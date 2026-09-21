"""Train several DD-Koop configurations from a single packed copy of the data.

Packing the training subsets into tensors costs far more than one epoch, so
this driver packs once and trains the requested configurations back to back
in the same process. The seed is reset before each configuration and the
packed data, including the per-sample AWGN draw, is identical for all of
them, which is what makes the comparison between configurations meaningful.

Usage::

    python train_sweep.py z_artifacts/config/ours/sweep/tdd_caplong.yaml \
        --max-epochs 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning.pytorch as pl
import torch

import src.cp.models.ours.dd_koop
import src.cp.models.ours.dd_koop_tap
import src.cp.models.baseline.mambacsp
import src.cp.models.baseline.baselines_suite
import src.cp.models.baseline.new_baselines

from src.utils.main_utils import (
    make_checkpoint_callback,
    make_config,
    make_early_stopping_callback,
    make_logger,
    make_output_dir,
    make_tensorboard_logger,
)
from src.utils.model_utils import load_model
from src.utils.data_utils import collect_fn_gather_antennas, collect_fn_separate_antennas

from train_ours import LowMemoryDataModule
from src.cp.models.ours.dd_koop import least_squares_spectrum_map
from src.utils.data_utils import collect_fn_separate_antennas

class DistillDataset(torch.utils.data.Dataset):

    def __init__(self, base, teacher: torch.Tensor):
        self.base = base
        self.teacher = teacher

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        hist, target = self.base[index]
        return hist, target, self.teacher[index]

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

def collate_distill(batch):
    hist, target, teacher = zip(*batch)
    hist_real, target_real = collect_fn_separate_antennas(list(zip(hist, target)))
    teacher_real = torch.stack(teacher, dim=0).reshape(
        -1, teacher[0].shape[-2], teacher[0].shape[-1]
    )
    return hist_real, target_real, teacher_real

class PackOnceDataModule(LowMemoryDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._packed = False
        self.distilling = False

    def setup(self, stage=None):
        if self._packed:
            return
        super().setup(stage)
        self._packed = True

    def enable_distillation(self, payload: dict) -> None:

        from src.utils.data_utils import TOT_ANTENNAS

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(-1, TOT_ANTENNAS, tensor.shape[-2], tensor.shape[-1])

        self.train_dataset = DistillDataset(self.train_dataset, split(payload["train"]))
        self.val_dataset = DistillDataset(self.val_dataset, split(payload["val"]))
        self.distilling = True

    def _collate(self):
        if self.distilling:
            return collate_distill
        from src.utils.data_utils import collect_fn_gather_antennas

        return (
            collect_fn_separate_antennas
            if self.data_cfg.is_separate_antennas
            else collect_fn_gather_antennas
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=self.data_cfg.shuffle,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=self.workers > 0,
            collate_fn=self._collate(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=self.workers > 0,
            collate_fn=self._collate(),
        )

def _literal(raw: str):
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw

def _apply_override(config, key: str, value) -> None:
    if key == "seed":
        config.seed = int(value)
    elif key == "epochs":
        config.training.num_epochs = int(value)
    elif key == "patience":
        config.training.early_stopping_patience = int(value)
    elif key == "lr":
        config.optimizer.params["lr"] = float(value)
    else:
        config.model.params[key] = value

def preflight(model: pl.LightningModule, datamodule: PackOnceDataModule, steps: int = 3) -> None:

    collate = datamodule._collate()
    loader = torch.utils.data.DataLoader(
        datamodule.train_dataset,
        batch_size=max(4, min(16, datamodule.data_cfg.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    batch = next(iter(loader))
    device = next(model.parameters()).device
    batch = tuple(
        item.to(device) if isinstance(item, torch.Tensor) else item for item in batch
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, 0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"preflight: non-finite loss {float(loss)}")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("preflight: non-finite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="yaml config paths, trained in order")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a `model.params` entry, e.g. --set num_delay_taps=16",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="rename the run so overrides of one config file do not share an output directory",
    )
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        default=None,
        help="path to the tensor file written by distill.py; enables distillation",
    )
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    entries: list[tuple[str, str | None, dict]] = []
    for raw_entry in args.configs:
        parts = raw_entry.split("::")
        entry_tag = parts[1] if len(parts) > 1 and parts[1] else None
        overrides: dict = {}
        if len(parts) > 2 and parts[2]:
            for assignment in parts[2].split(","):
                key, _, value = assignment.partition("=")
                overrides[key] = _literal(value)
        entries.append((parts[0], entry_tag, overrides))

    reference = make_config(entries[0][0])
    pl.seed_everything(reference.seed, workers=True)
    datamodule = PackOnceDataModule(reference.data)
    packed_at = time.time()
    datamodule.setup()
    print(f"[sweep] packed in {time.time() - packed_at:.0f}s; "
          f"train={len(datamodule.train_dataset)} val={len(datamodule.val_dataset)}", flush=True)

    if args.teacher_targets is not None:
        payload = torch.load(args.teacher_targets, map_location="cpu")
        datamodule.enable_distillation(payload)
        print(f"[sweep] distillation enabled from {args.teacher_targets}: "
              f"train {tuple(payload['train'].shape)}", flush=True)

    summary_path = args.summary or Path("z_artifacts/outputs/sweep_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for path, entry_tag, entry_overrides in entries:
        config = make_config(path)
        for assignment in args.set:
            key, _, raw = assignment.partition("=")
            if not _:
                raise SystemExit(f"--set expects KEY=VALUE, got {assignment!r}")
            config.model.params[key] = _literal(raw)
        for key, value in entry_overrides.items():
            _apply_override(config, key, value)
        suffix = args.tag or entry_tag
        if suffix:
            config.experiment_name = f"{config.experiment_name}_{suffix}"
            config.output_dir = f"{config.output_dir}_{suffix}"
        pl.seed_everything(config.seed, workers=True)
        output_dir = make_output_dir(config)
        logger = make_logger(output_dir)
        logger.info(f"sweep config: {path}")
        logger.info(f"output: {output_dir}")

        model = load_model(config, device=device)
        logger.info(f"model: {model!s}")

        if config.model.params.get("warm_start"):
            network = getattr(model, "model", None)
            operator = getattr(network, "operator", None)
            if operator is not None and hasattr(operator, "warm_start"):
                hist = datamodule.train_dataset.H_hist[:256].to(device)
                pred = datamodule.train_dataset.H_pred[:256].to(device)
                weight = least_squares_spectrum_map(hist, pred, network.num_delay_taps)
                operator.warm_start(weight)
                logger.info(f"warm-started the spectrum map from {tuple(weight.shape)} LS solution")
                print(f"[sweep] {Path(path).stem}: warm start applied", flush=True)

        started = time.time()
        preflight_ok = True
        try:
            preflight(model, datamodule)
        except RuntimeError as error:
            preflight_ok = False
            logger.error(f"preflight failed for {path}: {error}")
            print(f"[sweep] {Path(path).stem}: PREFLIGHT FAILED ({error})", flush=True)

        best = None
        if preflight_ok:
            ckpt_cb = make_checkpoint_callback(config=config, output_dir=output_dir)
            earlystop_cb = make_early_stopping_callback(config=config)
            tb_logger = make_tensorboard_logger(config=config, output_dir=output_dir)
            training_cfg = config.training
            trainer = pl.Trainer(
                max_epochs=args.max_epochs or training_cfg.num_epochs,
                accelerator=config.accelerator,
                devices=config.devices,
                precision=config.precision,
                callbacks=[ckpt_cb, earlystop_cb],
                logger=tb_logger,
                log_every_n_steps=training_cfg.log_every_n_steps,
                gradient_clip_val=training_cfg.gradient_clip_val,
                accumulate_grad_batches=training_cfg.accumulate_grad_batches,
                check_val_every_n_epoch=training_cfg.check_val_every_n_epoch,
                enable_progress_bar=training_cfg.enable_progress_bar,
                enable_model_summary=training_cfg.enable_model_summary,
            )
            trainer.fit(model, datamodule)
            best = str(ckpt_cb.best_model_path)
            logger.info(f"best checkpoint: {best}")
            print(f"BEST_CHECKPOINT[{Path(path).stem}]={best}", flush=True)

        elapsed = time.time() - started
        print(f"[sweep] {Path(path).stem}: {elapsed / 60:.1f} min, best={best}", flush=True)
        records.append(
            {
                "config": path,
                "output_dir": str(output_dir),
                "best_checkpoint": best,
                "minutes": elapsed / 60.0,
                "preflight_ok": preflight_ok,
                "params": int(sum(p.numel() for p in model.parameters())),
            }
        )
        summary_path.write_text(json.dumps(records, indent=2))
        del model
        torch.cuda.empty_cache()

    print(f"[sweep] done, summary at {summary_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
