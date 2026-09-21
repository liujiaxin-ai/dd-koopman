"""Training entry point for the delay-domain Koopman predictor.

Alternative to `python -m src.cp.main`: normalises each training subset
*before* concatenation, which keeps the memory peak of the official entry
point (all raw, split and normalised copies coexist) from blowing up.

This script keeps the protocol bit-for-bit (same 27 subsets, same `train_ratio`,
same per-sample AWGN uniform in [0, 25] dB applied after normalisation, same
normalisation statistics, same model/optimiser/loss handling) but

  * normalises each subset immediately after loading it, so the raw copy dies
    early, and
  * writes the normalised, noised subsets straight into pre-allocated tensors
    instead of accumulating a list and calling `torch.cat`.

Peak resident memory becomes approximately the size of the final dataset rather
than several multiples of it.

Usage::

    python train_ours.py --hparams_csi_pred z_artifacts/config/ours/tdd_dk.yaml
"""

from __future__ import annotations

import argparse
import os
import random
from itertools import product
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from src.cp.config.config import DataConfig
from src.noise.noise import gen_vanilla_noise_snr
from src.utils.data_utils import (
    LIST_CHANNEL_MODEL,
    LIST_DELAY_SPREAD,
    LIST_MIN_SPEED_TRAIN,
    PRED_LEN,
    SNR_RANGE_GAUSSIAN_NOISE_TRAIN,
    TOT_ANTENNAS,
    CSIDataset,
    _load_data,
    collect_fn_gather_antennas,
    collect_fn_separate_antennas,
)
from src.utils.norm_utils import normalize_input

class LowMemoryDataModule(pl.LightningDataModule):

    def __init__(self, data_config: DataConfig, verbose: bool = True, workers: int = 6):
        super().__init__()
        self.data_cfg = data_config
        self.verbose = verbose
        self.workers = max(workers, data_config.num_workers)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def setup(self, stage=None):
        combos = list(product(LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN))

        selected = os.environ.get("CSI_SUBSETS", "").strip()
        if selected:
            wanted = {token.strip() for token in selected.split(",") if token.strip()}
            combos = [c for c in combos if f"{c[0]}:{c[1]:g}:{c[2]}" in wanted]
            if not combos:
                raise ValueError(f"CSI_SUBSETS matched no combination: {selected}")
            self._log(f"CSI_SUBSETS active: packing {len(combos)} of 27 combinations")
        per_subset = 1000
        train_ratio = self.data_cfg.train_ratio
        n_train_per = int(per_subset * train_ratio)
        n_val_per = per_subset - n_train_per

        hist_train = torch.empty(
            (n_train_per * len(combos), TOT_ANTENNAS, 16, 300), dtype=torch.complex64
        )
        pred_train = torch.empty(
            (n_train_per * len(combos), TOT_ANTENNAS, PRED_LEN, 300), dtype=torch.complex64
        )
        hist_val = torch.empty(
            (n_val_per * len(combos), TOT_ANTENNAS, 16, 300), dtype=torch.complex64
        )
        pred_val = torch.empty(
            (n_val_per * len(combos), TOT_ANTENNAS, PRED_LEN, 300), dtype=torch.complex64
        )
        self._log(
            f"pre-allocated: train hist {tuple(hist_train.shape)} "
            f"({hist_train.numel() * 8 / 1024**3:.1f} GiB), pred "
            f"({pred_train.numel() * 8 / 1024**3:.1f} GiB)"
        )

        cursor_train = 0
        cursor_val = 0
        for index, (cm, ds, ms) in enumerate(combos):
            hist = _load_data(
                dir_data=Path(self.data_cfg.dir_dataset),
                cm=cm,
                ds=ds,
                ms=ms,
                is_train=True,
                is_gen=False,
                is_hist=True,
                is_U2D=self.data_cfg.is_U2D,
            )
            pred = _load_data(
                dir_data=Path(self.data_cfg.dir_dataset),
                cm=cm,
                ds=ds,
                ms=ms,
                is_train=True,
                is_gen=False,
                is_hist=False,
                is_U2D=self.data_cfg.is_U2D,
            )

            hist, pred = normalize_input(hist, pred, is_U2D=self.data_cfg.is_U2D)
            assert hist is not None and pred is not None

            indices = torch.randperm(len(hist), generator=torch.Generator().manual_seed(42))
            train_idx = indices[:n_train_per]
            val_idx = indices[n_train_per:]

            for source_idx, target, cursor in (
                (train_idx, hist_train, cursor_train),
                (val_idx, hist_val, cursor_val),
            ):
                block = hist[source_idx]
                for offset in range(block.shape[0]):
                    sample = block[offset]
                    snr_val = random.uniform(*SNR_RANGE_GAUSSIAN_NOISE_TRAIN)
                    target[cursor + offset] = sample + gen_vanilla_noise_snr(sample, SNR=snr_val)

            for source_idx, target, cursor in (
                (train_idx, pred_train, cursor_train),
                (val_idx, pred_val, cursor_val),
            ):
                target[cursor : cursor + len(source_idx)] = pred[source_idx]

            cursor_train += n_train_per
            cursor_val += n_val_per
            del hist, pred
            self._log(
                f"[{index + 1}/{len(combos)}] CM={cm} DS={round(ds * 1e9)}ns MS={ms} packed"
            )

        self.train_dataset = CSIDataset(hist_train, pred_train)
        self.val_dataset = CSIDataset(hist_val, pred_val)
        self._log(
            f"datasets ready: train={len(self.train_dataset)} val={len(self.val_dataset)}"
        )

    def train_dataloader(self):
        collect_fn = (
            collect_fn_separate_antennas
            if self.data_cfg.is_separate_antennas
            else collect_fn_gather_antennas
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=self.data_cfg.shuffle,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=self.workers > 0,
            collate_fn=collect_fn,
        )

    def val_dataloader(self):
        collect_fn = (
            collect_fn_separate_antennas
            if self.data_cfg.is_separate_antennas
            else collect_fn_gather_antennas
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            persistent_workers=self.workers > 0,
            collate_fn=collect_fn,
        )

def main() -> int:
    from src.utils.main_utils import (
        make_checkpoint_callback,
        make_config,
        make_early_stopping_callback,
        make_logger,
        make_output_dir,
        make_tensorboard_logger,
    )
    from src.utils.model_utils import load_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hparams_csi_pred", required=True)
    parser.add_argument("--max-epochs", type=int, default=None, help="override num_epochs")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    config = make_config(args.hparams_csi_pred)
    pl.seed_everything(config.seed, workers=True)
    output_dir = make_output_dir(config)
    logger = make_logger(output_dir)
    logger.info(f"config: {args.hparams_csi_pred}")
    logger.info(f"output: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device=device)
    logger.info(f"model: {model!s}")

    datamodule = LowMemoryDataModule(config.data)
    logger.info("data module: low-memory variant")

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
    logger.info(f"best checkpoint: {ckpt_cb.best_model_path}")
    print(f"BEST_CHECKPOINT={ckpt_cb.best_model_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
