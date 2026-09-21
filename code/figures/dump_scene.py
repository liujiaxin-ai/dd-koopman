"""Dump one test scene (history, target, ours, published) for the F2 figure.

This runs the benchmark's own data loading, z-score normalisation, noise-free
evaluation path and denormalisation, then stores a small complex tensor set so
the figure script can draw the delay-Doppler spectrum and the four-slot
prediction without touching the harness again.

Usage (from this repository):

    python code/figures/dump_scene.py --scenario cm_B_ds_050_ms_003 --samples 6

Output: results/analysis/scene_dump_<scenario>.npz
    hist  [n, 32, 16, 300] complex64  historical CSI actually fed to the models
    gt    [n, 32,  4, 300] complex64  held-out target slots
    ours  [n, 32,  4, 300] complex64  reported model prediction
    pub   [n, 32,  4, 300] complex64  published model prediction
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

HARNESS = Path(os.environ.get("CSI_HARNESS", "../CSI-4CAST-main"))
OUT_DIR = ROOT / "results" / "analysis"
OFFICIAL_CKPT = Path(
    os.environ.get("CSI_OFFICIAL_CKPT",
                   "../CSI-4CAST-main/registry/registry_backup_model_102238.ckpt"))

sys.path.insert(0, str(HARNESS))
os.chdir(HARNESS)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import src.cp.models.ours.dd_koop  # noqa: E402,F401  (registers DD_KOOP_TDD)
from src.cp.models import PREDICTORS  # noqa: E402
from src.testing.get_models import get_eval_model  # noqa: E402
from src.utils.data_utils import (  # noqa: E402
    CSIDataset, collect_fn_gather_antennas, collect_fn_separate_antennas,
    load_data, make_folder_name,
)
from src.utils.dirs import DIR_DATA  # noqa: E402
from src.utils.norm_utils import denormalize_output, normalize_input  # noqa: E402


def predict(model, hist, device, batch_size: int):
    dataset = CSIDataset(hist, torch.zeros_like(hist[:, :, :4, :]))
    collate = (collect_fn_separate_antennas if model.is_separate_antennas
               else collect_fn_gather_antennas)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    preds = []
    with torch.no_grad():
        for batch_hist, _ in loader:
            batch_hist = batch_hist.to(device)
            preds.append(model(batch_hist).detach().cpu())
    return torch.cat(preds, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cm_B_ds_050_ms_003")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default=None,
                    help="'regular' or 'generalization'; default by channel model")
    ap.add_argument("--snr", type=float, default=10.0,
                    help="vanilla-noise SNR in dB applied to the history "
                         "(use a negative value for the noise-free input)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cm, ds_ns, ms = args.scenario.split("_")[1], int(args.scenario.split("_")[3]), int(args.scenario.split("_")[5])
    ds = ds_ns * 1e-9
    # channel models A/C/D live in the regular test split, B/E only exist in
    # the generalization split (they are absent from training by construction)
    is_gen = (args.split == "generalization") if args.split else (cm in {"B", "E"})
    device = torch.device(args.device)
    print(f"[dump] scenario={args.scenario} split="
          f"{'generalization' if is_gen else 'regular'} device={device}")

    hist = load_data(dir_data=Path(DIR_DATA), list_cm=[cm], list_ds=[ds],
                     list_ms=[ms], is_train=False, is_gen=is_gen, is_hist=True)
    target = load_data(dir_data=Path(DIR_DATA), list_cm=[cm], list_ds=[ds],
                       list_ms=[ms], is_train=False, is_gen=is_gen, is_hist=False)
    print(f"[dump] raw hist={tuple(hist.shape)} target={tuple(target.shape)}")
    hist_norm, target_norm = normalize_input(hist, target)
    if args.snr is not None and args.snr >= 0:
        # identical to the harness's vanilla-noise path for one SNR setting
        from src.noise.noise_testing import Noise
        torch.manual_seed(0)
        noise = Noise()
        for i in range(len(hist_norm)):
            hist_norm[i] = hist_norm[i] + noise.vanilla(hist_norm[i], args.snr)
        print(f"[dump] applied vanilla noise at SNR={args.snr:g} dB")

    n = min(args.samples, hist_norm.shape[0])
    outs = {"hist": hist_norm[:n].clone(), "gt": target_norm[:n].clone()}
    full_nmse = {}
    for label, model_name in (("ours", "DD_KOOP_TDD"), ("pub", "MODEL_OFFICIAL")):
        if label == "pub":
            # the local `model` weight slot holds the 91k capacity control, so
            # the published checkpoint is loaded explicitly here
            model = PREDICTORS.MODEL_TDD.load_from_checkpoint(str(OFFICIAL_CKPT))
            model.to(device).eval()
        else:
            model = get_eval_model(model_name=model_name, device=device,
                                   scenario="TDD")
        params = sum(p.numel() for p in model.parameters())
        print(f"[dump] {label}: {model_name} params={params:,} "
              f"separate_antennas={model.is_separate_antennas}")
        # score the whole scene (the claim is about the scene, not six samples)
        pred_all = predict(model, hist_norm, device, args.batch_size)
        pred_all_orig, target_all_orig = denormalize_output(pred_all, target_norm)
        full_nmse[label] = float(
            (torch.abs(pred_all_orig - target_all_orig) ** 2).mean()
            / (torch.abs(target_all_orig) ** 2).mean())
        print(f"[dump] {label}: whole-scene NMSE over "
              f"{hist_norm.shape[0]} samples = {full_nmse[label]:.4f}")
        outs[label] = pred_all_orig[:n]
        outs["gt"] = target_all_orig[:n]
        model.cpu()

    # denormalise the history back to the original CSI scale for the figure
    hist_orig, _ = denormalize_output(hist_norm[:n], target_norm[:n])
    outs["hist"] = hist_orig

    for key, tensor in outs.items():
        arr = tensor.numpy().astype(np.complex64)
        err = float(np.mean(np.abs(arr - outs["gt"].numpy()) ** 2)
                    / np.mean(np.abs(outs["gt"].numpy()) ** 2)) if key in ("ours", "pub") else float("nan")
        print(f"[dump] {key:5s} shape={arr.shape} "
              f"nmse={err:.4f}" if key in ("ours", "pub") else
              f"[dump] {key:5s} shape={arr.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"scene_dump_{args.scenario}.npz"
    np.savez_compressed(
        out_path,
        hist=outs["hist"].numpy().astype(np.complex64),
        gt=outs["gt"].numpy().astype(np.complex64),
        ours=outs["ours"].numpy().astype(np.complex64),
        pub=outs["pub"].numpy().astype(np.complex64),
        scenario=args.scenario, cm=cm, ds_ns=ds_ns, ms=ms, samples=n,
        is_gen=bool(is_gen),
        # whole-scene values over every test sample of this scenario, i.e. the
        # quantity the benchmark's own evaluation averages
        scene_nmse_ours=full_nmse["ours"], scene_nmse_pub=full_nmse["pub"],
        scene_samples=int(hist_norm.shape[0]), snr_db=float(args.snr),
    )
    print(f"[dump] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[dump] scene NMSE: ours={full_nmse['ours']:.4f} "
          f"published={full_nmse['pub']:.4f} "
          f"diff={full_nmse['ours'] - full_nmse['pub']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
