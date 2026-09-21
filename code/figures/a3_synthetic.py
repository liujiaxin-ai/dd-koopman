"""A3 - controlled off-grid Doppler experiment (mechanism causality).

The paper's motivation is that a predictor whose Doppler hypotheses live on the
16-point DFT grid cannot represent an off-grid Doppler, while a learned pole can.
That statement is testable in isolation: build channels from *known* specular
paths, place their Doppler frequencies on the grid or between grid points, and
score four predictors on exactly the same realisations.

    ours        DD_KOOP_TDD (reported configuration checkpoint)
    published   the benchmark's published model
    dftgrid     the fixed 16-point grid extrapolator (K=1), no training
    persistence repeat the last observed slot (the trivial control)

Expected shape of the result: on the grid the grid extrapolator is near exact
and the learned operator has nothing to add; as the Doppler moves off the grid
the grid extrapolator degrades sharply while the learned operator degrades
gracefully. If that is not what happens, the paper's motivation is wrong and we
want to know before submission.

Usage (repo root, GPU host):
    python a3_synthetic.py --scenes 32 --snr 20 --out <csv>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# machine-local default; set CSI4CAST_ROOT (or pass --out) on another host
HARNESS = Path(os.environ.get("CSI4CAST_ROOT", "/root/rivermind-data/csi4cast/code/CSI-4CAST-main"))
sys.path.insert(0, str(HARNESS))
os.chdir(HARNESS)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import src.cp.models.ours.dd_koop  # noqa: E402,F401  (registers DD_KOOP_TDD)
from src.cp.models.baseline.statistical.dftgrid import DFTGRIDMODEL  # noqa: E402
from src.cp.models import PREDICTORS  # noqa: E402
from src.testing.get_models import get_eval_model  # noqa: E402
from src.utils.norm_utils import denormalize_output, normalize_input  # noqa: E402

HIST, PRED, NSUB = 16, 4, 300


def build_scene(rng: np.random.Generator, n_scenes: int, delta: float,
                paths: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Known-path CSI: [N, 32, 16+4, 300] complex, Doppler offset ``delta`` bins."""
    ant = 32
    total = HIST + PRED
    slots = np.arange(total)[None, :]
    sub = np.arange(NSUB)[None, :]
    h = np.zeros((n_scenes, ant, total, NSUB), dtype=np.complex128)
    for n in range(n_scenes):
        for p in range(paths):
            bin_index = rng.integers(0, HIST)
            nu = (bin_index + delta) / HIST                # cycles per slot
            tau = rng.uniform(0, 15)                       # delay taps
            gain = rng.normal() + 1j * rng.normal()
            array = np.exp(1j * rng.uniform(0, 2 * np.pi, size=ant))
            time = np.exp(1j * 2 * np.pi * nu * slots)            # [1, T]
            freq = np.exp(-1j * 2 * np.pi * tau * sub / NSUB)     # [1, F]
            h[n] += (gain / np.sqrt(paths)) * array[:, None, None] * time[..., None] * freq[:, None, :]
    h /= np.sqrt(np.mean(np.abs(h) ** 2))              # unit average power
    return (torch.from_numpy(h[:, :, :HIST, :].astype(np.complex64)),
            torch.from_numpy(h[:, :, HIST:, :].astype(np.complex64)))


def add_noise(hist: torch.Tensor, snr_db: float, rng: torch.Generator) -> torch.Tensor:
    power = hist.abs().pow(2).mean()
    noise_power = power / (10 ** (snr_db / 10))
    noise = torch.complex(torch.randn(hist.shape, generator=rng),
                          torch.randn(hist.shape, generator=rng))
    return hist + noise * float(noise_power.sqrt() / np.sqrt(2))


def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().pow(2).mean() / target.abs().pow(2).mean())


def resolve_ours(ckpt: str | None):
    if ckpt:
        return PREDICTORS.DD_KOOP_TDD.load_from_checkpoint(ckpt)
    hits = sorted(glob.glob(str(HARNESS / "z_artifacts/outputs/sweep/CAPLONG_CAPNOAUX600/*/ckpts/epoch=*.ckpt")))
    if hits:
        return PREDICTORS.DD_KOOP_TDD.load_from_checkpoint(hits[-1])
    return get_eval_model(model_name="DD_KOOP_TDD", device=torch.device("cpu"), scenario="TDD")


def resolve_published():
    """Prefer the path whose parameter count is the published 21,913,750."""
    candidates = [HARNESS / "z_artifacts/weights/tdd/model/model.ckpt"]
    candidates.append(Path("/root/rivermind-data/csi4cast/registry_backup_model_102238.ckpt"))
    candidates += [Path(p) for p in glob.glob("/root/rivermind-data/csi4cast/**/registry_backup_model_*.ckpt", recursive=True)]
    for path in candidates:
        if not path.exists():
            continue
        model = PREDICTORS.MODEL_TDD.load_from_checkpoint(str(path))
        n = sum(p.numel() for p in model.parameters())
        if n == 21_913_750:
            print(f"[a3] published checkpoint: {path}")
            return model
        print(f"[a3] skip {path} ({n:,} params)")
    raise SystemExit("[a3] no published checkpoint with 21,913,750 parameters found")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=int, default=32)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ours-ckpt", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "results/analysis/a3_synthetic.csv"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator().manual_seed(args.seed)

    ours = resolve_ours(args.ours_ckpt).to(device).eval()
    published = resolve_published().to(device).eval()
    grid = DFTGRIDMODEL(num_modes=1).to(device).eval()
    print(f"[a3] ours={sum(p.numel() for p in ours.parameters()):,} "
          f"published={sum(p.numel() for p in published.parameters()):,} "
          f"device={device}")

    rows = []
    def forward_antennas(model, hist_4d, real_input: bool = True,
                         collapse: bool = True):
        """Run a benchmark model with the harness's own input contract.

        `collect_fn_separate_antennas` flattens the antenna axis, converts the
        complex history to the real 2K-wide layout the models are trained on,
        and the eval path folds the real prediction back to complex before
        denormalisation. Reproducing that here keeps A3 on the same contract as
        every other number in the paper.
        """
        b, a, t, f = hist_4d.shape
        if not collapse:            # grid extrapolator: 4-D complex (b, a, t, l)
            out = model(hist_4d.to(device))
            if out.shape[-1] != f:
                out = out.reshape(*out.shape[:-1], -1, 2).contiguous()
                out = torch.view_as_complex(out)
            return out
        x = hist_4d.reshape(b * a, t, f)
        if real_input:
            x = torch.view_as_real(x).reshape(x.shape[0], t, -1).to(device)
        else:
            x = x.to(device)
        out = model(x)
        if out.shape[-1] != f:      # real 2K output -> fold back to complex K
            out = out.reshape(out.shape[0], out.shape[1], -1, 2).contiguous()
            out = torch.view_as_complex(out)
        return out.reshape(b, a, out.shape[1], out.shape[2])

    for delta in (0.0, 0.1, 0.25, 0.4):
        hist, target = build_scene(rng, args.scenes, delta)
        hist = normalize_input(hist, target)[0]
        hist = add_noise(hist, args.snr, torch_rng)
        with torch.no_grad():
            preds = {
                "ours": forward_antennas(ours, hist),
                "published": forward_antennas(published, hist),
                "dftgrid": forward_antennas(grid, hist, real_input=False,
                                            collapse=False),
                "persistence": hist[:, :, -1:, :].repeat(1, 1, PRED, 1).to(device),
            }
            for name, pred in preds.items():
                pred_orig, target_orig = denormalize_output(pred.cpu(), target)
                rows.append({
                    "delta_bin": delta,
                    "snr_db": args.snr,
                    "scenes": args.scenes,
                    "model": name,
                    "nmse": round(nmse(pred_orig, target_orig), 4),
                })
        print(f"[a3] delta={delta:.2f} " +
              "  ".join(f"{r['model']}={r['nmse']:.4f}" for r in rows[-4:]), flush=True)

    import pandas as pd
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[a3] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
