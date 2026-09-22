"""Read-only diagnostics for the operator identity and the off-grid response certificate.

Everything here runs against an *existing* checkpoint with the harness's own model
classes. Nothing is trained, no model file is edited, and only the training split's
validation fold is touched (27 subsets x the last 100 indices of the seeded
permutation that `train_ours.py` uses, i.e. the 2,700 validation samples).

Conventions (handoff section 3):
  * the model input is subcarrier-domain CSI per antenna, [16, 300] complex,
    flattened to [16, 600] real exactly as `collect_fn_separate_antennas` does;
  * X = IDFT over subcarriers of the (normalised) input = the delay-domain history,
    which is what `_lift` produces and what the operator acts on along time;
  * A_j are the three branch inputs (base / mixer / decomposition residual);
  * W_j[h, m] = gain_complex[h, m] * bounded_j[h, m] * zeta_j[m]**h is the effective
    per-(horizon, mode) complex gain actually applied by the branch, taken from the
    model's own `pole_angles` and `modulator` outputs -- never from `poles()`;
  * P_j = W_j @ F_N with F_N[m, n] = exp(-2*pi*i*m*n/16), so that op(A) = P_j @ A.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# The diagnostics run against the *frozen* model sources (the 393-line v8 that the
# reported checkpoints were trained with), not the stale 09-11 copy that ships in
# the local harness checkout. `D:\csi_gridfix\CSI-4CAST-main` is a working copy of
# the harness whose `src/cp/models/ours/*` are the frozen files and whose
# `z_artifacts/data` is a junction to the benchmark data, so nothing is duplicated
# and no existing file is modified.
REMOTE = os.environ.get("DIAG_REMOTE") == "1"
ROOT = Path(r"D:\research\icassp-paper")
GRIDFIX_DIR = ROOT / "remote_recovery" / "gridfix"
THEORY = Path(os.environ.get("DIAG_THEORY", str(ROOT / "results" / "theory")))

if REMOTE:
    HARNESS = Path("<harness-root>")
    _SW = HARNESS / "z_artifacts/outputs/sweep"
    REPORTED = {
        42: _SW / "CAPLONG_CAPNOAUX600/2026-09-14_02-24-38/ckpts/epoch=192-step=146680-val_loss=0.141040.ckpt",
        43: _SW / "CAPLONG_CAPNOAUXS43/2026-09-14_09-59-08/ckpts/epoch=211-step=161120-val_loss=0.142389.ckpt",
        44: Path("<theory-out>_ckpts/seed44.ckpt"),
    }
    GRIDFIX = {
        42: _SW / "GRIDFIX_S42/2026-09-19_08-17-53/ckpts/epoch=296-step=225720-val_loss=0.140098.ckpt",
        43: _SW / "GRIDFIX_S43/2026-09-19_12-55-01/ckpts/epoch=207-step=158080-val_loss=0.140989.ckpt",
        44: _SW / "GRIDFIX_S44/2026-09-19_16-57-08/ckpts/epoch=177-step=135280-val_loss=0.139362.ckpt",
    }
    CONFIG_REPORTED = HARNESS / "z_artifacts/config/ours/sweep/tdd_caplong.yaml"
    GRIDFIX_YAML = HARNESS / "z_artifacts/config/ours/sweep/tdd_gridfix.yaml"
else:
    HARNESS = Path(r"D:\csi_gridfix\CSI-4CAST-main")
    REPORTED = {
        42: ROOT / "remote_recovery/host1/z_artifacts/outputs/sweep/CAPLONG_CAPNOAUX600"
            / "2026-09-14_02-24-38/ckpts/epoch=192-step=146680-val_loss=0.141040.ckpt",
        43: ROOT / "remote_recovery/host1/z_artifacts/outputs/sweep/CAPLONG_CAPNOAUXS43"
            / "2026-09-14_09-59-08/ckpts/epoch=211-step=161120-val_loss=0.142389.ckpt",
        44: ROOT / "remote_recovery/host2/checkpoints/z_artifacts/outputs/sweep/CAPLONG_HEADNOAUXS44"
            / "2026-09-14_11-18-00/ckpts/epoch=254-step=193800-val_loss=0.139419.ckpt",
    }
    GRIDFIX = {
        42: GRIDFIX_DIR / "GRIDFIX_S42.ckpt",
        43: GRIDFIX_DIR / "GRIDFIX_S43.ckpt",
        44: GRIDFIX_DIR / "GRIDFIX_S44.ckpt",
    }
    CONFIG_REPORTED = ROOT / "code/config/sweep/tdd_caplong.yaml"
    GRIDFIX_YAML = GRIDFIX_DIR / "tdd_gridfix.yaml"

N, T, L = 16, 4, 300


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_env() -> None:
    """Make the harness importable and run from its root (relative data paths)."""
    os.chdir(HARNESS)
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    import src.cp.models.ours.dd_koop_v6  # noqa: F401
    import src.cp.models.ours.dd_koop_v7  # noqa: F401
    import src.cp.models.ours.dd_koop_v8  # noqa: F401


# --- model loading ---------------------------------------------------------

def load_model(ckpt: Path, yaml_config: Path | None = None, grid_constraint: bool = False):
    """Load a reported/gridfix/snap model without touching any model source file.

    `grid_constraint=True` reproduces the online intervention used for the snap
    family: `raw_delta` is pinned to zero and the *conditional* angle offset is
    forced to zero, while the radius conditioning is left untouched. Gridfix
    checkpoints carry `grid_constrained: true` in their hyper-parameters; the key
    is dropped here (the frozen harness class does not know it) and the identical
    constraint is applied instead.
    """
    from src.cp.models.ours.dd_koop_v8 import DD_KOOP8_TDD
    from src.utils.main_utils import make_config

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = dict(payload["hyper_parameters"]["model"].params)
    grid_from_ckpt = bool(saved.pop("grid_constrained", False))

    cfg_path = yaml_config or (GRIDFIX_YAML if grid_from_ckpt else CONFIG_REPORTED)
    cfg = make_config(str(cfg_path))
    cfg.model.params = {**cfg.model.params, **saved}

    model = DD_KOOP8_TDD(cfg)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    if grid_from_ckpt or grid_constraint:
        apply_grid_constraint(model)
    return model, {"grid_from_ckpt": grid_from_ckpt, "grid_constraint": bool(grid_constraint),
                   "missing": list(missing), "unexpected": list(unexpected),
                   "config": str(cfg_path)}


def apply_grid_constraint(model) -> None:
    """Pin the total phase offset to zero on an already-built model (in place)."""
    op = model.model.operator
    with torch.no_grad():
        op.raw_delta.zero_()
    op.raw_delta.requires_grad_(False)

    def pole_angles_grid(doppler, _op=op):
        bins = torch.arange(_op.hist_len, device=doppler.device, dtype=torch.float32)
        base = 2 * math.pi * (bins + 0.5 * torch.tanh(_op.raw_delta)) / _op.hist_len
        power = torch.log(doppler.abs().pow(2).mean(dim=-1) + 1e-8)
        generated = _op.pole_modulator(_op.context_norm(power))
        radius_offset = torch.tanh(generated[:, _op.hist_len:])
        angle = base.unsqueeze(0)  # conditional offset pinned to zero
        radius = 0.5 * (1.0 + torch.tanh(_op.raw_radius.unsqueeze(0) + radius_offset))
        return angle, radius

    op.pole_angles = pole_angles_grid


# --- validation fold -------------------------------------------------------

COMBOS = None


def combos():
    global COMBOS
    if COMBOS is None:
        from itertools import product
        from src.utils.data_utils import LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN
        COMBOS = list(product(LIST_CHANNEL_MODEL, LIST_DELAY_SPREAD, LIST_MIN_SPEED_TRAIN))
    return COMBOS


def val_fold(cm, ds, ms):
    """The validation fold of one subset, normalised exactly as training does it.

    The split is the harness's own: `randperm(1000, generator=seed 42)[900:]`.
    Inputs are returned *without* the training-time AWGN draw: the operator
    identity and the response certificate hold pointwise for every input, and a
    clean fold is reproducible without replaying the trainer's RNG stream.

    If `DIAG_CACHE` points at a directory of per-cell `.pt` files written by
    `build_val_cache.py`, the fold is read from there instead of re-loading and
    re-normalising the 1.75 GB raw cells (the raw read is the bottleneck when
    several diagnostics run at once).
    """
    cache = os.environ.get("DIAG_CACHE", "").strip()
    if cache:
        blob = Path(cache) / ("cm%s_ds%03d_ms%03d.pt" % (cm, round(ds * 1e9), ms))
        if blob.exists():
            payload = torch.load(blob, map_location="cpu", weights_only=False)
            return payload["hist"], payload["pred"]
    from src.utils.data_utils import _load_data
    from src.utils.norm_utils import normalize_input

    hist = _load_data(dir_data=Path("z_artifacts/data"), cm=cm, ds=ds, ms=ms,
                      is_train=True, is_gen=False, is_hist=True, is_U2D=False)
    pred = _load_data(dir_data=Path("z_artifacts/data"), cm=cm, ds=ds, ms=ms,
                      is_train=True, is_gen=False, is_hist=False, is_U2D=False)
    hist, pred = normalize_input(hist, pred, is_U2D=False)
    idx = torch.randperm(len(hist), generator=torch.Generator().manual_seed(42))[900:]
    return hist[idx], pred[idx]


def to_model_input(delay_or_csi_complex: torch.Tensor) -> torch.Tensor:
    """[n, 32, 16, 300] complex -> [n*32, 16, 600] real (separate-antenna layout)."""
    x = delay_or_csi_complex
    b, a, t, s = x.shape
    x = x.reshape(b * a, t, s)
    return torch.view_as_real(x).reshape(b * a, t, s * 2).contiguous()


# --- extraction ------------------------------------------------------------

F_N = None


def f_matrix(device, dtype=torch.complex64) -> torch.Tensor:
    m = torch.arange(N, device=device, dtype=torch.float32)
    return torch.exp(-2j * math.pi * torch.outer(m, m) / N).to(dtype)


def branch_gain(op, spectrum: torch.Tensor) -> torch.Tensor:
    """W_j[h, m] for one branch input, from the model's own conditioned outputs."""
    b = spectrum.shape[0]
    angle, radius = op.pole_angles(spectrum)
    horizon = torch.arange(1, op.pred_len + 1, device=spectrum.device, dtype=torch.float32)
    magnitude = radius.unsqueeze(-2) ** horizon.reshape(1, -1, 1)
    advance = torch.polar(magnitude, angle.unsqueeze(-2) * horizon.reshape(1, -1, 1))
    base = torch.complex(op.gain_real, op.gain_imag).unsqueeze(0) * advance
    power = torch.log(spectrum.abs().pow(2).mean(dim=-1) + 1e-8)
    modulation = op.modulator(op.context_norm(power)).view(b, op.pred_len, op.hist_len, 2)
    bounded = 1.0 + torch.tanh(torch.view_as_complex(modulation.contiguous()))
    return base * bounded


def extract(model, x: torch.Tensor) -> dict:
    """Run the model's own forward step by step and return the theory quantities."""
    net = model.model
    t = x.shape[1]
    xin = net.instance_norm.normalize(x) if net.use_revin else x
    xd = net.denoiser(xin, t) if net.use_denoiser else xin
    delay = net._lift(xd)                       # X : the delay-domain history
    spec0 = net.spectrum_of(delay)
    out0 = net.operator(spec0)

    pred = out0
    branches = [(delay, spec0, out0, branch_gain(net.operator, spec0))]

    mixed = net.shared_mixer(net.arl(delay))
    spec1 = net.spectrum_of(mixed)
    out1 = net.operator(spec1)
    g1 = float(net.gate_mix)
    pred = pred + net.gate_mix * (out1 - out0)
    branches.append((mixed, spec1, out1, branch_gain(net.operator, spec1)))

    g2 = 0.0
    if net.use_delay_decomposition:
        _structured, residual = net.decomposition(delay)
        spec2 = net.spectrum_of(residual)
        out2 = net.operator(spec2)
        g2 = float(net.gate_split)
        pred = pred + net.gate_split * (out2 - out0)
        branches.append((residual, spec2, out2, branch_gain(net.operator, spec2)))

    # full model output, for the fidelity check
    taps = pred.shape[-1]
    padded = pred
    if taps < net.num_subcarriers:
        padded = torch.cat([pred, pred.new_zeros(pred.shape[0], net.pred_len,
                                                 net.num_subcarriers - taps)], dim=-1)
    h_pred = torch.fft.fft(padded, dim=-1)[..., : net.num_subcarriers]
    y_full = torch.view_as_real(h_pred).reshape(pred.shape[0], net.pred_len,
                                                net.num_subcarriers * 2)
    with torch.no_grad():
        y_model = net(x)

    device = pred.device
    fn = f_matrix(device)
    alphas = [1.0 - g1 - g2, g1, g2]
    p_x = torch.zeros(pred.shape[0], T, N, dtype=torch.complex64, device=device)
    recon = torch.zeros_like(pred)
    recon_mode = torch.zeros_like(pred)
    recon64 = torch.zeros_like(pred, dtype=torch.complex128)
    r_x = torch.zeros_like(pred)
    for alpha, (a_j, _s_j, _o_j, w_j) in zip(alphas, branches):
        p_j = torch.einsum("bpm,mn->bpn", w_j.to(torch.complex64), fn)  # P_j
        p_x = p_x + alpha * p_j
        recon = recon + alpha * torch.einsum("bpn,bnl->bpl", p_j, a_j)
        # the same branch contracted in the model's own mode basis, from *my*
        # extracted W: isolates "was W extracted correctly" from "what does the
        # basis change W -> W F_N cost in float32"
        recon_mode = recon_mode + alpha * torch.einsum(
            "bml,bpm->bpl", _s_j, w_j.to(torch.complex64))
        p_j64 = torch.einsum("bpm,mn->bpn", w_j.to(torch.complex128),
                             f_matrix(device, torch.complex128))
        recon64 = recon64 + alpha * torch.einsum("bpn,bnl->bpl", p_j64,
                                                 a_j.to(torch.complex128))
        r_x = r_x + alpha * torch.einsum("bpn,bnl->bpl", p_j, a_j - delay)

    denom = torch.linalg.matrix_norm(pred).clamp_min(1e-30)
    rel_a = (torch.linalg.matrix_norm(pred - recon) / denom)
    rel_a_mode = (torch.linalg.matrix_norm(pred - recon_mode) / denom)
    rel_a_fp64 = (torch.linalg.matrix_norm(pred.to(torch.complex128) - recon64) /
                  denom.to(torch.float64))
    rel_b = (torch.linalg.matrix_norm(recon - (torch.einsum("bpn,bnl->bpl", p_x, delay) + r_x)) / denom)
    rel_full = (torch.linalg.matrix_norm(y_model - y_full) /
                torch.linalg.matrix_norm(y_full).clamp_min(1e-30))
    # Attribution of whatever `rel_a` turns out to be. `pred` is composed the way
    # the model composes it, `Σ α_j out_j` applies the same gates to the branch
    # outputs the operator itself returned, and `Σ α_j P_j A_j` rebuilds those
    # outputs from the extracted gains. Comparing all three isolates gate
    # arithmetics from gain extraction.
    recomposed = torch.zeros_like(pred)
    for alpha, (_a_j, _s_j, o_j, _w_j) in zip(alphas, branches):
        recomposed = recomposed + alpha * o_j
    rel_gate = torch.linalg.matrix_norm(pred - recomposed) / denom
    rel_gain_vs_operator = torch.linalg.matrix_norm(recon - recomposed) / denom
    return {"p_x": p_x, "r_x": r_x, "pred": pred, "y_model": y_model,
            "rel_a": rel_a, "rel_a_mode": rel_a_mode, "rel_a_fp64": rel_a_fp64,
            "rel_b": rel_b, "rel_full": rel_full,
            "rel_gate": rel_gate, "rel_gain_vs_operator": rel_gain_vs_operator,
            "g1": g1, "g2": g2,
            "norm_px_f": torch.linalg.matrix_norm(p_x),
            "norm_px_2": torch.linalg.matrix_norm(p_x, ord=2),
            "norm_rx_f": torch.linalg.matrix_norm(r_x),
            "norm_x_f": torch.linalg.matrix_norm(delay),
            "norm_pred_f": torch.linalg.matrix_norm(pred)}


# --- certificate grid ------------------------------------------------------

def grid(radii, n_angle=257):
    theta = torch.linspace(-math.pi, math.pi, n_angle, dtype=torch.float32)
    zs = []
    for r in radii:
        zs.append(torch.polar(torch.full_like(theta, float(r)), theta))
    z = torch.cat(zs)
    return z.to(torch.complex64)


def l_v(rho):  # sum_{n=1}^{15} n^2 rho^{2n-2}
    return math.sqrt(sum(n * n * rho ** (2 * n - 2) for n in range(1, N)))


def l_u(rho):  # sum_{p=16}^{19} p^2 rho^{2p-2}
    return math.sqrt(sum(p * p * rho ** (2 * p - 2) for p in range(N, N + T)))


def defect(p_x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """max over the grid of ||P_X v_N(z) - u_T(z)||_2, per sample."""
    powers = torch.stack([z ** k for k in range(N + T)], dim=0)      # [N+T, nz]
    v = powers[:N]                                                   # [N, nz]
    u = powers[N:]                                                   # [T, nz]
    diff = torch.einsum("bpn,nz->bpz", p_x, v) - u.unsqueeze(0)      # [B, T, nz]
    return torch.linalg.vector_norm(diff, dim=1).max(dim=1).values   # [B]


def provenance() -> dict:
    setup_env()
    files = {
        "main_dd_koop.py": ROOT / "code/models_ours/dd_koop.py",
        "anon_dd_koop.py": ROOT / "anonymous_repo/code/models_ours/dd_koop.py",
        "dd_koop_v6.py": ROOT / "code/models_ours/dd_koop_v6.py",
        "dd_koop_v7.py": ROOT / "code/models_ours/dd_koop_v7.py",
        "dd_koop_v8.py": ROOT / "code/models_ours/dd_koop_v8.py",
        "harness_dd_koop_v8.py": HARNESS / "src/cp/models/ours/dd_koop_v8.py",
        "harness_dd_koop.py": HARNESS / "src/cp/models/ours/dd_koop.py",
        "harness_train_ours.py": HARNESS / "train_ours.py",
    }
    out = {k: sha256(p) for k, p in files.items() if Path(p).exists()}
    out["checkpoints"] = {
        f"reported_s{s}": {"path": str(p), "sha256": sha256(p)} for s, p in REPORTED.items()
    }
    out["checkpoints"].update({
        f"gridfix_s{s}": {"path": str(p), "sha256": sha256(p)} for s, p in GRIDFIX.items()
    })
    out["conventions"] = {
        "N": N, "T": T, "L": L,
        "input": "subcarrier-domain CSI per antenna, [16,300] complex -> [16,600] real",
        "X": "ifft over subcarriers (= model _lift output); time axis rows, delay taps columns",
        "fft_norm": "torch.fft.fft over the 16 time slots (no scaling); ifft over 300 subcarriers",
        "P_j": "W_j @ F_N, F_N[m,n] = exp(-2*pi*i*m*n/16)",
        "W_j": "gain_complex * bounded_j * zeta_j^h, from pole_angles() and modulator()",
        "Z1": "unit circle, 257 angles uniform on [-pi, pi], eta = pi/256",
        "Z2": "Z1 x radii {0.90, 0.95, 0.99, 1.00}, eta = max(pi/256, 0.025)",
        "validation_fold": "randperm(1000, seed 42)[900:] per subset, 27 subsets = 2,700 samples; clean (no AWGN)",
        "precision": "FP32 primary, FP64 probe on 32 samples",
    }
    out["generated"] = "2026-09-21"
    return out
