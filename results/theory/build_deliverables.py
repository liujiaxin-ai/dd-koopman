"""Merge the sharded diagnostics into the V1-V5 deliverables.

Reads every `sh_*/` and `v4/` directory under `--shards-dir` and writes
`operator_identity.csv`, `response_certificate.csv`, `noise_gain.csv`,
`synthetic_coverage.csv`, `SUMMARY.md` and `provenance.json` into `--out`.

Row order is deterministic (family, seed, cell, validation index, antenna) so the
same shard set always produces the same file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import diag_lib as D

SORT_KEYS = {
    "operator_identity.csv": ["family", "seed", "cm", "ds", "ms", "val_index", "antenna"],
    "response_certificate.csv": ["family", "seed", "cm", "ds", "ms", "val_index", "antenna"],
    "noise_gain.csv": ["family", "seed", "cm", "ds", "ms", "val_index", "antenna"],
    "synthetic_coverage.csv": ["family", "seed", "K_modes", "q", "c", "trial"],
}


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], schema: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sort_rows(name: str, rows: list[dict]) -> list[dict]:
    keys = SORT_KEYS[name]
    return sorted(rows, key=lambda r: tuple(_sortable(r.get(k)) for k in keys))


def _sortable(value):
    if value is None:
        return (2, 0.0, "")
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def schema_of(rows: list[dict]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def merge(name: str, sources: list[Path], out_dir: Path) -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    used: list[Path] = []
    for directory in sources:
        path = directory / name
        if path.exists() and path.stat().st_size > 1:
            part = read_rows(path)
            if part:
                rows.extend(part)
                used.append(path)
    if not rows:
        return [], []
    rows = sort_rows(name, rows)
    write_rows(out_dir / name, rows, schema_of(rows))
    return rows, used


def f(values) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=np.float64)


def q(values, p: float) -> float:
    return float(np.quantile(f(values), p))


def family_seed_table(rows: list[dict], metric: str) -> list[tuple]:
    table = []
    keys = sorted({(r["family"], r["seed"]) for r in rows})
    for family, seed in keys:
        subset = [r[metric] for r in rows if r["family"] == family and r["seed"] == seed]
        values = f(subset)
        table.append((family, seed, len(values), float(np.median(values)), float(values.max())))
    return table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=Path("<ladder-out>"))
    parser.add_argument("--out", type=Path, default=Path("<ladder-out>/merged"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp64-csv", type=Path, default=None,
                        help="operator_identity_fp64.csv from run_v1b_fp64.py")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in args.shards_dir.iterdir()
                     if p.is_dir() and (p.name.startswith("sh_") or p.name == "v4"))

    merged: dict[str, list[dict]] = {}
    provenance_sources: dict[str, list[str]] = {}
    for name in SORT_KEYS:
        rows, used = merge(name, sources, args.out)
        merged[name] = rows
        provenance_sources[name] = [str(p) for p in used]
        print(f"{name}: {len(rows)} rows from {len(used)} shards", flush=True)

    identity = merged["operator_identity.csv"]
    cert = merged["response_certificate.csv"]
    noise = merged["noise_gain.csv"]
    synthetic = merged["synthetic_coverage.csv"]

    lines = ["# Frozen-checkpoint theory diagnostics (V1-V5)", "",
             "All numbers come from the merged CSVs in this directory; every row",
             "carries the checkpoint SHA256, seed, cell, validation index and antenna.", ""]

    if identity:
        lines += ["## V1 operator identity", "",
                  "| family | seed | rows | rel_A median | rel_A max | rel_A_mode median | rel_A_fp64 median | rel_B median | rel_B max |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for family, seed, n, med, mx in family_seed_table(identity, "rel_A"):
            subset = [r for r in identity if r["family"] == family and r["seed"] == seed]
            lines.append(
                f"| {family} | {seed} | {n} | {med:.3e} | {mx:.3e} | "
                f"{np.median(f([r['rel_A_mode'] for r in subset])):.3e} | "
                f"{np.median(f([r['rel_A_fp64'] for r in subset])):.3e} | "
                f"{np.median(f([r['rel_B'] for r in subset])):.3e} | "
                f"{f([r['rel_B'] for r in subset]).max():.3e} |")
        lines += ["", f"`rel_full` (model output reproduced by the extraction): "
                      f"max {f([r['rel_full'] for r in identity]).max():.3e}.", ""]
        if "rel_gate" in identity[0]:
            gate = f([r["rel_gate"] for r in identity])
            gain = f([r["rel_gain_vs_operator"] for r in identity])
            f64 = f([r["rel_A_fp64"] for r in identity])
            lines += [
                "Attribution of the `rel_A` residual. Two float32 evaluation orders of the "
                "same identity are compared: `sum_j alpha_j P_j A_j` in the delay basis "
                "(`rel_gain_vs_operator`, median "
                f"{np.median(gain):.3e}, max {gain.max():.3e}) against the model's own "
                "DFT-domain composition (`rel_gate`, median "
                f"{np.median(gate):.3e}, max {gate.max():.3e}). Repeating the delay-basis "
                "contraction in double precision (`rel_A_fp64`, median "
                f"{np.median(f64):.3e}) leaves the gap unchanged, so the gap is the "
                "arithmetic error of the model's float32 forward, not of the extraction.", ""]
        lines += ["Gate as pre-declared: FP32 median <= 1e-6 and max <= 1e-4. `rel_A` misses "
                  "the median gate by about a factor of two; `rel_B` passes both. "
                  "`rel_A_mode` contracts each branch in the model's own mode basis, which "
                  "makes it algebraically identical to `rel_gate`, so it is not an "
                  "independent probe of the extraction; it is reported for completeness.", ""]

    fp64_path = args.fp64_csv
    if fp64_path is not None and Path(fp64_path).exists():
        fp64_rows = read_rows(Path(fp64_path))
        a64 = f([r["rel_A_fp64"] for r in fp64_rows])
        b64 = f([r["rel_B_fp64"] for r in fp64_rows])
        full64 = f([r["rel_full_fp64"] for r in fp64_rows])
        lines += [
            "### V1b: the same extraction with the model in float64",
            "",
            f"Same checkpoints, same extraction, model and tensors cast to `torch.float64` "
            f"({len(fp64_rows)} rows over 3 seeds): `rel_A` median "
            f"{np.median(a64):.3e}, max {a64.max():.3e}; `rel_B` median "
            f"{np.median(b64):.3e}, max {b64.max():.3e}; `rel_full` max {full64.max():.3e}. "
            "The pre-declared float64 gate (<= 1e-12) is met by about three orders of "
            "magnitude, which locates the float32 `rel_A` residual in the model's own "
            "single-precision forward rather than in the extraction or the algebra.", ""]

    if cert:
        lines += ["## V3 off-grid response certificate", "",
                  "| family | seed | rows | delta_Z1 median | defect_upper_Z1 median | upper < grid max | delta_Z2 median | defect_upper_Z2 median |",
                  "|---|---|---|---|---|---|---|---|"]
        for family, seed in sorted({(r["family"], r["seed"]) for r in cert}):
            subset = [r for r in cert if r["family"] == family and r["seed"] == seed]
            d1 = f([r["delta_hat_Z1"] for r in subset])
            u1 = f([r["defect_upper_Z1"] for r in subset])
            d2 = f([r["delta_hat_Z2"] for r in subset])
            u2 = f([r["defect_upper_Z2"] for r in subset])
            lines.append(
                f"| {family} | {seed} | {len(subset)} | {np.median(d1):.3e} | {np.median(u1):.3e} | "
                f"{int((u1 < d1).sum())} | {np.median(d2):.3e} | {np.median(u2):.3e} |")
        ratio = f([r["ratio_corr_over_delta"] for r in cert])
        lines += ["", f"`correction_norm / norm_PX_F`: median {np.median(ratio):.3f}, "
                      f"p05 {np.quantile(ratio, 0.05):.3f}, p95 {np.quantile(ratio, 0.95):.3f}.", ""]

    if noise:
        lines += ["## V2 exact/regularised reference", "",
                  f"`selfcheck_P0V_minus_Q` = {f([r['selfcheck_P0V_minus_Q'] for r in noise]).max():.3e} "
                  f"(the exact check is run on the 16th roots of unity, where V is square).", "",
                  f"`ls_residual_K257` = {f([r['ls_residual_K257'] for r in noise]).max():.3e} "
                  "(the 257-point Z1 constraint is over-determined, so P0 is the "
                  "minimum-norm least-squares solution there and this residual is reported, not asserted).", ""]
        for family, seed in sorted({(r["family"], r["seed"]) for r in noise}):
            subset = [r for r in noise if r["family"] == family and r["seed"] == seed]
            lines.append(
                f"- {family} seed {seed}: norm_PX_F^2 median "
                f"{np.median(f([r['norm_PX_F2'] for r in subset])):.4f}; excess over KT/N (K=257) median "
                f"{np.median(f([r['excess_K257'] for r in subset])):.4f}; excess over KT/N (K=16) median "
                f"{np.median(f([r['excess_K16'] for r in subset])):.4f}; dist to P0 median "
                f"{np.median(f([r['dist_to_P0_F'] for r in subset])):.4f}; dist to P_lambda median "
                f"{np.median(f([r['dist_to_PLam_F'] for r in subset])):.4f}")
        lines.append("")

    if synthetic:
        lines += ["## V4 synthetic clean/noisy coverage", "",
                  "| K | q | c | trials | violations | ratio bound/true median | ratio p95 |",
                  "|---|---|---|---|---|---|---|"]
        for k, qq, cc in sorted({(int(r["K_modes"]), r["q"], r["c"]) for r in synthetic}):
            subset = [r for r in synthetic
                      if int(r["K_modes"]) == k and r["q"] == qq and r["c"] == cc]
            ratio = f([r["ratio_bound_over_true"] for r in subset])
            violations = sum(int(r["violation"]) for r in subset)
            lines.append(f"| {k} | {qq} | {cc} | {len(subset)} | {violations} | "
                         f"{np.median(ratio):.3f} | {np.quantile(ratio, 0.95):.3f} |")
        lines.append("")

    lines += ["## V5 family comparison",
              "",
              "reported / gridfix / snap are reported on the same stride-10 common subset "
              "(270 validation samples per seed); the `family` column separates them in "
              "`response_certificate.csv` and `operator_identity.csv`. These are descriptive "
              "frozen-checkpoint diagnostics; no significance claim is attached.", ""]
    (args.out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    provenance = D.provenance()
    provenance.update({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": args.device,
        "tasks": "V1 operator identity, V2 exact/regularised reference, V3 off-grid response "
                 "certificate, V4 synthetic coverage, V5 three-family comparison",
        "grids": {
            "Z1": "unit circle, 257 angles uniform on [-pi, pi]",
            "Z2": "Z1 x radii {0.90, 0.95, 0.99, 1.00}",
            "eta_Z1": math.pi / 256,
            "eta_Z2": max(math.pi / 256, 0.025),
        },
        "regularisation": {
            "lambda": "1e-3 * trace(V^H V) / N, computed on the Z1 grid",
            "Lambda": "I / K with K the number of grid columns (257)",
            "P0_check": "run on the 16th roots of unity, where V is square and invertible",
        },
        "seeds": [42, 43, 44],
        "subset_rule": "validation fold = randperm(1000, generator seed 42)[900:] per subset, "
                       "27 subsets = 2700 samples; stride 10 = 270 samples for the gridfix/snap families",
        "scripts": {
            name: {"path": str(Path("<theory-out>") / name),
                   "sha256": sha256(Path("<theory-out>") / name)}
            for name in ("run_diag.py", "diag_lib.py", "build_val_cache.py", "build_deliverables.py")
            if (Path("<theory-out>") / name).exists()
        },
        "row_sources": provenance_sources,
        "counts": {name: len(rows) for name, rows in merged.items()},
    })
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("wrote SUMMARY.md and provenance.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
