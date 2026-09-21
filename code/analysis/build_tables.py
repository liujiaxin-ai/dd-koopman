"""Regenerate every headline table from the handover's own results directory.

The handover rule is that no number may come from memory, so the narrative is
written against the files this script produces. It reads only
`results/run_csv/*.csv` (one row per test setting, the four per-horizon NMSE
values stored as a printed array in `nmse_mean`), the published reference
`results/reference/official_per_setting.csv`, and the local profile output
`results/analysis/profiles_local_5060.txt`.

Run from the handover root:
    python code/analysis/build_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "run_csv"
REFERENCE = ROOT / "results" / "reference"
OUT = ROOT / "results" / "analysis"
RNG = np.random.default_rng(20260913)
BOOTSTRAP = 5000


def read_run(path: Path) -> pd.DataFrame:
    """Per-setting NMSE of one run, with the mobility and SNR axes kept."""
    frame = pd.read_csv(path)
    if "scenario" in frame:
        frame = frame[frame["scenario"] == "TDD"]
    values = frame["nmse_mean"].map(
        lambda text: float(np.mean([float(x) for x in re.findall(r"-?\d+\.?\d*(?:e[-+]?\d+)?", str(text))]))
    )
    out = pd.DataFrame({
        "cm": frame["cm"].astype(str),
        "ds": frame["ds"].astype(float).round(10),
        "ms": frame["ms"].astype(float),
        "nmse": values,
    })
    if "noise_degree" in frame:
        out["snr"] = frame["noise_degree"].astype(float)
    if "noise_type" in frame:
        out["noise_type"] = frame["noise_type"].astype(str)
    return out.reset_index(drop=True)


def published_regular() -> pd.DataFrame:
    """Published MODEL rows on TDD regular (162 settings)."""
    frame = pd.read_csv(REFERENCE / "official_per_setting.csv")
    subset = frame[
        (frame["model"] == "MODEL")
        & (frame["duplex"] == "TDD")
        & (frame["split"] == "regular")
        & (frame["noise_type"] == "vanilla")
    ].copy()
    return pd.DataFrame({
        "cm": subset["cm"].astype(str),
        "ds": subset["ds"].astype(float).round(10),
        "ms": subset["ms"].astype(float),
        "nmse": subset["nmse_mean"].astype(float),
    }).reset_index(drop=True)


def load_profiles() -> dict:
    """model name -> (params, macs) from the local profiling output."""
    table = {}
    path = OUT / "profiles_local_5060.txt"
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = re.match(r"\s*(\S+)\s+params=\s*([\d,]+)\s+MACs/sample=\s*([\d,]+)", line)
            if match:
                table[match.group(1)] = (int(match.group(2).replace(",", "")),
                                         int(match.group(3).replace(",", "")))
    # Architecture-only profiles computed on the CPU (the capacity points are
    # training on the GPU host while these counts are produced).
    extra = OUT / "capacity_profiles.csv"
    if extra.exists():
        for row in pd.read_csv(extra).itertuples(index=False):
            table[row.label] = (int(row.params), int(row.macs_per_sample))
    return table


def summarise(frame: pd.DataFrame) -> dict:
    """Mean NMSE overall and per mobility class."""
    means = frame.groupby("ms")["nmse"].mean()
    return {
        "nmse": float(frame["nmse"].mean()),
        "nmse_1": float(means.get(1.0, np.nan)),
        "nmse_10": float(means.get(10.0, np.nan)),
        "nmse_30": float(means.get(30.0, np.nan)),
    }


def paired(table: pd.DataFrame, left: str, right: str, ms: float | None = None):
    """Paired bootstrap over (channel model, delay spread, speed) cells."""
    a = table[table["tag"] == left]
    b = table[table["tag"] == right]
    if ms is not None:
        a = a[a["ms"] == ms]
        b = b[b["ms"] == ms]
    key = ["cm", "ds", "ms", "snr"]
    merged = a.merge(b, on=key, suffixes=("_l", "_r"))
    cells = merged.groupby(["cm", "ds", "ms"])[["nmse_l", "nmse_r"]].mean()
    diff = (cells["nmse_l"] - cells["nmse_r"]).to_numpy()
    if diff.size == 0:
        return None
    draws = RNG.choice(diff, size=(BOOTSTRAP, diff.size), replace=True).mean(axis=1)
    return {"n": int(diff.size), "mean": float(diff.mean()),
            "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5))}


# label -> (file, params key in the profile table)
MAIN = [
    ("ours-reported-seed42", "CAPNOAUX600_full162.csv", "CAPLONG"),
    ("ours-reported-seed43", "CAPNOAUXS43_full162.csv", "CAPLONG"),
    ("ours-reported-seed44", "HEADNOAUXS44_full162.csv", "CAPLONG"),
    ("ours-with-loss-seed42", "LONGFULL_full162.csv", "CAPLONG"),
    ("ours-with-loss-seed43", "SEED43_full162.csv", "CAPLONG"),
    ("ours-with-loss-240ep-seed42", "S49K_full162.csv", "CAPLONG"),
    ("ours-with-loss-240ep-seed43", "S49KS43_full162.csv", "CAPLONG"),
    ("published-architecture-91k", "P3CONV_full162.csv", "O93K"),
    # capacity sweep of the reported operator (same protocol, only the two
    # widths change); rows are added as each point finishes
    ("capacity-S-96k", "CAPLONG_CAPACITY_S_full162.csv", "CAPACITY_S"),
    ("capacity-XS-77k", "CAPLONG_CAPACITY_XS_full162.csv", "CAPACITY_XS"),
    ("capacity-TINY-58k", "CAPLONG_CAPACITY_TINY_full162.csv", "CAPACITY_TINY"),
    ("capacity-M-250k", "CAPLONG_CAPACITY_M_full162.csv", "CAPACITY_M"),
    ("capacity-L-327k", "CAPLONG_CAPACITY_L_full162.csv", "CAPACITY_L"),
    ("capacity-matched-transformer", "TRANSMATCHED_full162.csv", "TRANSFORMERS"),
    ("transformer-262k", "TRANSFORMER_full162.csv", "TRANSFORMER"),
    ("patchtst-617k", "PATCHTST_full162.csv", "PATCHTST"),
    ("mambacsp-plateau", "MAMBA64CONV_C_full162.csv", "MAMBA64"),
    ("timemixer-L-iclr24", "NBTIML_full162.csv", "TIMEMIXERL"),
    ("tsmixer-L-2023", "NBTSML_full162.csv", "TSMIXERL"),
    ("itransformer-L-iclr24", "NBITFL_full162.csv", "ITRANSFORMERL"),
    ("mamba-lite-L", "NBMAML_full162.csv", "MAMBALITEL"),
    ("dlinear-baseline", "DLINEAR_full162.csv", "DLINEAR"),
    ("tcn-baseline", "TCN_full162.csv", "TCN"),
    ("gru-baseline", "GRU_full162.csv", "GRU"),
    ("mlp-baseline", "MLP_full162.csv", "MLP"),
]

ABLATION = [
    ("ABLREF", "ABLREF_full162.csv", True, True, True, True),
    ("no-denoiser", "ABL1_full162.csv", False, True, True, True),
    ("no-decomposition", "ABL2_full162.csv", True, False, True, True),
    ("no-conditioned-poles", "ABL3_full162.csv", True, True, False, True),
    ("no-denoiser-decomposition", "ABL12_full162.csv", False, False, True, True),
    ("no-denoiser-poles", "ABL13_full162.csv", False, True, False, True),
    ("no-decomposition-poles", "ABL23_full162.csv", True, False, False, True),
    ("no-denoiser-decomposition-poles", "ABL123_full162.csv", False, False, False, True),
    ("no-spectral-loss", "ABLNOAUX_full162.csv", True, True, True, False),
    ("no-spectral-loss-no-poles", "REDUND2_full162.csv", True, True, False, False),
]

ROBUST = [
    ("ours-reported-seed42", "CAPNOAUX600_robust486.csv"),
    ("ours-reported-seed43", "CAPNOAUXS43_robust486.csv"),
    ("ours-reported-seed44", "HEADNOAUXS44_robust486.csv"),
    ("ours-with-loss-seed43", "SEED43_full162_robust486.csv"),
    ("ours-with-loss-240ep-seed43", "S49K_robust486.csv"),
    ("ours-T300C64", "T300C64_robust486.csv"),
    ("capacity-matched-transformer", "ROBUST_TRANSMATCHED.csv"),
    ("patchtst-617k", "ROBUST_PATCHTST.csv"),
    ("mambacsp-plateau", "MAMBA64CONV_C_robust486.csv"),
]


def main() -> int:
    """Write every derived table next to the raw results."""
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    published = published_regular()
    published_mean = float(published["nmse"].mean())

    rows = []
    for label, filename, profile_key in MAIN:
        path = RUNS / filename
        if not path.exists():
            rows.append({"label": label, "source": f"results/run_csv/{filename}", "status": "MISSING"})
            continue
        stats = summarise(read_run(path))
        params, macs = profiles.get(profile_key, (np.nan, np.nan))
        rows.append({
            "label": label,
            "source": f"results/run_csv/{filename}",
            "params": params,
            "macs_per_sample": macs,
            "nmse": round(stats["nmse"], 4),
            "nmse_1ms": round(stats["nmse_1"], 4),
            "nmse_10ms": round(stats["nmse_10"], 4),
            "nmse_30ms": round(stats["nmse_30"], 4),
            "ratio_vs_published": round(stats["nmse"] / published_mean, 3),
            "status": "ok",
        })
    rows.append({
        "label": "CSI-4CAST-published",
        "source": "results/reference/official_per_setting.csv (MODEL, TDD, regular, vanilla)",
        "params": profiles.get("OFFICIAL", (np.nan, np.nan))[0],
        "macs_per_sample": profiles.get("OFFICIAL", (np.nan, np.nan))[1],
        "nmse": round(published_mean, 4),
        "nmse_1ms": round(float(published[published["ms"] == 1.0]["nmse"].mean()), 4),
        "nmse_10ms": round(float(published[published["ms"] == 10.0]["nmse"].mean()), 4),
        "nmse_30ms": round(float(published[published["ms"] == 30.0]["nmse"].mean()), 4),
        "ratio_vs_published": 1.0,
        "status": "ok",
    })
    main_table = pd.DataFrame(rows).sort_values("nmse")
    main_table.to_csv(OUT / "main_table.csv", index=False)

    # ---- ablations -------------------------------------------------------
    reference_row = read_run(RUNS / "ABLREF_full162.csv")
    reference_nmse = float(reference_row["nmse"].mean())
    ablation_rows = []
    for label, filename, denoiser, decomp, poles, auxi in ABLATION:
        path = RUNS / filename
        stats = summarise(read_run(path))
        ablation_rows.append({
            "cell": label,
            "denoiser": denoiser,
            "delay_decomposition": decomp,
            "conditioned_poles": poles,
            "spectral_loss": auxi,
            "nmse": round(stats["nmse"], 4),
            "delta_vs_reference": round(stats["nmse"] - reference_nmse, 4),
            "relative_vs_reference": f"{(stats['nmse'] / reference_nmse - 1) * 100:+.1f}%",
            "nmse_30ms": round(stats["nmse_30"], 4),
            "source": f"results/run_csv/{filename}",
        })
    pd.DataFrame(ablation_rows).to_csv(OUT / "ablation_table.csv", index=False)

    # ---- the same factorial under the objective the paper reports ---------
    # Every row of Table 2 in the first draft carried the spectral auxiliary
    # loss (ABLREF = 0.1496), while the reported configuration switches it off
    # (0.1401). The ABLOFF_* queue re-runs the full denoiser x decomposition x
    # poles factorial with auxi_lambda=0.0 so the deltas are measured against
    # the reported objective; the reference cell double-scores ABLNOAUX.
    ABLATION_OFF = [
        ("ABLOFF_REF", "ABLOFF_REF_full162.csv", True, True, True),
        ("ABLOFF_NODENO", "ABLOFF_NODENO_full162.csv", False, True, True),
        ("ABLOFF_NODECOMP", "ABLOFF_NODECOMP_full162.csv", True, False, True),
        ("ABLOFF_NOPOLE", "ABLOFF_NOPOLE_full162.csv", True, True, False),
        ("ABLOFF_NODENO_DECOMP", "ABLOFF_NODENO_DECOMP_full162.csv", False, False, True),
        ("ABLOFF_NODENO_POLE", "ABLOFF_NODENO_POLE_full162.csv", False, True, False),
        ("ABLOFF_NODECOMP_POLE", "ABLOFF_NODECOMP_POLE_full162.csv", True, False, False),
        ("ABLOFF_NODENO_DECOMP_POLE", "ABLOFF_NODENO_DECOMP_POLE_full162.csv", False, False, False),
    ]
    off_reference_path = RUNS / "ABLOFF_REF_full162.csv"
    if off_reference_path.exists():
        off_reference = float(read_run(off_reference_path)["nmse"].mean())
        off_rows = []
        for cell, filename, denoiser, decomp, poles in ABLATION_OFF:
            path = RUNS / filename
            if not path.exists():
                off_rows.append({
                    "cell": cell, "denoiser": denoiser,
                    "delay_decomposition": decomp, "conditioned_poles": poles,
                    "spectral_loss": False, "nmse": np.nan,
                    "delta_vs_reference": np.nan,
                    "relative_vs_reference": "", "nmse_30ms": np.nan,
                    "source": f"results/run_csv/{filename}",
                    "status": "pending",
                })
                continue
            stats = summarise(read_run(path))
            off_rows.append({
                "cell": cell,
                "denoiser": denoiser,
                "delay_decomposition": decomp,
                "conditioned_poles": poles,
                "spectral_loss": False,
                "nmse": round(stats["nmse"], 4),
                "delta_vs_reference": round(stats["nmse"] - off_reference, 4),
                "relative_vs_reference": f"{(stats['nmse'] / off_reference - 1) * 100:+.1f}%",
                "nmse_30ms": round(stats["nmse_30"], 4),
                "source": f"results/run_csv/{filename}",
                "status": "ok",
            })
        pd.DataFrame(off_rows).to_csv(OUT / "ablation_table_off.csv", index=False)

    # ---- mechanism ablations under the same loss-off protocol -------------
    # These two are the modules the abstract credits but which never had a
    # full-protocol ablation before 2026-09-17 (the old evidence was a 9-subset
    # screen). Same protocol as ABLOFF_REF: auxi_lambda=0, 240 epochs,
    # patience 25, seed 42, 27 subsets, official 162 grid.
    ABLATION_MECH = [
        ("ABLOFF_NOWARM", "ABLOFF_NOWARM_full162.csv", "least-squares warm start"),
        ("ABLOFF_NOTAP", "ABLOFF_NOTAP_full162.csv", "tap encoding"),
    ]
    if off_reference_path.exists():
        mech_rows = []
        for cell, filename, removed in ABLATION_MECH:
            path = RUNS / filename
            if not path.exists():
                mech_rows.append({"cell": cell, "removed": removed, "nmse": np.nan,
                                  "delta_vs_reference": np.nan,
                                  "relative_vs_reference": "", "nmse_30ms": np.nan,
                                  "source": f"results/run_csv/{filename}",
                                  "status": "pending"})
                continue
            stats = summarise(read_run(path))
            mech_rows.append({
                "cell": cell, "removed": removed,
                "nmse": round(stats["nmse"], 4),
                "delta_vs_reference": round(stats["nmse"] - off_reference, 4),
                "relative_vs_reference": f"{(stats['nmse'] / off_reference - 1) * 100:+.1f}%",
                "nmse_30ms": round(stats["nmse_30"], 4),
                "source": f"results/run_csv/{filename}",
                "status": "ok",
            })
        pd.DataFrame(mech_rows).to_csv(OUT / "ablation_table_mech.csv", index=False)

    # ---- seeds -----------------------------------------------------------
    seed_rows = []
    for label, filename, seed, schedule, auxi in [
        ("ours-reported", "CAPNOAUX600_full162.csv", 42, 600, False),
        ("ours-reported", "CAPNOAUXS43_full162.csv", 43, 600, False),
        ("ours-reported", "HEADNOAUXS44_full162.csv", 44, 600, False),
        ("ours-reported", "ABLNOAUX_full162.csv", 42, 240, False),
        ("ours-with-loss", "LONGFULL_full162.csv", 42, 600, True),
        ("ours-with-loss", "SEED43_full162.csv", 43, 600, True),
        ("ours-with-loss", "S49K_full162.csv", 42, 240, True),
        ("ours-with-loss", "S49KS43_full162.csv", 43, 240, True),
    ]:
        stats = summarise(read_run(RUNS / filename))
        seed_rows.append({"family": label, "seed": seed, "schedule_epochs": schedule,
                          "spectral_loss": auxi, "nmse": round(stats["nmse"], 4),
                          "source": f"results/run_csv/{filename}"})
    pd.DataFrame(seed_rows).to_csv(OUT / "seeds_table.csv", index=False)

    # ---- robustness ------------------------------------------------------
    robustness_rows = []
    for label, filename in ROBUST:
        path = RUNS / filename
        if not path.exists():
            robustness_rows.append({"label": label, "source": f"results/run_csv/{filename}",
                                    "status": "MISSING"})
            continue
        frame = read_run(path)
        robustness_rows.append({"label": label, "status": "ok",
                                "rows": len(frame),
                                "nmse": round(float(frame["nmse"].mean()), 4),
                                "source": f"results/run_csv/{filename}"})
    robustness_rows.append({"label": "CSI-4CAST-published", "status": "ok", "rows": 486,
                            "nmse": round(float(pd.read_csv(REFERENCE / "official_per_setting.csv").query(
                                "model == 'MODEL' and duplex == 'TDD' and split == 'robustness'")["nmse_mean"].mean()), 4),
                            "source": "results/reference/official_per_setting.csv (MODEL, TDD, robustness)"})
    pd.DataFrame(robustness_rows).to_csv(OUT / "robustness_table.csv", index=False)

    # ---- generalization split -------------------------------------------
    official = pd.read_csv(REFERENCE / "official_per_setting.csv")
    gen_ref = official[
        (official["model"] == "MODEL") & (official["duplex"] == "TDD")
        & (official["split"] == "generalization")
    ].copy()
    gen_ref["cm"] = gen_ref["cm"].astype(str)
    gen_ref["ds"] = gen_ref["ds"].astype(float).round(10)
    gen_ref["ms"] = gen_ref["ms"].astype(float)
    gen_ref["snr"] = gen_ref["snr"].astype(float)
    gen_key = ["cm", "ds", "ms", "snr"]

    def generalization_row(label, filename):
        path = RUNS / filename
        if not path.exists():
            return {"label": label, "source": f"results/run_csv/{filename}", "status": "MISSING"}
        ours = read_run(path)
        ours["ds"] = ours["ds"].astype(float).round(10)
        merged = ours.merge(gen_ref[gen_key + ["nmse_mean"]], on=gen_key, how="inner")
        if merged.empty:
            return {"label": label, "source": f"results/run_csv/{filename}",
                    "status": "no overlap with the published grid"}
        merged = merged.rename(columns={"nmse_mean": "published_nmse"})
        merged["diff"] = merged["nmse"] - merged["published_nmse"]
        cells = merged.groupby(["cm", "ds", "ms"])["diff"].mean().to_numpy()
        draws = RNG.choice(cells, size=(BOOTSTRAP, cells.size), replace=True).mean(axis=1)
        return {
            "label": label,
            "status": "ok",
            "settings": int(len(merged)),
            "cells": int(cells.size),
            "ours_nmse": round(float(merged["nmse"].mean()), 4),
            "published_nmse": round(float(merged["published_nmse"].mean()), 4),
            "ratio": round(float(merged["nmse"].mean() / merged["published_nmse"].mean()), 3),
            "paired_mean_diff": round(float(cells.mean()), 4),
            "ci_low": round(float(np.percentile(draws, 2.5)), 4),
            "ci_high": round(float(np.percentile(draws, 97.5)), 4),
            "ours_10ms_or_less": round(float(merged[merged["ms"] <= 10]["nmse"].mean()), 4)
            if (merged["ms"] <= 10).any() else np.nan,
            "published_10ms_or_less": round(float(merged[merged["ms"] <= 10]["published_nmse"].mean()), 4)
            if (merged["ms"] <= 10).any() else np.nan,
            "source": f"results/run_csv/{filename}",
        }

    gen_rows = [
        generalization_row("ours-reported-seed42", "CAPNOAUX600_gen_gen.csv"),
        generalization_row("ours-with-loss-seed42", "LONGFULL72_gen.csv"),
        generalization_row("ours-with-loss-36combo-slice", "LONGFULL_gen.csv"),
        generalization_row("mambacsp-2026-882k", "MAMBA64_gen.csv"),
        generalization_row("capacity-matched-transformer", "TRANSMATCHED_gen.csv"),
    ]
    pd.DataFrame(gen_rows).to_csv(OUT / "generalization_table.csv", index=False)

    # ---- generalization grouped by unseen channel model ------------------
    import generalization_by_cm
    generalization_by_cm.run(
        np.random.default_rng(generalization_by_cm.RNG_SEED))

    # ---- paired statistics ----------------------------------------------
    tags = {}
    for tag, filename in [
        ("ours-reported-s42", "CAPNOAUX600_full162.csv"),
        ("ours-reported-s43", "CAPNOAUXS43_full162.csv"),
        ("ours-reported-s44", "HEADNOAUXS44_full162.csv"),
        ("ours-with-loss-s42", "LONGFULL_full162.csv"),
        ("ours-with-loss-s43", "SEED43_full162.csv"),
        ("abl-ref", "ABLREF_full162.csv"),
        ("abl-no-denoiser", "ABL1_full162.csv"),
        ("abl-no-decomp", "ABL2_full162.csv"),
        ("abl-no-poles", "ABL3_full162.csv"),
        ("abl-none", "ABL123_full162.csv"),
        ("capacity-matched-transformer", "TRANSMATCHED_full162.csv"),
    ]:
        frame = read_run(RUNS / filename)
        frame["tag"] = tag
        tags[tag] = frame
    published_frame = published.copy()
    if "snr" not in published_frame:
        official = pd.read_csv(REFERENCE / "official_per_setting.csv")
        official = official[
            (official["model"] == "MODEL") & (official["duplex"] == "TDD")
            & (official["split"] == "regular") & (official["noise_type"] == "vanilla")
        ].copy()
        published_frame["snr"] = official["snr"].astype(float).to_numpy()
    published_frame["tag"] = "published"
    tags["published"] = published_frame
    table = pd.concat(tags.values(), ignore_index=True)

    lines = ["comparison".ljust(34), "class".rjust(10), "n".rjust(5),
             "mean".rjust(10), "95% CI".rjust(24), "  significant"]
    header = f"{'comparison':<34}{'class':>10}{'n':>6}{'mean':>10}{'95% CI':>26}  significant"
    lines = [header]
    for left, right in [
        ("ours-reported-s42", "published"),
        ("ours-reported-s43", "published"),
        ("ours-reported-s44", "published"),
        ("ours-reported-s42", "ours-with-loss-s42"),
        ("ours-reported-s42", "abl-ref"),
        ("abl-ref", "abl-no-denoiser"),
        ("abl-ref", "abl-no-decomp"),
        ("abl-ref", "abl-no-poles"),
        ("abl-ref", "abl-none"),
        ("capacity-matched-transformer", "ours-with-loss-s42"),
    ]:
        for label, ms in (("all", None), ("1 m/s", 1.0), ("10 m/s", 10.0), ("30 m/s", 30.0)):
            stats = paired(table, left, right, ms)
            if stats is None:
                continue
            significant = "yes" if (stats["low"] > 0 or stats["high"] < 0) else "no"
            lines.append(f"{left + ' - ' + right:<34}{label:>10}{stats['n']:>6}"
                         f"{stats['mean']:>10.4f}"
                         f"{'[' + f'{stats[chr(108)+chr(111)+chr(119)]:+.4f}, ' + f'{stats[chr(104)+chr(105)+chr(103)+chr(104)]:+.4f}' + ']':>26}"
                         f"  {significant}")
    (OUT / "paired_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- harness reproduction -------------------------------------------
    repro = read_run(REFERENCE / "repro_MODEL_TDD_regular.csv")
    reference_value = float(pd.read_csv(REFERENCE / "official_per_setting.csv")
                            .query("model == 'MODEL' and duplex == 'TDD' and split == 'regular' "
                                   "and noise_type == 'vanilla'")["nmse_mean"].mean())
    (OUT / "harness_reproduction.txt").write_text(
        "published-model reproduction through the official harness\n"
        f"  our fresh run (results/reference/repro_MODEL_TDD_regular.csv): {repro['nmse'].mean():.6f}\n"
        f"  official shipped reference on the same 162 settings:          {reference_value:.6f}\n"
        f"  relative deviation: {(repro['nmse'].mean() / reference_value - 1) * 100:+.3f}%\n",
        encoding="utf-8")

    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

