"""Aggregate the V1/V3 diagnostics into the per-checkpoint table the paper cites.

Source: `results/theory/operator_identity.csv` and
`results/theory/response_certificate.csv` (rows are per validation sample and
antenna, so the aggregates below are over rows, not over independent samples).

Writes `results/analysis/theory_certificate_summary.csv`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
THEORY = ROOT / "results" / "theory"
OUT = ROOT / "results" / "analysis" / "theory_certificate_summary.csv"
OUT_COMMON = ROOT / "results" / "analysis" / "theory_family_common_subset.csv"

# The gridfix and snap families were run on every tenth validation sample
# (stride 10), i.e. the rows whose validation index is 900 + 10 k. The reported
# family was run on the full fold, so its stride-1 rows are filtered to the same
# indices to make the three-family comparison like for like.
COMMON_MASK = lambda frame: (frame["val_index"] - 900) % 10 == 0  # noqa: E731


def summarise_frame(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, seed), group in merged.groupby(["family", "seed"], sort=True):
        rows.append({
            "family": family,
            "seed": int(seed),
            "rows": len(group),
            "rel_A_median": group["rel_A"].median(),
            "rel_A_max": group["rel_A"].max(),
            "rel_A_mode_median": group["rel_A_mode"].median(),
            "rel_A_fp64_median": group["rel_A_fp64"].median(),
            "rel_B_median": group["rel_B"].median(),
            "rel_B_max": group["rel_B"].max(),
            "rel_full_max": group["rel_full"].max(),
            "rel_gate_median": group["rel_gate"].median(),
            "rel_gain_vs_operator_median": group["rel_gain_vs_operator"].median(),
            "delta_hat_Z1_median": group["delta_hat_Z1"].median(),
            "delta_hat_Z1_p95": group["delta_hat_Z1"].quantile(0.95),
            "delta_hat_Z1_max": group["delta_hat_Z1"].max(),
            "defect_upper_Z1_median": group["defect_upper_Z1"].median(),
            "defect_upper_Z1_max": group["defect_upper_Z1"].max(),
            "upper_below_gridmax_rows": int((group["defect_upper_Z1"] < group["delta_hat_Z1"]).sum()),
            "slack_Z1_median": group["slack_Z1"].median(),
            "delta_hat_Z2_median": group["delta_hat_Z2"].median(),
            "defect_upper_Z2_median": group["defect_upper_Z2"].median(),
            "noise_gain_2_median": group["noise_gain_2"].median(),
            "norm_PX_F_median": group["norm_PX_F"].median(),
            "correction_norm_median": group["correction_norm"].median(),
            "corr_over_PX_F_median": (group["correction_norm"] / group["norm_PX_F"]).median(),
            "corr_over_delta_Z1_median": group["corr_over_delta_Z1"].median(),
            "corr_over_delta_Z1_p05": group["corr_over_delta_Z1"].quantile(0.05),
            "g1_median": group["g1"].median(),
            "g2_median": group["g2"].median(),
        })
    return pd.DataFrame(rows)


def summarise() -> pd.DataFrame:
    identity = pd.read_csv(THEORY / "operator_identity.csv")
    cert = pd.read_csv(THEORY / "response_certificate.csv")

    # `cert` repeats `rel_A`; keep the copy from the identity table only, so the
    # merged column has a single unambiguous provenance.
    cert = cert.drop(columns=["rel_A"])
    merged = cert.merge(
        identity[["family", "seed", "sample_id", "antenna", "rel_A", "rel_A_mode",
                  "rel_A_fp64", "rel_B", "rel_full", "rel_gate", "rel_gain_vs_operator",
                  "g1", "g2"]],
        on=["family", "seed", "sample_id", "antenna"],
        how="inner",
    )
    merged["corr_over_delta_Z1"] = (
        merged["correction_norm"] / merged["delta_hat_Z1"].replace(0.0, np.nan)
    )
    merged["slack_Z1"] = merged["defect_upper_Z1"] - merged["delta_hat_Z1"]

    return summarise_frame(merged)


def main() -> int:
    table = summarise()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {OUT}")

    identity = pd.read_csv(THEORY / "operator_identity.csv")
    cert = pd.read_csv(THEORY / "response_certificate.csv").drop(columns=["rel_A"])
    common = cert[COMMON_MASK(cert)].merge(
        identity[["family", "seed", "sample_id", "antenna", "rel_A", "rel_A_mode",
                  "rel_A_fp64", "rel_B", "rel_full", "rel_gate", "rel_gain_vs_operator",
                  "g1", "g2"]],
        on=["family", "seed", "sample_id", "antenna"],
        how="inner",
    )
    common["corr_over_delta_Z1"] = (
        common["correction_norm"] / common["delta_hat_Z1"].replace(0.0, np.nan)
    )
    common["slack_Z1"] = common["defect_upper_Z1"] - common["delta_hat_Z1"]
    frame = summarise_frame(common)
    frame.to_csv(OUT_COMMON, index=False)
    print("\ncommon stride-10 subset (same validation indices for all families)")
    print(frame[["family", "seed", "rows", "delta_hat_Z1_median", "noise_gain_2_median",
                 "rel_A_median", "rel_B_median"]].to_string(index=False))
    print(f"\nwrote {OUT_COMMON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
