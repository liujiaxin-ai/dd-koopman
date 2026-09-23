"""Figure F6 (v3, compact polar) - the certificate quantity: spectral defect
of the learned transfer around the unit circle.

FIGURE CONTRACT (nature-figure)
-------------------------------
1. Core conclusion: releasing exactness is not a narrow leak. The defect of
   the fitted transfer d_P(z) = ||P_X v_N(z) - u_T(z)|| forms a smooth,
   lobed profile around the unit circle - it dips to 2.10 at z = 1, peaks at
   7.1 in the mid-band, and its peak lies between 4.1 and 8.3 in every one
   of the 27 propagation cells - so the finite-scan certificate
   (Corollary 1) is governed by a broad band of angles, not an outlier.
2. Evidence chain, one inferential role per element:
   - terracotta median curve (hero): per-angle median over 86,400 rows
     (27 cells x 100 val samples x 32 antennas); radius = defect;
   - grey fine curves: the per-cell medians of the 27 propagation cells,
     the cell-to-cell variation behind the certificate's universality;
   - blue bands: 25th-75th and 5th-95th percentiles over all rows;
   - 16 bin dots on the curve: the DFT bin phases k*pi/8, where
     Proposition 1 fixes the exact transfer (the grid support).
   Polar form is chosen because the domain of the certificate IS the unit
   circle (a genuine angular profile, not a categorical radar chart): the
   geometry carries the meaning, and the 16-fold DFT structure becomes the
   visual motif.
3. Archetype: single hero profile, no panels.
4. Backend: Python - matplotlib primitives on the shared _pubstyle system
   (the same backend and export contract as fig_pareto / fig_arch).
5. Export contract: designed at final print size (0.5 * 88.9 mm column,
   square), editable SVG/PDF text, all glyphs >= 5 pt, source data
   results/theory/response_curve_seed42.csv and
   response_curve_seed42_by_cell.csv (same Z1 grid and operator as the
   response certificate CSV).

Statistics: descriptive profile of one frozen checkpoint (seed 42); no
significance language.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pubstyle as S  # noqa: E402

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

S.use_style(base_pt=6.0)

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = S.repo_root(HERE)
THEORY = ROOT / "results" / "theory"
OUT = HERE / "fig_defect"

W_MM = H_MM = 40.0          # final printed size: wrap-column side figure
RMAX = 11.5
BINS = np.arange(16) * np.pi / 8


def load_curve():
    rows = list(csv.DictReader(open(THEORY / "response_curve_seed42.csv")))
    assert len(rows) == 257, f"expected 257 angles, got {len(rows)}"
    order = np.argsort([float(r["angle_rad"]) for r in rows])
    ang = np.array([float(rows[i]["angle_rad"]) for i in order])
    med = np.array([float(rows[i]["median"]) for i in order])
    q25 = np.array([float(rows[i]["q25"]) for i in order])
    q75 = np.array([float(rows[i]["q75"]) for i in order])
    p05 = np.array([float(rows[i]["p05"]) for i in order])
    p95 = np.array([float(rows[i]["p95"]) for i in order])
    assert set(r["family"] for r in rows) == {"reported"}
    assert set(r["seed"] for r in rows) == {"42"}
    return ang, med, q25, q75, p05, p95


def load_cell_curves():
    """27 per-cell median curves, one per propagation condition."""
    rows = list(csv.DictReader(open(THEORY / "response_curve_seed42_by_cell.csv")))
    cells = sorted(set((r["cm"], r["ds"], r["ms"]) for r in rows))
    assert len(cells) == 27, f"expected 27 cells, got {len(cells)}"
    ang = np.array([float(r["angle_rad"]) for r in rows
                    if (r["cm"], r["ds"], r["ms"]) == cells[0]])
    order = np.argsort(ang)
    ang = ang[order]
    curves = []
    for c in cells:
        m = np.array([float(r["median"]) for r in rows
                      if (r["cm"], r["ds"], r["ms"]) == c])[order]
        curves.append(m)
    return ang, np.array(curves)


def per_cell_peaks(curves) -> tuple[float, float]:
    peaks = curves.max(axis=1)
    return float(peaks.min()), float(peaks.max())


def main():
    ang, med, q25, q75, p05, p95 = load_curve()
    cang, curves = load_cell_curves()
    lo, hi = per_cell_peaks(curves)

    # self-checks against the delivered readings
    i0 = int(np.argmin(np.abs(ang)))
    assert abs(ang[i0]) < 1e-4 and abs(med[i0] - 2.0955) < 0.01
    ip = int(np.argmax(med))
    assert abs(med[ip] - 7.0656) < 0.01
    assert abs(hi - 8.294) < 0.01 and abs(lo - 4.113) < 0.01, (lo, hi)

    fig = S.figure(W_MM, H_MM)
    ax = fig.add_axes((3.8 / W_MM, 3.6 / H_MM, 32.4 / W_MM, 32.4 / H_MM),
                      polar=True)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, RMAX)
    ax.set_rgrids([4, 8], labels=["4", "8"],
                  angle=247, fontsize=5.4, color=S.INK3)
    ax.set_thetagrids(np.degrees([0, np.pi / 2, np.pi, 3 * np.pi / 2]),
                      ["$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"],
                      fontsize=5.8, color=S.INK2)
    ax.tick_params(pad=-5.0)          # pull theta labels onto the rim
    ax.grid(True, axis="y", color=S.HAIR, lw=0.5)
    ax.grid(False, axis="x")
    ax.spines["polar"].set_color(S.INK3)
    ax.spines["polar"].set_linewidth(0.55)
    ax.spines["start"].set_visible(False)
    ax.spines["end"].set_visible(False)
    ax.set_facecolor("white")

    # data angles are in [-pi, pi]; wrap negatives into [0, 2pi). The curve
    # already closes at theta = +/-pi (identical direction, equal medians).
    th = np.where(ang < 0, ang + 2 * np.pi, ang)
    cth = np.where(cang < 0, cang + 2 * np.pi, cang)

    ax.fill_between(th, p05, p95, color=S.PUB_L, alpha=0.14, lw=0, zorder=2)
    for row in curves:                     # 27 cell medians, quiet grey
        ax.plot(cth, row, color=S.INK3, lw=0.42, alpha=0.42, zorder=3)
    ax.fill_between(th, q75, med, color=S.OURS, alpha=0.14, lw=0, zorder=4)
    ax.plot(th, med, color=S.OURS, lw=1.15, zorder=5)

    # the 16 DFT bin phases, marked on the curve (Proposition 1 grid support)
    med_at_bins = np.array([med[int(np.argmin(np.abs(ang - b)))]
                            for b in BINS])
    ax.plot(BINS, med_at_bins, "o", ms=1.9, mfc=S.INK2, mec="white",
            mew=0.35, zorder=6)

    # dip at z = 1 (East) and mid-band peak, direct-labelled (twin dots)
    ax.plot([0.0], [med[i0]], "o", ms=2.8, mfc=S.OURS, mec="white",
            mew=0.45, zorder=7)
    ax.plot([ang[ip]], [med[ip]], "o", ms=2.8, mfc=S.OURS, mec="white",
            mew=0.45, zorder=7)
    ax.annotate("2.10 at $z{=}1$", xy=(0.0, med[i0]),
                xytext=(11, -12), textcoords="offset points",
                color=S.INK2, fontsize=5.4, va="center", zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK3,
                                shrinkA=1, shrinkB=1.5))
    ax.annotate("median peak 7.1", xy=(ang[ip], med[ip]),
                xytext=(6, 9), textcoords="offset points",
                color=S.INK2, fontsize=5.4, va="bottom", zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK3,
                                shrinkA=1, shrinkB=1.5))

    # supporting labels: text in the genuinely empty sectors, short leaders
    ax.annotate("DFT bins", xy=(BINS[1], med_at_bins[1]),
                xytext=(np.deg2rad(15), 10.8), ha="right", va="center",
                color=S.INK3, fontsize=5.2, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK3,
                                shrinkA=1, shrinkB=1.5))
    ax.annotate("27 cells", xy=(np.deg2rad(218.0), 7.2),
                xytext=(np.deg2rad(232), 10.2), ha="right", va="center",
                color=S.INK3, fontsize=5.2, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=S.INK3,
                                shrinkA=1, shrinkB=1.5))

    paths = S.save(fig, OUT, W_MM, H_MM)
    print("qa: min %.4f at theta=0 | median peak %.4f at %.4f rad | "
          "p95 max %.3f | per-cell peaks %.3f..%.3f | files %s"
          % (med[i0], med[ip], ang[ip], p95.max(), lo, hi,
             [p.name for p in paths]))


if __name__ == "__main__":
    main()
