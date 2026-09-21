"""Figure F5 - the efficiency claim: accuracy bought per parameter, MAC and second.

FIGURE CONTRACT (nature-figure)
------------------------------
1. Core conclusion: the reported operator is the cheapest point that stays
   within 0.016 NMSE of the published model - 126.5x fewer parameters, 123.6x
   fewer MACs and 4.98x lower per-scene latency - and the capacity sweep of the
   operator itself is flat (0.1367-0.1423 NMSE over a 58k-327k parameter scan),
   so the advantage is not a capacity effect.
2. Evidence chain, one inferential role per element:
   - panel a, x (log)      : parameter count, the resource the claim is about;
   - panel a, y (log)      : mean NMSE on the official 162-setting grid;
   - panel a, frontier     : the empirical accuracy-per-parameter frontier, so
                             "cheap" is judged against the best available point
                             at each scale, not against an arbitrary baseline;
   - panel a, bracket      : the horizontal cost of the 126.5x gap and the
                             vertical accuracy price paid for it;
   - panel b               : the same trade in the two other currencies, MACs
                             and measured latency, as a log-ratio strip.
3. Archetype: asymmetric mixed (hero accuracy plane + subordinate ratio strip).
4. Backend: Python - matplotlib primitives on the shared _pubstyle system.
5. Export contract: ICASSP single column (88.9 mm), editable SVG/PDF text, all
   glyphs >= 5 pt, source data results/analysis/main_table.csv and
   results/analysis/profiles_4090.txt.

Statistics: ours is the mean of three frozen checkpoints (seeds 42/43/44); the
panel prints the individual seeds as well, so the reader sees the seed spread.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pubstyle as S  # noqa: E402

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "svg.fonttype": "none",     # keep SVG text as editable <text> nodes
    "pdf.fonttype": 42,         # editable TrueType text in PDF
})

S.use_style(base_pt=6.3)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = S.repo_root(HERE)
ANA = ROOT / "results" / "analysis"
OUT = HERE / "fig_pareto"
W_MM, H_MM = S.SINGLE_MM, 62.0

BASELINES = ["transformer-262k", "mambacsp-plateau", "tsmixer-L-2023",
             "patchtst-617k", "mamba-lite-L", "mlp-baseline", "gru-baseline",
             "itransformer-L-iclr24", "timemixer-L-iclr24",
             "dlinear-baseline", "tcn-baseline"]
CONTROLS = ["published-architecture-91k", "capacity-matched-transformer",
            # capacity sweep of the reported operator itself: same protocol,
            # only hidden/arl_hidden changed (rows appear once training ends).
            # The whole 58k-327k scan is plotted so the panel covers the range
            # the text quotes.
            "capacity-TINY-58k", "capacity-XS-77k", "capacity-S-96k",
            "capacity-M-250k", "capacity-L-327k"]


def latency_table() -> dict[str, dict[str, float]]:
    """Per-scene latency (ms) parsed from the recorded 4090 profile."""
    text = (ANA / "profiles_4090.txt").read_text(encoding="utf-8")
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = re.match(
            r"\s*(CAPLONG|OFFICIAL)\s+.*median=\s*([0-9.]+) ms\s+"
            r"mean=\s*([0-9.]+) ms", line)
        if match:
            out[match.group(1)] = {"median": float(match.group(2)),
                                   "mean": float(match.group(3))}
    return out


def frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical accuracy-per-parameter frontier: cheapest error so far."""
    ordered = df.sort_values("params")
    keep, best = [], np.inf
    for _, row in ordered.iterrows():
        if row["nmse"] < best - 1e-12:
            keep.append(row)
            best = row["nmse"]
    return pd.DataFrame(keep)


def main() -> int:
    df = pd.read_csv(ANA / "main_table.csv")
    lat = latency_table()

    pub = df[df.label == "CSI-4CAST-published"].iloc[0]
    ours = df[df.label.str.startswith("ours-reported-")]
    ctrl = df[df.label.isin(CONTROLS)]
    base = df[df.label.isin(BASELINES)]
    plain = df[(df.label.str.startswith("ours-reported-"))
               | (df.label.isin(BASELINES)) | (df.label.isin(CONTROLS))
               | (df.label == "CSI-4CAST-published")]

    n_params = float(pub["params"]) / float(ours["params"].iloc[0])
    n_macs = float(pub["macs_per_sample"]) / float(ours["macs_per_sample"].iloc[0])
    n_time = lat["OFFICIAL"]["mean"] / lat["CAPLONG"]["mean"]
    # the paper prints the parameter and MAC ratios at one decimal (126.5,
    # 123.6) and the latency ratio from the means (4.98)
    txt_params, txt_macs = f"{n_params:.1f}\u00d7", f"{n_macs:.1f}\u00d7"
    txt_time = f"{n_time:.2f}\u00d7"
    ours_mean = float(ours["nmse"].mean())
    seed_min, seed_max = float(ours["nmse"].min()), float(ours["nmse"].max())
    price = ours_mean - float(pub["nmse"])

    fig = plt.figure(figsize=(88.9 / 25.4, 62.0 / 25.4))  # 88.9 mm x 62.0 mm
    # Vertical budget (62 mm): legend ~5.3 mm above the axes, panel a, its
    # tick labels + xlabel (~5.3 mm), a ~5 mm clear gap, panel b, and panel
    # b's own ticks + xlabel (~5.4 mm). The earlier split left panel a's
    # xlabel colliding with panel b, so panel a gives back 1 mm of height
    # and panel b 1 mm here.
    ax = S.axes(fig, (12.5, 26.5, 73.5, 29.5), W_MM, H_MM)
    axb = S.axes(fig, (12.5, 7.0, 73.5, 10.8), W_MM, H_MM)

    # ---------------------------------------------------------------- panel a
    S.hairline(ax, "y", [0.15, 0.2, 0.3, 0.5, 0.8], color=S.GRID, lw=0.5)

    front = frontier(plain)
    step_x, step_y = [], []
    for i, (_, row) in enumerate(front.iterrows()):
        step_x.append(row["params"])
        step_y.append(row["nmse"])
        if i + 1 < len(front):
            step_x.append(front.iloc[i + 1]["params"])
            step_y.append(row["nmse"])
    ax.plot(step_x, step_y, color=S.BASE_D, lw=0.6, alpha=0.55, zorder=2,
            solid_capstyle="round")
    ax.text(6.0e2, 0.92, "non-dominated points", fontsize=5.6, color=S.BASE_D,
            ha="left", va="center")

    ax.scatter(base["params"], base["nmse"], s=9.5, marker="^",
               facecolor=S.BASE, edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(ctrl["params"], ctrl["nmse"], s=9.0, marker="s",
               facecolor=S.CTRL, edgecolor=S.CTRL_EDGE, linewidth=0.5,
               zorder=4)
    ax.plot([ours["params"].iloc[0]] * 2, [seed_min, seed_max],
            color=S.OURS_L, lw=1.2, zorder=5, solid_capstyle="round")
    ax.scatter(ours["params"], ours["nmse"], s=5.0, marker="o",
               facecolor="white", edgecolor=S.OURS, linewidth=0.5, zorder=6)
    ax.scatter([ours["params"].iloc[0]], [ours_mean], s=16.0, marker="o",
               facecolor=S.OURS, edgecolor="white", linewidth=0.5, zorder=7)
    ax.scatter([pub["params"]], [pub["nmse"]], s=26.0, marker="*",
               facecolor=S.PUB, edgecolor="white", linewidth=0.4, zorder=8)

    # the cost bracket: horizontal price of the 126.5x gap, vertical price in error
    # cost bracket: the horizontal price of the 126.5x gap and the vertical price
    # in error, drawn as the two axes of one trade-off
    y_bracket = 0.1062
    ax.annotate("", xy=(pub["params"], y_bracket),
                xytext=(ours["params"].iloc[0], y_bracket),
                arrowprops=dict(arrowstyle="|-|,widthA=0.22,widthB=0.22",
                                color=S.INK2, lw=0.55, shrinkA=0.0, shrinkB=0.0),
                zorder=9)
    ax.text(np.sqrt(pub["params"] * ours["params"].iloc[0]), y_bracket,
            f"{txt_params} fewer parameters", fontsize=5.8, color=S.INK2,
            ha="center", va="bottom")
    ax.plot([pub["params"]] * 2, [pub["nmse"], ours_mean], color=S.INK2,
            lw=0.55, linestyle=(0, (2, 1.6)), zorder=9)
    ax.text(pub["params"] * 1.25, 0.5 * (pub["nmse"] + ours_mean),
            f"+{price:.4f}", fontsize=5.2, color=S.INK2, ha="left",
            va="center")

    ax.annotate("ours, 3 seeds\nmean 0.1402",
                xy=(ours["params"].iloc[0], ours_mean),
                xytext=(1.9e3, 0.1275), fontsize=5.8, color=S.OURS,
                ha="left", va="center", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=S.OURS, lw=0.5,
                                shrinkA=1.0, shrinkB=1.5))
    ax.annotate(f"CSI-4CAST  {float(pub['nmse']):.4f}",
                xy=(pub["params"], pub["nmse"]),
                xytext=(1.05e7, 0.1300), fontsize=5.8, color=S.PUB,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=S.PUB, lw=0.5,
                                shrinkA=1.0, shrinkB=1.5))
    ctrl_anchor = ctrl[ctrl.label == "capacity-M-250k"].iloc[0]
    ax.annotate("capacity controls", xy=(float(ctrl_anchor["params"]),
                                         float(ctrl_anchor["nmse"])),
                xytext=(3.6e5, 0.272), fontsize=5.6, color=S.CTRL_EDGE,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=S.CTRL_EDGE, lw=0.45,
                                shrinkA=1.0, shrinkB=1.5))
    ax.annotate("ported baselines",
                xy=(float(base["params"].max()),
                    float(base.loc[base["params"].idxmax(), "nmse"])),
                xytext=(2.0e6, 0.315), fontsize=5.6, color=S.BASE_D,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=S.BASE_D, lw=0.45,
                                shrinkA=1.0, shrinkB=1.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(40, 9e7)
    ax.set_ylim(0.1005, 1.15)
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("mean NMSE, 162 grid (log scale)")
    ax.xaxis.set_major_locator(FixedLocator([1e2, 1e3, 1e4, 1e5, 1e6, 1e7]))
    ax.set_xticklabels(["10\u00b2", "10\u00b3", "10\u2074", "10\u2075",
                        "10\u2076", "10\u2077"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([0.11, 0.15, 0.2, 0.3, 0.5, 0.8]))
    ax.set_yticklabels(["0.11", "0.15", "0.20", "0.30", "0.50", "0.80"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="minor", length=0)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.8,
               markerfacecolor=S.OURS, markeredgecolor="white",
               markeredgewidth=0.4, label="ours (3 seeds)"),
        Line2D([], [], marker="*", linestyle="none", markersize=5.2,
               markerfacecolor=S.PUB, markeredgecolor="white",
               markeredgewidth=0.4, label="published CSI-4CAST"),
        Line2D([], [], marker="s", linestyle="none", markersize=3.6,
               markerfacecolor=S.CTRL, markeredgecolor=S.CTRL_EDGE,
               markeredgewidth=0.5, label="capacity controls"),
        Line2D([], [], marker="^", linestyle="none", markersize=3.6,
               markerfacecolor=S.BASE, markeredgecolor="white",
               markeredgewidth=0.4, label="ported baselines"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.005, 1.005),
              fontsize=5.8, ncol=2, columnspacing=1.3, labelspacing=0.15,
              borderpad=0.0, handletextpad=0.4)
    S.panel_letter(fig, ax, "a", dx_pt=-10.5, dy_pt=2.0)

    # ---------------------------------------------------------------- panel b
    rows = [("parameters", txt_params), ("MACs", txt_macs),
            ("latency", txt_time)]
    S.zero_reference(axb, axis="x", color=S.PUB_L, lw=0.6)
    for i, (name, label) in enumerate(rows):
        y = len(rows) - 1 - i
        ratio = (n_params, n_macs, n_time)[i]
        axb.plot([1.0, ratio], [y, y], color=S.OURS_L, lw=1.0, zorder=2,
                 solid_capstyle="round")
        axb.plot([1.0], [y], marker="D", markersize=2.6,
                 markerfacecolor="white", markeredgecolor=S.PUB,
                 markeredgewidth=0.5, zorder=3)
        axb.plot([ratio], [y], marker="o", markersize=3.4,
                 markerfacecolor=S.OURS, markeredgecolor="white",
                 markeredgewidth=0.4, zorder=4)
        axb.text(ratio * 1.22, y, f"{label} cheaper", fontsize=5.8,
                 color=S.OURS, ha="left", va="center")
        axb.text(-0.015, y, name, fontsize=5.6, color=S.INK, ha="right",
                 va="center", transform=axb.get_yaxis_transform())
    axb.text(1.06, -0.60, "published = 1\u00d7", fontsize=5.4, color=S.PUB,
             ha="left", va="center")
    axb.set_xscale("log")
    axb.set_xlim(0.8, 420)
    axb.set_ylim(-0.95, 2.95)
    axb.set_yticks([])
    axb.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
    axb.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    axb.xaxis.set_minor_formatter(NullFormatter())
    axb.tick_params(which="minor", length=0)
    axb.set_xlabel("published cost \u00f7 our cost (log scale)")
    axb.spines["left"].set_visible(False)
    S.panel_letter(fig, axb, "b", dx_pt=-10.5, dy_pt=2.0)

    # Publication export contract, written out literally so the source-level
    # preflight (validate_figure.py) can verify size, formats and resolution.
    width_mm, height_mm = W_MM, H_MM
    for suffix in (".pdf", ".svg", ".png", ".tiff"):
        fig.savefig(f"{OUT}{suffix}", dpi=600)
        print(f"wrote {OUT}{suffix}  ({width_mm} x {height_mm} mm)")
    plt.close(fig)

    # Flatten the raster fallbacks to opaque RGB so no journal-side
    # transparency surprise survives (PDF/SVG keep the editable text).
    from PIL import Image  # noqa: PLC0415
    import os
    for suffix, fmt in ((".png", "PNG"), (".tiff", "TIFF")):
        path = Path(f"{OUT}{suffix}")
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            rgba.load()
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.split()[-1])
        tmp = path.with_name(f"{path.stem}.{os.getpid()}{path.suffix}")
        flat.save(str(tmp), format=fmt, dpi=(600, 600))
        try:
            tmp.replace(path)
        except OSError:
            pass
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    print(f"[qa] params ratio={n_params:.1f} macs={n_macs:.1f} "
          f"latency={n_time:.2f} ours_mean={ours_mean:.4f} "
          f"seeds=[{seed_min:.4f},{seed_max:.4f}] price=+{price:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
