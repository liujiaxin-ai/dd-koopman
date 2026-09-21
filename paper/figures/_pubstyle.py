"""Shared publication design system for the DD-Koopman figure set (v2).

Skills composed here (each contributes a specific rule, not a mood):
  * academic-figure-skill    - typography baseline (Arial, 5 pt floor), semantic
                               palette roles, 89/183 mm column contract, vector
                               first export, QA checklist.
  * nature-figure            - figure contract, hero-panel composition, editable
                               SVG text (svg.fonttype=none), Type-42 PDF, panel
                               lettering and glyph-floor audit.
  * publication-chart-skill  - chart choice per claim, publication QA pass,
                               companion-table discipline.
  * scientific-visualization - honest encodings, uncertainty always explicit,
                               redundant colour+shape coding, provenance notes.
  * academic-figure-designer - reading order, semantic colour binding, text
                               capacity per region, no decorative ink.
  * multipanel-layout        - anti-redundancy audit, hero panel, narrative.

Physical contract: ICASSP single column = 88.9 mm, double column = 181.9 mm.
Figures are created at their final size and exported without bbox_inches
rescaling, so the exported glyph sizes are the glyph sizes the reader sees.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: F401


# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #
def repo_root(start=None) -> Path:
    """Locate the repository root from any script position or layout.

    The figure scripts sit in <root>/figures in the development worktree and in
    <root>/paper/figures in the released package; walking up to the directory
    that holds results/analysis keeps both layouts working.
    """
    here = Path(start or __file__).resolve()
    here = here if here.is_dir() else here.parent
    for cand in (here, *here.parents):
        if (cand / "results" / "analysis").is_dir():
            return cand
    return here

# --------------------------------------------------------------------------- #
# Physical layout
# --------------------------------------------------------------------------- #
MM = 1.0 / 25.4
SINGLE_MM = 88.9          # IEEE / ICASSP single column
DOUBLE_MM = 181.9         # IEEE / ICASSP double column (\textwidth)

# --------------------------------------------------------------------------- #
# Semantics-first palette. One family per role; colour is never the only code.
#   ours  = terracotta (the reported operator)
#   pub   = deep blue  (the published 21.9 M model)
#   ctrl  = sand       (capacity controls that share our parameter scale)
#   base  = greys      (re-implemented baselines)
#   slate = neutral ordered ramp for physics quantities that belong to no model
# --------------------------------------------------------------------------- #
OURS = "#BC4A2B"
OURS_D = "#8A3419"
OURS_L = "#E9A686"
OURS_XL = "#F8E3D7"
PUB = "#1F4E79"
PUB_D = "#123A5C"
PUB_L = "#9CBBD8"
PUB_XL = "#E4EDF5"
CTRL = "#E6C4A7"
CTRL_EDGE = "#8A6446"
BASE = "#B6BABE"
BASE_D = "#787D83"
INK = "#1A1A1A"
INK2 = "#4A4F55"
INK3 = "#8A9096"
HAIR = "#DFE3E6"
GRID = "#EFF1F3"
WHITE = "#FFFFFF"

SLATE_RAMP = ["#EDEFF1", "#C6CDD3", "#98A2AA", "#63707A", "#2E3941"]

# Channel-model identity (dataset axis, not a model axis): Okabe-Ito hues with a
# distinct marker each, so the three models survive greyscale and colour-vision
# deficiency without competing with the coral/navy model semantics.
# Palette decision (academic-figure-workflow/palettes.md): branch = scene
# ("data behaviour / comparison"), profile = classic-technical.
CM_COLOUR = {"A": "#E69F00", "C": "#009E73", "D": "#0072B2"}
CM_MARKER = {"A": "o", "C": "s", "D": "^"}
CM_LABEL = {"A": "cm A", "C": "cm C", "D": "cm D"}


def use_style(base_pt: float = 6.3, lw: float = 0.6) -> None:
    """Apply the shared rcParams (mandatory editable-text rules included)."""
    warnings.filterwarnings("ignore", message=".*Helvetica.*")
    mpl.rcParams.update({
        # --- mandatory, from the nature-figure python quick-start -------------
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "svg.fonttype": "none",     # keep SVG text as editable <text> nodes
        "pdf.fonttype": 42,         # embed TrueType, keep PDF text editable
        # --- type scale (5 pt floor enforced by the QA audit) ----------------
        "font.size": base_pt,
        "axes.titlesize": base_pt - 0.1,
        "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt - 0.4,
        "ytick.labelsize": base_pt - 0.4,
        "legend.fontsize": base_pt - 0.5,
        # --- ink -------------------------------------------------------------
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": lw,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": lw,
        "ytick.major.width": lw,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.32,
        "axes.axisbelow": True,
        "figure.dpi": 200,
        "savefig.transparent": False,
    })


def figure(width_mm: float, height_mm: float, **kwargs):
    """Create a figure whose page size is exactly width_mm x height_mm."""
    return plt.figure(figsize=(width_mm * MM, height_mm * MM), **kwargs)


def axes(fig, rect_mm: tuple[float, float, float, float], width_mm: float,
         height_mm: float, **kwargs):
    """Add axes addressed in millimetres from the figure's bottom-left corner."""
    left, bottom, width, height = rect_mm
    return fig.add_axes((left / width_mm, bottom / height_mm,
                         width / width_mm, height / height_mm), **kwargs)


def panel_letter(fig, ax, letter: str, dx_pt: float = -10.0, dy_pt: float = 1.5,
                 size: float = 7.6) -> None:
    """Bold lowercase panel letter offset from the axes' top-left corner.

    Offsets are in points, so the letter keeps the same visual distance from
    its panel whatever the panel's aspect ratio is.
    """
    bb = ax.get_position()
    width_in, height_in = fig.get_size_inches()
    dx = dx_pt / 72.0 / float(width_in)
    dy = dy_pt / 72.0 / float(height_in)
    fig.text(bb.x0 + dx, bb.y1 + dy, letter, fontsize=size,
             fontweight="bold", color=INK, ha="left", va="bottom")


def panel_label(ax, letter: str, dx: float = -0.13, dy: float = 1.06,
                size: float = 7.0) -> None:
    """Axes-relative variant kept for scripts that predate panel_letter."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", color=INK, ha="left", va="bottom")


def hairline(ax, axis: str = "y", values=None, color: str = GRID,
             lw: float = 0.5, **kwargs) -> None:
    """Faint guide lines, only where the eye genuinely needs them."""
    draw = ax.axhline if axis == "y" else ax.axvline
    for value in ([] if values is None else values):
        draw(value, color=color, lw=lw, zorder=0, **kwargs)


def zero_reference(ax, axis: str = "y", color: str = INK2, lw: float = 0.6):
    """Dashed zero line, the one reference convention shared by all panels."""
    draw = ax.axhline if axis == "y" else ax.axvline
    draw(0.0, color=color, lw=lw, linestyle=(0, (3, 2)), zorder=1)


def bare(ax, keep: str = "lb") -> None:
    """Strip spines and ticks for image panels, keeping the chosen spines."""
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)
    ax.tick_params(length=0)


def diverging_cmap() -> LinearSegmentedColormap:
    """Paired-difference map: navy = published better, coral = ours better."""
    return LinearSegmentedColormap.from_list(
        "paired", [PUB_D, PUB, PUB_L, "#F7F9FB", OURS_L, OURS, OURS_D], N=256)


def energy_cmap() -> LinearSegmentedColormap:
    """Sequential map for |H| energy: light floor -> deep ink peak.

    Light-low rather than dark-low, so the sparse delay-Doppler support reads
    as dark structure on paper-white instead of a black field and the ramp
    stays perceptually monotone in greyscale print.
    """
    return LinearSegmentedColormap.from_list(
        "energy", ["#FFFFFF", "#E7EDF2", "#BDD0DE", "#84A6C0", "#415F7D",
                   "#1B3A57"], N=256)


def slate_cmap() -> LinearSegmentedColormap:
    """Neutral ordered ramp (physics quantities that belong to no model)."""
    return LinearSegmentedColormap.from_list("slate", SLATE_RAMP, N=256)


def hue_cmap() -> LinearSegmentedColormap:
    """Ordered speed/condition ramp in the paper's own cool family."""
    return LinearSegmentedColormap.from_list(
        "speeds", ["#CFE0EE", "#7FA7CC", "#3F6EA8", "#1D4477", "#0E2C4F"],
        N=256)


def save(fig, base: str | Path, width_mm: float, height_mm: float,
         pubfig_engine: bool = False) -> list[Path]:
    """Export the exact-size publication bundle: pdf, svg, png, tiff.

    bbox_inches is never used, so the page keeps the final column width and the
    designed glyph sizes survive to print.
    """
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
                    ("tiff", {"dpi": 600})):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    return written


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000,
                 seed: int = 20260913, alpha: float = 0.05) -> tuple[float, float]:
    """Paired bootstrap CI over cells, matching the paper's statistics note."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def short_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "\u2212")


def sig_text(value: float, digits: int = 3) -> str:
    """Signed value with a true minus sign."""
    return f"{value:+.{digits}f}".replace("-", "\u2212")
