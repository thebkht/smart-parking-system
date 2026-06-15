#!/usr/bin/env python3
"""Shared color palette and matplotlib style for report graphs.

The palette mirrors the system architecture diagram
(``outputs/figures/architecture.tex``) and the annotated detection figure
(``figures/detection.jpg``) so every generated graph matches the rest of the
report. Green/red are reserved for the free/occupied semantics used in those
figures; the cyan/purple/magenta hues are used for neutral data series.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- Architecture palette (HTML hex from architecture.tex) -----------------
FREE_GREEN = "#1A7A4A"  # \definecolor{free}
OCC_RED = "#C0392B"  # \definecolor{occ}
PURPLE = "#8E44AD"  # \definecolor{pp}
CYAN = "#2E9BC0"  # \definecolor{cy}
MAGENTA = "#C0399B"  # \definecolor{mg}

INK = "#2B2B2B"
GRID = "#B9C2C9"

# Categorical sequence for multi-series charts. Green/red are kept out of the
# default rotation so they stay tied to the free/occupied meaning.
SERIES = [CYAN, PURPLE, MAGENTA, FREE_GREEN, OCC_RED]

# Sequential colormap (white -> cyan -> deep blue) for confusion matrices.
PALETTE_BLUES = LinearSegmentedColormap.from_list(
    "arch_blues", ["#ffffff", "#cfe8f1", CYAN, "#1a5d77", "#0e3346"]
)


def apply_style() -> None:
    """Apply the shared rcParams for a consistent report look."""
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
