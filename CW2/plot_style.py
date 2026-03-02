# Shared plot styling for CW2 figures.
# Import this module before creating any plots to apply consistent styling.

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# --- Colour palette (colorblind-friendly) ---
BLUE    = "#3C78D8"
GREEN   = "#6AA84F"
RED     = "#E06666"
GREY    = "#999999"
ORANGE  = "#E69F00"
DARK    = "#2F2F2F"

PALETTE = [BLUE, GREEN, ORANGE, RED, GREY]

# --- Figure defaults ---
DPI = 200
FIGSIZE = (8, 6)

# --- Marker defaults ---
MARKER_SIZE = 60
MARKER_ALPHA = 0.75
MARKER_EDGE  = "#2F2F2F"
MARKER_EDGE_WIDTH = 0.6


def apply_style():
    """Apply global Seaborn + matplotlib style settings."""
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
    mpl.rcParams.update({
        # Figure
        "figure.figsize": FIGSIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",

        # Font
        "axes.titlesize": 14,
        "axes.labelsize": 12,

        # Grid (soften the seaborn defaults)
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.loc": "best",

        # Lines
        "lines.linewidth": 1.8,
    })


def scatter_points(ax, x, y, color=BLUE, label="Data points", **kwargs):
    """Draw styled scatter points on an axis."""
    return ax.scatter(
        x, y,
        s=kwargs.pop("s", MARKER_SIZE),
        c=color,
        alpha=kwargs.pop("alpha", MARKER_ALPHA),
        edgecolors=kwargs.pop("edgecolors", MARKER_EDGE),
        linewidths=kwargs.pop("linewidths", MARKER_EDGE_WIDTH),
        label=label,
        zorder=3,
        **kwargs,
    )


def reference_line(ax, min_val, max_val, label="Perfect prediction (y = x)"):
    """Draw a styled y=x reference line."""
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        color=RED, linestyle="--", linewidth=2.0,
        label=label, zorder=2,
    )


def finish_figure(fig, path):
    """Save and close a figure."""
    fig.tight_layout()
    fig.savefig(path)
    plt.show()
    plt.close(fig)
