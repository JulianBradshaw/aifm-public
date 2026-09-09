"""
Centralized matplotlib styling for all plotting scripts.

This module provides consistent styling that matches the website's appearance.
The website uses an off-white background (#fffff8) which plots should match.

Usage:
    from plotting_utils.style import apply_style

    apply_style()  # Call once at start of script, after matplotlib import

    # For text boxes that should match the background:
    from plotting_utils.style import get_textbox_props
    plt.text(..., bbox=get_textbox_props())
"""

import matplotlib

# Website-matching colors (from app/globals.css --vivid-background)
BACKGROUND_COLOR = '#fffff8'

# Standard DPI for saved figures
DEFAULT_DPI = 150

# ---------------------------------------------------------------------------
# "AI-2030" plot style — matches the monorepo's `apps/ai2030-graphs` web app
# (app/globals.css + the figure chart components, e.g. TaskAutomationChart.tsx):
# a cream-paper theme in Courier New monospace, dark-gray text/spines, muted
# gray ticks/grid, and a deep-toned series palette (navy / red / olive / amber).
# ---------------------------------------------------------------------------
AI2030_BACKGROUND = '#FFFEF8'   # globals.css --background (cream paper)
AI2030_TEXT = '#333333'         # globals.css --foreground
AI2030_STROKE = '#374151'       # axis spines / bold labels
AI2030_MUTED = '#6b7280'        # tick labels / secondary text
AI2030_GRID = '#9ca3af'         # gridlines / borders

AI2030_MONOSPACE = [
    'Courier New', 'Nimbus Mono PS', 'Liberation Mono', 'DejaVu Sans Mono', 'monospace',
]

# Series palette from the ai2030-graphs charts (TaskAutomationChart / OutputShareChart).
AI2030_NAVY = '#1e3a5f'    # primary series ("Plan A")
AI2030_RED = '#5c1a1a'     # antagonist series ("Default World")
AI2030_OLIVE = '#3f6634'
AI2030_AMBER = '#c87a1f'
AI2030_PURPLE = '#412941'
AI2030_WARMGREY = '#78716c'
AI2030_LIGHTBLUE = '#4d77a8'
AI2030_LIGHTRED = '#a04444'

# Milestone reference-line palette, drawn from the app's deep-toned series colors.
# (The ai2030-graphs app has no AC/SAR/TED-AI milestone set of its own, so this
# maps the milestones onto on-brand hues.)
MILESTONE_COLORS = {
    'AC': AI2030_OLIVE,
    'SAR': AI2030_LIGHTBLUE,
    'TED-AI': AI2030_AMBER,
    'STRAT-AI': AI2030_PURPLE,
    'ASI': AI2030_LIGHTRED,
}


def apply_ai2030_style():
    """Apply the `ai2030-graphs` web-app theme to matplotlib.

    Matches apps/ai2030-graphs (globals.css + figure components): Courier New
    monospace, cream `#FFFEF8` background, dark-gray text/spines, muted gray tick
    labels and gridlines. Call once after importing matplotlib, before figures.
    """
    rc = matplotlib.rcParams
    rc['font.family'] = 'monospace'
    rc['font.monospace'] = AI2030_MONOSPACE
    rc['axes.facecolor'] = AI2030_BACKGROUND
    rc['figure.facecolor'] = AI2030_BACKGROUND
    rc['savefig.facecolor'] = AI2030_BACKGROUND
    rc['text.color'] = AI2030_TEXT
    rc['axes.labelcolor'] = AI2030_STROKE
    rc['axes.titlecolor'] = AI2030_TEXT
    rc['xtick.color'] = AI2030_MUTED
    rc['ytick.color'] = AI2030_MUTED
    rc['axes.edgecolor'] = AI2030_STROKE
    rc['grid.color'] = AI2030_GRID
    rc['savefig.dpi'] = 150


def apply_style():
    """Apply website-consistent styling to matplotlib.

    Call this once after importing matplotlib, before creating any figures.
    Sets:
    - Monospace font family
    - Off-white background color (#fffff8) for figures and axes
    - Default DPI of 150 for saved figures
    """
    matplotlib.rcParams['font.family'] = 'monospace'
    matplotlib.rcParams['figure.facecolor'] = BACKGROUND_COLOR
    matplotlib.rcParams['axes.facecolor'] = BACKGROUND_COLOR
    matplotlib.rcParams['savefig.facecolor'] = BACKGROUND_COLOR
    matplotlib.rcParams['savefig.dpi'] = DEFAULT_DPI


def get_textbox_props(alpha: float = 0.9, edgecolor: str = 'gray') -> dict:
    """Return consistent text box properties for annotations.

    Args:
        alpha: Transparency (default 0.9)
        edgecolor: Border color (default 'gray')

    Returns:
        dict suitable for bbox parameter in text/annotate calls
    """
    return {
        'facecolor': BACKGROUND_COLOR,
        'alpha': alpha,
        'edgecolor': edgecolor,
    }
