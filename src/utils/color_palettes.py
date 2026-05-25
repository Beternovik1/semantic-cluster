"""Color palette management for all visualization libraries.

Single source of truth for palette colors, backgrounds, and shape
encodings. No module in the codebase hardcodes a color or background
value directly — all consume this module.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.colors import to_hex

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

SUPPORTED_PALETTES = frozenset({"viridis", "cividis", "plasma", "inferno", "magma"})
DARK_BACKGROUND_PALETTES = frozenset({"inferno", "magma"})

MARKER_SHAPES: tuple[str, ...] = (
    "circle",
    "square",
    "triangle",
    "diamond",
    "inverted_triangle",
    "hex",
)


# ── Module-level helpers ───────────────────────────────────────────────────

def _validate(name: str) -> None:
    if name not in SUPPORTED_PALETTES:
        raise ValueError(
            f"Unsupported palette: '{name}'. "
            f"Supported palettes: {', '.join(sorted(SUPPORTED_PALETTES))}."
        )


def _sample_colors(name: str, n_colors: int) -> list[str]:
    """Return *n_colors* hex strings sampled evenly from the colormap."""
    import matplotlib  # delay import — no GUI backend needed

    cmap = matplotlib.colormaps[name]
    indices = np.linspace(0, 1, n_colors)
    return [to_hex(cmap(i)) for i in indices]


# ── Public helpers ─────────────────────────────────────────────────────────

def get_bokeh_palette(name: str, n_colors: int = 20) -> list[str]:
    """Return a list of hex colour strings for Bokeh."""
    _validate(name)
    return _sample_colors(name, n_colors)


def get_matplotlib_palette(name: str, n_colors: int = 20) -> list[str]:
    """Return a list of normalized RGBA tuples for Matplotlib."""
    _validate(name)

    import matplotlib

    cmap = matplotlib.colormaps[name]
    indices = np.linspace(0, 1, n_colors)
    return [cmap(i) for i in indices]


def get_plotly_palette(name: str, n_colors: int = 20) -> list[str]:
    """Return a list of hex colour strings for Plotly."""
    _validate(name)
    return _sample_colors(name, n_colors)


def get_background(name: str) -> str:
    """Return hex background colour appropriate for the palette."""
    _validate(name)
    return "#1a1a2e" if name in DARK_BACKGROUND_PALETTES else "#ffffff"


def get_plotly_template(name: str) -> str:
    """Return Plotly template name appropriate for the palette."""
    _validate(name)
    return "plotly_dark" if name in DARK_BACKGROUND_PALETTES else "plotly_white"


# ── PaletteManager class ───────────────────────────────────────────────────

class PaletteManager:
    """Holds a single palette and exposes it in every required format.

    Args:
        name: One of ``SUPPORTED_PALETTES``.

    Raises:
        ValueError: If *name* is not a supported palette.
    """

    def __init__(self, name: str) -> None:
        _validate(name)
        self.name = name
        self.is_dark = name in DARK_BACKGROUND_PALETTES

    # ── Colour accessors ───────────────────────────────────────────

    def get_bokeh_palette(self, n_colors: int = 20) -> list[str]:
        return get_bokeh_palette(self.name, n_colors)

    def get_matplotlib_palette(self, n_colors: int = 20) -> list[str]:
        return get_matplotlib_palette(self.name, n_colors)

    def get_plotly_palette(self, n_colors: int = 20) -> list[str]:
        return get_plotly_palette(self.name, n_colors)

    # ── Background / template ─────────────────────────────────────

    def get_background(self) -> str:
        return get_background(self.name)

    def get_plotly_template(self) -> str:
        return get_plotly_template(self.name)

    # ── Shapes (redundant encoding) ────────────────────────────────

    @staticmethod
    def get_shapes() -> tuple[str, ...]:
        """Marker shapes for Bokeh scatter plots (cycles if >6 topics)."""
        return MARKER_SHAPES

    def __repr__(self) -> str:
        return f"PaletteManager(name={self.name!r}, is_dark={self.is_dark})"
