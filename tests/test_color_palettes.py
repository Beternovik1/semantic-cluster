"""Unit tests for src/utils/color_palettes.py."""

import pytest

from src.utils.color_palettes import (
    DARK_BACKGROUND_PALETTES,
    MARKER_SHAPES,
    SUPPORTED_PALETTES,
    PaletteManager,
    get_background,
    get_bokeh_palette,
    get_matplotlib_palette,
    get_plotly_palette,
    get_plotly_template,
)


# ── Invalid palette ────────────────────────────────────────────────────────

class TestInvalidPalette:
    @pytest.mark.parametrize(
        "bad", ["unknown_palette", "rainbow", "jet", "", "viridis_light"],
    )
    def test_module_functions_raise_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Unsupported palette"):
            get_bokeh_palette(bad)

    def test_palette_manager_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported palette"):
            PaletteManager("unknown_palette")


# ── Colour count ───────────────────────────────────────────────────────────

class TestColourCount:
    @pytest.mark.parametrize("name", ["viridis", "cividis", "plasma", "inferno", "magma"])
    @pytest.mark.parametrize("n", [20, 5, 1, 50])
    def test_bokeh_palette_returns_correct_count(
        self, name: str, n: int,
    ) -> None:
        colours = get_bokeh_palette(name, n)
        assert len(colours) == n
        assert all(isinstance(c, str) and c.startswith("#") for c in colours)

    @pytest.mark.parametrize("name", ["viridis", "cividis", "plasma", "inferno", "magma"])
    @pytest.mark.parametrize("n", [20, 5, 1, 50])
    def test_plotly_palette_returns_correct_count(
        self, name: str, n: int,
    ) -> None:
        colours = get_plotly_palette(name, n)
        assert len(colours) == n
        assert all(isinstance(c, str) and c.startswith("#") for c in colours)

    @pytest.mark.parametrize("name", ["viridis", "cividis", "plasma", "inferno", "magma"])
    @pytest.mark.parametrize("n", [20, 5, 1, 50])
    def test_matplotlib_palette_returns_correct_count(
        self, name: str, n: int,
    ) -> None:
        colours = get_matplotlib_palette(name, n)
        assert len(colours) == n
        # Matplotlib returns RGBA tuples — check structure
        for c in colours:
            assert isinstance(c, tuple)
            assert len(c) == 4
            assert all(0.0 <= v <= 1.0 for v in c)


# ── Background logic ───────────────────────────────────────────────────────

class TestDarkBackground:
    @pytest.mark.parametrize("name", ["inferno", "magma"])
    def test_background_is_dark(self, name: str) -> None:
        assert get_background(name) == "#1a1a2e"

    @pytest.mark.parametrize("name", ["inferno", "magma"])
    def test_plotly_template_is_dark(self, name: str) -> None:
        assert get_plotly_template(name) == "plotly_dark"

    @pytest.mark.parametrize("name", ["inferno", "magma"])
    def test_palette_manager_is_dark_flag(self, name: str) -> None:
        pm = PaletteManager(name)
        assert pm.is_dark is True
        assert pm.get_background() == "#1a1a2e"
        assert pm.get_plotly_template() == "plotly_dark"


class TestLightBackground:
    @pytest.mark.parametrize("name", ["viridis", "cividis", "plasma"])
    def test_background_is_white(self, name: str) -> None:
        assert get_background(name) == "#ffffff"

    @pytest.mark.parametrize("name", ["viridis", "cividis", "plasma"])
    def test_plotly_template_is_light(self, name: str) -> None:
        assert get_plotly_template(name) == "plotly_white"

    @pytest.mark.parametrize("name", ["viridis", "cividis", "plasma"])
    def test_palette_manager_is_dark_false(self, name: str) -> None:
        pm = PaletteManager(name)
        assert pm.is_dark is False
        assert pm.get_background() == "#ffffff"
        assert pm.get_plotly_template() == "plotly_white"


# ── DARK_BACKGROUND_PALETTES ───────────────────────────────────────────────

class TestDarkPaletteSet:
    def test_contains_inferno_and_magma(self) -> None:
        assert DARK_BACKGROUND_PALETTES == {"inferno", "magma"}


# ── Marker shapes ──────────────────────────────────────────────────────────

class TestMarkerShapes:
    def test_exact_six_shapes(self) -> None:
        assert MARKER_SHAPES == (
            "circle",
            "square",
            "triangle",
            "diamond",
            "inverted_triangle",
            "hex",
        )

    def test_palette_manager_get_shapes_returns_same(self) -> None:
        assert PaletteManager.get_shapes() == MARKER_SHAPES


# ── SUPPORTED_PALETTES ─────────────────────────────────────────────────────

class TestSupportedPalettes:
    def test_exact_five_palettes(self) -> None:
        assert SUPPORTED_PALETTES == {"viridis", "cividis", "plasma", "inferno", "magma"}
