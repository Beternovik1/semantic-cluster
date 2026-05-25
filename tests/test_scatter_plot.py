"""Unit tests for src/viz/scatter_plot.py."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.viz.scatter_plot import ScatterPlot
from src.utils.validators import MAX_SCATTER_POINTS


# ── Helpers / fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def palette():
    pm = MagicMock()
    pm.get_bokeh_palette.return_value = [
        "#440154", "#3b528b", "#21918c", "#5ec962", "#fde725",
    ]
    pm.get_background.return_value = "#ffffff"
    pm.get_shapes.return_value = (
        "circle", "square", "triangle", "diamond",
    )
    return pm


@pytest.fixture
def df_pos():
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame(
        {
            "clean_text": [f"positive review {i}" for i in range(n)],
            "topic_id": [0, 1] * (n // 2),
            "topic_label": ["topic_a", "topic_b"] * (n // 2),
            "umap_x": rng.uniform(-5, 5, n),
            "umap_y": rng.uniform(-5, 5, n),
            "sentiment": "pos",
            "polarity": rng.uniform(0.1, 0.9, n),
            "representative_doc": [True] + [False] * (n - 1),
            "concept_similarity": rng.uniform(-0.3, 0.8, n),
        }
    )


@pytest.fixture
def df_neg():
    rng = np.random.default_rng(99)
    n = 20
    return pd.DataFrame(
        {
            "clean_text": [f"negative review {i}" for i in range(n)],
            "topic_id": [2, -1] * (n // 2),
            "topic_label": ["topic_c", "noise"] * (n // 2),
            "umap_x": rng.uniform(-5, 5, n),
            "umap_y": rng.uniform(-5, 5, n),
            "sentiment": "neg",
            "polarity": rng.uniform(-0.9, -0.1, n),
            "representative_doc": [False] * n,
            "concept_similarity": rng.uniform(-0.3, 0.8, n),
        }
    )


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path


@pytest.fixture
def scatter(palette):
    return ScatterPlot(palette)


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_palette(self, palette):
        s = ScatterPlot(palette)
        assert s._pm is palette


# ── Combination ──────────────────────────────────────────────────────────────


class TestCombinePartitions:
    def test_combines_pos_and_neg(self, scatter, df_pos, df_neg):
        df = scatter._combine_partitions(df_pos, df_neg)
        assert len(df) == len(df_pos) + len(df_neg)

    def test_combine_preserves_columns(self, scatter, df_pos, df_neg):
        df = scatter._combine_partitions(df_pos, df_neg)
        for col in df_pos.columns:
            assert col in df.columns

    def test_empty_partitions(self, scatter):
        empty = pd.DataFrame(
            {
                "clean_text": [],
                "topic_id": [],
                "topic_label": [],
                "umap_x": [],
                "umap_y": [],
                "sentiment": [],
                "polarity": [],
                "representative_doc": [],
            }
        )
        df = scatter._combine_partitions(empty, empty)
        assert len(df) == 0


# ── Down-sampling ────────────────────────────────────────────────────────────


class TestDownsample:
    def test_returns_same_when_under_limit(self, scatter):
        df = pd.DataFrame({"x": range(10)})
        result = scatter._downsample(df)
        assert len(result) == 10

    def test_reduces_when_over_limit(self, scatter):
        df = pd.DataFrame({"x": range(MAX_SCATTER_POINTS + 100)})
        result = scatter._downsample(df)
        assert len(result) == MAX_SCATTER_POINTS

    def test_info_logged(self, scatter, caplog):
        caplog.set_level(logging.INFO)
        df = pd.DataFrame({"x": range(MAX_SCATTER_POINTS + 100)})
        scatter._downsample(df)
        assert "Down-sampling" in caplog.text


class TestDownsamplePreservingRepresentatives:
    def test_keeps_all_representatives(self, scatter):
        n = MAX_SCATTER_POINTS + 100
        df = pd.DataFrame(
            {
                "x": range(n),
                "representative_doc": [True] * 5 + [False] * (n - 5),
            }
        )
        result = scatter._downsample_preserving_representatives(df)
        assert result["representative_doc"].sum() == 5

    def test_reduces_to_limit(self, scatter):
        n = MAX_SCATTER_POINTS + 200
        df = pd.DataFrame(
            {
                "x": range(n),
                "representative_doc": [True] * 5 + [False] * (n - 5),
            }
        )
        result = scatter._downsample_preserving_representatives(df)
        assert len(result) == MAX_SCATTER_POINTS

    def test_returns_same_when_under_limit(self, scatter):
        df = pd.DataFrame(
            {
                "x": range(50),
                "representative_doc": [True] * 3 + [False] * 47,
            }
        )
        result = scatter._downsample_preserving_representatives(df)
        assert len(result) == 50

    def test_representatives_exceed_limit(self, scatter):
        n = 2000
        df = pd.DataFrame(
            {
                "x": range(n),
                "representative_doc": [True] * (MAX_SCATTER_POINTS + 10)
                + [False] * (n - MAX_SCATTER_POINTS - 10),
            }
        )
        result = scatter._downsample_preserving_representatives(df)
        assert len(result) == MAX_SCATTER_POINTS
        assert result["representative_doc"].all()

    def test_info_logged(self, scatter, caplog):
        caplog.set_level(logging.INFO)
        n = MAX_SCATTER_POINTS + 100
        df = pd.DataFrame(
            {
                "x": range(n),
                "representative_doc": [True] * 5 + [False] * (n - 5),
            }
        )
        scatter._downsample_preserving_representatives(df)
        assert "Down-sampling" in caplog.text


# ── Topic scatter guard ──────────────────────────────────────────────────────


class TestTopicScatterGuard:
    def test_returns_none_when_umap_null(self, scatter, output_dir, caplog):
        df = pd.DataFrame(
            {
                "clean_text": ["a"],
                "topic_id": [-1],
                "topic_label": ["fallback"],
                "umap_x": [None],
                "umap_y": [None],
                "sentiment": ["neg"],
                "polarity": [0.0],
                "representative_doc": [False],
            }
        )
        caplog.set_level(logging.WARNING)
        result = scatter.generate_topic_scatter(
            df_pos=df,
            df_neg=pd.DataFrame(
                {
                    "clean_text": [],
                    "topic_id": [],
                    "topic_label": [],
                    "umap_x": [],
                    "umap_y": [],
                    "sentiment": [],
                    "polarity": [],
                    "representative_doc": [],
                }
            ),
            title="Test",
            output_dir=output_dir,
        )
        assert result is None
        assert "all null" in caplog.text


# ── Semantic scatter guard ───────────────────────────────────────────────────


class TestSemanticScatterGuard:
    def test_returns_none_when_umap_null(
        self, scatter, output_dir, caplog
    ):
        df = pd.DataFrame(
            {
                "clean_text": ["a"],
                "umap_x": [None],
                "umap_y": [None],
                "sentiment": ["neg"],
                "concept_similarity": [0.0],
            }
        )
        caplog.set_level(logging.WARNING)
        result = scatter.generate_semantic_scatter(
            df=df,
            title="Test",
            concept_name="test",
            output_dir=output_dir,
        )
        assert result is None
        assert "skipping" in caplog.text


# ── Topic scatter output ─────────────────────────────────────────────────────


class TestTopicScatterOutput:
    def test_returns_html_string(self, scatter, df_pos, df_neg, output_dir):
        result = scatter.generate_topic_scatter(
            df_pos=df_pos,
            df_neg=df_neg,
            title="Topics",
            output_dir=output_dir,
        )
        assert isinstance(result, str)
        assert result.startswith("<!DOCTYPE html>") or "<html" in result

    def test_saves_file(self, scatter, df_pos, df_neg, output_dir):
        scatter.generate_topic_scatter(
            df_pos=df_pos,
            df_neg=df_neg,
            title="Topics",
            output_dir=output_dir,
        )
        assert (output_dir / "scatter_topics.html").exists()

    def test_file_contains_bokeh_inline(self, scatter, df_pos, df_neg, output_dir):
        scatter.generate_topic_scatter(
            df_pos=df_pos,
            df_neg=df_neg,
            title="Topics",
            output_dir=output_dir,
        )
        content = (output_dir / "scatter_topics.html").read_text(
            encoding="utf-8"
        )
        # Bokeh INLINE embeds Bokeh JS directly — look for tell-tale signs
        assert "Bokeh" in content

    def test_html_standalone_no_cdn(self, scatter, df_pos, df_neg, output_dir):
        scatter.generate_topic_scatter(
            df_pos=df_pos,
            df_neg=df_neg,
            title="Topics",
            output_dir=output_dir,
        )
        content = (output_dir / "scatter_topics.html").read_text(
            encoding="utf-8"
        )
        assert "cdn.bokeh.org" not in content


# ── Semantic scatter output ──────────────────────────────────────────────────


class TestSemanticScatterOutput:
    def test_returns_html_string(self, scatter, df_pos, output_dir):
        result = scatter.generate_semantic_scatter(
            df=df_pos,
            title="Semantic",
            concept_name="test",
            output_dir=output_dir,
        )
        assert isinstance(result, str)
        assert result.startswith("<!DOCTYPE html>") or "<html" in result

    def test_saves_file(self, scatter, df_pos, output_dir):
        scatter.generate_semantic_scatter(
            df=df_pos,
            title="Semantic",
            concept_name="test",
            output_dir=output_dir,
        )
        assert (output_dir / "scatter_semantic.html").exists()

    def test_html_standalone_no_cdn(self, scatter, df_pos, output_dir):
        scatter.generate_semantic_scatter(
            df=df_pos,
            title="Semantic",
            concept_name="test",
            output_dir=output_dir,
        )
        content = (output_dir / "scatter_semantic.html").read_text(
            encoding="utf-8"
        )
        assert "cdn.bokeh.org" not in content


# ── generate() orchestrator ──────────────────────────────────────────────────


class TestGenerate:
    def test_returns_two_element_tuple(
        self, scatter, df_pos, df_neg, output_dir
    ):
        result = scatter.generate(
            df_pos=df_pos,
            df_neg=df_neg,
            concept="test concept",
            output_dir=output_dir,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_may_be_none(self, scatter, df_neg, output_dir):
        # pass empty pos so topic render is bypassed via nulls
        empty_pos = pd.DataFrame(
            {
                "clean_text": [],
                "topic_id": [],
                "topic_label": [],
                "umap_x": [],
                "umap_y": [],
                "sentiment": [],
                "polarity": [],
                "representative_doc": [],
                "concept_similarity": [],
            }
        )
        r1, r2 = scatter.generate(
            df_pos=empty_pos,
            df_neg=df_neg,
            concept="test",
            output_dir=output_dir,
        )
        assert r1 is None or isinstance(r1, str)
        assert isinstance(r2, str) or r2 is None

    def test_both_files_saved(
        self, scatter, df_pos, df_neg, output_dir
    ):
        scatter.generate(
            df_pos=df_pos,
            df_neg=df_neg,
            concept="test",
            output_dir=output_dir,
        )
        assert (output_dir / "scatter_topics.html").exists()
        assert (output_dir / "scatter_semantic.html").exists()
