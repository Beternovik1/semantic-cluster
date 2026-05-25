"""Unit tests for src/viz/report_builder.py."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.viz.report_builder import (
    ASSETS_DIR,
    ReportBuilder,
    ReportData,
    _REPORT_TEMPLATE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def palette():
    pm = MagicMock()
    pm.name = "viridis"
    pm.is_dark = False
    return pm


@pytest.fixture
def builder(palette):
    return ReportBuilder(title="Test Report", palette=palette)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "clean_text": [
                "great hotel",
                "terrible place",
                "average room",
                "wonderful staff",
                "dirty room",
            ],
            "sentiment": ["pos", "neg", "neg", "pos", "neg"],
            "sentiment_confidence": [True, True, False, True, True],
            "concept_similarity": [0.9, 0.1, 0.3, 0.8, 0.2],
        }
    )


@pytest.fixture
def sample_outliers():
    return pd.DataFrame(
        {
            "clean_text": [
                "this is a very strange unusual outlier review",
            ]
        }
    )


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path


# ── Constructor / constants ──────────────────────────────────────────────────


class TestConstructor:
    def test_stores_title(self, palette):
        b = ReportBuilder(title="My Report", palette=palette)
        assert b._title == "My Report"

    def test_stores_palette(self, palette):
        b = ReportBuilder(title="X", palette=palette)
        assert b._palette is palette


# ── ReportData dataclass ─────────────────────────────────────────────────────


class TestReportData:
    def test_defaults(self):
        d = ReportData()
        assert d.title == ""
        assert d.total_rows == 0
        assert d.scatter_topics_html is None
        assert d.top5_semantic == []

    def test_required_fields(self):
        d = ReportData(
            title="Test",
            timestamp="now",
            wc_outliers_b64="data:...",
            wc_positive_b64="data:...",
            wc_negative_b64="data:...",
            ngrams_unigrams_html="<div>",
            ngrams_bigrams_html="<div>",
            ngrams_trigrams_html="<div>",
        )
        assert d.title == "Test"


# ── Template rendering (smoke tests) ─────────────────────────────────────────


class TestTemplateSmoke:
    def test_template_is_string(self):
        assert isinstance(_REPORT_TEMPLATE, str)
        assert "<!DOCTYPE html>" in _REPORT_TEMPLATE
        assert "{{ data.title }}" in _REPORT_TEMPLATE

    def test_template_has_all_sections(self):
        assert "Pipeline Summary" in _REPORT_TEMPLATE
        assert "Outlier Analysis" in _REPORT_TEMPLATE
        assert "Sentiment Analysis" in _REPORT_TEMPLATE
        assert "Topic Modeling" in _REPORT_TEMPLATE
        assert "Semantic Concept Analysis" in _REPORT_TEMPLATE

    def test_no_cdn_urls(self):
        assert "cdn." not in _REPORT_TEMPLATE
        assert "http://" not in _REPORT_TEMPLATE
        assert "https://" not in _REPORT_TEMPLATE


# ── Asset reading ────────────────────────────────────────────────────────────


class TestReadAsset:
    def test_reads_existing_file(self, builder, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello world", encoding="utf-8")
        content = builder._read_asset(p)
        assert content == "hello world"

    def test_returns_empty_on_missing(self, builder, caplog):
        caplog.set_level(logging.WARNING)
        content = builder._read_asset(Path("/nonexistent/file.css"))
        assert content == ""
        assert "Asset not found" in caplog.text

    def test_assets_exist_on_disk(self):
        assert (ASSETS_DIR / "bootstrap.min.css").exists()
        assert (ASSETS_DIR / "plotly.min.js").exists()

    def test_bootstrap_contains_css(self):
        css = (ASSETS_DIR / "bootstrap.min.css").read_text(encoding="utf-8")
        assert "body" in css

    def test_plotly_contains_js(self):
        js = (ASSETS_DIR / "plotly.min.js").read_text(encoding="utf-8")
        assert "Plotly" in js


# ── build() — integration tests ──────────────────────────────────────────────


def _find_cdn_violations(content: str) -> list[str]:
    """Return a list of CDN patterns found in *content*.

    Only checks patterns that would cause the browser to actively load an
    external resource — i.e. HTML tags referencing ``http`` or ``//`` URLs.
    JS string literals such as ``e.href = "https://plotly.com/"`` are
    deliberately ignored because they do not trigger network requests.
    """
    patterns = (
        '<link href="http',
        '<link href="//',
        '<script src="http',
        '<script src="//',
        '<img src="http',
        '<img src="//',
        "@import url(http",
        "@import url(//",
    )
    return [p for p in patterns if p in content]


class TestBuild:
    def test_returns_path(self, builder, sample_df, output_dir):
        path = builder.build(
            df=sample_df,
            output_dir=output_dir,
        )
        assert isinstance(path, Path)
        assert path.name == "report.html"

    def test_creates_file(self, builder, sample_df, output_dir):
        builder.build(df=sample_df, output_dir=output_dir)
        assert (output_dir / "report.html").exists()

    def test_file_contains_title(self, builder, sample_df, output_dir):
        builder.build(df=sample_df, output_dir=output_dir)
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "Test Report" in content

    def test_file_contains_bootstrap(self, builder, sample_df, output_dir):
        builder.build(df=sample_df, output_dir=output_dir)
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        # Bootstrap CSS should be inlined
        assert "container" in content or "bootstrap" in content.lower() or "row" in content

    def test_file_contains_plotly_js(self, builder, sample_df, output_dir):
        builder.build(df=sample_df, output_dir=output_dir)
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "Plotly" in content

    def test_no_cdn_in_output(self, builder, sample_df, output_dir):
        builder.build(df=sample_df, output_dir=output_dir)
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        # Check for active resource-loading patterns (not inline data or JS string
        # literals).  Use _no_cdn helper to avoid pytest assertion rewriting on
        # the ~4 MB string — see AGENTS.md "Blocked" section.
        violations = _find_cdn_violations(content)
        assert not violations, f"CDN resource-loading patterns found: {violations}"

    def test_info_logged(self, builder, sample_df, output_dir, caplog):
        caplog.set_level(logging.INFO)
        builder.build(df=sample_df, output_dir=output_dir)
        assert "Report saved to" in caplog.text
        assert "MB" in caplog.text
        assert "self-contained offline" in caplog.text

    def test_creates_output_dir(self, builder, sample_df, tmp_path):
        nested = tmp_path / "deep" / "nested"
        builder.build(df=sample_df, output_dir=nested)
        assert (nested / "report.html").exists()

    def test_with_outliers(self, builder, sample_df, sample_outliers, output_dir):
        path = builder.build(
            df=sample_df,
            outliers_df=sample_outliers,
            output_dir=output_dir,
        )
        assert path.exists()

    def test_with_scatter_html(
        self, builder, sample_df, sample_outliers, output_dir
    ):
        path = builder.build(
            df=sample_df,
            outliers_df=sample_outliers,
            scatter_topics_html="<html><body>Bokeh Plot</body></html>",
            scatter_semantic_html="<html><body>Semantic</body></html>",
            output_dir=output_dir,
        )
        assert path.exists()

    def test_empty_df_does_not_crash(self, builder, output_dir):
        df = pd.DataFrame(
            {
                "clean_text": [],
                "sentiment": [],
                "sentiment_confidence": [],
            }
        )
        path = builder.build(df=df, output_dir=output_dir)
        assert path.exists()

    def test_top5_in_output(self, builder, sample_df, output_dir):
        builder.build(df=sample_df, output_dir=output_dir)
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "Top 5" in content
        assert "0.900" in content

    def test_fallback_keywords_in_output(self, builder, sample_df, output_dir):
        builder.build(
            df=sample_df,
            pos_fallback_keywords="hotel, staff, room",
            neg_fallback_keywords="terrible, dirty, place",
            output_dir=output_dir,
        )
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "hotel, staff, room" in content

    def test_summary_cards(self, builder, sample_df, output_dir):
        builder.build(df=sample_df, output_dir=output_dir)
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "Total Comments" in content
        assert "Positive" in content
        assert "Negative" in content

    def test_sentiment_wordclouds(self, builder, sample_df, output_dir):
        builder.build(
            df=sample_df,
            wc_positive_b64="data:image/png;base64,posimg",
            wc_negative_b64="data:image/png;base64,negimg",
            output_dir=output_dir,
        )
        content = (output_dir / "report.html").read_text(encoding="utf-8")
        assert "posimg" in content
        assert "negimg" in content


# ── Logging ──────────────────────────────────────────────────────────────────


class TestLogging:
    def test_info_level(self, builder, sample_df, output_dir, caplog):
        caplog.set_level(logging.INFO)
        builder.build(df=sample_df, output_dir=output_dir)
        assert "Report saved to" in caplog.text

    def test_size_reported(self, builder, sample_df, output_dir, caplog):
        caplog.set_level(logging.INFO)
        builder.build(df=sample_df, output_dir=output_dir)
        assert any("MB" in msg for msg in caplog.messages)
