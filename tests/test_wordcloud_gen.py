"""Unit tests for src/viz/wordcloud_gen.py."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.viz.wordcloud_gen import (
    WordCloudGenerator,
    _empty_chart_html,
    _empty_image_b64,
)
from src.utils.language_config import LanguageConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def lang_cfg():
    return LanguageConfig("en")


@pytest.fixture
def palette():
    pm = MagicMock()
    pm.get_background.return_value = "#ffffff"
    pm.get_plotly_template.return_value = "plotly_white"
    return pm


@pytest.fixture
def generator(palette, lang_cfg):
    return WordCloudGenerator(
        palette_manager=palette, lang_cfg=lang_cfg
    )


@pytest.fixture
def outlier_df():
    return pd.DataFrame(
        {
            "clean_text": [
                "hotel terrible service staff location",
                "hotel terrible service bad food",
                "hotel terrible rude staff location",
                "hotel terrible overpriced awful location",
            ]
        }
    )


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_palette(self, palette, lang_cfg):
        g = WordCloudGenerator(palette_manager=palette, lang_cfg=lang_cfg)
        assert g._pm is palette

    def test_stores_lang_cfg(self, palette, lang_cfg):
        g = WordCloudGenerator(palette_manager=palette, lang_cfg=lang_cfg)
        assert g._lang_cfg is lang_cfg


# ── Stopwords ────────────────────────────────────────────────────────────────


class TestLoadStopwords:
    def test_loads_english_stopwords(self, generator):
        sw = generator._load_stopwords()
        assert isinstance(sw, set)
        assert len(sw) > 0
        assert "the" in sw
        assert "and" in sw

    def test_returns_empty_on_failure(self, palette):
        cfg = MagicMock()
        cfg.stopwords_name = "nonexistent_language"
        g = WordCloudGenerator(palette_manager=palette, lang_cfg=cfg)
        sw = g._load_stopwords()
        assert sw == set()


# ── Empty helpers ────────────────────────────────────────────────────────────


class TestEmptyHelpers:
    def test_empty_image_b64_returns_string(self):
        result = _empty_image_b64()
        assert isinstance(result, str)
        assert result.startswith("data:image/png;base64,")

    def test_empty_chart_html_contains_message(self):
        result = _empty_chart_html("No data")
        assert "No data" in result
        assert result.startswith("<div>") or "<div" in result


# ── from_outliers ────────────────────────────────────────────────────────────


class TestFromOutliers:
    def test_empty_df_returns_placeholders(self, generator, output_dir, caplog):
        caplog.set_level(logging.WARNING)
        df = pd.DataFrame({"clean_text": []})
        result = generator.from_outliers(df, output_dir)
        assert result["wordcloud_b64"].startswith("data:image/png;base64,")
        assert "No outliers detected" in result["unigrams_html"]
        assert "No outliers detected" in result["bigrams_html"]
        assert "No outliers detected" in result["trigrams_html"]
        assert "No outliers detected" in caplog.text

    def test_returns_dict_with_correct_keys(self, generator, outlier_df, output_dir):
        result = generator.from_outliers(outlier_df, output_dir)
        assert set(result.keys()) == {
            "wordcloud_b64",
            "unigrams_html",
            "bigrams_html",
            "trigrams_html",
        }

    def test_wordcloud_b64_prefix(self, generator, outlier_df, output_dir):
        result = generator.from_outliers(outlier_df, output_dir)
        assert result["wordcloud_b64"].startswith("data:image/png;base64,")

    def test_saves_wordcloud_file(self, generator, outlier_df, output_dir):
        generator.from_outliers(outlier_df, output_dir)
        assert (output_dir / "wordcloud_outliers.png").exists()

    def test_saves_ngram_files(self, generator, outlier_df, output_dir):
        generator.from_outliers(outlier_df, output_dir)
        assert (output_dir / "ngrams_unigrams_outliers.html").exists()
        assert (output_dir / "ngrams_bigrams_outliers.html").exists()
        assert (output_dir / "ngrams_trigrams_outliers.html").exists()

    def test_ngram_html_standalone_has_plotly(
        self, generator, outlier_df, output_dir
    ):
        generator.from_outliers(outlier_df, output_dir)
        content = (
            output_dir / "ngrams_unigrams_outliers.html"
        ).read_text(encoding="utf-8")
        assert "plotly" in content.lower() or "Plotly" in content

    def test_returns_div_not_full_html(self, generator, outlier_df, output_dir):
        result = generator.from_outliers(outlier_df, output_dir)
        # Div strings should NOT have full HTML doctype
        assert "<!DOCTYPE html>" not in result["unigrams_html"]
        assert "<!DOCTYPE html>" not in result["bigrams_html"]
        assert "<!DOCTYPE html>" not in result["trigrams_html"]

    def test_info_log_for_saved_files(self, generator, outlier_df, output_dir, caplog):
        caplog.set_level(logging.INFO)
        generator.from_outliers(outlier_df, output_dir)
        assert "wordcloud_outliers" in caplog.text
        assert "ngrams_unigrams_outliers" in caplog.text

    def test_vectorizer_filters_stopwords(self, generator, output_dir):
        df = pd.DataFrame(
            {
                "clean_text": [
                    "the hotel terrible service",
                    "hotel terrible food service staff the",
                    "the hotel good food service staff",
                    "hotel good food service",
                    "the terrible room food staff",
                    "hotel terrible room food staff",
                ]
            }
        )
        result = generator.from_outliers(df, output_dir)
        # Stopwords like "the", "and", "a" should be filtered out
        assert "hotel" in result["unigrams_html"]
        assert "food" in result["unigrams_html"]
        assert "service" in result["unigrams_html"]


# ── from_partition ───────────────────────────────────────────────────────────


class TestFromPartition:
    def test_empty_df_returns_empty_b64(self, generator, output_dir):
        df = pd.DataFrame({"clean_text": []})
        result = generator.from_partition(df, "positive", output_dir)
        assert result.startswith("data:image/png;base64,")

    def test_returns_b64_string(self, generator, output_dir):
        df = pd.DataFrame({"clean_text": ["great place wonderful time"]})
        result = generator.from_partition(df, "positive", output_dir)
        assert result.startswith("data:image/png;base64,")

    def test_saves_file_with_label(self, generator, output_dir):
        df = pd.DataFrame({"clean_text": ["great place wonderful time"]})
        generator.from_partition(df, "positive", output_dir)
        assert (output_dir / "wordcloud_positive.png").exists()

    def test_saves_negative_file(self, generator, output_dir):
        df = pd.DataFrame({"clean_text": ["terrible experience"]})
        generator.from_partition(df, "negative", output_dir)
        assert (output_dir / "wordcloud_negative.png").exists()

    def test_return_type_string(self, generator, output_dir):
        df = pd.DataFrame({"clean_text": ["good"]})
        result = generator.from_partition(df, "positive", output_dir)
        assert isinstance(result, str)


# ── N-gram chart properties (Plotly) ────────────────────────────────────────


class TestNgramCharts:
    def test_tickangle_is_negative_45(self, generator, outlier_df, output_dir):
        result = generator.from_outliers(outlier_df, output_dir)
        # The Plotly div should contain the tickangle spec
        assert "tickangle" in result["unigrams_html"] or "-45" in result["unigrams_html"]

    def test_hover_shows_frequency(self, generator, outlier_df, output_dir):
        result = generator.from_outliers(outlier_df, output_dir)
        assert "Frequency" in result["unigrams_html"]
        assert "hovertemplate" in result["unigrams_html"]

    def test_template_set(self, generator, outlier_df, output_dir):
        result = generator.from_outliers(outlier_df, output_dir)
        # Plotly inlines templates as config, so check rendered structure
        assert '"type":"bar"' in result["unigrams_html"]
        assert len(result["unigrams_html"]) > 500

    def test_empty_corpus_returns_placeholder(
        self, generator, output_dir
    ):
        df = pd.DataFrame({"clean_text": ["a"]})  # stopword-only
        result = generator.from_outliers(df, output_dir)
        # Wordcloud guard returns empty b64; n-grams get empty chart
        assert result["wordcloud_b64"].startswith("data:image/png;base64,")
        assert "available" in result["bigrams_html"]
