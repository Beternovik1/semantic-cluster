"""Unit tests for src/pipeline/sentiment_analyzer.py."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.sentiment_analyzer import SentimentAnalyzer
from src.utils.language_config import LanguageConfig
from src.utils.validators import MIN_PARTITION_SIZE


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vader_cfg():
    return LanguageConfig("en")


@pytest.fixture
def pysentimiento_cfg():
    return LanguageConfig("es")


@pytest.fixture
def df_vader():
    return pd.DataFrame(
        {
            "clean_text": [
                "This hotel is absolutely amazing and wonderful",
                "Terrible experience, worst stay ever, very bad",
                "Average room, nothing special but not terrible",
                "Excellent service and friendly staff, highly recommend",
                "Dirty room and rude staff, would not come back",
            ]
        }
    )


@pytest.fixture
def df_pysentimiento():
    return pd.DataFrame(
        {
            "clean_text": [
                "El hotel es maravilloso y el personal excelente",
                " Pesima experiencia, horrible, no vuelvo nunca ",
                "La habitacion estaba bien, nada especial",
                "Me encanto la ubicacion y la comida deliciosa",
                "El servicio pesimo y la habitacion sucia",
            ]
        }
    )


def _mock_pysentimiento_result(output: str, pos: float, neg: float, neu: float):
    """Create a mock pysentimiento result with the given output and probas."""
    result = MagicMock()
    result.output = output
    result.probas = {"POS": pos, "NEG": neg, "NEU": neu}
    return result


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_lang_cfg(self, vader_cfg):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        assert a._lang_cfg is vader_cfg

    def test_stores_pysentimiento(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        assert a._pysentimiento is mock_analyzer

    def test_creates_vader_instance(self, vader_cfg):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        assert a._vader is not None


# ── VADER path ───────────────────────────────────────────────────────────────


class TestVaderPath:
    def test_adds_sentiment_column(self, vader_cfg, df_vader):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df_vader)
        assert "sentiment" in result.columns

    def test_adds_polarity_column(self, vader_cfg, df_vader):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df_vader)
        assert "polarity" in result.columns

    def test_adds_confidence_column(self, vader_cfg, df_vader):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df_vader)
        assert "sentiment_confidence" in result.columns

    def test_sentiment_is_binary(self, vader_cfg, df_vader):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df_vader)
        assert result["sentiment"].isin({"pos", "neg"}).all()

    def test_polarity_is_float(self, vader_cfg, df_vader):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df_vader)
        assert result["polarity"].dtype == float

    def test_confidence_is_bool(self, vader_cfg, df_vader):
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df_vader)
        assert result["sentiment_confidence"].dtype == bool

    def test_positive_text_gets_pos(self, vader_cfg):
        df = pd.DataFrame(
            {"clean_text": ["This is absolutely wonderful and perfect"]}
        )
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df)
        assert result["sentiment"].iloc[0] == "pos"

    def test_negative_text_gets_neg(self, vader_cfg):
        df = pd.DataFrame(
            {"clean_text": ["This is terrible awful horrible disgusting"]}
        )
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df)
        assert result["sentiment"].iloc[0] == "neg"

    def test_neutral_text_defaults_to_neg(self, vader_cfg):
        df = pd.DataFrame({"clean_text": ["the car is blue"]})
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df)
        assert result["sentiment"].iloc[0] == "neg"

    def test_empty_text_handled(self, vader_cfg):
        df = pd.DataFrame({"clean_text": ["", None, "good"]})
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df)
        assert result["sentiment"].notna().all()
        assert result["sentiment_confidence"].notna().all()

    def test_does_not_mutate_input(self, vader_cfg, df_vader):
        original_cols = set(df_vader.columns)
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        a.run(df_vader)
        assert set(df_vader.columns) == original_cols

    def test_vader_wins_on_disagreement(self, vader_cfg):
        # Find text where VADER says pos but TextBlob says neg,
        # or vice versa.  VADER should win.
        df = pd.DataFrame(
            {"clean_text": ["This is not bad at all, actually quite good"]}
        )
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        result = a.run(df)
        vs = a._vader.polarity_scores(df["clean_text"].iloc[0])
        expected = "pos" if vs["compound"] >= 0.05 else "neg"
        assert result["sentiment"].iloc[0] == expected


# ── pysentimiento path ───────────────────────────────────────────────────────


class TestPysentimientoPath:
    def test_adds_all_columns(self, pysentimiento_cfg, df_pysentimiento):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("POS", 0.8, 0.1, 0.1),
            _mock_pysentimiento_result("NEG", 0.1, 0.85, 0.05),
            _mock_pysentimiento_result("NEU", 0.4, 0.35, 0.25),
            _mock_pysentimiento_result("POS", 0.9, 0.05, 0.05),
            _mock_pysentimiento_result("NEG", 0.05, 0.9, 0.05),
        ]
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df_pysentimiento)
        assert "sentiment" in result.columns
        assert "polarity" in result.columns
        assert "sentiment_confidence" in result.columns

    def test_maps_pos_correctly(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("POS", 0.9, 0.05, 0.05),
        ]
        df = pd.DataFrame({"clean_text": ["texto positivo"]})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert result["sentiment"].iloc[0] == "pos"

    def test_maps_neg_correctly(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("NEG", 0.05, 0.9, 0.05),
        ]
        df = pd.DataFrame({"clean_text": ["texto negativo"]})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert result["sentiment"].iloc[0] == "neg"

    def test_resolves_neu_by_probas_pos(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("NEU", 0.6, 0.2, 0.2),
        ]
        df = pd.DataFrame({"clean_text": ["texto neutral sesgado a pos"]})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert result["sentiment"].iloc[0] == "pos"

    def test_resolves_neu_by_probas_neg(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("NEU", 0.2, 0.6, 0.2),
        ]
        df = pd.DataFrame({"clean_text": ["texto neutral sesgado a neg"]})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert result["sentiment"].iloc[0] == "neg"

    def test_confidence_true_when_not_neu(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("POS", 0.9, 0.05, 0.05),
            _mock_pysentimiento_result("NEG", 0.05, 0.9, 0.05),
        ]
        df = pd.DataFrame(
            {"clean_text": ["pos", "neg"]}
        )
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert result["sentiment_confidence"].all()

    def test_confidence_false_when_neu(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("NEU", 0.4, 0.35, 0.25),
        ]
        df = pd.DataFrame({"clean_text": ["texto neutral"]})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert not result["sentiment_confidence"].iloc[0]

    def test_polarity_is_pos_minus_neg(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("POS", 0.75, 0.15, 0.10),
        ]
        df = pd.DataFrame({"clean_text": ["text"]})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert result["polarity"].iloc[0] == 0.6  # 0.75 - 0.15

    def test_batch_processing(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("POS", 0.8, 0.1, 0.1),
            _mock_pysentimiento_result("NEG", 0.1, 0.8, 0.1),
        ]
        df = pd.DataFrame(
            {"clean_text": ["text1", "text2"]}
        )
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        a.run(df)
        # predict should have been called with a list of texts
        args, _ = mock_analyzer.predict.call_args
        assert len(args[0]) == 2

    def test_sentiment_is_binary(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("POS", 0.8, 0.1, 0.1),
            _mock_pysentimiento_result("NEG", 0.1, 0.8, 0.1),
            _mock_pysentimiento_result("NEU", 0.4, 0.35, 0.25),
        ]
        df = pd.DataFrame(
            {"clean_text": ["pos", "neg", "neu"]}
        )
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert result["sentiment"].isin({"pos", "neg"}).all()

    def test_empty_text_batched(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = []
        df = pd.DataFrame({"clean_text": []})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        result = a.run(df)
        assert len(result) == 0
        assert "sentiment" in result.columns

    def test_tqdm_progress_bar(self, pysentimiento_cfg):
        mock_analyzer = MagicMock()
        mock_analyzer.predict.return_value = [
            _mock_pysentimiento_result("POS", 0.8, 0.1, 0.1),
        ]
        df = pd.DataFrame({"clean_text": ["text"]})
        a = SentimentAnalyzer(
            lang_cfg=pysentimiento_cfg,
            pysentimiento_analyzer=mock_analyzer,
        )
        with patch(
            "src.pipeline.sentiment_analyzer.tqdm"
        ) as mock_tqdm:
            mock_tqdm.return_value.__iter__.return_value = [
                ["text"]
            ]
            a.run(df)
            mock_tqdm.assert_called_once()


# ── Logging ──────────────────────────────────────────────────────────────────


class TestLogging:
    def test_info_logs_counts(self, vader_cfg, df_vader, caplog):
        caplog.set_level(logging.INFO)
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        a.run(df_vader)
        assert "Sentiment:" in caplog.text
        assert "pos" in caplog.text
        assert "neg" in caplog.text

    def test_warning_if_below_min_size(self, vader_cfg, caplog):
        df = pd.DataFrame({"clean_text": ["good"] * 3})
        caplog.set_level(logging.WARNING)
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        a.run(df)
        assert "below MIN_PARTITION_SIZE" in caplog.text

    def test_no_warning_if_above_min_size(self, vader_cfg, caplog):
        texts = (
            ["positive and wonderful"] * (MIN_PARTITION_SIZE + 10)
            + ["terrible and awful"] * (MIN_PARTITION_SIZE + 10)
        )
        df = pd.DataFrame({"clean_text": texts})
        caplog.set_level(logging.WARNING)
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        a.run(df)
        assert "below MIN_PARTITION_SIZE" not in caplog.text

    def test_debug_logs_scores(self, vader_cfg, df_vader, caplog):
        caplog.set_level(logging.DEBUG)
        a = SentimentAnalyzer(
            lang_cfg=vader_cfg, pysentimiento_analyzer=None
        )
        a.run(df_vader)
        assert "VADER=" in caplog.text
