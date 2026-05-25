"""Unit tests for src/pipeline/topic_modeler.py."""

from __future__ import annotations

import logging
from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.topic_modeler import TopicModeler
from src.utils.language_config import LanguageConfig
from src.utils.validators import MIN_PARTITION_SIZE, RANDOM_SEED


# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture
def lang_cfg():
    return LanguageConfig("en")


@pytest.fixture
def fake_embedder():
    return MagicMock()


@pytest.fixture
def small_df():
    """Fallback-sized partition (below MIN_PARTITION_SIZE)."""
    return pd.DataFrame(
        {
            "clean_text": [f"review number {i}" for i in range(5)],
            "tokens": [
                ["great", "hotel", "food"],
                ["bad", "terrible", "awful"],
                ["good", "location", "staff"],
                ["wonderful", "friendly", "price"],
                ["dirty", "rude", "disappointed"],
            ],
        }
    )


@pytest.fixture
def large_df():
    """BERTopic-sized partition (>= MIN_PARTITION_SIZE rows)."""
    return pd.DataFrame(
        {
            "clean_text": [f"document about topic {i % 3}" for i in range(100)],
            "tokens": [
                ["word", str(i % 3)] for i in range(100)
            ],
        }
    )


@pytest.fixture
def large_embeddings():
    rng = np.random.default_rng(RANDOM_SEED)
    return rng.uniform(-1, 1, (100, 384))


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_model(self, fake_embedder, lang_cfg):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        assert m._model is fake_embedder

    def test_stores_lang_cfg(self, fake_embedder, lang_cfg):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        assert m._lang_cfg is lang_cfg


# ── Stopword loading ─────────────────────────────────────────────────────────


class TestLoadStopwords:
    def test_returns_set_for_english(self, fake_embedder, lang_cfg):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        sw = m._load_stopwords()
        assert isinstance(sw, set)
        assert len(sw) > 0
        assert "the" in sw

    def test_returns_empty_on_load_failure(self, fake_embedder):
        cfg = MagicMock()
        cfg.stopwords_name = "nonexistent_language"
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=cfg)
        sw = m._load_stopwords()
        assert sw == set()


# ── Fallback path ────────────────────────────────────────────────────────────


class TestFallback:
    def test_triggers_below_min_size(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, keywords, used_bertopic = m.run(small_df, np.empty((0, 384)))
        assert not used_bertopic

    def test_warning_logged(self, fake_embedder, lang_cfg, small_df, caplog):
        caplog.set_level(logging.WARNING)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        m.run(small_df, np.empty((0, 384)))
        assert "Falling back to word frequency" in caplog.text

    def test_topic_id_is_minus_one(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(small_df, np.empty((0, 384)))
        assert (df_out["topic_id"] == -1).all()

    def test_topic_label_is_fallback(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(small_df, np.empty((0, 384)))
        assert (df_out["topic_label"] == "word_frequency_fallback").all()

    def test_umap_columns_are_none(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(small_df, np.empty((0, 384)))
        assert df_out["umap_x"].isna().all()
        assert df_out["umap_y"].isna().all()

    def test_representative_doc_is_false(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(small_df, np.empty((0, 384)))
        assert not df_out["representative_doc"].any()

    def test_topic_keywords_populated(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, keywords, _ = m.run(small_df, np.empty((0, 384)))
        assert df_out["topic_keywords"].iloc[0] != ""
        assert -1 in keywords
        assert keywords[-1] != ""

    def test_stopwords_filtered(self, fake_embedder, lang_cfg):
        df = pd.DataFrame(
            {
                "clean_text": ["test"] * 3,
                "tokens": [
                    ["the", "hotel", "was", "great"],
                    ["a", "terrible", "experience", "the"],
                    ["service", "and", "food"],
                ],
            }
        )
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, keywords, _ = m.run(df, np.empty((0, 384)))
        kw_parts = set(keywords[-1].lower().split(", "))
        # Stopwords should NOT appear in keywords
        assert "the" not in kw_parts
        assert "and" not in kw_parts
        assert "a" not in kw_parts

    def test_empty_tokens_does_not_crash(self, fake_embedder, lang_cfg):
        df = pd.DataFrame(
            {
                "clean_text": [""],
                "tokens": [[]],
            }
        )
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, keywords, _ = m.run(df, np.empty((0, 384)))
        assert len(df_out) == 1

    def test_all_columns_present(self, fake_embedder, lang_cfg, small_df):
        required = {
            "topic_id", "topic_label", "umap_x", "umap_y",
            "topic_keywords", "representative_doc",
        }
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(small_df, np.empty((0, 384)))
        assert required.issubset(set(df_out.columns))

    def test_keywords_dict_key_is_int(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        _, keywords, _ = m.run(small_df, np.empty((0, 384)))
        for k in keywords:
            assert isinstance(k, int)

    def test_used_bertopic_flag_false(self, fake_embedder, lang_cfg, small_df):
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        _, _, used = m.run(small_df, np.empty((0, 384)))
        assert not used


# ── BERTopic path (mocked) ──────────────────────────────────────────────────


class TestBertopicPath:
    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_used_bertopic_flag_true(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        _setup_mock_bertopic(mock_bertopic, len(large_df))
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        _, _, used = m.run(large_df, large_embeddings)
        assert used

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_fit_transform_called(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        instance = _setup_mock_bertopic(mock_bertopic, len(large_df))
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        m.run(large_df, large_embeddings)
        instance.fit_transform.assert_called_once()

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_topic_id_column(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        _setup_mock_bertopic(mock_bertopic, len(large_df))
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(large_df, large_embeddings)
        assert "topic_id" in df_out.columns
        assert df_out["topic_id"].dtype == int

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_umap_columns_float(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        _setup_mock_bertopic(mock_bertopic, len(large_df))
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(large_df, large_embeddings)
        assert df_out["umap_x"].dtype == np.float64
        assert df_out["umap_y"].dtype == np.float64

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_representative_doc_one_per_topic(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        _setup_mock_bertopic(mock_bertopic, n)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(large_df, large_embeddings)

        # With mocked data having n_topics=3 and n_noise=10
        n_topics = 3
        n_rep = df_out["representative_doc"].sum()
        assert n_rep == n_topics

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_topic_keywords_dict(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        instance = _setup_mock_bertopic(mock_bertopic, n)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        _, keywords, _ = m.run(large_df, large_embeddings)

        # Should have 3 topics (0, 1, 2) but not -1
        assert 0 in keywords
        assert 1 in keywords
        assert 2 in keywords
        assert -1 not in keywords
        for kw in keywords.values():
            assert isinstance(kw, str)
            assert kw != ""

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_precomputed_embeddings_passed_to_fit_transform(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        instance = _setup_mock_bertopic(mock_bertopic, len(large_df))
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        m.run(large_df, large_embeddings)

        call_kwargs = instance.fit_transform.call_args[1]
        assert "embeddings" in call_kwargs
        np.testing.assert_array_equal(
            call_kwargs["embeddings"], large_embeddings
        )

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_all_required_columns(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        required = {
            "topic_id", "topic_label", "umap_x", "umap_y",
            "topic_keywords", "representative_doc",
        }
        n = len(large_df)
        _setup_mock_bertopic(mock_bertopic, n)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(large_df, large_embeddings)
        assert required.issubset(set(df_out.columns))

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_original_columns_preserved(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        _setup_mock_bertopic(mock_bertopic, n)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(large_df, large_embeddings)
        for col in large_df.columns:
            assert col in df_out.columns

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_debug_logging(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings, caplog,
    ):
        n = len(large_df)
        _setup_mock_bertopic(mock_bertopic, n)
        caplog.set_level(logging.INFO)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        m.run(large_df, large_embeddings)
        assert "BERTopic found" in caplog.text
        assert "min_cluster_size" in caplog.text


# ── BERTopic mock helpers ───────────────────────────────────────────────────


def _setup_mock_bertopic(mock_bertopic_cls, n_docs: int):
    """Configure a mock BERTopic instance with deterministic behaviour."""
    instance = MagicMock()

    # topics: assign 30 docs to topic 0, 30 to 1, 30 to 2, 10 to noise
    topics = np.array([0] * 30 + [1] * 30 + [2] * 30 + [-1] * (n_docs - 90))
    np.random.shuffle(topics)
    instance.fit_transform.return_value = (topics, None)

    # topic info
    def get_topic(tid):
        if tid == -1:
            return [("noise", 1.0)]
        return [(f"word_{tid}_a", 0.5), (f"word_{tid}_b", 0.3)]

    instance.get_topic.side_effect = get_topic

    # UMAP model
    rng = np.random.default_rng(RANDOM_SEED)
    umap_mock = MagicMock()
    umap_mock.transform.return_value = rng.uniform(-1, 1, (n_docs, 2))
    instance.umap_model = umap_mock

    # Silence maxsilence/cutoff warnings by setting specific attrs
    instance.topics_ = topics

    mock_bertopic_cls.return_value = instance

    # Also suppress UMAP/HDBSCAN from being instantiated by the import
    # (they will be patched so their constructors return mocks too)
    return instance


# ── Edge: all documents marked as noise by HDBSCAN ─────────────────────────


class TestAllNoiseEdgeCase:
    """BERTopic path where HDBSCAN labels every document as noise (topic -1)."""

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_does_not_crash(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        instance = MagicMock()
        topics = np.full(n, -1)  # all noise
        instance.fit_transform.return_value = (topics, None)
        instance.get_topic.return_value = []
        rng = np.random.default_rng(RANDOM_SEED)
        umap_mock = MagicMock()
        umap_mock.transform.return_value = rng.uniform(-1, 1, (n, 2))
        instance.umap_model = umap_mock
        instance.topics_ = topics
        mock_bertopic.return_value = instance

        df_out, keywords, used = TopicModeler(
            embedding_model=fake_embedder, lang_cfg=lang_cfg
        ).run(large_df, large_embeddings)

        assert used is True
        assert len(df_out) == n
        assert df_out["topic_id"].unique().tolist() == [-1]
        assert df_out["umap_x"].notna().all()
        assert df_out["umap_y"].notna().all()
        assert (df_out["topic_label"] == "").all()
        assert not df_out["representative_doc"].any()
        assert keywords == {}

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_reduce_topics_not_called(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        instance = MagicMock()
        topics = np.full(n, -1)
        instance.fit_transform.return_value = (topics, None)
        rng = np.random.default_rng(RANDOM_SEED)
        umap_mock = MagicMock()
        umap_mock.transform.return_value = rng.uniform(-1, 1, (n, 2))
        instance.umap_model = umap_mock
        instance.topics_ = topics
        mock_bertopic.return_value = instance

        TopicModeler(
            embedding_model=fake_embedder, lang_cfg=lang_cfg
        ).run(large_df, large_embeddings)

        instance.reduce_topics.assert_not_called()

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_all_required_columns_present(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        instance = MagicMock()
        topics = np.full(n, -1)
        instance.fit_transform.return_value = (topics, None)
        rng = np.random.default_rng(RANDOM_SEED)
        umap_mock = MagicMock()
        umap_mock.transform.return_value = rng.uniform(-1, 1, (n, 2))
        instance.umap_model = umap_mock
        instance.topics_ = topics
        mock_bertopic.return_value = instance

        df_out, _, _ = TopicModeler(
            embedding_model=fake_embedder, lang_cfg=lang_cfg
        ).run(large_df, large_embeddings)

        required = {
            "topic_id", "topic_label", "umap_x", "umap_y",
            "topic_keywords", "representative_doc",
        }
        assert required.issubset(set(df_out.columns))


# ── Edge: empty partition ───────────────────────────────────────────────────


class TestEmptyPartition:
    def test_empty_dataframe_returns_fallback(
        self, fake_embedder, lang_cfg
    ):
        df = pd.DataFrame({"clean_text": [], "tokens": []})
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, keywords, used = m.run(df, np.empty((0, 384)))
        assert not used
        assert len(df_out) == 0
        assert -1 in keywords

    def test_fallback_columns_on_empty(
        self, fake_embedder, lang_cfg
    ):
        df = pd.DataFrame({"clean_text": [], "tokens": []})
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        df_out, _, _ = m.run(df, np.empty((0, 384)))
        required = {
            "topic_id", "topic_label", "umap_x", "umap_y",
            "topic_keywords", "representative_doc",
        }
        assert required.issubset(set(df_out.columns))


# ── Keyword dict from BERTopic path ─────────────────────────────────────────


class TestKeywordsDict:
    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_does_not_include_noise_topic(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        _setup_mock_bertopic(mock_bertopic, n)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        _, keywords, _ = m.run(large_df, large_embeddings)
        assert -1 not in keywords

    @patch("src.pipeline.topic_modeler.UMAP")
    @patch("src.pipeline.topic_modeler.HDBSCAN")
    @patch("src.pipeline.topic_modeler.BERTopic")
    def test_keywords_format(
        self, mock_bertopic, mock_hdbscan, mock_umap,
        fake_embedder, lang_cfg, large_df, large_embeddings,
    ):
        n = len(large_df)
        _setup_mock_bertopic(mock_bertopic, n)
        m = TopicModeler(embedding_model=fake_embedder, lang_cfg=lang_cfg)
        _, keywords, _ = m.run(large_df, large_embeddings)
        for kw in keywords.values():
            parts = kw.split(", ")
            assert len(parts) >= 2
