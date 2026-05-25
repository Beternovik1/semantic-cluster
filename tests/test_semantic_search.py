"""Unit tests for src/pipeline/semantic_search.py."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.semantic_search import SemanticSearch
from src.utils.validators import RANDOM_SEED


# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeEncoder:
    """Stand-in for SentenceTransformer.

    ``encode`` returns deterministic embeddings seeded by input text
    content so the same string always produces the same vector.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        # Deterministic seed from input content
        seed = hash("|".join(texts)) % (2**31)
        rng = np.random.default_rng(seed)
        return rng.uniform(-1, 1, (n, self.dim))


@pytest.fixture
def fake_encoder():
    return _FakeEncoder()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "clean_text": [
                "great hotel and excellent service",
                "very bad experience terrible stay",
                "average room nothing special",
                "wonderful location friendly staff",
                "dirty room and rude staff",
                "amazing food and nice view",
            ]
        }
    )


@pytest.fixture
def sample_embeddings(fake_encoder):
    texts = ["dummy"] * 6
    return np.array(fake_encoder.encode(texts))


@pytest.fixture
def searcher(fake_encoder):
    return SemanticSearch(
        embedding_model=fake_encoder,
        concept="excellent great wonderful",
        top_n=3,
    )


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_model(self, fake_encoder):
        s = SemanticSearch(embedding_model=fake_encoder, concept="test")
        assert s._model is fake_encoder

    def test_default_top_n(self, fake_encoder):
        s = SemanticSearch(embedding_model=fake_encoder, concept="test")
        assert s._top_n == 5

    def test_custom_top_n(self, fake_encoder):
        s = SemanticSearch(
            embedding_model=fake_encoder, concept="test", top_n=3
        )
        assert s._top_n == 3

    def test_stores_concept(self, fake_encoder):
        s = SemanticSearch(
            embedding_model=fake_encoder, concept="mi concepto"
        )
        assert s._concept == "mi concepto"


# ── Run() — core functionality ───────────────────────────────────────────────


class TestRun:
    def test_returns_tuple_of_two(self, searcher, sample_df, sample_embeddings):
        result = searcher.run(sample_df, sample_embeddings)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_two_dataframes(self, searcher, sample_df, sample_embeddings):
        df_out, df_top = searcher.run(sample_df, sample_embeddings)
        assert isinstance(df_out, pd.DataFrame)
        assert isinstance(df_top, pd.DataFrame)

    def test_adds_concept_similarity_column(
        self, searcher, sample_df, sample_embeddings
    ):
        df_out, _ = searcher.run(sample_df, sample_embeddings)
        assert "concept_similarity" in df_out.columns

    def test_similarity_is_float(self, searcher, sample_df, sample_embeddings):
        df_out, _ = searcher.run(sample_df, sample_embeddings)
        assert df_out["concept_similarity"].dtype == np.float64

    def test_all_rows_have_similarity(
        self, searcher, sample_df, sample_embeddings
    ):
        df_out, _ = searcher.run(sample_df, sample_embeddings)
        assert len(df_out) == len(sample_df)
        assert df_out["concept_similarity"].notna().all()

    def test_top_n_returns_correct_number(
        self, searcher, sample_df, sample_embeddings
    ):
        _, df_top = searcher.run(sample_df, sample_embeddings)
        assert len(df_top) == 3  # top_n=3

    def test_top_n_with_default(self, fake_encoder, sample_df, sample_embeddings):
        s = SemanticSearch(
            embedding_model=fake_encoder, concept="test"
        )
        _, df_top = s.run(sample_df, sample_embeddings)
        assert len(df_top) == 5  # default top_n

    def test_top_n_sorted_descending(
        self, searcher, sample_df, sample_embeddings
    ):
        _, df_top = searcher.run(sample_df, sample_embeddings)
        scores = df_top["concept_similarity"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_original_columns_preserved(
        self, searcher, sample_df, sample_embeddings
    ):
        df_out, _ = searcher.run(sample_df, sample_embeddings)
        for col in sample_df.columns:
            assert col in df_out.columns

    def test_clean_text_in_top_n(
        self, searcher, sample_df, sample_embeddings
    ):
        _, df_top = searcher.run(sample_df, sample_embeddings)
        assert "clean_text" in df_top.columns
        assert df_top["clean_text"].notna().all()

    def test_does_not_mutate_input_df(
        self, searcher, sample_df, sample_embeddings
    ):
        original_cols = set(sample_df.columns)
        searcher.run(sample_df, sample_embeddings)
        assert set(sample_df.columns) == original_cols


# ── Similarity values (deterministic with seed) ──────────────────────────────


class TestSimilarityValues:
    def test_similarity_in_range(self, searcher, sample_df, sample_embeddings):
        df_out, _ = searcher.run(sample_df, sample_embeddings)
        sim = df_out["concept_similarity"]
        assert sim.between(-1.0, 1.0).all()

    def test_similarity_matches_sklearn_direct(
        self, fake_encoder, sample_df, sample_embeddings
    ):
        concept_vec = np.array(fake_encoder.encode(["some concept"]))
        expected = cosine_similarity(sample_embeddings, concept_vec).flatten()

        s = SemanticSearch(
            embedding_model=fake_encoder, concept="some concept"
        )
        df_out, _ = s.run(sample_df, sample_embeddings)

        assert np.allclose(df_out["concept_similarity"].values, expected)


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_row(self, fake_encoder):
        df = pd.DataFrame({"clean_text": ["only one review"]})
        emb = np.array(fake_encoder.encode(["only one review"]))
        s = SemanticSearch(
            embedding_model=fake_encoder, concept="test", top_n=1
        )
        df_out, df_top = s.run(df, emb)
        assert len(df_out) == 1
        assert len(df_top) == 1
        assert "concept_similarity" in df_out.columns

    def test_top_n_larger_than_dataset(self, fake_encoder):
        df = pd.DataFrame({"clean_text": ["a", "b"]})
        emb = np.array(fake_encoder.encode(["a", "b"]))
        s = SemanticSearch(
            embedding_model=fake_encoder, concept="test", top_n=10
        )
        _, df_top = s.run(df, emb)
        assert len(df_top) == 2  # clamped to available rows

    def test_empty_dataframe(self, fake_encoder):
        df = pd.DataFrame({"clean_text": []})
        emb = np.empty((0, 384))
        s = SemanticSearch(
            embedding_model=fake_encoder, concept="test"
        )
        df_out, df_top = s.run(df, emb)
        assert len(df_out) == 0
        assert len(df_top) == 0


# ── Logging ──────────────────────────────────────────────────────────────────


class TestLogging:
    def test_info_logs_top_similarity(
        self, searcher, sample_df, sample_embeddings, caplog
    ):
        caplog.set_level(logging.INFO)
        searcher.run(sample_df, sample_embeddings)
        assert "Top 1:" in caplog.text
        assert "Top 2:" in caplog.text
        assert "Top 3:" in caplog.text
        assert "similarity=" in caplog.text

    def test_debug_logs_distribution(
        self, searcher, sample_df, sample_embeddings, caplog
    ):
        caplog.set_level(logging.DEBUG)
        searcher.run(sample_df, sample_embeddings)
        assert "Similarity distribution" in caplog.text
        assert "min=" in caplog.text
        assert "max=" in caplog.text
        assert "mean=" in caplog.text
        assert "std=" in caplog.text
