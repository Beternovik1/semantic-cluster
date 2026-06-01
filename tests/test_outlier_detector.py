"""Unit tests for src/pipeline/outlier_detector.py."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from src.pipeline.outlier_detector import DEFAULT_CACHE_DIR, OutlierDetector
from src.utils.validators import RANDOM_SEED


# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakeEncoder:
    """Stand-in for SentenceTransformer that returns dummy embeddings."""

    def __init__(self, dim: int = 384, seed: int = RANDOM_SEED) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(seed)

    def encode(self, texts, show_progress_bar: bool = True):
        n = len(texts)
        return self._rng.uniform(-1, 1, (n, self.dim))


@pytest.fixture
def fake_encoder():
    return _FakeEncoder()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "clean_text": [
                "this is a normal review about a hotel",
                "great location and friendly staff",
                "very bad experience would not recommend",
                "average room nothing special",
                "excellent service and clean rooms",
                "this place is terrible awful disgusting",
            ]
        }
    )


@pytest.fixture
def detector(fake_encoder, tmp_path):
    return OutlierDetector(
        embedding_model=fake_encoder,
        contamination=0.15,
        cache_dir=tmp_path / "cache",
    )


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_model(self, fake_encoder):
        d = OutlierDetector(embedding_model=fake_encoder)
        assert d._model is fake_encoder

    def test_default_contamination(self, fake_encoder):
        d = OutlierDetector(embedding_model=fake_encoder)
        assert d._contamination == 0.05

    def test_custom_contamination(self, fake_encoder):
        d = OutlierDetector(embedding_model=fake_encoder, contamination=0.1)
        assert d._contamination == 0.1

    def test_default_cache_dir(self, fake_encoder):
        d = OutlierDetector(embedding_model=fake_encoder)
        assert d._cache_dir == DEFAULT_CACHE_DIR

    def test_custom_cache_dir(self, fake_encoder):
        d = OutlierDetector(
            embedding_model=fake_encoder,
            cache_dir=Path("/tmp/other"),
        )
        assert d._cache_dir == Path("/tmp/other")


# ── SHA-256 hash ─────────────────────────────────────────────────────────────


class TestComputeHash:
    def test_returns_string(self, detector, sample_df):
        h = detector._compute_hash(sample_df)
        assert isinstance(h, str)
        assert len(h) == 64
        assert re.match(r"^[a-f0-9]{64}$", h)

    def test_deterministic(self, detector, sample_df):
        h1 = detector._compute_hash(sample_df)
        h2 = detector._compute_hash(sample_df)
        assert h1 == h2

    def test_changes_with_data(self, detector, sample_df):
        h1 = detector._compute_hash(sample_df)
        df2 = sample_df.copy()
        df2.loc[0, "clean_text"] = "different text"
        h2 = detector._compute_hash(df2)
        assert h1 != h2

    def test_matches_expected_sha256(self, detector):
        texts = ["hello", "world"]
        df = pd.DataFrame({"clean_text": texts})
        expected = hashlib.sha256("helloworld".encode("utf-8")).hexdigest()
        assert detector._compute_hash(df) == expected


# ── Cache logic ──────────────────────────────────────────────────────────────


class TestCache:
    def test_cache_hit_loads_embeddings(self, detector, sample_df, caplog):
        embeddings = np.array([[0.1] * 384, [0.2] * 384])
        h = detector._compute_hash(sample_df)
        detector._save_cache(embeddings, h)

        caplog.set_level(logging.INFO)
        result = detector._get_embeddings(sample_df)

        assert_array_equal(result, embeddings)
        assert "Loaded embeddings from cache" in caplog.text

    def test_cache_miss_computes_fresh(self, detector, sample_df, caplog):
        caplog.set_level(logging.INFO)
        result = detector._get_embeddings(sample_df)

        assert result.shape == (len(sample_df), 384)
        assert "Computing embeddings" in caplog.text

    def test_cache_invalidated_on_hash_mismatch(
        self, detector, sample_df, caplog
    ):
        df_old = pd.DataFrame({"clean_text": ["old data"]})
        old_hash = detector._compute_hash(df_old)
        detector._save_cache(np.array([[0.5] * 384]), old_hash)

        caplog.set_level(logging.WARNING)
        result = detector._get_embeddings(sample_df)

        assert result.shape == (len(sample_df), 384)
        assert "Cache invalidated" in caplog.text

    def test_missing_embeddings_file_computes_fresh(
        self, detector, sample_df
    ):
        cache_dir = detector._cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        h_path = cache_dir / "cache_hash.txt"
        h_path.write_text("somehash", encoding="utf-8")

        result = detector._get_embeddings(sample_df)
        assert result.shape == (len(sample_df), 384)

    def test_missing_hash_file_computes_fresh(
        self, detector, sample_df
    ):
        cache_dir = detector._cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path = cache_dir / "embeddings.npy"
        np.save(emb_path, np.array([[0.1] * 384]))

        result = detector._get_embeddings(sample_df)
        assert result.shape == (len(sample_df), 384)

    def test_save_cache_creates_directory(self, detector, sample_df):
        cache_dir = detector._cache_dir / "subdir"
        emb_path = cache_dir / "embeddings.npy"
        h_path = cache_dir / "cache_hash.txt"

        d = OutlierDetector(
            embedding_model=detector._model,
            contamination=detector._contamination,
            cache_dir=cache_dir,
        )

        embeddings = np.array([[0.1] * 384])
        h = d._compute_hash(sample_df)
        d._save_cache(embeddings, h)

        assert emb_path.exists()
        assert h_path.exists()
        assert_array_equal(np.load(emb_path), embeddings)
        assert h_path.read_text(encoding="utf-8") == h


# ── PCA + IsolationForest ────────────────────────────────────────────────────


class TestDetection:
    def test_returns_boolean_mask(self, detector):
        embeddings = np.random.default_rng(RANDOM_SEED).uniform(
            -1, 1, (20, 384)
        )
        mask = detector._detect(embeddings)
        assert mask.dtype == bool
        assert len(mask) == 20

    def test_mask_sum_is_within_contamination(self, detector):
        rng = np.random.default_rng(RANDOM_SEED)
        embeddings = rng.uniform(-1, 1, (200, 384))
        mask = detector._detect(embeddings)
        outlier_ratio = 1.0 - mask.mean()
        assert outlier_ratio <= detector._contamination + 0.1

    def test_pca_n_components_logged(self, detector, caplog):
        caplog.set_level(logging.INFO)
        rng = np.random.default_rng(RANDOM_SEED)
        embeddings = rng.uniform(-1, 1, (30, 384))
        detector._detect(embeddings)
        assert "PCA n_components=" in caplog.text

    def test_pca_n_components_small_dataset(self, detector, caplog):
        caplog.set_level(logging.INFO)
        rng = np.random.default_rng(RANDOM_SEED)
        embeddings = rng.uniform(-1, 1, (3, 384))
        detector._detect(embeddings)
        assert "PCA n_components=2" in caplog.text

    def test_pca_n_components_large_dataset(self, detector, caplog):
        caplog.set_level(logging.INFO)
        rng = np.random.default_rng(RANDOM_SEED)
        embeddings = rng.uniform(-1, 1, (100, 384))
        detector._detect(embeddings)
        assert "PCA n_components=50" in caplog.text

    def test_debug_logs_explained_variance(self, detector, caplog):
        caplog.set_level(logging.DEBUG)
        rng = np.random.default_rng(RANDOM_SEED)
        embeddings = rng.uniform(-1, 1, (20, 384))
        detector._detect(embeddings)
        assert "PCA explained variance ratio" in caplog.text

    def test_debug_logs_iso_params(self, detector, caplog):
        caplog.set_level(logging.DEBUG)
        rng = np.random.default_rng(RANDOM_SEED)
        embeddings = rng.uniform(-1, 1, (20, 384))
        detector._detect(embeddings)
        assert "IsolationForest parameters" in caplog.text

    def test_random_seed_used(self, detector):
        rng = np.random.default_rng(RANDOM_SEED)
        embeddings = rng.uniform(-1, 1, (20, 384))
        mask1 = detector._detect(embeddings.copy())
        mask2 = detector._detect(embeddings.copy())
        assert_array_equal(mask1, mask2)

    def test_single_row_returns_all_inliers(self, detector):
        embeddings = np.random.default_rng(RANDOM_SEED).uniform(
            -1, 1, (1, 384)
        )
        mask = detector._detect(embeddings)
        assert mask.shape == (1,)
        assert mask[0]

    def test_zero_rows_crash_protection(self, detector):
        embeddings = np.empty((0, 384))
        mask = detector._detect(embeddings)
        assert mask.shape == (0,)


# ── Full run() — returns and alignment ───────────────────────────────────────


class TestRun:
    def test_returns_three_elements(self, detector, sample_df):
        result = detector.run(sample_df)
        assert len(result) == 3

    def test_returns_dataframes_and_array(self, detector, sample_df):
        df_clean, df_outliers, embeddings_clean = detector.run(sample_df)
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(df_outliers, pd.DataFrame)
        assert isinstance(embeddings_clean, np.ndarray)

    def test_total_rows_preserved(self, detector, sample_df):
        df_clean, df_outliers, _ = detector.run(sample_df)
        assert len(df_clean) + len(df_outliers) == len(sample_df)

    def test_embeddings_align_with_clean(self, detector, sample_df):
        df_clean, _, embeddings_clean = detector.run(sample_df)
        assert len(embeddings_clean) == len(df_clean)

    def test_embeddings_dimension(self, detector, sample_df):
        _, _, embeddings_clean = detector.run(sample_df)
        assert embeddings_clean.shape[1] == 384

    def test_original_columns_preserved(self, detector, sample_df):
        df_clean, df_outliers, _ = detector.run(sample_df)
        assert "clean_text" in df_clean.columns
        assert "clean_text" in df_outliers.columns
        for col in sample_df.columns:
            assert col in df_clean.columns

    def test_no_outliers_possible(self, detector):
        df = pd.DataFrame({"clean_text": ["same text"] * 6})
        df_clean, df_outliers, _ = detector.run(df)
        assert len(df_outliers) < len(df)

    def test_outlier_count_logged(self, detector, sample_df, caplog):
        caplog.set_level(logging.INFO)
        detector.run(sample_df)
        assert "Outliers:" in caplog.text


# ── Embedding dimensions ─────────────────────────────────────────────────────


class TestEmbeddingDimensions:
    def test_cache_stores_full_384d(self, detector, sample_df):
        detector.run(sample_df)
        cached = np.load(detector._cache_dir / "embeddings.npy")
        assert cached.shape[1] == 384

    def test_embeddings_clean_is_384d(self, detector, sample_df):
        _, _, embeddings_clean = detector.run(sample_df)
        assert embeddings_clean.shape[1] == 384
