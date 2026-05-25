"""Outlier detection via PCA + IsolationForest on sentence embeddings.

Computes or loads cached embeddings, applies dynamic PCA
dimensionality reduction, and flags semantic outliers.
Maintains strict row alignment between the returned DataFrame
and embeddings_clean numpy array.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.utils.validators import RANDOM_SEED

logger = logging.getLogger(__name__)

CACHE_DIR = Path("outputs")
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
HASH_PATH = CACHE_DIR / "cache_hash.txt"


class OutlierDetector:
    """Detect semantic outliers using PCA-reduced sentence embeddings.

    Args:
        embedding_model: A ``SentenceTransformer`` instance (injected).
        contamination: Expected proportion of outliers in the dataset.
            Must be between 0.01 and 0.15.
        cache_dir: Directory for ``embeddings.npy`` and
            ``cache_hash.txt``. Defaults to ``outputs/``.
    """

    def __init__(
        self,
        embedding_model,
        contamination: float = 0.05,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        self._model = embedding_model
        self._contamination = contamination
        self._cache_dir = Path(cache_dir)

    def run(self, df):
        """Detect outliers and return aligned clean/outlier splits.

        Args:
            df: DataFrame with a ``clean_text`` column.

        Returns:
            Tuple of ``(df_clean, df_outliers, embeddings_clean)``
            where ``embeddings_clean.shape[0] == len(df_clean)``.
        """
        embeddings = self._get_embeddings(df)
        mask = self._detect(embeddings)

        df_clean = df[mask].reset_index(drop=True)
        df_outliers = df[~mask].reset_index(drop=True)

        # Critical: apply the same boolean mask to the embeddings array
        embeddings_clean = embeddings[mask]

        logger.info(
            "Outliers: %d / %d (%.1f%%). Clean: %d.",
            len(df_outliers),
            len(df),
            (len(df_outliers) / len(df) * 100) if len(df) else 0.0,
            len(df_clean),
        )

        return df_clean, df_outliers, embeddings_clean

    # ── Embedding computation / cache ───────────────────────────────

    def _compute_hash(self, df) -> str:
        """SHA-256 of concatenated ``clean_text`` values."""
        combined = "".join(df["clean_text"].tolist()).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def _get_embeddings(self, df) -> np.ndarray:
        """Return full 384-dim embeddings (load cached or compute)."""
        current_hash = self._compute_hash(df)
        embeddings = self._try_load_cache(current_hash)

        if embeddings is not None:
            logger.info("Loaded embeddings from cache (hash match).")
            return embeddings

        logger.info("Computing embeddings for %d documents ...", len(df))
        embeddings = self._model.encode(
            df["clean_text"].tolist(),
            show_progress_bar=True,
        )
        embeddings = np.array(embeddings)

        self._save_cache(embeddings, current_hash)
        return embeddings

    def _try_load_cache(self, current_hash: str) -> np.ndarray | None:
        """Load cached embeddings if ``cache_hash.txt`` matches."""
        emb_path = self._cache_dir / "embeddings.npy"
        hash_path = self._cache_dir / "cache_hash.txt"

        if not emb_path.exists() or not hash_path.exists():
            return None

        saved_hash = hash_path.read_text(encoding="utf-8").strip()

        if saved_hash != current_hash:
            logger.warning(
                "Cache invalidated: text content changed. Recomputing."
            )
            return None

        return np.load(emb_path)

    def _save_cache(self, embeddings: np.ndarray, hash_val: str) -> None:
        """Persist embeddings and their hash for future runs."""
        emb_path = self._cache_dir / "embeddings.npy"
        hash_path = self._cache_dir / "cache_hash.txt"

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        hash_path.write_text(hash_val, encoding="utf-8")
        logger.info("Embeddings cached to %s.", emb_path)

    # ── PCA + IsolationForest ───────────────────────────────────────

    def _detect(self, embeddings: np.ndarray) -> np.ndarray:
        """Return boolean mask: ``True`` = clean (inlier)."""
        n = len(embeddings)

        if n <= 1:
            logger.warning(
                "Too few samples (%d) for PCA + IsolationForest. "
                "Marking all as inliers.",
                n,
            )
            return np.ones(n, dtype=bool)

        n_components = min(50, n - 1)
        logger.info(
            "PCA n_components=%d (embeddings shape: %s).",
            n_components,
            embeddings.shape,
        )

        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        reduced = pca.fit_transform(embeddings)

        logger.debug(
            "PCA explained variance ratio: %.4f (sum of %d components).",
            pca.explained_variance_ratio_.sum(),
            n_components,
        )

        iso = IsolationForest(
            contamination=self._contamination,
            random_state=RANDOM_SEED,
        )
        iso.fit(reduced)

        logger.debug(
            "IsolationForest parameters: contamination=%.2f, "
            "random_state=%d.",
            self._contamination,
            RANDOM_SEED,
        )

        # IsolationForest returns 1 for inliers, -1 for outliers
        predictions = iso.predict(reduced)
        return predictions == 1
