"""Semantic similarity search against a synthetic concept string.

Computes cosine similarity between pre-computed sentence embeddings
and a single concept embedding.  Never re-encodes the full dataset.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticSearch:
    """Score each comment by cosine similarity to a concept string.

    Args:
        embedding_model: A ``SentenceTransformer`` instance (injected).
        concept: The synthetic concept string to search for.
        top_n: Number of most similar comments to extract.
    """

    def __init__(
        self,
        embedding_model,
        concept: str,
        top_n: int = 5,
    ) -> None:
        self._model = embedding_model
        self._concept = concept
        self._top_n = top_n

    def run(self, df: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Add ``concept_similarity`` scores and return top-N matches.

        Args:
            df: DataFrame whose rows correspond to ``embeddings``.
            embeddings: Pre-computed 384-dim embeddings aligned to
                ``df`` rows by ``outlier_detector``.

        Returns:
            Tuple of ``(df, df_top_n)`` where *df* has been augmented
            with a ``concept_similarity`` column and *df_top_n*
            contains the ``top_n`` most similar rows.
        """
        if len(embeddings) == 0:
            df = df.copy()
            df["concept_similarity"] = np.array([], dtype=np.float64)
            return df, df.head(0)

        concept_embedding = self._model.encode([self._concept])
        similarities = cosine_similarity(embeddings, concept_embedding)

        # sklearn returns (N, 1) — flatten to (N,) for column assignment
        df = df.copy()
        df["concept_similarity"] = similarities.flatten()

        df_top_n = df.nlargest(self._top_n, "concept_similarity")

        self._log_results(df, df_top_n)
        return df, df_top_n

    # ── Logging helpers ─────────────────────────────────────────────

    def _log_results(self, df: pd.DataFrame, df_top_n: pd.DataFrame) -> None:
        """Emit INFO and DEBUG messages about similarity scores."""
        scores = df_top_n["concept_similarity"]

        for i, (_, row) in enumerate(df_top_n.iterrows(), 1):
            text = str(row.get("clean_text", "")).strip()
            truncated = text[:80] + "..." if len(text) > 80 else text
            logger.info(
                "Top %d: similarity=%.4f  |  %s",
                i,
                row["concept_similarity"],
                truncated,
            )

        sim = df["concept_similarity"]
        logger.debug(
            "Similarity distribution (n=%d): min=%.4f  max=%.4f  "
            "mean=%.4f  std=%.4f",
            len(sim),
            sim.min(),
            sim.max(),
            sim.mean(),
            sim.std(),
        )
