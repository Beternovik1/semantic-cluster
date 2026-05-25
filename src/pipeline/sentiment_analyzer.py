"""Language-aware sentiment analysis — pysentimiento (es/fr/pt/de) or VADER (en).

Processes text in batches with a ``tqdm`` progress bar for the
pysentimiento path.  Returns a strictly binary "pos" / "neg" label
with a polarity score and a confidence flag.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from src.utils.validators import MIN_PARTITION_SIZE, SENTIMENT_BATCH_SIZE

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Classify each document as positive or negative.

    Args:
        lang_cfg: A ``LanguageConfig`` instance that provides
            ``sentiment_backend`` ("pysentimiento" or "vader").
        pysentimiento_analyzer: A pre-instantiated pysentimiento
            sentiment analyzer, or ``None`` for English (VADER path).
    """

    def __init__(self, lang_cfg, pysentimiento_analyzer) -> None:
        self._lang_cfg = lang_cfg
        self._pysentimiento = pysentimiento_analyzer
        self._vader = SentimentIntensityAnalyzer()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``sentiment``, ``polarity``, and ``sentiment_confidence``.

        Args:
            df: DataFrame with a ``clean_text`` column.

        Returns:
            A copy of *df* with three new columns.
        """
        df = df.copy()

        if self._lang_cfg.sentiment_backend == "vader":
            self._vader_path(df)
        else:
            self._pysentimiento_path(df)

        self._log_summary(df)
        return df

    # ── VADER path (English) ───────────────────────────────────────

    def _vader_path(self, df: pd.DataFrame) -> None:
        """Annotate df using VADER compound score + TextBlob secondary."""
        sentiments: list[str] = []
        polarities: list[float] = []
        confidences: list[bool] = []

        for text in df["clean_text"]:
            if not isinstance(text, str) or not text.strip():
                sentiments.append("neg")
                polarities.append(0.0)
                confidences.append(False)
                continue

            vs = self._vader.polarity_scores(text)
            compound: float = vs["compound"]

            tb = TextBlob(text)
            tb_polarity: float = tb.sentiment.polarity

            label = "pos" if compound >= 0.05 else "neg"
            vader_label = label
            tb_label = "pos" if tb_polarity >= 0.05 else "neg"

            if vader_label != tb_label:
                label = vader_label  # VADER wins on disagreement

            sentiments.append(label)
            polarities.append(compound)
            confidences.append(vader_label == tb_label)

            logger.debug(
                "VADER=%.4f(%s) TextBlob=%.4f(%s) → %s",
                compound,
                vader_label,
                tb_polarity,
                tb_label,
                label,
            )

        df["sentiment"] = sentiments
        df["polarity"] = polarities
        df["sentiment_confidence"] = confidences

    # ── pysentimiento path (es, fr, pt, de) ────────────────────────

    def _pysentimiento_path(self, df: pd.DataFrame) -> None:
        """Annotate df using batched pysentimiento inference."""
        texts: list[str] = df["clean_text"].tolist()
        n: int = len(texts)

        sentiments: list[str] = []
        polarities: list[float] = []
        confidences: list[bool] = []

        batches: list[list[str]] = [
            texts[i : i + SENTIMENT_BATCH_SIZE]
            for i in range(0, n, SENTIMENT_BATCH_SIZE)
        ]

        for batch in tqdm(batches, desc="Analyzing sentiment", unit="batch"):
            results = self._pysentimiento.predict(batch)

            for result in results:
                probas: dict[str, float] = result.probas
                raw_output: str = result.output

                # Resolve NEU using model confidence
                if raw_output == "NEU":
                    label = "pos" if probas["POS"] >= probas["NEG"] else "neg"
                    confident = False
                else:
                    label = "pos" if raw_output == "POS" else "neg"
                    confident = True

                polarity: float = probas["POS"] - probas["NEG"]

                sentiments.append(label)
                polarities.append(polarity)
                confidences.append(confident)

                logger.debug(
                    "pysentimiento output=%s probas=POS:%.3f NEG:%.3f "
                    "NEU:%.3f → %s (confident=%s)",
                    raw_output,
                    probas.get("POS", 0.0),
                    probas.get("NEG", 0.0),
                    probas.get("NEU", 0.0),
                    label,
                    confident,
                )

        df["sentiment"] = sentiments
        df["polarity"] = polarities
        df["sentiment_confidence"] = confidences

    # ── Summary logging ────────────────────────────────────────────

    def _log_summary(self, df: pd.DataFrame) -> None:
        """Emit INFO and WARNING messages about classification results."""
        pos_count: int = int((df["sentiment"] == "pos").sum())
        neg_count: int = int((df["sentiment"] == "neg").sum())
        total: int = len(df)
        conf_rate: float = (
            df["sentiment_confidence"].sum() / total if total else 0.0
        )

        logger.info(
            "Sentiment: %d pos / %d neg / %d total  (confidence=%.1f%%).",
            pos_count,
            neg_count,
            total,
            conf_rate * 100,
        )

        if pos_count < MIN_PARTITION_SIZE:
            logger.warning(
                "Positive group size (%d) below MIN_PARTITION_SIZE (%d). "
                "Topic modeling will use word-frequency fallback.",
                pos_count,
                MIN_PARTITION_SIZE,
            )

        if neg_count < MIN_PARTITION_SIZE:
            logger.warning(
                "Negative group size (%d) below MIN_PARTITION_SIZE (%d). "
                "Topic modeling will use word-frequency fallback.",
                neg_count,
                MIN_PARTITION_SIZE,
            )
