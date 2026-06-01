"""Text cleaning and tokenization with language-specific preprocessing.

Truncates, cleans, and prepares text for downstream modules
(sentiment analysis, embedding computation). The processing depth
differs radically between the VADER and pysentimiento paths.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from src.utils.language_config import LanguageConfig
from src.utils.validators import MAX_CHARS

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_PATTERN = re.compile(r"\s+")


class Preprocessor:
    """Clean and tokenise text according to language-specific rules.

    Args:
        config: Language configuration that determines which
            preprocessing path to follow (VADER or pysentimiento).
    """

    def __init__(self, config: LanguageConfig) -> None:
        self.config = config
        self._truncation_count = 0

        if config.sentiment_backend == "vader":
            from nltk.corpus import stopwords as nltk_stopwords

            import spacy

            self.nlp = spacy.load(config.spacy_model)
            self._stopwords = set(
                nltk_stopwords.words(config.stopwords_name)
            )
            self._negation_set = set(config.negation_tokens)
        else:
            self.nlp = None
            self._stopwords = None
            self._negation_set = None

    # ── Public entry point ─────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to every row and add ``clean_text`` / ``tokens``.

        Args:
            df: DataFrame with a ``raw_text`` column.

        Returns:
            DataFrame with added ``clean_text`` and ``tokens`` columns.
        """
        self._truncation_count = 0
        clean_texts: list[str] = []
        token_lists: list[list[str]] = []

        for idx, text in enumerate(df["raw_text"]):
            if not isinstance(text, str):
                text = ""

            text = self._truncate(text)
            text = self._clean(text)

            tokens_pre = text.split()
            if len(tokens_pre) < 2:
                logger.warning(
                    "Row %d dropped: < 2 tokens after cleaning. "
                    "Preview: '%s'",
                    idx, text[:80],
                )
                clean_texts.append("")
                token_lists.append([])
                continue

            if self.config.sentiment_backend == "vader":
                text = self._process_vader(text)
            else:
                text = self._process_transformer(text)

            tokens = text.split()
            clean_texts.append(text)
            token_lists.append(tokens)

        df = df.copy()
        df["clean_text"] = clean_texts
        df["tokens"] = token_lists

        avg_len = (
            sum(len(t) for t in token_lists) / len(token_lists)
            if token_lists
            else 0.0
        )
        total = len(df)
        logger.info(
            "Preprocessing complete — %d docs, avg token length %.1f, "
            "%d truncations (%.1f%%).",
            total,
            avg_len,
            self._truncation_count,
            (self._truncation_count / total * 100) if total else 0.0,
        )
        return df

    # ── Step 1: Truncation ─────────────────────────────────────────

    def _truncate(self, text: str) -> str:
        """Truncate to ``MAX_CHARS`` and log a warning for every cut."""
        if len(text) > MAX_CHARS:
            logger.warning(
                "Text truncated from %d to %d chars. Preview: '%s...'",
                len(text),
                MAX_CHARS,
                text[:50],
            )
            self._truncation_count += 1
            return text[:MAX_CHARS]
        return text

    # ── Step 2: Universal cleaning ──────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Lowercase, remove URLs, numbers, and special characters."""
        text = URL_PATTERN.sub("", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = text.replace("_", "")
        text = text.lower()
        text = MULTISPACE_PATTERN.sub(" ", text).strip()
        return text

    # ── Step 3a: VADER path (English) ───────────────────────────────

    def _process_vader(self, text: str) -> str:
        """Stopword removal → spaCy lemmatisation → NEG_ prefix."""
        doc = self.nlp(text)

        lemmas: list[str] = []
        for token in doc:
            # Keep negation tokens even if they are stopwords
            if token.text in self._negation_set:
                lemmas.append(token.text)
            elif token.text not in self._stopwords:
                lemmas.append(token.lemma_)

        # NEG_ prefix: mark the 3 tokens after each negation token
        negated: list[str] = []
        i = 0
        while i < len(lemmas):
            if lemmas[i] in self._negation_set:
                negated.append(lemmas[i])
                for _ in range(3):
                    i += 1
                    if i < len(lemmas):
                        negated.append(f"NEG_{lemmas[i]}")
            else:
                negated.append(lemmas[i])
            i += 1

        return " ".join(negated)

    # ── Step 3b: Transformer path (pysentimiento) ───────────────────

    @staticmethod
    def _process_transformer(text: str) -> str:
        """Syntax left intact — no stopword removal, no lemmatisation.

        Returns the universally cleaned text as-is. Transformers
        handle stopwords, morphology, and negation natively through
        bidirectional attention.
        """
        return text
