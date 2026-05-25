"""CSV file loading with encoding fallback and structural validation.

This is the only module permitted to read input files from disk.
It handles encoding negotiation, null-row removal, and column
standardisation — no NLP processing of any kind.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Read a CSV into a sanitised DataFrame with a standard ``raw_text`` column.

    Args:
        file_path: Path to the input CSV.
        text_column: Name of the column containing the text to analyse.

    Raises:
        FileNotFoundError: *file_path* does not exist.
        ValueError: *text_column* is missing from the CSV, or the
            DataFrame is empty after dropping null rows.
    """

    def __init__(self, file_path: Path, text_column: str) -> None:
        self.file_path = Path(file_path)
        self.text_column = text_column

    def load(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {self.file_path}"
            )

        df = self._read_csv()
        self._validate_column(df)
        df = self._standardise(df)
        df = self._drop_null(df)
        self._log_summary(df)
        return df

    # ── Internal helpers ───────────────────────────────────────────

    def _read_csv(self) -> pd.DataFrame:
        """Attempt UTF-8 first, fall back to Latin-1 on decode error."""
        try:
            return pd.read_csv(self.file_path, encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decoding failed for %s. Retrying with latin-1 encoding.",
                self.file_path,
            )
            return pd.read_csv(self.file_path, encoding="latin-1")

    def _validate_column(self, df: pd.DataFrame) -> None:
        if self.text_column not in df.columns:
            raise ValueError(
                f"Column '{self.text_column}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

    def _standardise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``raw_text`` column while preserving the original column."""
        df["raw_text"] = df[self.text_column]
        return df

    def _drop_null(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where ``raw_text`` is null.

        NaN values in the text column would propagate through
        embedding computation and corrupt downstream models.
        Dropping early avoids silent NaN handling everywhere else.
        """
        before = len(df)
        df = df.dropna(subset=["raw_text"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(
                "Dropped %d row(s) with null raw_text.", dropped,
            )
        return df

    def _log_summary(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError(
                f"DataFrame is empty after loading {self.file_path}. "
                "No rows remaining after null removal."
            )
        logger.info(
            "Loaded %s — %d rows, %d columns.",
            self.file_path,
            len(df),
            len(df.columns),
        )
