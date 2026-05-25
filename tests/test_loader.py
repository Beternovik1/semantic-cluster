"""Unit tests for src/pipeline/loader.py."""

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.loader import DataLoader


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def utf8_csv(tmp_path: Path) -> Path:
    """Standard UTF-8 CSV with three columns."""
    path = tmp_path / "utf8.csv"
    path.write_text(
        "review_text,rating,date\n"
        "Great hotel,5,2024-01-01\n"
        "Bad service,1,2024-01-02\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def latin1_csv(tmp_path: Path) -> Path:
    """CSV containing ISO-8859-1 bytes that will trigger UnicodeDecodeError.

    ``\\xe1`` = á (latin-1), ``\\xf1`` = ñ (latin-1).
    Both are invalid start bytes in UTF-8.
    """
    path = tmp_path / "latin1.csv"
    path.write_bytes(b"texto,puntaje\nMa\xf1ana,5\nCami\xf3n,3\n")
    return path


# ── Successful UTF-8 load ──────────────────────────────────────────────────

class TestUtf8Load:
    def test_returns_dataframe(self, utf8_csv: Path) -> None:
        loader = DataLoader(utf8_csv, "review_text")
        df = loader.load()
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, utf8_csv: Path) -> None:
        df = DataLoader(utf8_csv, "review_text").load()
        assert len(df) == 2

    def test_retains_additional_columns(self, utf8_csv: Path) -> None:
        df = DataLoader(utf8_csv, "review_text").load()
        assert "rating" in df.columns
        assert "date" in df.columns


# ── Latin-1 fallback ───────────────────────────────────────────────────────

class TestLatin1Fallback:
    def test_loads_successfully(self, latin1_csv: Path) -> None:
        df = DataLoader(latin1_csv, "texto").load()
        assert len(df) == 2

    def test_decodes_accented_characters(self, latin1_csv: Path) -> None:
        df = DataLoader(latin1_csv, "texto").load()
        assert "Mañana" in df["raw_text"].values
        assert "Camión" in df["raw_text"].values

    def test_logs_warning(self, latin1_csv: Path, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("WARNING")
        DataLoader(latin1_csv, "texto").load()
        assert "UTF-8 decoding failed" in caplog.text
        assert "latin-1" in caplog.text


# ── Missing column ─────────────────────────────────────────────────────────

class TestMissingColumn:
    def test_raises_value_error(self, utf8_csv: Path) -> None:
        loader = DataLoader(utf8_csv, "nonexistent_column")
        with pytest.raises(ValueError, match="not found in CSV"):
            loader.load()

    def test_error_lists_available_columns(self, utf8_csv: Path) -> None:
        loader = DataLoader(utf8_csv, "wrong_col")
        with pytest.raises(ValueError, match="review_text"):
            loader.load()


# ── Null dropping ──────────────────────────────────────────────────────────

class TestNullDropping:
    @pytest.fixture
    def partial_nulls_csv(self, tmp_path: Path) -> Path:
        """Two valid rows, one blank, one explicit NaN."""
        path = tmp_path / "partial_nulls.csv"
        path.write_text(
            "text,val\nhello,1\n,2\nworld,3\nNaN,4\n",
            encoding="utf-8",
        )
        return path

    def test_drops_null_rows(self, partial_nulls_csv: Path) -> None:
        df = DataLoader(partial_nulls_csv, "text").load()
        assert len(df) == 2  # "hello" and "world"

    def test_keeps_valid_rows(self, partial_nulls_csv: Path) -> None:
        df = DataLoader(partial_nulls_csv, "text").load()
        assert "hello" in df["raw_text"].values
        assert "world" in df["raw_text"].values

    def test_logs_drop_count(self, partial_nulls_csv: Path, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("WARNING")
        DataLoader(partial_nulls_csv, "text").load()
        assert "Dropped" in caplog.text
        assert "null" in caplog.text


# ── All-null → empty → ValueError ──────────────────────────────────────────

class TestAllNull:
    @pytest.fixture
    def all_nulls_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "all_nulls.csv"
        path.write_text("text,val\n,\n,\n", encoding="utf-8")
        return path

    def test_raises_value_error(self, all_nulls_csv: Path) -> None:
        loader = DataLoader(all_nulls_csv, "text")
        with pytest.raises(ValueError, match="DataFrame is empty after loading"):
            loader.load()


# ── Column mapping ─────────────────────────────────────────────────────────

class TestColumnMapping:
    def test_raw_text_column_exists(self, utf8_csv: Path) -> None:
        df = DataLoader(utf8_csv, "review_text").load()
        assert "raw_text" in df.columns

    def test_original_column_preserved(self, utf8_csv: Path) -> None:
        df = DataLoader(utf8_csv, "review_text").load()
        assert "review_text" in df.columns

    def test_raw_text_contains_original_data(self, utf8_csv: Path) -> None:
        df = DataLoader(utf8_csv, "review_text").load()
        assert df["raw_text"].iloc[0] == "Great hotel"

    @pytest.mark.parametrize("col_name", ["text", "comment", "body"])
    def test_different_column_names_mapped_correctly(
        self, tmp_path: Path, col_name: str,
    ) -> None:
        path = tmp_path / f"{col_name}.csv"
        path.write_text(f"{col_name},val\nHello,1\n", encoding="utf-8")
        df = DataLoader(path, col_name).load()
        assert "raw_text" in df.columns
        assert col_name in df.columns
        assert df["raw_text"].iloc[0] == "Hello"
        assert df[col_name].iloc[0] == "Hello"
