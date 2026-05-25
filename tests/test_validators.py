"""Unit tests for src/utils/validators.py."""

from argparse import Namespace
from pathlib import Path

import pytest

from src.utils.validators import (
    MAX_CHARS,
    MAX_CONTAMINATION,
    MAX_SCATTER_POINTS,
    MIN_PARTITION_SIZE,
    RANDOM_SEED,
    SENTIMENT_BATCH_SIZE,
    SUPPORTED_LANGUAGES,
    validate_args,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Write a minimal 2-line CSV to a temp directory and return its path."""
    path = tmp_path / "reviews.csv"
    path.write_text("review_text,rating,date\nGreat hotel,5,2024-01-01\n", encoding="utf-8")
    return path


@pytest.fixture
def valid_args(valid_csv: Path, tmp_path: Path) -> Namespace:
    """Return a Namespace with fully valid arguments."""
    return Namespace(
        file=valid_csv,
        column="review_text",
        lang="es",
        title="Test Report",
        palette="viridis",
        output=tmp_path / "out",
        concept="precio valor costo",
        contamination=0.05,
        verbose=False,
    )


# ── Constants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_randome_seed(self) -> None:
        assert RANDOM_SEED == 42

    def test_min_partition_size(self) -> None:
        assert MIN_PARTITION_SIZE == 100

    def test_max_scatter_points(self) -> None:
        assert MAX_SCATTER_POINTS == 1000

    def test_sentiment_batch_size(self) -> None:
        assert SENTIMENT_BATCH_SIZE == 32

    def test_max_contamination(self) -> None:
        assert MAX_CONTAMINATION == 0.15

    def test_max_chars(self) -> None:
        assert MAX_CHARS == 600

    def test_supported_languages(self) -> None:
        assert SUPPORTED_LANGUAGES == {"es", "en", "fr", "de", "pt"}


# ── File validation ────────────────────────────────────────────────────────

class TestFileValidation:
    def test_missing_file_raises_value_error(self) -> None:
        fake = Path("/does/not/exist.csv")
        args = Namespace(
            file=fake, column="text", lang="es", title="T",
            palette="viridis", output=Path("/tmp"), concept="x",
            contamination=0.05, verbose=False,
        )
        with pytest.raises(ValueError, match="CSV file not found"):
            validate_args(args)


# ── Column validation ──────────────────────────────────────────────────────

class TestColumnValidation:
    def test_wrong_column_raises_value_error(self, valid_csv: Path, tmp_path: Path) -> None:
        args = Namespace(
            file=valid_csv, column="nonexistent", lang="es", title="T",
            palette="viridis", output=tmp_path / "out", concept="x",
            contamination=0.05, verbose=False,
        )
        with pytest.raises(ValueError, match="Column 'nonexistent' not found in CSV"):
            validate_args(args)

    def test_did_you_mean_suggestion(self, valid_csv: Path, tmp_path: Path) -> None:
        args = Namespace(
            file=valid_csv, column="review-text", lang="es", title="T",
            palette="viridis", output=tmp_path / "out", concept="x",
            contamination=0.05, verbose=False,
        )
        with pytest.raises(ValueError, match="Did you mean 'review_text'"):
            validate_args(args)

    def test_error_message_lists_columns(self, valid_csv: Path, tmp_path: Path) -> None:
        args = Namespace(
            file=valid_csv, column="oops", lang="es", title="T",
            palette="viridis", output=tmp_path / "out", concept="x",
            contamination=0.05, verbose=False,
        )
        with pytest.raises(ValueError, match="Available columns: review_text, rating, date"):
            validate_args(args)


# ── Language validation ────────────────────────────────────────────────────

class TestLanguageValidation:
    @pytest.mark.parametrize("bad_lang", ["it", "ru", "zh", "ja", "ar", "xx"])
    def test_unsupported_language_raises_value_error(
        self, valid_csv: Path, tmp_path: Path, bad_lang: str,
    ) -> None:
        args = Namespace(
            file=valid_csv, column="review_text", lang=bad_lang, title="T",
            palette="viridis", output=tmp_path / "out", concept="x",
            contamination=0.05, verbose=False,
        )
        with pytest.raises(ValueError, match="Unsupported language code"):
            validate_args(args)


# ── Contamination validation ───────────────────────────────────────────────

class TestContaminationValidation:
    @pytest.mark.parametrize("bad_value", [0.0, 0.001, -0.01, 0.16, 0.5, 1.0])
    def test_out_of_bounds_contamination_raises_value_error(
        self, valid_csv: Path, tmp_path: Path, bad_value: float,
    ) -> None:
        args = Namespace(
            file=valid_csv, column="review_text", lang="es", title="T",
            palette="viridis", output=tmp_path / "out", concept="x",
            contamination=bad_value, verbose=False,
        )
        with pytest.raises(ValueError, match="Contamination must be between"):
            validate_args(args)

    @pytest.mark.parametrize("good_value", [0.01, 0.05, 0.10, 0.15])
    def test_valid_contamination_passes(
        self, valid_csv: Path, tmp_path: Path, good_value: float,
    ) -> None:
        args = Namespace(
            file=valid_csv, column="review_text", lang="es", title="T",
            palette="viridis", output=tmp_path / "out", concept="x",
            contamination=good_value, verbose=False,
        )
        validate_args(args)


# ── Happy path ─────────────────────────────────────────────────────────────

class TestValidArgs:
    def test_valid_args_pass_without_error(self, valid_args: Namespace) -> None:
        validate_args(valid_args)

    def test_valid_args_creates_output_directory(self, valid_args: Namespace) -> None:
        validate_args(valid_args)
        assert valid_args.output.exists()
        assert valid_args.output.is_dir()

    @pytest.mark.parametrize("lang", ["es", "en", "fr", "de", "pt"])
    def test_all_supported_languages_pass(
        self, valid_csv: Path, tmp_path: Path, lang: str,
    ) -> None:
        args = Namespace(
            file=valid_csv, column="review_text", lang=lang, title="T",
            palette="viridis", output=tmp_path / "out", concept="x",
            contamination=0.05, verbose=False,
        )
        validate_args(args)
