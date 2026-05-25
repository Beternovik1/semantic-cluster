"""
CLI argument validation and pipeline-wide constants.

Validates all user-provided inputs before the pipeline starts,
provides actionable error messages, and defines project-wide constants
imported by every module in the pipeline.
"""

import csv
import difflib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Project-wide constants ─────────────────────────────────────────────────

RANDOM_SEED = 42
MIN_PARTITION_SIZE = 100
MAX_SCATTER_POINTS = 1000
SENTIMENT_BATCH_SIZE = 32
MAX_CONTAMINATION = 0.15
MAX_CHARS = 600

SUPPORTED_LANGUAGES = frozenset({"es", "en", "fr", "de", "pt"})


# ── Validation ─────────────────────────────────────────────────────────────

def validate_args(args) -> None:
    """Validate CLI arguments and create the output directory.

    Each validation failure raises ``ValueError`` with a message that
    tells the user exactly what is wrong and how to fix it. The column
    check reads only the CSV header row via stdlib ``csv`` — pandas is
    never imported here per project convention.

    Args:
        args: ``argparse.Namespace`` produced by ``main.build_parser()``.

    Raises:
        ValueError: CSV file does not exist.
        ValueError: Column name is not present in the CSV header.
        ValueError: Language code is not in ``SUPPORTED_LANGUAGES``.
        ValueError: Contamination is outside the valid range.
    """
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    _check_file_exists(args.file)
    _check_column_in_csv(args.file, args.column)
    _check_language(args.lang)
    _check_contamination(args.contamination)

    args.output.mkdir(parents=True, exist_ok=True)

    logger.debug(
        "Validated parameters: file=%s, column=%s, lang=%s, title=%s, "
        "palette=%s, output=%s, concept=%s, contamination=%.3f, verbose=%s",
        args.file,
        args.column,
        args.lang,
        args.title,
        args.palette,
        args.output,
        args.concept,
        args.contamination,
        args.verbose,
    )


def _check_file_exists(path: Path) -> None:
    if not path.exists():
        raise ValueError(
            f"CSV file not found: {path}. "
            "Please verify the --file path and try again."
        )


def _check_column_in_csv(path: Path, column: str) -> None:
    """Read only the CSV header row to validate the column name."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                headers = next(csv.reader(f))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"Could not read CSV file headers: {path}. "
            "File encoding is neither UTF-8 nor Latin-1."
        )

    if column not in headers:
        msg = f"Column '{column}' not found in CSV."
        match = difflib.get_close_matches(column, headers, n=1, cutoff=0.6)
        if match:
            msg += f" Did you mean '{match[0]}'?"
        msg += f"\nAvailable columns: {', '.join(headers)}"
        raise ValueError(msg)


def _check_language(lang: str) -> None:
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language code: '{lang}'. "
            f"Supported codes: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
        )


def _check_contamination(value: float) -> None:
    low, high = 0.01, MAX_CONTAMINATION
    if not low <= value <= high:
        raise ValueError(
            f"Contamination must be between {low} and {high}. Got: {value}."
        )
