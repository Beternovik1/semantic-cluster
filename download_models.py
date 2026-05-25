"""One-time setup script to warm the HuggingFace, NLTK, and spaCy model caches.

Run this with a working internet connection **before** running ``main.py``
offline.  Instantiates every NLP model the pipeline can use, forcing the
underlying libraries to download and cache their data into their native
directories (``~/.cache/huggingface``, ``~/nltk_data``, etc.).

Usage:
    python download_models.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time

import nltk
import spacy.cli
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── NLTK corpora required by TextBlob + VADER + stopword removal ──────
# TextBlob's POS tagger triggers auto-download if these are missing.
# Auto-download fails silently offline → pipeline crash.
_NLTK_RESOURCES: list[str] = [
    "stopwords",
    "vader_lexicon",
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "brown",
]

_SPACY_MODEL = "en_core_web_sm"
_SENTENCE_TRANSFORMER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _download_nltk_data() -> None:
    """Download all NLTK corpora and tokenizers required at runtime."""
    logger.info("Downloading NLTK data (%d resources) ...", len(_NLTK_RESOURCES))
    for resource in _NLTK_RESOURCES:
        try:
            nltk.download(resource, quiet=False, raise_on_error=True)
            logger.info("  NLTK '%s' — OK", resource)
        except Exception:
            logger.warning(
                "  NLTK '%s' — FAILED (pipeline may crash offline)", resource,
                exc_info=True,
            )


def _download_spacy_model() -> None:
    """Download the spaCy model used by the English VADER path.

    Only the English model is needed because pysentimiento languages
    (es/fr/pt/de) skip spaCy entirely in the preprocessor.
    """
    logger.info("Downloading spaCy model '%s' ...", _SPACY_MODEL)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", _SPACY_MODEL],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        logger.info("  spaCy '%s' — OK", _SPACY_MODEL)
    except subprocess.CalledProcessError:
        logger.warning(
            "  spaCy '%s' — FAILED (English sentiment path may crash)",
            _SPACY_MODEL,
            exc_info=True,
        )


def _load_sentence_transformer() -> None:
    """Instantiate the sentence embedding model to populate the HF cache."""
    logger.info(
        "Loading SentenceTransformer '%s' ...", _SENTENCE_TRANSFORMER_MODEL
    )
    try:
        model = SentenceTransformer(_SENTENCE_TRANSFORMER_MODEL)
        model.max_seq_length = 512
        logger.info("  SentenceTransformer — OK (max_seq_length=512)")
    except Exception:
        logger.warning(
            "  SentenceTransformer — FAILED (embeddings will not work)",
            exc_info=True,
        )


def _load_pysentimiento_spanish() -> None:
    """Instantiate the pysentimiento analyzer for Spanish to cache its model.

    Spanish is the primary non-English demo language.  French, Portuguese,
    and German analyzers are not loaded here to save bandwidth and disk.
    """
    logger.info("Loading pysentimiento analyzer (lang=es) ...")
    try:
        from pysentimiento import create_analyzer

        analyzer = create_analyzer(task="sentiment", lang="es")
        # Run one trivial prediction to verify the pipeline is wired
        result = analyzer.predict("Muy bueno")
        logger.info(
            "  pysentimiento (es) — OK (sanity check: '%s')", result.output
        )
    except Exception:
        logger.warning(
            "  pysentimiento (es) — FAILED (Spanish sentiment will crash)",
            exc_info=True,
        )


def main() -> None:
    """Warm all model caches so the pipeline can run fully offline.

    Each download/instantiation is wrapped in ``try/except`` so a single
    failure does not prevent the remaining models from being cached.
    """
    start = time.time()

    _download_nltk_data()
    _download_spacy_model()
    _load_sentence_transformer()
    _load_pysentimiento_spanish()

    elapsed = time.time() - start
    logger.info(
        "download_models.py finished in %.1f seconds. "
        "The pipeline should now run fully offline.",
        elapsed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
