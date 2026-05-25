"""Language-specific NLP metadata configuration.

Returns strings, booleans, and tuples — never imports or instantiates
any NLP model. Downstream modules (preprocessor, sentiment analyzer)
are responsible for loading the actual model libraries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Language registry ──────────────────────────────────────────────────────
# Each entry contains metadata only — no model instances, no imports.

_LANG_CONFIGS: dict[str, dict] = {
    "es": {
        "spacy_model": "es_core_news_sm",
        "stopwords_name": "spanish",
        "use_lemmatization": False,
        "negation_tokens": (
            "no", "nunca", "jamás", "tampoco", "sin", "ni",
            "nada", "nadie", "ningún", "ninguna",
        ),
        "sentiment_backend": "pysentimiento",
    },
    "en": {
        "spacy_model": "en_core_web_sm",
        "stopwords_name": "english",
        "use_lemmatization": True,
        "negation_tokens": (
            "not", "never", "no", "neither", "nor",
            "without", "hardly", "barely", "scarcely",
        ),
        "sentiment_backend": "vader",
    },
    "fr": {
        "spacy_model": "fr_core_news_sm",
        "stopwords_name": "french",
        "use_lemmatization": False,
        "negation_tokens": (),
        "sentiment_backend": "pysentimiento",
    },
    "de": {
        "spacy_model": "de_core_news_sm",
        "stopwords_name": "german",
        "use_lemmatization": False,
        "negation_tokens": (),
        "sentiment_backend": "pysentimiento",
    },
    "pt": {
        "spacy_model": "pt_core_news_sm",
        "stopwords_name": "portuguese",
        "use_lemmatization": False,
        "negation_tokens": (),
        "sentiment_backend": "pysentimiento",
    },
}

SUPPORTED_LANGUAGES = frozenset(_LANG_CONFIGS)


# ── Public interface ───────────────────────────────────────────────────────

class LanguageConfig:
    """NLP configuration metadata for a single language.

    Attributes:
        lang: ISO 639-1 language code.
        spacy_model: Name of the spaCy model to load.
        stopwords_name: NLTK stopwords corpus name.
        use_lemmatization: Whether the preprocessor should lemmatize
            (``True`` for VADER/English, ``False`` for transformer-based).
        negation_tokens: Tokens that flip sentiment polarity. Only
            meaningful for the VADER (English) path — never applied
            when ``sentiment_backend`` is ``"pysentimiento"``.
        sentiment_backend: ``"pysentimiento"`` or ``"vader"``.
    """

    def __init__(self, lang: str) -> None:
        config = _LANG_CONFIGS.get(lang)
        if config is None:
            raise ValueError(
                f"Unsupported language code: '{lang}'. "
                f"Supported codes: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
            )
        self.lang: str = lang
        self.spacy_model: str = config["spacy_model"]
        self.stopwords_name: str = config["stopwords_name"]
        self.use_lemmatization: bool = config["use_lemmatization"]
        self.negation_tokens: tuple[str, ...] = config["negation_tokens"]
        self.sentiment_backend: str = config["sentiment_backend"]

    def __repr__(self) -> str:
        return (
            f"LanguageConfig(lang={self.lang!r}, "
            f"backend={self.sentiment_backend!r})"
        )
