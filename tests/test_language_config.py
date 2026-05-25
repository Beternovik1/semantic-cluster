"""Unit tests for src/utils/language_config.py."""

import pytest

from src.utils.language_config import LanguageConfig, SUPPORTED_LANGUAGES


# ── Unsupported language ───────────────────────────────────────────────────

class TestUnsupportedLanguage:
    @pytest.mark.parametrize("bad_lang", ["it", "ru", "zh", "ja", "ar", "xx"])
    def test_raises_value_error(self, bad_lang: str) -> None:
        with pytest.raises(ValueError, match="Unsupported language code"):
            LanguageConfig(bad_lang)


# ── English specific ───────────────────────────────────────────────────────

class TestEnglishConfig:
    @pytest.fixture
    def cfg(self) -> LanguageConfig:
        return LanguageConfig("en")

    def test_sentiment_backend_is_vader(self, cfg: LanguageConfig) -> None:
        assert cfg.sentiment_backend == "vader"

    def test_use_lemmatization_is_true(self, cfg: LanguageConfig) -> None:
        assert cfg.use_lemmatization is True

    def test_spacy_model(self, cfg: LanguageConfig) -> None:
        assert cfg.spacy_model == "en_core_web_sm"

    def test_negation_tokens_exist(self, cfg: LanguageConfig) -> None:
        assert len(cfg.negation_tokens) > 0
        assert isinstance(cfg.negation_tokens, tuple)

    def test_negation_tokens_contain_standard_negators(
        self, cfg: LanguageConfig,
    ) -> None:
        assert "not" in cfg.negation_tokens
        assert "never" in cfg.negation_tokens
        assert "no" in cfg.negation_tokens


# ── Spanish specific ───────────────────────────────────────────────────────

class TestSpanishConfig:
    @pytest.fixture
    def cfg(self) -> LanguageConfig:
        return LanguageConfig("es")

    def test_sentiment_backend_is_pysentimiento(self, cfg: LanguageConfig) -> None:
        assert cfg.sentiment_backend == "pysentimiento"

    def test_use_lemmatization_is_false(self, cfg: LanguageConfig) -> None:
        assert cfg.use_lemmatization is False

    def test_spacy_model(self, cfg: LanguageConfig) -> None:
        assert cfg.spacy_model == "es_core_news_sm"

    def test_negation_tokens_exist(self, cfg: LanguageConfig) -> None:
        assert len(cfg.negation_tokens) > 0

    def test_stopwords_name(self, cfg: LanguageConfig) -> None:
        assert cfg.stopwords_name == "spanish"


# ── Transformer languages (es, fr, pt, de) ─────────────────────────────────

class TestTransformerLanguages:
    @pytest.mark.parametrize("lang", ["es", "fr", "pt", "de"])
    def test_use_lemmatization_is_false(self, lang: str) -> None:
        cfg = LanguageConfig(lang)
        assert cfg.use_lemmatization is False

    @pytest.mark.parametrize("lang", ["es", "fr", "pt", "de"])
    def test_sentiment_backend_is_pysentimiento(self, lang: str) -> None:
        cfg = LanguageConfig(lang)
        assert cfg.sentiment_backend == "pysentimiento"


# ── SUPPORTED_LANGUAGES ────────────────────────────────────────────────────

class TestSupportedLanguages:
    def test_all_five_codes(self) -> None:
        assert SUPPORTED_LANGUAGES == {"es", "en", "fr", "de", "pt"}
