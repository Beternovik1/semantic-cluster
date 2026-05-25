"""Unit tests for src/pipeline/preprocessor.py."""

import pandas as pd
import pytest

from src.pipeline.preprocessor import Preprocessor
from src.utils.language_config import LanguageConfig
from src.utils.validators import MAX_CHARS


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def en_preprocessor():
    """VADER-path Preprocessor (English). Loads spaCy once per class."""
    return Preprocessor(LanguageConfig("en"))


@pytest.fixture(scope="class")
def es_preprocessor():
    """Transformer-path Preprocessor (Spanish)."""
    return Preprocessor(LanguageConfig("es"))


# ── Helpers ────────────────────────────────────────────────────────────────

def _run(preprocessor: Preprocessor, texts: list[str]) -> pd.DataFrame:
    return preprocessor.run(pd.DataFrame({"raw_text": texts}))


# ════════════════════════════════════════════════════════════════════════════
# Truncation
# ════════════════════════════════════════════════════════════════════════════

class TestTruncation:
    def test_truncates_long_text(self, en_preprocessor: Preprocessor) -> None:
        long = "a" * (MAX_CHARS + 50)
        df = _run(en_preprocessor, [long])
        assert len(df.iloc[0]["clean_text"]) == MAX_CHARS

    def test_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("WARNING")
        prep = Preprocessor(LanguageConfig("en"))
        _run(prep, ["x" * (MAX_CHARS + 1)])
        assert "Text truncated from" in caplog.text

    def test_does_not_truncate_short_text(self, en_preprocessor: Preprocessor) -> None:
        short = "hello world"
        df = _run(en_preprocessor, [short])
        assert df.iloc[0]["clean_text"] == "hello world"

    def test_counts_truncations_in_info_log(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        prep = Preprocessor(LanguageConfig("en"))
        _run(prep, ["a" * (MAX_CHARS + 10), "short", "b" * (MAX_CHARS + 5)])
        assert "1 truncations" in caplog.text or "2 truncations" in caplog.text


# ════════════════════════════════════════════════════════════════════════════
# Universal cleaning (applied to ALL languages)
# ════════════════════════════════════════════════════════════════════════════

class TestUniversalCleaning:
    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_removes_urls(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["Visit http://example.com now"])
        assert "http" not in df.iloc[0]["clean_text"]
        assert "example" not in df.iloc[0]["clean_text"]

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_removes_numbers(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["Score 10 out of 100"])
        assert all(not c.isdigit() for c in df.iloc[0]["clean_text"])

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_removes_special_characters(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["Hello!!! How are you? (fine)."])
        clean = df.iloc[0]["clean_text"]
        assert "!" not in clean
        assert "?" not in clean
        assert "(" not in clean
        assert ")" not in clean
        assert "." not in clean

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_lowercases(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["HELLO WORLD"])
        assert df.iloc[0]["clean_text"] == "hello world"

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_combined_cleaning(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["Hello! Visit http://x.com. Rate 5/5."])
        assert df.iloc[0]["clean_text"] == "hello visit rate"


# ════════════════════════════════════════════════════════════════════════════
# VADER path (English)
# ════════════════════════════════════════════════════════════════════════════

class TestVaderStopwords:
    def test_removes_stopwords(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["the dog is running in the park"])
        clean = df.iloc[0]["clean_text"]
        assert "the" not in clean.split()
        assert "is" not in clean.split()
        assert "in" not in clean.split()

    def test_retains_negation_tokens(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["this is not good at all"])
        clean = df.iloc[0]["clean_text"]
        assert "not" in clean.split()


class TestVaderLemmatization:
    def test_lemmatizes_plural_nouns(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["the dogs are running fast"])
        clean = df.iloc[0]["clean_text"]
        assert "dog" in clean.split()
        assert "dogs" not in clean.split()

    def test_lemmatizes_verbs(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["he was running and eating"])
        clean = df.iloc[0]["clean_text"]
        assert "run" in clean.split()
        assert "eat" in clean.split()


class TestVaderNegPrefix:
    def test_neg_prefix_single_negation(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["not bad at all"])
        clean = df.iloc[0]["clean_text"]
        assert "NEG_bad" in clean.split()
        assert "not" in clean.split()

    def test_neg_prefix_three_tokens(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["not good service staff"])
        clean = df.iloc[0]["clean_text"]
        tokens = clean.split()
        assert tokens[0] == "not"
        assert tokens[1] == "NEG_good"
        assert tokens[2] == "NEG_service"
        assert tokens[3] == "NEG_staff"

    def test_no_neg_prefix_without_negation(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["great hotel amazing staff"])
        clean = df.iloc[0]["clean_text"]
        assert "NEG_" not in clean


# ════════════════════════════════════════════════════════════════════════════
# Pysentimiento path (es, fr, pt, de)
# ════════════════════════════════════════════════════════════════════════════

class TestPysentimientoStopwords:
    @pytest.mark.parametrize("lang,text", [
        ("es", "el hotel y la comida"),
        ("fr", "le service et la qualité"),
        ("pt", "o quarto e a localização"),
        ("de", "das zimmer und das essen"),
    ])
    def test_stopwords_not_removed(self, lang: str, text: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, [text])
        for token in text.split():
            assert token in df.iloc[0]["clean_text"], (
                f"'{token}' was removed for lang '{lang}'"
            )


class TestPysentimientoNoLemmatization:
    @pytest.mark.parametrize("lang,inflected", [
        ("es", "estuvieron"),
        ("fr", "étaient"),
        ("pt", "estiveram"),
        ("de", "gelaufen"),
    ])
    def test_inflected_words_preserved(self, lang: str, inflected: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, [inflected])
        assert inflected in df.iloc[0]["clean_text"], (
            f"'{inflected}' was lemmatized for lang '{lang}'"
        )


class TestPysentimientoNoNegPrefix:
    @pytest.mark.parametrize("lang,text", [
        ("es", "no estuvo mal"),
        ("fr", "pas mal du tout"),
        ("pt", "não foi ruim"),
        ("de", "nicht schlecht"),
    ])
    def test_no_neg_prefix_applied(self, lang: str, text: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, [text])
        assert "NEG_" not in df.iloc[0]["clean_text"], (
            f"NEG_ prefix incorrectly applied for lang '{lang}'"
        )


# ════════════════════════════════════════════════════════════════════════════
# Output structure
# ════════════════════════════════════════════════════════════════════════════

class TestOutputStructure:
    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_clean_text_column_present(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["hello world"])
        assert "clean_text" in df.columns

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_tokens_column_present(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["hello world"])
        assert "tokens" in df.columns

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de", "pt"])
    def test_tokens_list_matches_clean_text(self, lang: str) -> None:
        prep = Preprocessor(LanguageConfig(lang))
        df = _run(prep, ["hello world"])
        assert df.iloc[0]["tokens"] == ["hello", "world"]

    def test_raw_text_preserved(self, en_preprocessor: Preprocessor) -> None:
        df = _run(en_preprocessor, ["original text"])
        assert "raw_text" in df.columns
        assert df.iloc[0]["raw_text"] == "original text"
