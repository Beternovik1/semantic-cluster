"""Word cloud images and interactive Plotly n-gram charts.

Generates fully offline, self-contained visual assets for outlier
analysis and sentiment partition summaries.  Returns in-memory
base64-encoded images and Plotly div strings while also saving
files to disk.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS

from src.utils.validators import MIN_PARTITION_SIZE

logger = logging.getLogger(__name__)

NGRAM_TOP_N = 10


def _empty_image_b64() -> str:
    """Return a small transparent PNG as a base64 data URI."""
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.axis("off")
    fig.savefig(buf, format="png", dpi=50, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _empty_chart_html(message: str) -> str:
    """Return a minimal Plotly div with a text annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig.to_html(
        include_plotlyjs=False, full_html=False
    )


class WordCloudGenerator:
    """Generate word clouds and n-gram bar charts for the pipeline.

    Args:
        palette_manager: A ``PaletteManager`` instance providing
            colours, background, and plotly template.
        lang_cfg: A ``LanguageConfig`` instance whose ``stopwords_name``
            is used to filter out stopwords at the visualisation layer.
    """

    def __init__(self, palette_manager, lang_cfg) -> None:
        self._pm = palette_manager
        self._lang_cfg = lang_cfg

    # ── Public API ─────────────────────────────────────────────────

    def from_outliers(self, df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
        """Generate word cloud + unigram/bigram/trigram charts for outliers.

        Args:
            df: Outlier DataFrame with a ``clean_text`` column.
            output_dir: Directory to save image and HTML files.

        Returns:
            Dict with keys ``wordcloud_b64``, ``unigrams_html``,
            ``bigrams_html``, ``trigrams_html``.
        """
        if df.empty:
            logger.warning("No outliers detected. Returning empty placeholders.")
            return {
                "wordcloud_b64": _empty_image_b64(),
                "unigrams_html": _empty_chart_html("No outliers detected"),
                "bigrams_html": _empty_chart_html("No outliers detected"),
                "trigrams_html": _empty_chart_html("No outliers detected"),
            }

        stopwords = self._load_stopwords()
        text = " ".join(df["clean_text"].dropna().tolist())

        wc_b64 = self._generate_wordcloud(text, stopwords, output_dir, "outliers")

        unigrams_html = self._generate_ngram_chart(
            df, 1, stopwords, output_dir, "unigrams_outliers"
        )
        bigrams_html = self._generate_ngram_chart(
            df, 2, stopwords, output_dir, "bigrams_outliers"
        )
        trigrams_html = self._generate_ngram_chart(
            df, 3, stopwords, output_dir, "trigrams_outliers"
        )

        return {
            "wordcloud_b64": wc_b64,
            "unigrams_html": unigrams_html,
            "bigrams_html": bigrams_html,
            "trigrams_html": trigrams_html,
        }

    def from_partition(
        self, df: pd.DataFrame, label: str, output_dir: Path
    ) -> str:
        """Generate a word cloud for a sentiment partition (pos / neg).

        Args:
            df: Partition DataFrame with a ``clean_text`` column.
            label: ``"positive"`` or ``"negative"`` (used in filename).
            output_dir: Directory to save the PNG file.

        Returns:
            Base64 data URI string of the word cloud image.
        """
        if df.empty:
            return _empty_image_b64()

        stopwords = self._load_stopwords()
        text = " ".join(df["clean_text"].dropna().tolist())

        return self._generate_wordcloud(text, stopwords, output_dir, label)

    # ── Word cloud ────────────────────────────────────────────────

    def _generate_wordcloud(
        self,
        text: str,
        stopwords: set[str],
        output_dir: Path,
        suffix: str,
    ) -> str:
        """Build, save, and base64-encode a word cloud image."""
        wc_stop = stopwords | WC_STOPWORDS

        # Guard: all-stopword text produces no tokens
        tokens = [w for w in text.split() if w.lower() not in wc_stop]
        if not tokens:
            logger.warning(
                "No non-stopword tokens for '%s' word cloud. "
                "Returning empty image.",
                suffix,
            )
            return _empty_image_b64()

        wc = WordCloud(
            background_color=self._pm.get_background(),
            stopwords=wc_stop,
            width=800,
            height=400,
            random_state=42,
        ).generate(text)

        out_path = output_dir / f"wordcloud_{suffix}.png"
        wc.to_file(str(out_path))
        logger.info("Saved word cloud to %s", out_path)

        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("utf-8")

        return f"data:image/png;base64,{data}"

    # ── N-gram charts (Plotly) ─────────────────────────────────────

    def _generate_ngram_chart(
        self,
        df: pd.DataFrame,
        n: int,
        stopwords: set[str],
        output_dir: Path,
        suffix: str,
    ) -> str:
        """Build, save, and return a Plotly div for *n*-gram frequencies."""
        stop_list = list(stopwords) if stopwords else None

        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            stop_words=stop_list,
            max_features=NGRAM_TOP_N,
            min_df=2,
            max_df=0.9,
        )

        try:
            matrix = vectorizer.fit_transform(df["clean_text"].dropna())
        except ValueError:
            return _empty_chart_html(f"No {n}-grams available")

        frequencies = matrix.sum(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()
        sorted_idx = frequencies.argsort()[::-1]
        top_features = feature_names[sorted_idx][:NGRAM_TOP_N]
        top_frequencies = frequencies[sorted_idx][:NGRAM_TOP_N]

        if len(top_features) == 0:
            return _empty_chart_html(f"No {n}-grams available")

        fig = px.bar(
            x=top_features,
            y=top_frequencies,
            labels={"x": "N-gram", "y": "Frequency"},
            title=f"Top {n}-grams in Outliers",
            template=self._pm.get_plotly_template(),
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Frequency: %{y}<extra></extra>"
        )

        # Save standalone file (embedded Plotly JS)
        html_standalone = fig.to_html(include_plotlyjs=True, full_html=True)
        out_path = output_dir / f"ngrams_{suffix}.html"
        out_path.write_text(html_standalone, encoding="utf-8")
        logger.info("Saved n-gram chart to %s", out_path)

        # Return in-memory div (no Plotly JS — report_builder injects once)
        return fig.to_html(include_plotlyjs=False, full_html=False)

    # ── Stopwords ─────────────────────────────────────────────────

    def _load_stopwords(self) -> set[str]:
        """Return NLTK stopwords for the configured language, or empty set."""
        try:
            from nltk.corpus import stopwords

            return set(stopwords.words(self._lang_cfg.stopwords_name))
        except Exception:
            logger.debug(
                "Could not load NLTK stopwords for '%s'. Using empty set.",
                self._lang_cfg.stopwords_name,
            )
            return set()
