"""Self-contained offline HTML report via Jinja2 and Bootstrap 5.

Assembles all pipeline outputs — word clouds, n-gram charts, Bokeh
scatter plots, and metric summaries — into a single ``report.html``
with all external resources (Bootstrap CSS, Plotly JS) embedded inline.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Template

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


# ── Template ─────────────────────────────────────────────────────────────────

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ data.title }} — TextLens Report</title>
<style>{{ bootstrap_css }}</style>
<style>
  body { padding-top: 1.5rem; padding-bottom: 2rem; }
  .section-title { margin-top: 2rem; margin-bottom: 1rem; }
  .summary-card { margin-bottom: 1rem; }
  iframe.bokeh-frame { width: 100%; height: 800px; border: none; }
  .top5-table td, .top5-table th { vertical-align: middle; }
  footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 0.9rem; }
  .section-title h2 { border-bottom: 3px solid {{ data.accent_color }}; padding-bottom: 0.5rem; }
  .card.summary-card { border-top: 3px solid {{ data.accent_color }}; }
</style>
</head>
<body>
<div class="container">

  <!-- 1. Header -->
  <div class="row">
    <div class="col-12">
      <h1 class="display-5">{{ data.title }}</h1>
      <p class="text-muted">Generated {{ data.timestamp }}</p>
      {% if data.cli_params %}
      <details>
        <summary class="text-muted" style="cursor:pointer;">Technical Configuration</summary>
        <table class="table table-sm table-bordered" style="max-width: 600px; margin-top: 0.5rem;">
          <caption class="visually-hidden">CLI Parameters</caption>
          <tbody>
            {% for key, value in data.cli_params.items() %}
            <tr><th scope="row" class="w-25">{{ key }}</th><td>{{ value }}</td></tr>
            {% endfor %}
          </tbody>
        </table>
      </details>
      {% endif %}
    </div>
  </div>

  <!-- 2. Executive Summary -->
  <div class="row section-title">
    <div class="col-12"><h2>Executive Summary</h2></div>
  </div>
  <p class="text-muted">High-level overview of the key findings from the analysis.</p>
  <div class="row">
    <div class="col-12">
      <ul>
        <li>Analyzed <strong>{{ data.total_rows }}</strong> comments in total.</li>
        <li>Sentiment split: <strong>{{ "%.1f"|format(data.pos_pct * 100) }}% positive</strong> ({{ data.pos_count }}), <strong>{{ "%.1f"|format(data.neg_pct * 100) }}% negative</strong> ({{ data.neg_count }}).</li>
        <li>Detected <strong>{{ data.outliers_count }}</strong> outlier(s) — removed before downstream analysis.</li>
        {% if data.top_positive_topic %}
        <li>Top positive topic: <strong>{{ data.top_positive_topic }}</strong>.</li>
        {% endif %}
        {% if data.top_negative_topic %}
        <li>Top negative topic: <strong>{{ data.top_negative_topic }}</strong>.</li>
        {% endif %}
        {% if data.top_concept_score > 0 %}
        <li>Highest semantic similarity to the target concept: <strong>{{ "%.3f"|format(data.top_concept_score) }}</strong>.</li>
        {% endif %}
        <li>Confidence rate: <strong>{{ "%.1f"|format(data.confidence_rate * 100) }}%</strong>.</li>
      </ul>
    </div>
  </div>

  <!-- 3. Pipeline Summary -->
  <div class="row section-title">
    <div class="col-12"><h2>Pipeline Summary</h2></div>
  </div>
  <p class="text-muted">Overview of the dataset after preprocessing, outlier removal, and sentiment classification.</p>
  <div class="row">
    <div class="col-md-3">
      <div class="card summary-card text-center">
        <div class="card-body"><h5 class="card-title">{{ data.total_rows }}</h5><small class="text-muted">Total Comments</small></div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card summary-card text-center">
        <div class="card-body"><h5 class="card-title">{{ data.outliers_count }}</h5><small class="text-muted">Outliers Removed</small></div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card summary-card text-center">
        <div class="card-body"><h5 class="card-title">{{ data.pos_count }}</h5><small class="text-muted">Positive</small></div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card summary-card text-center">
        <div class="card-body"><h5 class="card-title">{{ data.neg_count }}</h5><small class="text-muted">Negative</small></div>
      </div>
    </div>
  </div>
  <p>Confidence rate: <strong>{{ "%.1f"|format(data.confidence_rate * 100) }}%</strong></p>

  <!-- 4. Outlier Analysis -->
  <div class="row section-title">
    <div class="col-12"><h2>Outlier Analysis</h2></div>
  </div>
  <p class="text-muted">Comments flagged as statistically unusual are shown below. The word cloud highlights frequent terms; n-gram charts reveal multi-word patterns among outliers.</p>
  <div class="row">
    <div class="col-md-6"><img src="{{ data.wc_outliers_b64 }}" class="img-fluid" alt="Outliers Word Cloud"></div>
    <div class="col-md-6">
      <h5>Unigrams</h5>
      {{ data.ngrams_unigrams_html | safe }}
      <h5>Bigrams</h5>
      {{ data.ngrams_bigrams_html | safe }}
      <h5>Trigrams</h5>
      {{ data.ngrams_trigrams_html | safe }}
    </div>
  </div>

  <!-- 5. Sentiment Analysis Summary -->
  <div class="row section-title">
    <div class="col-12"><h2>Sentiment Analysis</h2></div>
  </div>
  <p class="text-muted">Each comment was classified as positive or negative. The word clouds below show the most frequent terms for each sentiment group.</p>
  <div class="row">
    <div class="col-md-6"><img src="{{ data.wc_positive_b64 }}" class="img-fluid" alt="Positive Word Cloud"></div>
    <div class="col-md-6"><img src="{{ data.wc_negative_b64 }}" class="img-fluid" alt="Negative Word Cloud"></div>
  </div>

  <!-- 6. Topic Modeling Overview -->
  <div class="row section-title">
    <div class="col-12"><h2>Topic Modeling Overview</h2></div>
  </div>
  <p class="text-muted">Topics are automatically extracted from positive and negative comments. Each point represents a comment; color and shape encode its topic cluster. Hover over a point to see the comment text.</p>
  <div class="row">
    <div class="col-12">
      {% if data.scatter_topics_html %}
      <p class="text-muted">Both positive and negative comments are shown on the same plot. Each point is a comment, colored by its topic cluster. Hover over a point to see the comment text and sentiment.</p>
      <iframe srcdoc="{{ data.scatter_topics_html | e }}" class="bokeh-frame"></iframe>
      {% else %}
      <div class="alert alert-info" role="alert">
        Fewer than {{ data.min_partition_size }} comments in one or both partitions. Word frequency shown instead of topic modeling.
      </div>
      {% if data.pos_fallback_keywords %}
      <p><strong>Positive — Top keywords:</strong> {{ data.pos_fallback_keywords }}</p>
      {% endif %}
      {% if data.neg_fallback_keywords %}
      <p><strong>Negative — Top keywords:</strong> {{ data.neg_fallback_keywords }}</p>
      {% endif %}
      {% endif %}
    </div>
  </div>

  <!-- 7. Semantic Concept Analysis -->
  <div class="row section-title">
    <div class="col-12"><h2>Semantic Concept Analysis</h2></div>
  </div>
  <p class="text-muted">Each comment is scored by cosine similarity to a target concept. Brighter / higher points are more semantically related. The table below lists the five most relevant comments.</p>
  <div class="row">
    <div class="col-12">
      {% if data.scatter_semantic_html %}
      <iframe srcdoc="{{ data.scatter_semantic_html | e }}" class="bokeh-frame"></iframe>
      {% endif %}
    </div>
  </div>
  {% if data.top5_semantic %}
  <div class="row">
    <div class="col-12">
      <h5>Top 5 Most Similar Comments</h5>
      <table class="table table-striped top5-table">
        <caption>Top 5 comments by concept similarity</caption>
        <thead><tr><th>#</th><th>Comment</th><th>Similarity</th><th>Sentiment</th></tr></thead>
        <tbody>
          {% for row in data.top5_semantic %}
          <tr><td>{{ loop.index }}</td><td>{{ row.text }}</td><td>{{ "%.3f"|format(row.similarity) }}</td><td>{{ row.sentiment }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  {% endif %}

  <!-- 8. Footer -->
  <footer>
    <p>Generated by <strong>TextLens</strong> &mdash; Fully self-contained offline report.</p>
  </footer>

</div>
<script>{{ plotly_js }}</script>
</body>
</html>"""


# ── Data container ───────────────────────────────────────────────────────────


@dataclass
class ReportData:
    """All data required by the Jinja2 template."""

    title: str = ""
    timestamp: str = ""
    cli_params: dict[str, Any] = field(default_factory=dict)

    total_rows: int = 0
    outliers_count: int = 0
    clean_count: int = 0
    pos_count: int = 0
    neg_count: int = 0
    confidence_rate: float = 0.0
    min_partition_size: int = 100

    scatter_topics_html: str | None = None
    scatter_semantic_html: str | None = None
    wc_outliers_b64: str = ""
    wc_positive_b64: str = ""
    wc_negative_b64: str = ""
    ngrams_unigrams_html: str = ""
    ngrams_bigrams_html: str = ""
    ngrams_trigrams_html: str = ""

    pos_fallback_keywords: str | None = None
    neg_fallback_keywords: str | None = None
    top5_semantic: list[dict[str, Any]] = field(default_factory=list)

    pos_pct: float = 0.0
    neg_pct: float = 0.0
    top_positive_topic: str = ""
    top_negative_topic: str = ""
    top_concept_score: float = 0.0

    palette_name: str = "viridis"
    accent_color: str = "#0d6efd"
    is_dark_bg: bool = False


# ── Report builder ───────────────────────────────────────────────────────────


class ReportBuilder:
    """Assemble all pipeline outputs into a single offline HTML report.

    Args:
        title: Report title displayed in the header.
        palette: A ``PaletteManager`` instance (used for metadata).
    """

    def __init__(self, title: str, palette) -> None:
        self._title = title
        self._palette = palette

    def build(
        self,
        df=None,
        outliers_df=None,
        scatter_topics_html: str | None = None,
        scatter_semantic_html: str | None = None,
        wc_outliers_b64: str = "",
        wc_positive_b64: str = "",
        wc_negative_b64: str = "",
        ngrams_unigrams_html: str = "",
        ngrams_bigrams_html: str = "",
        ngrams_trigrams_html: str = "",
        pos_fallback_keywords: str | None = None,
        neg_fallback_keywords: str | None = None,
        pos_topic_keywords: dict[int, str] | None = None,
        neg_topic_keywords: dict[int, str] | None = None,
        cli_params: dict[str, Any] | None = None,
        output_dir: Path | str = Path("outputs"),
    ) -> Path:
        """Render and save the complete HTML report.

        Args:
            df: Full clean DataFrame (must contain ``sentiment``,
                ``concept_similarity``, and ``clean_text`` columns).
            outliers_df: Outlier DataFrame.
            scatter_topics_html: Bokeh topic scatter HTML or ``None``.
            scatter_semantic_html: Bokeh semantic scatter HTML.
            wc_outliers_b64: Base64 word cloud for outliers.
            wc_positive_b64: Base64 word cloud for positive partition.
            wc_negative_b64: Base64 word cloud for negative partition.
            ngrams_unigrams_html: Plotly div for unigrams.
            ngrams_bigrams_html: Plotly div for bigrams.
            ngrams_trigrams_html: Plotly div for trigrams.
            pos_fallback_keywords: Fallback keywords for positive
                partition, or ``None`` if BERTopic was used.
            neg_fallback_keywords: Fallback keywords for negative
                partition, or ``None`` if BERTopic was used.
            pos_topic_keywords: Full topic keywords dict for positive
                partition, or ``None``.
            neg_topic_keywords: Full topic keywords dict for negative
                partition, or ``None``.
            cli_params: CLI arguments as a dict for the technical
                configuration table, or ``None``.
            output_dir: Directory to save ``report.html``.

        Returns:
            Path to the generated ``report.html``.
        """
        data = self._build_data(
            df=df,
            outliers_df=outliers_df,
            scatter_topics_html=scatter_topics_html,
            scatter_semantic_html=scatter_semantic_html,
            wc_outliers_b64=wc_outliers_b64,
            wc_positive_b64=wc_positive_b64,
            wc_negative_b64=wc_negative_b64,
            ngrams_unigrams_html=ngrams_unigrams_html,
            ngrams_bigrams_html=ngrams_bigrams_html,
            ngrams_trigrams_html=ngrams_trigrams_html,
            pos_fallback_keywords=pos_fallback_keywords,
            neg_fallback_keywords=neg_fallback_keywords,
            pos_topic_keywords=pos_topic_keywords,
            neg_topic_keywords=neg_topic_keywords,
            cli_params=cli_params,
        )

        bootstrap_css = self._read_asset(ASSETS_DIR / "bootstrap.min.css")
        plotly_js = self._read_asset(ASSETS_DIR / "plotly.min.js")

        template = Template(_REPORT_TEMPLATE)
        html = template.render(
            data=data,
            bootstrap_css=bootstrap_css,
            plotly_js=plotly_js,
        )

        out_path = Path(output_dir) / "report.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Report saved to %s (%.1f MB). "
            "First browser render may take 2-3 seconds — "
            "expected behavior for a fully self-contained offline report.",
            out_path,
            size_mb,
        )

        return out_path

    # ── Data assembly ──────────────────────────────────────────────

    def _build_data(
        self,
        df=None,
        outliers_df=None,
        scatter_topics_html: str | None = None,
        scatter_semantic_html: str | None = None,
        wc_outliers_b64: str = "",
        wc_positive_b64: str = "",
        wc_negative_b64: str = "",
        ngrams_unigrams_html: str = "",
        ngrams_bigrams_html: str = "",
        ngrams_trigrams_html: str = "",
        pos_fallback_keywords: str | None = None,
        neg_fallback_keywords: str | None = None,
        pos_topic_keywords: dict[int, str] | None = None,
        neg_topic_keywords: dict[int, str] | None = None,
        cli_params: dict[str, Any] | None = None,
    ) -> ReportData:
        """Populate a ``ReportData`` instance from pipeline outputs."""
        data = ReportData(
            title=self._title,
            timestamp=datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            scatter_topics_html=scatter_topics_html,
            scatter_semantic_html=scatter_semantic_html,
            wc_outliers_b64=wc_outliers_b64,
            wc_positive_b64=wc_positive_b64,
            wc_negative_b64=wc_negative_b64,
            ngrams_unigrams_html=ngrams_unigrams_html,
            ngrams_bigrams_html=ngrams_bigrams_html,
            ngrams_trigrams_html=ngrams_trigrams_html,
            pos_fallback_keywords=pos_fallback_keywords,
            neg_fallback_keywords=neg_fallback_keywords,
            palette_name=self._palette.name,
            accent_color=self._palette.get_accent_color(),
            is_dark_bg=self._palette.is_dark,
        )

        if cli_params:
            data.cli_params = cli_params

        if df is not None:
            data.total_rows = len(df)
            n = len(df)
            data.pos_count = int((df["sentiment"] == "pos").sum())
            data.neg_count = int((df["sentiment"] == "neg").sum())
            data.pos_pct = data.pos_count / n if n else 0.0
            data.neg_pct = data.neg_count / n if n else 0.0
            data.confidence_rate = float(
                df["sentiment_confidence"].mean()
            ) if "sentiment_confidence" in df.columns else 0.0

            if "concept_similarity" in df.columns:
                data.top_concept_score = float(
                    df["concept_similarity"].max()
                )
                top5 = (
                    df.nlargest(5, "concept_similarity")[
                        ["clean_text", "concept_similarity", "sentiment"]
                    ]
                )
                data.top5_semantic = [
                    {
                        "text": row["clean_text"][:200],
                        "similarity": row["concept_similarity"],
                        "sentiment": row["sentiment"],
                    }
                    for _, row in top5.iterrows()
                ]

        if outliers_df is not None:
            data.outliers_count = len(outliers_df)
            data.clean_count = max(
                0, data.total_rows - data.outliers_count
            )

        # Topic keywords for executive summary
        if pos_topic_keywords:
            non_noise = {
                k: v for k, v in pos_topic_keywords.items() if k != -1 and v
            }
            if non_noise:
                from src.pipeline.topic_modeler import _readable_topic_name
                first_kw = next(iter(non_noise.values()))
                data.top_positive_topic = _readable_topic_name(first_kw)
        if neg_topic_keywords:
            non_noise = {
                k: v for k, v in neg_topic_keywords.items() if k != -1 and v
            }
            if non_noise:
                from src.pipeline.topic_modeler import _readable_topic_name
                first_kw = next(iter(non_noise.values()))
                data.top_negative_topic = _readable_topic_name(first_kw)

        return data

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _read_asset(path: Path) -> str:
        """Read an asset file, returning empty string on failure."""
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("Asset not found: %s", path)
            return ""
        except OSError:
            logger.warning("Could not read asset: %s", path)
            return ""
