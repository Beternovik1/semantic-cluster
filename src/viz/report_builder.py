"""Self-contained offline HTML report via Jinja2 and Bootstrap 5.

Assembles all pipeline outputs — word clouds, n-gram charts, Bokeh
scatter plots, and metric summaries — into a single ``report.html``
with all external resources (Bootstrap CSS, Plotly JS) embedded inline.

Visual style mirrors the Falcon dashboard design system.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


# ── Falcon dashboard CSS (design tokens + component styles) ──────────────────

_FALCON_CSS = """
  /* ── DESIGN TOKENS ── */
  :root {
    --sidebar-width: 260px;
    --primary: #2c7be5;
    --primary-light: #e8f1fd;
    --accent: #27bcfd;
    --success: #00d27a;
    --warning: #f5803e;
    --danger: #e63757;
    --text-dark: #344050;
    --text-muted: #748194;
    --bg-light: #edf2f9;
    --card-bg: #ffffff;
    --sidebar-bg: #ffffff;
    --border: #d8e2ef;
    --font-body: 'Segoe UI', system-ui, sans-serif;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow-hover: 0 4px 20px rgba(44,123,229,.1);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: var(--font-body);
    background: var(--bg-light);
    color: var(--text-dark);
    padding-bottom: 3rem;
  }

  /* ── HERO / HEADER ── */
  .rpt-hero {
    background: linear-gradient(135deg, var(--primary) 0%, #1a56c4 100%);
    color: #fff;
    padding: 2rem 0 1.75rem;
    margin-bottom: 2rem;
  }
  .rpt-hero h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 0.2rem; }
  .rpt-hero .subtitle { opacity: .85; font-size: .9rem; }
  .rpt-hero .meta { opacity: .65; font-size: .78rem; margin-top: .4rem; }
  .rpt-hero .badge-palette {
    display: inline-block;
    background: rgba(255,255,255,.2);
    padding: .15rem .6rem;
    border-radius: 20px;
    font-size: .72rem; font-weight: 500;
  }

  /* ── SECTION TITLES ── */
  .rpt-section { margin: 2rem 0 .75rem; }
  .rpt-section h2 {
    font-size: 1.1rem; font-weight: 700;
    color: var(--text-dark);
    border-bottom: 3px solid var(--primary);
    padding-bottom: .4rem;
    display: inline-block;
  }
  .rpt-section-desc {
    font-size: .85rem; color: var(--text-muted);
    margin-bottom: 1.25rem;
  }

  /* ── STAT / KPI CARDS ── */
  .rpt-stat {
    background: var(--card-bg);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    padding: 1.25rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: box-shadow .2s;
  }
  .rpt-stat:hover { box-shadow: var(--shadow-hover); }
  .rpt-stat .stat-value { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
  .rpt-stat .stat-label {
    font-size: .72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em;
    color: var(--text-muted); margin-top: .2rem;
  }
  .rpt-stat.pos { border-top-color: var(--success); }
  .rpt-stat.pos .stat-value { color: var(--success); }
  .rpt-stat.neg { border-top-color: var(--danger); }
  .rpt-stat.neg .stat-value { color: var(--danger); }
  .rpt-stat.warn { border-top-color: var(--warning); }
  .rpt-stat.warn .stat-value { color: var(--warning); }

  /* ── GENERIC CONTENT CARD ── */
  .rpt-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
  }
  .rpt-card-title {
    font-size: .9rem; font-weight: 700;
    color: var(--text-dark); margin-bottom: .75rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid var(--border);
  }

  /* ── BADGE CHANGES ── */
  .badge-up   { background: #d0f4e9; color: var(--success); }
  .badge-down { background: #fde8ec; color: var(--danger); }
  .badge-neutral { background: var(--primary-light); color: var(--primary); }
  .badge-change {
    display: inline-flex; align-items: center; gap: 3px;
    font-size: .72rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
  }

  /* ── SENTIMENT BADGES ── */
  .badge-pos { display:inline-block; background:#dcfce7; color:#166534; padding:.1rem .5rem; border-radius:20px; font-size:.72rem; font-weight:500; }
  .badge-neg { display:inline-block; background:#fee2e2; color:#991b1b; padding:.1rem .5rem; border-radius:20px; font-size:.72rem; font-weight:500; }

  /* ── EXECUTIVE SUMMARY LIST ── */
  .rpt-summary-list { list-style: none; padding: 0; }
  .rpt-summary-list li {
    padding: .4rem 0 .4rem 1.2rem;
    position: relative;
    font-size: .88rem;
    border-bottom: 1px solid var(--border);
    color: var(--text-dark);
  }
  .rpt-summary-list li:last-child { border-bottom: none; }
  .rpt-summary-list li::before {
    content: "";
    position: absolute; left: 0; top: 50%;
    transform: translateY(-50%);
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--primary);
  }

  /* ── SCATTER IFRAMES ── */
  iframe.rpt-scatter {
    width: 100%; height: 750px;
    border: none; border-radius: var(--radius);
  }

  /* ── TOPIC CARDS ── */
  .rpt-topic {
    border-left: 4px solid var(--primary);
    padding: .85rem 1rem;
    margin-bottom: .65rem;
    border-radius: 0 var(--radius) var(--radius) 0;
    background: var(--bg-light);
  }
  .rpt-topic h6 { font-weight: 700; font-size: .85rem; margin-bottom: .15rem; }
  .rpt-topic .topic-kw { font-size: .78rem; color: var(--text-muted); }
  .rpt-topic blockquote {
    font-size: .82rem; font-style: italic;
    color: #475569;
    border-left: 2px solid var(--border);
    padding-left: .65rem;
    margin: .35rem 0 0;
  }

  /* ── SIMILARITY BAR ── */
  .sim-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; min-width: 60px; }
  .sim-bar-fill { height: 100%; background: var(--primary); border-radius: 3px; }

  /* ── WORD CLOUDS ── */
  .rpt-wc img {
    width: 100%; border-radius: var(--radius);
    border: 1px solid var(--border);
  }
  .rpt-wc .wc-label {
    font-size: .72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .05em;
    color: var(--text-muted); margin-bottom: .4rem;
  }

  /* ── NGRAM GRID ── */
  .rpt-ngram-grid { display: flex; gap: 1rem; flex-wrap: wrap; }
  .rpt-ngram-grid > div { flex: 1; min-width: 180px; }
  .rpt-ngram-grid h6 { font-size: .8rem; font-weight: 700; color: var(--text-dark); margin-bottom: .35rem; }

  /* ── STORAGE STRIP ── */
  .storage-strip { height: 8px; border-radius: 6px; overflow: hidden; display: flex; gap: 2px; }
  .stor-seg { height: 100%; border-radius: 3px; }

  /* ── ALERT CARD ── */
  .rpt-alert-warn {
    background: linear-gradient(135deg, #fff5ef, #fff);
    border: 1px solid #ffd4b7;
    border-radius: var(--radius);
    padding: 1.25rem;
  }
  .rpt-alert-warn .alert-title { color: var(--warning); font-weight: 700; font-size: .9rem; }
  .rpt-alert-warn p { font-size: .82rem; color: var(--text-muted); margin: .5rem 0 .75rem; }

  /* ── TABLES ── */
  .rpt-table { width: 100%; border-collapse: collapse; font-size: .84rem; }
  .rpt-table thead tr { border-bottom: 2px solid var(--primary-light); }
  .rpt-table th { font-weight: 700; color: var(--text-muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; padding: .5rem .75rem; }
  .rpt-table td { padding: .6rem .75rem; border-bottom: 1px solid var(--border); vertical-align: middle; }
  .rpt-table tbody tr:last-child td { border-bottom: none; }
  .rpt-table tbody tr:hover td { background: var(--primary-light); }

  /* ── CLI PARAMS TABLE ── */
  .rpt-cli-table { font-size: .78rem; max-width: 560px; margin-top: .5rem; }
  .rpt-cli-table tr td:first-child { font-weight: 600; padding-right: 1rem; color: var(--text-muted); white-space: nowrap; }

  /* ── FOOTER ── */
  .rpt-footer {
    margin-top: 3rem; padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    font-size: .78rem; color: var(--text-muted);
    text-align: center;
  }

  /* ── UTILITIES ── */
  .text-primary-falcon { color: var(--primary) !important; }
  .text-success-falcon { color: var(--success) !important; }
  .text-danger-falcon  { color: var(--danger)  !important; }
  .text-warn-falcon    { color: var(--warning) !important; }
  .text-muted-falcon   { color: var(--text-muted) !important; }

  @media (max-width: 768px) {
    .rpt-hero h1 { font-size: 1.35rem; }
    .rpt-stat .stat-value { font-size: 1.25rem; }
    iframe.rpt-scatter { height: 480px; }
  }
"""


# ── Template ─────────────────────────────────────────────────────────────────

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ data.title }} — TextLens Report</title>
<style>{{ bootstrap_css }}</style>
<style>{{ falcon_css }}</style>
<script>{{ plotly_js }}</script>
</head>
<body>

<!-- ═══ HERO ═══ -->
<div class="rpt-hero">
  <div class="container">
    <div class="d-flex justify-content-between align-items-start flex-wrap gap-2">
      <div>
        <h1>{{ data.title }}</h1>
        <div class="subtitle">Automated NLP Analysis Report</div>
        <div class="meta">{{ data.timestamp }}</div>
      </div>
    </div>
    {% if data.cli_params %}
    <details style="margin-top:.75rem;opacity:.75;">
      <summary style="cursor:pointer;font-size:.8rem;">Technical Configuration</summary>
      <table class="rpt-cli-table" style="margin-top:.4rem;">
        {% for key, value in data.cli_params.items() %}
        <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
        {% endfor %}
      </table>
    </details>
    {% endif %}
  </div>
</div>

<div class="container">

  <!-- ── 1. OVERVIEW KPI CARDS ── -->
  <div class="rpt-section"><h2>Overview</h2></div>
  <p class="rpt-section-desc">Key metrics from the analysis of {{ data.total_rows }} comments.</p>
  <div class="row g-3 mb-4">
    <div class="col-6 col-md-3">
      <div class="rpt-stat"><div class="stat-value">{{ data.total_rows }}</div><div class="stat-label">Total Comments</div></div>
    </div>
    <div class="col-6 col-md-3">
      <div class="rpt-stat pos">
        <div class="stat-value">{{ "%.1f"|format(data.pos_pct * 100) }}%</div>
        <div class="stat-label">Positive ({{ data.pos_count }})</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="rpt-stat neg">
        <div class="stat-value">{{ "%.1f"|format(data.neg_pct * 100) }}%</div>
        <div class="stat-label">Negative ({{ data.neg_count }})</div>
      </div>
    </div>
    <div class="col-6 col-md-3">
      <div class="rpt-stat warn">
        <div class="stat-value">{{ data.outliers_count }}</div>
        <div class="stat-label">Outliers Removed</div>
      </div>
    </div>
  </div>

  <!-- ── 2. EXECUTIVE SUMMARY ── -->
  <div class="rpt-section"><h2>Executive Summary</h2></div>
  <p class="rpt-section-desc">High-level overview of the key findings from the analysis.</p>
  <div class="rpt-card">
    <ul class="rpt-summary-list">
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

  <!-- ── 3. PIPELINE SUMMARY ── -->
  <div class="rpt-section"><h2>Pipeline Summary</h2></div>
  <p class="rpt-section-desc">Overview of the dataset after preprocessing, outlier removal, and sentiment classification.</p>
  <div class="row g-3 mb-3">
    <div class="col-6 col-md-3">
      <div class="rpt-stat"><div class="stat-value">{{ data.total_rows }}</div><div class="stat-label">Total Comments</div></div>
    </div>
    <div class="col-6 col-md-3">
      <div class="rpt-stat warn"><div class="stat-value">{{ data.outliers_count }}</div><div class="stat-label">Outliers Removed</div></div>
    </div>
    <div class="col-6 col-md-3">
      <div class="rpt-stat pos"><div class="stat-value">{{ data.pos_count }}</div><div class="stat-label">Positive</div></div>
    </div>
    <div class="col-6 col-md-3">
      <div class="rpt-stat neg"><div class="stat-value">{{ data.neg_count }}</div><div class="stat-label">Negative</div></div>
    </div>
  </div>
  <div class="rpt-card" style="padding:.9rem 1.25rem;">
    <span style="font-size:.85rem;">Confidence rate: <strong class="text-primary-falcon">{{ "%.1f"|format(data.confidence_rate * 100) }}%</strong></span>
  </div>

  <!-- ── 4. OUTLIER ANALYSIS ── -->
  <div class="rpt-section"><h2>Outlier Analysis</h2></div>
  <p class="rpt-section-desc">Comments flagged as statistically unusual are shown below. The word cloud highlights frequent terms; n-gram charts reveal multi-word patterns among outliers.</p>
  <div class="row g-3 mb-4">
    <div class="col-md-5">
      <div class="rpt-card rpt-wc h-100">
        <div class="wc-label">Word Cloud — Outliers</div>
        <img src="{{ data.wc_outliers_b64 }}" alt="Outliers Word Cloud">
      </div>
    </div>
    <div class="col-md-7">
      <div class="rpt-card h-100">
        <div class="rpt-card-title">N-gram Frequency</div>
        <div class="rpt-ngram-grid">
          <div><h6>Unigrams</h6>{{ data.ngrams_unigrams_html | safe }}</div>
          <div><h6>Bigrams</h6>{{ data.ngrams_bigrams_html | safe }}</div>
          <div><h6>Trigrams</h6>{{ data.ngrams_trigrams_html | safe }}</div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── 5. SENTIMENT ANALYSIS ── -->
  <div class="rpt-section"><h2>Sentiment Analysis</h2></div>
  <p class="rpt-section-desc">Each comment was classified as positive or negative. The word clouds below show the most frequent terms for each sentiment group.</p>
  <div class="row g-3 mb-4">
    {% if data.sentiment_pie_html %}
    <div class="col-md-5">
      <div class="rpt-card" style="text-align:center;">
        {{ data.sentiment_pie_html | safe }}
      </div>
    </div>
    <div class="col-md-7">
    {% else %}
    <div class="col-12">
    {% endif %}
      <div class="row g-2">
        <div class="col-6">
          <div class="rpt-card rpt-wc" style="padding:.75rem;">
            <div class="wc-label">Positive</div>
            <img src="{{ data.wc_positive_b64 }}" alt="Positive Word Cloud">
          </div>
        </div>
        <div class="col-6">
          <div class="rpt-card rpt-wc" style="padding:.75rem;">
            <div class="wc-label">Negative</div>
            <img src="{{ data.wc_negative_b64 }}" alt="Negative Word Cloud">
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ── 6. TOPIC MODELING ── -->
  <div class="rpt-section"><h2>Topic Modeling Overview</h2></div>
  <p class="rpt-section-desc">Topics are automatically extracted from positive and negative comments. Each point represents a comment; color and shape encode its topic cluster.</p>
  <div class="row g-3 mb-4">
    <div class="col-12">
      {% if data.scatter_topics_html %}
      <div class="rpt-card" style="padding:.75rem;">
        <iframe srcdoc="{{ data.scatter_topics_html | e }}" class="rpt-scatter"></iframe>
      </div>
      {% else %}
      <div class="rpt-card">
        <div class="alert alert-info mb-0" role="alert">
          Fewer than {{ data.min_partition_size }} comments in one or both partitions. Word frequency shown instead of topic modeling.
        </div>
        {% if data.pos_fallback_keywords %}
        <p style="margin-top:.75rem;font-size:.85rem;"><strong>Positive — Top keywords:</strong> {{ data.pos_fallback_keywords }}</p>
        {% endif %}
        {% if data.neg_fallback_keywords %}
        <p style="font-size:.85rem;"><strong>Negative — Top keywords:</strong> {{ data.neg_fallback_keywords }}</p>
        {% endif %}
      </div>
      {% endif %}
    </div>

    {% if data.pos_topic_summaries or data.neg_topic_summaries %}
    <div class="col-md-6">
      <div style="font-weight:700;font-size:.88rem;color:#16a34a;margin-bottom:.6rem;">Positive Topics</div>
      {% for t in data.pos_topic_summaries %}
      <div class="rpt-topic">
        <h6>{{ t.label }}</h6>
        <div class="topic-kw">{{ t.keywords }}</div>
        {% if t.representative %}<blockquote>&ldquo;{{ t.representative }}&rdquo;</blockquote>{% endif %}
        <small class="text-muted-falcon">{{ t.count }} comments</small>
      </div>
      {% endfor %}
    </div>
    <div class="col-md-6">
      <div style="font-weight:700;font-size:.88rem;color:#dc2626;margin-bottom:.6rem;">Negative Topics</div>
      {% for t in data.neg_topic_summaries %}
      <div class="rpt-topic" style="border-left-color:var(--danger);">
        <h6>{{ t.label }}</h6>
        <div class="topic-kw">{{ t.keywords }}</div>
        {% if t.representative %}<blockquote>&ldquo;{{ t.representative }}&rdquo;</blockquote>{% endif %}
        <small class="text-muted-falcon">{{ t.count }} comments</small>
      </div>
      {% endfor %}
    </div>
    {% endif %}
  </div>

  <!-- ── 7. SEMANTIC CONCEPT ANALYSIS ── -->
  <div class="rpt-section"><h2>Semantic Concept Analysis</h2></div>
  <p class="rpt-section-desc">Each comment is scored by cosine similarity to the target concept{% if data.concept_name %} <strong>&ldquo;{{ data.concept_name }}&rdquo;</strong>{% endif %}. Brighter / higher points are more semantically related.</p>
  <div class="row g-3 mb-4">
    {% if data.scatter_semantic_html %}
    <div class="col-12">
      <div class="rpt-card" style="padding:.75rem;">
        <iframe srcdoc="{{ data.scatter_semantic_html | e }}" class="rpt-scatter"></iframe>
      </div>
    </div>
    {% endif %}
    {% if data.top5_semantic %}
    <div class="col-12">
      <div class="rpt-card">
        <div class="rpt-card-title">Top 5 Most Similar Comments</div>
        <div class="table-responsive">
          <table class="rpt-table">
            <caption class="visually-hidden">Top 5 comments by concept similarity</caption>
            <thead>
              <tr>
                <th style="width:36px;">#</th>
                <th>Comment</th>
                <th style="width:130px;">Similarity</th>
                <th style="width:90px;">Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {% for row in data.top5_semantic %}
              <tr>
                <td>{{ loop.index }}</td>
                <td style="max-width:420px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="{{ row.text }}">{{ row.text }}</td>
                <td>
                  <div class="d-flex align-items-center gap-2">
                    <div class="sim-bar flex-grow-1">
                      <div class="sim-bar-fill" style="width:{{ "%.0f"|format(row.similarity * 100) }}%;"></div>
                    </div>
                    <span style="font-size:.82rem;font-weight:600;">{{ "%.3f"|format(row.similarity) }}</span>
                  </div>
                </td>
                <td>
                  <span class="{% if row.sentiment == 'pos' %}badge-pos{% else %}badge-neg{% endif %}">{{ row.sentiment }}</span>
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>
    {% endif %}
  </div>

  <!-- ── FOOTER ── -->
  <div class="rpt-footer">
    <p>Generated by <strong>TextLens</strong> &mdash; Fully self-contained offline report.</p>
  </div>

</div><!-- /container -->

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

    sentiment_pie_html: str = ""

    pos_topic_summaries: list[dict[str, Any]] = field(default_factory=list)
    neg_topic_summaries: list[dict[str, Any]] = field(default_factory=list)

    concept_name: str = ""


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
        concept_name: str = "",
        output_dir: Path | str = Path("outputs"),
        filename: str = "report.html",
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
            concept_name: The concept string used for semantic search.
            output_dir: Directory to save the report.
            filename: Output filename (default ``"report.html"``).

        Returns:
            Path to the generated report file.
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
            concept_name=concept_name,
        )

        if data.pos_count + data.neg_count > 0:
            data.sentiment_pie_html = self._make_sentiment_pie(
                data.pos_count, data.neg_count
            )

        bootstrap_css = self._read_asset(ASSETS_DIR / "bootstrap.min.css")
        plotly_js = self._read_asset(ASSETS_DIR / "plotly.min.js")

        template = Template(_REPORT_TEMPLATE)
        html = template.render(
            data=data,
            bootstrap_css=bootstrap_css,
            falcon_css=_FALCON_CSS,
            plotly_js=plotly_js,
        )

        out_path = Path(output_dir) / filename
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
        concept_name: str = "",
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
                    df.drop_duplicates("clean_text")
                    .nlargest(5, "concept_similarity")[
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

        if concept_name:
            data.concept_name = concept_name
        elif cli_params and "Concept" in cli_params:
            data.concept_name = cli_params["Concept"]

        # Per-topic summaries
        if df is not None and "topic_label" in df.columns and "representative_doc" in df.columns:
            for sentiment_val, target in [("pos", data.pos_topic_summaries), ("neg", data.neg_topic_summaries)]:
                mask = df["sentiment"] == sentiment_val
                subset = df[mask]
                if subset.empty:
                    continue
                for topic_label in subset["topic_label"].unique():
                    if pd.isna(topic_label) or topic_label == "word_frequency_fallback":
                        continue
                    group = subset[subset["topic_label"] == topic_label]
                    rep_rows = group[group["representative_doc"] == True]
                    rep_text = rep_rows.iloc[0]["clean_text"][:300] if not rep_rows.empty else ""
                    first = group.iloc[0]
                    target.append({
                        "label": topic_label,
                        "keywords": first.get("topic_keywords", ""),
                        "count": len(group),
                        "representative": rep_text,
                    })

        return data

    # ── Styled helpers ────────────────────────────────────────────

    def _make_sentiment_pie(self, pos: int, neg: int) -> str:
        import plotly.graph_objects as go
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Positive", "Negative"],
                    values=[pos, neg],
                    marker=dict(colors=["#00d27a", "#e63757"]),
                    textinfo="label+percent",
                    hole=0.4,
                )
            ]
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=0, b=0, l=0, r=0),
            height=260,
            template="plotly_white",
        )
        return fig.to_html(include_plotlyjs=False, full_html=False)

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