"""
LAST USED MINIMAL
Chart-style offline HTML report via Jinja2 with sidebar layout.

Shares the same data assembly as ``ReportBuilder`` but renders a
different template (sidebar + topbar layout).
All external resources (Bootstrap CSS, Bootstrap Icons, Plotly JS) are
embedded inline — fully self-contained, zero CDN dependencies.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from jinja2 import Template

from src.viz.report_builder import ReportBuilder

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"


class ReportChartBuilder:
    """Build a chart-style HTML report with sidebar dashboard template.

    Uses ``ReportBuilder._build_data()`` internally so the data assembly
    logic is identical — only the visual template differs.

    Args:
        title: Report title displayed in the header.
        palette: A ``PaletteManager`` instance (used for metadata).
    """

    def __init__(self, title: str, palette) -> None:
        self._title = title
        self._palette = palette
        self._builder = ReportBuilder(title, palette)

    # ── Public API ─────────────────────────────────────────────────

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
        """Render and save the chart-style HTML report.

        Args: (same as ``ReportBuilder.build()`` — no ``styled`` param)

        Returns:
            Path to the generated report file.
        """
        data = self._builder._build_data(
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
            data.sentiment_pie_html = self._builder._make_sentiment_pie(
                data.pos_count, data.neg_count
            )

        bootstrap_css = ReportBuilder._read_asset(
            ASSETS_DIR / "bootstrap.min.css"
        )
        plotly_js = ReportBuilder._read_asset(
            ASSETS_DIR / "plotly.min.js"
        )
        bootstrap_icons_css = self._build_bootstrap_icons_css()

        template = Template(_CHART_TEMPLATE)
        html = template.render(
            data=data,
            bootstrap_css=bootstrap_css,
            bootstrap_icons_css=bootstrap_icons_css,
            plotly_js=plotly_js,
        )

        out_path = Path(output_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Chart report saved to %s (%.1f MB). "
            "First browser render may take 2-3 seconds — "
            "expected behavior for a fully self-contained offline report.",
            out_path,
            size_mb,
        )
        return out_path

    def _build_bootstrap_icons_css(self) -> str:
        """Read Bootstrap Icons CSS and embed the woff2 font as base64."""
        css_path = ASSETS_DIR / "bootstrap-icons.min.css"
        woff2_path = ASSETS_DIR / "bootstrap-icons.woff2"
        try:
            css = css_path.read_text(encoding="utf-8")
            woff2_b64 = base64.b64encode(
                woff2_path.read_bytes()
            ).decode("ascii")
            data_uri = (
                "data:font/woff2;base64," + woff2_b64
            )
            css = css.replace(
                "url(\"fonts/bootstrap-icons.woff2\")",
                f"url(\"{data_uri}\")",
            )
            return css
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "Bootstrap Icons assets not found (%s). "
                "Icons will not render.", exc,
            )
            return ""


# ════════════════════════════════════════════════════════
#  CHART TEMPLATE
# ════════════════════════════════════════════════════════

_CHART_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ data.title }} — Edgar Alfaro Report</title>
<style>{{ bootstrap_css }}</style>
<style>{{ bootstrap_icons_css }}</style>
<style>
/* ── DESIGN TOKENS ────────────────────────────────────────────── */
:root {
  --sb: 240px;

  /* Primary */
  --primary:        #3B82F6;
  --primary-dk:     #2563EB;
  --primary-lt:     #60A5FA;
  --primary-light:  #EFF6FF;
  --primary-glow:   rgba(59,130,246,.15);

  /* Semantic */
  --success:        #059669;
  --success-bg:     #D1FAE5;
  --warning:        #D97706;
  --warning-bg:     #FEF3C7;
  --danger:         #DC2626;
  --danger-bg:      #FEE2E2;

  /* Sidebar */
  --sb-bg:          #0F1117;
  --sb-surface:     #1A1D27;
  --sb-border:      rgba(255,255,255,.06);
  --sb-text:        rgba(255,255,255,.45);
  --sb-text-hv:     rgba(255,255,255,.88);
  --sb-active-bg:   rgba(59,130,246,.16);
  --sb-active-glow: rgba(59,130,246,.25);
  --sb-section:     rgba(255,255,255,.22);

  /* Surfaces */
  --bg:             #F8F9FB;
  --surface:        #FFFFFF;
  --surface-2:      #F3F4F6;
  --surface-3:      #E5E7EB;

  /* Borders */
  --border:         #E5E7EB;
  --border-2:       #D1D5DB;

  /* Text */
  --text:           #111827;
  --text-2:         #374151;
  --text-3:         #6B7280;

  /* Typography */
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI",
          "Helvetica Neue", Arial, system-ui, sans-serif;
  --mono: "SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace;

  /* Geometry */
  --r-sm:   6px;
  --r-md:   10px;
  --r-lg:   14px;
  --r-xl:   18px;
  --r-2xl:  24px;
  --r-pill: 999px;

  /* Shadows */
  --sh:     0 1px 2px rgba(0,0,0,.05), 0 1px 3px rgba(0,0,0,.04);
  --sh-md:  0 4px 6px rgba(0,0,0,.05), 0 2px 4px rgba(0,0,0,.04);
  --sh-lg:  0 10px 15px rgba(0,0,0,.06), 0 4px 6px rgba(0,0,0,.04);
  --sh-in:  inset 0 1px 0 rgba(255,255,255,.85);
  --sh-in-dk: inset 0 1px 0 rgba(255,255,255,.06);

  /* Easing */
  --ease:       cubic-bezier(.4, 0, .2, 1);
  --t-fast: 150ms var(--ease);
  --t-base: 200ms var(--ease);
}

/* ── RESET ───────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; text-size-adjust: 100%; }
body {
  font-family: var(--font);
  background-color: var(--bg);
  color: var(--text);
  display: flex;
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

/* ── SIDEBAR ─────────────────────────────────────────────────── */
#sidebar {
  width: var(--sb);
  min-height: 100vh;
  background: var(--sb-bg);
  border-right: 1px solid var(--sb-border);
  position: fixed; top: 0; left: 0;
  overflow-y: auto; z-index: 1000;
  display: flex; flex-direction: column;
  transition: transform var(--t-base);
}
#sidebar::-webkit-scrollbar { width: 3px; }
#sidebar::-webkit-scrollbar-track { background: transparent; }
#sidebar::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,.08);
  border-radius: var(--r-pill);
}

/* Brand */
.sb-brand {
  display: flex; align-items: center; gap: 12px;
  padding: 24px 20px 20px;
  border-bottom: 1px solid var(--sb-border);
  text-decoration: none; flex-shrink: 0;
}
.sb-logo {
  width: 36px; height: 36px;
  background: var(--primary);
  border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 800; font-size: .75rem;
  flex-shrink: 0;
}
.sb-name {
  font-size: .92rem; font-weight: 700;
  color: rgba(255,255,255,.92);
}

/* Sidebar sections & links */
.sb-section {
  padding: 20px 20px 6px;
  font-size: .58rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.2px;
  color: var(--sb-section);
}
.sb-a {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 12px;
  margin: 1px 8px;
  color: var(--sb-text);
  font-size: .8rem; font-weight: 500;
  text-decoration: none;
  border-radius: var(--r-md);
  transition: color var(--t-fast), background var(--t-fast);
}
.sb-a:hover { color: var(--sb-text-hv); background: rgba(255,255,255,.06); }
.sb-a.active {
  color: rgba(255,255,255,.9);
  background: var(--sb-active-bg);
  font-weight: 600;
}
.sb-a i { font-size: .88rem; width: 16px; flex-shrink: 0; }
.sb-badge {
  margin-left: auto;
  font-size: .58rem; font-weight: 600;
  padding: 1px 8px; border-radius: var(--r-pill);
  background: rgba(255,255,255,.08);
  color: rgba(255,255,255,.45);
  font-family: var(--mono);
}
.sb-badge.warn { background: rgba(217,119,6,.15); color: #FBBF24; }
.sb-badge.dng  { background: rgba(220,38,38,.15); color: #FCA5A5; }

/* Sidebar footer */
.sb-footer {
  margin-top: auto; padding: 16px 16px 20px;
  border-top: 1px solid var(--sb-border); flex-shrink: 0;
}
.sb-footer-text {
  font-size: .68rem; color: var(--sb-text);
  text-align: center;
}

/* ── TOPBAR ──────────────────────────────────────────────────── */
#topbar {
  background: rgba(255,255,255,.92);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  padding: 0 28px;
  height: 60px;
  display: flex; align-items: center; gap: 12px;
  position: sticky; top: 0; z-index: 900;
}
.tb-toggle {
  display: none; background: none; border: none;
  font-size: 1.2rem; color: var(--text); cursor: pointer;
  padding: 6px; border-radius: var(--r-sm);
}
.tb-toggle:hover { background: var(--surface-2); }
.tb-bc { display: flex; align-items: center; gap: 8px; font-size: .78rem; }
.tb-bc .bc-root { color: var(--text-3); font-weight: 500; }
.tb-bc .bc-sep  { color: var(--border-2); font-size: .6rem; }
.tb-bc .bc-cur  { font-weight: 700; color: var(--text); }
.tb-actions { margin-left: auto; display: flex; align-items: center; gap: 6px; }
.tb-chip {
  padding: 5px 12px; border-radius: var(--r-pill);
  background: var(--surface-2); border: 1px solid var(--border);
  font-size: .68rem; font-weight: 500; color: var(--text-3);
}
.tb-icon-btn {
  width: 36px; height: 36px; border-radius: var(--r-md);
  background: var(--surface-2); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  color: var(--text-3); font-size: .9rem; cursor: pointer;
  transition: background var(--t-fast);
}
.tb-icon-btn:hover { background: var(--primary-light); color: var(--primary); }

/* ── MAIN LAYOUT ─────────────────────────────────────────────── */
#main { margin-left: var(--sb); flex: 1; display: flex; flex-direction: column; min-height: 100vh; }
#content { padding: 36px 36px 72px; flex: 1; }
.mb-sec { margin-bottom: 54px; }

/* Section header */
.sec-hd { margin-bottom: 28px; }
.sec-tag {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: .59rem; font-weight: 800;
  text-transform: uppercase; letter-spacing: 1.3px;
  color: var(--primary); background: var(--primary-light);
  padding: 5px 14px; border-radius: var(--r-pill);
  margin-bottom: 12px;
  border: 1px solid rgba(79,70,229,.14);
  box-shadow: 0 1px 6px rgba(79,70,229,.10), var(--sh-in);
}
.sec-hd h2 {
  font-size: 1.45rem; font-weight: 800; color: var(--text);
  letter-spacing: -.65px; margin-bottom: 8px; line-height: 1.2;
}
.sec-hd p  {
  font-size: .875rem; color: var(--text-3); line-height: 1.72;
  max-width: 640px;
}
.sec-hd p strong { color: var(--text-2); font-weight: 700; }

/* ── KPI CARDS ───────────────────────────────────────────────── */
.kpi {
  background: var(--surface);
  border-radius: var(--r-xl);
  border: 1px solid var(--border);
  padding: 24px 20px 20px;
  height: 100%;
  box-shadow: var(--sh);
}
.kpi-icon {
  width: 40px; height: 40px;
  border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; margin-bottom: 16px;
  color: var(--primary);
  background: var(--primary-light);
}
.kpi.suc .kpi-icon { color: var(--success); background: var(--success-bg); }
.kpi.dng .kpi-icon { color: var(--danger);  background: var(--danger-bg); }
.kpi.wrn .kpi-icon { color: var(--warning); background: var(--warning-bg); }
.kpi-lbl {
  font-size: .6rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--text-3); margin-bottom: 6px;
}
.kpi-val {
  font-size: 2.4rem; font-weight: 700; line-height: 1;
  letter-spacing: -2px; margin-bottom: 8px; color: var(--text);
  font-variant-numeric: tabular-nums;
}
.kpi.suc .kpi-val { color: var(--success); }
.kpi.dng .kpi-val { color: var(--danger); }
.kpi.wrn .kpi-val { color: var(--warning); }
.kpi-sub { font-size: .72rem; color: var(--text-3); }

/* ── PANELS ──────────────────────────────────────────────────── */
.panel {
  background: var(--surface);
  border-radius: var(--r-xl);
  border: 1px solid var(--border);
  box-shadow: var(--sh);
  height: 100%; overflow: hidden;
}
.panel-hd {
  padding: 18px 24px 16px;
  display: flex; align-items: flex-start; gap: 12px;
  border-bottom: 1px solid var(--border);
}
.panel-hd .ph-icon {
  width: 36px; height: 36px; border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  font-size: .85rem; flex-shrink: 0; margin-top: 1px;
  background: var(--surface-2); color: var(--text-3);
}
.ph-title { font-size: .9rem; font-weight: 700; color: var(--text); }
.ph-desc  { font-size: .72rem; color: var(--text-3); margin-top: 3px; }
.panel-body { padding: 20px 24px 24px; }

/* ── EXECUTIVE SUMMARY LIST ──────────────────────────────────── */
.sum-list { list-style: none; }
.sum-list li {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
  font-size: .86rem; line-height: 1.65; color: var(--text-2);
}
.sum-list li:last-child { border-bottom: none; padding-bottom: 0; }
.sum-list li strong { color: var(--text); font-weight: 700; }
.sum-ico {
  width: 32px; height: 32px; border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  font-size: .78rem; flex-shrink: 0; margin-top: 2px;
  background: var(--surface-2); color: var(--text-3);
}

/* ── PIPELINE — MINI STATS ───────────────────────────────────── */
.ms {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  padding: 20px 16px 18px;
  text-align: center;
  box-shadow: var(--sh);
  height: 100%;
}
.ms-val {
  font-size: 1.8rem; font-weight: 700; color: var(--text);
  letter-spacing: -1px; line-height: 1;
  margin-bottom: 6px;
  font-variant-numeric: tabular-nums;
}
.ms-lbl {
  font-size: .6rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .08em;
  color: var(--text-3);
}

/* Distribution strip */
.stor-strip {
  height: 11px; border-radius: var(--r-pill); overflow: hidden;
  display: flex; gap: 2px; margin: 14px 0;
  background: var(--border);
  box-shadow: inset 0 1px 4px rgba(0,0,0,.08);
}
.stor-seg { height: 100%; }
.stor-seg:first-child { border-radius: var(--r-pill) 0 0 var(--r-pill); }
.stor-seg:last-child  { border-radius: 0 var(--r-pill) var(--r-pill) 0; }

/* ── CONFIDENCE BAR ──────────────────────────────────────────── */
.conf-wrap {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 18px 22px;
  display: flex; align-items: center; gap: 24px;
  margin-top: 4px;
}
.conf-label {
  font-size: .6rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--text-3); margin-bottom: 10px;
}
.conf-track {
  flex: 1; height: 8px;
  background: var(--border); border-radius: var(--r-pill); overflow: hidden;
}
.conf-fill {
  height: 100%; border-radius: var(--r-pill);
  background: var(--success);
  transition: width 1.2s ease;
}
.conf-score {
  font-size: 1.8rem; font-weight: 700; color: var(--success);
  white-space: nowrap; letter-spacing: -1px;
  font-variant-numeric: tabular-nums;
}

/* ── WORD CLOUD CONTAINERS ───────────────────────────────────── */
.wc-box {
  height: 220px; border-radius: var(--r-xl);
  border: 1px solid var(--border);
  background: var(--surface);
  overflow: hidden;
}
.wc-box img { width: 100%; height: 100%; object-fit: contain; }

/* ── SCATTER IFRAMES ─────────────────────────────────────────── */
iframe.scatter-frame {
  width: 100%; height: 720px;
  border: none; border-radius: var(--r-xl);
  display: block;
  box-shadow: 0 2px 18px rgba(6,15,40,.07);
}

/* ── TOPIC CARDS ─────────────────────────────────────────────── */
.topic-col-hd {
  display: flex; align-items: center; gap: 8px;
  font-size: .8rem; font-weight: 700;
  padding: 10px 14px; border-radius: var(--r-md); margin-bottom: 12px;
  border: 1px solid var(--border);
  background: var(--surface-2); color: var(--text-2);
}
.topic-col-hd.pos { border-left: 3px solid var(--success); }
.topic-col-hd.neg { border-left: 3px solid var(--danger); }
.tc {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 16px 18px;
  margin-bottom: 10px;
  border-left: 3px solid var(--primary);
}
.tc.neg { border-left-color: var(--danger); }
.tc-name  { font-size: .86rem; font-weight: 700; color: var(--text); margin-bottom: 3px; }
.tc-kw    { font-size: .7rem; color: var(--text-3); margin-bottom: 8px; line-height: 1.5; }
.tc-quote {
  font-size: .78rem; font-style: italic; color: var(--text-2);
  background: var(--surface-2);
  border-radius: var(--r-md);
  padding: 8px 12px;
  line-height: 1.55; margin-bottom: 8px;
}
.tc-meta  { font-size: .68rem; font-weight: 500; color: var(--text-3); }

/* ── SIMILARITY TABLE ────────────────────────────────────────── */
.sim-tbl { width: 100%; border-collapse: collapse; }
.sim-tbl th {
  font-size: .6rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--text-3);
  padding: 10px 14px; border-bottom: 2px solid var(--border);
  text-align: left; white-space: nowrap;
}
.sim-tbl td {
  padding: 11px 14px; border-bottom: 1px solid var(--border);
  font-size: .82rem; vertical-align: middle; color: var(--text-2);
}
.sim-tbl tbody tr:hover td { background: var(--surface-2); }
.sim-tbl .td-rank { font-size: .68rem; font-weight: 600; color: var(--text-3); width: 32px; }
.sim-tbl .td-text { max-width: 360px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sim-row { display: flex; align-items: center; gap: 8px; }
.sim-track { flex: 1; height: 6px; background: var(--border); border-radius: var(--r-pill); overflow: hidden; min-width: 60px; }
.sim-fill { height: 100%; border-radius: var(--r-pill); background: var(--primary); }
.sim-score { font-size: .75rem; font-weight: 600; color: var(--primary); white-space: nowrap; font-family: var(--mono); min-width: 36px; text-align: right; }
.badge-sent { display: inline-block; padding: 2px 10px; border-radius: var(--r-pill); font-size: .65rem; font-weight: 600; }
.badge-sent.pos { background: var(--success-bg); color: #065F46; }
.badge-sent.neg { background: var(--danger-bg);  color: #7F1D1D; }

/* ── CLI PARAMS ──────────────────────────────────────────────── */
details.cli-details {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 8px 14px;
  font-size: .78rem;
}
details.cli-details summary {
  cursor: pointer; font-weight: 600; color: var(--text-3);
  user-select: none;
}
details.cli-details[open] summary { margin-bottom: 10px; }
.cli-tbl { font-size: .74rem; width: 100%; }
.cli-tbl tr td { padding: 3px 8px; }
.cli-tbl td:first-child { font-weight: 600; color: var(--text-3); white-space: nowrap; padding-right: 16px; }
.cli-tbl td:last-child  { font-family: var(--mono); font-size: .69rem; color: var(--text-2); }

/* ── FALLBACK ALERT ──────────────────────────────────────────── */
.alert-info {
  background: var(--primary-light);
  border: 1px solid rgba(59,130,246,.18);
  border-radius: var(--r-md);
  padding: 12px 16px;
  font-size: .84rem; color: var(--text-2);
}

/* ── REPORT FOOTER ───────────────────────────────────────────── */
.rpt-footer {
  padding: 18px 36px;
  border-top: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 10px;
  font-size: .72rem; color: var(--text-3);
}
.rpt-footer strong { color: var(--text-2); font-weight: 600; }

/* ── MOBILE ──────────────────────────────────────────────────── */
@media (max-width: 768px) {
  #sidebar { transform: translateX(-100%); box-shadow: none; }
  #sidebar.open { transform: translateX(0); box-shadow: 8px 0 60px rgba(4,11,32,.75); }
  #main { margin-left: 0; }
  .tb-toggle { display: flex; align-items: center; justify-content: center; }
  #content { padding: 22px 18px 52px; }
  .tb-chip { display: none; }
  #readProg { left: 0; }
  .kpi-val  { font-size: 2.2rem; letter-spacing: -1.5px; }
  .ms-val   { font-size: 1.65rem; }
  .conf-wrap { flex-direction: column; align-items: flex-start; gap: 12px; }
  .conf-score { font-size: 1.6rem; }
  iframe.scatter-frame { height: 480px; }
  .mb-sec { margin-bottom: 38px; }
}
@media (max-width: 480px) {
  .kpi-val { font-size: 1.9rem; letter-spacing: -1px; }
  .ms-val  { font-size: 1.45rem; }
  .sim-tbl .td-text { max-width: 160px; }
  #content { padding: 16px 14px 48px; }
  .sec-hd h2 { font-size: 1.2rem; }
}

/* ── PRINT ───────────────────────────────────────────────────── */
@media print {
  #sidebar, #topbar { display: none !important; }
  #main { margin-left: 0 !important; }
  .kpi, .panel, .tc, .ms { box-shadow: none !important; break-inside: avoid; }
  .mb-sec { margin-bottom: 24px; }
  body { background: #fff !important; }
}
</style>
<script>{{ plotly_js }}</script>
</head>
<body>

<nav id="sidebar">
  <a href="#overview" class="sb-brand">
    <div class="sb-logo">B</div>
    <div class="sb-name">Beternovik</div>
  </a>

  <div class="sb-section">Report</div>
  <a href="#overview" class="sb-a active"><i class="bi bi-grid-1x2-fill"></i> Overview <span class="sb-badge">{{ data.total_rows }}</span></a>
  <a href="#summary"  class="sb-a"><i class="bi bi-list-check"></i> Executive Summary</a>
  <a href="#pipeline" class="sb-a"><i class="bi bi-diagram-3-fill"></i> Pipeline Summary</a>

  <div class="sb-section">Analysis</div>
  <a href="#outliers"  class="sb-a"><i class="bi bi-exclamation-triangle-fill"></i> Outlier Analysis <span class="sb-badge warn">{{ data.outliers_count }}</span></a>
  <a href="#sentiment" class="sb-a"><i class="bi bi-bar-chart-fill"></i> Sentiment Analysis</a>
  <a href="#topics"    class="sb-a"><i class="bi bi-diagram-2-fill"></i> Topic Modeling</a>
  <a href="#semantic"  class="sb-a"><i class="bi bi-broadcast-pin"></i> Semantic Analysis</a>

  <div class="sb-footer">
    <div class="sb-footer-text">Beternovik &mdash; Semantic Cluster</div>
  </div>
</nav>

<!-- ═══════════════════════════════════════════════════════════
     MAIN
═══════════════════════════════════════════════════════════ -->
<div id="main">

  <!-- Topbar -->
  <header id="topbar">
    <button class="tb-toggle" onclick="document.getElementById('sidebar').classList.toggle('open')" aria-label="Toggle sidebar">
      <i class="bi bi-list"></i>
    </button>
    <div class="tb-bc">
      <span class="bc-root">Beternovik</span>
      <i class="bi bi-chevron-right bc-sep"></i>
      <span class="bc-cur">{{ data.title }}</span>
    </div>
    <div class="tb-actions">
      <div class="tb-chip">{{ data.timestamp }}</div>
      <button class="tb-icon-btn" onclick="window.print()" title="Print report">
        <i class="bi bi-printer"></i>
      </button>
    </div>
  </header>

  <div id="content">

    <!-- Overview -->
    <section id="overview" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-grid-1x2-fill"></i> Overview</div>
        <h2>Key Metrics at a Glance</h2>
        <p>Top-level numbers from the analysis of <strong>{{ data.total_rows }}</strong> comments across all pipeline stages.</p>
      </div>
      <div class="row g-3">
        <div class="col-6 col-xl-3">
          <div class="kpi">
            <div class="kpi-icon"><i class="bi bi-chat-text-fill"></i></div>
            <div class="kpi-lbl">Total Comments</div>
            <div class="kpi-val">{{ data.total_rows }}</div>
            <div class="kpi-sub">All records processed</div>
          </div>
        </div>
        <div class="col-6 col-xl-3">
          <div class="kpi suc">
            <div class="kpi-icon"><i class="bi bi-emoji-smile-fill"></i></div>
            <div class="kpi-lbl">Positive</div>
            <div class="kpi-val">{{ "%.1f"|format(data.pos_pct * 100) }}%</div>
            <div class="kpi-sub">{{ data.pos_count }} comments</div>
          </div>
        </div>
        <div class="col-6 col-xl-3">
          <div class="kpi dng">
            <div class="kpi-icon"><i class="bi bi-emoji-frown-fill"></i></div>
            <div class="kpi-lbl">Negative</div>
            <div class="kpi-val">{{ "%.1f"|format(data.neg_pct * 100) }}%</div>
            <div class="kpi-sub">{{ data.neg_count }} comments</div>
          </div>
        </div>
        <div class="col-6 col-xl-3">
          <div class="kpi wrn">
            <div class="kpi-icon"><i class="bi bi-exclamation-triangle-fill"></i></div>
            <div class="kpi-lbl">Outliers Removed</div>
            <div class="kpi-val">{{ data.outliers_count }}</div>
            <div class="kpi-sub">Filtered before analysis</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Executive Summary -->
    <section id="summary" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-list-check"></i> Executive Summary</div>
        <h2>Key Findings</h2>
        <p>High-level overview of the most important findings from all pipeline stages.</p>
      </div>
      <div class="panel">
        <div class="panel-hd">
          <div class="ph-icon blue"><i class="bi bi-lightning-fill"></i></div>
          <div>
            <div class="ph-title">Summary of Results</div>
            <div class="ph-desc">Automated highlights from all pipeline stages</div>
          </div>
        </div>
        <div class="panel-body">
          <ul class="sum-list">
            <li>
              <div class="sum-ico"><i class="bi bi-chat-text-fill"></i></div>
              <span>Analyzed <strong>{{ data.total_rows }}</strong> comments in total.</span>
            </li>
            <li>
              <div class="sum-ico"><i class="bi bi-activity"></i></div>
              <span>Sentiment split: <strong>{{ "%.1f"|format(data.pos_pct * 100) }}% positive</strong> ({{ data.pos_count }} comments), <strong>{{ "%.1f"|format(data.neg_pct * 100) }}% negative</strong> ({{ data.neg_count }} comments).</span>
            </li>
            <li>
              <div class="sum-ico"><i class="bi bi-exclamation-triangle-fill"></i></div>
              <span>Detected <strong>{{ data.outliers_count }}</strong> outlier(s) — removed before downstream analysis.</span>
            </li>
            {% if data.top_positive_topic %}
            <li>
              <div class="sum-ico"><i class="bi bi-tag-fill"></i></div>
              <span>Top positive topic: <strong>{{ data.top_positive_topic }}</strong>.</span>
            </li>
            {% endif %}
            {% if data.top_negative_topic %}
            <li>
              <div class="sum-ico"><i class="bi bi-tag-fill"></i></div>
              <span>Top negative topic: <strong>{{ data.top_negative_topic }}</strong>.</span>
            </li>
            {% endif %}
            {% if data.top_concept_score > 0 %}
            <li>
              <div class="sum-ico"><i class="bi bi-vector-pen"></i></div>
              <span>Highest semantic similarity to the target concept: <strong>{{ "%.3f"|format(data.top_concept_score) }}</strong>.</span>
            </li>
            {% endif %}
            <li>
              <div class="sum-ico"><i class="bi bi-shield-check-fill"></i></div>
              <span>Confidence rate: <strong>{{ "%.1f"|format(data.confidence_rate * 100) }}%</strong>.</span>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <!-- Pipeline Summary -->
    <section id="pipeline" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-diagram-3-fill"></i> Pipeline Summary</div>
        <h2>Data Processing Overview</h2>
        <p>Overview of the dataset after preprocessing, outlier removal, and sentiment classification.</p>
      </div>
      <div class="row g-3 mb-3">
        <div class="col-6 col-md-3"><div class="ms"><div class="ms-val">{{ data.total_rows }}</div><div class="ms-lbl">Total Comments</div></div></div>
        <div class="col-6 col-md-3"><div class="ms"><div class="ms-val" style="color:var(--warning)">{{ data.outliers_count }}</div><div class="ms-lbl">Outliers Removed</div></div></div>
        <div class="col-6 col-md-3"><div class="ms"><div class="ms-val" style="color:var(--success)">{{ data.pos_count }}</div><div class="ms-lbl">Positive</div></div></div>
        <div class="col-6 col-md-3"><div class="ms"><div class="ms-val" style="color:var(--danger)">{{ data.neg_count }}</div><div class="ms-lbl">Negative</div></div></div>
      </div>
      <div class="panel">
        <div class="panel-hd">
          <div class="ph-icon blue"><i class="bi bi-pie-chart-fill"></i></div>
          <div>
            <div class="ph-title">Dataset Breakdown &amp; Model Confidence</div>
            <div class="ph-desc">Proportional distribution across sentiment classes and confidence score</div>
          </div>
        </div>
        <div class="panel-body">
          <div class="d-flex align-items-center justify-content-between mb-1">
            <div style="font-size:.78rem;font-weight:700;color:var(--text)">Distribution</div>
            <div style="font-size:.73rem;color:var(--text-3)">Positive &middot; Negative &middot; Outliers</div>
          </div>
          {% set total = data.total_rows if data.total_rows > 0 else 1 %}
          {% set pos_w = (data.pos_count / total * 100) %}
          {% set neg_w = (data.neg_count / total * 100) %}
          {% set out_w = (data.outliers_count / total * 100) %}
          <div class="stor-strip">
            <div class="stor-seg" style="width:{{ pos_w }}%;background:var(--success);"></div>
            <div class="stor-seg" style="width:{{ neg_w }}%;background:var(--danger);"></div>
            <div class="stor-seg" style="width:{{ out_w }}%;background:var(--warning);"></div>
          </div>
          <div class="d-flex flex-wrap gap-3 mb-4" style="font-size:.73rem;">
            <span><i class="bi bi-circle-fill me-1" style="color:var(--success);font-size:.5rem;vertical-align:middle;"></i>Positive — {{ data.pos_count }} ({{ "%.1f"|format(pos_w) }}%)</span>
            <span><i class="bi bi-circle-fill me-1" style="color:var(--danger);font-size:.5rem;vertical-align:middle;"></i>Negative — {{ data.neg_count }} ({{ "%.1f"|format(neg_w) }}%)</span>
            <span><i class="bi bi-circle-fill me-1" style="color:var(--warning);font-size:.5rem;vertical-align:middle;"></i>Outliers — {{ data.outliers_count }} ({{ "%.1f"|format(out_w) }}%)</span>
          </div>
          <div class="conf-wrap">
            <div style="flex:1;">
              <div class="conf-label">Model Confidence Rate</div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{{ "%.1f"|format(data.confidence_rate * 100) }}%"></div>
              </div>
            </div>
            <div style="text-align:right;">
              <div class="conf-label">Score</div>
              <div class="conf-score">{{ "%.1f"|format(data.confidence_rate * 100) }}%</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Outlier Analysis -->
    <section id="outliers" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-exclamation-triangle-fill"></i> Outlier Analysis</div>
        <h2>Flagged Comments</h2>
        <p>Comments flagged as statistically unusual. The word cloud highlights frequent terms; n-gram charts reveal multi-word patterns among outliers.</p>
      </div>
      <div class="row g-3 mb-3">
        <div class="col-12">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-cloud-fill"></i></div>
              <div>
                <div class="ph-title">Word Cloud — Outliers</div>
                <div class="ph-desc">Most frequent terms in {{ data.outliers_count }} flagged comments</div>
              </div>
            </div>
            <div class="panel-body">
              <div class="wc-box" style="height:280px;">
                {% if data.wc_outliers_b64 %}
                <img src="{{ data.wc_outliers_b64 }}" alt="Outliers Word Cloud">
                {% else %}
                <div style="padding:80px 20px;text-align:center;color:var(--text-3);font-size:.85rem;">No outliers detected</div>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="row g-3">
        <div class="col-md-4">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-bar-chart-horizontal-fill"></i></div>
              <div>
                <div class="ph-title">Unigrams</div>
                <div class="ph-desc">Single-word frequency</div>
              </div>
            </div>
            <div class="panel-body">{{ data.ngrams_unigrams_html | safe }}</div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-bar-chart-horizontal-fill"></i></div>
              <div>
                <div class="ph-title">Bigrams</div>
                <div class="ph-desc">Two-word patterns</div>
              </div>
            </div>
            <div class="panel-body">{{ data.ngrams_bigrams_html | safe }}</div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-bar-chart-horizontal-fill"></i></div>
              <div>
                <div class="ph-title">Trigrams</div>
                <div class="ph-desc">Three-word patterns</div>
              </div>
            </div>
            <div class="panel-body">{{ data.ngrams_trigrams_html | safe }}</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Sentiment Analysis -->
    <section id="sentiment" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-bar-chart-fill"></i> Sentiment Analysis</div>
        <h2>Comment Sentiment Distribution</h2>
        <p>Each comment was classified as positive or negative. The word clouds show the most frequent terms for each sentiment group.</p>
      </div>
      <div class="row g-3 mb-3">
        <div class="col-12">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-pie-chart-fill"></i></div>
              <div>
                <div class="ph-title">Sentiment Split</div>
                <div class="ph-desc">Overall distribution across {{ data.total_rows }} comments</div>
              </div>
            </div>
            <div class="panel-body" style="text-align:center;">
              <div style="max-width:320px;margin:0 auto;">
                {{ data.sentiment_pie_html | safe }}
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="row g-3">
        <div class="col-md-6">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-cloud-fill"></i></div>
              <div>
                <div class="ph-title">Word Cloud — Positive</div>
                <div class="ph-desc">{{ data.pos_count }} comments</div>
              </div>
            </div>
            <div class="panel-body">
              <div class="wc-box" style="height:260px;">
                {% if data.wc_positive_b64 %}
                <img src="{{ data.wc_positive_b64 }}" alt="Positive Word Cloud">
                {% else %}
                <div style="padding:80px 20px;text-align:center;color:var(--text-3);font-size:.85rem;">No positive comments</div>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
        <div class="col-md-6">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-cloud-fill"></i></div>
              <div>
                <div class="ph-title">Word Cloud — Negative</div>
                <div class="ph-desc">{{ data.neg_count }} comments</div>
              </div>
            </div>
            <div class="panel-body">
              <div class="wc-box" style="height:260px;">
                {% if data.wc_negative_b64 %}
                <img src="{{ data.wc_negative_b64 }}" alt="Negative Word Cloud">
                {% else %}
                <div style="padding:80px 20px;text-align:center;color:var(--text-3);font-size:.85rem;">No negative comments</div>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Topic Modeling -->
    <section id="topics" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-diagram-2-fill"></i> Topic Modeling</div>
        <h2>Topic Modeling Overview</h2>
        <p>Topics automatically extracted from positive and negative comments. Each point is a comment; color and shape encode the cluster.</p>
      </div>
      <div class="row g-3">
        <div class="col-12">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-diagram-3-fill"></i></div>
              <div>
                <div class="ph-title">Topic Cluster Scatter</div>
                <div class="ph-desc">2D projection of comment embeddings — color-coded by topic cluster</div>
              </div>
            </div>
            <div class="panel-body">
              {% if data.scatter_topics_html %}
              <iframe srcdoc="{{ data.scatter_topics_html | e }}" class="scatter-frame"></iframe>
              {% else %}
              <div class="alert-info">
                Fewer than {{ data.min_partition_size }} comments in one or both partitions. Word frequency shown instead.
              </div>
              {% endif %}
            </div>
          </div>
        </div>

        {% if data.pos_topic_summaries or data.neg_topic_summaries %}
        <div class="col-md-6">
          <div class="topic-col-hd pos"><i class="bi bi-emoji-smile-fill"></i> Positive Topics</div>
          {% for t in data.pos_topic_summaries %}
          <div class="tc">
            <div class="tc-name">{{ t.label }}</div>
            <div class="tc-kw">{{ t.keywords }}</div>
            {% if t.representative %}<div class="tc-quote">{{ t.representative }}</div>{% endif %}
            <div class="tc-meta">{{ t.count }} comments</div>
          </div>
          {% endfor %}
        </div>
        <div class="col-md-6">
          <div class="topic-col-hd neg"><i class="bi bi-emoji-frown-fill"></i> Negative Topics</div>
          {% for t in data.neg_topic_summaries %}
          <div class="tc neg">
            <div class="tc-name">{{ t.label }}</div>
            <div class="tc-kw">{{ t.keywords }}</div>
            {% if t.representative %}<div class="tc-quote">{{ t.representative }}</div>{% endif %}
            <div class="tc-meta">{{ t.count }} comments</div>
          </div>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </section>

    <!-- Semantic Concept Analysis -->
    <section id="semantic" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-broadcast-pin"></i> Semantic Analysis</div>
        <h2>Semantic Concept Analysis</h2>
        <p>Each comment is scored by cosine similarity to the target concept{% if data.concept_name %} <strong>&ldquo;{{ data.concept_name }}&rdquo;</strong>{% endif %}. Brighter / higher points are more semantically related.</p>
      </div>
      <div class="row g-3">
        <div class="col-12">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-broadcast-pin"></i></div>
              <div>
                <div class="ph-title">Concept Similarity Scatter</div>
                <div class="ph-desc">2D projection colored by cosine similarity to concept{% if data.concept_name %} <strong>&ldquo;{{ data.concept_name }}&rdquo;</strong>{% endif %}</div>
              </div>
            </div>
            <div class="panel-body">
              {% if data.scatter_semantic_html %}
              <iframe srcdoc="{{ data.scatter_semantic_html | e }}" class="scatter-frame"></iframe>
              {% else %}
              <div style="padding:60px 20px;text-align:center;color:var(--text-3);font-size:.85rem;">Semantic scatter not available.</div>
              {% endif %}
            </div>
          </div>
        </div>

        {% if data.top5_semantic %}
        <div class="col-12">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon"><i class="bi bi-trophy-fill"></i></div>
              <div>
                <div class="ph-title">Top 5 Most Similar Comments</div>
                <div class="ph-desc">Comments ranked by highest cosine similarity to the target concept</div>
              </div>
            </div>
            <div class="panel-body">
              <div class="table-responsive">
                <table class="sim-tbl">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Comment</th>
                      <th style="width:200px;">Similarity</th>
                      <th style="width:110px;">Sentiment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {% for row in data.top5_semantic %}
                    <tr>
                      <td class="td-rank">{{ loop.index }}</td>
                      <td class="td-text" title="{{ row.text }}">{{ row.text }}</td>
                      <td>
                        <div class="sim-row">
                          <div class="sim-track">
                            <div class="sim-fill" style="width:{{ "%.0f"|format(row.similarity * 100) }}%;"></div>
                          </div>
                          <span class="sim-score">{{ "%.3f"|format(row.similarity) }}</span>
                        </div>
                      </td>
                      <td><span class="badge-sent {{ row.sentiment }}"><span class="dot"></span> {{ row.sentiment }}</span></td>
                    </tr>
                    {% endfor %}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
        {% endif %}
      </div>
    </section>

  </div><!-- /content -->

  <!-- Footer -->
  <footer class="rpt-footer">
    <span><strong>Developed by Edgar Alfaro Hernandez</strong> &mdash; Fully self-contained offline NLP report.</span>
  </footer>

</div><!-- /main -->

<script>
/* ══ Sidebar toggle (mobile) ══ */
document.querySelector('.tb-toggle')?.addEventListener('click', function() {
  document.getElementById('sidebar').classList.toggle('open');
});

/* ══ Close sidebar on outside click (mobile) ══ */
document.addEventListener('click', function(e) {
  var sb = document.getElementById('sidebar');
  if (!sb) return;
  if (sb.classList.contains('open') &&
      !sb.contains(e.target) &&
      !e.target.closest('.tb-toggle')) {
    sb.classList.remove('open');
  }
});
</script>
</body>
</html>"""