"""Chart-style offline HTML report via Jinja2 with Falcon sidebar layout.

Shares the same data assembly as ``ReportBuilder`` but renders a
different template (sidebar + topbar layout with Falcon design system).
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
    """Build a chart-style HTML report with Falcon sidebar dashboard template.

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


# ══════════════════════════════════════════════════════════════════════
#  CHART TEMPLATE  —  Meridian Premium Dashboard
# ══════════════════════════════════════════════════════════════════════

_CHART_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ data.title }} — Semantic Cluster Report</title>
<style>{{ bootstrap_css }}</style>
<style>{{ bootstrap_icons_css }}</style>
<style>
/* ════════════════════════════════════════════════════════════════
   MERIDIAN DESIGN SYSTEM v3.0
   Premium analytics dashboard — deep space sidebar + light surface
════════════════════════════════════════════════════════════════ */

/* ── DESIGN TOKENS ────────────────────────────────────────────── */
:root {
  --sb: 280px;

  /* Primary — Sophisticated Indigo */
  --primary:        #4F46E5;
  --primary-dk:     #3730A3;
  --primary-lt:     #818CF8;
  --primary-lter:   #A5B4FC;
  --primary-light:  #EEF2FF;
  --primary-glow:   rgba(79,70,229,.18);

  /* Warm Accent — Amber-Orange */
  --accent:         #F97316;
  --accent-warm:    #FBBF24;
  --accent-glow:    rgba(249,115,22,.15);

  /* Semantic */
  --success:        #059669;
  --success-dk:     #047857;
  --success-bg:     #D1FAE5;
  --success-mid:    #6EE7B7;
  --warning:        #D97706;
  --warning-dk:     #B45309;
  --warning-bg:     #FEF3C7;
  --danger:         #DC2626;
  --danger-dk:      #B91C1C;
  --danger-bg:      #FEE2E2;
  --danger-mid:     #FCA5A5;
  --purple:         #7C3AED;
  --purple-lt:      #A78BFA;
  --purple-bg:      #EDE9FE;
  --teal:           #0891B2;
  --teal-bg:        #E0F2FE;

  /* Sidebar — Deep Cosmic Navy */
  --sb-bg:          #040B20;
  --sb-bg-mid:      #07102E;
  --sb-bg-lt:       #0C1840;
  --sb-border:      rgba(255,255,255,.055);
  --sb-text:        rgba(172,186,228,.60);
  --sb-text-hv:     rgba(255,255,255,.94);
  --sb-active-bg:   rgba(79,70,229,.20);
  --sb-active-glow: rgba(79,70,229,.35);
  --sb-section:     rgba(172,186,228,.28);

  /* Surfaces */
  --bg:             #F0F4FF;
  --surface:        #FFFFFF;
  --surface-2:      #F6F9FF;
  --surface-3:      #EEF2FF;

  /* Borders */
  --border:         #E2E9F6;
  --border-2:       #C4D1EA;

  /* Text */
  --text:           #060F28;
  --text-2:         #1B2E5A;
  --text-3:         #526098;

  /* Typography */
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI Variable", "Segoe UI",
          "Helvetica Neue", Arial, system-ui, sans-serif;
  --mono: "SF Mono", "Cascadia Code", "Fira Code", Consolas, monospace;

  /* Geometry */
  --r-xs:   5px;
  --r-sm:   8px;
  --r-md:   13px;
  --r-lg:   18px;
  --r-xl:   22px;
  --r-2xl:  28px;
  --r-pill: 999px;

  /* Shadow system — ambient + directional */
  --sh:     0 1px 3px rgba(6,15,40,.05),
            0 4px 14px rgba(79,70,229,.06);
  --sh-md:  0 6px 20px rgba(6,15,40,.08),
            0 14px 44px rgba(79,70,229,.09);
  --sh-lg:  0 14px 44px rgba(6,15,40,.10),
            0 30px 72px rgba(79,70,229,.12);
  --sh-xl:  0 22px 64px rgba(6,15,40,.13),
            0 48px 96px rgba(79,70,229,.15);
  --sh-in:  inset 0 1px 0 rgba(255,255,255,.90),
            inset 0 -1px 0 rgba(6,15,40,.04);
  --sh-in-dk: inset 0 1px 0 rgba(255,255,255,.07),
              inset 0 -1px 0 rgba(0,0,0,.15);

  /* Easing */
  --ease-out:    cubic-bezier(.16, 1,   .30, 1);
  --ease-spring: cubic-bezier(.34, 1.2, .64, 1);
  --ease-smooth: cubic-bezier(.40, 0,   .20, 1);
  --t-fast: 150ms var(--ease-smooth);
  --t-base: 240ms var(--ease-smooth);
  --t-slow: 420ms var(--ease-out);
}

/* ── RESET ───────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; text-size-adjust: 100%; }
body {
  font-family: var(--font);
  background-color: var(--bg);
  background-image:
    radial-gradient(ellipse 75% 55% at 68% -8%,  rgba(79,70,229,.08) 0%, transparent 56%),
    radial-gradient(ellipse 55% 48% at 98% 95%,  rgba(124,58,237,.05) 0%, transparent 52%),
    radial-gradient(ellipse 48% 40% at 3%  65%,  rgba(249,115,22,.04) 0%, transparent 48%);
  background-attachment: fixed;
  color: var(--text);
  display: flex;
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ── READING PROGRESS ────────────────────────────────────────── */
#readProg {
  position: fixed;
  top: 0; left: var(--sb); right: 0;
  height: 2px;
  background: linear-gradient(90deg,
    var(--primary) 0%, var(--primary-lt) 30%,
    var(--accent) 60%, var(--primary) 100%);
  background-size: 300% 100%;
  animation: progressFlow 4s linear infinite;
  z-index: 1100;
  width: 0%;
  transition: width 60ms linear;
  border-radius: 0 0 var(--r-xs) 0;
  box-shadow: 0 0 12px rgba(79,70,229,.55), 0 0 24px rgba(79,70,229,.22);
}
@keyframes progressFlow {
  0%   { background-position:  300% 0; }
  100% { background-position: -300% 0; }
}

/* ── SIDEBAR ─────────────────────────────────────────────────── */
#sidebar {
  width: var(--sb);
  min-height: 100vh;
  background: linear-gradient(175deg,
    var(--sb-bg)     0%,
    var(--sb-bg-mid) 40%,
    var(--sb-bg-lt)  100%);
  border-right: 1px solid rgba(255,255,255,.028);
  position: fixed; top: 0; left: 0;
  overflow-y: auto; z-index: 1000;
  display: flex; flex-direction: column;
  transition: transform var(--t-slow);
  box-shadow:
    6px 0 64px rgba(4,11,32,.78),
    2px 0 12px rgba(4,11,32,.45),
    1px 0 0 rgba(255,255,255,.028);
}
#sidebar::-webkit-scrollbar { width: 3px; }
#sidebar::-webkit-scrollbar-track { background: transparent; }
#sidebar::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,.09);
  border-radius: var(--r-pill);
}
#sidebar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,.18); }

/* Brand */
.sb-brand {
  display: flex; align-items: center; gap: 14px;
  padding: 26px 22px 23px;
  border-bottom: 1px solid var(--sb-border);
  text-decoration: none; flex-shrink: 0;
  position: relative; overflow: hidden;
}
.sb-brand::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 150% 140% at 0% 0%,
      rgba(79,70,229,.28) 0%, transparent 55%),
    radial-gradient(ellipse 90% 90% at 100% 100%,
      rgba(249,115,22,.09) 0%, transparent 48%);
  pointer-events: none;
}
.sb-logo {
  width: 46px; height: 46px;
  background: linear-gradient(135deg, #4F46E5 0%, #7468F0 45%, #F97316 100%);
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 900; font-size: .9rem;
  letter-spacing: -.5px; flex-shrink: 0;
  position: relative; z-index: 1;
  box-shadow:
    0 0 0 1px rgba(255,255,255,.22),
    0 4px 20px rgba(79,70,229,.68),
    0 12px 48px rgba(79,70,229,.32);
}
.sb-names { display: flex; flex-direction: column; position: relative; z-index: 1; }
.sb-name {
  font-size: 1.06rem; font-weight: 800;
  color: rgba(255,255,255,.96);
  letter-spacing: -.5px; line-height: 1.15;
}
.sb-tag {
  font-size: .56rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .16em;
  color: var(--sb-text); margin-top: 3px;
}

/* Nav section labels */
.sb-section {
  padding: 20px 22px 7px;
  font-size: .55rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.7px;
  color: var(--sb-section);
}

/* Nav links */
.sb-a {
  display: flex; align-items: center; gap: 10px;
  padding: 9.5px 12px 9.5px 15px;
  margin: 2px 10px;
  color: var(--sb-text);
  font-size: .82rem; font-weight: 500;
  text-decoration: none;
  border-radius: var(--r-lg);
  transition: color var(--t-fast), background var(--t-fast), box-shadow var(--t-fast);
  position: relative;
}
.sb-a:hover {
  color: var(--sb-text-hv);
  background: rgba(255,255,255,.068);
}
.sb-a.active {
  color: rgba(255,255,255,.96);
  background: var(--sb-active-bg);
  font-weight: 600;
  box-shadow: var(--sh-in-dk), 0 2px 18px var(--sb-active-glow);
}
.sb-a.active::before {
  content: '';
  position: absolute; left: 0; top: 50%;
  transform: translateY(-50%);
  width: 3px; height: 52%;
  background: linear-gradient(180deg, var(--primary-lter), var(--primary-lt));
  border-radius: 0 var(--r-xs) var(--r-xs) 0;
  box-shadow: 0 0 10px var(--primary-lt), 0 0 20px rgba(79,70,229,.45);
}
.sb-a i {
  font-size: .92rem; width: 18px;
  opacity: .68; flex-shrink: 0;
  transition: opacity var(--t-fast), color var(--t-fast);
}
.sb-a:hover i, .sb-a.active i { opacity: 1; }
.sb-a.active i { color: var(--primary-lter); }

/* Sidebar badges */
.sb-badge {
  margin-left: auto;
  font-size: .57rem; font-weight: 700;
  padding: 2px 9px; border-radius: var(--r-pill);
  background: rgba(255,255,255,.07);
  color: rgba(255,255,255,.52);
  border: 1px solid rgba(255,255,255,.08);
  font-family: var(--mono);
  letter-spacing: .02em;
}
.sb-badge.warn { background: rgba(217,119,6,.16);  color: #FDE68A; border-color: rgba(217,119,6,.22); }
.sb-badge.suc  { background: rgba(5,150,105,.16);  color: #6EE7B7; border-color: rgba(5,150,105,.22); }
.sb-badge.dng  { background: rgba(220,38,38,.16);  color: #FCA5A5; border-color: rgba(220,38,38,.22); }

/* Sidebar info box */
.sb-infobox {
  margin: 18px 12px 0;
  background: rgba(255,255,255,.033);
  border: 1px solid var(--sb-border);
  border-radius: var(--r-xl);
  padding: 14px 16px;
}
.sb-infobox .info-title {
  font-weight: 700; font-size: .69rem;
  color: rgba(255,255,255,.42);
  margin-bottom: 10px;
  display: flex; align-items: center; gap: 7px;
}
.sb-infobox .info-title i { color: var(--primary-lter); font-size: .78rem; }
.sb-infobox .info-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 0; font-size: .68rem; color: var(--sb-text);
  border-bottom: 1px solid rgba(255,255,255,.038);
}
.sb-infobox .info-row:last-child { border-bottom: none; padding-bottom: 0; }
.sb-infobox .info-row span:last-child {
  color: rgba(255,255,255,.72);
  font-weight: 600; font-size: .67rem;
  font-family: var(--mono);
}

/* Sidebar footer */
.sb-footer {
  margin-top: auto; padding: 16px 16px 22px;
  border-top: 1px solid var(--sb-border); flex-shrink: 0;
}
.sb-user { display: flex; align-items: center; gap: 11px; }
.sb-avatar {
  width: 38px; height: 38px; border-radius: 12px;
  background: linear-gradient(135deg, #6875F5, #7C3AED);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-weight: 700; font-size: .73rem; flex-shrink: 0;
  box-shadow:
    0 3px 14px rgba(124,58,237,.48),
    0 0 0 1px rgba(255,255,255,.14);
}
.sb-uname { font-size: .82rem; font-weight: 700; color: rgba(255,255,255,.85); }
.sb-urole { font-size: .67rem; color: var(--sb-text); margin-top: 2px; }

/* ── TOPBAR ──────────────────────────────────────────────────── */
#topbar {
  background: rgba(255,255,255,.88);
  -webkit-backdrop-filter: blur(28px) saturate(1.9) brightness(1.04);
  backdrop-filter: blur(28px) saturate(1.9) brightness(1.04);
  border-bottom: 1px solid rgba(226,233,246,.96);
  padding: 0 32px;
  height: 64px;
  display: flex; align-items: center; gap: 14px;
  position: sticky; top: 0; z-index: 900;
  box-shadow:
    0 1px 0 rgba(226,233,246,1),
    0 4px 28px rgba(6,15,40,.05),
    0 12px 56px rgba(79,70,229,.05);
}
.tb-toggle {
  display: none; background: none; border: none;
  font-size: 1.25rem; color: var(--text); cursor: pointer;
  padding: 6px; border-radius: var(--r-sm);
  transition: color var(--t-fast), background var(--t-fast);
}
.tb-toggle:hover { background: var(--primary-light); color: var(--primary); }
.tb-bc { display: flex; align-items: center; gap: 9px; font-size: .78rem; }
.tb-bc .bc-root { color: var(--text-3); font-weight: 500; }
.tb-bc .bc-sep  { color: var(--border-2); font-size: .62rem; }
.tb-bc .bc-cur  {
  font-weight: 800; font-size: .92rem;
  letter-spacing: -.3px;
  background: linear-gradient(135deg, var(--text) 25%, var(--primary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.tb-actions { margin-left: auto; display: flex; align-items: center; gap: 8px; }
.tb-chip {
  display: inline-flex; align-items: center; gap: 9px;
  padding: 6px 14px; border-radius: var(--r-pill);
  background: var(--surface-2); border: 1px solid var(--border);
  font-size: .7rem; font-weight: 600; color: var(--text-3);
  box-shadow: var(--sh-in);
}
.tb-chip .pulse-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 0 3px var(--success-bg);
  animation: pulseGreen 2.8s ease-in-out infinite;
}
@keyframes pulseGreen {
  0%, 100% { box-shadow: 0 0 0 3px var(--success-bg); opacity: 1; }
  50%       { box-shadow: 0 0 0 6px rgba(5,150,105,.15); opacity: .8; }
}
.tb-icon-btn {
  width: 38px; height: 38px; border-radius: var(--r-md);
  background: var(--surface-2); border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  color: var(--text-3); font-size: .95rem; cursor: pointer;
  transition: all var(--t-base);
  box-shadow: var(--sh-in);
}
.tb-icon-btn:hover {
  background: var(--primary-light); color: var(--primary);
  border-color: rgba(79,70,229,.25);
  box-shadow: 0 0 0 3px var(--primary-glow), var(--sh-in);
  transform: translateY(-1px);
}
.btn-export {
  display: flex; align-items: center; gap: 7px;
  padding: 9px 18px; border-radius: var(--r-md);
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-lt) 100%);
  color: #fff; font-size: .78rem; font-weight: 700;
  border: none; cursor: pointer; text-decoration: none;
  box-shadow: 0 3px 12px rgba(79,70,229,.40), var(--sh-in-dk);
  transition: all var(--t-base);
  letter-spacing: -.1px;
}
.btn-export:hover {
  background: linear-gradient(135deg, var(--primary-dk) 0%, var(--primary) 100%);
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(79,70,229,.52), var(--sh-in-dk);
  color: #fff;
}

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
  border-radius: var(--r-2xl);
  border: 1px solid var(--border);
  padding: 28px 24px 24px;
  position: relative; overflow: hidden;
  height: 100%;
  box-shadow: var(--sh), var(--sh-in);
  transition: box-shadow var(--t-base), transform var(--t-base);
  cursor: default;
}
.kpi:hover {
  box-shadow: var(--sh-lg), var(--sh-in);
  transform: translateY(-7px);
}
/* Gradient top accent bar */
.kpi::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, var(--primary), var(--primary-lt) 55%, rgba(79,70,229,.10));
  border-radius: var(--r-2xl) var(--r-2xl) 0 0;
}
.kpi.suc::before { background: linear-gradient(90deg, var(--success),  var(--success-mid) 55%, rgba(5,150,105,.10)); }
.kpi.dng::before { background: linear-gradient(90deg, var(--danger),   var(--danger-mid)  55%, rgba(220,38,38,.10)); }
.kpi.wrn::before { background: linear-gradient(90deg, var(--warning),  var(--accent-warm) 55%, rgba(217,119,6,.10)); }
.kpi.pur::before { background: linear-gradient(90deg, var(--purple),   var(--purple-lt)   55%, rgba(124,58,237,.10)); }

/* Ambient corner glow */
.kpi-ring {
  position: absolute; bottom: -36px; right: -36px;
  width: 130px; height: 130px; border-radius: 50%;
  background: radial-gradient(circle, rgba(79,70,229,.07) 0%, transparent 72%);
  pointer-events: none;
}
.kpi.suc .kpi-ring { background: radial-gradient(circle, rgba(5,150,105,.08)  0%, transparent 72%); }
.kpi.dng .kpi-ring { background: radial-gradient(circle, rgba(220,38,38,.08)  0%, transparent 72%); }
.kpi.wrn .kpi-ring { background: radial-gradient(circle, rgba(217,119,6,.08)  0%, transparent 72%); }
.kpi.pur .kpi-ring { background: radial-gradient(circle, rgba(124,58,237,.08) 0%, transparent 72%); }

/* Icon */
.kpi-icon {
  width: 52px; height: 52px;
  border-radius: var(--r-lg);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.15rem; margin-bottom: 22px;
  background: var(--primary-light); color: var(--primary);
  box-shadow: 0 2px 14px rgba(79,70,229,.15), var(--sh-in);
  position: relative; z-index: 1;
}
.kpi.suc .kpi-icon { background: var(--success-bg); color: var(--success); box-shadow: 0 2px 14px rgba(5,150,105,.17), var(--sh-in); }
.kpi.dng .kpi-icon { background: var(--danger-bg);  color: var(--danger);  box-shadow: 0 2px 14px rgba(220,38,38,.17), var(--sh-in); }
.kpi.wrn .kpi-icon { background: var(--warning-bg); color: var(--warning); box-shadow: 0 2px 14px rgba(217,119,6,.17), var(--sh-in); }
.kpi.pur .kpi-icon { background: var(--purple-bg);  color: var(--purple);  box-shadow: 0 2px 14px rgba(124,58,237,.17), var(--sh-in); }

.kpi-lbl {
  font-size: .62rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: var(--text-3); margin-bottom: 8px;
  position: relative; z-index: 1;
}
.kpi-val {
  font-size: 2.8rem; font-weight: 800; line-height: .96;
  letter-spacing: -2.5px; margin-bottom: 10px; color: var(--text);
  font-variant-numeric: tabular-nums;
  position: relative; z-index: 1;
}
.kpi.suc .kpi-val { color: var(--success); }
.kpi.dng .kpi-val { color: var(--danger); }
.kpi.wrn .kpi-val { color: var(--warning); }
.kpi.pur .kpi-val { color: var(--purple); }
.kpi-sub { font-size: .74rem; color: var(--text-3); line-height: 1.45; position: relative; z-index: 1; }

/* ── PANELS ──────────────────────────────────────────────────── */
.panel {
  background: var(--surface);
  border-radius: var(--r-2xl);
  border: 1px solid var(--border);
  box-shadow: var(--sh), var(--sh-in);
  height: 100%; overflow: hidden;
  transition: box-shadow var(--t-base);
}
.panel:hover { box-shadow: var(--sh-md), var(--sh-in); }

.panel-hd {
  padding: 20px 26px 18px;
  display: flex; align-items: flex-start; gap: 14px;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(180deg, #FAFCFF 0%, #F5F8FF 100%);
}
.panel-hd .ph-icon {
  width: 44px; height: 44px; border-radius: var(--r-lg);
  display: flex; align-items: center; justify-content: center;
  font-size: .96rem; flex-shrink: 0; margin-top: 1px;
}
.panel-hd .ph-icon.blue   { background: var(--primary-light); color: var(--primary);  box-shadow: 0 2px 10px rgba(79,70,229,.15), var(--sh-in); }
.panel-hd .ph-icon.green  { background: var(--success-bg);    color: var(--success);  box-shadow: 0 2px 10px rgba(5,150,105,.15), var(--sh-in); }
.panel-hd .ph-icon.red    { background: var(--danger-bg);     color: var(--danger);   box-shadow: 0 2px 10px rgba(220,38,38,.15), var(--sh-in); }
.panel-hd .ph-icon.orange { background: var(--warning-bg);    color: var(--warning);  box-shadow: 0 2px 10px rgba(217,119,6,.15), var(--sh-in); }
.panel-hd .ph-icon.purple { background: var(--purple-bg);     color: var(--purple);   box-shadow: 0 2px 10px rgba(124,58,237,.15), var(--sh-in); }
.panel-hd .ph-icon.teal   { background: var(--teal-bg);       color: var(--teal);     box-shadow: 0 2px 10px rgba(8,145,178,.15), var(--sh-in); }
.ph-title { font-size: .96rem; font-weight: 800; color: var(--text); letter-spacing: -.3px; }
.ph-desc  { font-size: .74rem; color: var(--text-3); margin-top: 4px; line-height: 1.55; }
.panel-body { padding: 24px 26px 26px; }

/* ── EXECUTIVE SUMMARY LIST ──────────────────────────────────── */
.sum-list { list-style: none; }
.sum-list li {
  display: flex; align-items: flex-start; gap: 14px;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
  font-size: .875rem; line-height: 1.7; color: var(--text-2);
}
.sum-list li:last-child { border-bottom: none; padding-bottom: 0; }
.sum-list li strong { color: var(--text); font-weight: 700; }
.sum-ico {
  width: 34px; height: 34px; border-radius: var(--r-md);
  display: flex; align-items: center; justify-content: center;
  font-size: .82rem; flex-shrink: 0; margin-top: 2px;
  box-shadow: var(--sh-in);
}

/* ── PIPELINE — MINI STATS ───────────────────────────────────── */
.ms {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-2xl);
  padding: 22px 18px 20px;
  text-align: center;
  box-shadow: var(--sh), var(--sh-in);
  transition: box-shadow var(--t-base), transform var(--t-base);
  height: 100%;
  position: relative; overflow: hidden;
}
.ms::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--primary), var(--primary-lt));
  opacity: .38;
}
.ms:hover { box-shadow: var(--sh-md), var(--sh-in); transform: translateY(-4px); }
.ms-val {
  font-size: 2rem; font-weight: 800; color: var(--text);
  letter-spacing: -1.5px; line-height: 1;
  margin-bottom: 8px;
  font-variant-numeric: tabular-nums;
}
.ms-lbl {
  font-size: .62rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .1em;
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
  background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-3) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  padding: 20px 24px;
  display: flex; align-items: center; gap: 28px;
  box-shadow: var(--sh-in), var(--sh);
  margin-top: 4px;
}
.conf-label {
  font-size: .62rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: var(--text-3); margin-bottom: 11px;
}
.conf-track {
  flex: 1; height: 9px;
  background: var(--border); border-radius: var(--r-pill); overflow: hidden;
  box-shadow: inset 0 1px 4px rgba(0,0,0,.1);
}
.conf-fill {
  height: 100%; border-radius: var(--r-pill);
  background: linear-gradient(90deg,
    var(--success) 0%, var(--success-mid) 35%,
    var(--teal) 65%, var(--success) 100%);
  background-size: 300% 100%;
  animation: shimmerBar 4.5s linear infinite;
  transition: width 2s cubic-bezier(.22, 1, .36, 1);
}
@keyframes shimmerBar {
  0%   { background-position:  300% 0; }
  100% { background-position: -300% 0; }
}
.conf-score {
  font-size: 2rem; font-weight: 800; color: var(--success);
  white-space: nowrap; letter-spacing: -1.5px;
  font-variant-numeric: tabular-nums;
}

/* ── WORD CLOUD CONTAINERS ───────────────────────────────────── */
.wc-header {
  font-size: .61rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .1em; color: var(--text-3); margin-bottom: 10px;
}
.wc-box {
  height: 220px; border-radius: var(--r-xl);
  border: 1px solid var(--border);
  position: relative; overflow: hidden;
  background: linear-gradient(145deg, var(--primary-light), #DDE6FF);
  transition: box-shadow var(--t-base), transform var(--t-base);
  box-shadow: var(--sh);
}
.wc-box:hover { box-shadow: var(--sh-md); transform: scale(1.005); }
.wc-box.wc-pos { background: linear-gradient(145deg, #EDFCF5, #BBEFDA); border-color: rgba(5,150,105,.18); }
.wc-box.wc-neg { background: linear-gradient(145deg, #FFF4F4, #FFD4D4); border-color: rgba(220,38,38,.18); }
.wc-box.wc-out { background: var(--surface-2); border-color: var(--border); }
.wc-box img { width: 100%; height: 100%; object-fit: contain; border-radius: var(--r-xl); }

/* ── SCATTER IFRAMES ─────────────────────────────────────────── */
iframe.scatter-frame {
  width: 100%; height: 720px;
  border: none; border-radius: var(--r-xl);
  display: block;
  box-shadow: 0 2px 18px rgba(6,15,40,.07);
}

/* ── TOPIC CARDS ─────────────────────────────────────────────── */
.topic-col-hd {
  display: flex; align-items: center; gap: 9px;
  font-size: .82rem; font-weight: 800;
  padding: 11px 16px; border-radius: var(--r-xl); margin-bottom: 14px;
  box-shadow: var(--sh-in);
}
.topic-col-hd.pos {
  background: linear-gradient(135deg, var(--success-bg), #A7F3D0);
  color: #065F46; border: 1px solid rgba(5,150,105,.22);
}
.topic-col-hd.neg {
  background: linear-gradient(135deg, var(--danger-bg), #FECACA);
  color: #7F1D1D; border: 1px solid rgba(220,38,38,.22);
}
.tc {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-xl);
  padding: 17px 20px 17px 24px;
  margin-bottom: 13px;
  position: relative; overflow: hidden;
  box-shadow: var(--sh), var(--sh-in);
  transition: box-shadow var(--t-base), transform var(--t-base), border-color var(--t-base);
}
/* Left accent stripe with gradient */
.tc::before {
  content: '';
  position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
  background: linear-gradient(180deg, var(--primary), var(--primary-lt));
  border-radius: 5px 0 0 5px;
}
.tc.neg::before { background: linear-gradient(180deg, var(--danger), var(--danger-mid)); }
.tc:hover {
  box-shadow: var(--sh-md), var(--sh-in);
  transform: translateX(5px);
  border-color: rgba(79,70,229,.22);
}
.tc.neg:hover { border-color: rgba(220,38,38,.22); }
.tc-name  { font-size: .9rem; font-weight: 800; color: var(--text); margin-bottom: 4px; letter-spacing: -.25px; }
.tc-kw    { font-size: .72rem; color: var(--text-3); margin-bottom: 10px; line-height: 1.5; }
.tc-quote {
  font-size: .79rem; font-style: italic; color: var(--text-2);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 9px 13px 9px 18px;
  line-height: 1.6; margin-bottom: 10px;
  position: relative;
}
.tc-quote::before {
  content: '\201C';
  position: absolute; top: -1px; left: 6px;
  font-size: 2.2rem; color: var(--primary-light);
  font-family: Georgia, 'Times New Roman', serif;
  line-height: 1; pointer-events: none;
}
.tc-meta  {
  font-size: .7rem; font-weight: 600; color: var(--text-3);
  display: flex; align-items: center; gap: 5px;
}

/* ── SIMILARITY TABLE ────────────────────────────────────────── */
.sim-tbl { width: 100%; border-collapse: separate; border-spacing: 0; }
.sim-tbl thead tr {
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface-3) 100%);
}
.sim-tbl th {
  font-size: .61rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: .1em; color: var(--text-3);
  padding: 12px 16px;
  border-bottom: 2px solid var(--border-2);
  white-space: nowrap;
}
.sim-tbl th:first-child { border-radius: var(--r-md) 0 0 0; padding-left: 20px; }
.sim-tbl th:last-child  { border-radius: 0 var(--r-md) 0 0; padding-right: 20px; }
.sim-tbl td {
  padding: 13px 16px; border-bottom: 1px solid var(--border);
  font-size: .855rem; vertical-align: middle; color: var(--text-2);
  transition: background var(--t-fast);
}
.sim-tbl td:first-child { padding-left: 20px; }
.sim-tbl td:last-child  { padding-right: 20px; }
.sim-tbl tbody tr:last-child td { border-bottom: none; }
.sim-tbl tbody tr:hover td { background: var(--surface-2); }
.sim-tbl .td-rank {
  font-size: .7rem; font-weight: 800; color: var(--text-3);
  width: 38px; font-family: var(--mono);
}
.sim-tbl .td-text {
  max-width: 360px; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; color: var(--text); font-weight: 500;
}
.sim-row { display: flex; align-items: center; gap: 10px; }
.sim-track {
  flex: 1; height: 7px; background: var(--border);
  border-radius: var(--r-pill); overflow: hidden; min-width: 60px;
  box-shadow: inset 0 1px 3px rgba(0,0,0,.08);
}
.sim-fill {
  height: 100%; border-radius: var(--r-pill);
  background: linear-gradient(90deg, var(--primary), var(--primary-lt));
}
.sim-score {
  font-size: .78rem; font-weight: 700; color: var(--primary);
  white-space: nowrap; font-family: var(--mono);
  min-width: 38px; text-align: right;
}
.badge-sent {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 11px; border-radius: var(--r-pill);
  font-size: .67rem; font-weight: 700; white-space: nowrap;
  box-shadow: var(--sh-in);
}
.badge-sent.pos { background: var(--success-bg); color: #065F46; border: 1px solid rgba(5,150,105,.22); }
.badge-sent.neg { background: var(--danger-bg);  color: #7F1D1D; border: 1px solid rgba(220,38,38,.22); }
.badge-sent .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.badge-sent.pos .dot { background: var(--success); }
.badge-sent.neg .dot { background: var(--danger); }

/* ── CLI PARAMS ──────────────────────────────────────────────── */
.cli-toggle {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 12px 18px;
  cursor: pointer; display: flex; align-items: center; gap: 10px;
  font-size: .79rem; font-weight: 600; color: var(--text-3);
  transition: all var(--t-base); width: 100%; text-align: left;
  margin-bottom: 3px; box-shadow: var(--sh-in);
}
.cli-toggle:hover {
  background: var(--primary-light); color: var(--primary);
  border-color: rgba(79,70,229,.25);
  box-shadow: 0 0 0 3px var(--primary-glow), var(--sh-in);
}
.cli-panel {
  display: none; background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: 14px 18px;
}
.cli-panel.open { display: block; }
.cli-tbl { font-size: .76rem; width: 100%; }
.cli-tbl tr td { padding: 4px 8px; }
.cli-tbl td:first-child { font-weight: 600; color: var(--text-3); white-space: nowrap; padding-right: 18px; }
.cli-tbl td:last-child  { font-family: var(--mono); font-size: .71rem; color: var(--text-2); }

/* ── FALLBACK ALERT ──────────────────────────────────────────── */
.alert-info {
  background: linear-gradient(135deg, var(--primary-light), #E4EAFF);
  border: 1px solid rgba(79,70,229,.18);
  border-radius: var(--r-lg);
  padding: 14px 18px;
  font-size: .86rem; color: var(--text-2);
  font-weight: 500;
  box-shadow: var(--sh-in);
}

/* ── REPORT FOOTER ───────────────────────────────────────────── */
.rpt-footer {
  padding: 20px 36px;
  border-top: 1px solid var(--border);
  background: linear-gradient(180deg, var(--surface) 0%, var(--surface-2) 100%);
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px;
  font-size: .74rem; color: var(--text-3);
  box-shadow: 0 -1px 0 var(--border), 0 -4px 24px rgba(6,15,40,.04);
}
.rpt-footer strong { color: var(--text-2); font-weight: 700; }
.rpt-footer-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 16px; border-radius: var(--r-pill);
  background: var(--primary-light); color: var(--primary);
  font-size: .68rem; font-weight: 700;
  border: 1px solid rgba(79,70,229,.16);
  box-shadow: 0 1px 8px rgba(79,70,229,.12), var(--sh-in);
}

/* ── SCROLL-DRIVEN ENTRANCE ANIMATIONS ──────────────────────── */
/* Start hidden; JS IntersectionObserver adds .in-view */
.a1, .a2, .a3, .a4, .a5, .a6, .a7, .a8 {
  opacity: 0;
  transform: translateY(22px);
  transition: opacity .55s var(--ease-out), transform .55s var(--ease-out);
  will-change: opacity, transform;
}
.a1.in-view { opacity: 1; transform: none; transition-delay: .04s; }
.a2.in-view { opacity: 1; transform: none; transition-delay: .09s; }
.a3.in-view { opacity: 1; transform: none; transition-delay: .14s; }
.a4.in-view { opacity: 1; transform: none; transition-delay: .19s; }
.a5.in-view { opacity: 1; transform: none; transition-delay: .24s; }
.a6.in-view { opacity: 1; transform: none; transition-delay: .29s; }
.a7.in-view { opacity: 1; transform: none; transition-delay: .34s; }
.a8.in-view { opacity: 1; transform: none; transition-delay: .39s; }

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
  #sidebar, #topbar, #readProg { display: none !important; }
  #main { margin-left: 0 !important; }
  .kpi, .panel, .tc, .ms { box-shadow: none !important; break-inside: avoid; }
  .mb-sec { margin-bottom: 24px; }
  body { background: #fff !important; background-image: none !important; }
  .a1,.a2,.a3,.a4,.a5,.a6,.a7,.a8 { opacity: 1 !important; transform: none !important; }
}
</style>
<script>{{ plotly_js }}</script>
</head>
<body>

<!-- ══ Reading progress bar ══ -->
<div id="readProg" aria-hidden="true"></div>

<!-- ═══════════════════════════════════════════════════════════
     SIDEBAR
═══════════════════════════════════════════════════════════ -->
<nav id="sidebar">
  <a href="#overview" class="sb-brand">
    <div class="sb-logo">B</div>
    <div class="sb-names">
      <div class="sb-name">Beternovik</div>
      <div class="sb-tag">NLP Analysis Report</div>
    </div>
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

  <div class="sb-infobox">
    <div class="info-title"><i class="bi bi-file-earmark-bar-graph"></i> Report Details</div>
    <div class="info-row"><span>Generated</span><span>{{ data.timestamp }}</span></div>
    <div class="info-row"><span>Min Cluster</span><span>{{ data.min_partition_size }}</span></div>
    {% if data.concept_name %}
    <div class="info-row"><span>Concept</span><span>{{ data.concept_name }}</span></div>
    {% endif %}
  </div>

  <div class="sb-footer">
    <div class="sb-user">
      <div class="sb-avatar">TL</div>
      <div>
        <div class="sb-uname">Beternovik</div>
        <div class="sb-urole">NLP Pipeline v2.4</div>
      </div>
    </div>
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
      <div class="tb-chip">
        <span class="pulse-dot"></span>
        {{ data.timestamp }}
      </div>
      <button class="tb-icon-btn" onclick="window.print()" title="Print report">
        <i class="bi bi-printer"></i>
      </button>
    </div>
  </header>

  <div id="content">

    <!-- ╔══════════════════════════════════════╗
         §1  OVERVIEW
         ╚══════════════════════════════════════╝ -->
    <section id="overview" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-grid-1x2-fill"></i> Overview</div>
        <h2>Key Metrics at a Glance</h2>
        <p>Top-level numbers from the analysis of <strong>{{ data.total_rows }}</strong> comments across all pipeline stages.</p>
      </div>
      <div class="row g-3">
        <div class="col-6 col-xl-3 a1">
          <div class="kpi">
            <div class="kpi-ring"></div>
            <div class="kpi-icon"><i class="bi bi-chat-text-fill"></i></div>
            <div class="kpi-lbl">Total Comments</div>
            <div class="kpi-val">{{ data.total_rows }}</div>
            <div class="kpi-sub">All records processed</div>
          </div>
        </div>
        <div class="col-6 col-xl-3 a2">
          <div class="kpi suc">
            <div class="kpi-ring"></div>
            <div class="kpi-icon"><i class="bi bi-emoji-smile-fill"></i></div>
            <div class="kpi-lbl">Positive</div>
            <div class="kpi-val">{{ "%.1f"|format(data.pos_pct * 100) }}%</div>
            <div class="kpi-sub">{{ data.pos_count }} comments</div>
          </div>
        </div>
        <div class="col-6 col-xl-3 a3">
          <div class="kpi dng">
            <div class="kpi-ring"></div>
            <div class="kpi-icon"><i class="bi bi-emoji-frown-fill"></i></div>
            <div class="kpi-lbl">Negative</div>
            <div class="kpi-val">{{ "%.1f"|format(data.neg_pct * 100) }}%</div>
            <div class="kpi-sub">{{ data.neg_count }} comments</div>
          </div>
        </div>
        <div class="col-6 col-xl-3 a4">
          <div class="kpi wrn">
            <div class="kpi-ring"></div>
            <div class="kpi-icon"><i class="bi bi-exclamation-triangle-fill"></i></div>
            <div class="kpi-lbl">Outliers Removed</div>
            <div class="kpi-val">{{ data.outliers_count }}</div>
            <div class="kpi-sub">Filtered before analysis</div>
          </div>
        </div>
      </div>
    </section>

    <!-- ╔══════════════════════════════════════╗
         §2  EXECUTIVE SUMMARY
         ╚══════════════════════════════════════╝ -->
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
            <li class="a1">
              <div class="sum-ico" style="background:var(--primary-light);color:var(--primary)"><i class="bi bi-chat-text-fill"></i></div>
              <span>Analyzed <strong>{{ data.total_rows }}</strong> comments in total.</span>
            </li>
            <li class="a2">
              <div class="sum-ico" style="background:var(--success-bg);color:var(--success)"><i class="bi bi-activity"></i></div>
              <span>Sentiment split: <strong>{{ "%.1f"|format(data.pos_pct * 100) }}% positive</strong> ({{ data.pos_count }} comments), <strong>{{ "%.1f"|format(data.neg_pct * 100) }}% negative</strong> ({{ data.neg_count }} comments).</span>
            </li>
            <li class="a3">
              <div class="sum-ico" style="background:var(--warning-bg);color:var(--warning)"><i class="bi bi-exclamation-triangle-fill"></i></div>
              <span>Detected <strong>{{ data.outliers_count }}</strong> outlier(s) — removed before downstream analysis.</span>
            </li>
            {% if data.top_positive_topic %}
            <li class="a4">
              <div class="sum-ico" style="background:var(--success-bg);color:var(--success)"><i class="bi bi-tag-fill"></i></div>
              <span>Top positive topic: <strong>{{ data.top_positive_topic }}</strong>.</span>
            </li>
            {% endif %}
            {% if data.top_negative_topic %}
            <li class="a5">
              <div class="sum-ico" style="background:var(--danger-bg);color:var(--danger)"><i class="bi bi-tag-fill"></i></div>
              <span>Top negative topic: <strong>{{ data.top_negative_topic }}</strong>.</span>
            </li>
            {% endif %}
            {% if data.top_concept_score > 0 %}
            <li class="a6">
              <div class="sum-ico" style="background:var(--purple-bg);color:var(--purple)"><i class="bi bi-vector-pen"></i></div>
              <span>Highest semantic similarity to the target concept: <strong>{{ "%.3f"|format(data.top_concept_score) }}</strong>.</span>
            </li>
            {% endif %}
            <li class="a7">
              <div class="sum-ico" style="background:var(--primary-light);color:var(--primary)"><i class="bi bi-shield-check-fill"></i></div>
              <span>Confidence rate: <strong>{{ "%.1f"|format(data.confidence_rate * 100) }}%</strong>.</span>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <!-- ╔══════════════════════════════════════╗
         §3  PIPELINE SUMMARY
         ╚══════════════════════════════════════╝ -->
    <section id="pipeline" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-diagram-3-fill"></i> Pipeline Summary</div>
        <h2>Data Processing Overview</h2>
        <p>Overview of the dataset after preprocessing, outlier removal, and sentiment classification.</p>
      </div>
      <div class="row g-3 mb-3">
        <div class="col-6 col-md-3 a1"><div class="ms"><div class="ms-val">{{ data.total_rows }}</div><div class="ms-lbl">Total Comments</div></div></div>
        <div class="col-6 col-md-3 a2"><div class="ms"><div class="ms-val" style="color:var(--warning)">{{ data.outliers_count }}</div><div class="ms-lbl">Outliers Removed</div></div></div>
        <div class="col-6 col-md-3 a3"><div class="ms"><div class="ms-val" style="color:var(--success)">{{ data.pos_count }}</div><div class="ms-lbl">Positive</div></div></div>
        <div class="col-6 col-md-3 a4"><div class="ms"><div class="ms-val" style="color:var(--danger)">{{ data.neg_count }}</div><div class="ms-lbl">Negative</div></div></div>
      </div>
      <div class="panel a5">
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
                <div class="conf-fill" id="confFill" style="width:0%"></div>
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

    <!-- ╔══════════════════════════════════════╗
         §4  OUTLIER ANALYSIS
         ╚══════════════════════════════════════╝ -->
    <section id="outliers" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-exclamation-triangle-fill"></i> Outlier Analysis</div>
        <h2>Flagged Comments</h2>
        <p>Comments flagged as statistically unusual. The word cloud highlights frequent terms; n-gram charts reveal multi-word patterns among outliers.</p>
      </div>
      <div class="row g-3 mb-3">
        <div class="col-12 a1">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon orange"><i class="bi bi-cloud-fill"></i></div>
              <div>
                <div class="ph-title">Word Cloud — Outliers</div>
                <div class="ph-desc">Most frequent terms in {{ data.outliers_count }} flagged comments</div>
              </div>
            </div>
            <div class="panel-body">
              <div class="wc-box wc-out" style="height:280px;">
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
        <div class="col-md-4 a2">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon blue"><i class="bi bi-bar-chart-horizontal-fill"></i></div>
              <div>
                <div class="ph-title">Unigrams</div>
                <div class="ph-desc">Single-word frequency</div>
              </div>
            </div>
            <div class="panel-body">{{ data.ngrams_unigrams_html | safe }}</div>
          </div>
        </div>
        <div class="col-md-4 a3">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon teal"><i class="bi bi-bar-chart-horizontal-fill"></i></div>
              <div>
                <div class="ph-title">Bigrams</div>
                <div class="ph-desc">Two-word patterns</div>
              </div>
            </div>
            <div class="panel-body">{{ data.ngrams_bigrams_html | safe }}</div>
          </div>
        </div>
        <div class="col-md-4 a4">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon purple"><i class="bi bi-bar-chart-horizontal-fill"></i></div>
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

    <!-- ╔══════════════════════════════════════╗
         §5  SENTIMENT ANALYSIS
         ╚══════════════════════════════════════╝ -->
    <section id="sentiment" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-bar-chart-fill"></i> Sentiment Analysis</div>
        <h2>Comment Sentiment Distribution</h2>
        <p>Each comment was classified as positive or negative. The word clouds show the most frequent terms for each sentiment group.</p>
      </div>
      <div class="row g-3 mb-3">
        <div class="col-12 a1">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon blue"><i class="bi bi-pie-chart-fill"></i></div>
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
        <div class="col-md-6 a2">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon green"><i class="bi bi-cloud-fill"></i></div>
              <div>
                <div class="ph-title">Word Cloud — Positive</div>
                <div class="ph-desc">{{ data.pos_count }} comments</div>
              </div>
            </div>
            <div class="panel-body">
              <div class="wc-box wc-pos" style="height:260px;">
                {% if data.wc_positive_b64 %}
                <img src="{{ data.wc_positive_b64 }}" alt="Positive Word Cloud">
                {% else %}
                <div style="padding:80px 20px;text-align:center;color:var(--text-3);font-size:.85rem;">No positive comments</div>
                {% endif %}
              </div>
            </div>
          </div>
        </div>
        <div class="col-md-6 a3">
          <div class="panel h-100">
            <div class="panel-hd">
              <div class="ph-icon red"><i class="bi bi-cloud-fill"></i></div>
              <div>
                <div class="ph-title">Word Cloud — Negative</div>
                <div class="ph-desc">{{ data.neg_count }} comments</div>
              </div>
            </div>
            <div class="panel-body">
              <div class="wc-box wc-neg" style="height:260px;">
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

    <!-- ╔══════════════════════════════════════╗
         §6  TOPIC MODELING
         ╚══════════════════════════════════════╝ -->
    <section id="topics" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-diagram-2-fill"></i> Topic Modeling</div>
        <h2>Topic Modeling Overview</h2>
        <p>Topics automatically extracted from positive and negative comments. Each point is a comment; color and shape encode the cluster.</p>
      </div>
      <div class="row g-3">
        <div class="col-12 a1">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon blue"><i class="bi bi-diagram-3-fill"></i></div>
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
        <div class="col-md-6 a2">
          <div class="topic-col-hd pos"><i class="bi bi-emoji-smile-fill"></i> Positive Topics</div>
          {% for t in data.pos_topic_summaries %}
          <div class="tc a{{ loop.index }}">
            <div class="tc-name">{{ t.label }}</div>
            <div class="tc-kw">{{ t.keywords }}</div>
            {% if t.representative %}<div class="tc-quote">&ldquo;{{ t.representative }}&rdquo;</div>{% endif %}
            <div class="tc-meta"><i class="bi bi-chat-fill"></i> {{ t.count }} comments</div>
          </div>
          {% endfor %}
        </div>
        <div class="col-md-6 a3">
          <div class="topic-col-hd neg"><i class="bi bi-emoji-frown-fill"></i> Negative Topics</div>
          {% for t in data.neg_topic_summaries %}
          <div class="tc neg a{{ loop.index }}">
            <div class="tc-name">{{ t.label }}</div>
            <div class="tc-kw">{{ t.keywords }}</div>
            {% if t.representative %}<div class="tc-quote">&ldquo;{{ t.representative }}&rdquo;</div>{% endif %}
            <div class="tc-meta"><i class="bi bi-chat-fill"></i> {{ t.count }} comments</div>
          </div>
          {% endfor %}
        </div>
        {% endif %}
      </div>
    </section>

    <!-- ╔══════════════════════════════════════╗
         §7  SEMANTIC CONCEPT ANALYSIS
         ╚══════════════════════════════════════╝ -->
    <section id="semantic" class="mb-sec">
      <div class="sec-hd">
        <div class="sec-tag"><i class="bi bi-broadcast-pin"></i> Semantic Analysis</div>
        <h2>Semantic Concept Analysis</h2>
        <p>Each comment is scored by cosine similarity to the target concept{% if data.concept_name %} <strong>&ldquo;{{ data.concept_name }}&rdquo;</strong>{% endif %}. Brighter / higher points are more semantically related.</p>
      </div>
      <div class="row g-3">
        <div class="col-12 a1">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon purple"><i class="bi bi-broadcast-pin"></i></div>
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
        <div class="col-12 a2">
          <div class="panel">
            <div class="panel-hd">
              <div class="ph-icon teal"><i class="bi bi-trophy-fill"></i></div>
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
    <span>Developed by <strong>Beternovik</strong> &mdash; Fully self-contained offline report.</span>
    <span class="rpt-footer-badge"><i class="bi bi-cpu"></i> NLP Pipeline v2.4</span>
  </footer>

</div><!-- /main -->

<script>
/* ══ Confidence bar — delayed animate-in ══ */
setTimeout(() => {
  const f = document.getElementById('confFill');
  if (f) f.style.width = '{{ "%.1f"|format(data.confidence_rate * 100) }}%';
}, 600);

/* ══ Reading progress bar ══ */
(function () {
  const bar = document.getElementById('readProg');
  if (!bar) return;
  const update = () => {
    const h = document.documentElement;
    const scrollable = Math.max(h.scrollHeight - h.clientHeight, 1);
    bar.style.width = ((window.scrollY / scrollable) * 100).toFixed(1) + '%';
  };
  window.addEventListener('scroll', update, { passive: true });
  update();
})();

/* ══ Scroll-triggered entrance animations ══ */
(function () {
  const elems = document.querySelectorAll('.a1,.a2,.a3,.a4,.a5,.a6,.a7,.a8');
  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('in-view');
          obs.unobserve(e.target);
        }
      });
    }, { rootMargin: '0px 0px -48px 0px', threshold: 0.05 });
    elems.forEach(el => obs.observe(el));
  } else {
    elems.forEach(el => el.classList.add('in-view'));
  }
})();

/* ══ Active sidebar link on scroll ══ */
const sections = document.querySelectorAll('section[id]');
const links    = document.querySelectorAll('.sb-a[href^="#"]');
const secObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(l => l.classList.remove('active'));
      const a = document.querySelector('.sb-a[href="#' + e.target.id + '"]');
      if (a) a.classList.add('active');
    }
  });
}, { rootMargin: '-15% 0px -72% 0px', threshold: 0 });
sections.forEach(s => secObs.observe(s));

/* ══ CLI params toggle ══ */
function toggleCli() {
  const p = document.getElementById('cliPanel');
  if (p) p.classList.toggle('open');
}

/* ══ Close sidebar on outside click (mobile) ══ */
document.addEventListener('click', e => {
  const sb = document.getElementById('sidebar');
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