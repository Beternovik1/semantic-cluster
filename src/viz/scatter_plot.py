"""Interactive Bokeh scatter plots for topic distribution and semantic similarity.

Generates fully self-contained offline HTML files using ``INLINE``
resources.  Down-samples large datasets while preserving representative
documents (topic centroids).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.embed import file_html
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    Legend,
)
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.transform import factor_cmap, factor_mark

from src.utils.validators import MAX_SCATTER_POINTS

logger = logging.getLogger(__name__)

# Shared marker and colour defaults
FILL_ALPHA = 0.6
LINE_ALPHA = 0.8

NOISE_COLOR = "#888888"
NOISE_SHAPE = "circle"

CENTROID_SHAPE = "star"
CENTROID_SIZE = 10
NORMAL_SIZE = 7


class ScatterPlot:
    """Bokeh scatter-plot factory for topic and semantic views.

    Args:
        palette_manager: A ``PaletteManager`` instance that provides
            colours, background, and marker shapes.
    """

    def __init__(self, palette_manager) -> None:
        self._pm = palette_manager

    # ── Public orchestration (called from main.py) ─────────────────

    def generate(
        self,
        df_pos: pd.DataFrame,
        df_neg: pd.DataFrame,
        concept: str,
        output_dir: Path,
    ) -> tuple[str | None, str]:
        """Generate both scatter plots and return their HTML strings.

        Args:
            df_pos: Partition with positive sentiment (may have
                ``umap_x`` / ``umap_y`` and ``concept_similarity``
                columns).
            df_neg: Partition with negative sentiment.
            concept: The concept string used for the semantic plot.
            output_dir: Directory to save HTML files.

        Returns:
            Tuple of ``(scatter_topics_html, scatter_semantic_html)``.
            The topic scatter may be ``None`` when all ``umap_x``
            values are null (fallback path).
        """
        topic_html = self.generate_topic_scatter(
            df_pos=df_pos,
            df_neg=df_neg,
            title="Topic Distribution — Positive vs Negative",
            output_dir=output_dir,
        )
        combined = self._combine_partitions(df_pos, df_neg)
        semantic_html = self.generate_semantic_scatter(
            df=combined,
            title=f"Semantic Similarity — \"{concept}\"",
            concept_name=concept,
            output_dir=output_dir,
        )
        return topic_html, semantic_html

    # ── Topic scatter ──────────────────────────────────────────────

    def generate_topic_scatter(
        self,
        df_pos: pd.DataFrame,
        df_neg: pd.DataFrame,
        title: str,
        output_dir: Path,
    ) -> str | None:
        """Generate a topic-coloured scatter of positive *and* negative docs.

        Returns:
            HTML string, or ``None`` when UMAP coordinates are absent
            (word-frequency fallback path).
        """
        df = self._combine_partitions(df_pos, df_neg)

        if df["umap_x"].isna().all():
            logger.warning(
                "umap_x is all null — topic modeling used word-frequency "
                "fallback. Skipping topic scatter."
            )
            return None

        df = self._downsample_preserving_representatives(df)
        palette = self._pm.get_bokeh_palette()
        shapes = self._pm.get_shapes()
        background = self._pm.get_background()

        # Map topic_id → colour and shape
        topic_ids = sorted(
            tid for tid in df["topic_id"].unique() if tid != -1
        )
        n_topics = len(topic_ids)

        colour_map = {
            tid: palette[i % len(palette)] for i, tid in enumerate(topic_ids)
        }
        shape_map = {
            tid: shapes[i % len(shapes)] for i, tid in enumerate(topic_ids)
        }
        colour_map[-1] = NOISE_COLOR
        shape_map[-1] = NOISE_SHAPE

        # Add marker columns
        df["_colour"] = df["topic_id"].map(colour_map)
        df["_shape"] = df["topic_id"].map(shape_map)
        df["_size"] = np.where(df["representative_doc"], CENTROID_SIZE, NORMAL_SIZE)

        source = ColumnDataSource(df)

        p = figure(
            title=title,
            background_fill_color=background,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            sizing_mode="stretch_width",
            height=600,
        )

        # Render topic groups for legend
        renderers: dict[int, object] = {}
        for tid in sorted(df["topic_id"].unique()):
            mask = df["topic_id"] == tid
            sub = df[mask]
            sub_source = ColumnDataSource(sub)
            r = p.scatter(
                x="umap_x",
                y="umap_y",
                source=sub_source,
                color=colour_map[tid],
                marker=shape_map[tid],
                size="_size",
                fill_alpha=FILL_ALPHA,
                line_alpha=LINE_ALPHA,
                legend_label=f"Topic {tid}" if tid != -1 else "Noise (-1)",
            )
            renderers[tid] = r

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        # Hover tooltip
        hover = HoverTool(
            tooltips="""
            <div style="max-width:300px; word-wrap:break-word;">
              <b>Topic:</b> @topic_label<br>
              <b>Sentiment:</b> @sentiment<br>
              <b>Polarity:</b> @polarity<br>
              <b>Text:</b> @clean_text
            </div>
            """
        )
        p.add_tools(hover)

        # Axis labels
        p.xaxis.axis_label = "UMAP 1"
        p.yaxis.axis_label = "UMAP 2"

        html = file_html(p, INLINE, title)
        out_path = output_dir / "scatter_topics.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Saved topic scatter to %s", out_path)

        return html

    # ── Semantic scatter ───────────────────────────────────────────

    def generate_semantic_scatter(
        self,
        df: pd.DataFrame,
        title: str,
        concept_name: str,
        output_dir: Path,
    ) -> str | None:
        """Generate a similarity-coloured scatter against a concept.

        Returns:
            HTML string, or ``None`` when UMAP coordinates are absent.
        """
        if df["umap_x"].isna().all():
            logger.warning(
                "umap_x is all null — skipping semantic scatter."
            )
            return None

        df = self._downsample(df)
        palette = self._pm.get_bokeh_palette(256)
        background = self._pm.get_background()

        color_mapper = LinearColorMapper(
            palette=palette,
            low=df["concept_similarity"].min(),
            high=df["concept_similarity"].max(),
        )

        source = ColumnDataSource(df)

        p = figure(
            title=title,
            background_fill_color=background,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            sizing_mode="stretch_width",
            height=600,
        )

        p.scatter(
            x="umap_x",
            y="umap_y",
            source=source,
            color={"field": "concept_similarity", "transform": color_mapper},
            marker="circle",
            size=NORMAL_SIZE,
            fill_alpha=FILL_ALPHA,
            line_alpha=LINE_ALPHA,
        )

        hover = HoverTool(
            tooltips="""
            <div style="max-width:300px; word-wrap:break-word;">
              <b>Similarity:</b> @concept_similarity{0.000}<br>
              <b>Sentiment:</b> @sentiment<br>
              <b>Text:</b> @clean_text
            </div>
            """
        )
        p.add_tools(hover)

        p.xaxis.axis_label = "UMAP 1"
        p.yaxis.axis_label = "UMAP 2"

        html = file_html(p, INLINE, title)
        out_path = output_dir / "scatter_semantic.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("Saved semantic scatter to %s", out_path)

        return html

    # ── Down-sampling ──────────────────────────────────────────────

    @staticmethod
    def _downsample(df: pd.DataFrame) -> pd.DataFrame:
        """Randomly sample at most ``MAX_SCATTER_POINTS`` rows."""
        if len(df) <= MAX_SCATTER_POINTS:
            return df
        logger.info(
            "Down-sampling scatter from %d to %d points.",
            len(df),
            MAX_SCATTER_POINTS,
        )
        return df.sample(n=MAX_SCATTER_POINTS, random_state=42).reset_index(
            drop=True
        )

    @staticmethod
    def _downsample_preserving_representatives(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Down-sample but keep all ``representative_doc`` rows."""
        if len(df) <= MAX_SCATTER_POINTS:
            return df

        rep = df[df["representative_doc"]]
        non_rep = df[~df["representative_doc"]]

        remaining = MAX_SCATTER_POINTS - len(rep)
        if remaining <= 0:
            logger.info(
                "Representative docs (%d) already exceed limit. "
                "Truncating to %d.",
                len(rep),
                MAX_SCATTER_POINTS,
            )
            return rep.head(MAX_SCATTER_POINTS).reset_index(drop=True)

        sampled = non_rep.sample(
            n=remaining, random_state=42
        ).reset_index(drop=True)

        result = pd.concat([rep, sampled], ignore_index=True)
        logger.info(
            "Down-sampling scatter from %d to %d "
            "(preserving %d representative docs).",
            len(df),
            len(result),
            len(rep),
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _combine_partitions(
        df_pos: pd.DataFrame, df_neg: pd.DataFrame
    ) -> pd.DataFrame:
        """Concatenate positive and negative partitions."""
        return pd.concat(
            [df_pos, df_neg], ignore_index=True
        )
