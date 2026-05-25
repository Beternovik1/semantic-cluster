"""
TextLens — Modular NLP Analysis Pipeline

Sets HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE before any
other import to prevent HuggingFace hub pings at runtime.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="textlens",
        description=(
            "TextLens: Modular NLP pipeline for text analysis and "
            "interactive visualization. Runs 100%% offline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --file reviews.csv --column text "
            "--lang es --title \"My Report\" --palette viridis\n"
            "  python main.py --file jobs.csv --column description "
            "--lang en --title \"Job Market\" --palette cividis\n"
        ),
    )

    parser.add_argument(
        "--file", "-f",
        type=Path,
        required=True,
        help="Path to the CSV file containing the data.",
    )
    parser.add_argument(
        "--column", "-c",
        type=str,
        required=True,
        help="Name of the column that contains the text to analyze.",
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        required=True,
        choices=["es", "en", "fr", "de", "pt"],
        help="Language of the text (ISO 639-1 code).",
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        required=True,
        help="Title for the generated report and visualizations.",
    )
    parser.add_argument(
        "--palette", "-p",
        type=str,
        required=True,
        choices=["viridis", "cividis", "plasma", "inferno", "magma"],
        help="Color palette for plots. All options are colorblind-safe.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where outputs will be saved. Default: outputs/",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="precio valor costo",
        help=(
            "Synthetic concept for semantic similarity analysis. "
            "Default: 'precio valor costo'"
        ),
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected proportion of outliers (0.01-0.15). Default: 0.05",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(args: argparse.Namespace) -> None:
    logger.info("Initializing TextLens: %s", args.title)

    # Shared model instances — loaded once, injected into modules
    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_model.max_seq_length = 512
    logger.info(
        "Embedding model loaded. max_seq_length overridden to 512."
    )

    pysentimiento_analyzer = None
    if args.lang != "en":
        from pysentimiento import create_analyzer
        pysentimiento_analyzer = create_analyzer(
            task="sentiment", lang=args.lang
        )
        logger.info(
            "pysentimiento analyzer loaded for lang=%s.", args.lang
        )

    # Imports stay inside run() so --help works before modules built
    from src.utils.validators import validate_args
    from src.utils.language_config import LanguageConfig
    from src.utils.color_palettes import PaletteManager
    from src.pipeline.loader import DataLoader
    from src.pipeline.preprocessor import Preprocessor
    from src.pipeline.outlier_detector import OutlierDetector
    from src.pipeline.sentiment_analyzer import SentimentAnalyzer
    from src.pipeline.topic_modeler import TopicModeler
    from src.pipeline.semantic_search import SemanticSearch
    from src.viz.scatter_plot import ScatterPlot
    from src.viz.wordcloud_gen import WordCloudGenerator
    from src.viz.report_builder import ReportBuilder

    # ── 0. Validate inputs, load config ───────────────────────────
    validate_args(args)
    lang_cfg = LanguageConfig(args.lang)
    palette = PaletteManager(args.palette)

    # Clean output directory (preserve embeddings cache)
    if args.output.exists():
        for p in args.output.iterdir():
            if p.name not in ("embeddings.npy", "cache_hash.txt"):
                p.unlink()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────
    logger.info("[1/6] Loading data from %s ...", args.file)
    loader = DataLoader(args.file, args.column)
    df = loader.load()

    # ── 2. Preprocess ─────────────────────────────────────────────
    logger.info("[2/6] Preprocessing text ...")
    preprocessor = Preprocessor(lang_cfg)
    df = preprocessor.run(df)

    # ── 3. Outlier detection ──────────────────────────────────────
    logger.info("[3/6] Detecting outliers ...")
    detector = OutlierDetector(
        embedding_model=embedding_model,
        contamination=args.contamination,
    )
    df, outliers_df, embeddings_clean = detector.run(df)

    # ── 4. Sentiment analysis ─────────────────────────────────────
    logger.info("[4/6] Analyzing sentiment ...")
    analyzer = SentimentAnalyzer(
        lang_cfg=lang_cfg,
        pysentimiento_analyzer=pysentimiento_analyzer,
    )
    df = analyzer.run(df)

    # Split into positive and negative partitions
    pos_mask = df["sentiment"] == "pos"
    neg_mask = df["sentiment"] == "neg"
    df_pos = df[pos_mask].reset_index(drop=True)
    df_neg = df[neg_mask].reset_index(drop=True)
    logger.info(
        "Positive: %d, Negative: %d", len(df_pos), len(df_neg)
    )

    # ── 5. Topic modeling ─────────────────────────────────────────
    logger.info("[5/6] Modeling topics ...")
    modeler = TopicModeler(
        embedding_model=embedding_model,
        lang_cfg=lang_cfg,
    )
    df_pos, pos_topic_keywords, pos_used_bertopic = modeler.run(
        df_pos, embeddings_clean[pos_mask]
    )
    df_neg, neg_topic_keywords, neg_used_bertopic = modeler.run(
        df_neg, embeddings_clean[neg_mask]
    )

    # ── 6. Semantic search ────────────────────────────────────────
    logger.info("[6/6] Running semantic search ...")
    searcher = SemanticSearch(
        embedding_model=embedding_model,
        concept=args.concept,
    )
    df, _top5_df = searcher.run(df, embeddings_clean)

    # Merge concept_similarity into topic-modeled partitions
    # (df_pos/df_neg were split from df preserving row order)
    pos_mask = df["sentiment"] == "pos"
    neg_mask = df["sentiment"] == "neg"
    df_pos["concept_similarity"] = df.loc[pos_mask, "concept_similarity"].values
    df_neg["concept_similarity"] = df.loc[neg_mask, "concept_similarity"].values

    # ── Visualizations ────────────────────────────────────────────
    logger.info("[VIZ] Generating scatter plots ...")
    scatter = ScatterPlot(palette)
    scatter_topics_html, scatter_semantic_html = scatter.generate(
        df_pos=df_pos,
        df_neg=df_neg,
        concept=args.concept,
        output_dir=args.output,
    )

    logger.info("[VIZ] Generating word clouds and n-grams ...")
    wc = WordCloudGenerator(palette, lang_cfg)
    outliers_viz = wc.from_outliers(outliers_df, args.output)
    wc_positive_b64 = wc.from_partition(df_pos, "positive", args.output)
    wc_negative_b64 = wc.from_partition(df_neg, "negative", args.output)

    logger.info("[VIZ] Building HTML report ...")
    builder = ReportBuilder(
        title=args.title,
        palette=palette,
    )
    pos_fallback_keywords = (
        None
        if pos_used_bertopic
        else pos_topic_keywords.get(-1, "")
    )
    neg_fallback_keywords = (
        None
        if neg_used_bertopic
        else neg_topic_keywords.get(-1, "")
    )
    report_path = builder.build(
        df=df,
        outliers_df=outliers_df,
        scatter_topics_html=scatter_topics_html,
        scatter_semantic_html=scatter_semantic_html,
        wc_outliers_b64=outliers_viz["wordcloud_b64"],
        wc_positive_b64=wc_positive_b64,
        wc_negative_b64=wc_negative_b64,
        ngrams_unigrams_html=outliers_viz["unigrams_html"],
        ngrams_bigrams_html=outliers_viz["bigrams_html"],
        ngrams_trigrams_html=outliers_viz["trigrams_html"],
        pos_fallback_keywords=pos_fallback_keywords,
        neg_fallback_keywords=neg_fallback_keywords,
        output_dir=args.output,
    )

    logger.info("Pipeline complete. Report: %s", report_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger.debug("CLI arguments: %s", args)
    run(args)


if __name__ == "__main__":
    main()
