# Semantic Cluster

A fully offline, end-to-end NLP pipeline for analyzing textual data (reviews, job postings, product feedback) using topic modeling, sentiment analysis, and semantic search. Produces a self-contained interactive HTML dashboard with no external runtime dependencies.

## Key Features

- **100% Offline Execution** — No external API calls at runtime. All neural models, NLTK corpora, spaCy pipelines, and JavaScript/CSS assets are cached locally before the first run. Works without internet after initial setup.
- **Dynamic NLP Routing** — Language-aware sentiment analysis dispatches to pysentimiento (es, fr, pt, de) or VADER with TextBlob secondary (en). Preprocessing adapts per language: NEG_ prefix and lemmatization for VADER; raw syntax preserved for transformer-based models.
- **Topic Modeling** — BERTopic with UMAP dimensionality reduction, HDBSCAN clustering scaled to partition size (min_cluster_size = max(10, n/10)), MaximalMarginalRelevance keyword ablation (diversity=0.7), and automatic topic reduction. Falls back to word-frequency analysis when a sentiment partition is below 100 rows.
- **Semantic Search** — Cosine similarity against a synthetic concept string using SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2). Pre-computed embeddings are reused across outlier detection, topic modeling, and semantic search.
- **Interactive Visualizations** — Bokeh scatter plots (topic-colored and semantic-similarity) with hover, zoom, pan, and redundant encoding (color + shape). Plotly n-gram bar charts for outlier analysis. Word clouds for outliers and each sentiment partition.
- **Self-Contained HTML Dashboard** — Jinja2 + Bootstrap 5 report with base64-embedded images, inline Bokeh/Plotly assets, and zero CDN references. Opens in any browser offline.

## Architecture / Tech Stack

| Layer | Library |
|---|---|
| CLI | `argparse` (stdlib) |
| Data processing | `pandas`, `numpy` |
| Sentence embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, max_seq_length overridden to 512) |
| Embedding cache | SHA-256 content hash stored alongside `embeddings.npy` |
| Outlier detection | PCA (384 to 50 dims) + `sklearn.ensemble.IsolationForest` |
| Sentiment (es/fr/pt/de) | `pysentimiento` (robertuya-base-sentiment-analysis, batched at 32) |
| Sentiment (en) | `vaderSentiment` primary, `textblob` secondary |
| Topic modeling | `bertopic`, `umap-learn`, `hdbscan` |
| MMR keyword ablation | `bertopic.representation.MaximalMarginalRelevance` |
| Interactive scatter plots | `bokeh` (INLINE resources, max 1000 points, downsampled preserving centroids) |
| N-gram charts | `plotly` |
| Word clouds | `wordcloud` (Pillow) |
| Report | `jinja2` + Bootstrap 5, base64-embedded assets |
| Language tooling | `spaCy` (lemmatization), `nltk` (stopwords) |
| Testing | `pytest` (424 tests) |

## Installation and Setup

### Prerequisites

Python 3.10 or later. A C++ compiler toolchain is required for `hdbscan` and `umap-learn` (install `build-essential` on Ubuntu, `xcode-select --install` on macOS).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/semantic-cluster.git
cd semantic-cluster

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Warm the local cache (requires internet — one-time setup)
# This downloads neural models, NLTK corpora, spaCy pipelines, and
# pysentimiento weights so the pipeline runs fully offline afterward.
python download_models.py
```

### Cache Warm-Up (Step 4)

`download_models.py` downloads the following into your local cache directory (`~/.cache/`):

- SentenceTransformer model: `paraphrase-multilingual-MiniLM-L12-v2`
- pysentimiento sentiment model: `robertuya-base-sentiment-analysis`
- spaCy pipeline: `es_core_news_sm` (and `en_core_web_sm` for English)
- NLTK corpora: `stopwords`, `punkt`, `punkt_tab`, `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`, `wordnet`, `brown`
- HuggingFace `transformers` and `sentence-transformers` base models

After this step completes, disconnect from the internet and verify the pipeline runs. The pipeline sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` at startup to prevent any accidental network calls.

## Usage

```bash
python main.py \
  --file data/samples/reviews_sample.csv \
  --column review_text \
  --lang es \
  --title "Analisis de Resenas de Hoteles" \
  --palette viridis \
  --concept "precio valor costo" \
  --verbose
```

### CLI Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--file` | Yes | — | Path to input CSV file |
| `--column` | Yes | — | Name of the text column to analyze |
| `--lang` | Yes | — | Language code: `es`, `en`, `fr`, `de`, `pt` |
| `--title` | Yes | — | Report title, included in all output files |
| `--palette` | Yes | — | Color palette: `viridis`, `cividis`, `plasma`, `inferno`, `magma` |
| `--concept` | No | `"precio valor costo"` | Synthetic concept for semantic similarity search |
| `--contamination` | No | `0.05` | Expected outlier ratio (0.01–0.15) |
| `--verbose` | No | `False` | Enable DEBUG-level logging |

### Outputs

All generated files are written to `outputs/`:

| File | Description |
|---|---|
| `report.html` | Fully self-contained offline dashboard (~9 MB). Opens in any browser. |
| `scatter_topics.html` | Bokeh scatter plot colored by topic (positive vs negative). |
| `scatter_semantic.html` | Bokeh scatter plot colored by semantic similarity to the concept. |
| `wordcloud_*.png` | Word clouds for outliers, positive, and negative partitions. |
| `ngrams_*.html` | Plotly n-gram frequency charts (unigrams, bigrams, trigrams for outliers). |
| `embeddings.npy` | Cached 384-dimensional sentence embeddings (reused across runs). |
| `cache_hash.txt` | SHA-256 hash of input text for cache validation. |

## Project Structure

```
main.py                          # Entry point — orchestrates the full pipeline
download_models.py               # One-time setup script (internet required)
src/
  pipeline/
    loader.py                    # CSV ingestion and column validation
    preprocessor.py              # Text cleaning, tokenization, negation handling
    outlier_detector.py          # PCA + IsolationForest, embedding cache
    sentiment_analyzer.py        # Language-aware sentiment classification
    topic_modeler.py             # BERTopic or word-frequency fallback
    semantic_search.py           # Cosine similarity against concept
  viz/
    scatter_plot.py              # Bokeh interactive scatter plots
    wordcloud_gen.py             # Word clouds and Plotly n-gram charts
    report_builder.py            # Jinja2 + Bootstrap 5 HTML report
  utils/
    color_palettes.py            # Colorblind-safe palette manager
    language_config.py           # Per-language preprocessing configuration
    validators.py                # CLI argument validation and constants
assets/
  bootstrap.min.css              # Bootstrap 5 (injected into report)
  plotly.min.js                  # Plotly JS (injected into report)
data/
  samples/
    reviews_sample.csv           # 500 synthetic Spanish hotel reviews (demo)
tests/                           # 424 pytest tests
```

## Pipeline Flow

```
CSV Input
  [1] Loader              raw_text
  [2] Preprocessor        clean_text, tokens
  [3] Outlier Detector    is_outlier, embeddings (cached)
       |-- outliers  --> [VIZ] Word clouds + n-grams
       |
       df_clean
  [4] Sentiment Analyzer   sentiment, polarity
       |-- positive partition
       |-- negative partition
  [5] Topic Modeler        topic_id, umap_x/y, keywords, representative doc
  [6] Semantic Search      concept_similarity (vs. synthetic concept)
  [VIZ] Scatter plots, word clouds, n-grams, HTML report
```

## Design Decisions

- **Offline-first**: All neural models, linguistic data, and web assets are cached locally. No CDN, no API keys, no internet at runtime.
- **Embedding reuse**: SentenceTransformers encodes the corpus once. The same embedding matrix is shared across outlier detection, topic modeling, and semantic search.
- **Embedding cache invalidation**: A SHA-256 hash of concatenated `clean_text` values is computed on every run. If the hash matches the cached `cache_hash.txt`, the full `embeddings.npy` is loaded — no re-encoding.
- **Partition-aware clustering**: BERTopic's HDBSCAN `min_cluster_size` scales with partition size (max(10, n/10)). Below 100 rows, the pipeline falls back to word frequency.
- **Language preprocessing separation**: English text receives stopword removal, lemmatization, and VADER-style NEG_ prefix. Non-English text is lowercased and stripped of special characters only — syntax is preserved for transformer attention.
- **Downsampled scatters**: Datasets larger than 1000 points are downsampled (preserving representative centroid documents) to keep Bokeh HTML output manageable.

## Testing

```bash
pytest tests/      # 424 tests
```

## License

MIT
