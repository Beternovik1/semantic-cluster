"""Topic modeling via BERTopic with scaled clustering and MMR ablation.

Falls back to word-frequency analysis when the partition is below
``MIN_PARTITION_SIZE`` rows.  Both paths produce the same six-column
contract required by downstream visualizations.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from src.utils.validators import MIN_PARTITION_SIZE, RANDOM_SEED

logger = logging.getLogger(__name__)

_TOPIC_NAME_KEYWORDS: dict[str, str] = {
    "precio": "Pricing",
    "costo": "Pricing",
    "coste": "Pricing",
    "dinero": "Pricing",
    "caro": "Pricing",
    "barato": "Pricing",
    "price": "Pricing",
    "cost": "Pricing",
    "value": "Pricing",
    "expensive": "Pricing",
    "cheap": "Pricing",
    "prix": "Pricing",
    "coût": "Pricing",
    "cher": "Pricing",
    "preis": "Pricing",
    "kosten": "Pricing",
    "teuer": "Pricing",
    "billig": "Pricing",
    "preço": "Pricing",
    "custo": "Pricing",
    "calidad": "Quality",
    "quality": "Quality",
    "qualité": "Quality",
    "qualität": "Quality",
    "qualidade": "Quality",
    "servicio": "Service",
    "serviço": "Service",
    "service": "Service",
    "atención": "Service",
    "personal": "Service",
    "staff": "Service",
    "personnel": "Service",
    "dienst": "Service",
    "pessoal": "Service",
    "ubicación": "Location",
    "ubicacion": "Location",
    "lugar": "Location",
    "location": "Location",
    "emplacement": "Location",
    "lage": "Location",
    "localização": "Location",
    "habitación": "Room",
    "habitacion": "Room",
    "room": "Room",
    "chambre": "Room",
    "zimmer": "Room",
    "quarto": "Room",
    "limpieza": "Cleanliness",
    "limpio": "Cleanliness",
    "clean": "Cleanliness",
    "cleanliness": "Cleanliness",
    "propreté": "Cleanliness",
    "sauberkeit": "Cleanliness",
    "sauber": "Cleanliness",
    "limpeza": "Cleanliness",
    "comida": "Food",
    "desayuno": "Food",
    "food": "Food",
    "breakfast": "Food",
    "restaurant": "Food",
    "nourriture": "Food",
    "petit-déjeuner": "Food",
    "frühstück": "Food",
    "essen": "Food",
    "restaurante": "Food",
    "experiencia": "Experience",
    "experience": "Experience",
    "ambiente": "Atmosphere",
    "atmosphere": "Atmosphere",
    "decoración": "Atmosphere",
    "wifi": "Amenities",
    "parking": "Amenities",
    "estacionamiento": "Amenities",
    "piscina": "Amenities",
    "pool": "Amenities",
    "gym": "Amenities",
    "transporte": "Amenities",
}


def _readable_topic_name(keywords_str: str) -> str:
    """Map a comma-separated keyword string to a human-readable topic name.

    Args:
        keywords_str: Comma-separated keywords from topic modeler.

    Returns:
        A short human-readable name like "Pricing" or "Service".
        Falls back to first 2 keywords formatted as title if no match.
    """
    if not keywords_str:
        return "General"
    keywords = [k.strip().lower() for k in keywords_str.split(",")]
    scores: dict[str, int] = {}
    for kw in keywords:
        for pattern, name in _TOPIC_NAME_KEYWORDS.items():
            if pattern in kw or kw in pattern:
                scores[name] = scores.get(name, 0) + 1
                break
    if scores:
        return max(scores, key=scores.get)
    parts = [k.strip().title() for k in keywords_str.split(",")[:2]]
    return " / ".join(parts)


class TopicModeler:
    """Apply BERTopic or fallback word-frequency topic modeling.

    Args:
        embedding_model: A ``SentenceTransformer`` instance (injected,
            unused directly — BERTopic receives pre-computed embeddings).
        lang_cfg: A ``LanguageConfig`` instance (used for stopwords in
            the word-frequency fallback path).
    """

    def __init__(self, embedding_model, lang_cfg) -> None:
        self._model = embedding_model
        self._lang_cfg = lang_cfg

    def run(
        self, df: pd.DataFrame, embeddings: np.ndarray
    ) -> tuple[pd.DataFrame, dict[int, str], bool]:
        """Fit topic model on the partition and annotate every row.

        Args:
            df: Partition DataFrame (positive or negative).
            embeddings: 384-dim embeddings aligned to *df* rows.

        Returns:
            Tuple of ``(df_updated, topic_keywords, used_bertopic)``:
            - ``df_updated``: original DataFrame with topic columns added.
            - ``topic_keywords``: mapping ``{topic_id: keywords_string}``.
            - ``used_bertopic``: ``True`` if BERTopic was used,
              ``False`` if word-frequency fallback ran.
        """
        if len(df) < MIN_PARTITION_SIZE:
            return self._fallback(df)

        return self._bertopic_path(df, embeddings)

    # ── BERTopic path ──────────────────────────────────────────────

    def _bertopic_path(
        self, df: pd.DataFrame, embeddings: np.ndarray
    ) -> tuple[pd.DataFrame, dict[int, str], bool]:
        n = len(df)
        min_cluster_size = max(10, n // 10)
        logger.info(
            "Running BERTopic on %d documents (min_cluster_size=%d).",
            n,
            min_cluster_size,
        )

        umap_model = UMAP(n_components=2, random_state=RANDOM_SEED)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        representation_model = MaximalMarginalRelevance(diversity=0.7)

        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
            nr_topics=None,
            verbose=False,
        )

        topics, _ = topic_model.fit_transform(
            df["clean_text"].tolist(),
            embeddings=embeddings,
        )

        # Guarded topic reduction — only merge when ≥2 non-noise topics
        unique_topics = sorted(set(t for t in topics if t != -1))
        if len(unique_topics) >= 2:
            logger.info(
                "Reducing topics (auto) on %d non-noise topics.",
                len(unique_topics),
            )
            topic_model.reduce_topics(
                df["clean_text"].tolist(),
                nr_topics="auto",
            )
            topics = topic_model.topics_
            unique_topics = sorted(set(t for t in topics if t != -1))

        # UMAP 2-D coordinates
        umap_coords = topic_model.umap_model.transform(embeddings)

        # Build per-document columns
        df_out = df.copy()
        df_out["topic_id"] = topics
        df_out["umap_x"] = umap_coords[:, 0]
        df_out["umap_y"] = umap_coords[:, 1]

        # Build topic metadata dict
        topic_keywords: dict[int, str] = {}

        for tid in unique_topics:
            words = topic_model.get_topic(tid)
            if words:
                kw_str = ", ".join(w for w, _ in words)
            else:
                kw_str = ""
            topic_keywords[tid] = kw_str

        # Map each row to its topic_label (readable name) and topic_keywords (raw)
        def _label(tid: int) -> str:
            return _readable_topic_name(topic_keywords.get(tid, ""))

        def _keywords(tid: int) -> str:
            return topic_keywords.get(tid, "")

        df_out["topic_label"] = df_out["topic_id"].apply(_label)
        df_out["topic_keywords"] = df_out["topic_id"].apply(_keywords)

        # Representative document (centroid)
        df_out["representative_doc"] = False
        for tid in unique_topics:
            mask = df_out["topic_id"] == tid
            topic_embeddings = embeddings[mask.values]
            centroid = topic_embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(topic_embeddings, centroid).flatten()
            best_idx = np.argmax(sims)
            # Map back to index in df_out
            df_idx = df_out.index[mask.values][best_idx]
            df_out.at[df_idx, "representative_doc"] = True

        logger.info(
            "BERTopic found %d topics (+ noise) in partition.",
            len(unique_topics),
        )

        return df_out, topic_keywords, True

    # ── Fallback path ──────────────────────────────────────────────

    def _fallback(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[int, str], bool]:
        logger.warning(
            "Partition size (%d) below MIN_PARTITION_SIZE (%d). "
            "Falling back to word frequency.",
            len(df),
            MIN_PARTITION_SIZE,
        )

        # Gather all tokens, optionally filtering stopwords
        stopwords_set = self._load_stopwords()
        counter: Counter[str] = Counter()

        for tokens in df["tokens"]:
            for token in tokens:
                token_lower = token.lower()
                if stopwords_set and token_lower in stopwords_set:
                    continue
                counter[token_lower] += 1

        top_words = [w for w, _ in counter.most_common(20)]
        kw_str = ", ".join(top_words) if top_words else ""

        df_out = df.copy()
        df_out["topic_id"] = -1
        df_out["topic_label"] = "word_frequency_fallback"
        df_out["umap_x"] = None
        df_out["umap_y"] = None
        df_out["topic_keywords"] = kw_str
        df_out["representative_doc"] = False

        topic_keywords: dict[int, str] = {-1: kw_str}

        logger.info(
            "Fallback keywords (%d unique tokens): %s",
            len(top_words),
            kw_str[:200],
        )

        return df_out, topic_keywords, False

    # ── Helpers ────────────────────────────────────────────────────

    def _load_stopwords(self) -> set[str]:
        """Return NLTK stopwords for the configured language, or empty."""
        try:
            from nltk.corpus import stopwords

            return set(stopwords.words(self._lang_cfg.stopwords_name))
        except Exception:
            logger.debug(
                "Could not load NLTK stopwords for '%s'. Skipping.",
                self._lang_cfg.stopwords_name,
            )
            return set()
