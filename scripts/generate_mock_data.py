"""Generate a synthetic dataset of Spanish hotel / restaurant reviews.

Output: ``data/samples/reviews_sample.csv`` (500 rows).

Content distribution:
  - 200 clearly positive reviews (40 %)
  - 200 clearly negative reviews (40 %)
  - 100 reviews containing the semantic-search target words (20 %):
    "precio", "costo", "caro", "barato", "valor", "economico"

Usage:
    python scripts/generate_mock_data.py
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
NUM_ROWS = 500
POSITIVE_COUNT = 200
NEGATIVE_COUNT = 200
SEMANTIC_COUNT = 100
OUTPUT_PATH = Path("data/samples/reviews_sample.csv")

# ── Template data ──────────────────────────────────────────────────────────────

HOTEL_NAMES: list[str] = [
    "Hotel Bahia del Sol",
    "Parador de Granada",
    "Hostal Madrid Centro",
    "Hotel Mediterraneo",
    "Posada Real Barcelona",
    "Hotel Alhambra Palace",
    "Hacienda San Miguel",
    "Hotel Costa del Sol",
    "Palacio de la Luz",
    "Villa Marbella Suites",
    "Hotel Santiago Apostol",
    "Casa Rural Los Olivos",
    "Hotel Mirador del Valle",
    "Gran Hotel Colon",
    "Hotel Santa Maria",
]

POSITIVE_FRAGMENTS: list[str] = [
    "Excelente servicio y atencion al cliente.",
    "Habitaciones muy limpias y amplias.",
    "La comida es deliciosa, volveremos sin duda.",
    "Muy buena ubicacion, cerca de todo.",
    "El personal fue muy atento y amable.",
    "Las instalaciones son modernas y comodas.",
    "Desayuno buffet variado y de calidad.",
    "Vistas espectaculares desde la habitacion.",
    "Todo perfecto, supero nuestras expectativas.",
    "Cama muy comoda, dormimos estupendamente.",
    "Excelente opcion para viajar en familia.",
    "El ambiente es acogedor y tranquilo.",
    "La piscina y el son maravillosos.",
    "Servicio rapido y profesional en todo momento.",
    "Decoracion elegante y muy cuidada.",
    "El jardin es precioso y muy bien mantenido.",
    "Silencio total por la noche, se descansa muy bien.",
    "El personal del restaurante es excepcional.",
    "Las habitaciones tienen vistas increibles al mar.",
    "Muy recomendable, repetiremos el proximo ano.",
]

NEGATIVE_FRAGMENTS: list[str] = [
    "Pesima atencion al cliente, no volvere.",
    "Las habitaciones estan sucias y viejas.",
    "La comida es mala.",
    "El ruido de la calle no deja dormir.",
    "Personal antipatico y poco profesional.",
    "Muy caro para lo que ofrecen.",
    "El wifi no funciona en las habitaciones.",
    "El bano tenia moho y mal olor.",
    "Servicio muy lento, esperamos una hora.",
    "Las fotos no coinciden con la realidad.",
    "Camas incomodas y almohadas duras.",
    "El aire acondicionado no funcionaba.",
    "Mala relacion calidad precio.",
    "El hotel necesita una reforma urgente.",
    "La ubicacion es mala y peligrosa.",
    "La calefaccion no funciona correctamente.",
    "El desayuno es escaso y de baja calidad.",
    "Las toallas estaban sucias al llegar.",
    "Los precios del minibar son abusivos.",
    "El ascensor no funciona constantemente.",
]

# Phrases containing semantic-search target words, mixed sentiment
SEMANTIC_PHRASES: list[str] = [
    "El precio es muy razonable para la calidad que ofrecen.",
    "El costo de la habitacion es excesivo, no lo recomiendo.",
    "Buena relacion calidad precio, muy satisfecho.",
    "Es el hotel mas caro de la zona y no lo vale.",
    "Opciones economicas para viajeros con presupuesto ajustado.",
    "La mejor relacion valor por dinero que he encontrado.",
    "Precio competitivo comparado con otros hoteles similares.",
    "Demasiado caro para los servicios que realmente dan.",
    "Vale la pena cada euro que pagas por la experiencia.",
    "El valor agregado del desayuno incluido es estupendo.",
    "Coste elevado pero justificado por la ubicacion.",
    "Oferta economica con descuento por reserva anticipada.",
    "Precio desorbitado para una habitacion tan pequena.",
    "Buena opcion de bajo costo sin sacrificar calidad.",
    "Relacion precio calidad inmejorable, todo perfecto.",
    "El precio incluye desayuno y cena, muy buen valor.",
    "Caro pero vale la pena por las vistas.",
    "Alternativa economica en pleno centro historico.",
    "El costo adicional del son no esta justificado.",
    "Precio ajustado para tratarse de un hotel de cuatro estrellas.",
]


# ── Generation helpers ─────────────────────────────────────────────────────────


def _random_date(start_year: int = 2023, end_year: int = 2025) -> str:
    """Return a random date string ``YYYY-MM-DD`` between *start_year* and *end_year*."""
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    max_day = 28 if month == 2 else 30 if month in {4, 6, 9, 11} else 31
    day = random.randint(1, max_day)
    return f"{year}-{month:02d}-{day:02d}"


def _build_positive_review() -> str:
    """Combine 1-3 positive fragments into a single review string."""
    count = random.randint(1, 3)
    selected = random.sample(POSITIVE_FRAGMENTS, min(count, len(POSITIVE_FRAGMENTS)))
    return " ".join(selected)


def _build_negative_review() -> str:
    """Combine 1-3 negative fragments into a single review string."""
    count = random.randint(1, 3)
    selected = random.sample(NEGATIVE_FRAGMENTS, min(count, len(NEGATIVE_FRAGMENTS)))
    return " ".join(selected)


def _build_semantic_review() -> str:
    """Return a semantic-target phrase, optionally preceded by a sentiment fragment."""
    if random.random() < 0.5:
        phrase = random.choice(SEMANTIC_PHRASES)
        return phrase
    sentiment = random.choice(POSITIVE_FRAGMENTS + NEGATIVE_FRAGMENTS)
    phrase = random.choice(SEMANTIC_PHRASES)
    return f"{sentiment} {phrase}"


def _rating_for_sentiment(sentiment_type: str) -> int:
    """Map review type to a plausible rating (1-5)."""
    if sentiment_type == "positive":
        return random.choices([5, 4, 3], weights=[60, 30, 10])[0]
    if sentiment_type == "negative":
        return random.choices([1, 2, 3], weights=[50, 35, 15])[0]
    # semantic — mixed, spans full range
    return random.choices([5, 4, 3, 2, 1], weights=[20, 20, 20, 20, 20])[0]


def _generate_rows() -> list[dict]:
    """Build the full 500-row dataset as a list of dicts."""
    random.seed(RANDOM_SEED)

    reviews: list[dict] = []
    row_id = 1

    for _ in range(POSITIVE_COUNT):
        hotel = random.choice(HOTEL_NAMES)
        reviews.append(
            {
                "id": row_id,
                "review_text": _build_positive_review(),
                "rating": _rating_for_sentiment("positive"),
                "date": _random_date(),
                "hotel_name": hotel,
            }
        )
        row_id += 1

    for _ in range(NEGATIVE_COUNT):
        hotel = random.choice(HOTEL_NAMES)
        reviews.append(
            {
                "id": row_id,
                "review_text": _build_negative_review(),
                "rating": _rating_for_sentiment("negative"),
                "date": _random_date(),
                "hotel_name": hotel,
            }
        )
        row_id += 1

    for _ in range(SEMANTIC_COUNT):
        hotel = random.choice(HOTEL_NAMES)
        reviews.append(
            {
                "id": row_id,
                "review_text": _build_semantic_review(),
                "rating": _rating_for_sentiment("semantic"),
                "date": _random_date(),
                "hotel_name": hotel,
            }
        )
        row_id += 1

    random.shuffle(reviews)
    return reviews


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    """Generate the demo CSV and write it to ``data/samples/reviews_sample.csv``."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = _generate_rows()
    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    logger.info(
        "Dataset saved to %s (%d rows, %d columns).",
        OUTPUT_PATH.resolve(),
        len(df),
        len(df.columns),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
