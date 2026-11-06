"""Synthetic data generator that seeds the mock database.

The pricing relationship is defined explicitly (not hidden inside some
downloaded dataset) so we know exactly what a model *should* learn, which
makes debugging the pipeline much easier. Every coefficient is a function
argument with a sane default rather than a hardcoded literal -- this is
what will let a future "phase 2" (see docs/roadmap-phase2.md) simulate
data/concept drift by simply calling this function again with different
coefficients for a later batch, instead of needing a new data source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from house_prices.config import DB_PATH, RANDOM_SEED
from house_prices.db import get_connection, init_schema, reset_business_tables

NEIGHBORHOODS = [
    ("Riverside", 0.82),
    ("Old Town", 0.55),
    ("Maple Heights", 0.68),
    ("Sunset District", 0.74),
    ("Industrial Row", 0.35),
]


@dataclass
class PricingCoefficients:
    """The ground-truth data-generating process for sale_price.

    sale_price = base_price
               + price_per_sqft * sqft
               + bedroom_bonus * bedrooms
               + bathroom_bonus * bathrooms
               - age_penalty * (current_year - year_built)
               + location_multiplier * location_score * base_price_scale
               + noise

    Changing these values (or calling generate_batch with different ones)
    is exactly how a later "drift" batch would be produced.
    """

    base_price: float = 120_000.0
    price_per_sqft: float = 145.0
    bedroom_bonus: float = 8_000.0
    bathroom_bonus: float = 12_000.0
    age_penalty_per_year: float = 900.0
    location_multiplier: float = 180_000.0
    noise_std: float = 18_000.0
    current_year: int = 2026


def _sample_properties(n_rows: int, rng: np.random.Generator, null_fraction: float) -> pd.DataFrame:
    location_ids = rng.integers(1, len(NEIGHBORHOODS) + 1, size=n_rows)
    sqft = rng.normal(1800, 550, size=n_rows).clip(400, 6000)
    bedrooms = rng.integers(1, 6, size=n_rows)
    bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], size=n_rows)
    year_built = rng.integers(1950, 2024, size=n_rows)
    lot_size = rng.normal(6500, 2200, size=n_rows).clip(1000, 20000)

    # Real, SQL-level NULLs -- not a post-hoc pandas NaN -- to give
    # features.py a genuine imputation problem to solve.
    null_mask = rng.random(n_rows) < null_fraction
    lot_size = lot_size.astype(object)
    lot_size[null_mask] = None

    return pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "sqft": sqft.round(0),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "lot_size": lot_size,
            "location_id": location_ids,
        }
    )


def _price_for_row(row: pd.Series, coeffs: PricingCoefficients, rng: np.random.Generator, location_score: float) -> float:
    age = coeffs.current_year - row["year_built"]
    price = (
        coeffs.base_price
        + coeffs.price_per_sqft * row["sqft"]
        + coeffs.bedroom_bonus * row["bedrooms"]
        + coeffs.bathroom_bonus * row["bathrooms"]
        - coeffs.age_penalty_per_year * age
        + coeffs.location_multiplier * location_score
        + rng.normal(0, coeffs.noise_std)
    )
    return max(price, 25_000.0)


def generate_batch(
    n_rows: int = 2000,
    seed: int = RANDOM_SEED,
    null_fraction: float = 0.08,
    coeffs: PricingCoefficients | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates (locations_df, properties_df, sales_df) ready for SQL insertion."""
    coeffs = coeffs or PricingCoefficients()
    rng = np.random.default_rng(seed)

    locations_df = pd.DataFrame(
        [(i + 1, name, score) for i, (name, score) in enumerate(NEIGHBORHOODS)],
        columns=["location_id", "neighborhood_name", "location_score"],
    )
    location_score_by_id = dict(zip(locations_df.location_id, locations_df.location_score))

    properties_df = _sample_properties(n_rows, rng, null_fraction)

    sale_prices = [
        _price_for_row(row, coeffs, rng, location_score_by_id[row["location_id"]])
        for _, row in properties_df.iterrows()
    ]
    sale_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 400, size=n_rows), unit="D"
    )
    sales_df = pd.DataFrame(
        {
            "property_id": properties_df["id"],
            "sale_price": np.round(sale_prices, 2),
            "sale_date": sale_dates.strftime("%Y-%m-%d"),
        }
    )

    return locations_df, properties_df, sales_df


def seed_database(
    n_rows: int = 2000,
    seed: int = RANDOM_SEED,
    null_fraction: float = 0.08,
    db_path: Path | str = DB_PATH,
    coeffs: PricingCoefficients | None = None,
) -> None:
    init_schema(db_path)
    reset_business_tables(db_path)
    locations_df, properties_df, sales_df = generate_batch(n_rows, seed, null_fraction, coeffs)

    with get_connection(db_path) as conn:
        locations_df.to_sql("locations", conn, if_exists="append", index=False)
        properties_df.to_sql("properties", conn, if_exists="append", index=False)
        sales_df.to_sql("sales", conn, if_exists="append", index=False)
        conn.commit()

    print(
        f"Seeded {len(properties_df)} properties across {len(locations_df)} "
        f"neighborhoods into {db_path} (lot_size null rate target={null_fraction:.0%})."
    )
