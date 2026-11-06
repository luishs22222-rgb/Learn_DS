"""The mock relational database.

This is a SQLite file standing in for "the production database." The point
is to practice real SQL (DDL, JOINs, parameterized inserts) exactly like you
would against Postgres/MySQL -- SQLite is just the zero-setup local stand-in.

Schema (three related tables, not one flat file):
    locations(location_id PK, neighborhood_name, location_score)
    properties(id PK, sqft, bedrooms, bathrooms, year_built, lot_size, location_id FK)
    sales(property_id FK, sale_price, sale_date)

`fetch_training_data()` assembles the training table with a single JOIN
query, the same way you'd pull a training set out of a real warehouse.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from house_prices.config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS locations (
    location_id     INTEGER PRIMARY KEY,
    neighborhood_name TEXT NOT NULL,
    location_score  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS properties (
    id          INTEGER PRIMARY KEY,
    sqft        REAL NOT NULL,
    bedrooms    INTEGER NOT NULL,
    bathrooms   REAL NOT NULL,
    year_built  INTEGER NOT NULL,
    lot_size    REAL,                 -- intentionally nullable: ~8% missing in generated data
    location_id INTEGER NOT NULL REFERENCES locations(location_id)
);

CREATE TABLE IF NOT EXISTS sales (
    property_id INTEGER NOT NULL REFERENCES properties(id),
    sale_price  REAL NOT NULL,
    sale_date   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    input_json      TEXT NOT NULL,
    predicted_price REAL NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL
);
"""

TRAINING_QUERY = """
SELECT
    p.id               AS property_id,
    p.sqft,
    p.bedrooms,
    p.bathrooms,
    p.year_built,
    p.lot_size,
    l.neighborhood_name,
    l.location_score,
    s.sale_price,
    s.sale_date
FROM properties p
JOIN locations l ON l.location_id = p.location_id
JOIN sales s      ON s.property_id = p.id
"""


@contextmanager
def get_connection(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def init_schema(db_path: Path | str = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def reset_business_tables(db_path: Path | str = DB_PATH) -> None:
    """Wipes locations/properties/sales (used before reseeding). Leaves predictions intact."""
    with get_connection(db_path) as conn:
        conn.executescript(
            "DELETE FROM sales; DELETE FROM properties; DELETE FROM locations;"
        )
        conn.commit()


def fetch_training_data(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Real SQL fetch: joins properties + locations + sales into one training frame."""
    with get_connection(db_path) as conn:
        return pd.read_sql(TRAINING_QUERY, conn)


def insert_prediction(
    input_data: dict,
    predicted_price: float,
    model_name: str,
    model_version: str,
    db_path: Path | str = DB_PATH,
) -> int:
    from datetime import datetime, timezone

    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (created_at, input_json, predicted_price, model_name, model_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                float(predicted_price),
                model_name,
                model_version,
            ),
        )
        conn.commit()
        return cur.lastrowid


def fetch_recent_predictions(limit: int = 20, db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
