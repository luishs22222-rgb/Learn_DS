"""The mock relational database.

This is a SQLite file standing in for "the production database." The point
is to practice real SQL (DDL, JOINs, GROUP BY aggregation, parameterized
inserts) exactly like you would against Postgres/MySQL -- SQLite is just the
zero-setup local stand-in.

Schema (three related tables, not one flat file):
    customers(customer_id PK, tenure_months, contract_type, monthly_charges,
              total_charges, signup_date)
    support_tickets(ticket_id PK, customer_id FK, opened_date)  -- zero or
              many rows per customer
    churn_labels(customer_id FK, churned, as_of_date)

`fetch_training_data()` assembles the training table with a JOIN plus a
GROUP BY aggregation subquery (support_tickets -> support_ticket_count) --
real SQL aggregation, not just SELECT *.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from customer_churn.config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id     INTEGER PRIMARY KEY,
    tenure_months   INTEGER NOT NULL,
    contract_type   TEXT NOT NULL,
    monthly_charges REAL NOT NULL,
    total_charges   REAL,              -- intentionally nullable: ~7% missing (brand-new, not yet billed)
    signup_date     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    opened_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS churn_labels (
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    churned     INTEGER NOT NULL,
    as_of_date  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    input_json            TEXT NOT NULL,
    predicted_class       INTEGER NOT NULL,
    predicted_probability REAL NOT NULL,
    model_name            TEXT NOT NULL,
    model_version         TEXT NOT NULL
);
"""

# LEFT JOIN + GROUP BY subquery + COALESCE: customers with zero support
# tickets have no rows in support_tickets at all (there's nothing to be
# "null" about), so this is a join-cardinality gap, not a missing-value
# problem -- COALESCE(..., 0) is the correct fix here, distinct from the
# median/most-frequent imputation features.py does for genuine NULLs like
# total_charges.
TRAINING_QUERY = """
SELECT
    c.customer_id,
    c.tenure_months,
    c.contract_type,
    c.monthly_charges,
    c.total_charges,
    COALESCE(t.support_ticket_count, 0) AS support_ticket_count,
    cl.churned
FROM customers c
JOIN churn_labels cl ON cl.customer_id = c.customer_id
LEFT JOIN (
    SELECT customer_id, COUNT(*) AS support_ticket_count
    FROM support_tickets
    GROUP BY customer_id
) t ON t.customer_id = c.customer_id
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
    """Wipes customers/support_tickets/churn_labels (used before reseeding). Leaves predictions intact."""
    with get_connection(db_path) as conn:
        conn.executescript(
            "DELETE FROM churn_labels; DELETE FROM support_tickets; DELETE FROM customers;"
        )
        conn.commit()


def fetch_training_data(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """Real SQL fetch: joins customers + churn_labels, aggregates support_tickets via GROUP BY."""
    with get_connection(db_path) as conn:
        return pd.read_sql(TRAINING_QUERY, conn)


def insert_prediction(
    input_data: dict,
    predicted_class: int,
    predicted_probability: float,
    model_name: str,
    model_version: str,
    db_path: Path | str = DB_PATH,
) -> int:
    from datetime import datetime, timezone

    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (created_at, input_json, predicted_class, predicted_probability, model_name, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_data),
                int(predicted_class),
                float(predicted_probability),
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
