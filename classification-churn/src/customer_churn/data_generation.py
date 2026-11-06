"""Synthetic data generator that seeds the mock database.

The churn relationship is defined explicitly (not hidden inside some
downloaded dataset) so we know exactly what a model *should* learn, which
makes debugging the pipeline much easier. Every coefficient is a function
argument with a sane default rather than a hardcoded literal -- this is
what will let a future "phase 2" (see docs/roadmap-phase2.md) simulate
data/concept drift by simply calling this function again with different
coefficients for a later batch, instead of needing a new data source.

Deliberately produces an *imbalanced* target (~15-25% churned) rather than
a clean 50/50 split -- this is what makes accuracy misleading here and the
rest of the classification-metrics story in the README meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from customer_churn.config import DB_PATH, RANDOM_SEED
from customer_churn.db import get_connection, init_schema, reset_business_tables

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
CONTRACT_PROBS = [0.55, 0.30, 0.15]


@dataclass
class ChurnCoefficients:
    """The ground-truth data-generating process for `churned`.

    logit = intercept
          + tenure_coef * tenure_months
          + monthly_charges_coef * monthly_charges
          + support_ticket_coef * support_ticket_count
          + contract_bonus[contract_type]
          + noise
    churn_probability = sigmoid(logit)
    churned ~ Bernoulli(churn_probability)

    Changing these values (or calling generate_batch with different ones)
    is exactly how a later "drift" batch would be produced.
    """

    intercept: float = -0.95
    tenure_coef: float = -0.045
    monthly_charges_coef: float = 0.010
    support_ticket_coef: float = 0.32
    contract_bonus: dict[str, float] = field(
        default_factory=lambda: {"month-to-month": 0.55, "one-year": -0.35, "two-year": -1.30}
    )
    logit_noise_std: float = 0.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_customers(n_rows: int, rng: np.random.Generator, null_fraction: float) -> pd.DataFrame:
    tenure_months = rng.integers(1, 73, size=n_rows)
    contract_type = rng.choice(CONTRACT_TYPES, size=n_rows, p=CONTRACT_PROBS)
    monthly_charges = rng.normal(70, 25, size=n_rows).clip(15, 150).round(2)

    # total_charges roughly tracks tenure * monthly rate with some noise --
    # brand-new customers (tenure <= 1 month) haven't been billed yet, which
    # is where the real NULLs come from (not a post-hoc pandas NaN).
    total_charges = (tenure_months * monthly_charges * rng.uniform(0.9, 1.1, size=n_rows)).round(2)
    total_charges = total_charges.astype(object)
    new_customer_mask = tenure_months <= 1
    extra_null_mask = rng.random(n_rows) < max(null_fraction - new_customer_mask.mean(), 0.0)
    null_mask = new_customer_mask | extra_null_mask
    total_charges[null_mask] = None

    signup_start = pd.to_datetime("2024-01-01")
    signup_date = signup_start - pd.to_timedelta(tenure_months * 30, unit="D")

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n_rows + 1),
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
        }
    )


def _sample_support_ticket_counts(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    return rng.poisson(lam=0.7, size=n_rows).clip(0, 8)


def generate_batch(
    n_rows: int = 2500,
    seed: int = RANDOM_SEED,
    null_fraction: float = 0.07,
    coeffs: ChurnCoefficients | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates (customers_df, support_tickets_df, churn_labels_df) ready for SQL insertion."""
    coeffs = coeffs or ChurnCoefficients()
    rng = np.random.default_rng(seed)

    customers_df = _sample_customers(n_rows, rng, null_fraction)
    ticket_counts = _sample_support_ticket_counts(n_rows, rng)

    # Expand per-customer ticket counts into individual ticket rows -- this
    # is why support_tickets needs a GROUP BY in db.py to become a feature.
    ticket_customer_ids = np.repeat(customers_df["customer_id"].to_numpy(), ticket_counts)
    n_tickets = len(ticket_customer_ids)
    ticket_dates = pd.to_datetime("2025-06-01") - pd.to_timedelta(
        rng.integers(0, 200, size=n_tickets), unit="D"
    )
    support_tickets_df = pd.DataFrame(
        {"customer_id": ticket_customer_ids, "opened_date": ticket_dates.strftime("%Y-%m-%d")}
    )

    contract_bonus = np.array([coeffs.contract_bonus[c] for c in customers_df["contract_type"]])
    # total_charges may contain None; churn probability is driven by the
    # *billing rate* (monthly_charges) and tenure, not total_charges, so the
    # generator doesn't need to touch the nullable column at all.
    logit = (
        coeffs.intercept
        + coeffs.tenure_coef * customers_df["tenure_months"].to_numpy()
        + coeffs.monthly_charges_coef * customers_df["monthly_charges"].to_numpy()
        + coeffs.support_ticket_coef * ticket_counts
        + contract_bonus
        + rng.normal(0, coeffs.logit_noise_std, size=n_rows)
    )
    churn_probability = _sigmoid(logit)
    churned = (rng.random(n_rows) < churn_probability).astype(int)

    churn_labels_df = pd.DataFrame(
        {
            "customer_id": customers_df["customer_id"],
            "churned": churned,
            "as_of_date": "2025-12-01",
        }
    )

    return customers_df, support_tickets_df, churn_labels_df


def seed_database(
    n_rows: int = 2500,
    seed: int = RANDOM_SEED,
    null_fraction: float = 0.07,
    db_path: Path | str = DB_PATH,
    coeffs: ChurnCoefficients | None = None,
) -> None:
    init_schema(db_path)
    reset_business_tables(db_path)
    customers_df, support_tickets_df, churn_labels_df = generate_batch(n_rows, seed, null_fraction, coeffs)

    with get_connection(db_path) as conn:
        customers_df.to_sql("customers", conn, if_exists="append", index=False)
        if len(support_tickets_df):
            support_tickets_df.to_sql("support_tickets", conn, if_exists="append", index=False)
        churn_labels_df.to_sql("churn_labels", conn, if_exists="append", index=False)
        conn.commit()

    churn_rate = churn_labels_df["churned"].mean()
    print(
        f"Seeded {len(customers_df)} customers ({len(support_tickets_df)} support tickets) "
        f"into {db_path} (churn rate={churn_rate:.1%}, total_charges null rate="
        f"{customers_df['total_charges'].isna().mean():.1%})."
    )
