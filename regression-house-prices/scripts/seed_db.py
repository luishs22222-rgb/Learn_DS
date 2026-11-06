"""CLI: (re)seed the mock database with a fresh synthetic batch.

Usage:
    python scripts/seed_db.py [--rows 2000] [--seed 42] [--null-fraction 0.08]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from house_prices.data_generation import seed_database  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the house_prices mock database.")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-fraction", type=float, default=0.08)
    args = parser.parse_args()

    seed_database(n_rows=args.rows, seed=args.seed, null_fraction=args.null_fraction)


if __name__ == "__main__":
    main()
