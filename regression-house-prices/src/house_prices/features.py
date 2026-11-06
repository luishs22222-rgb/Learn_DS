"""Feature engineering: null handling, one-hot encoding, scaling.

Preprocessing and the model are bundled into a single sklearn Pipeline so
that whatever transformation is fit at training time is applied identically
at prediction time -- see docs/glossary.md ("scikit-learn Pipeline /
ColumnTransformer") for why this matters.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_prices.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_SEED, TARGET_COLUMN


def build_preprocessor() -> ColumnTransformer:
    """Median-imputes + scales numeric columns, one-hot encodes categoricals.

    Median imputation (not mean) is used for lot_size because it's robust
    to the skew/outliers typical of lot sizes. Scaling matters for the
    linear model; tree-based models ignore it (scale-invariant) but it does
    them no harm, so we apply it uniformly for simplicity.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[feature_columns].copy(), df[TARGET_COLUMN].copy()


def train_test_split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_SEED
):
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
