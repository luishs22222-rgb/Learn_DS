import numpy as np
import pandas as pd
import pytest

from house_prices.features import build_preprocessor, train_test_split_data


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "sqft": [1200, 1800, None, 2400, 1500, 3000, 1100, 2000],
            "bedrooms": [2, 3, 3, 4, 2, 5, 1, 3],
            "bathrooms": [1.0, 2.0, 2.0, 3.0, 1.5, 3.5, 1.0, 2.0],
            "year_built": [1990, 2000, 1985, 2010, 1995, 2015, 1975, 2005],
            "lot_size": [5000, None, 6000, 7000, None, 8000, 4000, 6500],
            "neighborhood_name": ["Riverside", "Old Town", "Riverside", "Maple Heights",
                                   "Sunset District", "Riverside", "Old Town", "Maple Heights"],
            "sale_price": [300000, 350000, 320000, 500000, 310000, 600000, 250000, 420000],
        }
    )


def test_preprocessor_imputes_nulls(sample_df):
    pre = build_preprocessor()
    X = sample_df.drop(columns=["sale_price"])
    transformed = pre.fit_transform(X)
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()


def test_preprocessor_one_hot_expands_categories(sample_df):
    pre = build_preprocessor()
    X = sample_df.drop(columns=["sale_price"])
    transformed = pre.fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    n_numeric = 5
    n_categories = X["neighborhood_name"].nunique()
    assert dense.shape[1] == n_numeric + n_categories


def test_train_test_split_proportions(sample_df):
    X_train, X_test, y_train, y_test = train_test_split_data(sample_df, test_size=0.25, random_state=1)
    assert len(X_test) == 2
    assert len(X_train) == 6
    assert set(X_train.index).isdisjoint(set(X_test.index))
