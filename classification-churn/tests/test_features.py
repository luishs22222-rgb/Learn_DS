import numpy as np
import pandas as pd
import pytest

from customer_churn.features import build_preprocessor, train_test_split_data


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "tenure_months": [2, 40, 5, 60, 12, 55, 3, 30, 20, 45],
            "contract_type": ["month-to-month", "two-year", "month-to-month", "two-year",
                               "one-year", "two-year", "month-to-month", "one-year",
                               "month-to-month", "two-year"],
            "monthly_charges": [85.0, 45.0, 95.0, 40.0, 65.0, 42.0, 90.0, 60.0, 80.0, 50.0],
            "total_charges": [170.0, None, 475.0, 2400.0, None, 2310.0, 270.0, 1800.0, 1600.0, 2250.0],
            "support_ticket_count": [2, 0, 3, 0, 1, 0, 2, 1, 1, 0],
            "churned": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        }
    )


def test_preprocessor_imputes_nulls(sample_df):
    pre = build_preprocessor()
    X = sample_df.drop(columns=["churned"])
    transformed = pre.fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(dense).any()


def test_preprocessor_one_hot_expands_categories(sample_df):
    pre = build_preprocessor()
    X = sample_df.drop(columns=["churned"])
    transformed = pre.fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    n_numeric = 4
    n_categories = X["contract_type"].nunique()
    assert dense.shape[1] == n_numeric + n_categories


def test_train_test_split_is_stratified(sample_df):
    X_train, X_test, y_train, y_test = train_test_split_data(sample_df, test_size=0.4, random_state=1)
    assert len(X_test) == 4
    assert len(X_train) == 6
    assert set(X_train.index).isdisjoint(set(X_test.index))
    # stratified: both classes must appear on both sides of the split. With
    # only 10 rows a *non*-stratified split could easily put all churners on
    # one side by chance -- this is exactly the failure mode stratifying
    # guards against (see features.py's docstring).
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
