import numpy as np

from house_prices.evaluate import compute_regression_metrics


def test_metrics_match_hand_computed_values():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 320.0, 380.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    expected_mae = np.mean(np.abs(y_true - y_pred))  # (10+10+20+20)/4 = 15
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))  # sqrt((100+100+400+400)/4) = sqrt(250)

    assert metrics["mae"] == expected_mae == 15.0
    assert abs(metrics["rmse"] - expected_rmse) < 1e-9
    assert 0.0 <= metrics["r2"] <= 1.0


def test_perfect_predictions_give_zero_error_and_r2_one():
    y_true = np.array([10.0, 20.0, 30.0])
    metrics = compute_regression_metrics(y_true, y_true)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
