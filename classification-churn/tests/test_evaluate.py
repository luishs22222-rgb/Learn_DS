import numpy as np

from customer_churn.evaluate import compute_classification_metrics


def test_metrics_match_hand_computed_values():
    # 10 customers: 4 actually churned (1), 6 didn't (0).
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # Predictions: catches 2 of 4 churners (2 false negatives), and
    # incorrectly flags 1 non-churner (1 false positive).
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    # Probabilities consistent with those hard predictions, used for ROC-AUC.
    y_proba = np.array([0.9, 0.8, 0.4, 0.3, 0.6, 0.2, 0.1, 0.15, 0.05, 0.25])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    # Confusion matrix by hand: TP=2, FN=2, FP=1, TN=5.
    assert metrics["true_positives"] == 2.0
    assert metrics["false_negatives"] == 2.0
    assert metrics["false_positives"] == 1.0
    assert metrics["true_negatives"] == 5.0

    # accuracy = (TP+TN)/total = 7/10
    assert metrics["accuracy"] == 0.7
    # precision = TP/(TP+FP) = 2/3
    assert abs(metrics["precision"] - 2 / 3) < 1e-9
    # recall = TP/(TP+FN) = 2/4 = 0.5
    assert metrics["recall"] == 0.5
    # f1 = 2*P*R/(P+R) = 2*(2/3)*(0.5)/((2/3)+0.5) = 0.5714...
    assert abs(metrics["f1"] - 0.5714285714) < 1e-6


def test_perfect_ranking_gives_roc_auc_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    # Every positive scored strictly higher than every negative.
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_always_predict_negative_has_high_accuracy_but_zero_recall():
    """The exact failure mode accuracy hides under class imbalance -- this is
    why train.py promotes on ROC-AUC, not accuracy (see evaluate.py docstring
    and the README's metrics section)."""
    y_true = np.array([0] * 8 + [1] * 2)  # 20% churn rate
    y_pred = np.zeros_like(y_true)  # always predicts "no churn"
    y_proba = np.zeros_like(y_true, dtype=float)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 0.8  # looks good...
    assert metrics["recall"] == 0.0  # ...but it never catches a single churner
