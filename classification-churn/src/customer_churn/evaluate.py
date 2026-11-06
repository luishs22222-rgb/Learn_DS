"""Classification evaluation metrics.

Note the distinction from training: LogisticRegression/RandomForestClassifier
internally minimize a *loss* during fit() (log-loss / Gini impurity). These
functions compute *metrics* -- chosen for human interpretability, not for
being optimizable -- to judge the result afterward. See docs/glossary.md
("Loss function vs. evaluation metric").

Accuracy is included but is NOT the promotion metric (see train.py) --
with only ~15-25% of customers churning, a model that always predicts "no
churn" would still score ~75-85% accuracy while being useless. ROC-AUC is
used instead: it measures how well the model *ranks* churners above
non-churners across every possible decision threshold, independent of the
class imbalance. See docs/glossary.md and this project's README for the
full walkthrough with real numbers.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "true_negatives": float(tn),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "true_positives": float(tp),
    }
