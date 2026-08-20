import numpy as np
import pandas as pd


def classification_metrics_from_oof(oof: pd.DataFrame) -> dict[str, float]:
    y = oof["y_true"].to_numpy(dtype=int)
    p = oof["y_pred"].to_numpy(dtype=int)

    tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))
    tp = np.sum((y == 1) & (p == 1))

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "recall_from_oof": float(recall),
        "specificity_from_oof": float(specificity),
        "precision_from_oof": float(precision),
        "f1_from_oof": float(f1),
        "balanced_accuracy_from_oof": float(balanced_accuracy),
    }


def balanced_accuracy_from_oof(oof: pd.DataFrame) -> float:
    return classification_metrics_from_oof(oof)["balanced_accuracy_from_oof"]
