from __future__ import annotations

import numpy as np


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes classification accuracy.

    Вычисляет accuracy классификации.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred))


def top_share(values: np.ndarray, k: int) -> float:
    """
    Returns the fraction of total activity held by the top-k entries.

    Возвращает долю общей активности, приходящуюся на top-k элементов.
    """
    values = np.asarray(values, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    return float(np.sum(np.sort(values)[::-1][:k]) / total)
