from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ReservoirEvaluationResult:
    """Stores reservoir classification metrics. / Хранит метрики reservoir-классификации."""

    y_true: np.ndarray
    y_pred: np.ndarray
    confusion_matrix: np.ndarray
    accuracy: float


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        if 0 <= int(true_label) < n_classes and 0 <= int(predicted_label) < n_classes:
            matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def evaluate_reservoir_classifier(
    runner,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ReservoirEvaluationResult:
    """
    Evaluates a fitted reservoir readout on a deterministic test subset.

    Оценивает обученный reservoir-readout на детерминированной test-подвыборке.
    """
    if runner.state.readout_model is None:
        raise RuntimeError("Reservoir readout is not fitted")

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    n_samples = min(max(0, int(n_samples)), len(x_test))
    indices = np.arange(len(x_test), dtype=np.int64)[:n_samples]

    # EN: Use the runner prediction method because it already isolates the RNG
    #     stream from later continued training.
    # RU: Используем runner.predict, потому что он уже изолирует RNG-поток от
    #     последующего дообучения.
    y_pred = runner.predict(x_test[indices], seed=seed)
    if progress_callback is not None:
        progress_callback(n_samples, n_samples, "test")
    y_true = y_test[indices].astype(np.int64, copy=True)
    n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1 if len(y_true) else 0
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    return ReservoirEvaluationResult(
        y_true=y_true,
        y_pred=np.asarray(y_pred, dtype=np.int64),
        confusion_matrix=_confusion_matrix(y_true, y_pred, n_classes),
        accuracy=accuracy,
    )
