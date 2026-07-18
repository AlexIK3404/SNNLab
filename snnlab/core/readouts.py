from __future__ import annotations

from typing import Any

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC


def build_readout(
    *,
    kind: str,
    use_feature_selection: bool = False,
    select_k: int | None = None,
) -> Any:
    """
    Builds an external readout model for reservoir features.

    Создаёт внешний readout для признаков резервуара.
    """
    if kind == "ridge":
        estimator = RidgeClassifier(alpha=1.0)
    elif kind == "logreg":
        estimator = LogisticRegression(max_iter=2000, solver="lbfgs")
    elif kind == "linear_svm":
        estimator = LinearSVC(C=1.0, max_iter=5000)
    elif kind == "rbf_svm":
        estimator = SVC(C=2.0, gamma="scale", kernel="rbf")
    else:
        raise ValueError(f"Unknown readout {kind!r}. Available: ridge, logreg, linear_svm, rbf_svm")

    if not use_feature_selection:
        return estimator

    if select_k is None or select_k <= 0:
        raise ValueError("select_k must be positive when feature selection is enabled")

    return Pipeline(
        [
            ("select", SelectKBest(score_func=f_classif, k=select_k)),
            ("readout", estimator),
        ]
    )
