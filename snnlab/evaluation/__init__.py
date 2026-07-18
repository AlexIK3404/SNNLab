"""Evaluation pipelines for trained SNN models. / Пайплайны оценки обученных SNN-моделей."""

from .dci import (
    DCIEvaluationConfig,
    DCIEvaluationData,
    DCIEvaluationResult,
    DCILabelAssignment,
    collect_dci_evaluation_data,
    decode_dci_evaluation_data,
    evaluate_dci_classifier,
)
from .reservoir import ReservoirEvaluationResult, evaluate_reservoir_classifier

__all__ = [
    "DCIEvaluationConfig",
    "DCIEvaluationData",
    "DCIEvaluationResult",
    "DCILabelAssignment",
    "collect_dci_evaluation_data",
    "decode_dci_evaluation_data",
    "evaluate_dci_classifier",
    "ReservoirEvaluationResult",
    "evaluate_reservoir_classifier",
]
