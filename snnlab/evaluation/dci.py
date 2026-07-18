from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from snnlab.architectures.dci import DCIModel, create_dci_state

HomeostasisMode = Literal["frozen", "zero"]
NetworkStateMode = Literal["fresh_continuous", "trained_continuous"]
PredictionRule = Literal["mean_response", "sum_response", "balanced_topk"]
AssignmentPolicy = Literal["full_train_pool", "exclude_training", "training_subset"]


@dataclass(frozen=True, slots=True)
class DCIEvaluationConfig:
    """
    Configures SNN-native label assignment and evaluation for DCI.

    The decoder never fits an external classifier. It only uses class-wise E
    responses measured from the spiking network. Quality thresholds and a
    per-class top-k cap can suppress silent, weak, or non-selective neurons.

    Настраивает SNN-native назначение меток и оценку DCI.

    Decoder не обучает внешний классификатор. Он использует только отклики
    E-нейронов, измеренные в спайковой сети. Пороги качества и ограничение
    top-k по классам позволяют исключать молчащие, слабые и неселективные
    нейроны.
    """

    assignment_samples: int = 500
    test_samples: int = 200
    seed: int = 52
    assignment_policy: AssignmentPolicy = "full_train_pool"
    homeostasis_mode: HomeostasisMode = "frozen"
    network_state_mode: NetworkStateMode = "fresh_continuous"
    prediction_rule: PredictionRule = "mean_response"

    # EN: Zero values preserve the legacy "assign every active neuron" rule.
    # RU: Нулевые значения сохраняют старое правило «назначать любой активный нейрон».
    min_best_response: float = 0.0
    min_absolute_margin: float = 0.0
    min_relative_margin: float = 0.0
    top_k_per_class: int = 0

    shuffle_assignment: bool = True
    shuffle_test: bool = False

    def response_signature(self) -> tuple[object, ...]:
        """
        Returns the subset of settings that changes simulated E responses.

        Decoder thresholds and voting rules are intentionally excluded, so the
        same expensive response collection can be reused for fast decoder
        experiments.

        Возвращает настройки, которые меняют симулируемые E-отклики.

        Пороги decoder-а и правило голосования намеренно исключены, чтобы один
        дорогой сбор откликов можно было использовать для быстрых экспериментов
        с decoder-ом.
        """
        return (
            int(self.assignment_samples),
            int(self.test_samples),
            int(self.seed),
            str(self.assignment_policy),
            str(self.homeostasis_mode),
            str(self.network_state_mode),
            bool(self.shuffle_assignment),
            bool(self.shuffle_test),
        )


@dataclass(frozen=True, slots=True)
class DCIResponseSet:
    """Stores per-sample E responses. / Хранит E-отклики по sample."""

    responses: np.ndarray
    labels: np.ndarray
    sample_indices: np.ndarray
    accepted: np.ndarray
    attempts: np.ndarray


@dataclass(frozen=True, slots=True)
class DCIEvaluationData:
    """
    Stores expensive simulated responses independently from decoder settings.

    Хранит дорогие симулированные отклики отдельно от настроек decoder-а.
    """

    response_signature: tuple[object, ...]
    assignment_responses: DCIResponseSet
    test_responses: DCIResponseSet
    n_classes: int


@dataclass(frozen=True, slots=True)
class DCILabelAssignment:
    """
    Stores quality-aware class assignment of E neurons.

    Хранит назначение классов E-нейронам с учётом качества.
    """

    neuron_labels: np.ndarray
    candidate_labels: np.ndarray
    class_mean_response: np.ndarray
    best_response: np.ndarray
    second_response: np.ndarray
    selectivity_margin: np.ndarray
    relative_margin: np.ndarray
    silent_mask: np.ndarray
    weak_response_mask: np.ndarray
    low_absolute_margin_mask: np.ndarray
    low_relative_margin_mask: np.ndarray
    pruned_by_top_k_mask: np.ndarray
    class_counts: np.ndarray

    @property
    def assigned_mask(self) -> np.ndarray:
        """Returns neurons retained by the decoder. / Возвращает оставленные decoder-ом нейроны."""
        return self.neuron_labels >= 0

    @property
    def quality_rejected_mask(self) -> np.ndarray:
        """Returns non-silent neurons rejected by quality thresholds.

        Возвращает не-молчащие нейроны, отклонённые порогами качества.
        """
        return (
            self.weak_response_mask | self.low_absolute_margin_mask | self.low_relative_margin_mask
        ) & ~self.silent_mask


@dataclass(frozen=True, slots=True)
class DCIEvaluationDiagnostics:
    """
    Stores additional diagnostics for interpreting DCI evaluation.

    Хранит дополнительные диагностики для интерпретации DCI-evaluation.
    """

    true_counts: np.ndarray
    prediction_counts: np.ndarray
    unclassified_count: int
    per_class_accuracy: np.ndarray
    assigned_count: int
    unassigned_count: int
    silent_count: int
    weak_rejected_count: int
    absolute_margin_rejected_count: int
    relative_margin_rejected_count: int
    quality_rejected_count: int
    top_k_pruned_count: int
    low_selectivity_count: int
    low_selectivity_fraction: float
    weak_response_count: int
    weak_response_fraction: float
    mean_margin: float
    median_margin: float
    mean_relative_margin: float
    median_relative_margin: float
    mean_best_response: float
    median_best_response: float
    margin_low_threshold: float
    weak_response_threshold: float
    sorted_neuron_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class DCIEvaluationResult:
    """Complete DCI classification evaluation result. / Полный результат оценки DCI."""

    config: DCIEvaluationConfig
    assignment: DCILabelAssignment
    y_true: np.ndarray
    y_pred: np.ndarray
    class_scores: np.ndarray
    confusion_matrix: np.ndarray
    accuracy: float
    assigned_fraction: float
    accepted_fraction_assignment: float
    accepted_fraction_test: float
    diagnostics: DCIEvaluationDiagnostics
    response_data: DCIEvaluationData


def _sample_order(
    n_items: int,
    n_samples: int,
    *,
    seed: int,
    shuffle: bool,
) -> np.ndarray:
    n_samples = min(max(0, int(n_samples)), int(n_items))
    indices = np.arange(n_items, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    return indices[:n_samples]


def _clone_for_evaluation(
    model: DCIModel,
    *,
    config: DCIEvaluationConfig,
    seed_offset: int,
) -> DCIModel:
    """
    Creates an isolated evaluation model without changing the trainable model.

    Создаёт изолированную evaluation-модель, не изменяя обучаемую модель.
    """
    clone = copy.deepcopy(model)

    if config.network_state_mode == "fresh_continuous":
        clone.network_state = create_dci_state(
            clone.cfg,
            seed=clone.cfg.seed + int(seed_offset),
        )
    elif config.network_state_mode != "trained_continuous":
        raise ValueError(f"Unsupported network_state_mode: {config.network_state_mode!r}")

    if config.homeostasis_mode == "zero":
        clone.homeostasis_state.exc_current.fill(0.0)
    elif config.homeostasis_mode != "frozen":
        raise ValueError(f"Unsupported homeostasis_mode: {config.homeostasis_mode!r}")

    # EN: Freeze the sample-level current during evaluation. The simulation
    #     still reads the current, but neither decay nor adaptation changes it.
    # RU: Замораживаем sample-level ток во время evaluation. Симуляция всё ещё
    #     использует ток, но ни затухание, ни адаптация его не изменяют.
    clone.homeostasis_cfg = replace(
        clone.homeostasis_cfg,
        tau_ms=float("inf"),
        learning_rate=0.0,
    )
    return clone


def collect_dci_responses(
    model: DCIModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    shuffle: bool,
    config: DCIEvaluationConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
    phase: str = "responses",
    seed_offset: int = 10_000,
) -> DCIResponseSet:
    """
    Collects final-attempt E spike counts without learning.

    Собирает E spike counts финальной попытки без обучения.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    indices = _sample_order(len(x), n_samples, seed=seed, shuffle=shuffle)
    eval_model = _clone_for_evaluation(model, config=config, seed_offset=seed_offset)
    rng = np.random.default_rng(seed + seed_offset + 1)

    responses = np.zeros((len(indices), model.cfg.n_exc), dtype=np.float64)
    accepted = np.zeros(len(indices), dtype=bool)
    attempts = np.zeros(len(indices), dtype=np.int64)

    for position, sample_index in enumerate(indices, start=1):
        presentation = eval_model.simulate_image_no_learning(
            x[int(sample_index)],
            rng=rng,
        )
        responses[position - 1] = np.sum(
            presentation.stimulus_result.exc_spikes,
            axis=0,
        )
        accepted[position - 1] = bool(presentation.accepted)
        attempts[position - 1] = int(presentation.accepted_attempt + 1)
        if progress_callback is not None:
            progress_callback(position, len(indices), phase)

    return DCIResponseSet(
        responses=responses,
        labels=y[indices].astype(np.int64, copy=True),
        sample_indices=indices,
        accepted=accepted,
        attempts=attempts,
    )


def collect_dci_evaluation_data(
    model: DCIModel,
    x_assignment: np.ndarray,
    y_assignment: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    config: DCIEvaluationConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> DCIEvaluationData:
    """
    Simulates assignment and test responses once for later decoder experiments.

    Один раз симулирует assignment- и test-отклики для последующих экспериментов
    с decoder-ом без повторного запуска SNN.
    """
    n_classes = int(max(np.max(y_assignment), np.max(y_test))) + 1
    assignment_responses = collect_dci_responses(
        model,
        x_assignment,
        y_assignment,
        n_samples=config.assignment_samples,
        seed=config.seed,
        shuffle=config.shuffle_assignment,
        config=config,
        progress_callback=progress_callback,
        phase="assignment",
        seed_offset=10_000,
    )
    test_responses = collect_dci_responses(
        model,
        x_test,
        y_test,
        n_samples=config.test_samples,
        seed=config.seed + 1,
        shuffle=config.shuffle_test,
        config=config,
        progress_callback=progress_callback,
        phase="test",
        seed_offset=20_000,
    )
    return DCIEvaluationData(
        response_signature=config.response_signature(),
        assignment_responses=assignment_responses,
        test_responses=test_responses,
        n_classes=n_classes,
    )


def assign_labels_to_exc_neurons(
    responses: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int,
    min_best_response: float = 0.0,
    min_absolute_margin: float = 0.0,
    min_relative_margin: float = 0.0,
    top_k_per_class: int = 0,
) -> DCILabelAssignment:
    """
    Assigns E neurons using explicit quality filters and an optional class cap.

    A neuron is first associated with the class that produces its largest mean
    response. It is then rejected when it is silent, too weak, insufficiently
    separated from its second-best class, or outside the best ``top_k`` neurons
    of that class.

    Назначает E-нейроны классам с явными фильтрами качества и необязательным
    ограничением числа нейронов на класс.

    Сначала нейрон связывается с классом максимального среднего отклика. Затем
    он исключается, если молчит, имеет слишком слабый отклик, недостаточно
    отделён от второго класса или не входит в лучшие ``top_k`` своего класса.
    """
    responses = np.asarray(responses, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if responses.ndim != 2 or len(responses) != len(labels):
        raise ValueError("responses must be [samples, neurons] and match labels")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive")

    min_best_response = max(0.0, float(min_best_response))
    min_absolute_margin = max(0.0, float(min_absolute_margin))
    min_relative_margin = max(0.0, float(min_relative_margin))
    top_k_per_class = max(0, int(top_k_per_class))

    class_mean = np.zeros((n_classes, responses.shape[1]), dtype=np.float64)
    for class_id in range(n_classes):
        mask = labels == class_id
        if np.any(mask):
            class_mean[class_id] = np.mean(responses[mask], axis=0)

    candidate_labels = np.argmax(class_mean, axis=0).astype(np.int64)
    best = np.max(class_mean, axis=0)

    if n_classes > 1:
        partitioned = np.partition(class_mean, kth=n_classes - 2, axis=0)
        second = partitioned[-2]
    else:
        second = np.zeros_like(best)

    margin = best - second
    relative_margin = np.divide(
        margin,
        np.maximum(best, 1e-12),
        out=np.zeros_like(margin),
        where=best > 0.0,
    )

    silent = best <= 0.0
    weak = (~silent) & (best < min_best_response)
    low_absolute = (~silent) & (margin < min_absolute_margin)
    low_relative = (~silent) & (relative_margin < min_relative_margin)

    neuron_labels = candidate_labels.copy()
    neuron_labels[silent | weak | low_absolute | low_relative] = -1
    pruned_by_top_k = np.zeros_like(silent)

    if top_k_per_class > 0:
        for class_id in range(n_classes):
            indices = np.flatnonzero(neuron_labels == class_id)
            if indices.size <= top_k_per_class:
                continue

            # EN: Rank primarily by relative selectivity and secondarily by
            #     response strength. This is deterministic and uses no trained
            #     external weights.
            # RU: Сортируем прежде всего по относительной селективности, затем
            #     по силе отклика. Алгоритм детерминирован и не использует
            #     обучаемые внешние веса.
            local_order = np.lexsort((-best[indices], -relative_margin[indices]))
            rejected = indices[local_order[top_k_per_class:]]
            neuron_labels[rejected] = -1
            pruned_by_top_k[rejected] = True

    class_counts = np.array(
        [np.count_nonzero(neuron_labels == class_id) for class_id in range(n_classes)],
        dtype=np.int64,
    )
    return DCILabelAssignment(
        neuron_labels=neuron_labels,
        candidate_labels=candidate_labels,
        class_mean_response=class_mean,
        best_response=best,
        second_response=second,
        selectivity_margin=margin,
        relative_margin=relative_margin,
        silent_mask=silent,
        weak_response_mask=weak,
        low_absolute_margin_mask=low_absolute,
        low_relative_margin_mask=low_relative,
        pruned_by_top_k_mask=pruned_by_top_k,
        class_counts=class_counts,
    )


def predict_from_assignment(
    responses: np.ndarray,
    assignment: DCILabelAssignment,
    *,
    n_classes: int,
    rule: PredictionRule,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts classes directly from grouped E-neuron spike responses.

    ``balanced_topk`` normalizes each retained neuron's response by its own
    assignment-set best response and weights it by relative selectivity. Scores
    are weighted means, so classes with more neurons receive no automatic
    advantage.

    Предсказывает классы непосредственно по сгруппированным спайковым откликам
    E-нейронов.

    ``balanced_topk`` нормирует отклик каждого оставленного нейрона на его
    лучший отклик в assignment-наборе и взвешивает относительной селективностью.
    Используется взвешенное среднее, поэтому большее число нейронов не даёт
    классу автоматического преимущества.
    """
    responses = np.asarray(responses, dtype=np.float64)
    scores = np.full((len(responses), n_classes), -np.inf, dtype=np.float64)

    for class_id in range(n_classes):
        mask = assignment.neuron_labels == class_id
        if not np.any(mask):
            continue
        selected = responses[:, mask]

        if rule == "mean_response":
            scores[:, class_id] = np.mean(selected, axis=1)
        elif rule == "sum_response":
            scores[:, class_id] = np.sum(selected, axis=1)
        elif rule == "balanced_topk":
            normalization = np.maximum(assignment.best_response[mask], 1e-12)
            normalized = selected / normalization[None, :]
            weights = np.clip(assignment.relative_margin[mask], 0.0, 1.0)
            if not np.any(weights > 0.0):
                weights = np.ones_like(weights)
            scores[:, class_id] = np.sum(normalized * weights[None, :], axis=1) / np.sum(weights)
        else:
            raise ValueError(f"Unsupported prediction rule: {rule!r}")

    # EN: When no class owns any active neuron, return -1 instead of silently
    #     mapping the sample to class zero through argmax(-inf, ...).
    # RU: Если ни один класс не имеет активного назначенного нейрона, возвращаем
    #     -1, а не молча отображаем sample в класс 0 через argmax(-inf, ...).
    assigned_mask = assignment.neuron_labels >= 0
    active_any = (
        np.sum(responses[:, assigned_mask], axis=1) > 0.0
        if np.any(assigned_mask)
        else np.zeros(len(responses), dtype=bool)
    )
    finite_any = np.any(np.isfinite(scores), axis=1) & active_any
    predictions = np.full(len(responses), -1, dtype=np.int64)
    predictions[finite_any] = np.argmax(scores[finite_any], axis=1)
    return predictions, scores


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        if 0 <= int(true_label) < n_classes and 0 <= int(predicted_label) < n_classes:
            matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def _build_diagnostics(
    *,
    assignment: DCILabelAssignment,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confusion: np.ndarray,
    n_classes: int,
) -> DCIEvaluationDiagnostics:
    """
    Builds diagnostic summaries that explain why an evaluation behaves as it does.

    Строит диагностические сводки, объясняющие поведение evaluation.
    """
    true_counts = np.array(
        [np.count_nonzero(y_true == class_id) for class_id in range(n_classes)],
        dtype=np.int64,
    )
    prediction_counts = np.array(
        [np.count_nonzero(y_pred == class_id) for class_id in range(n_classes)],
        dtype=np.int64,
    )
    unclassified_count = int(np.count_nonzero(y_pred < 0))

    per_class_accuracy = np.full(n_classes, np.nan, dtype=np.float64)
    for class_id in range(n_classes):
        if true_counts[class_id] > 0:
            per_class_accuracy[class_id] = confusion[class_id, class_id] / true_counts[class_id]

    assigned_mask = assignment.assigned_mask
    assigned_count = int(np.count_nonzero(assigned_mask))
    unassigned_count = int(len(assignment.neuron_labels) - assigned_count)
    silent_count = int(np.count_nonzero(assignment.silent_mask))
    weak_rejected_count = int(
        np.count_nonzero(assignment.weak_response_mask & ~assignment.silent_mask)
    )
    absolute_margin_rejected_count = int(
        np.count_nonzero(assignment.low_absolute_margin_mask & ~assignment.silent_mask)
    )
    relative_margin_rejected_count = int(
        np.count_nonzero(assignment.low_relative_margin_mask & ~assignment.silent_mask)
    )
    quality_rejected_count = int(np.count_nonzero(assignment.quality_rejected_mask))
    top_k_pruned_count = int(np.count_nonzero(assignment.pruned_by_top_k_mask))

    assigned_margins = assignment.selectivity_margin[assigned_mask]
    assigned_relative = assignment.relative_margin[assigned_mask]
    assigned_best = assignment.best_response[assigned_mask]

    if assigned_margins.size:
        # EN: This threshold is a diagnostic marker, not an assignment filter.
        # RU: Этот порог — диагностический маркер, а не фильтр назначения.
        margin_low_threshold = float(max(1e-12, 0.1 * np.nanmax(assigned_margins)))
        low_selectivity_count = int(np.count_nonzero(assigned_margins <= margin_low_threshold))
        mean_margin = float(np.nanmean(assigned_margins))
        median_margin = float(np.nanmedian(assigned_margins))
        mean_relative_margin = float(np.nanmean(assigned_relative))
        median_relative_margin = float(np.nanmedian(assigned_relative))
    else:
        margin_low_threshold = 0.0
        low_selectivity_count = 0
        mean_margin = float("nan")
        median_margin = float("nan")
        mean_relative_margin = float("nan")
        median_relative_margin = float("nan")

    if assigned_best.size:
        weak_response_threshold = float(max(1e-12, 0.1 * np.nanmax(assigned_best)))
        weak_response_count = int(np.count_nonzero(assigned_best <= weak_response_threshold))
        mean_best = float(np.nanmean(assigned_best))
        median_best = float(np.nanmedian(assigned_best))
    else:
        weak_response_threshold = 0.0
        weak_response_count = 0
        mean_best = float("nan")
        median_best = float("nan")

    labels_for_sort = np.where(
        assignment.neuron_labels >= 0, assignment.neuron_labels, n_classes + 1
    )
    sorted_indices = np.lexsort((-assignment.relative_margin, labels_for_sort)).astype(np.int64)

    denominator = max(assigned_count, 1)
    return DCIEvaluationDiagnostics(
        true_counts=true_counts,
        prediction_counts=prediction_counts,
        unclassified_count=unclassified_count,
        per_class_accuracy=per_class_accuracy,
        assigned_count=assigned_count,
        unassigned_count=unassigned_count,
        silent_count=silent_count,
        weak_rejected_count=weak_rejected_count,
        absolute_margin_rejected_count=absolute_margin_rejected_count,
        relative_margin_rejected_count=relative_margin_rejected_count,
        quality_rejected_count=quality_rejected_count,
        top_k_pruned_count=top_k_pruned_count,
        low_selectivity_count=low_selectivity_count,
        low_selectivity_fraction=float(low_selectivity_count / denominator),
        weak_response_count=weak_response_count,
        weak_response_fraction=float(weak_response_count / denominator),
        mean_margin=mean_margin,
        median_margin=median_margin,
        mean_relative_margin=mean_relative_margin,
        median_relative_margin=median_relative_margin,
        mean_best_response=mean_best,
        median_best_response=median_best,
        margin_low_threshold=margin_low_threshold,
        weak_response_threshold=weak_response_threshold,
        sorted_neuron_indices=sorted_indices,
    )


def decode_dci_evaluation_data(
    response_data: DCIEvaluationData,
    *,
    config: DCIEvaluationConfig,
) -> DCIEvaluationResult:
    """
    Applies a cheap SNN-native decoder to already collected E responses.

    Применяет быстрый SNN-native decoder к уже собранным E-откликам.
    """
    if response_data.response_signature != config.response_signature():
        raise ValueError("Cached response data does not match the evaluation protocol")

    assignment_responses = response_data.assignment_responses
    test_responses = response_data.test_responses
    assignment = assign_labels_to_exc_neurons(
        assignment_responses.responses,
        assignment_responses.labels,
        n_classes=response_data.n_classes,
        min_best_response=config.min_best_response,
        min_absolute_margin=config.min_absolute_margin,
        min_relative_margin=config.min_relative_margin,
        top_k_per_class=config.top_k_per_class,
    )
    y_pred, class_scores = predict_from_assignment(
        test_responses.responses,
        assignment,
        n_classes=response_data.n_classes,
        rule=config.prediction_rule,
    )
    y_true = test_responses.labels
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else float("nan")
    confusion = _confusion_matrix(y_true, y_pred, response_data.n_classes)
    diagnostics = _build_diagnostics(
        assignment=assignment,
        y_true=y_true,
        y_pred=y_pred,
        confusion=confusion,
        n_classes=response_data.n_classes,
    )

    return DCIEvaluationResult(
        config=config,
        assignment=assignment,
        y_true=y_true,
        y_pred=y_pred,
        class_scores=class_scores,
        confusion_matrix=confusion,
        accuracy=accuracy,
        assigned_fraction=float(np.mean(assignment.assigned_mask)),
        accepted_fraction_assignment=(
            float(np.mean(assignment_responses.accepted))
            if len(assignment_responses.accepted)
            else float("nan")
        ),
        accepted_fraction_test=(
            float(np.mean(test_responses.accepted))
            if len(test_responses.accepted)
            else float("nan")
        ),
        diagnostics=diagnostics,
        response_data=response_data,
    )


def evaluate_dci_classifier(
    model: DCIModel,
    x_assignment: np.ndarray,
    y_assignment: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    config: DCIEvaluationConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
    response_data: DCIEvaluationData | None = None,
) -> DCIEvaluationResult:
    """
    Runs isolated response collection and SNN-native DCI decoding.

    When compatible ``response_data`` is supplied, only the decoder is
    recalculated. This makes threshold and top-k experiments nearly instant.

    Выполняет изолированный сбор откликов и SNN-native декодирование DCI.

    Если переданы совместимые ``response_data``, пересчитывается только decoder.
    Поэтому эксперименты с порогами и top-k выполняются почти мгновенно.
    """
    if response_data is None:
        response_data = collect_dci_evaluation_data(
            model,
            x_assignment,
            y_assignment,
            x_test,
            y_test,
            config=config,
            progress_callback=progress_callback,
        )
    return decode_dci_evaluation_data(response_data, config=config)
