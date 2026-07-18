import numpy as np

from snnlab.evaluation.dci import (
    DCIEvaluationConfig,
    DCIEvaluationData,
    DCIResponseSet,
    assign_labels_to_exc_neurons,
    decode_dci_evaluation_data,
    predict_from_assignment,
)


def test_quality_filters_and_top_k_are_applied_per_class() -> None:
    """
    Keeps selective neurons, rejects ambiguous/silent ones, and enforces top-k.

    Сохраняет селективные нейроны, исключает неоднозначные/молчащие и
    применяет top-k отдельно для каждого класса.
    """
    # Classes 0 and 1, six E neurons.
    responses = np.array(
        [
            [4.0, 3.0, 1.0, 0.0, 0.2, 0.0],
            [4.0, 2.5, 1.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 1.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    assignment = assign_labels_to_exc_neurons(
        responses,
        labels,
        n_classes=2,
        min_relative_margin=0.20,
        top_k_per_class=1,
    )

    # Neuron 0 is the best class-0 specialist, neuron 3 class-1 specialist.
    assert assignment.neuron_labels.tolist() == [0, -1, -1, 1, -1, -1]
    assert assignment.class_counts.tolist() == [1, 1]
    assert assignment.silent_mask[-1]
    assert assignment.low_relative_margin_mask[2]
    assert np.count_nonzero(assignment.pruned_by_top_k_mask) == 2


def test_balanced_top_k_vote_avoids_class_size_bias() -> None:
    """
    Balanced voting must not reward a class merely for owning more neurons.

    Сбалансированное голосование не должно награждать класс только за большее
    число назначенных нейронов.
    """
    assignment_responses = np.array(
        [
            [2.0, 2.0, 2.0, 0.0],
            [2.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    assignment = assign_labels_to_exc_neurons(
        assignment_responses,
        labels,
        n_classes=2,
    )

    test_responses = np.array([[1.0, 1.0, 1.0, 3.0]], dtype=np.float64)
    pred_sum, _ = predict_from_assignment(
        test_responses,
        assignment,
        n_classes=2,
        rule="sum_response",
    )
    pred_balanced, _ = predict_from_assignment(
        test_responses,
        assignment,
        n_classes=2,
        rule="balanced_topk",
    )

    assert pred_sum.tolist() == [0]
    assert pred_balanced.tolist() == [1]


def test_decoder_can_be_recomputed_from_cached_responses() -> None:
    """
    Decoder settings can change without rerunning the spiking simulation.

    Настройки decoder-а можно менять без повторного запуска спайковой симуляции.
    """
    assignment_set = DCIResponseSet(
        responses=np.array(
            [
                [3.0, 0.0],
                [2.0, 0.0],
                [0.0, 4.0],
                [0.0, 5.0],
            ],
            dtype=np.float64,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        sample_indices=np.arange(4, dtype=np.int64),
        accepted=np.ones(4, dtype=bool),
        attempts=np.ones(4, dtype=np.int64),
    )
    test_set = DCIResponseSet(
        responses=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        labels=np.array([0, 1], dtype=np.int64),
        sample_indices=np.arange(2, dtype=np.int64),
        accepted=np.ones(2, dtype=bool),
        attempts=np.ones(2, dtype=np.int64),
    )
    config = DCIEvaluationConfig(
        assignment_samples=4,
        test_samples=2,
        seed=52,
        prediction_rule="balanced_topk",
        min_relative_margin=0.1,
        top_k_per_class=1,
        shuffle_assignment=True,
        shuffle_test=False,
    )
    data = DCIEvaluationData(
        response_signature=config.response_signature(),
        assignment_responses=assignment_set,
        test_responses=test_set,
        n_classes=2,
    )

    result = decode_dci_evaluation_data(data, config=config)

    assert result.accuracy == 1.0
    assert result.response_data is data
    assert result.assignment.class_counts.tolist() == [1, 1]
