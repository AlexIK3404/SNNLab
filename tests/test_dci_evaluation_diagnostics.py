import numpy as np

from snnlab.evaluation.dci import (
    _build_diagnostics,
    _confusion_matrix,
    assign_labels_to_exc_neurons,
    predict_from_assignment,
)


def test_dci_evaluation_diagnostics_are_consistent() -> None:
    """
    Checks diagnostic counters derived from DCI assignment and predictions.

    Проверяет согласованность диагностических счётчиков, построенных по
    назначению E-нейронов и предсказаниям.
    """
    responses = np.array(
        [
            [3.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 4.0, 0.0, 1.0],
            [0.0, 5.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    assignment = assign_labels_to_exc_neurons(responses, labels, n_classes=2)
    test_responses = np.array(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 3.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_true = np.array([0, 1, 1], dtype=np.int64)
    y_pred, _ = predict_from_assignment(
        test_responses,
        assignment,
        n_classes=2,
        rule="mean_response",
    )
    confusion = _confusion_matrix(y_true, y_pred, n_classes=2)
    diagnostics = _build_diagnostics(
        assignment=assignment,
        y_true=y_true,
        y_pred=y_pred,
        confusion=confusion,
        n_classes=2,
    )

    assert diagnostics.true_counts.tolist() == [1, 2]
    assert diagnostics.prediction_counts.tolist() == [1, 1]
    assert diagnostics.unclassified_count == 1
    assert diagnostics.silent_count == 1
    assert diagnostics.assigned_count == 3
    assert diagnostics.unassigned_count == 1
    assert diagnostics.per_class_accuracy[0] == 1.0
    assert diagnostics.per_class_accuracy[1] == 0.5
    assert diagnostics.sorted_neuron_indices.shape == (4,)
