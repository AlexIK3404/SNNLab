import numpy as np

from snnlab.architectures.dci import build_dci_model
from snnlab.configs.dci import (
    DCIConfig,
    DCIHomeostasisConfig,
    DCIPresentationConfig,
    DCISTDPConfig,
    make_dci_dynamics,
)
from snnlab.evaluation.dci import DCIEvaluationConfig, evaluate_dci_classifier


def test_dci_evaluation_does_not_mutate_training_model() -> None:
    cfg = DCIConfig(
        seed=9,
        n_input=8,
        n_exc=6,
        n_inh=6,
        dt_ms=0.5,
        stimulus_ms=2.0,
        rest_ms=1.0,
        refractory_exc_ms=0.5,
        refractory_inh_ms=0.5,
        input_weight_sum=1.0,
    )
    model = build_dci_model(
        cfg=cfg,
        dynamics=make_dci_dynamics(cfg, target_total_inhibition=2.0),
        presentation_cfg=DCIPresentationConfig(
            base_max_rate_hz=250.0,
            rate_increment_hz=100.0,
            min_exc_spikes=0,
            max_attempts=1,
        ),
        stdp_cfg=DCISTDPConfig(w_max=0.50),
        homeostasis_cfg=DCIHomeostasisConfig(),
    )
    x = np.linspace(0.0, 1.0, 80, dtype=np.float64).reshape(10, 8)
    y = np.arange(10, dtype=np.int64) % 2

    weights_before = model.connectivity.w_input_exc.copy()
    exc_v_before = model.network_state.exc.v.copy()
    homeo_before = model.homeostasis_state.exc_current.copy()

    result = evaluate_dci_classifier(
        model,
        x,
        y,
        x,
        y,
        config=DCIEvaluationConfig(
            assignment_samples=6,
            test_samples=4,
            seed=3,
        ),
    )

    assert result.confusion_matrix.shape == (2, 2)
    assert 0.0 <= result.assigned_fraction <= 1.0
    np.testing.assert_array_equal(model.connectivity.w_input_exc, weights_before)
    np.testing.assert_array_equal(model.network_state.exc.v, exc_v_before)
    np.testing.assert_array_equal(model.homeostasis_state.exc_current, homeo_before)
