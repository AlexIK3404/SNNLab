import numpy as np

from snnlab.architectures.dci import NeuronPopulationState, _population_step, build_dci_model
from snnlab.configs.dci import (
    DCIConfig,
    DCIHomeostasisConfig,
    DCIPresentationConfig,
    DCISTDPConfig,
    make_dci_dynamics,
)


def test_small_dci_train_sample_smoke() -> None:
    cfg = DCIConfig(
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
            base_max_rate_hz=200.0,
            rate_increment_hz=100.0,
            min_exc_spikes=0,
            max_attempts=1,
        ),
        stdp_cfg=DCISTDPConfig(w_max=0.50),
        homeostasis_cfg=DCIHomeostasisConfig(),
    )
    frames: list[dict] = []
    metrics = model.train_one_sample(
        np.linspace(0.0, 1.0, 8),
        rng=np.random.default_rng(2),
        frame_callback=frames.append,
        frame_every_steps=2,
    )
    assert frames
    assert {frame["phase"] for frame in frames} == {"stimulus", "rest"}
    assert metrics["exc_counts"].shape == (6,)
    assert metrics["inh_counts"].shape == (6,)
    assert np.isclose(np.sum(model.connectivity.w_input_exc, axis=0), 1.0).all()


def test_implicit_dci_population_produces_spikes() -> None:
    v = np.full(6, -65.0)
    state = NeuronPopulationState(
        v=v, u=0.2 * v, g_exc=np.zeros(6), g_inh=np.zeros(6),
        refractory_left_ms=np.zeros(6),
    )
    spike_count = 0
    for _ in range(300):
        spikes = _population_step(
            state, a=0.02, b=0.2, c=-65.0, d=8.0, v_peak=30.0,
            refractory_ms=0.4, dt_ms=0.4, e_exc=0.0, e_inh=-80.0,
            bias_current=12.0, numerical_v_min=-500.0, numerical_v_max=500.0,
            numerical_method="implicit_euler",
        )
        spike_count += int(spikes.sum())
        assert np.all(np.isfinite(state.v)) and np.all(np.isfinite(state.u))
    assert spike_count > 0
