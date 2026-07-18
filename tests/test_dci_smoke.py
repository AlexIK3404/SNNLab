import numpy as np

from snnlab.architectures.dci import build_dci_model
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
