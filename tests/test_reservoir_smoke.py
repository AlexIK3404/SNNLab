import numpy as np

from snnlab.architectures.reservoir import build_reservoir_network
from snnlab.configs.reservoir import ReservoirConfig
from snnlab.core.encoders import PoissonEncoder


def test_small_reservoir_smoke() -> None:
    cfg = ReservoirConfig(
        n_input=4,
        n_reservoir=12,
        simulation_ms=2.0,
        dt_ms=0.5,
        select_k=4,
        use_feature_selection=False,
    )
    network = build_reservoir_network(cfg)
    encoder = PoissonEncoder(n_features=4, max_rate_hz=50.0)
    frames: list[dict] = []
    result = network.extract_features(
        np.array([0.0, 0.3, 0.6, 1.0]),
        encoder=encoder,
        rng=np.random.default_rng(1),
        frame_callback=frames.append,
        frame_every_steps=2,
    )
    assert frames
    assert frames[-1]["step"] == cfg.n_steps
    assert result.features.shape == (12,)
    assert result.spike_counts.shape == (12,)


def test_reservoir_runner_emits_activity_diagnostics(tmp_path) -> None:
    from snnlab.core.events import FrameworkEvent
    from snnlab.experiments.reservoir_runner import ReservoirRunner, ReservoirRunnerState

    class Collector:
        def __init__(self) -> None:
            self.events: list[FrameworkEvent] = []

        def handle(self, event: FrameworkEvent) -> None:
            self.events.append(event)

    cfg = ReservoirConfig(
        seed=3,
        n_input=4,
        n_reservoir=10,
        simulation_ms=4.0,
        dt_ms=0.5,
        input_scale=2.0,
        recurrent_scale=1.0,
        select_k=4,
        use_feature_selection=False,
    )
    network = build_reservoir_network(cfg)
    encoder = PoissonEncoder(n_features=4, max_rate_hz=100.0)
    state = ReservoirRunnerState(
        network=network,
        train_sample_indices=np.asarray([0], dtype=np.int64),
    )
    collector = Collector()
    runner = ReservoirRunner(
        x_train=np.asarray([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32),
        y_train=np.asarray([2], dtype=np.int64),
        cfg=cfg,
        encoder=encoder,
        state=state,
        run_dir=tmp_path,
        observer=collector,
        emit_spike_logs=True,
    )

    runner.collect_one_sample()
    payload = next(event.payload for event in collector.events if event.name == "sample_end")

    assert payload["n_reservoir"] == cfg.n_reservoir
    assert payload["n_steps"] == cfg.n_steps
    assert payload["spike_log"].shape == (cfg.n_steps, cfg.n_reservoir)
    assert payload["excitatory_spikes"] + payload["inhibitory_spikes"] == payload["spikes"]
    assert payload["active_exc"] + payload["active_inh"] == payload["active"]
    assert np.isclose(
        payload["spike_occupancy"],
        payload["spikes"] / (cfg.n_reservoir * cfg.n_steps),
    )
    assert np.isfinite(payload["mean_rate_hz"])
