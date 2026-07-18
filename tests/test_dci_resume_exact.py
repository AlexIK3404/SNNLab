from pathlib import Path

import numpy as np

from snnlab.architectures.dci import build_dci_model
from snnlab.configs.dci import (
    DCIConfig,
    DCIHomeostasisConfig,
    DCIPresentationConfig,
    DCISTDPConfig,
    make_dci_dynamics,
)
from snnlab.experiments.dci_trainer import DCITrainer, DCITrainerState
from snnlab.runtime.checkpoint import CheckpointManager


def _build_model() -> object:
    cfg = DCIConfig(
        seed=7,
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
    return build_dci_model(
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


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    x = np.linspace(0.0, 1.0, 48, dtype=np.float64).reshape(6, 8)
    y = np.arange(6, dtype=np.int64) % 2
    schedule = np.arange(6, dtype=np.int64)

    full_state = DCITrainerState(model=_build_model(), sample_indices=schedule.copy())
    full = DCITrainer(
        x_train=x,
        y_train=y,
        state=full_state,
        run_dir=tmp_path / "full",
        checkpoint_every=0,
    )
    full.train()

    split_state = DCITrainerState(model=_build_model(), sample_indices=schedule.copy())
    split = DCITrainer(
        x_train=x,
        y_train=y,
        state=split_state,
        run_dir=tmp_path / "split",
        checkpoint_every=0,
    )
    for _ in range(3):
        split.train_one_sample()
    checkpoint_path = split.save_checkpoint("midpoint")

    payload = CheckpointManager(tmp_path / "split").load(checkpoint_path)
    resumed = DCITrainer(
        x_train=x,
        y_train=y,
        state=payload["state"],
        run_dir=tmp_path / "resumed",
        checkpoint_every=0,
    )
    resumed.train()

    np.testing.assert_allclose(
        resumed.state.model.connectivity.w_input_exc,
        full.state.model.connectivity.w_input_exc,
    )
    np.testing.assert_allclose(
        resumed.state.model.network_state.exc.v,
        full.state.model.network_state.exc.v,
    )
    np.testing.assert_allclose(
        resumed.state.model.homeostasis_state.exc_current,
        full.state.model.homeostasis_state.exc_current,
    )
    assert resumed.state.position == full.state.position == 6
    assert resumed.state.history == full.state.history
