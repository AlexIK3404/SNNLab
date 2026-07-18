from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from snnlab.architectures.reservoir import ReservoirNetwork
from snnlab.configs.reservoir import ReservoirConfig
from snnlab.core.encoders import SpikeEncoder
from snnlab.core.events import EventObserver, FrameworkEvent, NullObserver
from snnlab.core.readouts import build_readout
from snnlab.experiments.base import extend_sample_schedule
from snnlab.runtime.checkpoint import CheckpointManager
from snnlab.runtime.control import RunControl
from snnlab.runtime.experiment_io import update_data_protocol_training


@dataclass(slots=True)
class ReservoirRunnerState:
    """
    Stores all mutable state required to continue reservoir feature collection.

    Хранит всё изменяемое состояние, необходимое для продолжения сбора reservoir-признаков.
    """

    network: ReservoirNetwork
    train_sample_indices: np.ndarray
    position: int = 0
    features: list[np.ndarray] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    readout_model: Any | None = None
    rng_state: dict[str, Any] | None = None


class ReservoirRunner:
    """
    Manages feature collection, checkpointing, continuation, and readout fitting.

    Управляет сбором признаков, checkpoint, продолжением и обучением readout.
    """

    def __init__(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        cfg: ReservoirConfig,
        encoder: SpikeEncoder,
        state: ReservoirRunnerState,
        run_dir: str | Path,
        observer: EventObserver | None = None,
        control: RunControl | None = None,
        checkpoint_every: int = 100,
        emit_spike_logs: bool = False,
        emit_live_frames: bool = False,
        live_frame_every_steps: int = 10,
        data_spec: dict[str, Any] | None = None,
    ):
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        if len(self.x_train) != len(self.y_train):
            raise ValueError("x_train and y_train must have the same length")
        if encoder.output_size != cfg.n_input:
            raise ValueError("Encoder output size does not match cfg.n_input")

        self.cfg = cfg
        self.encoder = encoder
        self.state = state
        self.observer = observer or NullObserver()
        self.control = control or RunControl()
        self.checkpoint_every = max(0, int(checkpoint_every))
        self.emit_spike_logs = bool(emit_spike_logs)
        self.emit_live_frames = bool(emit_live_frames)
        self.live_frame_every_steps = max(1, int(live_frame_every_steps))
        # EN: Keep the dataset recipe with checkpoints for portable resume.
        # RU: Храним рецепт датасета вместе с checkpoint для переносимого resume.
        self.data_spec = dict(data_spec or {})
        self.checkpoints = CheckpointManager(run_dir)

        self.rng = np.random.default_rng(cfg.seed + 1)
        if self.state.rng_state is not None:
            self.rng.bit_generator.state = self.state.rng_state

    @property
    def total_samples(self) -> int:
        return int(len(self.state.train_sample_indices))

    @property
    def finished(self) -> bool:
        return self.state.position >= self.total_samples

    def extend_training(
        self,
        *,
        additional_samples: int,
        seed: int,
        allow_repeats: bool = True,
    ) -> None:
        """
        Appends more samples for later readout fitting or refitting.

        Добавляет дополнительные sample для последующего обучения или переобучения readout.
        """
        self.state.train_sample_indices = extend_sample_schedule(
            self.state.train_sample_indices,
            n_items=len(self.x_train),
            additional_samples=additional_samples,
            seed=seed,
            allow_repeats=allow_repeats,
        )

    def collect_features(self) -> ReservoirRunnerState:
        self._emit("run_start", {"architecture": "reservoir", "position": self.state.position})
        self._emit("stage_start", {"stage": "collect_train_features"})

        while not self.finished:
            if self.control.stop_requested:
                self.save_checkpoint("stopped")
                self._emit("stopped", {"position": self.state.position})
                return self.state

            self.collect_one_sample()
            self._handle_safe_point()

            if self.checkpoint_every and self.state.position % self.checkpoint_every == 0:
                self.save_checkpoint(f"features_{self.state.position:06d}")

        self.save_checkpoint("features_complete")
        self._emit("stage_end", {"stage": "collect_train_features"})
        self._emit("run_end", {"position": self.state.position})
        return self.state

    def collect_one_sample(self) -> None:
        sample_index = int(self.state.train_sample_indices[self.state.position])
        label = int(self.y_train[sample_index])

        def frame_callback(frame: dict[str, Any]) -> None:
            self._emit(
                "simulation_frame",
                {
                    "position": self.state.position + 1,
                    "total_samples": self.total_samples,
                    "sample_index": sample_index,
                    "label": label,
                    **self._epoch_payload(self.state.position + 1),
                    **frame,
                },
            )

        result = self.state.network.extract_features(
            self.x_train[sample_index],
            encoder=self.encoder,
            rng=self.rng,
            return_spike_log=self.emit_spike_logs,
            frame_callback=frame_callback if self.emit_live_frames else None,
            frame_every_steps=self.live_frame_every_steps,
        )
        self.state.features.append(result.features.copy())
        self.state.labels.append(label)
        self.state.position += 1
        self.state.rng_state = self.rng.bit_generator.state

        signs = np.asarray(self.state.network.neuron_signs, dtype=np.float32)
        excitatory_mask = signs >= 0.0
        inhibitory_mask = ~excitatory_mask
        excitatory_spikes = int(np.sum(result.spike_counts[excitatory_mask]))
        inhibitory_spikes = int(np.sum(result.spike_counts[inhibitory_mask]))
        active_exc = int(np.count_nonzero(result.spike_counts[excitatory_mask]))
        active_inh = int(np.count_nonzero(result.spike_counts[inhibitory_mask]))
        possible_spikes = max(1, int(self.cfg.n_reservoir * self.cfg.n_steps))
        spike_occupancy = float(result.total_spikes / possible_spikes)
        duration_s = max(float(self.cfg.simulation_ms) / 1000.0, 1e-12)
        mean_rate_hz = float(result.total_spikes / self.cfg.n_reservoir / duration_s)

        payload: dict[str, Any] = {
            "position": self.state.position,
            "total_samples": self.total_samples,
            "sample_index": sample_index,
            "label": label,
            **self._epoch_payload(self.state.position),
            "spikes": result.total_spikes,
            "active": result.active_neurons,
            "spike_counts": result.spike_counts.copy(),
            # EN: Explicit reservoir diagnostics keep GUI plots on comparable
            #     scales and make pathological saturation visible immediately.
            # RU: Явные reservoir-метрики дают сравнимые шкалы GUI и сразу
            #     показывают патологическое насыщение сети.
            "n_reservoir": int(self.cfg.n_reservoir),
            "n_steps": int(self.cfg.n_steps),
            "simulation_ms": float(self.cfg.simulation_ms),
            "spike_occupancy": spike_occupancy,
            "mean_rate_hz": mean_rate_hz,
            "excitatory_spikes": excitatory_spikes,
            "inhibitory_spikes": inhibitory_spikes,
            "active_exc": active_exc,
            "active_inh": active_inh,
        }
        if result.spike_log is not None:
            payload["spike_log"] = result.spike_log
        self._emit("sample_end", payload)

    def _epoch_payload(self, position: int) -> dict[str, int]:
        """
        Returns epoch coordinates when the schedule was created by the GUI.

        Возвращает координаты эпохи, если расписание было создано GUI.
        """
        samples_per_epoch = int((self.data_spec or {}).get("samples_per_epoch", 0))
        n_epochs = int((self.data_spec or {}).get("epochs", 0))
        if samples_per_epoch <= 0 or n_epochs <= 0:
            return {}
        zero_based = max(0, int(position) - 1)
        return {
            "epoch": min(n_epochs, zero_based // samples_per_epoch + 1),
            "n_epochs": n_epochs,
            "sample_in_epoch": zero_based % samples_per_epoch + 1,
            "samples_per_epoch": samples_per_epoch,
        }

    def fit_readout(self, *, allow_partial: bool = True) -> Any:
        """
        Fits or refits a readout using all features collected so far.

        When allow_partial=True, a stopped run can still produce a usable model.

        Обучает или переобучает readout на всех уже собранных признаках.

        При allow_partial=True даже остановленный запуск может дать используемую модель.
        """
        if not self.state.features:
            raise RuntimeError("No reservoir features have been collected")
        if not allow_partial and not self.finished:
            raise RuntimeError("Feature collection is incomplete")

        x_features = np.vstack(self.state.features)
        y = np.asarray(self.state.labels, dtype=np.int64)
        if np.unique(y).size < 2:
            raise RuntimeError("Readout fitting requires at least two classes")

        select_k = min(self.cfg.select_k, x_features.shape[1])
        model = build_readout(
            kind=self.cfg.readout,
            use_feature_selection=self.cfg.use_feature_selection,
            select_k=select_k if self.cfg.use_feature_selection else None,
        )
        model.fit(x_features, y)
        self.state.readout_model = model
        self.save_checkpoint("readout_fitted")
        self._emit("stage_end", {"stage": "fit_readout", "n_samples": len(y)})
        return model

    def predict(self, x: np.ndarray, *, seed: int | None = None) -> np.ndarray:
        """
        Predicts labels without advancing the training RNG stream.

        Предсказывает метки, не сдвигая RNG-поток обучения.
        """
        if self.state.readout_model is None:
            raise RuntimeError("Readout is not fitted")

        # EN: Evaluation uses an isolated RNG stream so inference cannot change
        #     the exact trajectory of a later continue-training operation.
        # RU: Evaluation использует отдельный RNG-поток, чтобы inference не менял
        #     точную траекторию последующего дообучения.
        eval_rng = np.random.default_rng(self.cfg.seed + 20_000 if seed is None else seed)
        features = [
            self.state.network.extract_features(sample, encoder=self.encoder, rng=eval_rng).features
            for sample in np.asarray(x)
        ]
        return self.state.readout_model.predict(np.vstack(features))

    def save_checkpoint(self, name: str) -> Path:
        path = self.checkpoints.save(
            name=name,
            payload={
                "kind": "reservoir",
                "cfg": self.cfg,
                "state": self.state,
                "data_spec": self.data_spec,
            },
            metadata={
                "kind": "reservoir",
                "position": self.state.position,
                "total_samples": self.total_samples,
                "readout_fitted": self.state.readout_model is not None,
            },
        )
        update_data_protocol_training(
            run_dir=self.checkpoints.run_dir,
            kind="reservoir",
            data_spec=self.data_spec,
            schedule=self.state.train_sample_indices,
            position=self.state.position,
        )
        self._emit("checkpoint_saved", {"path": str(path), "position": self.state.position})
        return path

    def _handle_safe_point(self) -> None:
        if not self.control.pause_requested:
            return
        self.save_checkpoint("paused")
        self._emit("paused", {"position": self.state.position})
        if self.control.wait_if_paused():
            self._emit("resumed", {"position": self.state.position})

    def _emit(self, name: str, payload: dict[str, Any]) -> None:
        self.observer.handle(FrameworkEvent.create(name, "reservoir_runner", payload))
