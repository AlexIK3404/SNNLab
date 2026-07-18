from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from snnlab.architectures.dci import DCIModel
from snnlab.core.events import EventObserver, FrameworkEvent, NullObserver
from snnlab.experiments.base import extend_sample_schedule
from snnlab.runtime.checkpoint import CheckpointManager
from snnlab.runtime.control import RunControl
from snnlab.runtime.experiment_io import update_data_protocol_training


@dataclass(slots=True)
class DCITrainerState:
    """
    Stores the model, schedule, history, and RNG state required for exact continuation.

    Хранит модель, расписание, историю и RNG-состояние для точного продолжения.
    """

    model: DCIModel
    sample_indices: np.ndarray
    position: int = 0
    history: dict[str, list[Any]] = field(default_factory=dict)
    total_exc_spikes_by_neuron: np.ndarray | None = None
    active_samples_by_neuron: np.ndarray | None = None
    rng_state: dict[str, Any] | None = None


class DCITrainer:
    """
    Manages clean DCI training with pause, stop, resume, and continued training.

    Управляет clean DCI-обучением с pause, stop, resume и дообучением.
    """

    HISTORY_KEYS = (
        "sample_index",
        "label",
        "accepted",
        "attempts",
        "final_rate_hz",
        "exc_spikes",
        "inh_spikes",
        "active_exc",
        "active_inh",
        "max_sync_exc",
        "max_sync_inh",
        "rest_exc_spikes",
        "rest_inh_spikes",
        "mean_abs_delta",
        "max_abs_delta",
        "weight_min",
        "weight_max",
        "column_sum_min",
        "column_sum_max",
        "homeo_mean",
        "homeo_std",
        "homeo_min",
        "homeo_max",
    )

    def __init__(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        state: DCITrainerState,
        run_dir: str | Path,
        observer: EventObserver | None = None,
        control: RunControl | None = None,
        checkpoint_every: int = 50,
        normalize_after_each: bool = True,
        emit_spike_logs: bool = False,
        emit_live_frames: bool = False,
        live_frame_every_steps: int = 50,
        emit_learning_snapshots: bool = False,
        learning_snapshot_every_samples: int = 10,
        data_spec: dict[str, Any] | None = None,
    ):
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        if len(self.x_train) != len(self.y_train):
            raise ValueError("x_train and y_train must have the same length")

        self.state = state
        self.observer = observer or NullObserver()
        self.control = control or RunControl()
        self.checkpoint_every = max(0, int(checkpoint_every))
        self.normalize_after_each = bool(normalize_after_each)
        self.emit_spike_logs = bool(emit_spike_logs)
        self.emit_live_frames = bool(emit_live_frames)
        self.live_frame_every_steps = max(1, int(live_frame_every_steps))
        self.emit_learning_snapshots = bool(emit_learning_snapshots)
        self.learning_snapshot_every_samples = max(1, int(learning_snapshot_every_samples))
        # EN: Persist the dataset construction recipe so a checkpoint can be
        #     resumed without relying on hidden GUI state.
        # RU: Сохраняем рецепт построения датасета, чтобы checkpoint можно было
        #     продолжить без зависимости от скрытого состояния GUI.
        self.data_spec = dict(data_spec or {})
        self.checkpoints = CheckpointManager(run_dir)

        self.rng = np.random.default_rng(self.state.model.cfg.seed + 1)
        if self.state.rng_state is not None:
            self.rng.bit_generator.state = self.state.rng_state

        self._initialize_accumulators()

    @property
    def total_samples(self) -> int:
        return int(len(self.state.sample_indices))

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
        Appends a continuation schedule without changing already processed samples.

        Добавляет расписание дообучения без изменения уже обработанных sample.
        """
        self.state.sample_indices = extend_sample_schedule(
            self.state.sample_indices,
            n_items=len(self.x_train),
            additional_samples=additional_samples,
            seed=seed,
            allow_repeats=allow_repeats,
        )

    def train(self) -> DCITrainerState:
        self._emit("run_start", {"architecture": "dci", "position": self.state.position})

        while not self.finished:
            if self.control.stop_requested:
                self.save_checkpoint("stopped")
                self._emit("stopped", {"position": self.state.position})
                return self.state

            self.train_one_sample()
            self._handle_safe_point()

            if self.checkpoint_every and self.state.position % self.checkpoint_every == 0:
                self.save_checkpoint(f"step_{self.state.position:06d}")

        self.save_checkpoint("final")
        self._emit("run_end", {"position": self.state.position})
        return self.state

    def train_one_sample(self) -> None:
        sample_index = int(self.state.sample_indices[self.state.position])
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

        metrics = self.state.model.train_one_sample(
            self.x_train[sample_index],
            rng=self.rng,
            normalize_after_each=self.normalize_after_each,
            emit_spike_logs=self.emit_spike_logs,
            frame_callback=frame_callback if self.emit_live_frames else None,
            frame_every_steps=self.live_frame_every_steps,
        )

        exc_counts = np.asarray(metrics.pop("exc_counts"), dtype=np.int64)
        inh_counts = np.asarray(metrics.pop("inh_counts"), dtype=np.int64)
        exc_spike_log = metrics.pop("exc_spike_log", None)
        inh_spike_log = metrics.pop("inh_spike_log", None)

        self.state.total_exc_spikes_by_neuron += exc_counts
        self.state.active_samples_by_neuron += (exc_counts > 0).astype(np.int64)

        self.state.history["sample_index"].append(sample_index)
        self.state.history["label"].append(label)
        for key in self.HISTORY_KEYS:
            if key in {"sample_index", "label"}:
                continue
            self.state.history[key].append(metrics[key])

        self.state.position += 1
        self.state.rng_state = self.rng.bit_generator.state

        payload: dict[str, Any] = {
            "position": self.state.position,
            "total_samples": self.total_samples,
            "sample_index": sample_index,
            "label": label,
            **self._epoch_payload(self.state.position),
            **metrics,
            "exc_counts": exc_counts,
            "inh_counts": inh_counts,
        }
        if exc_spike_log is not None:
            payload["exc_spike_log"] = exc_spike_log
            payload["inh_spike_log"] = inh_spike_log
        self._emit("sample_end", payload)

        if self.emit_learning_snapshots and (
            self.state.position == 1
            or self.state.position % self.learning_snapshot_every_samples == 0
            or self.finished
        ):
            # EN: Emit copied arrays only at a throttled sample interval. This
            #     keeps the GUI thread independent from mutable trainer state.
            # RU: Отправляем копии массивов только с ограниченной частотой.
            #     Так GUI-поток не зависит от изменяемого состояния trainer-а.
            model = self.state.model
            self._emit(
                "learning_snapshot",
                {
                    "architecture": "dci",
                    "position": self.state.position,
                    "total_samples": self.total_samples,
                    "w_input_exc": model.connectivity.w_input_exc.copy(),
                    "total_exc_spikes_by_neuron": self.state.total_exc_spikes_by_neuron.copy(),
                    "homeostasis_current": model.homeostasis_state.exc_current.copy(),
                    "w_min": float(model.stdp_cfg.w_min),
                    "w_max": float(model.stdp_cfg.w_max),
                },
            )

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

    def save_checkpoint(self, name: str) -> Path:
        path = self.checkpoints.save(
            name=name,
            payload={"kind": "dci", "state": self.state, "data_spec": self.data_spec},
            metadata={
                "kind": "dci",
                "position": self.state.position,
                "total_samples": self.total_samples,
                "exc_method": self.state.model.cfg.exc_numerical_method,
                "inh_method": self.state.model.cfg.inh_numerical_method,
            },
        )
        update_data_protocol_training(
            run_dir=self.checkpoints.run_dir,
            kind="dci",
            data_spec=self.data_spec,
            schedule=self.state.sample_indices,
            position=self.state.position,
        )
        self._emit("checkpoint_saved", {"path": str(path), "position": self.state.position})
        return path

    def _initialize_accumulators(self) -> None:
        n_exc = self.state.model.cfg.n_exc
        if self.state.total_exc_spikes_by_neuron is None:
            self.state.total_exc_spikes_by_neuron = np.zeros(n_exc, dtype=np.int64)
        if self.state.active_samples_by_neuron is None:
            self.state.active_samples_by_neuron = np.zeros(n_exc, dtype=np.int64)
        if not self.state.history:
            self.state.history = {key: [] for key in self.HISTORY_KEYS}

    def _handle_safe_point(self) -> None:
        if not self.control.pause_requested:
            return
        self.save_checkpoint("paused")
        self._emit("paused", {"position": self.state.position})
        if self.control.wait_if_paused():
            self._emit("resumed", {"position": self.state.position})

    def _emit(self, name: str, payload: dict[str, Any]) -> None:
        self.observer.handle(FrameworkEvent.create(name, "dci_trainer", payload))
