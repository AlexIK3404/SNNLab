from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from snnlab.configs.reservoir import ReservoirConfig
from snnlab.core.encoders import SpikeEncoder
from snnlab.core.numerical_methods import IzhikevichParameters, get_stepper


@dataclass(slots=True)
class ReservoirSimulationResult:
    """
    Stores one-sample reservoir features and optional spike log.

    Хранит признаки одного sample и необязательный spike log резервуара.
    """

    features: np.ndarray
    spike_counts: np.ndarray
    spike_log: np.ndarray | None
    total_spikes: int
    active_neurons: int


@dataclass(slots=True)
class ReservoirNetwork:
    """
    Represents a fixed random recurrent reservoir.

    Представляет фиксированный случайный рекуррентный резервуар.
    """

    cfg: ReservoirConfig
    w_input: np.ndarray
    w_recurrent: np.ndarray
    neuron_signs: np.ndarray

    def extract_features(
        self,
        x: np.ndarray,
        *,
        encoder: SpikeEncoder,
        rng: np.random.Generator,
        return_spike_log: bool = False,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
        frame_every_steps: int = 10,
    ) -> ReservoirSimulationResult:
        """
        Encodes one sample and simulates the reservoir from a fresh state.

        Кодирует один sample и моделирует резервуар из нового начального состояния.
        """
        input_spikes = encoder.sample_spikes(
            x,
            n_steps=self.cfg.n_steps,
            dt_ms=self.cfg.dt_ms,
            rng=rng,
        )
        return self.simulate(
            input_spikes,
            return_spike_log=return_spike_log,
            frame_callback=frame_callback,
            frame_every_steps=frame_every_steps,
        )

    def simulate(
        self,
        input_spikes: np.ndarray,
        *,
        return_spike_log: bool = False,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
        frame_every_steps: int = 10,
    ) -> ReservoirSimulationResult:
        """
        Simulates one input spike train on CPU.

        Моделирует один входной spike train на CPU.
        """
        input_spikes = np.asarray(input_spikes, dtype=np.float32)
        if input_spikes.ndim != 2 or input_spikes.shape[1] != self.cfg.n_input:
            raise ValueError(
                f"input_spikes must have shape [steps, {self.cfg.n_input}], got {input_spikes.shape}"
            )

        n_steps = input_spikes.shape[0]
        n_res = self.cfg.n_reservoir
        v = np.full(n_res, self.cfg.neuron_c, dtype=np.float32)
        u = np.full(n_res, self.cfg.neuron_b * self.cfg.neuron_c, dtype=np.float32)
        syn = np.zeros(n_res, dtype=np.float32)
        previous_spikes = np.zeros(n_res, dtype=np.float32)
        spike_counts = np.zeros(n_res, dtype=np.int64)
        spike_log = np.zeros((n_steps, n_res), dtype=bool) if return_spike_log else None

        params = IzhikevichParameters(a=self.cfg.neuron_a, b=self.cfg.neuron_b)
        stepper = get_stepper(self.cfg.numerical_method)
        decay = float(np.exp(-self.cfg.dt_ms / self.cfg.tau_syn_ms))
        frame_every_steps = max(1, int(frame_every_steps))

        for step in range(n_steps):
            input_current = input_spikes[step] @ self.w_input
            recurrent_current = previous_spikes @ self.w_recurrent
            syn = syn * decay + input_current + recurrent_current
            current = syn + self.cfg.bias_current

            v, u = stepper(v, u, current, self.cfg.dt_ms, params)

            # EN: Cast back to float32 to preserve the numerical regime of the current notebook reservoir.
            # RU: Возвращаем float32, чтобы сохранить численный режим текущего reservoir из блокнота.
            v = np.asarray(v, dtype=np.float32)
            u = np.asarray(u, dtype=np.float32)

            unstable = (~np.isfinite(v)) | (~np.isfinite(u)) | (v > 500.0) | (v < -500.0)
            if np.any(unstable):
                v[unstable] = self.cfg.neuron_c
                u[unstable] = self.cfg.neuron_b * self.cfg.neuron_c

            spikes = v >= self.cfg.v_peak
            if np.any(spikes):
                v[spikes] = self.cfg.neuron_c
                u[spikes] += self.cfg.neuron_d

            spike_counts += spikes.astype(np.int64)
            previous_spikes = spikes.astype(np.float32)
            if spike_log is not None:
                spike_log[step] = spikes

            if frame_callback is not None and (
                (step + 1) % frame_every_steps == 0 or step + 1 == n_steps
            ):
                # EN: Emit lightweight snapshots during simulation; the future GUI
                #     can render them without waiting for the whole sample to finish.
                # RU: Отправляем лёгкие snapshot во время симуляции; будущий GUI
                #     сможет рисовать их, не ожидая завершения всего sample.
                frame_callback(
                    {
                        "step": step + 1,
                        "total_steps": n_steps,
                        # EN: Reservoir samples start from a fresh dynamic state, so
                        #     time inside the current sample is more informative than
                        #     a global GUI frame counter.
                        # RU: Каждый sample резервуара стартует из свежего состояния,
                        #     поэтому время внутри текущего sample информативнее
                        #     глобального номера GUI-кадра.
                        "time_ms": float((step + 1) * self.cfg.dt_ms),
                        "spikes": spikes.copy(),
                        "membrane_v": v.copy(),
                        "synaptic_state": syn.copy(),
                        "cumulative_spike_counts": spike_counts.copy(),
                    }
                )

        features = (spike_counts.astype(np.float32) / max(1, n_steps)).astype(np.float32)
        return ReservoirSimulationResult(
            features=features,
            spike_counts=spike_counts,
            spike_log=spike_log,
            total_spikes=int(np.sum(spike_counts)),
            active_neurons=int(np.count_nonzero(spike_counts)),
        )

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the model parameters required for persistence.

        Возвращает параметры модели, необходимые для сохранения.
        """
        return {
            "cfg": self.cfg,
            "w_input": self.w_input.copy(),
            "w_recurrent": self.w_recurrent.copy(),
            "neuron_signs": self.neuron_signs.copy(),
        }


def build_reservoir_network(cfg: ReservoirConfig, *, seed: int | None = None) -> ReservoirNetwork:
    """
    Builds the fixed random reservoir used by the current notebook branch.

    Создаёт фиксированный случайный резервуар из текущей ветки блокнота.
    """
    rng = np.random.RandomState(cfg.seed if seed is None else seed)

    w_input = rng.uniform(0.2, 1.0, size=(cfg.n_input, cfg.n_reservoir))
    input_mask = rng.rand(cfg.n_input, cfg.n_reservoir) < cfg.input_density
    w_input = (w_input * input_mask * cfg.input_scale).astype(np.float32)

    neuron_signs = np.ones(cfg.n_reservoir, dtype=np.float32)
    n_exc = int(round(cfg.n_reservoir * cfg.excitatory_ratio))
    inhibitory_indices = rng.permutation(cfg.n_reservoir)[n_exc:]
    neuron_signs[inhibitory_indices] = -1.0

    w_recurrent = rng.uniform(0.0, 1.0, size=(cfg.n_reservoir, cfg.n_reservoir))
    recurrent_mask = rng.rand(cfg.n_reservoir, cfg.n_reservoir) < cfg.recurrent_density
    np.fill_diagonal(recurrent_mask, False)

    # EN: The sign belongs to the presynaptic neuron, so it multiplies rows.
    # RU: Знак принадлежит пресинаптическому нейрону, поэтому умножает строки.
    w_recurrent = w_recurrent * recurrent_mask * neuron_signs[:, None] * cfg.recurrent_scale

    return ReservoirNetwork(
        cfg=cfg,
        w_input=w_input.astype(np.float32),
        w_recurrent=w_recurrent.astype(np.float32),
        neuron_signs=neuron_signs,
    )
