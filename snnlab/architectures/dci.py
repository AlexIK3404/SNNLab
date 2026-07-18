from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from snnlab.configs.dci import (
    DCIConfig,
    DCIDynamicsConfig,
    DCIHomeostasisConfig,
    DCIPresentationConfig,
    DCISTDPConfig,
)
from snnlab.core.numerical_methods import IzhikevichParameters, get_stepper


@dataclass(slots=True)
class DCIConnectivity:
    w_input_exc: np.ndarray
    mask_exc_inh: np.ndarray
    mask_inh_exc: np.ndarray


@dataclass(slots=True)
class NeuronPopulationState:
    v: np.ndarray
    u: np.ndarray
    g_exc: np.ndarray
    g_inh: np.ndarray
    refractory_left_ms: np.ndarray


@dataclass(slots=True)
class DCINetworkState:
    exc: NeuronPopulationState
    inh: NeuronPopulationState
    last_exc_spikes: np.ndarray
    last_inh_spikes: np.ndarray


@dataclass(slots=True)
class DCIHomeostasisState:
    exc_current: np.ndarray


@dataclass(slots=True)
class DCISimulationResult:
    exc_spikes: np.ndarray
    inh_spikes: np.ndarray
    recorded_exc_v: np.ndarray
    recorded_inh_v: np.ndarray
    mean_exc_g_exc: np.ndarray
    mean_exc_g_inh: np.ndarray
    mean_inh_g_exc: np.ndarray
    record_exc_indices: np.ndarray
    record_inh_indices: np.ndarray


@dataclass(slots=True)
class DCIPresentationResult:
    accepted: bool
    accepted_attempt: int
    attempted_rates_hz: list[float]
    attempt_exc_spikes: list[int]
    attempt_inh_spikes: list[int]
    attempt_active_exc: list[int]
    accepted_input_spikes: np.ndarray
    stimulus_result: DCISimulationResult
    rest_result: DCISimulationResult


@dataclass(slots=True)
class DCIModel:
    """
    Holds the complete mutable DCI model state required for training continuation.

    Хранит полное изменяемое состояние DCI, необходимое для продолжения обучения.
    """

    cfg: DCIConfig
    dynamics: DCIDynamicsConfig
    presentation_cfg: DCIPresentationConfig
    stdp_cfg: DCISTDPConfig
    homeostasis_cfg: DCIHomeostasisConfig
    connectivity: DCIConnectivity
    network_state: DCINetworkState
    homeostasis_state: DCIHomeostasisState

    def train_one_sample(
        self,
        x: np.ndarray,
        *,
        rng: np.random.Generator,
        normalize_after_each: bool = True,
        emit_spike_logs: bool = False,
        frame_callback: Callable[[dict[str, Any]], None] | None = None,
        frame_every_steps: int = 50,
    ) -> dict[str, Any]:
        """
        Performs one checkpoint-safe clean DCI training sample.

        Probe attempts contain no STDP update. STDP and target-rate homeostasis
        are applied exactly once using the final attempt.

        Выполняет один checkpoint-safe sample clean DCI-обучения.

        Probe-попытки не обновляют STDP. STDP и target-rate гомеостаз применяются
        ровно один раз по финальной попытке.
        """
        weights_before = self.connectivity.w_input_exc.copy()

        presentation = present_image(
            x,
            model=self,
            rng=rng,
            record_exc_indices=np.empty(0, dtype=np.int64),
            record_inh_indices=np.empty(0, dtype=np.int64),
            frame_callback=frame_callback,
            frame_every_steps=frame_every_steps,
        )

        stimulus = presentation.stimulus_result
        rest = presentation.rest_result
        exc_counts = np.sum(stimulus.exc_spikes, axis=0).astype(np.int64)
        inh_counts = np.sum(stimulus.inh_spikes, axis=0).astype(np.int64)

        apply_stdp_from_logs(
            input_spikes=presentation.accepted_input_spikes,
            exc_spikes=stimulus.exc_spikes,
            connectivity=self.connectivity,
            cfg=self.cfg,
            stdp_cfg=self.stdp_cfg,
        )
        update_target_rate_homeostasis(
            exc_counts=exc_counts,
            homeostasis_cfg=self.homeostasis_cfg,
            homeostasis_state=self.homeostasis_state,
        )

        if normalize_after_each:
            normalize_input_columns(
                self.connectivity.w_input_exc,
                target_sum=self.cfg.input_weight_sum,
            )
            np.clip(
                self.connectivity.w_input_exc,
                self.stdp_cfg.w_min,
                self.stdp_cfg.w_max,
                out=self.connectivity.w_input_exc,
            )

        exc_per_step = np.sum(stimulus.exc_spikes, axis=1)
        inh_per_step = np.sum(stimulus.inh_spikes, axis=1)
        weight_delta = np.abs(self.connectivity.w_input_exc - weights_before)
        column_sums = np.sum(self.connectivity.w_input_exc, axis=0)
        homeo = self.homeostasis_state.exc_current

        metrics: dict[str, Any] = {
            "accepted": bool(presentation.accepted),
            "attempts": int(presentation.accepted_attempt + 1),
            "final_rate_hz": float(presentation.attempted_rates_hz[-1]),
            "exc_spikes": int(np.sum(exc_counts)),
            "inh_spikes": int(np.sum(inh_counts)),
            "active_exc": int(np.count_nonzero(exc_counts)),
            "active_inh": int(np.count_nonzero(inh_counts)),
            "max_sync_exc": int(np.max(exc_per_step, initial=0)),
            "max_sync_inh": int(np.max(inh_per_step, initial=0)),
            "rest_exc_spikes": int(np.sum(rest.exc_spikes)),
            "rest_inh_spikes": int(np.sum(rest.inh_spikes)),
            "mean_abs_delta": float(np.mean(weight_delta)),
            "max_abs_delta": float(np.max(weight_delta)),
            "weight_min": float(np.min(self.connectivity.w_input_exc)),
            "weight_max": float(np.max(self.connectivity.w_input_exc)),
            "column_sum_min": float(np.min(column_sums)),
            "column_sum_max": float(np.max(column_sums)),
            "homeo_mean": float(np.mean(homeo)),
            "homeo_std": float(np.std(homeo)),
            "homeo_min": float(np.min(homeo)),
            "homeo_max": float(np.max(homeo)),
            "exc_counts": exc_counts,
            "inh_counts": inh_counts,
        }
        if emit_spike_logs:
            metrics["exc_spike_log"] = stimulus.exc_spikes.copy()
            metrics["inh_spike_log"] = stimulus.inh_spikes.copy()
        return metrics

    def simulate_image_no_learning(
        self,
        x: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> DCIPresentationResult:
        """
        Runs the current model without changing weights or sample-level homeostasis.

        Запускает текущую модель без изменения весов и sample-level гомеостаза.
        """
        return present_image(x, model=self, rng=rng)


def normalize_input_columns(weights: np.ndarray, target_sum: float) -> None:
    """
    Normalizes incoming Input -> E weights column-wise.

    Нормализует входящие веса Input -> E по столбцам.
    """
    column_sums = np.sum(weights, axis=0)
    valid = column_sums > 1e-12
    weights[:, valid] *= (target_sum / column_sums[valid])[None, :]


def build_dci_connectivity(cfg: DCIConfig) -> DCIConnectivity:
    """
    Builds all-to-all Input -> E, one-to-one E -> I, and lateral I -> E topology.

    Создаёт all-to-all Input -> E, one-to-one E -> I и латеральную I -> E топологию.
    """
    rng = np.random.default_rng(cfg.seed)
    w_input_exc = rng.uniform(0.5, 1.5, size=(cfg.n_input, cfg.n_exc)).astype(np.float64)
    normalize_input_columns(w_input_exc, cfg.input_weight_sum)

    mask_exc_inh = np.eye(cfg.n_exc, cfg.n_inh, dtype=bool)
    mask_inh_exc = np.ones((cfg.n_inh, cfg.n_exc), dtype=bool)
    np.fill_diagonal(mask_inh_exc, False)

    return DCIConnectivity(w_input_exc, mask_exc_inh, mask_inh_exc)


def create_population_state(
    n_neurons: int,
    *,
    b: float,
    c: float,
    rng: np.random.Generator,
    v_jitter_mv: float = 5.0,
) -> NeuronPopulationState:
    v = rng.uniform(c, c + v_jitter_mv, size=n_neurons).astype(np.float64)
    return NeuronPopulationState(
        v=v,
        u=(b * v).astype(np.float64),
        g_exc=np.zeros(n_neurons, dtype=np.float64),
        g_inh=np.zeros(n_neurons, dtype=np.float64),
        refractory_left_ms=np.zeros(n_neurons, dtype=np.float64),
    )


def create_dci_state(cfg: DCIConfig, *, seed: int | None = None) -> DCINetworkState:
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    return DCINetworkState(
        exc=create_population_state(cfg.n_exc, b=cfg.exc_b, c=cfg.exc_c, rng=rng),
        inh=create_population_state(cfg.n_inh, b=cfg.inh_b, c=cfg.inh_c, rng=rng),
        last_exc_spikes=np.zeros(cfg.n_exc, dtype=bool),
        last_inh_spikes=np.zeros(cfg.n_inh, dtype=bool),
    )


def create_homeostasis_state(cfg: DCIConfig) -> DCIHomeostasisState:
    return DCIHomeostasisState(exc_current=np.zeros(cfg.n_exc, dtype=np.float64))


def build_dci_model(
    *,
    cfg: DCIConfig,
    dynamics: DCIDynamicsConfig,
    presentation_cfg: DCIPresentationConfig,
    stdp_cfg: DCISTDPConfig,
    homeostasis_cfg: DCIHomeostasisConfig,
    seed: int | None = None,
) -> DCIModel:
    """
    Builds a fresh DCI model.

    Создаёт новую DCI-модель.
    """
    return DCIModel(
        cfg=cfg,
        dynamics=dynamics,
        presentation_cfg=presentation_cfg,
        stdp_cfg=stdp_cfg,
        homeostasis_cfg=homeostasis_cfg,
        connectivity=build_dci_connectivity(cfg),
        network_state=create_dci_state(cfg, seed=seed),
        homeostasis_state=create_homeostasis_state(cfg),
    )


def _population_step(
    state: NeuronPopulationState,
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    v_peak: float,
    refractory_ms: float,
    dt_ms: float,
    e_exc: float,
    e_inh: float,
    bias_current: float,
    numerical_v_min: float,
    numerical_v_max: float,
    numerical_method: str,
    additional_inhibitory_current: np.ndarray | None = None,
) -> np.ndarray:
    """
    Advances one Izhikevich population while preserving the current refractory semantics.

    Продвигает одну популяцию Ижикевича, сохраняя текущую семантику refractory.
    """
    v_old = state.v.copy()
    u_old = state.u.copy()
    refractory_mask = state.refractory_left_ms > 0.0
    state.refractory_left_ms = np.maximum(state.refractory_left_ms - dt_ms, 0.0)

    synaptic_current = state.g_exc * (e_exc - v_old) + state.g_inh * (e_inh - v_old) + bias_current
    if additional_inhibitory_current is not None:
        current = np.asarray(additional_inhibitory_current, dtype=np.float64)
        if current.shape != state.v.shape:
            raise ValueError("additional_inhibitory_current has invalid shape")
        synaptic_current -= current

    active_mask = ~refractory_mask
    if np.any(active_mask):
        stepper = get_stepper(numerical_method)
        params = IzhikevichParameters(a=a, b=b)
        v_new, u_new = stepper(
            v_old[active_mask],
            u_old[active_mask],
            synaptic_current[active_mask],
            dt_ms,
            params,
        )
        state.v[active_mask] = v_new
        state.u[active_mask] = u_new

    # EN: During refractory time v is clamped, while u keeps evolving as in the current notebook code.
    # RU: Во время refractory v фиксируется, а u продолжает эволюционировать как в текущем коде блокнота.
    state.v[refractory_mask] = c
    state.u[refractory_mask] = u_old[refractory_mask] + dt_ms * a * (b * c - u_old[refractory_mask])

    finite = np.isfinite(state.v) & np.isfinite(state.u)
    spikes = active_mask & finite & (state.v >= v_peak) & (state.v <= numerical_v_max)
    if np.any(spikes):
        state.v[spikes] = c
        state.u[spikes] += d
        state.refractory_left_ms[spikes] = refractory_ms

    unstable = (~finite) | (state.v < numerical_v_min) | (state.v > numerical_v_max)
    unstable &= ~spikes
    if np.any(unstable):
        state.v[unstable] = c
        state.u[unstable] = b * c
        state.refractory_left_ms[unstable] = 0.0

    return spikes


def simulate_dci(
    input_spikes: np.ndarray,
    *,
    model: DCIModel,
    record_exc_indices: np.ndarray | None = None,
    record_inh_indices: np.ndarray | None = None,
    frame_callback: Callable[[dict[str, Any]], None] | None = None,
    frame_every_steps: int = 50,
    frame_context: dict[str, Any] | None = None,
) -> DCISimulationResult:
    """
    Simulates a continuous DCI segment without resetting mutable network state.

    Моделирует непрерывный участок DCI без сброса изменяемого состояния сети.
    """
    cfg = model.cfg
    dyn = model.dynamics
    state = model.network_state
    conn = model.connectivity

    input_spikes = np.asarray(input_spikes, dtype=np.float64)
    if input_spikes.ndim != 2 or input_spikes.shape[1] != cfg.n_input:
        raise ValueError(f"input_spikes must have shape [steps, {cfg.n_input}]")

    record_exc_indices = (
        np.array([0], dtype=np.int64)
        if record_exc_indices is None
        else np.asarray(record_exc_indices, dtype=np.int64)
    )
    record_inh_indices = (
        np.array([0], dtype=np.int64)
        if record_inh_indices is None
        else np.asarray(record_inh_indices, dtype=np.int64)
    )

    n_steps = input_spikes.shape[0]
    exc_log = np.zeros((n_steps, cfg.n_exc), dtype=bool)
    inh_log = np.zeros((n_steps, cfg.n_inh), dtype=bool)
    recorded_exc_v = np.zeros((n_steps, len(record_exc_indices)), dtype=np.float64)
    recorded_inh_v = np.zeros((n_steps, len(record_inh_indices)), dtype=np.float64)
    mean_exc_g_exc = np.zeros(n_steps, dtype=np.float64)
    mean_exc_g_inh = np.zeros(n_steps, dtype=np.float64)
    mean_inh_g_exc = np.zeros(n_steps, dtype=np.float64)

    decay_g_exc = float(np.exp(-cfg.dt_ms / cfg.tau_g_exc_ms))
    decay_g_inh = float(np.exp(-cfg.dt_ms / cfg.tau_g_inh_ms))
    homeo_decay = float(np.exp(-cfg.dt_ms / model.homeostasis_cfg.tau_ms))
    frame_every_steps = max(1, int(frame_every_steps))
    frame_context = dict(frame_context or {})

    for step in range(n_steps):
        state.exc.g_exc *= decay_g_exc
        state.exc.g_inh *= decay_g_inh
        state.inh.g_exc *= decay_g_exc
        state.inh.g_inh *= decay_g_inh

        active_inputs = np.flatnonzero(input_spikes[step] > 0.5)
        if active_inputs.size > 0:
            state.exc.g_exc += dyn.input_gain * np.sum(conn.w_input_exc[active_inputs], axis=0)

        # EN: Previous-step spikes are delivered now, matching the current notebook timing.
        # RU: Спайки прошлого шага доставляются сейчас, как в текущей временной схеме блокнота.
        state.inh.g_exc += dyn.weight_exc_inh * state.last_exc_spikes.astype(np.float64)

        total_inh_spikes = int(np.sum(state.last_inh_spikes))
        if total_inh_spikes > 0:
            state.exc.g_inh += dyn.weight_inh_exc * (
                total_inh_spikes - state.last_inh_spikes.astype(np.float64)
            )

        model.homeostasis_state.exc_current *= homeo_decay

        exc_spikes = _population_step(
            state.exc,
            a=cfg.exc_a,
            b=cfg.exc_b,
            c=cfg.exc_c,
            d=cfg.exc_d,
            v_peak=cfg.v_peak,
            refractory_ms=cfg.refractory_exc_ms,
            dt_ms=cfg.dt_ms,
            e_exc=cfg.e_exc,
            e_inh=cfg.e_inh,
            bias_current=dyn.bias_exc,
            numerical_v_min=dyn.numerical_v_min,
            numerical_v_max=dyn.numerical_v_max,
            numerical_method=cfg.exc_numerical_method,
            additional_inhibitory_current=model.homeostasis_state.exc_current,
        )
        inh_spikes = _population_step(
            state.inh,
            a=cfg.inh_a,
            b=cfg.inh_b,
            c=cfg.inh_c,
            d=cfg.inh_d,
            v_peak=cfg.v_peak,
            refractory_ms=cfg.refractory_inh_ms,
            dt_ms=cfg.dt_ms,
            e_exc=cfg.e_exc,
            e_inh=cfg.e_inh,
            bias_current=dyn.bias_inh,
            numerical_v_min=dyn.numerical_v_min,
            numerical_v_max=dyn.numerical_v_max,
            numerical_method=cfg.inh_numerical_method,
        )

        state.last_exc_spikes = exc_spikes.copy()
        state.last_inh_spikes = inh_spikes.copy()
        exc_log[step] = exc_spikes
        inh_log[step] = inh_spikes
        recorded_exc_v[step] = state.exc.v[record_exc_indices]
        recorded_inh_v[step] = state.inh.v[record_inh_indices]
        mean_exc_g_exc[step] = float(np.mean(state.exc.g_exc))
        mean_exc_g_inh[step] = float(np.mean(state.exc.g_inh))
        mean_inh_g_exc[step] = float(np.mean(state.inh.g_exc))

        if frame_callback is not None and (
            (step + 1) % frame_every_steps == 0 or step + 1 == n_steps
        ):
            # EN: Emit throttled population snapshots for future real-time views.
            # RU: Отправляем throttled snapshot популяций для будущих real-time представлений.
            frame_callback(
                {
                    **frame_context,
                    "step": step + 1,
                    "total_steps": n_steps,
                    "exc_spikes": exc_spikes.copy(),
                    "inh_spikes": inh_spikes.copy(),
                    "exc_v": state.exc.v.copy(),
                    "inh_v": state.inh.v.copy(),
                    "mean_exc_g_exc": mean_exc_g_exc[step],
                    "mean_exc_g_inh": mean_exc_g_inh[step],
                    "mean_inh_g_exc": mean_inh_g_exc[step],
                }
            )

    return DCISimulationResult(
        exc_spikes=exc_log,
        inh_spikes=inh_log,
        recorded_exc_v=recorded_exc_v,
        recorded_inh_v=recorded_inh_v,
        mean_exc_g_exc=mean_exc_g_exc,
        mean_exc_g_inh=mean_exc_g_inh,
        mean_inh_g_exc=mean_inh_g_exc,
        record_exc_indices=record_exc_indices,
        record_inh_indices=record_inh_indices,
    )


def _prepare_flat_input(x: np.ndarray, expected_size: int) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size != expected_size:
        raise ValueError(f"Expected {expected_size} input values, got {values.size}")
    if values.size > 0 and np.max(values) > 1.0:
        values = values / 255.0
    return np.clip(values, 0.0, 1.0)


def _encode_poisson(
    x: np.ndarray,
    *,
    n_steps: int,
    dt_ms: float,
    max_rate_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rates = x * max_rate_hz
    probabilities = rates * dt_ms / 1000.0
    if np.any(probabilities > 1.0):
        raise ValueError("Input spike probability exceeded 1")
    return rng.random((n_steps, x.size)) < probabilities[None, :]


def present_image(
    x: np.ndarray,
    *,
    model: DCIModel,
    rng: np.random.Generator,
    record_exc_indices: np.ndarray | None = None,
    record_inh_indices: np.ndarray | None = None,
    frame_callback: Callable[[dict[str, Any]], None] | None = None,
    frame_every_steps: int = 50,
) -> DCIPresentationResult:
    """
    Presents one sample with rate escalation and real rest after each attempt.

    No weight or sample-level homeostasis update is performed here. The mutable
    neuronal state still evolves across attempts and rest periods, preserving the
    current clean notebook protocol.

    Предъявляет один sample с повышением частоты и реальным отдыхом после каждой попытки.

    Веса и sample-level гомеостаз здесь не обновляются. При этом изменяемое
    нейронное состояние продолжает эволюционировать между попытками и отдыхом,
    сохраняя текущий clean-протокол блокнота.
    """
    cfg = model.cfg
    p_cfg = model.presentation_cfg
    values = _prepare_flat_input(x, cfg.n_input)
    rest_input = np.zeros((cfg.rest_steps, cfg.n_input), dtype=bool)

    attempted_rates: list[float] = []
    attempt_exc: list[int] = []
    attempt_inh: list[int] = []
    attempt_active: list[int] = []

    final_input: np.ndarray | None = None
    final_stimulus: DCISimulationResult | None = None
    final_rest: DCISimulationResult | None = None
    accepted = False
    accepted_attempt = -1

    for attempt in range(p_cfg.max_attempts):
        max_rate = p_cfg.base_max_rate_hz + attempt * p_cfg.rate_increment_hz
        input_spikes = _encode_poisson(
            values,
            n_steps=cfg.stimulus_steps,
            dt_ms=cfg.dt_ms,
            max_rate_hz=max_rate,
            rng=rng,
        )
        stimulus = simulate_dci(
            input_spikes,
            model=model,
            record_exc_indices=record_exc_indices,
            record_inh_indices=record_inh_indices,
            frame_callback=frame_callback,
            frame_every_steps=frame_every_steps,
            frame_context={"phase": "stimulus", "attempt": attempt + 1},
        )
        exc_counts = np.sum(stimulus.exc_spikes, axis=0)
        total_exc = int(np.sum(exc_counts))
        total_inh = int(np.sum(stimulus.inh_spikes))
        active_exc = int(np.count_nonzero(exc_counts))

        attempted_rates.append(float(max_rate))
        attempt_exc.append(total_exc)
        attempt_inh.append(total_inh)
        attempt_active.append(active_exc)
        accepted = total_exc >= p_cfg.min_exc_spikes

        rest = simulate_dci(
            rest_input,
            model=model,
            record_exc_indices=record_exc_indices,
            record_inh_indices=record_inh_indices,
            frame_callback=frame_callback,
            frame_every_steps=frame_every_steps,
            frame_context={"phase": "rest", "attempt": attempt + 1},
        )

        final_input = input_spikes
        final_stimulus = stimulus
        final_rest = rest
        accepted_attempt = attempt
        if accepted:
            break

    assert final_input is not None and final_stimulus is not None and final_rest is not None
    return DCIPresentationResult(
        accepted=accepted,
        accepted_attempt=accepted_attempt,
        attempted_rates_hz=attempted_rates,
        attempt_exc_spikes=attempt_exc,
        attempt_inh_spikes=attempt_inh,
        attempt_active_exc=attempt_active,
        accepted_input_spikes=final_input,
        stimulus_result=final_stimulus,
        rest_result=final_rest,
    )


def apply_stdp_from_logs(
    *,
    input_spikes: np.ndarray,
    exc_spikes: np.ndarray,
    connectivity: DCIConnectivity,
    cfg: DCIConfig,
    stdp_cfg: DCISTDPConfig,
) -> None:
    """
    Applies Input -> E STDP once using final-attempt logs.

    Один раз применяет Input -> E STDP по логам финальной попытки.
    """
    input_spikes = np.asarray(input_spikes, dtype=bool)
    exc_spikes = np.asarray(exc_spikes, dtype=bool)
    if input_spikes.shape[0] != exc_spikes.shape[0]:
        raise ValueError("input_spikes and exc_spikes must have equal step count")
    if input_spikes.shape[1] != cfg.n_input or exc_spikes.shape[1] != cfg.n_exc:
        raise ValueError("Spike logs have incompatible shapes")

    pre_trace = np.zeros(cfg.n_input, dtype=np.float64)
    trace_decay = float(np.exp(-cfg.dt_ms / stdp_cfg.tau_pre_ms))

    for step in range(input_spikes.shape[0]):
        pre_trace *= trace_decay
        active_input = input_spikes[step]
        if np.any(active_input):
            pre_trace[active_input] += 1.0

        active_exc = exc_spikes[step]
        if not np.any(active_exc):
            continue

        post_indices = np.flatnonzero(active_exc)
        selected_weights = connectivity.w_input_exc[:, post_indices]
        distance_to_max = np.maximum(stdp_cfg.w_max - selected_weights, 0.0)
        delta = (
            stdp_cfg.eta
            * (pre_trace[:, None] - stdp_cfg.x_target)
            * np.power(distance_to_max, stdp_cfg.mu)
        )
        connectivity.w_input_exc[:, post_indices] = np.clip(
            selected_weights + delta,
            stdp_cfg.w_min,
            stdp_cfg.w_max,
        )


def update_target_rate_homeostasis(
    *,
    exc_counts: np.ndarray,
    homeostasis_cfg: DCIHomeostasisConfig,
    homeostasis_state: DCIHomeostasisState,
) -> None:
    """
    Applies one sample-level target-rate correction.

    Применяет одну sample-level target-rate коррекцию.
    """
    counts = np.asarray(exc_counts, dtype=np.float64)
    if counts.shape != homeostasis_state.exc_current.shape:
        raise ValueError("exc_counts has invalid shape")
    error = counts - homeostasis_cfg.target_spikes_per_sample
    homeostasis_state.exc_current += homeostasis_cfg.learning_rate * error
    np.clip(
        homeostasis_state.exc_current,
        homeostasis_cfg.min_current,
        homeostasis_cfg.max_current,
        out=homeostasis_state.exc_current,
    )
