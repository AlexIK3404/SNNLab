from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from time import perf_counter_ns

import numpy as np

from snnlab.core.numerical_methods import (
    IzhikevichParameters,
    available_methods,
    get_stepper,
    izhikevich_rhs,
)

CancelCallback = Callable[[], bool]
ProgressCallback = Callable[[str, int, int], None]


class StudyCancelled(RuntimeError):
    """Raised when a long single-neuron study is cancelled by the caller."""


@dataclass(frozen=True, slots=True)
class NeuronParameters:
    """Parameters of the two-state Izhikevich neuron and its reset event."""

    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    threshold_mv: float = 30.0


NEURON_PRESETS: dict[str, NeuronParameters] = {
    "regular_spiking": NeuronParameters(a=0.02, b=0.2, c=-65.0, d=8.0),
    "fast_spiking": NeuronParameters(a=0.10, b=0.2, c=-65.0, d=2.0),
    "intrinsically_bursting": NeuronParameters(a=0.02, b=0.2, c=-55.0, d=4.0),
    "chattering": NeuronParameters(a=0.02, b=0.2, c=-50.0, d=2.0),
    "low_threshold_spiking": NeuronParameters(a=0.02, b=0.25, c=-65.0, d=2.0),
}


@dataclass(frozen=True, slots=True)
class IntegratorDescriptor:
    method_id: str
    rhs_evaluations: int
    nominal_order: int
    parameter_name: str | None = None


INTEGRATORS: dict[str, IntegratorDescriptor] = {
    "explicit_euler": IntegratorDescriptor("explicit_euler", 1, 1),
    "semi_euler": IntegratorDescriptor("semi_euler", 1, 1),
    "semi_implicit_euler": IntegratorDescriptor("semi_implicit_euler", 2, 1),
    "implicit_euler": IntegratorDescriptor("implicit_euler", 1, 1),
    "midpoint": IntegratorDescriptor("midpoint", 2, 2),
    "semi_midpoint": IntegratorDescriptor("semi_midpoint", 2, 2),
    "semi_implicit_midpoint": IntegratorDescriptor("semi_implicit_midpoint", 3, 2),
    "euler_composition_real": IntegratorDescriptor("euler_composition_real", 2, 1, "composition_s"),
    "euler_composition_complex": IntegratorDescriptor(
        "euler_composition_complex", 2, 2, "composition_alpha"
    ),
}

DEFAULT_COMPARISON_METHODS = (
    "explicit_euler",
    "semi_euler",
    "midpoint",
    "euler_composition_real",
    "euler_composition_complex",
)
DEFAULT_DT_VALUES_MS = (0.4, 0.2, 0.1, 0.05, 0.025)
DEFAULT_ALPHA_VALUES = (0.25, 0.4, 0.5, 0.6, 0.75, 1.0)
_FRAMEWORK_METHODS = frozenset(available_methods())


@dataclass(frozen=True, slots=True)
class SingleNeuronConfig:
    """Configuration shared by trace, f-I and convergence experiments."""

    neuron: NeuronParameters = NeuronParameters()
    input_current: float = 12.0
    duration_ms: float = 2000.0
    burn_in_ms: float = 500.0
    dt_ms: float = 0.1
    method_id: str = "midpoint"
    reset_mode: str = "event"
    composition_s: float = 0.5
    composition_alpha: float = 0.5
    initial_voltage_mv: float | None = None
    initial_recovery: float | None = None
    state_limit: float = 500.0

    def __post_init__(self) -> None:
        neuron_values = (
            self.neuron.a,
            self.neuron.b,
            self.neuron.c,
            self.neuron.d,
            self.neuron.threshold_mv,
        )
        if not all(np.isfinite(value) for value in neuron_values):
            raise ValueError("All neuron parameters must be finite")
        if self.method_id not in INTEGRATORS:
            raise ValueError(f"Unknown single-neuron integrator: {self.method_id!r}")
        if self.reset_mode not in {"grid", "event"}:
            raise ValueError("reset_mode must be 'grid' or 'event'")
        if not np.isfinite(self.input_current):
            raise ValueError("input_current must be finite")
        if not np.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")
        if not np.isfinite(self.duration_ms) or self.duration_ms <= 0.0:
            raise ValueError("duration_ms must be finite and positive")
        if (
            not np.isfinite(self.burn_in_ms)
            or self.burn_in_ms < 0.0
            or self.burn_in_ms >= self.duration_ms
        ):
            raise ValueError("burn_in_ms must be in [0, duration_ms)")
        if not 0.0 <= self.composition_s <= 1.0:
            raise ValueError("composition_s must be in [0, 1]")
        if not np.isfinite(self.composition_alpha) or self.composition_alpha < 0.0:
            raise ValueError("composition_alpha must be finite and non-negative")
        for name, value in (
            ("initial_voltage_mv", self.initial_voltage_mv),
            ("initial_recovery", self.initial_recovery),
        ):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if not np.isfinite(self.state_limit) or self.state_limit <= 0.0:
            raise ValueError("state_limit must be finite and positive")
        if self.duration_ms / self.dt_ms > 10_000_000:
            raise ValueError("The requested trace exceeds the 10,000,000-step safety limit")


@dataclass(slots=True)
class SingleNeuronTrace:
    time_ms: np.ndarray
    voltage_mv: np.ndarray
    recovery: np.ndarray
    spike_times_ms: np.ndarray
    elapsed_ns: int
    simulated_steps: int
    diverged: bool = False
    divergence_time_ms: float | None = None


@dataclass(frozen=True, slots=True)
class TraceMetrics:
    spike_count: int
    firing_rate_hz: float
    mean_isi_ms: float
    isi_cv: float
    elapsed_ms: float
    nanoseconds_per_step: float
    diverged: bool


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    dt_ms: float
    period_ms: float
    period_error_ms: float
    drift_ms_per_s: float
    nanoseconds_per_step: float
    diverged: bool


@dataclass(frozen=True, slots=True)
class ConvergenceSeries:
    method_id: str
    observed_period_order: float
    local_order: float
    points: tuple[ConvergencePoint, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceStudy:
    reference_period_ms: float
    reference_dt_ms: float
    reset_mode: str
    series: tuple[ConvergenceSeries, ...]


@dataclass(frozen=True, slots=True)
class FICurve:
    method_id: str
    currents: np.ndarray
    rates_hz: np.ndarray
    diverged: np.ndarray


@dataclass(frozen=True, slots=True)
class AlphaSweepPoint:
    alpha: float
    local_order: float
    observed_period_order: float


@dataclass(slots=True)
class SingleNeuronStudy:
    config: SingleNeuronConfig
    trace: SingleNeuronTrace
    metrics: TraceMetrics
    convergence: ConvergenceStudy
    grid_convergence: ConvergenceStudy
    fi_curves: tuple[FICurve, ...]
    alpha_sweep: tuple[AlphaSweepPoint, ...]


def neuron_preset(preset_id: str) -> NeuronParameters:
    """Return a named, immutable neuron parameter set."""

    try:
        return NEURON_PRESETS[preset_id]
    except KeyError as exc:
        choices = ", ".join(NEURON_PRESETS)
        raise KeyError(f"Unknown neuron preset {preset_id!r}. Available: {choices}") from exc


def _rhs(v: complex, u: complex, current: float, p: NeuronParameters) -> tuple[complex, complex]:
    return izhikevich_rhs(v, u, current, _core_parameters(p.a, p.b))


@lru_cache(maxsize=64)
def _core_parameters(a: float, b: float) -> IzhikevichParameters:
    return IzhikevichParameters(a=a, b=b)


def step_single_neuron(
    method_id: str,
    v: float,
    u: float,
    current: float,
    dt_ms: float,
    parameters: NeuronParameters,
    *,
    composition_s: float = 0.5,
    composition_alpha: float = 0.5,
) -> tuple[float, float]:
    """Advance the smooth Izhikevich ODE once, without applying spike reset."""

    if method_id in _FRAMEWORK_METHODS:
        core_parameters = _core_parameters(parameters.a, parameters.b)
        v_new, u_new = get_stepper(method_id)(v, u, current, dt_ms, core_parameters)
        return float(v_new), float(u_new)

    if method_id == "euler_composition_real":
        c1 = float(composition_s)
        c2 = 1.0 - c1
        dv1, du1 = _rhs(v, u, current, parameters)
        v1 = v + c1 * dt_ms * dv1
        u1 = u + c1 * dt_ms * du1
        dv2, du2 = _rhs(v1, u1, current, parameters)
        return float(v1 + c2 * dt_ms * dv2), float(u1 + c2 * dt_ms * du2)

    if method_id == "euler_composition_complex":
        c1 = complex(0.5, composition_alpha)
        c2 = c1.conjugate()
        dv1, du1 = _rhs(complex(v), complex(u), current, parameters)
        v1 = v + c1 * dt_ms * dv1
        u1 = u + c1 * dt_ms * du1
        dv2, du2 = _rhs(v1, u1, current, parameters)
        v2 = v1 + c2 * dt_ms * dv2
        u2 = u1 + c2 * dt_ms * du2
        # The conjugate two-stage composition is real through order two. The
        # projection removes higher-order imaginary residue, matching the
        # real-valued neuron state studied in the MATLAB prototype.
        return float(v2.real), float(u2.real)

    raise ValueError(f"Unknown single-neuron integrator: {method_id!r}")


def _check_cancelled(cancel_callback: CancelCallback | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise StudyCancelled("Single-neuron study cancelled")


def _finite_state(v: float, u: float, limit: float) -> bool:
    return bool(np.isfinite(v) and np.isfinite(u) and abs(v) <= limit and abs(u) <= limit)


def _step_from_config(
    config: SingleNeuronConfig,
    v: float,
    u: float,
    dt_ms: float,
) -> tuple[float, float]:
    return step_single_neuron(
        config.method_id,
        v,
        u,
        config.input_current,
        dt_ms,
        config.neuron,
        composition_s=config.composition_s,
        composition_alpha=config.composition_alpha,
    )


def _event_crossing(
    config: SingleNeuronConfig,
    v: float,
    u: float,
    dt_ms: float,
) -> tuple[float, float]:
    """Locate the threshold crossing within a step using the chosen integrator."""

    low = 0.0
    high = dt_ms
    _, u_high = _step_from_config(config, v, u, high)
    for _ in range(36):
        middle = 0.5 * (low + high)
        v_middle, u_middle = _step_from_config(config, v, u, middle)
        if not _finite_state(v_middle, u_middle, config.state_limit):
            high = middle
            continue
        if v_middle >= config.neuron.threshold_mv:
            high = middle
            u_high = u_middle
        else:
            low = middle
        if high - low <= max(1e-12, 1e-10 * dt_ms):
            break
    return high, float(u_high)


def _advance_with_reset(
    config: SingleNeuronConfig,
    v: float,
    u: float,
    dt_ms: float,
) -> tuple[float, float, list[float], bool]:
    """Advance one grid interval and return spike offsets inside it."""

    remaining = dt_ms
    elapsed = 0.0
    spike_offsets: list[float] = []
    while remaining > 1e-12:
        v_new, u_new = _step_from_config(config, v, u, remaining)
        if not _finite_state(v_new, u_new, config.state_limit):
            return v_new, u_new, spike_offsets, True
        if v_new < config.neuron.threshold_mv:
            return v_new, u_new, spike_offsets, False

        if config.reset_mode == "grid":
            spike_offsets.append(elapsed + remaining)
            return config.neuron.c, u_new + config.neuron.d, spike_offsets, False

        crossing_dt, u_crossing = _event_crossing(config, v, u, remaining)
        spike_offsets.append(elapsed + crossing_dt)
        v = config.neuron.c
        u = u_crossing + config.neuron.d
        elapsed += crossing_dt
        remaining -= crossing_dt

        if len(spike_offsets) >= 8:
            return v, u, spike_offsets, True
        if crossing_dt <= 1e-12:
            remaining = max(0.0, remaining - 1e-12)

    return v, u, spike_offsets, False


def simulate_single_neuron(
    config: SingleNeuronConfig,
    *,
    record_state: bool = True,
    cancel_callback: CancelCallback | None = None,
) -> SingleNeuronTrace:
    """Simulate one neuron with either grid-aligned or event-corrected reset."""

    initial_v = (
        float(config.initial_voltage_mv)
        if config.initial_voltage_mv is not None
        else float(config.neuron.c)
    )
    initial_u = (
        float(config.initial_recovery)
        if config.initial_recovery is not None
        else float(config.neuron.b * initial_v)
    )
    v = initial_v
    u = initial_u
    times = [0.0]
    voltages = [v]
    recoveries = [u]
    spikes: list[float] = []
    time_ms = 0.0
    steps = 0
    diverged = False
    divergence_time: float | None = None

    started = perf_counter_ns()
    while time_ms < config.duration_ms - 1e-12:
        if steps % 4096 == 0:
            _check_cancelled(cancel_callback)
        step_dt = min(config.dt_ms, config.duration_ms - time_ms)
        v, u, offsets, step_diverged = _advance_with_reset(config, v, u, step_dt)
        spikes.extend(time_ms + offset for offset in offsets)
        time_ms += step_dt
        steps += 1
        if record_state:
            times.append(time_ms)
            voltages.append(v)
            recoveries.append(u)
        if step_diverged:
            diverged = True
            divergence_time = time_ms
            break
    elapsed_ns = perf_counter_ns() - started

    if not record_state:
        times = [0.0, time_ms]
        voltages = [initial_v, v]
        recoveries = [initial_u, u]

    return SingleNeuronTrace(
        time_ms=np.asarray(times, dtype=np.float64),
        voltage_mv=np.asarray(voltages, dtype=np.float64),
        recovery=np.asarray(recoveries, dtype=np.float64),
        spike_times_ms=np.asarray(spikes, dtype=np.float64),
        elapsed_ns=int(elapsed_ns),
        simulated_steps=int(steps),
        diverged=diverged,
        divergence_time_ms=divergence_time,
    )


def _spikes_after(trace: SingleNeuronTrace, burn_in_ms: float) -> np.ndarray:
    return trace.spike_times_ms[trace.spike_times_ms >= burn_in_ms]


def _period_ms(trace: SingleNeuronTrace, burn_in_ms: float) -> float:
    spikes = _spikes_after(trace, burn_in_ms)
    if trace.diverged or spikes.size < 5:
        return float("nan")
    indices = np.arange(spikes.size, dtype=np.float64)
    return float(np.polyfit(indices, spikes, 1)[0])


def trace_metrics(trace: SingleNeuronTrace, burn_in_ms: float) -> TraceMetrics:
    spikes = _spikes_after(trace, burn_in_ms)
    intervals = np.diff(spikes)
    mean_isi = float(np.mean(intervals)) if intervals.size else float("nan")
    firing_rate = 1000.0 / mean_isi if np.isfinite(mean_isi) and mean_isi > 0.0 else 0.0
    isi_cv = (
        float(np.std(intervals, ddof=1) / mean_isi)
        if intervals.size >= 2 and mean_isi > 0.0
        else float("nan")
    )
    ns_per_step = trace.elapsed_ns / max(1, trace.simulated_steps)
    return TraceMetrics(
        spike_count=int(spikes.size),
        firing_rate_hz=float(firing_rate),
        mean_isi_ms=mean_isi,
        isi_cv=isi_cv,
        elapsed_ms=trace.elapsed_ns / 1_000_000.0,
        nanoseconds_per_step=float(ns_per_step),
        diverged=trace.diverged,
    )


def _rk4_reference_step(
    v: float,
    u: float,
    current: float,
    h_ms: float,
    p: NeuronParameters,
    *,
    substeps: int = 400,
) -> tuple[float, float]:
    dt = h_ms / substeps
    for _ in range(substeps):
        k1v, k1u = _rhs(v, u, current, p)
        k2v, k2u = _rhs(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u, current, p)
        k3v, k3u = _rhs(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u, current, p)
        k4v, k4u = _rhs(v + dt * k3v, u + dt * k3u, current, p)
        v += dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        u += dt * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
    return float(v), float(u)


def estimate_local_order(
    config: SingleNeuronConfig,
    method_id: str,
    *,
    step_sizes_ms: Sequence[float] = (0.02, 0.01, 0.005, 0.0025, 0.00125),
    composition_alpha: float | None = None,
) -> float:
    """Estimate global order from local truncation error slope minus one."""

    v0 = -55.0
    u0 = -12.0
    errors: list[float] = []
    valid_h: list[float] = []
    alpha = config.composition_alpha if composition_alpha is None else composition_alpha
    for h in step_sizes_ms:
        v_ref, u_ref = _rk4_reference_step(
            v0,
            u0,
            config.input_current,
            float(h),
            config.neuron,
        )
        v_num, u_num = step_single_neuron(
            method_id,
            v0,
            u0,
            config.input_current,
            float(h),
            config.neuron,
            composition_s=config.composition_s,
            composition_alpha=alpha,
        )
        error = abs(v_num - v_ref) + abs(u_num - u_ref)
        if np.isfinite(error) and error > np.finfo(np.float64).eps:
            valid_h.append(float(h))
            errors.append(float(error))
    if len(errors) < 2:
        return float("nan")
    slope = float(np.polyfit(np.log(valid_h), np.log(errors), 1)[0])
    return slope - 1.0


def _observed_order(dts: Iterable[float], errors: Iterable[float]) -> float:
    pairs = sorted(
        (float(dt), float(error))
        for dt, error in zip(dts, errors, strict=True)
        if np.isfinite(error) and error > 0.0
    )
    if len(pairs) < 2:
        return float("nan")
    fit_pairs = pairs[: min(3, len(pairs))]
    return float(
        np.polyfit(
            np.log([item[0] for item in fit_pairs]),
            np.log([item[1] for item in fit_pairs]),
            1,
        )[0]
    )


def _reference_period(
    config: SingleNeuronConfig,
    dts: Sequence[float],
    cancel_callback: CancelCallback | None,
) -> tuple[float, float, SingleNeuronTrace]:
    fine_dt = min(float(value) for value in dts) / 10.0
    coarse_config = replace(
        config,
        method_id="midpoint",
        reset_mode="event",
        dt_ms=2.0 * fine_dt,
    )
    fine_config = replace(coarse_config, dt_ms=fine_dt)
    coarse = simulate_single_neuron(
        coarse_config,
        record_state=False,
        cancel_callback=cancel_callback,
    )
    fine = simulate_single_neuron(
        fine_config,
        record_state=False,
        cancel_callback=cancel_callback,
    )
    coarse_period = _period_ms(coarse, config.burn_in_ms)
    fine_period = _period_ms(fine, config.burn_in_ms)
    if not np.isfinite(coarse_period) or not np.isfinite(fine_period):
        raise RuntimeError(
            "The reference simulation did not produce enough stable spikes. "
            "Increase input current or duration."
        )
    richardson_period = fine_period + (fine_period - coarse_period) / 3.0
    return float(richardson_period), float(fine_dt), fine


def _drift_rate(
    trace: SingleNeuronTrace,
    reference: SingleNeuronTrace,
    burn_in_ms: float,
) -> float:
    count = min(trace.spike_times_ms.size, reference.spike_times_ms.size)
    if count < 5:
        return float("nan")
    reference_times = reference.spike_times_ms[:count]
    differences = trace.spike_times_ms[:count] - reference_times
    mask = reference_times >= burn_in_ms
    if int(np.count_nonzero(mask)) < 3:
        return float("nan")
    slope = np.polyfit(reference_times[mask], differences[mask], 1)[0]
    return float(abs(slope) * 1000.0)


def run_convergence_study(
    config: SingleNeuronConfig,
    *,
    method_ids: Sequence[str] = DEFAULT_COMPARISON_METHODS,
    dt_values_ms: Sequence[float] = DEFAULT_DT_VALUES_MS,
    cancel_callback: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    _reference_data: tuple[float, float, SingleNeuronTrace] | None = None,
) -> ConvergenceStudy:
    """Compare spike-period convergence and long-horizon timing drift."""

    if not method_ids or not dt_values_ms:
        raise ValueError("Convergence study requires at least one method and one dt")
    for method_id in method_ids:
        if method_id not in INTEGRATORS:
            raise ValueError(f"Unknown single-neuron integrator: {method_id!r}")
    if any(not np.isfinite(dt) or dt <= 0.0 for dt in dt_values_ms):
        raise ValueError("All dt values must be finite and positive")

    reference_period, reference_dt, reference_trace = (
        _reference_period(config, dt_values_ms, cancel_callback)
        if _reference_data is None
        else _reference_data
    )
    total = len(method_ids) * len(dt_values_ms)
    completed = 0
    result_series: list[ConvergenceSeries] = []

    for method_id in method_ids:
        points: list[ConvergencePoint] = []
        for dt_ms in dt_values_ms:
            _check_cancelled(cancel_callback)
            trial_config = replace(config, method_id=method_id, dt_ms=float(dt_ms))
            trace = simulate_single_neuron(
                trial_config,
                record_state=False,
                cancel_callback=cancel_callback,
            )
            period = _period_ms(trace, config.burn_in_ms)
            error = abs(period - reference_period) if np.isfinite(period) else float("nan")
            points.append(
                ConvergencePoint(
                    dt_ms=float(dt_ms),
                    period_ms=float(period),
                    period_error_ms=float(error),
                    drift_ms_per_s=_drift_rate(trace, reference_trace, config.burn_in_ms),
                    nanoseconds_per_step=trace.elapsed_ns / max(1, trace.simulated_steps),
                    diverged=trace.diverged,
                )
            )
            completed += 1
            if progress_callback is not None:
                progress_callback("convergence", completed, total)

        result_series.append(
            ConvergenceSeries(
                method_id=method_id,
                observed_period_order=_observed_order(
                    (point.dt_ms for point in points),
                    (point.period_error_ms for point in points),
                ),
                local_order=estimate_local_order(config, method_id),
                points=tuple(points),
            )
        )

    return ConvergenceStudy(
        reference_period_ms=reference_period,
        reference_dt_ms=reference_dt,
        reset_mode=config.reset_mode,
        series=tuple(result_series),
    )


def run_fi_curves(
    config: SingleNeuronConfig,
    *,
    method_ids: Sequence[str] = DEFAULT_COMPARISON_METHODS,
    currents: Sequence[float] = tuple(np.arange(0.0, 35.01, 1.0)),
    cancel_callback: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[FICurve, ...]:
    """Measure f-I curves without fitting a controller-specific reduced model."""

    if not method_ids or not currents:
        raise ValueError("f-I study requires at least one method and one current")
    for method_id in method_ids:
        if method_id not in INTEGRATORS:
            raise ValueError(f"Unknown single-neuron integrator: {method_id!r}")
    total = len(method_ids) * len(currents)
    completed = 0
    curves: list[FICurve] = []
    current_values = np.asarray(currents, dtype=np.float64)
    if not np.all(np.isfinite(current_values)):
        raise ValueError("All current values must be finite")
    for method_id in method_ids:
        rates: list[float] = []
        diverged: list[bool] = []
        for current in current_values:
            _check_cancelled(cancel_callback)
            trial = replace(config, method_id=method_id, input_current=float(current))
            trace = simulate_single_neuron(
                trial,
                record_state=False,
                cancel_callback=cancel_callback,
            )
            rates.append(trace_metrics(trace, config.burn_in_ms).firing_rate_hz)
            diverged.append(trace.diverged)
            completed += 1
            if progress_callback is not None:
                progress_callback("fi_curve", completed, total)
        curves.append(
            FICurve(
                method_id=method_id,
                currents=current_values.copy(),
                rates_hz=np.asarray(rates, dtype=np.float64),
                diverged=np.asarray(diverged, dtype=np.bool_),
            )
        )
    return tuple(curves)


def run_alpha_sweep(
    config: SingleNeuronConfig,
    *,
    alpha_values: Sequence[float] = DEFAULT_ALPHA_VALUES,
    dt_values_ms: Sequence[float] = DEFAULT_DT_VALUES_MS,
    cancel_callback: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
    _reference_period_ms: float | None = None,
) -> tuple[AlphaSweepPoint, ...]:
    """Test where the complex two-stage Euler composition reaches order two."""

    if not alpha_values or not dt_values_ms:
        raise ValueError("Alpha sweep requires at least one alpha and one dt")
    if any(not np.isfinite(alpha) or alpha < 0.0 for alpha in alpha_values):
        raise ValueError("All alpha values must be finite and non-negative")
    if any(not np.isfinite(dt) or dt <= 0.0 for dt in dt_values_ms):
        raise ValueError("All dt values must be finite and positive")
    reference_period = (
        _reference_period(config, dt_values_ms, cancel_callback)[0]
        if _reference_period_ms is None
        else float(_reference_period_ms)
    )
    total = len(alpha_values) * len(dt_values_ms)
    completed = 0
    points: list[AlphaSweepPoint] = []
    for alpha in alpha_values:
        errors: list[float] = []
        for dt_ms in dt_values_ms:
            _check_cancelled(cancel_callback)
            trial = replace(
                config,
                method_id="euler_composition_complex",
                composition_alpha=float(alpha),
                dt_ms=float(dt_ms),
            )
            trace = simulate_single_neuron(
                trial,
                record_state=False,
                cancel_callback=cancel_callback,
            )
            period = _period_ms(trace, config.burn_in_ms)
            errors.append(abs(period - reference_period) if np.isfinite(period) else float("nan"))
            completed += 1
            if progress_callback is not None:
                progress_callback("alpha_sweep", completed, total)
        points.append(
            AlphaSweepPoint(
                alpha=float(alpha),
                local_order=estimate_local_order(
                    config,
                    "euler_composition_complex",
                    composition_alpha=float(alpha),
                ),
                observed_period_order=_observed_order(dt_values_ms, errors),
            )
        )
    return tuple(points)


def run_single_neuron_study(
    config: SingleNeuronConfig,
    *,
    method_ids: Sequence[str] = DEFAULT_COMPARISON_METHODS,
    dt_values_ms: Sequence[float] = DEFAULT_DT_VALUES_MS,
    currents: Sequence[float] = tuple(np.arange(0.0, 35.01, 1.0)),
    alpha_values: Sequence[float] = DEFAULT_ALPHA_VALUES,
    cancel_callback: CancelCallback | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SingleNeuronStudy:
    """Run the complete single-neuron layer of the numerical-method ablation."""

    if progress_callback is not None:
        progress_callback("trace", 0, 1)
    trace = simulate_single_neuron(config, cancel_callback=cancel_callback)
    if progress_callback is not None:
        progress_callback("trace", 1, 1)
    event_config = replace(config, reset_mode="event")
    reference_data = _reference_period(event_config, dt_values_ms, cancel_callback)
    convergence = run_convergence_study(
        event_config,
        method_ids=method_ids,
        dt_values_ms=dt_values_ms,
        cancel_callback=cancel_callback,
        progress_callback=progress_callback,
        _reference_data=reference_data,
    )
    grid_convergence = run_convergence_study(
        replace(config, reset_mode="grid"),
        method_ids=method_ids,
        dt_values_ms=dt_values_ms,
        cancel_callback=cancel_callback,
        progress_callback=(
            None
            if progress_callback is None
            else lambda _stage, value, maximum: progress_callback(
                "grid_convergence", value, maximum
            )
        ),
        _reference_data=reference_data,
    )
    curves = run_fi_curves(
        event_config,
        method_ids=method_ids,
        currents=currents,
        cancel_callback=cancel_callback,
        progress_callback=progress_callback,
    )
    alpha_sweep = run_alpha_sweep(
        event_config,
        alpha_values=alpha_values,
        dt_values_ms=dt_values_ms,
        cancel_callback=cancel_callback,
        progress_callback=progress_callback,
        _reference_period_ms=reference_data[0],
    )
    return SingleNeuronStudy(
        config=config,
        trace=trace,
        metrics=trace_metrics(trace, config.burn_in_ms),
        convergence=convergence,
        grid_convergence=grid_convergence,
        fi_curves=curves,
        alpha_sweep=alpha_sweep,
    )
