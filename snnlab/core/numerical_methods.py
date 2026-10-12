from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray
Stepper = Callable[[Array, Array, Array, float, "IzhikevichParameters"], tuple[Array, Array]]


@dataclass(frozen=True, slots=True)
class IzhikevichParameters:
    """
    Stores Izhikevich recovery parameters used by integration methods.

    Хранит параметры восстановления модели Ижикевича, используемые методами интегрирования.
    """

    a: float = 0.02
    b: float = 0.20


def izhikevich_rhs(
    v: Array,
    u: Array,
    current: Array,
    params: IzhikevichParameters,
) -> tuple[Array, Array]:
    """Evaluate the smooth Izhikevich ODE before threshold/reset handling."""

    dv = 0.04 * v * v + 5.0 * v + 140.0 - u + current
    du = params.a * (params.b * v - u)
    return dv, du


def explicit_euler(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Performs one explicit-Euler step.

    Выполняет один шаг явного метода Эйлера.
    """
    dv, du = izhikevich_rhs(v, u, current, params)
    return v + dt * dv, u + dt * du


def semi_euler(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Performs the semi-explicit Euler variant where u uses v_(n+1).

    Выполняет полуявный вариант Эйлера, где u использует v_(n+1).
    """
    v_new = v + dt * (0.04 * v * v + 5.0 * v + 140.0 - u + current)
    u_new = u + dt * params.a * (params.b * v_new - u)
    return v_new, u_new


def semi_euler_uv(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """Semi-explicit Euler with the reverse update order: u then v."""
    u_new = u + dt * params.a * (params.b * v - u)
    v_new = v + dt * (0.04 * v * v + 5.0 * v + 140.0 - u_new + current)
    return v_new, u_new


def semi_implicit_euler(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Performs the current semi-implicit Euler approximation.

    Выполняет текущий полунеявный вариант Эйлера.
    """
    u_new = (u + dt * params.a * params.b * v) / (1.0 + dt * params.a)
    v_tmp = v + dt * (0.04 * v * v + 5.0 * v + 140.0 - u_new + current)
    v_new = v + dt * (0.04 * v_tmp * v_tmp + 5.0 * v_tmp + 140.0 - u_new + current)
    return v_new, u_new


def implicit_euler(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Solve the quadratic backward-Euler equation on the branch continuous at dt=0.

    The nonlinear equation need not have a real root near the spike. Return NaN
    in that case; the caller must shorten the step or report that it failed.
    """
    h = np.asarray(dt, dtype=np.float64)
    v_old = np.asarray(v, dtype=np.float64)
    u_old = np.asarray(u, dtype=np.float64)
    denominator = 1.0 + h * params.a
    linear = 1.0 - 5.0 * h + h * h * params.a * params.b / denominator
    constant = v_old + h * (140.0 + current - u_old / denominator)
    discriminant = linear * linear - 0.16 * h * constant
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        radical = np.sqrt(discriminant)
        v_new = np.where(
            linear > 0.0,
            2.0 * constant / (linear + radical),
            (linear - radical) / (0.08 * h),
        )
        u_new = (u_old + h * params.a * params.b * v_new) / denominator
    return v_new, u_new


def implicit_euler_step_doubling(
    v: Array,
    u: Array,
    current: Array,
    dt: Array | float,
    params: IzhikevichParameters,
    *,
    rtol: float = 1e-3,
    atol_v: float = 0.05,
    atol_u: float = 0.01,
) -> tuple[Array, Array, Array, Array, Array]:
    """Two half steps and a full step; their difference estimates local error.

    Keep the two-half-step solution, without Richardson extrapolation, so the
    accepted smooth integrator remains first-order backward Euler.
    """
    full_v, full_u = implicit_euler(v, u, current, dt, params)
    middle_v, middle_u = implicit_euler(v, u, current, np.asarray(dt) / 2, params)
    fine_v, fine_u = implicit_euler(
        middle_v, middle_u, current, np.asarray(dt) / 2, params
    )
    voltage_scale = atol_v + rtol * np.maximum(np.abs(v), np.abs(fine_v))
    recovery_scale = atol_u + rtol * np.maximum(np.abs(u), np.abs(fine_u))
    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.maximum(
            np.abs(fine_v - full_v) / voltage_scale,
            np.abs(fine_u - full_u) / recovery_scale,
        )
    return middle_v, middle_u, fine_v, fine_u, error


def implicit_euler_grid_step(
    v: Array,
    u: Array,
    current: Array,
    dt: float,
    params: IzhikevichParameters,
    *,
    threshold: float,
    reset_voltage: float,
    reset_recovery: float,
    state_limit: float = 500.0,
    rtol: float = 1e-3,
    atol_v: float = 0.05,
    atol_u: float = 0.01,
    return_solves: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
    """Error-controlled backward Euler for one or many grid-reset neurons.

    A full step and two half steps control each neuron's local error independently.
    A spike found inside the interval is reported at its outer grid boundary.
    The optional solve count records three quadratic solves per attempted step.
    """
    voltage = np.array(v, dtype=np.float64, copy=True)
    recovery = np.array(u, dtype=np.float64, copy=True)
    drive = np.broadcast_to(np.asarray(current, dtype=np.float64), voltage.shape)
    remaining = np.full(voltage.shape, dt, dtype=np.float64)
    proposed = remaining.copy()
    spikes = np.zeros(voltage.shape, dtype=np.bool_)
    failed = np.zeros(voltage.shape, dtype=np.bool_)
    solves = np.zeros(voltage.shape, dtype=np.int64)
    for _ in range(4096):
        active = (remaining > 1e-12) & ~spikes & ~failed
        if not np.any(active):
            break
        indices = np.flatnonzero(active)
        h = np.minimum(proposed[active], remaining[active])
        solves.flat[indices] += 3
        v_mid, u_mid, v_new, u_new, error = implicit_euler_step_doubling(
            voltage[active], recovery[active], drive[active], h, params,
            rtol=rtol, atol_v=atol_v, atol_u=atol_u,
        )
        valid = (
            np.isfinite(v_mid) & np.isfinite(u_mid)
            & np.isfinite(v_new) & np.isfinite(u_new) & np.isfinite(error)
            & (np.abs(v_mid) <= state_limit) & (np.abs(u_mid) <= state_limit)
            & (np.abs(v_new) <= state_limit) & (np.abs(u_new) <= state_limit)
        )
        accepted = valid & (error <= 1.0)
        rejected = ~accepted
        if np.any(rejected):
            retry = indices[rejected]
            safe_error = np.where(np.isfinite(error[rejected]), error[rejected], np.inf)
            factor = np.clip(0.9 / np.sqrt(np.maximum(safe_error, 1e-12)), 0.2, 0.5)
            proposed.flat[retry] = h[rejected] * factor
            failed.flat[retry] = proposed.flat[retry] < 1e-9
        crossed = accepted & (v_new >= threshold)
        if np.any(crossed):
            hit = indices[crossed]
            first = v_mid[crossed] >= threshold
            old_v = np.where(first, voltage.flat[hit], v_mid[crossed])
            old_u = np.where(first, recovery.flat[hit], u_mid[crossed])
            new_v = np.where(first, v_mid[crossed], v_new[crossed])
            new_u = np.where(first, u_mid[crossed], u_new[crossed])
            fraction = np.clip((threshold - old_v) / (new_v - old_v), 0.0, 1.0)
            voltage.flat[hit] = reset_voltage
            recovery.flat[hit] = old_u + fraction * (new_u - old_u) + reset_recovery
            spikes.flat[hit] = True
            remaining.flat[hit] = 0.0
        keep_mask = accepted & ~crossed
        if np.any(keep_mask):
            keep = indices[keep_mask]
            voltage.flat[keep] = v_new[keep_mask]
            recovery.flat[keep] = u_new[keep_mask]
            remaining.flat[keep] = np.maximum(0.0, remaining.flat[keep] - h[keep_mask])
            factor = np.clip(0.9 / np.sqrt(np.maximum(error[keep_mask], 1e-12)), 0.5, 2.0)
            proposed.flat[keep] = np.minimum(remaining.flat[keep], h[keep_mask] * factor)
    else:
        failed |= (remaining > 1e-12) & ~spikes
    voltage[failed] = np.nan
    recovery[failed] = np.nan
    if return_solves:
        return voltage, recovery, spikes, solves
    return voltage, recovery, spikes


def midpoint(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Performs one classical midpoint step.

    Выполняет один шаг классического метода средней точки.
    """
    half_dt = 0.5 * dt
    dv1, du1 = izhikevich_rhs(v, u, current, params)
    v_mid = v + half_dt * dv1
    u_mid = u + half_dt * du1
    dv2, du2 = izhikevich_rhs(v_mid, u_mid, current, params)
    return v + dt * dv2, u + dt * du2


def semi_midpoint(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Performs the semi-explicit midpoint variant.

    Выполняет полуявный вариант метода средней точки.
    """
    half_dt = 0.5 * dt
    v_mid = v + half_dt * (0.04 * v * v + 5.0 * v + 140.0 - u + current)
    u_mid = u + half_dt * params.a * (params.b * v_mid - u)
    dv2, du2 = izhikevich_rhs(v_mid, u_mid, current, params)
    return v + dt * dv2, u + dt * du2


def semi_midpoint_uv(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """Semi-explicit midpoint with u then v at the intermediate stage."""
    half_dt = 0.5 * dt
    u_mid = u + half_dt * params.a * (params.b * v - u)
    v_mid = v + half_dt * (0.04 * v * v + 5.0 * v + 140.0 - u_mid + current)
    dv2, du2 = izhikevich_rhs(v_mid, u_mid, current, params)
    return v + dt * dv2, u + dt * du2


def semi_implicit_midpoint(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Performs the current semi-implicit midpoint approximation.

    Выполняет текущий полунеявный вариант метода средней точки.
    """
    half_dt = 0.5 * dt
    u_mid = (u + half_dt * params.a * params.b * v) / (1.0 + half_dt * params.a)
    v_tmp = v + half_dt * (0.04 * v * v + 5.0 * v + 140.0 - u_mid + current)
    v_mid = v + half_dt * (0.04 * v_tmp * v_tmp + 5.0 * v_tmp + 140.0 - u_mid + current)
    dv2, du2 = izhikevich_rhs(v_mid, u_mid, current, params)
    return v + dt * dv2, u + dt * du2


_METHODS: dict[str, Stepper] = {
    "explicit_euler": explicit_euler,
    "semi_euler": semi_euler,
    "semi_euler_uv": semi_euler_uv,
    "semi_implicit_euler": semi_implicit_euler,
    "implicit_euler": implicit_euler,
    "midpoint": midpoint,
    "semi_midpoint": semi_midpoint,
    "semi_midpoint_uv": semi_midpoint_uv,
    "semi_implicit_midpoint": semi_implicit_midpoint,
}


def get_stepper(method_id: str) -> Stepper:
    """
    Returns a numerical method by stable English identifier.

    Возвращает численный метод по стабильному английскому идентификатору.
    """
    try:
        return _METHODS[method_id]
    except KeyError as exc:
        available = ", ".join(sorted(_METHODS))
        raise KeyError(f"Unknown numerical method {method_id!r}. Available: {available}") from exc


def available_methods() -> tuple[str, ...]:
    """
    Returns all registered numerical-method identifiers.

    Возвращает все зарегистрированные идентификаторы численных методов.
    """
    return tuple(sorted(_METHODS))
