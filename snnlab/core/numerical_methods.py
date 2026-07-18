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


def _rhs(v: Array, u: Array, current: Array, params: IzhikevichParameters) -> tuple[Array, Array]:
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
    dv, du = _rhs(v, u, current, params)
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
    Performs vectorized implicit Euler using bounded Newton iterations.

    Выполняет векторизованный неявный Эйлер с ограниченным числом итераций Ньютона.
    """
    v_new = np.array(v, dtype=np.float64, copy=True)
    u_new = np.array(u, dtype=np.float64, copy=True)

    # EN: Eight iterations preserve the current reservoir implementation's cost profile.
    # RU: Восемь итераций сохраняют текущий профиль вычислительной стоимости reservoir-ветки.
    max_iter = 8
    tol = 1e-7

    for _ in range(max_iter):
        f1 = v_new - v - dt * (0.04 * v_new * v_new + 5.0 * v_new + 140.0 - u_new + current)
        f2 = u_new - u - dt * params.a * (params.b * v_new - u_new)

        if max(float(np.max(np.abs(f1))), float(np.max(np.abs(f2)))) < tol:
            break

        j11 = 1.0 - dt * (0.08 * v_new + 5.0)
        j12 = dt
        j21 = -dt * params.a * params.b
        j22 = 1.0 + dt * params.a

        det = j11 * j22 - j12 * j21
        det = np.where(np.abs(det) < 1e-10, np.where(det < 0.0, -1e-10, 1e-10), det)

        dv = -(j22 * f1 - j12 * f2) / det
        du = -(-j21 * f1 + j11 * f2) / det

        v_new += dv
        u_new += du

    return v_new, u_new


def midpoint(
    v: Array, u: Array, current: Array, dt: float, params: IzhikevichParameters
) -> tuple[Array, Array]:
    """
    Performs one classical midpoint step.

    Выполняет один шаг классического метода средней точки.
    """
    half_dt = 0.5 * dt
    dv1, du1 = _rhs(v, u, current, params)
    v_mid = v + half_dt * dv1
    u_mid = u + half_dt * du1
    dv2, du2 = _rhs(v_mid, u_mid, current, params)
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
    dv2, du2 = _rhs(v_mid, u_mid, current, params)
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
    dv2, du2 = _rhs(v_mid, u_mid, current, params)
    return v + dt * dv2, u + dt * du2


_METHODS: dict[str, Stepper] = {
    "explicit_euler": explicit_euler,
    "semi_euler": semi_euler,
    "semi_implicit_euler": semi_implicit_euler,
    "implicit_euler": implicit_euler,
    "midpoint": midpoint,
    "semi_midpoint": semi_midpoint,
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
