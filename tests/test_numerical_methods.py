import numpy as np

from snnlab.core.numerical_methods import (
    IzhikevichParameters,
    available_methods,
    get_stepper,
    implicit_euler_grid_step,
)


def test_all_methods_produce_expected_shapes() -> None:
    v = np.array([-65.0, -64.0])
    u = 0.2 * v
    current = np.array([5.0, 6.0])
    params = IzhikevichParameters()

    assert len(available_methods()) == 9
    for method_id in available_methods():
        v_new, u_new = get_stepper(method_id)(v, u, current, 0.1, params)
        assert v_new.shape == v.shape
        assert u_new.shape == u.shape
        assert np.all(np.isfinite(v_new))
        assert np.all(np.isfinite(u_new))


def test_implicit_euler_solves_equation_and_rejects_missing_real_root() -> None:
    params = IzhikevichParameters()
    v, u, current, h = -61.0, -12.5, 9.0, 0.1
    v_new, u_new = get_stepper("implicit_euler")(v, u, current, h, params)
    assert abs(v_new - v - h * (0.04 * v_new**2 + 5 * v_new + 140 - u_new + current)) < 1e-10
    assert abs(u_new - u - h * params.a * (params.b * v_new - u_new)) < 1e-10
    missing, _ = get_stepper("implicit_euler")(0.0, 0.0, 12.0, 0.2, params)
    assert np.isnan(missing)


def test_implicit_population_resolves_spikes_at_coarse_step() -> None:
    params = IzhikevichParameters()
    v = np.full(8, -65.0)
    u = params.b * v
    total_spikes = np.zeros(8, dtype=np.int64)
    for _ in range(300):
        v, u, spikes = implicit_euler_grid_step(
            v, u, np.linspace(10.0, 18.0, 8), 0.4, params,
            threshold=30.0, reset_voltage=-65.0, reset_recovery=8.0,
        )
        total_spikes += spikes
        assert np.all(np.isfinite(v)) and np.all(np.isfinite(u))
    assert np.all(total_spikes > 0)
