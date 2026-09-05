from dataclasses import replace

import numpy as np

from snnlab.core.numerical_methods import IzhikevichParameters, get_stepper
from snnlab.experiments.single_neuron import (
    SingleNeuronConfig,
    _period_ms,
    estimate_local_order,
    run_convergence_study,
    run_fi_curves,
    run_single_neuron_study,
    simulate_single_neuron,
    step_single_neuron,
    trace_metrics,
)
from snnlab.runtime.single_neuron_io import save_single_neuron_study


def test_scalar_integrators_match_network_steppers() -> None:
    config = SingleNeuronConfig(duration_ms=10.0, burn_in_ms=1.0)
    v = -61.0
    u = -12.5
    current = 9.0
    dt_ms = 0.1
    vector_parameters = IzhikevichParameters(a=config.neuron.a, b=config.neuron.b)

    for method_id in (
        "explicit_euler",
        "semi_euler",
        "semi_implicit_euler",
        "implicit_euler",
        "midpoint",
        "semi_midpoint",
        "semi_implicit_midpoint",
    ):
        scalar_v, scalar_u = step_single_neuron(
            method_id,
            v,
            u,
            current,
            dt_ms,
            config.neuron,
        )
        vector_v, vector_u = get_stepper(method_id)(
            np.asarray([v]),
            np.asarray([u]),
            np.asarray([current]),
            dt_ms,
            vector_parameters,
        )
        np.testing.assert_allclose([scalar_v, scalar_u], [vector_v[0], vector_u[0]])


def test_recordless_trace_preserves_custom_initial_state() -> None:
    config = SingleNeuronConfig(
        duration_ms=1.0,
        burn_in_ms=0.5,
        initial_voltage_mv=-70.0,
        initial_recovery=-15.0,
    )
    trace = simulate_single_neuron(config, record_state=False)

    assert trace.voltage_mv[0] == -70.0
    assert trace.recovery[0] == -15.0


def test_event_reset_records_substep_spike_times_and_reduces_period_error() -> None:
    base = SingleNeuronConfig(
        input_current=12.0,
        duration_ms=1200.0,
        burn_in_ms=300.0,
        dt_ms=0.2,
        method_id="midpoint",
        reset_mode="event",
    )
    reference = simulate_single_neuron(replace(base, dt_ms=0.0025), record_state=False)
    event_trace = simulate_single_neuron(base, record_state=False)
    grid_trace = simulate_single_neuron(replace(base, reset_mode="grid"), record_state=False)

    reference_period = _period_ms(reference, base.burn_in_ms)
    event_error = abs(_period_ms(event_trace, base.burn_in_ms) - reference_period)
    grid_error = abs(_period_ms(grid_trace, base.burn_in_ms) - reference_period)

    assert np.any(np.mod(event_trace.spike_times_ms, base.dt_ms) > 1e-6)
    assert event_error < grid_error


def test_complex_composition_is_second_order_only_at_alpha_half() -> None:
    config = SingleNeuronConfig(duration_ms=100.0, burn_in_ms=10.0)

    order_at_half = estimate_local_order(
        config,
        "euler_composition_complex",
        composition_alpha=0.5,
    )
    order_away = estimate_local_order(
        config,
        "euler_composition_complex",
        composition_alpha=0.75,
    )
    real_composition_order = estimate_local_order(config, "euler_composition_real")

    assert 1.8 < order_at_half < 2.2
    assert 0.8 < order_away < 1.2
    assert 0.8 < real_composition_order < 1.2


def test_trace_metrics_and_fi_curve_are_finite() -> None:
    config = SingleNeuronConfig(duration_ms=700.0, burn_in_ms=200.0)
    trace = simulate_single_neuron(config)
    metrics = trace_metrics(trace, config.burn_in_ms)
    curves = run_fi_curves(
        config,
        method_ids=("midpoint",),
        currents=(0.0, 12.0, 20.0),
    )

    assert not trace.diverged
    assert metrics.spike_count > 0
    assert metrics.firing_rate_hz > 0.0
    assert curves[0].rates_hz[0] == 0.0
    assert np.all(np.isfinite(curves[0].rates_hz))
    assert curves[0].rates_hz[-1] > curves[0].rates_hz[1]


def test_small_convergence_study_reports_expected_order_separation() -> None:
    config = SingleNeuronConfig(duration_ms=1000.0, burn_in_ms=250.0)
    result = run_convergence_study(
        config,
        method_ids=("explicit_euler", "midpoint"),
        dt_values_ms=(0.2, 0.1, 0.05),
    )
    by_method = {series.method_id: series for series in result.series}

    assert 0.7 < by_method["explicit_euler"].observed_period_order < 1.3
    assert by_method["midpoint"].observed_period_order > 1.5
    assert by_method["midpoint"].local_order > 1.8


def test_full_study_runs_both_reset_controls_and_persists(tmp_path) -> None:
    config = SingleNeuronConfig(duration_ms=500.0, burn_in_ms=100.0)
    study = run_single_neuron_study(
        config,
        method_ids=("midpoint",),
        dt_values_ms=(0.2, 0.1),
        currents=(0.0, 12.0),
        alpha_values=(0.5,),
    )
    run_dir = save_single_neuron_study(study, root=tmp_path)

    assert study.convergence.reset_mode == "event"
    assert study.grid_convergence.reset_mode == "grid"
    assert (run_dir / "study.json").is_file()
    assert (run_dir / "environment.json").is_file()
    with np.load(run_dir / "trace.npz") as trace:
        np.testing.assert_allclose(trace["spike_times_ms"], study.trace.spike_times_ms)
