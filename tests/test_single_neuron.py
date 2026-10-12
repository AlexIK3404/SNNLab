import json
from dataclasses import replace

import numpy as np
import pytest

from snnlab.architectures.dci import NeuronPopulationState, _population_step
from snnlab.core.numerical_methods import IzhikevichParameters, get_stepper
from snnlab.experiments.single_neuron import (
    SingleNeuronConfig,
    StudySelection,
    _period_ms,
    estimate_local_order,
    observed_trace_arrays,
    run_convergence_study,
    run_fi_curves,
    run_single_neuron_study,
    run_trial_variability,
    simulate_single_neuron,
    step_single_neuron,
    trace_metrics,
)
from snnlab.runtime.single_neuron_io import load_single_neuron_study, save_single_neuron_study


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
        "semi_euler_uv",
        "semi_implicit_euler",
        "implicit_euler",
        "midpoint",
        "semi_midpoint",
        "semi_midpoint_uv",
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


def test_semi_explicit_commutations_use_opposite_stage_order() -> None:
    params = IzhikevichParameters(a=0.02, b=0.2)
    v, u, current, dt = -61.0, -12.5, 9.0, 0.2
    dv = 0.04 * v * v + 5 * v + 140 - u + current
    du = params.a * (params.b * v - u)
    v_first, u_second = get_stepper("semi_euler")(v, u, current, dt, params)
    v_second, u_first = get_stepper("semi_euler_uv")(v, u, current, dt, params)
    assert v_first == pytest.approx(v + dt * dv)
    assert u_first == pytest.approx(u + dt * du)
    assert v_second != pytest.approx(v_first, abs=1e-9)
    assert u_second != pytest.approx(u_first, abs=1e-9)

    midpoint_vu = get_stepper("semi_midpoint")(v, u, current, dt, params)
    midpoint_uv = get_stepper("semi_midpoint_uv")(v, u, current, dt, params)
    assert not np.allclose(midpoint_vu, midpoint_uv, atol=1e-10, rtol=0.0)


def test_semi_explicit_midpoint_commutations_preserve_second_order() -> None:
    config = SingleNeuronConfig(duration_ms=100.0, burn_in_ms=10.0)
    for method_id in ("semi_midpoint", "semi_midpoint_uv"):
        assert 1.8 < estimate_local_order(config, method_id) < 2.2


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


@pytest.mark.parametrize("preset", ["regular_spiking", "fast_spiking"])
@pytest.mark.parametrize("reset", ["grid", "event"])
def test_implicit_euler_adapts_before_spikes(preset: str, reset: str) -> None:
    from snnlab.experiments.single_neuron import neuron_preset

    config = SingleNeuronConfig(
        neuron=neuron_preset(preset), input_current=12.0,
        duration_ms=200.0, burn_in_ms=20.0, dt_ms=0.4,
        method_id="implicit_euler", reset_mode=reset,
    )
    trace = simulate_single_neuron(config, record_state=False)
    assert not trace.diverged
    assert trace.spike_times_ms.size >= 4
    assert trace.implicit_step_attempts > trace.simulated_steps
    assert np.isfinite(_period_ms(trace, config.burn_in_ms))


def test_grid_implicit_euler_matches_dci_population_with_same_conditions() -> None:
    config = SingleNeuronConfig(
        duration_ms=120.0, burn_in_ms=0.0, dt_ms=0.4,
        method_id="implicit_euler", reset_mode="grid", input_current=12.0,
        implicit_rtol=2e-4, implicit_atol_v=0.02, implicit_atol_u=0.005,
    )
    trace = simulate_single_neuron(config)
    state = NeuronPopulationState(
        v=np.array([-65.0]), u=np.array([-13.0]),
        g_exc=np.zeros(1), g_inh=np.zeros(1),
        refractory_left_ms=np.zeros(1),
    )
    voltages = [-65.0]
    recoveries = [-13.0]
    spikes = []
    for index in range(300):
        fired = _population_step(
            state, a=0.02, b=0.2, c=-65.0, d=8.0, v_peak=30.0,
            refractory_ms=0.0, dt_ms=0.4, e_exc=0.0, e_inh=-80.0,
            bias_current=12.0, numerical_v_min=-500.0, numerical_v_max=500.0,
            numerical_method="implicit_euler", implicit_rtol=config.implicit_rtol,
            implicit_atol_v=config.implicit_atol_v, implicit_atol_u=config.implicit_atol_u,
        )
        voltages.append(state.v[0])
        recoveries.append(state.u[0])
        if fired[0]:
            spikes.append((index + 1) * 0.4)
    np.testing.assert_allclose(trace.voltage_mv, voltages, atol=1e-10)
    np.testing.assert_allclose(trace.recovery, recoveries, atol=1e-10)
    np.testing.assert_allclose(trace.spike_times_ms, spikes, atol=1e-9)


def test_implicit_error_tolerance_controls_cost_and_smooth_accuracy() -> None:
    config = SingleNeuronConfig(
        input_current=2.0, duration_ms=4.0, burn_in_ms=0.0,
        dt_ms=0.4, method_id="implicit_euler", reset_mode="grid",
        initial_voltage_mv=-61.0, initial_recovery=-12.5,
    )
    loose = simulate_single_neuron(replace(config, implicit_rtol=0.05, implicit_atol_v=1.0))
    tight = simulate_single_neuron(
        replace(config, implicit_rtol=1e-5, implicit_atol_v=0.0001, implicit_atol_u=0.0001)
    )
    reference = simulate_single_neuron(replace(config, dt_ms=0.0005))
    assert not loose.diverged and not tight.diverged and not reference.diverged
    assert tight.implicit_step_attempts > loose.implicit_step_attempts
    assert abs(tight.voltage_mv[-1] - reference.voltage_mv[-1]) < abs(
        loose.voltage_mv[-1] - reference.voltage_mv[-1]
    )


def test_failed_step_does_not_append_invalid_plot_point(monkeypatch) -> None:
    import snnlab.experiments.single_neuron as single_neuron

    monkeypatch.setattr(
        single_neuron, "_advance_with_reset",
        lambda *_args: (1e9, -1e9, [], True, 3),
    )
    trace = simulate_single_neuron(SingleNeuronConfig(burn_in_ms=0.0, duration_ms=1.0))
    assert trace.diverged
    np.testing.assert_array_equal(trace.voltage_mv, [-65.0])
    np.testing.assert_array_equal(trace.recovery, [-13.0])


def test_implicit_inner_step_count_survives_saved_study(tmp_path) -> None:
    config = SingleNeuronConfig(
        duration_ms=120.0, burn_in_ms=20.0, dt_ms=0.4,
        method_id="implicit_euler", reset_mode="grid",
    )
    study = run_single_neuron_study(
        config, method_ids=("implicit_euler",), trace_dt_values_ms=(0.4,),
        selection=StudySelection(True, False, False, False, False),
    )
    reopened = load_single_neuron_study(save_single_neuron_study(study, root=tmp_path))
    assert reopened.comparison_traces[0].trace.implicit_step_attempts > 0
    assert (reopened.comparison_traces[0].trace.implicit_step_attempts
            == study.comparison_traces[0].trace.implicit_step_attempts)


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
    config = SingleNeuronConfig(duration_ms=1000.0, burn_in_ms=250.0, reset_mode="event")
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

    assert study.convergence.reset_mode == "grid"
    assert study.reset_control.reset_mode == "event"
    for series in study.convergence.series:
        finest = min(series.points, key=lambda point: point.dt_ms)
        trace = simulate_single_neuron(
            replace(config, method_id=series.method_id, dt_ms=finest.dt_ms),
            record_state=False,
        )
        assert finest.spike_count == trace_metrics(trace, config.burn_in_ms).spike_count
    assert study.convergence.reference_uncertainty_ms >= 0.0
    assert [item.method_id for item in study.comparison_traces] == ["midpoint"]
    np.testing.assert_allclose(
        study.comparison_traces[0].trace.time_ms,
        study.trace.time_ms,
    )
    assert (run_dir / "study.json").is_file()
    assert (run_dir / "environment.json").is_file()
    saved = json.loads((run_dir / "study.json").read_text())
    assert saved["axes"]["dt_values_ms"] == [0.2, 0.1]
    assert saved["axes"]["alpha_values"] == [0.5]
    assert len(saved["source_fingerprints_sha256"]["experiments/single_neuron.py"]) == 64
    with np.load(run_dir / "trace.npz") as trace:
        np.testing.assert_allclose(trace["spike_times_ms"], study.trace.spike_times_ms)
        np.testing.assert_allclose(
            trace["comparison__0000__voltage_mv"],
            study.comparison_traces[0].trace.voltage_mv,
        )


def test_full_study_records_selected_methods_on_one_time_grid() -> None:
    config = SingleNeuronConfig(
        duration_ms=500.0,
        burn_in_ms=100.0,
        dt_ms=0.2,
        method_id="explicit_euler",
    )
    study = run_single_neuron_study(
        config,
        method_ids=("explicit_euler", "midpoint"),
        dt_values_ms=(0.2, 0.1),
        currents=(12.0,),
        alpha_values=(0.5,),
    )

    assert [item.method_id for item in study.comparison_traces] == [
        "explicit_euler",
        "midpoint",
    ]
    np.testing.assert_allclose(
        study.comparison_traces[0].trace.time_ms,
        study.comparison_traces[1].trace.time_ms,
    )


def test_trace_only_study_skips_unselected_protocols(monkeypatch, tmp_path) -> None:
    from snnlab.experiments import single_neuron as experiment

    def forbidden(*args, **kwargs):
        raise AssertionError("Unselected protocol was run")

    for name in ("_reference_period", "run_convergence_study", "run_fi_curves", "run_alpha_sweep"):
        monkeypatch.setattr(experiment, name, forbidden)
    study = run_single_neuron_study(
        SingleNeuronConfig(duration_ms=100.0, burn_in_ms=10.0),
        method_ids=("semi_euler", "semi_euler_uv"),
        dt_values_ms=(),
        currents=(),
        alpha_values=(),
        selection=StudySelection(
            traces=True, convergence=False, fi=False, alpha=False, summary=False
        ),
    )
    assert study.trace is not None
    assert len(study.comparison_traces) == 2
    assert study.convergence is None and study.reset_control is None
    assert study.fi_curves == study.alpha_sweep == ()
    run_dir = save_single_neuron_study(study, root=tmp_path)
    assert json.loads((run_dir / "study.json").read_text())["selection"]["alpha"] is False


def test_alpha_only_study_does_not_require_methods_or_other_outputs(tmp_path) -> None:
    config = SingleNeuronConfig(duration_ms=500.0, burn_in_ms=100.0)
    study = run_single_neuron_study(
        config,
        method_ids=(),
        dt_values_ms=(0.2, 0.1),
        currents=(),
        alpha_values=(0.5,),
        selection=StudySelection(
            traces=False, convergence=False, fi=False, alpha=True, summary=False
        ),
    )
    assert study.trace is None and study.metrics is None
    assert study.convergence is None and study.reset_control is None
    assert len(study.alpha_sweep) == 1
    path = save_single_neuron_study(study, root=tmp_path)
    with np.load(path / "trace.npz") as arrays:
        assert arrays.files == []


def test_fi_only_does_not_build_reference_or_alpha_sweep(monkeypatch) -> None:
    from snnlab.experiments import single_neuron as experiment

    def forbidden(*args, **kwargs):
        raise AssertionError("Unselected protocol was run")

    monkeypatch.setattr(experiment, "_reference_period", forbidden)
    monkeypatch.setattr(experiment, "run_alpha_sweep", forbidden)
    study = run_single_neuron_study(
        SingleNeuronConfig(duration_ms=100.0, burn_in_ms=10.0),
        method_ids=("midpoint",),
        dt_values_ms=(),
        currents=(0.0, 12.0),
        alpha_values=(),
        selection=StudySelection(False, False, True, False, False),
    )
    assert len(study.fi_curves) == 1
    assert study.trace is study.convergence is study.reset_control is None


def test_convergence_plot_without_summary_skips_grid_control(monkeypatch) -> None:
    from snnlab.experiments import single_neuron as experiment

    original = experiment.run_convergence_study

    def only_event(config, **kwargs):
        assert config.reset_mode == "event"
        return original(config, **kwargs)

    monkeypatch.setattr(experiment, "run_convergence_study", only_event)
    study = run_single_neuron_study(
        SingleNeuronConfig(duration_ms=500.0, burn_in_ms=100.0, reset_mode="event"),
        method_ids=("midpoint",),
        dt_values_ms=(0.2, 0.1),
        currents=(),
        alpha_values=(),
        selection=StudySelection(False, True, False, False, False),
    )
    assert study.convergence is not None
    assert study.reset_control is None


def test_empty_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="Select at least one"):
        StudySelection(False, False, False, False, False)


def test_noise_is_reproducible_and_constant_input_does_not_depend_on_seed() -> None:
    base = SingleNeuronConfig(duration_ms=130.0, burn_in_ms=30.0, dt_ms=0.3)
    constant_a = simulate_single_neuron(base)
    constant_b = simulate_single_neuron(replace(base, seed=19))
    np.testing.assert_array_equal(constant_a.voltage_mv, constant_b.voltage_mv)

    noisy = replace(base, noise_std=6.0, noise_interval_ms=2.0, seed=123)
    first = simulate_single_neuron(noisy)
    second = simulate_single_neuron(noisy)
    different = simulate_single_neuron(replace(noisy, seed=124))
    np.testing.assert_array_equal(first.voltage_mv, second.voltage_mv)
    assert not np.array_equal(first.voltage_mv, different.voltage_mv)
    assert first.time_ms[-1] == pytest.approx(base.burn_in_ms + base.duration_ms)


def test_transient_is_separate_from_observation_window() -> None:
    config = SingleNeuronConfig(duration_ms=50.0, burn_in_ms=120.0, dt_ms=0.2)
    trace = simulate_single_neuron(config)
    time_ms, voltage, recovery, spikes = observed_trace_arrays(trace, config.burn_in_ms)
    assert trace.time_ms[-1] == pytest.approx(170.0)
    assert time_ms[0] >= 0.0
    assert time_ms[0] < config.dt_ms + 1e-9
    assert time_ms[-1] == pytest.approx(config.duration_ms)
    assert voltage.size == recovery.size == time_ms.size
    assert np.all((spikes >= 0.0) & (spikes <= config.duration_ms))
    assert trace_metrics(trace, config.burn_in_ms).firing_rate_hz == pytest.approx(
        spikes.size * 1000.0 / config.duration_ms
    )


def test_trajectory_dt_grid_round_trip_without_recalculation(tmp_path) -> None:
    config = SingleNeuronConfig(duration_ms=80.0, burn_in_ms=20.0, dt_ms=0.1)
    selection = StudySelection(
        traces=True, convergence=False, fi=False, alpha=False, summary=False
    )
    study = run_single_neuron_study(
        config,
        method_ids=("explicit_euler", "midpoint"),
        trace_dt_values_ms=(0.2, 0.1, 0.05),
        dt_values_ms=(), currents=(), alpha_values=(), selection=selection,
    )
    assert [(item.method_id, item.dt_ms) for item in study.comparison_traces] == [
        (method, dt) for dt in (0.2, 0.1, 0.05) for method in ("explicit_euler", "midpoint")
    ]
    path = save_single_neuron_study(study, root=tmp_path)
    reopened = load_single_neuron_study(path / "study.json")
    assert reopened.trace_dt_values_ms == (0.2, 0.1, 0.05)
    assert reopened.selection == study.selection
    for before, after in zip(study.comparison_traces, reopened.comparison_traces, strict=True):
        assert (before.method_id, before.dt_ms) == (after.method_id, after.dt_ms)
        np.testing.assert_array_equal(before.trace.time_ms, after.trace.time_ms)
        np.testing.assert_array_equal(before.trace.voltage_mv, after.trace.voltage_mv)
        np.testing.assert_array_equal(before.trace.spike_times_ms, after.trace.spike_times_ms)
        assert before.trace.elapsed_ns == after.trace.elapsed_ns


def test_reopen_summary_and_fi_results(tmp_path) -> None:
    config = SingleNeuronConfig(duration_ms=300.0, burn_in_ms=50.0)
    study = run_single_neuron_study(
        config,
        method_ids=("midpoint",), dt_values_ms=(0.2, 0.1),
        trace_dt_values_ms=(), currents=(0.0, 12.0), alpha_values=(),
        selection=StudySelection(
            traces=False, convergence=True, fi=True, alpha=False, summary=True
        ),
    )
    reopened = load_single_neuron_study(save_single_neuron_study(study, root=tmp_path))
    assert reopened.trace is None and reopened.comparison_traces == ()
    assert reopened.convergence.reset_mode == study.convergence.reset_mode
    assert reopened.reset_control.reset_mode == study.reset_control.reset_mode
    for before, after in zip(
        study.convergence.series[0].points,
        reopened.convergence.series[0].points,
        strict=True,
    ):
        np.testing.assert_allclose(
            [before.period_ms, before.period_error_ms, before.drift_ms_per_s],
            [after.period_ms, after.period_error_ms, after.drift_ms_per_s],
            equal_nan=True,
        )
    np.testing.assert_array_equal(reopened.fi_curves[0].rates_hz, study.fi_curves[0].rates_hz)


def test_reopen_old_single_dt_trace_with_original_total_duration(tmp_path) -> None:
    config = SingleNeuronConfig(duration_ms=60.0, burn_in_ms=20.0)
    study = run_single_neuron_study(
        config, method_ids=("midpoint",), dt_values_ms=(), currents=(), alpha_values=(),
        selection=StudySelection(True, False, False, False, False),
    )
    path = save_single_neuron_study(study, root=tmp_path)
    metadata_path = path / "study.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 4
    metadata["config"]["duration_ms"] = 80.0
    metadata["effective_protocols"]["trajectory"]["duration_ms"] = 80.0
    metadata.pop("comparison_trace_entries")
    metadata.pop("trace_metadata")
    metadata["axes"].pop("trajectory_dt_values_ms")
    with np.load(path / "trace.npz") as arrays:
        legacy = {key.replace("comparison__0000__", "comparison__midpoint__"): arrays[key]
                  for key in arrays.files}
    np.savez_compressed(path / "trace.npz", **legacy)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    reopened = load_single_neuron_study(path)
    assert reopened.config.duration_ms == pytest.approx(60.0)
    assert reopened.trace_dt_values_ms == (config.dt_ms,)
    np.testing.assert_array_equal(reopened.trace.time_ms, study.trace.time_ms)


def test_fi_repeats_and_variability_use_the_same_seed_sequence(tmp_path) -> None:
    config = SingleNeuronConfig(
        duration_ms=210.0,
        burn_in_ms=40.0,
        dt_ms=0.2,
        seed=31,
        noise_std=5.0,
        noise_interval_ms=3.0,
    )
    methods = ("explicit_euler", "midpoint")
    curves = run_fi_curves(config, method_ids=methods, currents=(12.0,), n_runs=3)
    points = run_trial_variability(config, method_ids=methods, n_runs=3)
    for curve in curves:
        trial_points = [point for point in points if point.method_id == curve.method_id]
        assert [point.seed for point in trial_points] == [31, 32, 33]
        np.testing.assert_allclose(
            curve.trial_rates_hz[:, 0],
            [point.firing_rate_hz for point in trial_points],
        )
        assert curve.valid_counts[0] == 3
        assert curve.rates_sd_hz[0] >= 0

    study = run_single_neuron_study(
        config,
        method_ids=methods,
        currents=(12.0,),
        dt_values_ms=(),
        alpha_values=(),
        n_runs=3,
        selection=StudySelection(
            traces=False,
            convergence=False,
            fi=True,
            alpha=False,
            summary=False,
            variability=True,
        ),
    )
    path = save_single_neuron_study(study, root=tmp_path)
    saved = json.loads((path / "study.json").read_text())
    assert saved["n_runs"] == 3
    assert saved["config"]["seed"] == 31
    assert len(saved["trial_points"]) == 6


def test_phase_only_and_mixed_noise_protocols(tmp_path) -> None:
    config = SingleNeuronConfig(duration_ms=100.0, burn_in_ms=20.0)
    phase = StudySelection(
        traces=False,
        convergence=False,
        fi=False,
        alpha=False,
        summary=False,
        phase=True,
    )
    study = run_single_neuron_study(config, method_ids=(), selection=phase)
    assert study.trace is not None
    assert study.comparison_traces == ()
    with np.load(save_single_neuron_study(study, root=tmp_path) / "trace.npz") as data:
        assert "voltage_mv" in data

    comparison = run_single_neuron_study(
        config,
        method_ids=("explicit_euler", "midpoint"),
        trace_dt_values_ms=(0.2, 0.1),
        selection=phase,
    )
    assert [(item.method_id, item.dt_ms) for item in comparison.comparison_traces] == [
        ("explicit_euler", 0.2), ("midpoint", 0.2),
        ("explicit_euler", 0.1), ("midpoint", 0.1),
    ]
    reopened = load_single_neuron_study(save_single_neuron_study(comparison, root=tmp_path))
    assert len(reopened.comparison_traces) == 4
    assert reopened.trace_dt_values_ms == (0.2, 0.1)

    mixed = run_single_neuron_study(
        replace(config, duration_ms=350.0, burn_in_ms=80.0, noise_std=2.0),
        method_ids=("midpoint",),
        dt_values_ms=(0.2, 0.1),
        currents=(12.0,),
        n_runs=2,
        selection=StudySelection(
            traces=False,
            convergence=True,
            fi=True,
            alpha=False,
            summary=False,
            variability=True,
        ),
    )
    deterministic = run_convergence_study(
        replace(config, duration_ms=350.0, burn_in_ms=80.0),
        method_ids=("midpoint",),
        dt_values_ms=(0.2, 0.1),
    )
    assert mixed.convergence.reference_period_ms == deterministic.reference_period_ms
    assert len(mixed.trial_points) == 2
    saved = json.loads((save_single_neuron_study(mixed, root=tmp_path) / "study.json").read_text())
    assert saved["order_protocol_noise_std"] == 0.0


def test_deterministic_repeats_are_permitted_and_have_zero_spread() -> None:
    study = run_single_neuron_study(
        SingleNeuronConfig(duration_ms=140.0, burn_in_ms=20.0),
        method_ids=("midpoint",),
        dt_values_ms=(),
        currents=(12.0,),
        n_runs=3,
        selection=StudySelection(
            traces=False,
            convergence=False,
            fi=True,
            alpha=False,
            summary=False,
            variability=True,
        ),
    )
    assert len(study.trial_points) == 3
    assert study.fi_curves[0].rates_sd_hz[0] == 0.0
    assert len({point.firing_rate_hz for point in study.trial_points}) == 1


def test_grid_reset_is_the_primary_protocol_when_selected(tmp_path) -> None:
    config = SingleNeuronConfig(
        duration_ms=500.0, burn_in_ms=100.0, reset_mode="grid", dt_ms=0.2
    )
    study = run_single_neuron_study(
        config,
        method_ids=("explicit_euler", "midpoint"),
        dt_values_ms=(0.2, 0.1),
        currents=(12.0,),
        alpha_values=(),
        selection=StudySelection(True, True, True, False, True),
    )
    assert study.convergence.reset_mode == "grid"
    assert study.reset_control.reset_mode == "event"
    direct = run_fi_curves(config, method_ids=("explicit_euler", "midpoint"), currents=(12.0,))
    for obtained, expected in zip(study.fi_curves, direct, strict=True):
        np.testing.assert_array_equal(obtained.rates_hz, expected.rates_hz)
    saved = json.loads((save_single_neuron_study(study, root=tmp_path) / "study.json").read_text())
    assert saved["effective_protocols"]["fi"]["reset_mode"] == "grid"
    assert saved["effective_protocols"]["convergence"]["reset_mode"] == "grid"
    assert saved["effective_protocols"]["reset_control"]["reset_mode"] == "event"


def test_phase_uses_its_own_composition_coefficient(tmp_path) -> None:
    phase_config = SingleNeuronConfig(
        method_id="euler_composition_complex",
        composition_alpha=0.5,
        duration_ms=130.0,
        burn_in_ms=20.0,
        dt_ms=0.4,
        reset_mode="event",
    )
    comparison_config = replace(phase_config, composition_alpha=0.9, reset_mode="grid")
    study = run_single_neuron_study(
        comparison_config,
        phase_config=phase_config,
        method_ids=("euler_composition_complex",),
        selection=StudySelection(True, False, False, False, False, phase=True),
    )
    np.testing.assert_array_equal(
        study.trace.voltage_mv, simulate_single_neuron(phase_config).voltage_mv
    )
    assert not np.array_equal(study.trace.voltage_mv, study.comparison_traces[0].trace.voltage_mv)
    saved = json.loads((save_single_neuron_study(study, root=tmp_path) / "study.json").read_text())
    assert saved["effective_protocols"]["trajectory"]["composition_alpha"] == 0.5
    assert saved["effective_protocols"]["trajectory"]["reset_mode"] == "event"
    assert saved["config"]["composition_alpha"] == 0.9
    assert saved["effective_protocols"]["comparison_traces"]["reset_mode"] == "grid"


def test_single_spike_has_finite_window_rate_and_unmatched_drift_is_invalid() -> None:
    from snnlab.experiments.single_neuron import SingleNeuronTrace, _drift_rate

    trace = SingleNeuronTrace(
        time_ms=np.asarray([0.0, 100.0]),
        voltage_mv=np.asarray([-65.0, -60.0]),
        recovery=np.asarray([-13.0, -12.0]),
        spike_times_ms=np.asarray([75.0]),
        elapsed_ns=100,
        simulated_steps=1,
    )
    metrics = trace_metrics(trace, burn_in_ms=50.0)
    assert metrics.firing_rate_hz == pytest.approx(20.0)
    assert np.isnan(metrics.mean_isi_ms)
    reference = replace(trace, spike_times_ms=np.asarray([20., 40., 60., 80., 95.]))
    unmatched = replace(trace, spike_times_ms=np.asarray([20., 40., 60., 80.]))
    assert np.isnan(_drift_rate(unmatched, reference, burn_in_ms=20.0))


def test_drift_ignores_spikes_before_observation_window() -> None:
    from snnlab.experiments.single_neuron import SingleNeuronTrace, _drift_rate

    base = SingleNeuronTrace(
        time_ms=np.asarray([0.0, 200.0]),
        voltage_mv=np.asarray([-65.0, -65.0]),
        recovery=np.asarray([-13.0, -13.0]),
        spike_times_ms=np.asarray([5.0, 12.0, 110.0, 130.0, 150.0, 170.0, 190.0]),
        elapsed_ns=0,
        simulated_steps=1,
    )
    observed = replace(
        base,
        spike_times_ms=np.asarray([2.0, 110.1, 130.2, 150.3, 170.4, 190.5]),
    )
    assert _drift_rate(observed, base, burn_in_ms=100.0) == pytest.approx(5.0)
    assert trace_metrics(observed, 100.0).spike_count == 5
