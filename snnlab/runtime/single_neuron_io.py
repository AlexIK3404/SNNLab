from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

from snnlab.experiments.single_neuron import (
    AlphaSweepPoint,
    ConvergencePoint,
    ConvergenceSeries,
    ConvergenceStudy,
    FICurve,
    MethodTrace,
    NeuronParameters,
    SingleNeuronConfig,
    SingleNeuronStudy,
    SingleNeuronTrace,
    StudySelection,
    TraceMetrics,
    TrialPoint,
)
from snnlab.runtime.experiment_io import collect_environment, save_json


def _source_fingerprints() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    files = (
        "core/numerical_methods.py",
        "experiments/single_neuron.py",
        "runtime/single_neuron_io.py",
        "gui/widgets/single_neuron_view.py",
    )
    return {name: sha256((package / name).read_bytes()).hexdigest() for name in files}


def _unique_study_directory(root: str | Path) -> Path:
    base = Path(root)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def save_single_neuron_study(
    study: SingleNeuronStudy,
    *,
    root: str | Path = "runs/gui/single_neuron",
) -> Path:
    """Persist one study as human-readable metadata plus a compact trace archive."""

    target = _unique_study_directory(root)
    effective_protocols = {
        "trajectory": asdict(study.trace_config) if study.trace_config is not None else None,
        "comparison_traces": {
            "reset_mode": study.config.reset_mode,
            "noise_std": study.config.noise_std,
        }
        if study.comparison_traces
        else None,
        "convergence": {
            "reset_mode": study.convergence.reset_mode,
            "noise_std": 0.0,
            "reference_method": "midpoint",
            "reference_reset_mode": "event",
            "reference_dt_ms": study.convergence.reference_dt_ms,
            "reference_uncertainty_ms": study.convergence.reference_uncertainty_ms,
        }
        if study.convergence is not None
        else None,
        "reset_control": {
            "reset_mode": study.reset_control.reset_mode,
            "noise_std": 0.0,
        }
        if study.reset_control is not None
        else None,
        "fi": {
            "reset_mode": study.config.reset_mode,
            "noise_std": study.config.noise_std,
            "seeds": list(range(study.config.seed, study.config.seed + study.n_runs)),
            "rate_definition": "spike_count_per_observation_window",
        }
        if study.selection.fi
        else None,
        "alpha": {
            "reset_mode": study.config.reset_mode,
            "noise_std": 0.0,
            "reference_reset_mode": "event",
        }
        if study.selection.alpha
        else None,
        "variability": {
            "reset_mode": study.config.reset_mode,
            "noise_std": study.config.noise_std,
            "seeds": list(range(study.config.seed, study.config.seed + study.n_runs)),
        }
        if study.selection.variability
        else None,
    }
    save_json(
        target / "study.json",
        {
            "format_version": 6,
            "time_protocol": {
                "duration_ms": "observation_window_after_burn_in",
                "total_simulation_ms": study.config.burn_in_ms + study.config.duration_ms,
                "trace_arrays": "absolute_simulation_time_including_transient",
                "plots": "observation_window_time_rebased_to_zero",
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": asdict(study.config),
            "selection": asdict(study.selection),
            "effective_protocols": effective_protocols,
            "axes": {
                "dt_values_ms": study.dt_values_ms,
                "trajectory_dt_values_ms": study.trace_dt_values_ms,
                "currents": study.currents,
                "alpha_values": study.alpha_values,
            },
            "source_fingerprints_sha256": _source_fingerprints(),
            "n_runs": study.n_runs,
            "order_protocol_noise_std": 0.0
            if study.convergence is not None or study.alpha_sweep
            else None,
            "metrics": asdict(study.metrics) if study.metrics is not None else None,
            "convergence": asdict(study.convergence) if study.convergence is not None else None,
            "reset_control": asdict(study.reset_control)
            if study.reset_control is not None
            else None,
            "fi_curves": [asdict(curve) for curve in study.fi_curves],
            "alpha_sweep": [asdict(point) for point in study.alpha_sweep],
            "trial_points": [asdict(point) for point in study.trial_points],
            "comparison_trace_methods": [
                method_trace.method_id for method_trace in study.comparison_traces
            ],
            "trace_metadata": _trace_metadata(study.trace) if study.trace is not None else None,
            "comparison_trace_entries": [
                {
                    "method_id": item.method_id,
                    "dt_ms": item.dt_ms if item.dt_ms is not None else study.config.dt_ms,
                    "prefix": f"comparison__{index:04d}__",
                    **_trace_metadata(item.trace),
                }
                for index, item in enumerate(study.comparison_traces)
            ],
        },
    )
    save_json(target / "environment.json", collect_environment())

    trace_path = target / "trace.npz"
    temporary = target / "trace.npz.tmp"
    with temporary.open("wb") as stream:
        arrays = {}
        if study.trace is not None:
            arrays.update(
                {
                    "time_ms": study.trace.time_ms,
                    "voltage_mv": study.trace.voltage_mv,
                    "recovery": study.trace.recovery,
                    "spike_times_ms": study.trace.spike_times_ms,
                }
            )
        for index, method_trace in enumerate(study.comparison_traces):
            prefix = f"comparison__{index:04d}__"
            arrays[f"{prefix}time_ms"] = method_trace.trace.time_ms
            arrays[f"{prefix}voltage_mv"] = method_trace.trace.voltage_mv
            arrays[f"{prefix}recovery"] = method_trace.trace.recovery
            arrays[f"{prefix}spike_times_ms"] = method_trace.trace.spike_times_ms
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, trace_path)
    return target


def _trace_metadata(trace: SingleNeuronTrace) -> dict[str, object]:
    return {
        "elapsed_ns": trace.elapsed_ns,
        "simulated_steps": trace.simulated_steps,
        "diverged": trace.diverged,
        "divergence_time_ms": trace.divergence_time_ms,
        "implicit_step_attempts": trace.implicit_step_attempts,
    }


def _load_config(data: dict) -> SingleNeuronConfig:
    values = dict(data)
    values["neuron"] = NeuronParameters(**values["neuron"])
    return SingleNeuronConfig(**values)


def _load_convergence(data: dict | None) -> ConvergenceStudy | None:
    if data is None:
        return None
    series = tuple(
        ConvergenceSeries(
            method_id=item["method_id"],
            observed_period_order=item["observed_period_order"],
            local_order=item["local_order"],
            points=tuple(ConvergencePoint(**point) for point in item["points"]),
        )
        for item in data["series"]
    )
    return ConvergenceStudy(
        reference_period_ms=data["reference_period_ms"],
        reference_dt_ms=data["reference_dt_ms"],
        reset_mode=data["reset_mode"],
        series=series,
        reference_uncertainty_ms=data.get("reference_uncertainty_ms", float("nan")),
        reference_spike_count=data.get("reference_spike_count", 0),
    )


def _load_trace(arrays: np.lib.npyio.NpzFile, prefix: str, metadata: dict | None) -> SingleNeuronTrace:
    fields = ("time_ms", "voltage_mv", "recovery", "spike_times_ms")
    values = [np.asarray(arrays[prefix + field], dtype=np.float64) for field in fields]
    if any(value.ndim != 1 for value in values):
        raise ValueError("Trace arrays must be one-dimensional")
    if not values[0].size or any(value.size != values[0].size for value in values[1:3]):
        raise ValueError("Trace arrays have inconsistent lengths")
    metadata = metadata or {}
    elapsed_ns = metadata.get("elapsed_ns")
    if elapsed_ns is None:
        elapsed_ns = round(float(metadata.get("elapsed_ms", 0.0)) * 1_000_000)
    return SingleNeuronTrace(
        time_ms=values[0],
        voltage_mv=values[1],
        recovery=values[2],
        spike_times_ms=values[3],
        elapsed_ns=int(elapsed_ns),
        simulated_steps=int(metadata.get("simulated_steps", max(1, values[0].size - 1))),
        diverged=bool(metadata.get("diverged", False)),
        divergence_time_ms=metadata.get("divergence_time_ms"),
        implicit_step_attempts=int(metadata.get("implicit_step_attempts", 0)),
    )


def load_single_neuron_study(directory: str | Path) -> SingleNeuronStudy:
    """Open a saved study without running simulations; support old single-dt studies."""
    root = Path(directory)
    if root.is_file() and root.name == "study.json":
        root = root.parent
    with (root / "study.json").open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    version = data.get("format_version")
    if version not in (4, 5, 6):
        raise ValueError(f"Unsupported single-neuron study format: {version!r}")
    def compatible_config(values: dict) -> SingleNeuronConfig:
        values = dict(values)
        if version == 4:
            values["duration_ms"] = values["duration_ms"] - values["burn_in_ms"]
        return _load_config(values)

    config = compatible_config(data["config"])
    selection = StudySelection(**data["selection"])
    protocol = data.get("effective_protocols", {}).get("trajectory")
    trace_config = compatible_config(protocol) if protocol else None
    axes = data.get("axes", {})
    with np.load(root / "trace.npz", allow_pickle=False) as arrays:
        trace = (
            _load_trace(arrays, "", data.get("trace_metadata") or data.get("metrics"))
            if "time_ms" in arrays
            else None
        )
        comparison_traces = []
        if version == 6:
            entries = data.get("comparison_trace_entries", [])
        else:
            entries = [
                {"method_id": method_id, "dt_ms": config.dt_ms,
                 "prefix": f"comparison__{method_id}__"}
                for method_id in data.get("comparison_trace_methods", [])
            ]
        for entry in entries:
            comparison_traces.append(
                MethodTrace(
                    method_id=entry["method_id"],
                    trace=_load_trace(arrays, entry["prefix"], entry),
                    dt_ms=float(entry["dt_ms"]),
                )
            )
    if trace is not None and trace_config is None:
        trace_config = config
    curves = tuple(
        FICurve(
            method_id=item["method_id"],
            currents=np.asarray(item["currents"], dtype=np.float64),
            rates_hz=np.asarray(item["rates_hz"], dtype=np.float64),
            diverged=np.asarray(item["diverged"], dtype=np.bool_),
            rates_sd_hz=np.asarray(item["rates_sd_hz"], dtype=np.float64)
            if item.get("rates_sd_hz") is not None else None,
            valid_counts=np.asarray(item["valid_counts"], dtype=np.int64)
            if item.get("valid_counts") is not None else None,
            trial_rates_hz=np.asarray(item["trial_rates_hz"], dtype=np.float64)
            if item.get("trial_rates_hz") is not None else None,
        )
        for item in data.get("fi_curves", [])
    )
    if selection.phase and trace is None:
        raise ValueError("Saved phase portrait is missing its trace arrays")
    if selection.traces and trace is None and not comparison_traces:
        raise ValueError("Saved trajectories are missing their trace arrays")
    if (selection.convergence or selection.summary) and data.get("convergence") is None:
        raise ValueError("Saved convergence results are missing")
    if selection.summary and data.get("reset_control") is None:
        raise ValueError("Saved reset-control results are missing")
    if selection.fi and not curves:
        raise ValueError("Saved f-I curves are missing")
    if selection.alpha and not data.get("alpha_sweep"):
        raise ValueError("Saved alpha sweep is missing")
    if selection.variability and not data.get("trial_points"):
        raise ValueError("Saved trial results are missing")
    return SingleNeuronStudy(
        config=config,
        trace=trace,
        metrics=TraceMetrics(**data["metrics"]) if data.get("metrics") is not None else None,
        convergence=_load_convergence(data.get("convergence")),
        reset_control=_load_convergence(data.get("reset_control")),
        fi_curves=curves,
        alpha_sweep=tuple(AlphaSweepPoint(**item) for item in data.get("alpha_sweep", [])),
        comparison_traces=tuple(comparison_traces),
        selection=selection,
        trial_points=tuple(TrialPoint(**item) for item in data.get("trial_points", [])),
        n_runs=int(data.get("n_runs", 1)),
        trace_config=trace_config,
        dt_values_ms=tuple(axes.get("dt_values_ms", ())),
        trace_dt_values_ms=tuple(axes.get("trajectory_dt_values_ms", ()))
        or ((config.dt_ms,) if comparison_traces else ()),
        currents=tuple(axes.get("currents", ())),
        alpha_values=tuple(axes.get("alpha_values", ())),
    )
