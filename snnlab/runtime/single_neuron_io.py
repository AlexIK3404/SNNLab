from __future__ import annotations

import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from snnlab.experiments.single_neuron import SingleNeuronStudy
from snnlab.runtime.experiment_io import collect_environment, save_json


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
    save_json(
        target / "study.json",
        {
            "format_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": asdict(study.config),
            "metrics": asdict(study.metrics),
            "convergence": asdict(study.convergence),
            "grid_convergence": asdict(study.grid_convergence),
            "fi_curves": [asdict(curve) for curve in study.fi_curves],
            "alpha_sweep": [asdict(point) for point in study.alpha_sweep],
        },
    )
    save_json(target / "environment.json", collect_environment())

    trace_path = target / "trace.npz"
    temporary = target / "trace.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            time_ms=study.trace.time_ms,
            voltage_mv=study.trace.voltage_mv,
            recovery=study.trace.recovery,
            spike_times_ms=study.trace.spike_times_ms,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, trace_path)
    return target
