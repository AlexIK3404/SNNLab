"""
Generates a frozen regression fixture from the current clean DCI notebook code.

The tool extracts only the required class/function definitions from the exported
Python file, executes the exact notebook implementation, and stores deterministic
reference outputs in a compressed NPZ fixture.

Генерирует зафиксированный regression fixture из текущего clean DCI-кода блокнота.

Инструмент извлекает только необходимые определения классов/функций из
экспортированного Python-файла, выполняет точную реализацию блокнота и сохраняет
детерминированные эталонные результаты в сжатый NPZ fixture.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REFERENCE_DEFINITIONS = {
    "DCIConfig",
    "DCIDynamicsConfig",
    "DCIPresentationConfig",
    "DCISTDPConfig",
    "DCIHomeostasisConfig",
    "DCIHomeostasisState",
    "create_dci_homeostasis_state",
    "update_target_rate_homeostasis",
    "scaled_inh_exc_weight",
    "make_dci_dyn",
    "DCIConnectivity",
    "normalize_input_columns",
    "build_dci_connectivity",
    "clone_dci_connectivity",
    "NeuronPopulationState",
    "DCINetworkState",
    "create_population_state",
    "create_dci_state",
    "izhikevich_population_step_cpu",
    "apply_dci_stdp_from_logs",
    "DCISimulationResult",
    "simulate_dci_cpu",
    "prepare_flat_mnist_image",
    "encode_poisson_image_cpu",
    "DCIPresentationResult",
    "present_image_dci_cpu",
}


def _load_reference_module(source_path: Path) -> types.ModuleType:
    """
    Extracts the exact clean DCI definitions from an exported notebook script.

    Извлекает точные clean DCI-определения из экспортированного скрипта блокнота.
    """
    source = source_path.read_text(encoding="utf-8")

    # EN: Colab shell commands are invalid Python and are irrelevant to the
    #     deterministic DCI reference implementation.
    # RU: Shell-команды Colab не являются валидным Python и не относятся к
    #     детерминированной эталонной реализации DCI.
    sanitized = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("!"))
    tree = ast.parse(sanitized, filename=str(source_path))

    selected_nodes: list[ast.stmt] = []
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        if node.name not in REFERENCE_DEFINITIONS or node.name in seen:
            continue
        selected_nodes.append(node)
        seen.add(node.name)

    missing = REFERENCE_DEFINITIONS - seen
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(f"Reference definitions not found: {missing_text}")

    module_ast = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module_ast)

    module = types.ModuleType("snnlab_notebook_dci_reference")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        {
            "np": np,
            "math": math,
            "dataclass": dataclass,
        }
    )
    exec(
        compile(module_ast, filename=str(source_path), mode="exec"),
        module.__dict__,
    )
    return module


def _build_fixture(reference: types.ModuleType) -> dict[str, np.ndarray]:
    """
    Runs a small but non-trivial deterministic DCI sequence.

    Запускает небольшой, но нетривиальный детерминированный DCI-сценарий.
    """
    seed = 52
    data_seed = 999
    n_input = 784
    n_exc = 8
    n_samples = 6

    cfg = reference.DCIConfig(
        seed=seed,
        n_input=n_input,
        n_exc=n_exc,
        n_inh=n_exc,
        dt_ms=0.5,
        stimulus_ms=20.0,
        rest_ms=5.0,
        exc_a=0.02,
        exc_b=0.2,
        exc_c=-65.0,
        exc_d=8.0,
        inh_a=0.1,
        inh_b=0.2,
        inh_c=-65.0,
        inh_d=2.0,
        v_peak=30.0,
        e_exc=0.0,
        e_inh=-80.0,
        tau_g_exc_ms=5.0,
        tau_g_inh_ms=10.0,
        refractory_exc_ms=5.0,
        refractory_inh_ms=2.0,
        input_weight_sum=1.0,
    )
    dyn = reference.make_dci_dyn(
        cfg,
        target_total_inhibition=3.0,
        weight_exc_inh=0.3,
        input_gain=0.6,
    )
    presentation_cfg = reference.DCIPresentationConfig(
        base_max_rate_hz=63.75,
        rate_increment_hz=32.0,
        min_exc_spikes=2,
        max_attempts=3,
    )
    stdp_cfg = reference.DCISTDPConfig()
    homeo_cfg = reference.DCIHomeostasisConfig()

    data_rng = np.random.default_rng(data_seed)
    x = data_rng.random((12, n_input), dtype=np.float64)
    y = (np.arange(12, dtype=np.int64) % 3).astype(np.int64)

    order_rng = np.random.default_rng(seed)
    sample_indices = np.arange(len(x), dtype=np.int64)
    order_rng.shuffle(sample_indices)
    sample_indices = sample_indices[:n_samples]

    conn = reference.build_dci_connectivity(cfg)
    state = reference.create_dci_state(cfg, seed=seed)
    homeo_state = reference.create_dci_homeostasis_state(cfg)
    spike_rng = np.random.default_rng(seed + 1)

    initial_weights = conn.w_input_exc.copy()
    initial_exc_v = state.exc.v.copy()
    initial_inh_v = state.inh.v.copy()

    scalar_metric_keys = (
        "accepted",
        "attempts",
        "final_rate_hz",
        "exc_spikes",
        "inh_spikes",
        "active_exc",
        "active_inh",
        "max_sync_exc",
        "max_sync_inh",
        "rest_exc_spikes",
        "rest_inh_spikes",
        "mean_abs_delta",
        "max_abs_delta",
        "weight_min",
        "weight_max",
        "column_sum_min",
        "column_sum_max",
        "homeo_mean",
        "homeo_std",
        "homeo_min",
        "homeo_max",
    )
    metric_rows: list[list[float]] = []
    exc_counts_rows: list[np.ndarray] = []
    inh_counts_rows: list[np.ndarray] = []
    exc_logs: list[np.ndarray] = []
    inh_logs: list[np.ndarray] = []

    for sample_index in sample_indices:
        weights_before = conn.w_input_exc.copy()

        presentation = reference.present_image_dci_cpu(
            image=x[sample_index],
            state=state,
            conn=conn,
            cfg=cfg,
            dyn=dyn,
            presentation_cfg=presentation_cfg,
            rng=spike_rng,
            record_exc_indices=[],
            record_inh_indices=[],
            verbose=False,
            stdp_cfg=None,
            learning=False,
            homeo_cfg=homeo_cfg,
            homeo_state=homeo_state,
            adapt_homeostasis=False,
        )

        stimulus = presentation.stimulus_result
        rest = presentation.rest_result
        exc_counts = np.sum(stimulus.exc_spikes, axis=0).astype(np.int64)
        inh_counts = np.sum(stimulus.inh_spikes, axis=0).astype(np.int64)

        reference.apply_dci_stdp_from_logs(
            input_spikes=presentation.accepted_input_spikes,
            exc_spikes=stimulus.exc_spikes,
            conn=conn,
            cfg=cfg,
            stdp_cfg=stdp_cfg,
        )
        reference.update_target_rate_homeostasis(
            exc_counts=exc_counts,
            homeo_cfg=homeo_cfg,
            homeo_state=homeo_state,
        )
        reference.normalize_input_columns(
            conn.w_input_exc,
            target_sum=cfg.input_weight_sum,
        )
        np.clip(
            conn.w_input_exc,
            stdp_cfg.w_min,
            stdp_cfg.w_max,
            out=conn.w_input_exc,
        )

        exc_per_step = np.sum(stimulus.exc_spikes, axis=1)
        inh_per_step = np.sum(stimulus.inh_spikes, axis=1)
        weight_delta = np.abs(conn.w_input_exc - weights_before)
        column_sums = np.sum(conn.w_input_exc, axis=0)
        homeo = homeo_state.exc_current

        metrics: dict[str, float | int | bool] = {
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
            "weight_min": float(np.min(conn.w_input_exc)),
            "weight_max": float(np.max(conn.w_input_exc)),
            "column_sum_min": float(np.min(column_sums)),
            "column_sum_max": float(np.max(column_sums)),
            "homeo_mean": float(np.mean(homeo)),
            "homeo_std": float(np.std(homeo)),
            "homeo_min": float(np.min(homeo)),
            "homeo_max": float(np.max(homeo)),
        }
        metric_rows.append([float(metrics[key]) for key in scalar_metric_keys])
        exc_counts_rows.append(exc_counts.copy())
        inh_counts_rows.append(inh_counts.copy())
        exc_logs.append(stimulus.exc_spikes.copy())
        inh_logs.append(stimulus.inh_spikes.copy())

    config = {
        "fixture_version": 1,
        "source_protocol": "clean_target_rate_final_attempt_stdp",
        "seed": seed,
        "data_seed": data_seed,
        "n_input": n_input,
        "n_exc": n_exc,
        "n_inh": n_exc,
        "dt_ms": 0.5,
        "stimulus_ms": 20.0,
        "rest_ms": 5.0,
        "target_total_inhibition": 3.0,
        "weight_exc_inh": 0.3,
        "input_gain": 0.6,
        "presentation": {
            "base_max_rate_hz": 63.75,
            "rate_increment_hz": 32.0,
            "min_exc_spikes": 2,
            "max_attempts": 3,
        },
        "n_samples": n_samples,
        "metric_keys": scalar_metric_keys,
    }

    return {
        "config_json": np.asarray(json.dumps(config, ensure_ascii=False)),
        "x": x,
        "y": y,
        "sample_indices": sample_indices,
        "initial_weights": initial_weights,
        "initial_exc_v": initial_exc_v,
        "initial_inh_v": initial_inh_v,
        "metric_matrix": np.asarray(metric_rows, dtype=np.float64),
        "exc_counts": np.asarray(exc_counts_rows, dtype=np.int64),
        "inh_counts": np.asarray(inh_counts_rows, dtype=np.int64),
        "exc_spike_logs": np.asarray(exc_logs, dtype=bool),
        "inh_spike_logs": np.asarray(inh_logs, dtype=bool),
        "final_weights": conn.w_input_exc.copy(),
        "final_exc_v": state.exc.v.copy(),
        "final_exc_u": state.exc.u.copy(),
        "final_exc_g_exc": state.exc.g_exc.copy(),
        "final_exc_g_inh": state.exc.g_inh.copy(),
        "final_exc_refractory": state.exc.refractory_left_ms.copy(),
        "final_inh_v": state.inh.v.copy(),
        "final_inh_u": state.inh.u.copy(),
        "final_inh_g_exc": state.inh.g_exc.copy(),
        "final_inh_g_inh": state.inh.g_inh.copy(),
        "final_inh_refractory": state.inh.refractory_left_ms.copy(),
        "final_last_exc_spikes": state.last_exc_spikes.copy(),
        "final_last_inh_spikes": state.last_inh_spikes.copy(),
        "final_homeostasis": homeo_state.exc_current.copy(),
        "final_rng_state_json": np.asarray(json.dumps(spike_rng.bit_generator.state)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _load_reference_module(args.source)
    fixture = _build_fixture(reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **fixture)
    print(f"Fixture written: {args.output}")


if __name__ == "__main__":
    main()
