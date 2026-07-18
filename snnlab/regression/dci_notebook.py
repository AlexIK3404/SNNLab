from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from snnlab.architectures.dci import build_dci_model
from snnlab.configs.dci import (
    DCIConfig,
    DCIHomeostasisConfig,
    DCIPresentationConfig,
    DCISTDPConfig,
    make_dci_dynamics,
)
from snnlab.experiments.base import create_sample_schedule


@dataclass(frozen=True, slots=True)
class RegressionCheck:
    """
    Stores one regression assertion and its diagnostic details.

    Хранит одну regression-проверку и её диагностические детали.
    """

    name: str
    passed: bool
    details: str


@dataclass(frozen=True, slots=True)
class DCIRegressionReport:
    """
    Summarizes DCI equivalence against the frozen notebook reference.

    Суммирует эквивалентность DCI относительно зафиксированного эталона блокнота.
    """

    checks: tuple[RegressionCheck, ...]

    @property
    def passed(self) -> bool:
        """Returns True only when every regression check passes.

        Возвращает True только если прошли все regression-проверки.
        """
        return all(check.passed for check in self.checks)


def _default_fixture_path() -> Path:
    """
    Resolves the packaged notebook-reference fixture.

    Возвращает путь к упакованному notebook-reference fixture.
    """
    return Path(files("snnlab.regression.fixtures").joinpath("dci_notebook_clean_v1.npz"))


def _record_exact(
    checks: list[RegressionCheck],
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    """
    Records an exact array comparison.

    Записывает точное сравнение массивов.
    """
    passed = np.array_equal(actual, expected)
    details = (
        "exact match"
        if passed
        else (
            f"shape actual={actual.shape}, expected={expected.shape}; "
            f"mismatched={int(np.count_nonzero(actual != expected)) if actual.shape == expected.shape else 'n/a'}"
        )
    )
    checks.append(RegressionCheck(name=name, passed=passed, details=details))


def _record_close(
    checks: list[RegressionCheck],
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
) -> None:
    """
    Records a zero-relative-tolerance floating-point comparison.

    Записывает сравнение floating-point массивов с нулевой относительной погрешностью.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        checks.append(
            RegressionCheck(
                name=name,
                passed=False,
                details=f"shape actual={actual.shape}, expected={expected.shape}",
            )
        )
        return

    difference = np.abs(actual - expected)
    max_abs_diff = float(np.max(difference, initial=0.0))
    exact_mismatches = int(np.count_nonzero(actual != expected))
    passed = bool(np.allclose(actual, expected, rtol=0.0, atol=atol, equal_nan=True))
    checks.append(
        RegressionCheck(
            name=name,
            passed=passed,
            details=(
                f"exact_mismatches={exact_mismatches}; "
                f"max_abs_diff={max_abs_diff:.3e}; atol={atol:.1e}"
            ),
        )
    )


def run_dci_notebook_regression(
    fixture_path: str | Path | None = None,
) -> DCIRegressionReport:
    """
    Compares Stage-1 DCI with a fixture generated from the clean notebook code.

    The fixture was produced by executing the current notebook implementation
    with fixed initial conditions, sample order, and RNG streams. Discrete spike
    events, counts, weights, homeostasis, and continuous state are compared.

    Сравнивает Stage-1 DCI с fixture, сгенерированным из clean-кода блокнота.

    Fixture получен выполнением текущей реализации блокнота при фиксированных
    начальных условиях, порядке sample и RNG-потоках. Сравниваются дискретные
    спайковые события, counts, веса, гомеостаз и непрерывное состояние.
    """
    path = Path(fixture_path) if fixture_path is not None else _default_fixture_path()
    checks: list[RegressionCheck] = []

    with np.load(path, allow_pickle=False) as fixture:
        config = json.loads(str(fixture["config_json"].item()))
        x = fixture["x"]
        expected_schedule = fixture["sample_indices"]

        cfg = DCIConfig(
            seed=int(config["seed"]),
            n_input=int(config["n_input"]),
            n_exc=int(config["n_exc"]),
            n_inh=int(config["n_inh"]),
            dt_ms=float(config["dt_ms"]),
            stimulus_ms=float(config["stimulus_ms"]),
            rest_ms=float(config["rest_ms"]),
            exc_numerical_method="explicit_euler",
            inh_numerical_method="explicit_euler",
        )
        dynamics = make_dci_dynamics(
            cfg,
            target_total_inhibition=float(config["target_total_inhibition"]),
            weight_exc_inh=float(config["weight_exc_inh"]),
            input_gain=float(config["input_gain"]),
        )
        p_cfg_raw = config["presentation"]
        presentation_cfg = DCIPresentationConfig(
            base_max_rate_hz=float(p_cfg_raw["base_max_rate_hz"]),
            rate_increment_hz=float(p_cfg_raw["rate_increment_hz"]),
            min_exc_spikes=int(p_cfg_raw["min_exc_spikes"]),
            max_attempts=int(p_cfg_raw["max_attempts"]),
        )
        stdp_cfg = DCISTDPConfig()
        homeostasis_cfg = DCIHomeostasisConfig()
        model = build_dci_model(
            cfg=cfg,
            dynamics=dynamics,
            presentation_cfg=presentation_cfg,
            stdp_cfg=stdp_cfg,
            homeostasis_cfg=homeostasis_cfg,
            seed=cfg.seed,
        )

        schedule = create_sample_schedule(
            n_items=len(x),
            n_samples=int(config["n_samples"]),
            seed=cfg.seed,
            shuffle=True,
            allow_repeats=False,
        )
        _record_exact(checks, "sample_schedule", schedule, expected_schedule)
        _record_exact(
            checks,
            "initial_weights",
            model.connectivity.w_input_exc,
            fixture["initial_weights"],
        )
        _record_exact(checks, "initial_exc_v", model.network_state.exc.v, fixture["initial_exc_v"])
        _record_exact(checks, "initial_inh_v", model.network_state.inh.v, fixture["initial_inh_v"])

        rng = np.random.default_rng(cfg.seed + 1)
        metric_keys = tuple(config["metric_keys"])
        metric_rows: list[list[float]] = []
        exc_counts_rows: list[np.ndarray] = []
        inh_counts_rows: list[np.ndarray] = []
        exc_logs: list[np.ndarray] = []
        inh_logs: list[np.ndarray] = []

        for sample_index in schedule:
            metrics = model.train_one_sample(
                x[int(sample_index)],
                rng=rng,
                normalize_after_each=True,
                emit_spike_logs=True,
            )
            exc_counts_rows.append(np.asarray(metrics["exc_counts"], dtype=np.int64))
            inh_counts_rows.append(np.asarray(metrics["inh_counts"], dtype=np.int64))
            exc_logs.append(np.asarray(metrics["exc_spike_log"], dtype=bool))
            inh_logs.append(np.asarray(metrics["inh_spike_log"], dtype=bool))
            metric_rows.append([float(metrics[key]) for key in metric_keys])

        _record_close(
            checks,
            "sample_metrics",
            np.asarray(metric_rows, dtype=np.float64),
            fixture["metric_matrix"],
            atol=1e-14,
        )
        _record_exact(checks, "exc_counts", np.asarray(exc_counts_rows), fixture["exc_counts"])
        _record_exact(checks, "inh_counts", np.asarray(inh_counts_rows), fixture["inh_counts"])
        _record_exact(checks, "exc_spike_logs", np.asarray(exc_logs), fixture["exc_spike_logs"])
        _record_exact(checks, "inh_spike_logs", np.asarray(inh_logs), fixture["inh_spike_logs"])

        # EN: Final weights are continuous floating-point results. Bitwise equality
        #     across OS / Python / NumPy builds is not guaranteed because STDP uses
        #     fractional powers and column-wise floating-point normalization.
        #     Keep a strict absolute tolerance and zero relative tolerance.
        # RU: Итоговые веса — непрерывный floating-point результат. Битовое
        #     совпадение между разными ОС / сборками Python / NumPy не гарантируется,
        #     потому что STDP использует дробную степень и нормализацию по столбцам.
        #     Сохраняем строгий абсолютный допуск и нулевой относительный допуск.
        _record_close(
            checks,
            "final_weights",
            model.connectivity.w_input_exc,
            fixture["final_weights"],
            atol=1e-15,
        )
        _record_close(
            checks, "final_exc_v", model.network_state.exc.v, fixture["final_exc_v"], atol=1e-12
        )
        _record_close(
            checks, "final_exc_u", model.network_state.exc.u, fixture["final_exc_u"], atol=1e-12
        )
        _record_exact(
            checks, "final_exc_g_exc", model.network_state.exc.g_exc, fixture["final_exc_g_exc"]
        )
        _record_exact(
            checks, "final_exc_g_inh", model.network_state.exc.g_inh, fixture["final_exc_g_inh"]
        )
        _record_exact(
            checks,
            "final_exc_refractory",
            model.network_state.exc.refractory_left_ms,
            fixture["final_exc_refractory"],
        )
        _record_close(
            checks, "final_inh_v", model.network_state.inh.v, fixture["final_inh_v"], atol=1e-12
        )
        _record_close(
            checks, "final_inh_u", model.network_state.inh.u, fixture["final_inh_u"], atol=1e-12
        )
        _record_exact(
            checks, "final_inh_g_exc", model.network_state.inh.g_exc, fixture["final_inh_g_exc"]
        )
        _record_exact(
            checks, "final_inh_g_inh", model.network_state.inh.g_inh, fixture["final_inh_g_inh"]
        )
        _record_exact(
            checks,
            "final_inh_refractory",
            model.network_state.inh.refractory_left_ms,
            fixture["final_inh_refractory"],
        )
        _record_exact(
            checks,
            "final_last_exc_spikes",
            model.network_state.last_exc_spikes,
            fixture["final_last_exc_spikes"],
        )
        _record_exact(
            checks,
            "final_last_inh_spikes",
            model.network_state.last_inh_spikes,
            fixture["final_last_inh_spikes"],
        )
        _record_exact(
            checks,
            "final_homeostasis",
            model.homeostasis_state.exc_current,
            fixture["final_homeostasis"],
        )

        expected_rng = json.loads(str(fixture["final_rng_state_json"].item()))
        actual_rng = rng.bit_generator.state
        rng_passed = json.dumps(actual_rng, sort_keys=True) == json.dumps(
            expected_rng, sort_keys=True
        )
        checks.append(
            RegressionCheck(
                name="final_rng_state",
                passed=rng_passed,
                details="exact JSON state match" if rng_passed else "RNG state differs",
            )
        )

    return DCIRegressionReport(checks=tuple(checks))
