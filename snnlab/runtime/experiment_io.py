from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from snnlab import __version__

EXPERIMENT_CONFIG_FORMAT_VERSION = 1
DATA_PROTOCOL_FORMAT_VERSION = 1


def _jsonable(value: Any) -> Any:
    """Converts nested runtime values into stable JSON/YAML data.

    Преобразует вложенные runtime-значения в стабильные JSON/YAML-данные.
    """
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_text(path: Path, text: str) -> None:
    """Writes text atomically to avoid half-written release artifacts.

    Атомарно записывает текст, чтобы не оставлять повреждённые артефакты.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    """Saves a human-readable UTF-8 YAML document atomically.

    Сохраняет читаемый UTF-8 YAML-документ атомарно.
    """
    target = Path(path)
    text = yaml.safe_dump(
        _jsonable(payload),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    _atomic_text(target, text)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Loads and validates a mapping from YAML.

    Загружает YAML и проверяет, что корневой объект является словарём.
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration root must be a mapping")
    return payload


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Saves an indented UTF-8 JSON document atomically.

    Сохраняет форматированный UTF-8 JSON-документ атомарно.
    """
    target = Path(path)
    text = json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n"
    _atomic_text(target, text)
    return target


def schedule_sha256(indices: np.ndarray | list[int]) -> str:
    """Returns a deterministic hash of the exact sample schedule.

    Возвращает детерминированный hash точного расписания sample.
    """
    array = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_user_configuration(
    *,
    architecture: str,
    task: str,
    backend: str,
    locale: str,
    parameters: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Builds the portable GUI configuration document.

    Создаёт переносимый документ конфигурации GUI.
    """
    return {
        "format_version": EXPERIMENT_CONFIG_FORMAT_VERSION,
        "snnlab_version": __version__,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "architecture": str(architecture),
        "task": str(task),
        "backend": str(backend),
        "locale": str(locale),
        "parameters": _jsonable(parameters),
        "evaluation": _jsonable(evaluation),
    }


def validate_user_configuration(payload: dict[str, Any]) -> dict[str, Any]:
    """Validates the public configuration schema before applying it.

    Проверяет публичную схему конфигурации перед применением.
    """
    version = payload.get("format_version")
    if version != EXPERIMENT_CONFIG_FORMAT_VERSION:
        raise ValueError(f"Unsupported experiment configuration version: {version!r}")
    architecture = payload.get("architecture")
    if architecture not in {"dci", "reservoir"}:
        raise ValueError(f"Unsupported architecture: {architecture!r}")
    if not isinstance(payload.get("parameters"), dict):
        raise ValueError("Configuration field 'parameters' must be a mapping")
    if not isinstance(payload.get("evaluation", {}), dict):
        raise ValueError("Configuration field 'evaluation' must be a mapping")
    return payload


def _git_commit() -> str | None:
    """Returns the current Git commit when the source is inside a repository.

    Возвращает текущий Git commit, если исходники находятся в репозитории.
    """
    try:
        source_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def collect_environment() -> dict[str, Any]:
    """Collects reproducibility metadata without importing heavy packages.

    Собирает metadata воспроизводимости без импорта тяжёлых библиотек.
    """
    packages = {}
    for package_name in (
        "numpy",
        "scikit-learn",
        "PyYAML",
        "PySide6",
        "pyqtgraph",
        "tensorflow",
    ):
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None

    return {
        "snnlab_version": __version__,
        "git_commit": _git_commit(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
    }


def build_data_protocol(
    *,
    kind: str,
    data_spec: dict[str, Any] | None,
    schedule: np.ndarray | list[int],
    position: int = 0,
) -> dict[str, Any]:
    """Builds an explicit, reproducible training-data protocol.

    Создаёт явный воспроизводимый протокол обучающих данных.
    """
    schedule_array = np.asarray(schedule, dtype=np.int64)
    unique_indices = np.unique(schedule_array)
    spec = dict(data_spec or {})
    return {
        "format_version": DATA_PROTOCOL_FORMAT_VERSION,
        "architecture": str(kind),
        "dataset": spec.get("dataset"),
        "train_pool_size": int(spec.get("train_pool_size", spec.get("train_limit", 0)) or 0),
        "test_pool_size": int(spec.get("test_pool_size", spec.get("test_limit", 0)) or 0),
        "subset_seed": int(spec.get("subset_seed", 0) or 0),
        "samples_per_epoch": int(spec.get("samples_per_epoch", 0) or 0),
        "epochs": int(spec.get("epochs", 0) or 0),
        "total_presentations": int(schedule_array.size),
        "unique_training_samples": int(unique_indices.size),
        "repeated_presentations": int(schedule_array.size - unique_indices.size),
        "position": int(position),
        "schedule_sha256": schedule_sha256(schedule_array),
        # EN: Full indices are intentionally persisted for exact reproducibility.
        # RU: Полные индексы намеренно сохраняются для точной воспроизводимости.
        "training_schedule_indices": schedule_array.tolist(),
        "unique_training_indices": unique_indices.tolist(),
        "evaluation": {
            "assignment_policy": "full_train_pool",
            "assignment_indices": None,
            "test_indices": None,
            "training_assignment_overlap": None,
        },
    }


def write_run_manifest(
    *,
    run_dir: str | Path,
    user_configuration: dict[str, Any],
    data_protocol: dict[str, Any],
) -> None:
    """Writes the standard release-ready run artifacts.

    Записывает стандартные артефакты запуска для публичного релиза.
    """
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    save_yaml(target / "experiment.yaml", user_configuration)
    save_json(target / "environment.json", collect_environment())
    save_json(target / "data_protocol.json", data_protocol)
    (target / "logs").mkdir(parents=True, exist_ok=True)
    (target / "models").mkdir(parents=True, exist_ok=True)


def update_data_protocol_evaluation(
    *,
    run_dir: str | Path,
    assignment_policy: str,
    assignment_indices: np.ndarray,
    test_indices: np.ndarray,
    training_indices: np.ndarray,
) -> Path:
    """Updates protocol metadata after an evaluation response set is collected.

    Обновляет metadata протокола после сбора evaluation-откликов.
    """
    path = Path(run_dir) / "data_protocol.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = {"format_version": DATA_PROTOCOL_FORMAT_VERSION, "evaluation": {}}

    assignment = np.asarray(assignment_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    training = np.asarray(training_indices, dtype=np.int64)
    overlap = np.intersect1d(np.unique(training), np.unique(assignment), assume_unique=False)
    payload["evaluation"] = {
        "assignment_policy": str(assignment_policy),
        "assignment_indices": assignment.tolist(),
        "test_indices": test.tolist(),
        "training_assignment_overlap": int(overlap.size),
        "assignment_sha256": schedule_sha256(assignment),
        "test_sha256": schedule_sha256(test),
    }
    return save_json(path, payload)


def update_data_protocol_reservoir_evaluation(
    *,
    run_dir: str | Path,
    test_indices: np.ndarray,
) -> Path:
    """Stores the deterministic reservoir test subset in the data protocol.

    Сохраняет детерминированную test-подвыборку reservoir в протоколе данных.
    """
    path = Path(run_dir) / "data_protocol.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = {"format_version": DATA_PROTOCOL_FORMAT_VERSION}

    test = np.asarray(test_indices, dtype=np.int64)
    payload["evaluation"] = {
        "assignment_policy": None,
        "assignment_indices": None,
        "test_indices": test.tolist(),
        "training_assignment_overlap": None,
        "test_sha256": schedule_sha256(test),
    }
    return save_json(path, payload)


def update_data_protocol_training(
    *,
    run_dir: str | Path,
    kind: str,
    data_spec: dict[str, Any] | None,
    schedule: np.ndarray | list[int],
    position: int,
) -> Path:
    """Refreshes schedule and progress metadata after checkpoint/extension.

    Обновляет расписание и прогресс после checkpoint или дообучения.
    """
    path = Path(run_dir) / "data_protocol.json"
    previous_evaluation = None
    if path.exists():
        try:
            previous_evaluation = json.loads(path.read_text(encoding="utf-8")).get("evaluation")
        except (OSError, json.JSONDecodeError):
            previous_evaluation = None
    payload = build_data_protocol(
        kind=kind,
        data_spec=data_spec,
        schedule=schedule,
        position=position,
    )
    if previous_evaluation is not None:
        payload["evaluation"] = previous_evaluation
    return save_json(path, payload)


def append_run_log(run_dir: str | Path, text: str) -> Path:
    """Appends one UTF-8 line to the persistent run log.

    Добавляет одну UTF-8 строку в постоянный лог запуска.
    """
    path = Path(run_dir) / "logs" / "snnlab.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{timestamp} | {text.rstrip()}\n")
    return path


def diagnostic_report(
    *,
    current_configuration: dict[str, Any],
    run_dir: str | Path | None,
    status: str,
    position: int | None,
    total_samples: int | None,
    traceback_text: str | None = None,
    last_log_lines: int = 80,
) -> dict[str, Any]:
    """Builds a shareable diagnostic report for issues and bug reports.

    Создаёт диагностический отчёт для issue и сообщений об ошибках.
    """
    lines: list[str] = []
    if run_dir is not None:
        log_path = Path(run_dir) / "logs" / "snnlab.log"
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()[-last_log_lines:]
            except OSError:
                lines = ["<failed to read persistent log>"]

    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": collect_environment(),
        "status": str(status),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "position": position,
        "total_samples": total_samples,
        "configuration": _jsonable(current_configuration),
        "traceback": traceback_text,
        "last_log_lines": lines,
    }
