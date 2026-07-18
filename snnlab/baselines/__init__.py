"""Frozen reproducible reference configurations. / Зафиксированные эталонные конфигурации."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import yaml


def available_baselines() -> tuple[str, ...]:
    """Returns packaged baseline identifiers. / Возвращает ID встроенных baseline."""
    directory = files(__name__)
    return tuple(sorted(path.stem for path in directory.iterdir() if path.name.endswith(".yaml")))


def load_baseline(baseline_id: str) -> dict[str, Any]:
    """Loads one packaged baseline YAML by identifier. / Загружает baseline YAML по ID."""
    resource = files(__name__).joinpath(f"{baseline_id}.yaml")
    if not resource.is_file():
        raise KeyError(f"Unknown baseline {baseline_id!r}. Available: {available_baselines()}")
    with resource.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Baseline {baseline_id!r} must contain a YAML mapping")
    return data
