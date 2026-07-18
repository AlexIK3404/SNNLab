from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

CHECKPOINT_FORMAT_VERSION = 1


def to_jsonable(value: Any) -> Any:
    """
    Converts metadata into a JSON-compatible representation.

    Преобразует metadata в JSON-совместимое представление.
    """
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class CheckpointManager:
    """
    Saves atomic Python checkpoints plus human-readable JSON metadata.

    Pickle is deliberately limited to Stage 1 because it preserves numpy arrays,
    sklearn models, dataclasses, and RNG states with minimal migration code.
    Only trusted checkpoint files must be loaded.

    Сохраняет атомарные Python-checkpoint и читаемую JSON metadata.

    Pickle намеренно используется только на Этапе 1: он без лишнего кода
    сохраняет numpy-массивы, sklearn-модели, dataclass и состояния RNG.
    Загружать можно только доверенные checkpoint-файлы.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        name: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        metadata_path = self.checkpoint_dir / f"{name}.json"

        envelope = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        metadata_envelope = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "created_at_utc": envelope["created_at_utc"],
            **(metadata or {}),
        }

        self._atomic_pickle(checkpoint_path, envelope)
        self._atomic_json(metadata_path, to_jsonable(metadata_envelope))
        return checkpoint_path

    def load(self, path: str | Path) -> dict[str, Any]:
        """
        Loads a trusted checkpoint and returns its payload.

        Загружает доверенный checkpoint и возвращает payload.
        """
        with Path(path).open("rb") as stream:
            envelope = pickle.load(stream)
        if envelope.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint version {envelope.get('format_version')!r}")
        return envelope["payload"]

    @staticmethod
    def _atomic_pickle(path: Path, value: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
