from __future__ import annotations

import copy
import json
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelSnapshotInfo:
    """Describes a saved inference-oriented model snapshot. / Описывает сохранённый model snapshot."""

    kind: str
    created_at_utc: str
    training_position: int
    path: Path


class ModelSnapshotManager:
    """
    Saves model-oriented snapshots separately from exact-resume checkpoints.

    Model snapshots intentionally omit schedule, history, and training RNG state.
    They are meant for evaluation/inference, not exact training continuation.

    Сохраняет model snapshot отдельно от checkpoint точного resume.

    Model snapshot намеренно не содержит schedule, history и training RNG state.
    Он предназначен для evaluation/inference, а не точного продолжения обучения.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.models_dir = self.run_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        kind: str,
        training_position: int,
        payload: dict[str, Any],
        name: str | None = None,
    ) -> ModelSnapshotInfo:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stem = name or f"model_{int(training_position):06d}_{timestamp}"
        pkl_path = self.models_dir / f"{stem}.pkl"
        json_path = self.models_dir / f"{stem}.json"
        created = datetime.now(UTC).isoformat()

        envelope = {
            "format_version": 1,
            "kind": str(kind),
            "created_at_utc": created,
            "training_position": int(training_position),
            **copy.deepcopy(payload),
        }
        with pkl_path.open("wb") as file:
            pickle.dump(envelope, file, protocol=pickle.HIGHEST_PROTOCOL)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "format_version": 1,
                    "kind": str(kind),
                    "created_at_utc": created,
                    "training_position": int(training_position),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        return ModelSnapshotInfo(
            kind=str(kind),
            created_at_utc=created,
            training_position=int(training_position),
            path=pkl_path,
        )
