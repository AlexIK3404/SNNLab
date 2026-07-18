from pathlib import Path

from snnlab.runtime.model_snapshot import ModelSnapshotManager


def test_model_snapshot_writes_separate_model_artifacts(tmp_path: Path) -> None:
    manager = ModelSnapshotManager(tmp_path)
    info = manager.save(
        kind="dci",
        training_position=12,
        payload={"model": {"weights": [1, 2, 3]}},
        name="test_model",
    )
    assert info.path.exists()
    assert info.path.parent.name == "models"
    assert info.path.with_suffix(".json").exists()
