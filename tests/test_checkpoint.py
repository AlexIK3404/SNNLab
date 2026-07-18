from pathlib import Path

import numpy as np

from snnlab.runtime.checkpoint import CheckpointManager


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "run")
    path = manager.save(name="test", payload={"array": np.arange(5)}, metadata={"step": 3})
    payload = manager.load(path)
    np.testing.assert_array_equal(payload["array"], np.arange(5))
