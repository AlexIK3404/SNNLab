from pathlib import Path

import numpy as np

from snnlab.runtime.experiment_io import (
    build_data_protocol,
    build_user_configuration,
    load_yaml,
    save_yaml,
    update_data_protocol_evaluation,
    update_data_protocol_reservoir_evaluation,
    write_run_manifest,
)


def test_portable_configuration_roundtrip(tmp_path: Path) -> None:
    """Portable YAML preserves GUI and evaluation parameters.

    Переносимый YAML сохраняет GUI- и evaluation-параметры.
    """
    payload = build_user_configuration(
        architecture="dci",
        task="classification",
        backend="python_cpu",
        locale="ru",
        parameters={"seed": 52, "n_exc": 400},
        evaluation={"assignment_policy": "exclude_training", "test_samples": 500},
    )
    path = save_yaml(tmp_path / "experiment.yaml", payload)
    restored = load_yaml(path)
    assert restored["architecture"] == "dci"
    assert restored["parameters"]["n_exc"] == 400
    assert restored["evaluation"]["assignment_policy"] == "exclude_training"


def test_run_manifest_and_protocol_are_explicit(tmp_path: Path) -> None:
    """Release artifacts include exact schedule and evaluation overlap.

    Release-артефакты содержат точное расписание и overlap evaluation.
    """
    schedule = np.array([2, 0, 1, 2], dtype=np.int64)
    protocol = build_data_protocol(
        kind="dci",
        data_spec={
            "dataset": "mnist",
            "train_limit": 10,
            "test_limit": 4,
            "subset_seed": 52,
            "samples_per_epoch": 2,
            "epochs": 2,
        },
        schedule=schedule,
    )
    config = build_user_configuration(
        architecture="dci",
        task="classification",
        backend="python_cpu",
        locale="en",
        parameters={"seed": 52},
        evaluation={},
    )
    write_run_manifest(run_dir=tmp_path, user_configuration=config, data_protocol=protocol)
    assert (tmp_path / "experiment.yaml").is_file()
    assert (tmp_path / "environment.json").is_file()
    assert (tmp_path / "data_protocol.json").is_file()
    assert protocol["unique_training_samples"] == 3
    assert protocol["repeated_presentations"] == 1

    update_data_protocol_evaluation(
        run_dir=tmp_path,
        assignment_policy="full_train_pool",
        assignment_indices=np.array([1, 4, 7]),
        test_indices=np.array([0, 2]),
        training_indices=schedule,
    )
    import json

    saved = json.loads((tmp_path / "data_protocol.json").read_text(encoding="utf-8"))
    assert saved["evaluation"]["training_assignment_overlap"] == 1


def test_reservoir_evaluation_protocol_records_test_indices(tmp_path: Path) -> None:
    """Reservoir evaluation persists the exact deterministic test subset.

    Reservoir-evaluation сохраняет точную детерминированную test-подвыборку.
    """
    protocol = build_data_protocol(
        kind="reservoir",
        data_spec={"dataset": "iris", "train_pool_size": 100, "test_pool_size": 50},
        schedule=np.array([0, 1, 2], dtype=np.int64),
    )
    config = build_user_configuration(
        architecture="reservoir",
        task="classification",
        backend="python_cpu",
        locale="en",
        parameters={},
        evaluation={},
    )
    write_run_manifest(run_dir=tmp_path, user_configuration=config, data_protocol=protocol)
    update_data_protocol_reservoir_evaluation(
        run_dir=tmp_path,
        test_indices=np.array([0, 1, 2, 3], dtype=np.int64),
    )

    import json

    saved = json.loads((tmp_path / "data_protocol.json").read_text(encoding="utf-8"))
    assert saved["evaluation"]["assignment_policy"] is None
    assert saved["evaluation"]["test_indices"] == [0, 1, 2, 3]
