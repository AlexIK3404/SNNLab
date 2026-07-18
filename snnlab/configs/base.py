from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunConfig:
    """
    Defines architecture-independent experiment-run settings.

    Stores reproducibility parameters, output location, checkpoint cadence,
    and dataset limits. Architecture-specific parameters belong to separate
    configuration classes.

    Определяет настройки запуска, не зависящие от архитектуры.

    Хранит параметры воспроизводимости, путь результатов, частоту checkpoint
    и ограничения выборки. Параметры конкретной архитектуры должны находиться
    в отдельных конфигурационных классах.
    """

    run_name: str = "debug_run"
    seed: int = 52
    backend_id: str = "python_cpu"
    n_train_samples: int = 500
    n_test_samples: int = 200
    run_dir: str = "runs/debug_run"
    checkpoint_every: int = 50

    @property
    def run_path(self) -> Path:
        """
        Returns the run directory as a Path.

        Возвращает директорию запуска как Path.
        """
        return Path(self.run_dir)
