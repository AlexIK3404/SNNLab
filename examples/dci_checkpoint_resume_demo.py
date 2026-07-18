"""
Shows how to load a DCI checkpoint and continue training.

Показывает, как загрузить DCI-checkpoint и продолжить обучение.
"""

from __future__ import annotations

from pathlib import Path

from snnlab.data import load_mnist_dataset
from snnlab.experiments.dci_trainer import DCITrainer
from snnlab.runtime.checkpoint import CheckpointManager

RUN_DIR = Path("runs/dci_mnist")
CHECKPOINT = RUN_DIR / "checkpoints" / "stopped.pkl"

x_train, y_train, _, _ = load_mnist_dataset()
payload = CheckpointManager(RUN_DIR).load(CHECKPOINT)
state = payload["state"]

# EN: The loaded state already contains model weights, membrane/synaptic state,
#     homeostasis, sample position, history, and RNG state.
# RU: Загруженное состояние уже содержит веса модели, мембранное/синаптическое
#     состояние, гомеостаз, позицию sample, историю и состояние RNG.
trainer = DCITrainer(
    x_train=x_train,
    y_train=y_train,
    state=state,
    run_dir=RUN_DIR,
    checkpoint_every=50,
)

# EN: Append 200 additional presentations. Set allow_repeats=False to use only
#     dataset items absent from the existing schedule.
# RU: Добавляем 200 предъявлений. allow_repeats=False использует только объекты,
#     отсутствующие в существующем расписании.
trainer.extend_training(
    additional_samples=200,
    seed=state.model.cfg.seed + 300,
    allow_repeats=False,
)
trainer.train()
