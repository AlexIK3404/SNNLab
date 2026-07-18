# Experiments

Experiment orchestration: sample schedules, training loops, progress events, and architecture-specific runner/trainer state.

Оркестрация экспериментов: расписания sample, циклы обучения, события прогресса и состояние runner/trainer.

## Main modules

- `dci_trainer.py` — checkpoint-safe DCI training lifecycle.
- `reservoir_runner.py` — Reservoir feature collection and readout lifecycle.
- `base.py` — shared experiment history and deterministic sample-order helpers.

This package coordinates components but should not duplicate neuron/synapse mathematics from `architectures/`.

Пакет связывает компоненты, но не должен дублировать математическую логику нейронов и синапсов из `architectures/`.