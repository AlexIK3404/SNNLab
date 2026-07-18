# `snnlab` package

This directory contains the installable SNNLab Python package. The code is split by responsibility so that simulation logic, experiment orchestration, GUI code, evaluation, and persistence can evolve independently.

Эта директория содержит устанавливаемый Python-пакет SNNLab. Код разделён по ответственности, чтобы вычислительное ядро, управление экспериментами, GUI, evaluation и сохранение состояния могли развиваться независимо.

| Package | Purpose |
|---|---|
| [`architectures/`](architectures/) | SNN implementations: DCI and Reservoir |
| [`baselines/`](baselines/) | Frozen reference configurations/results used for comparison |
| [`cli/`](cli/) | Command-line entry points |
| [`configs/`](configs/) | Typed experiment and architecture configuration objects |
| [`core/`](core/) | Shared low-level primitives: numerical methods, encoders, events, registries |
| [`data/`](data/) | Dataset loading and deterministic data preparation |
| [`evaluation/`](evaluation/) | Architecture-specific evaluation and SNN decoding |
| [`experiments/`](experiments/) | Training/running orchestration and experiment state |
| [`gui/`](gui/) | PySide6 desktop interface and interactive visualizations |
| [`i18n/`](i18n/) | RU/EN interface strings and study-mode help |
| [`regression/`](regression/) | Regression checks against frozen historical reference behavior |
| [`runtime/`](runtime/) | Checkpoints, model snapshots, run control, experiment I/O |

## Dependency direction

Keep lower-level code independent of higher-level interfaces whenever possible:

```text
core/configs/data
      ↓
architectures
      ↓
experiments/evaluation/runtime
      ↓
gui/cli
```

In particular, the simulation core should not depend on the GUI.

По возможности нижние уровни не должны зависеть от верхних. В частности, вычислительное ядро не должно импортировать GUI.