# Runtime

Persistence and run-lifecycle infrastructure shared by experiments and the GUI.

Инфраструктура сохранения состояния и жизненного цикла запусков, общая для экспериментов и GUI.

## Contents

- `checkpoint.py` — exact-resume checkpoints;
- `model_snapshot.py` — inference/evaluation-oriented model snapshots;
- `control.py` — pause/stop/resume control flags;
- `experiment_io.py` — experiment configuration, metadata, logs, and run artifacts.

A checkpoint and a model snapshot are intentionally different concepts: checkpoints preserve training continuation state, while model snapshots are intended for using an already obtained model.

Checkpoint и model snapshot — разные сущности: checkpoint сохраняет состояние для точного продолжения обучения, а model snapshot предназначен для использования уже полученной модели.