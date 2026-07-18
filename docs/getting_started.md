# Getting started / Быстрый старт

## 1. Create an environment / Создайте окружение

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gui,mnist,dev]"
```

## 2. Verify the installation / Проверьте установку

```powershell
python -m pytest
snnlab-dci-regression --locale en
snnlab-gui
```

## 3. Import a preset / Импортируйте preset

Open **Configuration → Import configuration** and select one of the YAML files in `examples/`.

Откройте **Конфигурация → Загрузить конфигурацию** и выберите YAML из `examples/`.

## 4. Start, pause, and continue / Запуск, пауза и продолжение

- **Pause** waits for a safe sample boundary and writes `paused.pkl`.
- **Stop** writes `stopped.pkl`; the current model remains usable.
- Loading an unfinished checkpoint enables **Continue run** and resumes the exact stored schedule.
- **Extend training** appears after the schedule is complete and appends a new schedule.

## 5. Evaluate / Оценка

Use a fixed evaluation protocol when comparing checkpoints. Changing only decoder thresholds reuses cached E responses; changing the assignment policy, seed, state mode, or homeostasis mode reruns the SNN response collection.
