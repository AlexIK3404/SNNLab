# GUI

PySide6 desktop interface for configuring, running, visualizing, pausing/resuming, and evaluating SNN experiments.

Desktop-интерфейс PySide6 для настройки, запуска, визуализации, паузы/продолжения и оценки SNN-экспериментов.

## Structure

- `app.py` — application entry point.
- `main_window.py` — main window, menus, tabs, and lifecycle coordination.
- `session.py` — GUI-side experiment/session construction and state.
- `event_bridge.py` / `workers.py` — Qt-safe bridge between background experiment execution and widgets.
- `widgets/` — reusable views for parameters, live activity, training, evaluation, metrics, and network diagrams.

GUI code should call public framework APIs rather than reimplementing simulation logic.

GUI должен использовать API фреймворка, а не повторно реализовывать вычислительную логику.