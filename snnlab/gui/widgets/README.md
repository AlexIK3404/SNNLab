# GUI widgets

Reusable PySide6/pyqtgraph views used by the main window.

Переиспользуемые PySide6/pyqtgraph-компоненты главного окна.

- `parameter_panel.py` — architecture/run parameters and study-mode help buttons.
- `live_view.py` — current input, raster plots, population state, live activity.
- `training_view.py` — receptive fields and Reservoir learning/readout views.
- `evaluation_view.py` — evaluation protocol, confusion matrix, diagnostics, class responses.
- `network_view.py` — architecture diagrams.
- `metrics_view.py` — tabular experiment metrics.
- `help_dialog.py` — extended study-mode explanations.

Widgets should receive data/events from the GUI/session layer and avoid directly mutating simulation state.