from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from snnlab.experiments.single_neuron import (
    DEFAULT_ALPHA_VALUES,
    DEFAULT_COMPARISON_METHODS,
    DEFAULT_DT_VALUES_MS,
    INTEGRATORS,
    NEURON_PRESETS,
    NeuronParameters,
    SingleNeuronConfig,
    SingleNeuronStudy,
    SingleNeuronTrace,
    neuron_preset,
    simulate_single_neuron,
    trace_metrics,
)
from snnlab.gui.workers import SingleNeuronStudyWorker

_PLOT_COLORS = (
    "#62a0ea",
    "#57e389",
    "#f6d32d",
    "#ff7800",
    "#dc8add",
    "#99c1f1",
    "#8ff0a4",
    "#f9f06b",
    "#ffbe6f",
)


class SingleNeuronView(QWidget):
    """Interactive first layer of the numerical-method ablation."""

    running_changed = Signal(bool)

    def __init__(self, translator, parent=None) -> None:
        super().__init__(parent)
        self._translator = translator
        self._thread: QThread | None = None
        self._worker: SingleNeuronStudyWorker | None = None
        self._last_study: SingleNeuronStudy | None = None
        self._form_labels: dict[str, QLabel] = {}
        self._last_stage = ""

        self._build_ui()
        self._connect_actions()
        self._populate_static_choices()
        self._apply_preset("regular_spiking")
        self._update_method_parameter_state()
        self.retranslate(translator)
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)

        self.title_label = QLabel()
        self.title_label.setObjectName("workspace_title")
        self.title_label.setStyleSheet("font-size: 17pt; font-weight: 700;")
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setObjectName("research_note")
        self.description_label.setStyleSheet(
            "QLabel { background: #202631; border: 1px solid #344054; "
            "border-radius: 7px; padding: 9px; color: #cfd7e6; }"
        )
        root.addWidget(self.title_label)
        root.addWidget(self.description_label)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.main_splitter, 1)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_scroll.setMinimumWidth(300)
        settings_scroll.setMaximumWidth(430)
        settings_content = QWidget()
        self.settings_layout = QVBoxLayout(settings_content)
        self.settings_layout.setContentsMargins(4, 4, 8, 4)
        self.settings_layout.setSpacing(8)
        settings_scroll.setWidget(settings_content)
        self.main_splitter.addWidget(settings_scroll)

        self.neuron_group = QGroupBox()
        neuron_form = QFormLayout(self.neuron_group)
        self.preset_combo = QComboBox()
        self.a_spin = self._double_spin(0.001, 1.0, 0.001, 0.02, 4)
        self.b_spin = self._double_spin(-2.0, 2.0, 0.01, 0.2, 4)
        self.c_spin = self._double_spin(-120.0, 30.0, 1.0, -65.0, 2)
        self.d_spin = self._double_spin(-20.0, 50.0, 0.5, 8.0, 2)
        self.threshold_spin = self._double_spin(-20.0, 100.0, 1.0, 30.0, 2)
        self._add_form_row(neuron_form, "preset", self.preset_combo)
        self._add_form_row(neuron_form, "a", self.a_spin)
        self._add_form_row(neuron_form, "b", self.b_spin)
        self._add_form_row(neuron_form, "c", self.c_spin)
        self._add_form_row(neuron_form, "d", self.d_spin)
        self._add_form_row(neuron_form, "threshold", self.threshold_spin)
        self.settings_layout.addWidget(self.neuron_group)

        self.simulation_group = QGroupBox()
        simulation_form = QFormLayout(self.simulation_group)
        self.current_spin = self._double_spin(-50.0, 200.0, 0.5, 12.0, 3)
        self.duration_spin = self._double_spin(100.0, 20_000.0, 100.0, 2000.0, 1)
        self.burn_in_spin = self._double_spin(0.0, 19_900.0, 50.0, 500.0, 1)
        self.dt_spin = self._double_spin(0.0001, 10.0, 0.025, 0.1, 5)
        self.method_combo = QComboBox()
        self.reset_combo = QComboBox()
        for combo in (self.method_combo, self.reset_combo):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(14)
        self.composition_s_spin = self._double_spin(0.0, 1.0, 0.05, 0.5, 3)
        self.alpha_spin = self._double_spin(0.0, 3.0, 0.05, 0.5, 3)
        self._add_form_row(simulation_form, "current", self.current_spin)
        self._add_form_row(simulation_form, "duration", self.duration_spin)
        self._add_form_row(simulation_form, "burn_in", self.burn_in_spin)
        self._add_form_row(simulation_form, "dt", self.dt_spin)
        self._add_form_row(simulation_form, "method", self.method_combo)
        self._add_form_row(simulation_form, "reset", self.reset_combo)
        self._add_form_row(simulation_form, "composition_s", self.composition_s_spin)
        self._add_form_row(simulation_form, "composition_alpha", self.alpha_spin)
        self.settings_layout.addWidget(self.simulation_group)

        self.ablation_group = QGroupBox()
        ablation_layout = QVBoxLayout(self.ablation_group)
        self.methods_label = QLabel()
        self.method_list = QListWidget()
        self.method_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.method_list.setMinimumHeight(155)
        self.dt_values_edit = QLineEdit(", ".join(str(value) for value in DEFAULT_DT_VALUES_MS))
        self.current_range_edit = QLineEdit("0, 35, 1")
        self.alpha_values_edit = QLineEdit(", ".join(str(value) for value in DEFAULT_ALPHA_VALUES))
        ablation_layout.addWidget(self.methods_label)
        ablation_layout.addWidget(self.method_list)
        ablation_form = QFormLayout()
        self._add_form_row(ablation_form, "dt_values", self.dt_values_edit)
        self._add_form_row(ablation_form, "current_range", self.current_range_edit)
        self._add_form_row(ablation_form, "alpha_values", self.alpha_values_edit)
        ablation_layout.addLayout(ablation_form)
        self.settings_layout.addWidget(self.ablation_group)

        action_row = QHBoxLayout()
        self.trace_button = QPushButton()
        self.study_button = QPushButton()
        self.cancel_button = QPushButton()
        action_row.addWidget(self.trace_button)
        action_row.addWidget(self.study_button)
        action_row.addWidget(self.cancel_button)
        self.settings_layout.addLayout(action_row)
        self.study_progress = QProgressBar()
        self.study_progress.setRange(0, 1)
        self.study_progress.setValue(0)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.settings_layout.addWidget(self.study_progress)
        self.settings_layout.addWidget(self.status_label)
        self.settings_layout.addStretch(1)

        self.result_tabs = QTabWidget()
        self.main_splitter.addWidget(self.result_tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([350, 1050])

        self.trace_tab = QWidget()
        trace_layout = QVBoxLayout(self.trace_tab)
        self.trace_metrics_label = QLabel()
        self.trace_metrics_label.setWordWrap(True)
        self.voltage_plot = pg.PlotWidget()
        self.recovery_plot = pg.PlotWidget()
        self.voltage_plot.setDownsampling(auto=True, mode="peak")
        self.recovery_plot.setDownsampling(auto=True, mode="peak")
        self.recovery_plot.setXLink(self.voltage_plot)
        trace_layout.addWidget(self.trace_metrics_label)
        trace_layout.addWidget(self.voltage_plot, 2)
        trace_layout.addWidget(self.recovery_plot, 1)

        self.convergence_tab = QWidget()
        convergence_layout = QVBoxLayout(self.convergence_tab)
        self.convergence_note = QLabel()
        self.convergence_note.setWordWrap(True)
        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setLogMode(x=True, y=True)
        self.convergence_plot.addLegend(offset=(8, 8))
        convergence_layout.addWidget(self.convergence_note)
        convergence_layout.addWidget(self.convergence_plot, 1)

        self.fi_tab = QWidget()
        fi_layout = QVBoxLayout(self.fi_tab)
        self.fi_note = QLabel()
        self.fi_note.setWordWrap(True)
        self.fi_plot = pg.PlotWidget()
        self.fi_plot.addLegend(offset=(8, 8))
        fi_layout.addWidget(self.fi_note)
        fi_layout.addWidget(self.fi_plot, 1)

        self.alpha_tab = QWidget()
        alpha_layout = QVBoxLayout(self.alpha_tab)
        self.alpha_note = QLabel()
        self.alpha_note.setWordWrap(True)
        self.alpha_plot = pg.PlotWidget()
        self.alpha_plot.addLegend(offset=(8, 8))
        alpha_layout.addWidget(self.alpha_note)
        alpha_layout.addWidget(self.alpha_plot, 1)

        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.summary_note = QLabel()
        self.summary_note.setWordWrap(True)
        self.summary_table = QTableWidget(0, 8)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.output_path_label = QLabel()
        self.output_path_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_note)
        summary_layout.addWidget(self.summary_table, 1)
        summary_layout.addWidget(self.output_path_label)

        for tab in (
            self.trace_tab,
            self.convergence_tab,
            self.fi_tab,
            self.alpha_tab,
            self.summary_tab,
        ):
            self.result_tabs.addTab(tab, "")

        for plot in (
            self.voltage_plot,
            self.recovery_plot,
            self.convergence_plot,
            self.fi_plot,
            self.alpha_plot,
        ):
            plot.showGrid(x=True, y=True, alpha=0.22)

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        editor = QDoubleSpinBox()
        editor.setRange(minimum, maximum)
        editor.setSingleStep(step)
        editor.setDecimals(decimals)
        editor.setValue(value)
        return editor

    def _add_form_row(self, form: QFormLayout, key: str, editor: QWidget) -> None:
        label = QLabel()
        self._form_labels[key] = label
        form.addRow(label, editor)

    def _connect_actions(self) -> None:
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.method_combo.currentIndexChanged.connect(self._update_method_parameter_state)
        self.duration_spin.valueChanged.connect(self._duration_changed)
        self.trace_button.clicked.connect(self.run_trace)
        self.study_button.clicked.connect(self.run_study)
        self.cancel_button.clicked.connect(self.cancel_study)

    def _populate_static_choices(self) -> None:
        self.preset_combo.clear()
        for preset_id in NEURON_PRESETS:
            self.preset_combo.addItem(preset_id, preset_id)

        self.method_combo.clear()
        self.method_list.clear()
        for method_id in INTEGRATORS:
            self.method_combo.addItem(method_id, method_id)
            item = QListWidgetItem(method_id)
            item.setData(Qt.ItemDataRole.UserRole, method_id)
            self.method_list.addItem(item)
            if method_id in DEFAULT_COMPARISON_METHODS:
                item.setSelected(True)
        self._set_combo_data(self.method_combo, "midpoint")

        self.reset_combo.clear()
        self.reset_combo.addItem("event", "event")
        self.reset_combo.addItem("grid", "grid")

    def _preset_changed(self) -> None:
        preset_id = self.preset_combo.currentData()
        if preset_id:
            self._apply_preset(str(preset_id))

    def _apply_preset(self, preset_id: str) -> None:
        parameters = neuron_preset(preset_id)
        self.a_spin.setValue(parameters.a)
        self.b_spin.setValue(parameters.b)
        self.c_spin.setValue(parameters.c)
        self.d_spin.setValue(parameters.d)
        self.threshold_spin.setValue(parameters.threshold_mv)

    def _duration_changed(self, duration: float) -> None:
        self.burn_in_spin.setMaximum(max(0.0, duration - 0.1))
        if self.burn_in_spin.value() >= duration:
            self.burn_in_spin.setValue(max(0.0, 0.25 * duration))

    def _update_method_parameter_state(self) -> None:
        method_id = str(self.method_combo.currentData() or "")
        self.composition_s_spin.setEnabled(method_id == "euler_composition_real")
        self.alpha_spin.setEnabled(method_id == "euler_composition_complex")

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _retranslate_combo(self, combo: QComboBox, label_for_data) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        for index in range(combo.count()):
            combo.setItemText(index, label_for_data(str(combo.itemData(index))))
        self._set_combo_data(combo, str(current))
        combo.blockSignals(False)

    def _method_name(self, method_id: str) -> str:
        return self._translator.tr(f"components.numerical_methods.{method_id}")

    def retranslate(self, translator) -> None:
        self._translator = translator
        self.title_label.setText(translator.tr("gui.single_neuron.title"))
        self.description_label.setText(translator.tr("gui.single_neuron.description"))
        self.neuron_group.setTitle(translator.tr("gui.single_neuron.groups.neuron"))
        self.simulation_group.setTitle(translator.tr("gui.single_neuron.groups.simulation"))
        self.ablation_group.setTitle(translator.tr("gui.single_neuron.groups.ablation"))

        for key, label in self._form_labels.items():
            label.setText(translator.tr(f"gui.single_neuron.params.{key}"))
        self.methods_label.setText(translator.tr("gui.single_neuron.params.methods"))
        self.trace_button.setText(translator.tr("gui.single_neuron.buttons.trace"))
        self.study_button.setText(translator.tr("gui.single_neuron.buttons.study"))
        self.cancel_button.setText(translator.tr("gui.single_neuron.buttons.cancel"))

        self._retranslate_combo(
            self.preset_combo,
            lambda value: translator.tr(f"gui.single_neuron.presets.{value}"),
        )
        self._retranslate_combo(self.method_combo, self._method_name)
        self._retranslate_combo(
            self.reset_combo,
            lambda value: translator.tr(f"gui.single_neuron.reset_modes.{value}"),
        )
        for index in range(self.method_list.count()):
            item = self.method_list.item(index)
            item.setText(self._method_name(str(item.data(Qt.ItemDataRole.UserRole))))

        tab_keys = ("trace", "convergence", "fi", "alpha", "summary")
        for index, key in enumerate(tab_keys):
            self.result_tabs.setTabText(index, translator.tr(f"gui.single_neuron.tabs.{key}"))
        self.convergence_note.setText(translator.tr("gui.single_neuron.notes.convergence"))
        self.fi_note.setText(translator.tr("gui.single_neuron.notes.fi"))
        self.alpha_note.setText(translator.tr("gui.single_neuron.notes.alpha"))
        self.summary_note.setText(translator.tr("gui.single_neuron.notes.summary"))
        self.summary_table.setHorizontalHeaderLabels(
            [
                translator.tr(f"gui.single_neuron.table.{key}")
                for key in (
                    "method",
                    "lte_order",
                    "isi_order",
                    "grid_order",
                    "error",
                    "drift",
                    "cost",
                    "status",
                )
            ]
        )
        if self._thread is None:
            self.status_label.setText(translator.tr("gui.single_neuron.status.ready"))
        self._set_plot_labels()

    def _set_plot_labels(self) -> None:
        tr = self._translator.tr
        self.voltage_plot.setLabel("bottom", tr("gui.single_neuron.axes.time_ms"))
        self.voltage_plot.setLabel("left", tr("gui.single_neuron.axes.voltage_mv"))
        self.recovery_plot.setLabel("bottom", tr("gui.single_neuron.axes.time_ms"))
        self.recovery_plot.setLabel("left", tr("gui.single_neuron.axes.recovery"))
        self.convergence_plot.setLabel("bottom", tr("gui.single_neuron.axes.dt_ms"))
        self.convergence_plot.setLabel("left", tr("gui.single_neuron.axes.isi_error_ms"))
        self.fi_plot.setLabel("bottom", tr("gui.single_neuron.axes.current"))
        self.fi_plot.setLabel("left", tr("gui.single_neuron.axes.rate_hz"))
        self.alpha_plot.setLabel("bottom", "α")
        self.alpha_plot.setLabel("left", tr("gui.single_neuron.axes.order"))

    def _configuration(self) -> SingleNeuronConfig:
        return SingleNeuronConfig(
            neuron=NeuronParameters(
                a=self.a_spin.value(),
                b=self.b_spin.value(),
                c=self.c_spin.value(),
                d=self.d_spin.value(),
                threshold_mv=self.threshold_spin.value(),
            ),
            input_current=self.current_spin.value(),
            duration_ms=self.duration_spin.value(),
            burn_in_ms=self.burn_in_spin.value(),
            dt_ms=self.dt_spin.value(),
            method_id=str(self.method_combo.currentData()),
            reset_mode=str(self.reset_combo.currentData()),
            composition_s=self.composition_s_spin.value(),
            composition_alpha=self.alpha_spin.value(),
        )

    @staticmethod
    def _parse_values(text: str, *, field: str, maximum_count: int) -> tuple[float, ...]:
        chunks = [chunk.strip() for chunk in text.replace(";", ",").split(",")]
        values = tuple(float(chunk) for chunk in chunks if chunk)
        if not values:
            raise ValueError(f"{field}: no values")
        if len(values) > maximum_count:
            raise ValueError(f"{field}: at most {maximum_count} values are allowed")
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"{field}: all values must be finite")
        return values

    def _selected_methods(self) -> tuple[str, ...]:
        selected = tuple(
            str(item.data(Qt.ItemDataRole.UserRole)) for item in self.method_list.selectedItems()
        )
        if not selected:
            raise ValueError(self._translator.tr("gui.single_neuron.errors.no_methods"))
        return selected

    def _study_axes(self) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        dt_values = self._parse_values(
            self.dt_values_edit.text(),
            field="dt",
            maximum_count=12,
        )
        if any(value <= 0.0 for value in dt_values):
            raise ValueError("dt: values must be positive")
        current_range = self._parse_values(
            self.current_range_edit.text(),
            field="I range",
            maximum_count=3,
        )
        if len(current_range) != 3:
            raise ValueError("I range: enter start, stop, step")
        start, stop, step = current_range
        if step <= 0.0 or stop < start:
            raise ValueError("I range: require step > 0 and stop >= start")
        currents = tuple(np.arange(start, stop + 0.5 * step, step, dtype=np.float64))
        if len(currents) > 200:
            raise ValueError("I range: at most 200 current samples are allowed")
        alpha_values = self._parse_values(
            self.alpha_values_edit.text(),
            field="alpha",
            maximum_count=20,
        )
        if any(value < 0.0 for value in alpha_values):
            raise ValueError("alpha: values must be non-negative")
        return dt_values, currents, alpha_values

    def run_trace(self) -> None:
        if self.is_running():
            return
        try:
            config = self._configuration()
            trace = simulate_single_neuron(config)
        except Exception as exc:  # noqa: BLE001 - GUI validation boundary.
            QMessageBox.critical(self, self._translator.tr("gui.dialogs.error_title"), str(exc))
            return
        self._display_trace(trace, config)
        self.result_tabs.setCurrentWidget(self.trace_tab)
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.trace_done"))

    def run_study(self) -> None:
        if self.is_running():
            return
        try:
            config = self._configuration()
            method_ids = self._selected_methods()
            dt_values, currents, alpha_values = self._study_axes()
        except Exception as exc:  # noqa: BLE001 - GUI validation boundary.
            QMessageBox.critical(self, self._translator.tr("gui.dialogs.error_title"), str(exc))
            return

        thread = QThread(self)
        worker = SingleNeuronStudyWorker(
            config=config,
            method_ids=method_ids,
            dt_values_ms=dt_values,
            currents=currents,
            alpha_values=alpha_values,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.result_ready.connect(self._on_study_ready)
        worker.failed.connect(self._on_study_failed)
        worker.cancelled.connect(self._on_study_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        self._last_stage = ""
        self.output_path_label.clear()
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.running"))
        self._set_running(True)
        thread.start()

    def cancel_study(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self.status_label.setText(self._translator.tr("gui.single_neuron.status.cancelling"))

    def _on_progress(self, stage: str, value: int, maximum: int) -> None:
        self._last_stage = stage
        self.study_progress.setRange(0, max(1, maximum))
        self.study_progress.setValue(min(value, max(1, maximum)))
        self.status_label.setText(
            self._translator.tr(
                "gui.single_neuron.status.stage",
                stage=self._translator.tr(f"gui.single_neuron.stages.{stage}"),
                value=value,
                maximum=maximum,
            )
        )

    def _on_study_ready(self, study: SingleNeuronStudy, run_dir: str) -> None:
        self._last_study = study
        self._display_trace(study.trace, study.config)
        self._display_convergence(study)
        self._display_fi_curves(study)
        self._display_alpha_sweep(study)
        self._display_summary(study)
        self.output_path_label.setText(
            self._translator.tr("gui.single_neuron.output_path", path=run_dir)
        )
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.completed"))
        self.result_tabs.setCurrentWidget(self.summary_tab)

    def _on_study_failed(self, traceback_text: str) -> None:
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.error"))
        QMessageBox.critical(
            self,
            self._translator.tr("gui.dialogs.error_title"),
            traceback_text,
        )

    def _on_study_cancelled(self) -> None:
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.cancelled"))

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.trace_button.setEnabled(not running)
        self.study_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        for widget in (
            self.preset_combo,
            self.a_spin,
            self.b_spin,
            self.c_spin,
            self.d_spin,
            self.threshold_spin,
            self.current_spin,
            self.duration_spin,
            self.burn_in_spin,
            self.dt_spin,
            self.method_combo,
            self.reset_combo,
            self.composition_s_spin,
            self.alpha_spin,
            self.method_list,
            self.dt_values_edit,
            self.current_range_edit,
            self.alpha_values_edit,
        ):
            widget.setEnabled(not running)
        if not running:
            self._update_method_parameter_state()
        self.running_changed.emit(running)

    @staticmethod
    def _format_number(value: float, digits: int = 3) -> str:
        if not np.isfinite(value):
            return "—"
        return f"{value:.{digits}g}"

    def _display_trace(self, trace: SingleNeuronTrace, config: SingleNeuronConfig) -> None:
        metrics = trace_metrics(trace, config.burn_in_ms)
        tr = self._translator.tr
        self.trace_metrics_label.setText(
            tr(
                "gui.single_neuron.trace_metrics",
                spikes=metrics.spike_count,
                rate=self._format_number(metrics.firing_rate_hz, 5),
                isi=self._format_number(metrics.mean_isi_ms, 5),
                cv=self._format_number(metrics.isi_cv, 4),
                runtime=self._format_number(metrics.elapsed_ms, 5),
            )
        )
        self.voltage_plot.clear()
        self.recovery_plot.clear()
        self.voltage_plot.plot(
            trace.time_ms,
            trace.voltage_mv,
            pen=pg.mkPen("#62a0ea", width=1.4),
        )
        if trace.spike_times_ms.size:
            self.voltage_plot.plot(
                trace.spike_times_ms,
                np.full(trace.spike_times_ms.shape, config.neuron.threshold_mv),
                pen=None,
                symbol="t1",
                symbolSize=7,
                symbolBrush="#ff7800",
                symbolPen=None,
            )
        self.recovery_plot.plot(
            trace.time_ms,
            trace.recovery,
            pen=pg.mkPen("#57e389", width=1.3),
        )
        self._set_plot_labels()

    def _display_convergence(self, study: SingleNeuronStudy) -> None:
        self.convergence_plot.clear()
        for index, series in enumerate(study.convergence.series):
            x = np.asarray([point.dt_ms for point in series.points], dtype=np.float64)
            y = np.asarray([point.period_error_ms for point in series.points], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            self.convergence_plot.plot(
                x[valid],
                y[valid],
                pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.8),
                symbol="o",
                symbolSize=6,
                name=self._method_name(series.method_id),
            )
        self._set_plot_labels()

    def _display_fi_curves(self, study: SingleNeuronStudy) -> None:
        self.fi_plot.clear()
        for index, curve in enumerate(study.fi_curves):
            valid = ~curve.diverged & np.isfinite(curve.rates_hz)
            self.fi_plot.plot(
                curve.currents[valid],
                curve.rates_hz[valid],
                pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.8),
                symbol="o",
                symbolSize=4,
                name=self._method_name(curve.method_id),
            )
        self._set_plot_labels()

    def _display_alpha_sweep(self, study: SingleNeuronStudy) -> None:
        self.alpha_plot.clear()
        alpha = np.asarray([point.alpha for point in study.alpha_sweep], dtype=np.float64)
        local = np.asarray([point.local_order for point in study.alpha_sweep], dtype=np.float64)
        period = np.asarray(
            [point.observed_period_order for point in study.alpha_sweep], dtype=np.float64
        )
        self.alpha_plot.plot(
            alpha,
            local,
            pen=pg.mkPen("#62a0ea", width=2),
            symbol="o",
            name=self._translator.tr("gui.single_neuron.legend.local_order"),
        )
        self.alpha_plot.plot(
            alpha,
            period,
            pen=pg.mkPen("#57e389", width=1.7),
            symbol="s",
            name=self._translator.tr("gui.single_neuron.legend.period_order"),
        )
        self.alpha_plot.addLine(y=1.0, pen=pg.mkPen("#777777", style=Qt.PenStyle.DotLine))
        self.alpha_plot.addLine(y=2.0, pen=pg.mkPen("#aaaaaa", style=Qt.PenStyle.DashLine))
        self.alpha_plot.addLine(x=0.5, pen=pg.mkPen("#f66151", style=Qt.PenStyle.DashLine))
        self._set_plot_labels()

    def _display_summary(self, study: SingleNeuronStudy) -> None:
        self.summary_table.setRowCount(len(study.convergence.series))
        grid_by_method = {series.method_id: series for series in study.grid_convergence.series}
        for row, series in enumerate(study.convergence.series):
            fine = min(series.points, key=lambda point: point.dt_ms)
            grid_series = grid_by_method[series.method_id]
            divergence_count = sum(point.diverged for point in series.points)
            status = (
                self._translator.tr(
                    "gui.single_neuron.table.diverged",
                    count=divergence_count,
                )
                if divergence_count
                else self._translator.tr("gui.single_neuron.table.ok")
            )
            values: tuple[Any, ...] = (
                self._method_name(series.method_id),
                self._format_number(series.local_order, 4),
                self._format_number(series.observed_period_order, 4),
                self._format_number(grid_series.observed_period_order, 4),
                self._format_number(fine.period_error_ms, 5),
                self._format_number(fine.drift_ms_per_s, 5),
                self._format_number(fine.nanoseconds_per_step, 5),
                status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.summary_table.setItem(row, column, item)

    def set_help_mode(self, enabled: bool) -> None:
        if enabled:
            self.dt_spin.setToolTip(self._translator.tr("gui.single_neuron.help.dt"))
            self.reset_combo.setToolTip(self._translator.tr("gui.single_neuron.help.reset"))
            self.alpha_spin.setToolTip(self._translator.tr("gui.single_neuron.help.alpha"))
        else:
            for widget in (self.dt_spin, self.reset_combo, self.alpha_spin):
                widget.setToolTip("")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def shutdown(self, timeout_ms: int = 5000) -> bool:
        if not self.is_running():
            return True
        self.cancel_study()
        assert self._thread is not None
        # The usual worker.finished -> thread.quit connection is queued to the
        # GUI thread. During closeEvent this method waits synchronously, so ask
        # the thread's event loop to exit directly before waiting.
        self._thread.quit()
        return bool(self._thread.wait(timeout_ms))

    def refresh_layout(self) -> None:
        for plot in (
            self.voltage_plot,
            self.recovery_plot,
            self.convergence_plot,
            self.fi_plot,
            self.alpha_plot,
        ):
            plot.updateGeometry()
            plot.getViewBox().update()
