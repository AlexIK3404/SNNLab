from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
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
from snnlab.gui.widgets.help_dialog import HelpDialog
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

_PARAMETER_HELP_TOPICS = {
    "preset": "single_neuron.preset",
    "a": "izhikevich.a",
    "b": "izhikevich.b",
    "c": "izhikevich.c",
    "d": "izhikevich.d",
    "threshold": "single_neuron.threshold",
    "current": "single_neuron.current",
    "duration": "single_neuron.duration",
    "burn_in": "single_neuron.burn_in",
    "dt": "simulation.dt_ms",
    "method": "single_neuron.method",
    "reset": "single_neuron.reset",
    "composition_s": "single_neuron.composition_s",
    "composition_alpha": "single_neuron.composition_alpha",
    "methods": "single_neuron.methods",
    "dt_values": "single_neuron.dt_values",
    "current_range": "single_neuron.current_range",
    "alpha_values": "single_neuron.alpha_values",
}


class ContextHelpButton(QToolButton):
    """Study-mode button that opens one localized structured help topic."""

    def __init__(self, topic_id: str, translator, parent=None) -> None:
        super().__init__(parent)
        self.topic_id = str(topic_id)
        self._translator = translator
        self.setText("?")
        self.setFixedSize(22, 22)
        self.setVisible(False)
        self.clicked.connect(self._open_help)
        self.retranslate(translator)

    def retranslate(self, translator) -> None:
        self._translator = translator
        try:
            topic = translator.help_topic(self.topic_id)
        except (KeyError, TypeError):
            self.setToolTip(self.topic_id)
            return
        short = str(topic.get("short", ""))
        long_text = str(topic.get("long", ""))
        self.setToolTip(f"{short}\n\n{long_text}" if short and long_text else short or long_text)

    def _open_help(self) -> None:
        HelpDialog(self._translator.help_topic(self.topic_id), parent=self).exec()


class ParameterHelpLabel(QWidget):
    """Compact form label with a help button shown only in learning mode."""

    def __init__(self, topic_id: str, translator, parent=None) -> None:
        super().__init__(parent)
        self.text_label = QLabel()
        self.help_button = ContextHelpButton(topic_id, translator, self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.text_label)
        layout.addWidget(self.help_button)
        layout.addStretch(1)

    def setText(self, text: str) -> None:  # noqa: N802 - QLabel-compatible API
        self.text_label.setText(text)

    def set_help_mode(self, enabled: bool) -> None:
        self.help_button.setVisible(bool(enabled))

    def retranslate_help(self, translator) -> None:
        self.help_button.retranslate(translator)


class SingleNeuronView(QWidget):
    """Interactive first layer of the numerical-method ablation."""

    running_changed = Signal(bool)

    def __init__(self, translator, parent=None) -> None:
        super().__init__(parent)
        self._translator = translator
        self._thread: QThread | None = None
        self._worker: SingleNeuronStudyWorker | None = None
        self._last_study: SingleNeuronStudy | None = None
        self._form_labels: dict[str, ParameterHelpLabel] = {}
        self._help_buttons: list[ContextHelpButton] = []
        self._help_mode = False
        self._last_stage = ""
        self._trace_items: dict[str, list[Any]] = {}
        self._trace_method_checks: dict[str, QCheckBox] = {}
        self._hover_series: dict[int, list[dict[str, Any]]] = {}
        self._mouse_proxies: list[Any] = []
        self._hover_hint = ""

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

        title_row = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("workspace_title")
        self.title_label.setStyleSheet("font-size: 17pt; font-weight: 700;")
        self.workspace_help_button = ContextHelpButton(
            "single_neuron.workspace", self._translator, self
        )
        self._help_buttons.append(self.workspace_help_button)
        title_row.addWidget(self.title_label)
        title_row.addWidget(self.workspace_help_button)
        title_row.addStretch(1)
        root.addLayout(title_row)

        run_status_row = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.study_progress = QProgressBar()
        self.study_progress.setRange(0, 1)
        self.study_progress.setValue(0)
        self.study_progress.setMaximumWidth(360)
        run_status_row.addWidget(self.status_label, 1)
        run_status_row.addWidget(self.study_progress)
        root.addLayout(run_status_row)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(self.main_splitter, 1)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.settings_scroll.setMinimumWidth(330)
        settings_content = QWidget()
        self.settings_layout = QVBoxLayout(settings_content)
        self.settings_layout.setContentsMargins(4, 4, 8, 4)
        self.settings_layout.setSpacing(8)
        self.settings_scroll.setWidget(settings_content)
        self.main_splitter.addWidget(self.settings_scroll)

        self.neuron_group = QGroupBox()
        neuron_form = QFormLayout(self.neuron_group)
        self._configure_form(neuron_form)
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
        self._configure_form(simulation_form)
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
        self.methods_label = ParameterHelpLabel(
            _PARAMETER_HELP_TOPICS["methods"], self._translator, self
        )
        self._form_labels["methods"] = self.methods_label
        self.method_list = QListWidget()
        self.method_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.method_list.setMinimumHeight(155)
        self.dt_values_edit = QLineEdit(", ".join(str(value) for value in DEFAULT_DT_VALUES_MS))
        self.current_range_edit = QLineEdit("0, 35, 1")
        self.alpha_values_edit = QLineEdit(", ".join(str(value) for value in DEFAULT_ALPHA_VALUES))
        ablation_layout.addWidget(self.methods_label)
        ablation_layout.addWidget(self.method_list)
        ablation_form = QFormLayout()
        self._configure_form(ablation_form)
        self._add_form_row(ablation_form, "dt_values", self.dt_values_edit)
        self._add_form_row(ablation_form, "current_range", self.current_range_edit)
        self._add_form_row(ablation_form, "alpha_values", self.alpha_values_edit)
        ablation_layout.addLayout(ablation_form)
        self.settings_layout.addWidget(self.ablation_group)

        # MainWindow places these controls in the same top toolbar used by the
        # network workspace. Keeping the actions out of the scrolling form
        # makes starting a run possible from any scroll position.
        self.trace_button = QPushButton(self)
        self.study_button = QPushButton(self)
        self.cancel_button = QPushButton(self)
        self.settings_layout.addStretch(1)

        self.result_tabs = QTabWidget()
        self.main_splitter.addWidget(self.result_tabs)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([430, 970])

        self.trace_tab = QWidget()
        trace_layout = QVBoxLayout(self.trace_tab)
        self.trace_metrics_label = QLabel()
        self.trace_metrics_label.setWordWrap(True)
        self.trace_method_bar = QWidget()
        self.trace_method_layout = QHBoxLayout(self.trace_method_bar)
        self.trace_method_layout.setContentsMargins(0, 0, 0, 0)
        self.trace_method_layout.setSpacing(10)
        self.trace_method_label = QLabel()
        self.trace_method_layout.addWidget(self.trace_method_label)
        self.trace_method_layout.addStretch(1)
        self.trace_method_bar.setVisible(False)
        self.trace_hover_label = self._hover_label()
        self.voltage_plot = pg.PlotWidget()
        self.recovery_plot = pg.PlotWidget()
        self.voltage_plot.addLegend(offset=(8, 8))
        self.voltage_plot.setDownsampling(auto=True, mode="peak")
        self.recovery_plot.setDownsampling(auto=True, mode="peak")
        self.recovery_plot.setXLink(self.voltage_plot)
        trace_layout.addWidget(self.trace_metrics_label)
        trace_layout.addWidget(self.trace_method_bar)
        trace_layout.addWidget(self.trace_hover_label)
        trace_layout.addWidget(self.voltage_plot, 2)
        trace_layout.addWidget(self.recovery_plot, 1)

        self.convergence_tab = QWidget()
        convergence_layout = QVBoxLayout(self.convergence_tab)
        self.convergence_hover_label = self._hover_label()
        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setLogMode(x=True, y=True)
        self.convergence_plot.addLegend(offset=(8, 8))
        convergence_layout.addWidget(self.convergence_hover_label)
        convergence_layout.addWidget(self.convergence_plot, 1)

        self.fi_tab = QWidget()
        fi_layout = QVBoxLayout(self.fi_tab)
        self.fi_hover_label = self._hover_label()
        self.fi_plot = pg.PlotWidget()
        self.fi_plot.addLegend(offset=(8, 8))
        fi_layout.addWidget(self.fi_hover_label)
        fi_layout.addWidget(self.fi_plot, 1)

        self.alpha_tab = QWidget()
        alpha_layout = QVBoxLayout(self.alpha_tab)
        self.alpha_hover_label = self._hover_label()
        self.alpha_plot = pg.PlotWidget()
        self.alpha_plot.addLegend(offset=(8, 8))
        alpha_layout.addWidget(self.alpha_hover_label)
        alpha_layout.addWidget(self.alpha_plot, 1)

        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        self.summary_table = QTableWidget(0, 8)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.output_path_label = QLabel()
        self.output_path_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_table, 1)
        summary_layout.addWidget(self.output_path_label)

        tabs = (
            (self.trace_tab, "single_neuron.trace"),
            (self.convergence_tab, "single_neuron.convergence"),
            (self.fi_tab, "single_neuron.fi"),
            (self.alpha_tab, "single_neuron.alpha"),
            (self.summary_tab, "single_neuron.summary"),
        )
        for tab, topic_id in tabs:
            index = self.result_tabs.addTab(tab, "")
            button = ContextHelpButton(topic_id, self._translator, self.result_tabs)
            self._help_buttons.append(button)
            self.result_tabs.tabBar().setTabButton(index, QTabBar.ButtonPosition.RightSide, button)

        for plot in (
            self.voltage_plot,
            self.recovery_plot,
            self.convergence_plot,
            self.fi_plot,
            self.alpha_plot,
        ):
            plot.showGrid(x=True, y=True, alpha=0.22)

        self._install_hover_handlers()

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

    @staticmethod
    def _configure_form(form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

    @staticmethod
    def _hover_label() -> QLabel:
        label = QLabel()
        label.setObjectName("plot_hover_info")
        label.setMinimumHeight(22)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setVisible(False)
        return label

    def _add_form_row(self, form: QFormLayout, key: str, editor: QWidget) -> None:
        label = ParameterHelpLabel(_PARAMETER_HELP_TOPICS[key], self._translator, self)
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
        self.neuron_group.setTitle(translator.tr("gui.single_neuron.groups.neuron"))
        self.simulation_group.setTitle(translator.tr("gui.single_neuron.groups.simulation"))
        self.ablation_group.setTitle(translator.tr("gui.single_neuron.groups.ablation"))

        for key, label in self._form_labels.items():
            label.setText(translator.tr(f"gui.single_neuron.params.{key}"))
            label.retranslate_help(translator)
        for button in self._help_buttons:
            button.retranslate(translator)
        self.trace_button.setText(translator.tr("gui.single_neuron.buttons.trace"))
        self.study_button.setText(translator.tr("gui.single_neuron.buttons.study"))
        self.cancel_button.setText(translator.tr("gui.single_neuron.buttons.cancel"))
        self.trace_button.setToolTip(translator.tr("gui.single_neuron.buttons.trace_tip"))
        self.study_button.setToolTip(translator.tr("gui.single_neuron.buttons.study_tip"))
        self.cancel_button.setToolTip(translator.tr("gui.single_neuron.buttons.cancel_tip"))
        self.trace_method_label.setText(translator.tr("gui.single_neuron.trace_visibility"))
        hover_hint = translator.tr("gui.single_neuron.hover_hint")
        self._hover_hint = hover_hint
        for label in (
            self.trace_hover_label,
            self.convergence_hover_label,
            self.fi_hover_label,
            self.alpha_hover_label,
        ):
            label.setText(hover_hint if self._help_mode else "")
            label.setVisible(self._help_mode)

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
        self._retranslate_summary_header()
        for method_id, checkbox in self._trace_method_checks.items():
            checkbox.setText(self._method_name(method_id))
        if self._thread is None:
            self.status_label.setText(translator.tr("gui.single_neuron.status.ready"))
        self._set_plot_labels()

    def _retranslate_summary_header(self) -> None:
        keys = (
            "method",
            "lte_order",
            "isi_order",
            "grid_order",
            "error",
            "drift",
            "cost",
            "status",
        )
        for column, key in enumerate(keys):
            item = QTableWidgetItem(self._translator.tr(f"gui.single_neuron.table.{key}"))
            if self._help_mode:
                item.setToolTip(self._translator.tr(f"gui.single_neuron.table_help.{key}"))
            self.summary_table.setHorizontalHeaderItem(column, item)

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
        self._display_trace_comparison(study)
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
        self._plot_traces(((config.method_id, trace),), config)

    def _display_trace_comparison(self, study: SingleNeuronStudy) -> None:
        traces = tuple((item.method_id, item.trace) for item in study.comparison_traces)
        if not traces:
            self._display_trace(study.trace, study.config)
            return
        self.trace_metrics_label.setText(
            self._translator.tr(
                "gui.single_neuron.comparison_context",
                count=len(traces),
                dt=self._format_number(study.config.dt_ms, 6),
                current=self._format_number(study.config.input_current, 6),
                reset=self._translator.tr(
                    f"gui.single_neuron.reset_modes.{study.config.reset_mode}"
                ),
            )
        )
        self._plot_traces(traces, study.config)

    def _plot_traces(
        self,
        traces: tuple[tuple[str, SingleNeuronTrace], ...],
        config: SingleNeuronConfig,
    ) -> None:
        self.voltage_plot.clear()
        self.recovery_plot.clear()
        self._trace_items.clear()
        voltage_series: list[dict[str, Any]] = []
        recovery_series: list[dict[str, Any]] = []
        for index, (method_id, trace) in enumerate(traces):
            color = _PLOT_COLORS[index % len(_PLOT_COLORS)]
            name = self._method_name(method_id)
            voltage_item = self.voltage_plot.plot(
                trace.time_ms,
                trace.voltage_mv,
                pen=pg.mkPen(color, width=1.5),
                name=name,
            )
            recovery_item = self.recovery_plot.plot(
                trace.time_ms,
                trace.recovery,
                pen=pg.mkPen(color, width=1.3),
            )
            items: list[Any] = [voltage_item, recovery_item]
            if trace.spike_times_ms.size:
                spike_item = self.voltage_plot.plot(
                    trace.spike_times_ms,
                    np.full(trace.spike_times_ms.shape, config.neuron.threshold_mv),
                    pen=None,
                    symbol="t1",
                    symbolSize=6,
                    symbolBrush=color,
                    symbolPen=None,
                )
                items.append(spike_item)
            self._trace_items[method_id] = items
            voltage_series.append(
                self._hover_entry(name, trace.time_ms, trace.voltage_mv, voltage_item)
            )
            recovery_series.append(
                self._hover_entry(name, trace.time_ms, trace.recovery, recovery_item)
            )
        self._set_hover_series(self.voltage_plot, voltage_series)
        self._set_hover_series(self.recovery_plot, recovery_series)
        self._rebuild_trace_method_controls(tuple(method_id for method_id, _ in traces))
        self._set_plot_labels()

    def _rebuild_trace_method_controls(self, method_ids: tuple[str, ...]) -> None:
        for checkbox in self._trace_method_checks.values():
            self.trace_method_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self._trace_method_checks.clear()
        insert_at = max(1, self.trace_method_layout.count() - 1)
        for method_id in method_ids:
            checkbox = QCheckBox(self._method_name(method_id))
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda visible, value=method_id: self._set_trace_method_visible(value, visible)
            )
            self.trace_method_layout.insertWidget(insert_at, checkbox)
            insert_at += 1
            self._trace_method_checks[method_id] = checkbox
        self.trace_method_bar.setVisible(len(method_ids) > 1)

    def _set_trace_method_visible(self, method_id: str, visible: bool) -> None:
        for item in self._trace_items.get(method_id, ()):  # curves and spike markers
            item.setVisible(bool(visible))

    def _display_convergence(self, study: SingleNeuronStudy) -> None:
        self.convergence_plot.clear()
        hover_series: list[dict[str, Any]] = []
        valid_series: list[tuple[np.ndarray, np.ndarray]] = []
        for index, series in enumerate(study.convergence.series):
            x = np.asarray([point.dt_ms for point in series.points], dtype=np.float64)
            y = np.asarray([point.period_error_ms for point in series.points], dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
            name = self._method_name(series.method_id)
            item = self.convergence_plot.plot(
                x[valid],
                y[valid],
                pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.8),
                symbol="o",
                symbolSize=6,
                name=name,
            )
            hover_series.append(
                self._hover_entry(name, x[valid], y[valid], item, log_x=True, log_y=True)
            )
            if np.count_nonzero(valid) >= 2:
                valid_series.append((x[valid], y[valid]))
        self._add_convergence_guides(valid_series)
        self._set_hover_series(self.convergence_plot, hover_series)
        self._set_plot_labels()

    def _add_convergence_guides(self, valid_series: list[tuple[np.ndarray, np.ndarray]]) -> None:
        if not valid_series:
            return
        all_x = np.concatenate([values[0] for values in valid_series])
        all_y = np.concatenate([values[1] for values in valid_series])
        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))
        if not (x_min > 0.0 and x_max > x_min):
            return
        near_fine = all_y[np.isclose(all_x, x_min, rtol=1e-9, atol=1e-15)]
        anchor_y = float(np.median(near_fine if near_fine.size else all_y))
        if not np.isfinite(anchor_y) or anchor_y <= 0.0:
            return
        guide_x = np.asarray([x_min, x_max], dtype=np.float64)
        first_order = anchor_y * (guide_x / x_min)
        second_order = 0.45 * anchor_y * (guide_x / x_min) ** 2
        self.convergence_plot.plot(
            guide_x,
            first_order,
            pen=pg.mkPen("#b8b8b8", width=1.2, style=Qt.PenStyle.DashLine),
            name=self._translator.tr("gui.single_neuron.legend.first_order"),
        )
        self.convergence_plot.plot(
            guide_x,
            second_order,
            pen=pg.mkPen("#888888", width=1.2, style=Qt.PenStyle.DotLine),
            name=self._translator.tr("gui.single_neuron.legend.second_order"),
        )

    def _display_fi_curves(self, study: SingleNeuronStudy) -> None:
        self.fi_plot.clear()
        hover_series: list[dict[str, Any]] = []
        for index, curve in enumerate(study.fi_curves):
            valid = ~curve.diverged & np.isfinite(curve.rates_hz)
            name = self._method_name(curve.method_id)
            item = self.fi_plot.plot(
                curve.currents[valid],
                curve.rates_hz[valid],
                pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.8),
                symbol="o",
                symbolSize=4,
                name=name,
            )
            hover_series.append(
                self._hover_entry(name, curve.currents[valid], curve.rates_hz[valid], item)
            )
        self._set_hover_series(self.fi_plot, hover_series)
        self._set_plot_labels()

    def _display_alpha_sweep(self, study: SingleNeuronStudy) -> None:
        self.alpha_plot.clear()
        alpha = np.asarray([point.alpha for point in study.alpha_sweep], dtype=np.float64)
        local = np.asarray([point.local_order for point in study.alpha_sweep], dtype=np.float64)
        period = np.asarray(
            [point.observed_period_order for point in study.alpha_sweep], dtype=np.float64
        )
        local_name = self._translator.tr("gui.single_neuron.legend.local_order")
        local_item = self.alpha_plot.plot(
            alpha,
            local,
            pen=pg.mkPen("#62a0ea", width=2),
            symbol="o",
            name=local_name,
        )
        period_name = self._translator.tr("gui.single_neuron.legend.period_order")
        period_item = self.alpha_plot.plot(
            alpha,
            period,
            pen=pg.mkPen("#57e389", width=1.7),
            symbol="s",
            name=period_name,
        )
        self.alpha_plot.addLine(y=1.0, pen=pg.mkPen("#777777", style=Qt.PenStyle.DotLine))
        self.alpha_plot.addLine(y=2.0, pen=pg.mkPen("#aaaaaa", style=Qt.PenStyle.DashLine))
        self.alpha_plot.addLine(x=0.5, pen=pg.mkPen("#f66151", style=Qt.PenStyle.DashLine))
        self._set_hover_series(
            self.alpha_plot,
            [
                self._hover_entry(local_name, alpha, local, local_item),
                self._hover_entry(period_name, alpha, period, period_item),
            ],
        )
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

    @staticmethod
    def _hover_entry(
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        item: Any,
        *,
        log_x: bool = False,
        log_y: bool = False,
    ) -> dict[str, Any]:
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if log_x:
            valid &= x_values > 0.0
        if log_y:
            valid &= y_values > 0.0
        x_values = x_values[valid]
        y_values = y_values[valid]
        if x_values.size > 1 and np.any(np.diff(x_values) < 0.0):
            order = np.argsort(x_values)
            x_values = x_values[order]
            y_values = y_values[order]
        display_x = np.log10(x_values) if log_x else x_values
        display_y = np.log10(y_values) if log_y else y_values
        return {
            "name": name,
            "x": x_values,
            "y": y_values,
            "display_x": display_x,
            "display_y": display_y,
            "item": item,
        }

    def _set_hover_series(self, plot: pg.PlotWidget, series: list[dict[str, Any]]) -> None:
        self._hover_series[id(plot)] = series

    def _install_hover_handlers(self) -> None:
        targets = (
            (self.voltage_plot, self.trace_hover_label),
            (self.recovery_plot, self.trace_hover_label),
            (self.convergence_plot, self.convergence_hover_label),
            (self.fi_plot, self.fi_hover_label),
            (self.alpha_plot, self.alpha_hover_label),
        )
        for plot, label in targets:
            self._mouse_proxies.append(
                pg.SignalProxy(
                    plot.scene().sigMouseMoved,
                    rateLimit=30,
                    slot=lambda event, target=plot, info=label: self._handle_plot_hover(
                        event, target, info
                    ),
                )
            )

    def _handle_plot_hover(
        self,
        event: Any,
        plot: pg.PlotWidget,
        label: QLabel,
    ) -> None:
        scene_pos = event[0] if isinstance(event, (tuple, list)) else event
        if not plot.sceneBoundingRect().contains(scene_pos):
            if not self._help_mode:
                label.setVisible(False)
            return
        view_pos = plot.plotItem.vb.mapSceneToView(scene_pos)
        best: tuple[float, dict[str, Any], int] | None = None
        for series in self._hover_series.get(id(plot), ()):  # user-visible data only
            item = series["item"]
            if not item.isVisible() or not series["x"].size:
                continue
            display_x = series["display_x"]
            insertion = int(np.searchsorted(display_x, view_pos.x()))
            for index in {max(0, insertion - 1), min(display_x.size - 1, insertion)}:
                point = plot.plotItem.vb.mapViewToScene(
                    QPointF(float(display_x[index]), float(series["display_y"][index]))
                )
                distance = (point.x() - scene_pos.x()) ** 2 + (point.y() - scene_pos.y()) ** 2
                if best is None or distance < best[0]:
                    best = (float(distance), series, index)
        if best is None or best[0] > 28.0**2:
            if not self._help_mode:
                label.setVisible(False)
            return
        _, series, index = best
        label.setText(
            self._translator.tr(
                "gui.single_neuron.hover_value",
                series=series["name"],
                x=self._format_number(float(series["x"][index]), 7),
                y=self._format_number(float(series["y"][index]), 7),
            )
        )
        label.setVisible(True)

    def set_help_mode(self, enabled: bool) -> None:
        self._help_mode = bool(enabled)
        for label in self._form_labels.values():
            label.set_help_mode(self._help_mode)
        for button in self._help_buttons:
            button.setVisible(self._help_mode)
        for label in (
            self.trace_hover_label,
            self.convergence_hover_label,
            self.fi_hover_label,
            self.alpha_hover_label,
        ):
            label.setText(self._hover_hint if self._help_mode else "")
            label.setVisible(self._help_mode)
        self._retranslate_summary_header()

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
