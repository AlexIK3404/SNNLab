from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
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
    PRESET_CURRENTS,
    NeuronParameters,
    SingleNeuronConfig,
    SingleNeuronStudy,
    SingleNeuronTrace,
    StudySelection,
    neuron_preset,
    observed_trace_arrays,
    simulate_single_neuron,
    trace_metrics,
)
from snnlab.gui.widgets.help_dialog import HelpDialog
from snnlab.gui.workers import SingleNeuronStudyWorker
from snnlab.runtime.single_neuron_io import load_single_neuron_study, save_single_neuron_study

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
    "comparison_dt": "single_neuron.comparison_dt",
    "current_range": "single_neuron.current_range",
    "alpha_values": "single_neuron.alpha_values",
    "runs": "single_neuron.runs",
    "seed": "single_neuron.seed",
    "noise_std": "single_neuron.noise_std",
    "noise_interval": "single_neuron.noise_interval",
    "comparison_s": "single_neuron.composition_s",
    "comparison_alpha": "single_neuron.composition_alpha",
    "comparison_reset": "single_neuron.reset",
    "implicit_rtol": "single_neuron.implicit_tolerance",
    "implicit_atol_v": "single_neuron.implicit_tolerance",
    "implicit_atol_u": "single_neuron.implicit_tolerance",
    "comparison_implicit_rtol": "single_neuron.implicit_tolerance",
    "comparison_implicit_atol_v": "single_neuron.implicit_tolerance",
    "comparison_implicit_atol_u": "single_neuron.implicit_tolerance",
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
        self._trace_method_actions: dict[str, QAction] = {}
        self._phase_items: dict[str, Any] = {}
        self._phase_method_actions: dict[str, QAction] = {}
        self._isi_items: dict[str, Any] = {}
        self._isi_method_actions: dict[str, QAction] = {}
        self._hover_series: dict[int, list[dict[str, Any]]] = {}
        self._mouse_proxies: list[Any] = []
        self._hover_hint = ""
        self._plot_export_buttons: list[QPushButton] = []
        self._export_buttons_by_plot: dict[pg.PlotWidget, QPushButton] = {}
        self._plot_dialog: QDialog | None = None
        self._view_buttons: dict[QToolButton, tuple[pg.PlotWidget, ...]] = {}
        self._phase_limits: tuple[float, float, float, float] | None = None

        self._build_ui()
        self._connect_actions()
        self._populate_static_choices()
        self._apply_preset("regular_spiking")
        self._update_method_parameter_state()
        self.retranslate(translator)
        self._set_running(False)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 8)
        root.setSpacing(4)

        self.workspace_help_button = ContextHelpButton(
            "single_neuron.workspace", self._translator, self
        )
        self._help_buttons.append(self.workspace_help_button)

        run_status_row = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.study_progress = QProgressBar()
        self.study_progress.setRange(0, 1)
        self.study_progress.setValue(0)
        self.study_progress.setMaximumWidth(360)
        self.study_progress.hide()
        run_status_row.addWidget(self.status_label)
        run_status_row.addWidget(self.workspace_help_button)
        self.output_path_label = QLabel()
        self.output_path_label.setMaximumWidth(370)
        self.output_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.output_path_label.hide()
        run_status_row.addWidget(self.output_path_label, 1)
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
        self.burn_in_spin = self._double_spin(0.0, 100_000.0, 50.0, 500.0, 1)
        self.dt_spin = self._double_spin(0.0001, 10.0, 0.025, 0.1, 5)
        self._add_form_row(simulation_form, "current", self.current_spin)
        self._add_form_row(simulation_form, "duration", self.duration_spin)
        self._add_form_row(simulation_form, "burn_in", self.burn_in_spin)
        self.settings_layout.addWidget(self.simulation_group)

        self.single_run_group = QGroupBox()
        single_form = QFormLayout(self.single_run_group)
        self._configure_form(single_form)
        self.method_combo = QComboBox()
        self.method_combo.view().setMinimumWidth(360)
        self.reset_combo = QComboBox()
        for combo in (self.method_combo, self.reset_combo):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(14)
        self._add_form_row(single_form, "dt", self.dt_spin)
        self.composition_s_spin = self._double_spin(0.0, 1.0, 0.05, 0.5, 3)
        self.alpha_spin = self._double_spin(0.0, 3.0, 0.05, 0.5, 3)
        self.implicit_rtol_spin = self._double_spin(1e-6, 0.1, 0.0001, 0.001, 6)
        self.implicit_atol_v_spin = self._double_spin(1e-4, 10.0, 0.01, 0.05, 4)
        self.implicit_atol_u_spin = self._double_spin(1e-4, 10.0, 0.005, 0.01, 4)
        self._add_form_row(single_form, "method", self.method_combo)
        self._add_form_row(single_form, "reset", self.reset_combo)
        self._add_form_row(single_form, "composition_s", self.composition_s_spin)
        self._add_form_row(single_form, "composition_alpha", self.alpha_spin)
        self._add_form_row(single_form, "implicit_rtol", self.implicit_rtol_spin)
        self._add_form_row(single_form, "implicit_atol_v", self.implicit_atol_v_spin)
        self._add_form_row(single_form, "implicit_atol_u", self.implicit_atol_u_spin)
        self.settings_layout.addWidget(self.single_run_group)

        self.ablation_group = QGroupBox()
        ablation_layout = QVBoxLayout(self.ablation_group)
        self.outputs_group = QGroupBox()
        outputs_layout = QVBoxLayout(self.outputs_group)
        self.outputs_help_button = ContextHelpButton(
            "single_neuron.outputs", self._translator, self.outputs_group
        )
        self._help_buttons.append(self.outputs_help_button)
        outputs_layout.addWidget(self.outputs_help_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.output_checks = {
            key: QCheckBox()
            for key in ("traces", "phase", "convergence", "fi", "alpha", "variability", "summary")
        }
        for key, checkbox in self.output_checks.items():
            checkbox.setChecked(key in ("traces", "convergence", "summary"))
            outputs_layout.addWidget(checkbox)
        ablation_layout.addWidget(self.outputs_group)
        self.methods_label = ParameterHelpLabel(
            _PARAMETER_HELP_TOPICS["methods"], self._translator, self
        )
        self._form_labels["methods"] = self.methods_label
        self.method_list = QListWidget()
        self.method_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.method_list.setMinimumHeight(155)
        self.method_list.setWordWrap(True)
        self.method_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.method_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dt_values_edit = QLineEdit(", ".join(str(value) for value in DEFAULT_DT_VALUES_MS))
        self.comparison_dt_spin = self._double_spin(0.0001, 10.0, 0.025, 0.1, 5)
        self.current_range_edit = QLineEdit("0, 35, 1")
        self.alpha_values_edit = QLineEdit(", ".join(str(value) for value in DEFAULT_ALPHA_VALUES))
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 100)
        self.runs_spin.setValue(1)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 101)
        self.noise_std_spin = self._double_spin(0.0, 100.0, 0.25, 0.0, 3)
        self.noise_interval_spin = self._double_spin(0.01, 1000.0, 1.0, 5.0, 2)
        self.comparison_s_spin = self._double_spin(0.0, 1.0, 0.05, 0.5, 3)
        self.comparison_alpha_spin = self._double_spin(0.0, 3.0, 0.05, 0.5, 3)
        self.comparison_implicit_rtol_spin = self._double_spin(1e-6, 0.1, 0.0001, 0.001, 6)
        self.comparison_implicit_atol_v_spin = self._double_spin(1e-4, 10.0, 0.01, 0.05, 4)
        self.comparison_implicit_atol_u_spin = self._double_spin(1e-4, 10.0, 0.005, 0.01, 4)
        ablation_layout.addWidget(self.methods_label)
        ablation_layout.addWidget(self.method_list)
        ablation_form = QFormLayout()
        self.ablation_form = ablation_form
        self._configure_form(ablation_form)
        ablation_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.comparison_reset_combo = QComboBox()
        self._add_form_row(ablation_form, "comparison_reset", self.comparison_reset_combo)
        self._add_form_row(ablation_form, "dt_values", self.dt_values_edit)
        self._add_form_row(ablation_form, "comparison_dt", self.comparison_dt_spin)
        self._add_form_row(ablation_form, "current_range", self.current_range_edit)
        self._add_form_row(ablation_form, "alpha_values", self.alpha_values_edit)
        for key, widget in (
            ("runs", self.runs_spin),
            ("seed", self.seed_spin),
            ("noise_std", self.noise_std_spin),
            ("noise_interval", self.noise_interval_spin),
        ):
            self._add_form_row(ablation_form, key, widget)
        self._add_form_row(ablation_form, "comparison_s", self.comparison_s_spin)
        self._add_form_row(ablation_form, "comparison_alpha", self.comparison_alpha_spin)
        self._add_form_row(
            ablation_form, "comparison_implicit_rtol", self.comparison_implicit_rtol_spin
        )
        self._add_form_row(
            ablation_form, "comparison_implicit_atol_v", self.comparison_implicit_atol_v_spin
        )
        self._add_form_row(
            ablation_form, "comparison_implicit_atol_u", self.comparison_implicit_atol_u_spin
        )
        ablation_layout.addLayout(ablation_form)
        self.settings_layout.addWidget(self.ablation_group)

        # MainWindow places these controls in the same top toolbar used by the
        # network workspace. Keeping the actions out of the scrolling form
        # makes starting a run possible from any scroll position.
        self.trace_button = QPushButton(self)
        self.study_button = QPushButton(self)
        self.open_button = QPushButton(self)
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
        self.trace_method_button = QToolButton(self.trace_tab)
        self.trace_method_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.trace_method_menu = QMenu(self.trace_method_button)
        self.trace_method_button.setMenu(self.trace_method_menu)
        self.trace_method_button.setVisible(False)
        self.trace_dt_combo = QComboBox(self.trace_tab)
        self.trace_dt_combo.setMinimumContentsLength(10)
        self.trace_dt_combo.setVisible(False)
        self.trace_hover_label = self._hover_label()
        self.voltage_plot = pg.PlotWidget()
        self.recovery_plot = pg.PlotWidget()
        self.voltage_plot.addLegend(offset=(8, 8))
        self.voltage_plot.setDownsampling(auto=True, mode="peak")
        self.recovery_plot.setDownsampling(auto=True, mode="peak")
        self.recovery_plot.setXLink(self.voltage_plot)
        trace_header = QHBoxLayout()
        trace_header.addWidget(self.trace_metrics_label, 1)
        trace_header.addWidget(self.trace_dt_combo)
        trace_header.addWidget(self.trace_method_button)
        trace_header.addWidget(self._view_button((self.voltage_plot, self.recovery_plot)))
        trace_header.addWidget(self._export_trace_button())
        trace_layout.addLayout(trace_header)
        trace_layout.addWidget(self.trace_hover_label)
        trace_layout.addWidget(self.voltage_plot, 2)
        trace_layout.addWidget(self.recovery_plot, 1)

        self.phase_tab = QWidget()
        phase_layout = QVBoxLayout(self.phase_tab)
        self.phase_hover_label = self._hover_label()
        self.phase_dt_combo = QComboBox(self.phase_tab)
        self.phase_dt_combo.setMinimumContentsLength(10)
        self.phase_method_button = QToolButton(self.phase_tab)
        self.phase_method_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.phase_method_menu = QMenu(self.phase_method_button)
        self.phase_method_button.setMenu(self.phase_method_menu)
        self.phase_method_button.setVisible(False)
        self.phase_plot = pg.PlotWidget()
        self.phase_plot.addLegend(offset=(8, 8))
        phase_layout.addWidget(self.phase_hover_label)
        phase_header = QHBoxLayout()
        phase_header.addStretch(1)
        phase_header.addWidget(self.phase_dt_combo)
        phase_header.addWidget(self.phase_method_button)
        phase_header.addWidget(self._view_button((self.phase_plot,)))
        phase_header.addWidget(self._export_button(self.phase_plot))
        phase_layout.addLayout(phase_header)
        phase_layout.addWidget(self.phase_plot, 1)

        self.isi_tab = QWidget()
        isi_layout = QVBoxLayout(self.isi_tab)
        self.isi_context_label = QLabel()
        self.isi_context_label.setWordWrap(True)
        self.isi_dt_combo = QComboBox(self.isi_tab)
        self.isi_dt_combo.setMinimumContentsLength(10)
        self.isi_method_button = QToolButton(self.isi_tab)
        self.isi_method_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.isi_method_menu = QMenu(self.isi_method_button)
        self.isi_method_button.setMenu(self.isi_method_menu)
        self.isi_method_button.setVisible(False)
        self.isi_plot = pg.PlotWidget()
        self.isi_plot.addLegend(offset=(8, 8))
        self.isi_hover_label = self._hover_label()
        isi_header = QHBoxLayout()
        isi_header.addWidget(self.isi_context_label, 1)
        isi_header.addWidget(self.isi_dt_combo)
        isi_header.addWidget(self.isi_method_button)
        isi_header.addWidget(self._view_button((self.isi_plot,)))
        isi_header.addWidget(self._export_button(self.isi_plot))
        isi_layout.addLayout(isi_header)
        isi_layout.addWidget(self.isi_hover_label)
        isi_layout.addWidget(self.isi_plot, 1)

        self.convergence_tab = QWidget()
        convergence_layout = QVBoxLayout(self.convergence_tab)
        self.convergence_hover_label = self._hover_label()
        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setLogMode(x=True, y=True)
        self.convergence_plot.addLegend(offset=(8, 8))
        convergence_layout.addWidget(self.convergence_hover_label)
        convergence_layout.addLayout(self._plot_toolbar(self.convergence_plot))
        convergence_layout.addWidget(self.convergence_plot, 1)

        self.fi_tab = QWidget()
        fi_layout = QVBoxLayout(self.fi_tab)
        self.fi_hover_label = self._hover_label()
        self.fi_plot = pg.PlotWidget()
        self.fi_plot.addLegend(offset=(8, 8))
        fi_layout.addWidget(self.fi_hover_label)
        fi_layout.addLayout(self._plot_toolbar(self.fi_plot))
        fi_layout.addWidget(self.fi_plot, 1)

        self.alpha_tab = QWidget()
        alpha_layout = QVBoxLayout(self.alpha_tab)
        self.alpha_hover_label = self._hover_label()
        self.alpha_plot = pg.PlotWidget()
        self.alpha_plot.addLegend(offset=(8, 8))
        alpha_layout.addWidget(self.alpha_hover_label)
        alpha_layout.addLayout(self._plot_toolbar(self.alpha_plot))
        alpha_layout.addWidget(self.alpha_plot, 1)

        self.variability_tab = QWidget()
        variability_layout = QVBoxLayout(self.variability_tab)
        self.variability_hover_label = self._hover_label()
        self.variability_plot = pg.PlotWidget()
        self.variability_plot.addLegend(offset=(8, 8))
        self.variability_summary_label = QLabel()
        self.variability_summary_label.setWordWrap(True)
        variability_layout.addWidget(self.variability_summary_label)
        variability_layout.addWidget(self.variability_hover_label)
        variability_layout.addLayout(self._plot_toolbar(self.variability_plot))
        variability_layout.addWidget(self.variability_plot, 1)

        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        summary_reference_row = QHBoxLayout()
        self.summary_reference_label = QLabel()
        self.summary_reference_label.setWordWrap(True)
        summary_reference_row.addWidget(self.summary_reference_label, 1)
        self.summary_reference_help = ContextHelpButton(
            "single_neuron.summary_reference", self._translator, self.summary_tab
        )
        self._help_buttons.append(self.summary_reference_help)
        summary_reference_row.addWidget(self.summary_reference_help)
        summary_layout.addLayout(summary_reference_row)
        self.summary_help_hint = QLabel()
        self.summary_help_hint.setWordWrap(True)
        self.summary_help_hint.setVisible(False)
        summary_layout.addWidget(self.summary_help_hint)
        self.summary_table = QTableWidget(0, 9)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.horizontalHeader().sectionClicked.connect(
            self._show_summary_column_help
        )
        summary_layout.addWidget(self.summary_table, 1)

        tabs = (
            (self.trace_tab, "single_neuron.trace"),
            (self.phase_tab, "single_neuron.phase"),
            (self.isi_tab, "single_neuron.isi_sequence"),
            (self.convergence_tab, "single_neuron.convergence"),
            (self.fi_tab, "single_neuron.fi"),
            (self.alpha_tab, "single_neuron.alpha"),
            (self.variability_tab, "single_neuron.variability"),
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
            self.phase_plot,
            self.isi_plot,
            self.convergence_plot,
            self.fi_plot,
            self.alpha_plot,
            self.variability_plot,
        ):
            plot.showGrid(x=True, y=True, alpha=0.22)
            self._install_plot_shortcuts(plot)

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

    def _export_button(self, plot: pg.PlotWidget) -> QPushButton:
        button = QPushButton(self)
        button.setEnabled(False)
        button.clicked.connect(lambda: self._export_plot(plot))
        self._plot_export_buttons.append(button)
        self._export_buttons_by_plot[plot] = button
        return button

    def _export_trace_button(self) -> QPushButton:
        button = QPushButton(self)
        button.setEnabled(False)
        menu = QMenu(button)
        menu.addAction("v(t)", lambda: self._export_plot(self.voltage_plot))
        menu.addAction("u(t)", lambda: self._export_plot(self.recovery_plot))
        button.setMenu(menu)
        self._plot_export_buttons.append(button)
        self._export_buttons_by_plot[self.voltage_plot] = button
        self._export_buttons_by_plot[self.recovery_plot] = button
        return button

    def _plot_toolbar(self, plot: pg.PlotWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._view_button((plot,)))
        row.addWidget(self._export_button(plot))
        return row

    def _view_button(self, plots: tuple[pg.PlotWidget, ...]) -> QToolButton:
        button = QToolButton(self)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(QMenu(button))
        self._view_buttons[button] = plots
        button.menu().aboutToShow.connect(lambda b=button, p=plots: self._rebuild_view_menu(b, p))
        self._rebuild_view_menu(button, plots)
        return button

    def _rebuild_view_menu(
        self, button: QToolButton, plots: tuple[pg.PlotWidget, ...]
    ) -> None:
        tr = self._translator.tr
        button.setText(tr("gui.single_neuron.view.button"))
        button.setToolTip(tr("gui.single_neuron.view.tip"))
        menu = button.menu()
        assert menu is not None
        menu.clear()
        for plot in plots:
            if len(plots) > 1:
                label = "v(t)" if plot is self.voltage_plot else "u(t)"
                plot_menu = menu.addMenu(label)
            else:
                plot_menu = menu
            selected_axes = plot.getViewBox().mouseEnabled()
            active_mode = (
                "both" if all(selected_axes) else "x" if selected_axes[0] else "y"
            )
            mode_group = QActionGroup(plot_menu)
            for mode in ("both", "x", "y"):
                action = plot_menu.addAction(tr(f"gui.single_neuron.view.{mode}"))
                action.setCheckable(True)
                action.setActionGroup(mode_group)
                action.setChecked(mode == active_mode)
                action.triggered.connect(
                    lambda _checked=False, p=plot, m=mode: self._set_plot_axis_mode(p, m)
                )
            plot_menu.addSeparator()
            drag_group = QActionGroup(plot_menu)
            for mode, constant in (("pan", pg.ViewBox.PanMode), ("rectangle", pg.ViewBox.RectMode)):
                action = plot_menu.addAction(tr(f"gui.single_neuron.view.{mode}"))
                action.setCheckable(True)
                action.setActionGroup(drag_group)
                action.setChecked(plot.getViewBox().state["mouseMode"] == constant)
                action.triggered.connect(
                    lambda _checked=False, p=plot, m=constant: p.getViewBox().setMouseMode(m)
                )
            plot_menu.addSeparator()
            reset = plot_menu.addAction(tr("gui.single_neuron.view.reset"))
            reset.triggered.connect(lambda _checked=False, p=plot: self._fit_plot(p))
            maximize = plot_menu.addAction(tr("gui.single_neuron.view.maximize"))
            maximize.triggered.connect(
                lambda _checked=False, p=plot: self._open_plot_dialog(p)
            )

    @staticmethod
    def _set_plot_axis_mode(plot: pg.PlotWidget, mode: str) -> None:
        box = plot.getViewBox()
        box.setMouseEnabled(x=mode != "y", y=mode != "x")

    def _install_plot_shortcuts(self, plot: pg.PlotWidget) -> None:
        """Keep navigation local to the focused plot, including an expanded plot."""
        actions = {
            "0": lambda: self._fit_plot(plot),
            "X": lambda: self._set_plot_axis_mode(plot, "x"),
            "Y": lambda: self._set_plot_axis_mode(plot, "y"),
            "B": lambda: self._set_plot_axis_mode(plot, "both"),
            "P": lambda: plot.getViewBox().setMouseMode(pg.ViewBox.PanMode),
            "Z": lambda: plot.getViewBox().setMouseMode(pg.ViewBox.RectMode),
            "F": lambda: self._open_plot_dialog(plot),
            "+": lambda: self._zoom_plot(plot, 0.8),
            "-": lambda: self._zoom_plot(plot, 1.25),
            "Ctrl+S": lambda: self._export_plot(plot),
        }
        for key, callback in actions.items():
            shortcut = QShortcut(QKeySequence(key), plot)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
        plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _fit_plot(self, plot: pg.PlotWidget) -> None:
        if plot is self.phase_plot and self._phase_limits is not None:
            x0, x1, y0, y1 = self._phase_limits
            plot.setXRange(x0, x1, padding=0)
            plot.setYRange(y0, y1, padding=0)
        else:
            plot.getViewBox().autoRange()

    @staticmethod
    def _zoom_plot(plot: pg.PlotWidget, factor: float) -> None:
        box = plot.getViewBox()
        x_enabled, y_enabled = box.mouseEnabled()
        box.scaleBy((factor if x_enabled else 1.0, factor if y_enabled else 1.0))

    def _open_plot_dialog(self, plot: pg.PlotWidget) -> None:
        if self._plot_dialog is not None:
            if plot.window() is self._plot_dialog:
                self._plot_dialog.close()
                return
            self._plot_dialog.close()
        parent = plot.parentWidget()
        if parent is None or parent.layout() is None:
            return
        original_layout = parent.layout()
        index = original_layout.indexOf(plot)
        if index < 0:
            return
        stretch = original_layout.stretch(index)
        original_layout.removeWidget(plot)

        dialog = QDialog(self)
        dialog.setWindowTitle(self._translator.tr("gui.single_neuron.view.window"))
        dialog_layout = QVBoxLayout(dialog)
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        toolbar.addWidget(self._view_button((plot,)))
        save = QPushButton(self._translator.tr("gui.single_neuron.export.button"), dialog)
        save.setEnabled(self._export_buttons_by_plot[plot].isEnabled())
        save.clicked.connect(lambda: self._export_plot(plot))
        toolbar.addWidget(save)
        close = QPushButton(self._translator.tr("gui.single_neuron.view.close"), dialog)
        close.clicked.connect(dialog.close)
        toolbar.addWidget(close)
        dialog_layout.addLayout(toolbar)
        dialog_layout.addWidget(plot, 1)
        dialog.finished.connect(
            lambda _result: self._restore_plot_dialog(
                dialog, plot, original_layout, index, stretch
            )
        )
        self._plot_dialog = dialog
        dialog.showMaximized()

    def _restore_plot_dialog(
        self, dialog: QDialog, plot: pg.PlotWidget, layout: QVBoxLayout, index: int, stretch: int
    ) -> None:
        dialog.layout().removeWidget(plot)
        layout.insertWidget(index, plot, stretch)
        plot.show()
        if self._plot_dialog is dialog:
            self._plot_dialog = None
        for button in tuple(self._view_buttons):
            if button.parentWidget() is dialog:
                self._view_buttons.pop(button)
        dialog.deleteLater()

    def _export_plot(self, plot: pg.PlotWidget) -> None:
        name = {
            self.voltage_plot: "voltage",
            self.recovery_plot: "recovery",
            self.phase_plot: "phase",
            self.convergence_plot: "convergence",
            self.fi_plot: "fi",
            self.alpha_plot: "alpha",
            self.variability_plot: "variability",
        }[plot]
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            self._translator.tr("gui.single_neuron.export.title"),
            f"{name}.png",
            "PNG (*.png);;SVG (*.svg)",
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() not in (".png", ".svg"):
            target = target.with_suffix(".svg" if "SVG" in selected_filter else ".png")
        try:
            if target.suffix.lower() == ".svg":
                SVGExporter(plot.plotItem).export(str(target))
            else:
                exporter = ImageExporter(plot.plotItem)
                exporter.parameters()["width"] = 1600
                exporter.export(str(target))
        except Exception as exc:  # noqa: BLE001 - GUI export boundary.
            QMessageBox.critical(self, self._translator.tr("gui.dialogs.error_title"), str(exc))

    def _enable_exports(self, *plots: pg.PlotWidget) -> None:
        for plot in plots:
            self._export_buttons_by_plot[plot].setEnabled(True)

    def _add_form_row(self, form: QFormLayout, key: str, editor: QWidget) -> None:
        label = ParameterHelpLabel(_PARAMETER_HELP_TOPICS[key], self._translator, self)
        self._form_labels[key] = label
        form.addRow(label, editor)

    def _connect_actions(self) -> None:
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.method_combo.currentIndexChanged.connect(self._update_method_parameter_state)
        self.method_list.itemSelectionChanged.connect(self._update_method_parameter_state)
        for checkbox in self.output_checks.values():
            checkbox.toggled.connect(self._update_study_controls)
        self.noise_std_spin.valueChanged.connect(self._update_study_controls)
        self.trace_button.clicked.connect(self.run_trace)
        self.study_button.clicked.connect(self.run_study)
        self.open_button.clicked.connect(self.open_study)
        self.cancel_button.clicked.connect(self.cancel_study)
        self.trace_dt_combo.currentIndexChanged.connect(self._show_selected_trace_dt)
        self.phase_dt_combo.currentIndexChanged.connect(self._show_selected_phase_dt)
        self.isi_dt_combo.currentIndexChanged.connect(self._show_selected_isi_dt)

    def _populate_static_choices(self) -> None:
        self.preset_combo.clear()
        for preset_id in NEURON_PRESETS:
            self.preset_combo.addItem(preset_id, preset_id)
        self.preset_combo.addItem("custom", "custom")

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
        self._set_combo_data(self.reset_combo, "grid")
        self.comparison_reset_combo.clear()
        self.comparison_reset_combo.addItem("grid", "grid")
        self.comparison_reset_combo.addItem("event", "event")

    def _preset_changed(self) -> None:
        preset_id = self.preset_combo.currentData()
        if preset_id in NEURON_PRESETS:
            self._apply_preset(str(preset_id))

    def _apply_preset(self, preset_id: str) -> None:
        parameters = neuron_preset(preset_id)
        self.a_spin.setValue(parameters.a)
        self.b_spin.setValue(parameters.b)
        self.c_spin.setValue(parameters.c)
        self.d_spin.setValue(parameters.d)
        self.threshold_spin.setValue(parameters.threshold_mv)
        if preset_id in PRESET_CURRENTS:
            self.current_spin.setValue(PRESET_CURRENTS[preset_id])

    def _update_method_parameter_state(self) -> None:
        method_id = str(self.method_combo.currentData() or "")
        if method_id:
            self.method_combo.setToolTip(self._method_name(method_id))
        for key, editor, relevant in (
            ("composition_s", self.composition_s_spin, "euler_composition_real"),
            ("composition_alpha", self.alpha_spin, "euler_composition_complex"),
        ):
            visible = method_id == relevant
            self.single_run_group.layout().setRowVisible(self._form_labels[key], visible)
            editor.setEnabled(visible and not self.is_running())
        selected = {
            str(item.data(Qt.ItemDataRole.UserRole)) for item in self.method_list.selectedItems()
        }
        implicit_visible = method_id == "implicit_euler"
        for key, editor in (
            ("implicit_rtol", self.implicit_rtol_spin),
            ("implicit_atol_v", self.implicit_atol_v_spin),
            ("implicit_atol_u", self.implicit_atol_u_spin),
        ):
            self.single_run_group.layout().setRowVisible(
                self._form_labels[key], implicit_visible
            )
            editor.setEnabled(implicit_visible and not self.is_running())
        comparing = any(
            self.output_checks[key].isChecked()
            for key in ("traces", "phase", "convergence", "fi", "summary", "variability")
        )
        for key, method in (
            ("comparison_s", "euler_composition_real"),
            ("comparison_alpha", "euler_composition_complex"),
        ):
            self.ablation_form.setRowVisible(
                self._form_labels[key], comparing and method in selected
            )
        for key in (
            "comparison_implicit_rtol",
            "comparison_implicit_atol_v",
            "comparison_implicit_atol_u",
        ):
            self.ablation_form.setRowVisible(
                self._form_labels[key], comparing and "implicit_euler" in selected
            )

    def _selection(self) -> StudySelection:
        selected = {key: check.isChecked() for key, check in self.output_checks.items()}
        if not any(selected.values()):
            raise ValueError(self._translator.tr("gui.single_neuron.errors.no_outputs"))
        return StudySelection(**selected)

    def _update_study_controls(self) -> None:
        checks = self.output_checks
        need_methods = any(
            checks[key].isChecked()
            for key in ("traces", "phase", "convergence", "fi", "summary", "variability")
        )
        self.methods_label.setVisible(need_methods)
        self.method_list.setVisible(need_methods)
        for key, needed in (
            (
                "dt_values",
                any(checks[item].isChecked() for item in ("traces", "phase", "convergence", "summary", "alpha")),
            ),
            ("current_range", checks["fi"].isChecked()),
            ("comparison_dt", checks["fi"].isChecked() or checks["variability"].isChecked()),
            ("alpha_values", checks["alpha"].isChecked()),
        ):
            self.ablation_form.setRowVisible(self._form_labels[key], needed)
        self.ablation_form.setRowVisible(
            self._form_labels["noise_interval"], self.noise_std_spin.value() > 0.0
        )
        self.runs_spin.setEnabled(
            not self.is_running()
            and self.noise_std_spin.value() > 0.0
            and (checks["fi"].isChecked() or checks["variability"].isChecked())
        )
        if (
            checks["variability"].isChecked()
            and self.noise_std_spin.value() > 0.0
            and self.runs_spin.value() < 2
        ):
            self.runs_spin.setValue(3)
        self._update_method_parameter_state()

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
        self.neuron_group.setTitle(translator.tr("gui.single_neuron.groups.neuron"))
        self.simulation_group.setTitle(translator.tr("gui.single_neuron.groups.simulation"))
        self.single_run_group.setTitle(translator.tr("gui.single_neuron.groups.single_run"))
        self.ablation_group.setTitle(translator.tr("gui.single_neuron.groups.ablation"))
        self.outputs_group.setTitle(translator.tr("gui.single_neuron.groups.outputs"))
        for key, checkbox in self.output_checks.items():
            checkbox.setText(translator.tr(f"gui.single_neuron.outputs.{key}"))

        for key, label in self._form_labels.items():
            label.setText(translator.tr(f"gui.single_neuron.params.{key}"))
            label.retranslate_help(translator)
        for button in self._help_buttons:
            button.retranslate(translator)
        self.summary_help_hint.setText(translator.tr("gui.single_neuron.summary_help_hint"))
        for button in self._plot_export_buttons:
            button.setText(translator.tr("gui.single_neuron.export.button"))
        for button, plots in self._view_buttons.items():
            self._rebuild_view_menu(button, plots)
        self.trace_button.setText(translator.tr("gui.single_neuron.buttons.trace"))
        self.study_button.setText(translator.tr("gui.single_neuron.buttons.study"))
        self.open_button.setText(translator.tr("gui.single_neuron.buttons.open"))
        self.cancel_button.setText(translator.tr("gui.single_neuron.buttons.cancel"))
        self.trace_button.setToolTip(translator.tr("gui.single_neuron.buttons.trace_tip"))
        self.study_button.setToolTip(translator.tr("gui.single_neuron.buttons.study_tip"))
        self.open_button.setToolTip(translator.tr("gui.single_neuron.buttons.open_tip"))
        self.cancel_button.setToolTip(translator.tr("gui.single_neuron.buttons.cancel_tip"))
        self.trace_method_button.setText(translator.tr("gui.single_neuron.trace_visibility"))
        self.phase_method_button.setText(translator.tr("gui.single_neuron.trace_visibility"))
        self.isi_method_button.setText(translator.tr("gui.single_neuron.trace_visibility"))
        hover_hint = translator.tr("gui.single_neuron.hover_hint")
        self._hover_hint = hover_hint
        for label in (
            self.trace_hover_label,
            self.phase_hover_label,
            self.isi_hover_label,
            self.convergence_hover_label,
            self.fi_hover_label,
            self.alpha_hover_label,
            self.variability_hover_label,
        ):
            label.setText(hover_hint if self._help_mode else "")
            label.setVisible(self._help_mode)

        self._retranslate_combo(
            self.preset_combo,
            lambda value: translator.tr(f"gui.single_neuron.presets.{value}"),
        )
        self._retranslate_combo(self.method_combo, self._method_name)
        self.method_combo.setToolTip(self._method_name(str(self.method_combo.currentData())))
        self._retranslate_combo(
            self.reset_combo,
            lambda value: translator.tr(f"gui.single_neuron.reset_modes.{value}"),
        )
        self._retranslate_combo(
            self.comparison_reset_combo,
            lambda value: translator.tr(f"gui.single_neuron.reset_modes.{value}"),
        )
        for index in range(self.method_list.count()):
            item = self.method_list.item(index)
            method_id = str(item.data(Qt.ItemDataRole.UserRole))
            item.setText(self._method_name(method_id))
            item.setToolTip(self._method_name(method_id))

        tab_keys = ("trace", "phase", "isi_sequence", "convergence", "fi", "alpha", "variability", "summary")
        for index, key in enumerate(tab_keys):
            self.result_tabs.setTabText(index, translator.tr(f"gui.single_neuron.tabs.{key}"))
        self._retranslate_summary_header()
        for method_id, action in self._trace_method_actions.items():
            action.setText(self._method_name(method_id))
            action.setToolTip(self._method_name(method_id))
        for method_id, action in self._phase_method_actions.items():
            action.setText(self._method_name(method_id))
            action.setToolTip(self._method_name(method_id))
        for method_id, action in self._isi_method_actions.items():
            action.setText(self._method_name(method_id))
            action.setToolTip(self._method_name(method_id))
        for index in range(self.trace_dt_combo.count()):
            self.trace_dt_combo.setItemText(
                index,
                translator.tr(
                    "gui.single_neuron.trace_dt_option",
                    dt=f"{float(self.trace_dt_combo.itemData(index)):g}",
                ),
            )
        for index in range(self.phase_dt_combo.count()):
            self.phase_dt_combo.setItemText(
                index,
                translator.tr(
                    "gui.single_neuron.trace_dt_option",
                    dt=f"{float(self.phase_dt_combo.itemData(index)):g}",
                ),
            )
        for index in range(self.isi_dt_combo.count()):
            self.isi_dt_combo.setItemText(
                index,
                translator.tr(
                    "gui.single_neuron.trace_dt_option",
                    dt=f"{float(self.isi_dt_combo.itemData(index)):g}",
                ),
            )
        if self._last_study is not None and self._last_study.selection.traces:
            self._show_selected_trace_dt()
            self._show_selected_isi_dt()
        if self._last_study is not None and self._last_study.selection.phase:
            self._show_selected_phase_dt()
        if self._thread is None:
            self.status_label.setText(translator.tr("gui.single_neuron.status.ready"))
        if self._last_study is not None and self._last_study.selection.summary:
            self._set_summary_reference(self._last_study)
        self._set_plot_labels()

    def _summary_column_keys(self) -> tuple[str, ...]:
        primary_reset = (
            self._last_study.convergence.reset_mode
            if self._last_study is not None and self._last_study.convergence is not None
            else str(self.reset_combo.currentData() or "grid")
        )
        control_reset = "event" if primary_reset == "grid" else "grid"
        return (
            "method",
            "lte_order",
            f"isi_{primary_reset}",
            f"isi_{control_reset}",
            "spike_counts",
            "error",
            "drift",
            "cost",
            "status",
        )

    def _retranslate_summary_header(self) -> None:
        keys = self._summary_column_keys()
        for column, key in enumerate(keys):
            item = QTableWidgetItem(self._translator.tr(f"gui.single_neuron.table.{key}"))
            if self._help_mode:
                item.setToolTip(self._translator.tr(f"gui.single_neuron.table_help.{key}"))
            self.summary_table.setHorizontalHeaderItem(column, item)
        for row in range(self.summary_table.rowCount()):
            for column, key in enumerate(keys):
                item = self.summary_table.item(row, column)
                if item is not None:
                    item.setToolTip(
                        self._translator.tr(f"gui.single_neuron.table_help.{key}")
                        if self._help_mode
                        else ""
                    )

    def _show_summary_column_help(self, column: int) -> None:
        if not self._help_mode:
            return
        keys = self._summary_column_keys()
        if not 0 <= column < len(keys):
            return
        key = keys[column]
        HelpDialog(
            {
                "title": self._translator.tr(f"gui.single_neuron.table.{key}"),
                "long": self._translator.tr(f"gui.single_neuron.table_help.{key}"),
            },
            parent=self.summary_table,
        ).exec()

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
        self.phase_plot.setLabel("bottom", tr("gui.single_neuron.axes.voltage_mv"))
        self.phase_plot.setLabel("left", tr("gui.single_neuron.axes.recovery"))
        self.isi_plot.setLabel("bottom", tr("gui.single_neuron.axes.isi_index"))
        self.isi_plot.setLabel("left", tr("gui.single_neuron.axes.isi_deviation_ms"))
        self.variability_plot.setLabel("bottom", tr("gui.single_neuron.axes.seed"))
        self.variability_plot.setLabel("left", tr("gui.single_neuron.axes.rate_hz"))

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
            implicit_rtol=self.implicit_rtol_spin.value(),
            implicit_atol_v=self.implicit_atol_v_spin.value(),
            implicit_atol_u=self.implicit_atol_u_spin.value(),
            seed=self.seed_spin.value(),
            noise_std=self.noise_std_spin.value(),
            noise_interval_ms=self.noise_interval_spin.value(),
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

    def _study_axes(
        self, selection: StudySelection
    ) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        dt_values: tuple[float, ...] = ()
        currents: tuple[float, ...] = ()
        alpha_values: tuple[float, ...] = ()
        if selection.traces or selection.phase or selection.convergence or selection.summary or selection.alpha:
            dt_values = self._parse_values(self.dt_values_edit.text(), field="dt", maximum_count=12)
            minimum = 2 if selection.convergence or selection.summary or selection.alpha else 1
            if len(set(dt_values)) < minimum or any(value <= 0.0 for value in dt_values):
                raise ValueError(f"dt: enter at least {minimum} distinct positive value(s)")
        if selection.fi:
            current_range = self._parse_values(
                self.current_range_edit.text(), field="I range", maximum_count=3
            )
            if len(current_range) != 3:
                raise ValueError("I range: enter start, stop, step")
            start, stop, step = current_range
            if step <= 0.0 or stop < start:
                raise ValueError("I range: require step > 0 and stop >= start")
            currents = tuple(np.arange(start, stop + 0.5 * step, step, dtype=np.float64))
            if len(currents) > 200:
                raise ValueError("I range: at most 200 current samples are allowed")
        if selection.alpha:
            alpha_values = self._parse_values(
                self.alpha_values_edit.text(), field="alpha", maximum_count=20
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
            study = SingleNeuronStudy(
                config=config,
                trace=trace,
                metrics=trace_metrics(trace, config.burn_in_ms),
                convergence=None,
                reset_control=None,
                fi_curves=(),
                alpha_sweep=(),
                selection=StudySelection(
                    traces=True,
                    convergence=False,
                    fi=False,
                    alpha=False,
                    summary=False,
                    phase=self.output_checks["phase"].isChecked(),
                ),
                trace_config=config,
            )
            run_dir = save_single_neuron_study(study)
        except Exception as exc:  # noqa: BLE001 - GUI validation boundary.
            QMessageBox.critical(self, self._translator.tr("gui.dialogs.error_title"), str(exc))
            return
        self._on_study_ready(study, str(run_dir))
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.trace_done"))

    def open_study(self) -> None:
        if self.is_running():
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self._translator.tr("gui.single_neuron.buttons.open_title"),
            str(Path("runs/gui/single_neuron").resolve()),
            "study.json (study.json)",
        )
        if not filename:
            return
        try:
            study = load_single_neuron_study(filename)
            self._restore_study_controls(study)
            self._on_study_ready(study, str(Path(filename).parent))
        except Exception as exc:  # noqa: BLE001 - Untrusted saved-study boundary.
            QMessageBox.critical(self, self._translator.tr("gui.dialogs.error_title"), str(exc))
            return
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.loaded"))

    def _restore_study_controls(self, study: SingleNeuronStudy) -> None:
        config = study.config
        phase_config = study.trace_config or config
        preset = next(
            (name for name, parameters in NEURON_PRESETS.items() if parameters == config.neuron),
            "custom",
        )
        self.preset_combo.blockSignals(True)
        self._set_combo_data(self.preset_combo, preset)
        self.preset_combo.blockSignals(False)
        for spin, value in (
            (self.a_spin, config.neuron.a),
            (self.b_spin, config.neuron.b),
            (self.c_spin, config.neuron.c),
            (self.d_spin, config.neuron.d),
            (self.threshold_spin, config.neuron.threshold_mv),
            (self.current_spin, config.input_current),
            (self.duration_spin, config.duration_ms),
            (self.burn_in_spin, config.burn_in_ms),
            (self.dt_spin, phase_config.dt_ms),
            (self.comparison_dt_spin, config.dt_ms),
            (self.composition_s_spin, phase_config.composition_s),
            (self.alpha_spin, phase_config.composition_alpha),
            (self.comparison_s_spin, config.composition_s),
            (self.comparison_alpha_spin, config.composition_alpha),
            (self.implicit_rtol_spin, phase_config.implicit_rtol),
            (self.implicit_atol_v_spin, phase_config.implicit_atol_v),
            (self.implicit_atol_u_spin, phase_config.implicit_atol_u),
            (self.comparison_implicit_rtol_spin, config.implicit_rtol),
            (self.comparison_implicit_atol_v_spin, config.implicit_atol_v),
            (self.comparison_implicit_atol_u_spin, config.implicit_atol_u),
            (self.noise_std_spin, config.noise_std),
            (self.noise_interval_spin, config.noise_interval_ms),
        ):
            spin.setValue(value)
        self.seed_spin.setValue(config.seed)
        self.runs_spin.setValue(study.n_runs)
        self._set_combo_data(self.method_combo, phase_config.method_id)
        self._set_combo_data(self.reset_combo, phase_config.reset_mode)
        self._set_combo_data(self.comparison_reset_combo, config.reset_mode)
        values = study.dt_values_ms or study.trace_dt_values_ms
        if values:
            self.dt_values_edit.setText(", ".join(f"{value:g}" for value in values))
        if study.alpha_values:
            self.alpha_values_edit.setText(", ".join(f"{value:g}" for value in study.alpha_values))
        if len(study.currents) > 1:
            spacing = np.diff(study.currents)
            if np.allclose(spacing, spacing[0]):
                self.current_range_edit.setText(
                    f"{study.currents[0]:g}, {study.currents[-1]:g}, {spacing[0]:g}"
                )
        methods = {item.method_id for item in study.comparison_traces}
        if not methods and study.convergence is not None:
            methods = {series.method_id for series in study.convergence.series}
        if not methods:
            methods = {curve.method_id for curve in study.fi_curves}
        for index in range(self.method_list.count()):
            item = self.method_list.item(index)
            item.setSelected(item.data(Qt.ItemDataRole.UserRole) in methods)
        for key, check in self.output_checks.items():
            check.setChecked(getattr(study.selection, key))
        self._update_study_controls()

    def run_study(self) -> None:
        if self.is_running():
            return
        try:
            phase_config = self._configuration()
            selection = self._selection()
            config = replace(
                phase_config,
                dt_ms=self.comparison_dt_spin.value(),
                reset_mode=str(self.comparison_reset_combo.currentData()),
                composition_s=self.comparison_s_spin.value(),
                composition_alpha=self.comparison_alpha_spin.value(),
                implicit_rtol=self.comparison_implicit_rtol_spin.value(),
                implicit_atol_v=self.comparison_implicit_atol_v_spin.value(),
                implicit_atol_u=self.comparison_implicit_atol_u_spin.value(),
            )
            method_ids = (
                self._selected_methods()
                if (
                    selection.traces
                    or selection.phase
                    or selection.convergence
                    or selection.fi
                    or selection.summary
                    or selection.variability
                )
                else ()
            )
            dt_values, currents, alpha_values = self._study_axes(selection)
            n_runs = (
                max(self.runs_spin.value(), 2 if selection.variability else 1)
                if (selection.fi or selection.variability) and config.noise_std > 0.0
                else 1
            )
            if selection.variability and config.noise_std == 0.0:
                raise ValueError(
                    self._translator.tr("gui.single_neuron.errors.variability_requires_noise")
                )
        except Exception as exc:  # noqa: BLE001 - GUI validation boundary.
            QMessageBox.critical(self, self._translator.tr("gui.dialogs.error_title"), str(exc))
            return

        thread = QThread(self)
        worker = SingleNeuronStudyWorker(
            config=config,
            phase_config=phase_config if selection.phase else None,
            method_ids=method_ids,
            dt_values_ms=dt_values,
            trace_dt_values_ms=dt_values if selection.traces or selection.phase else (),
            currents=currents,
            alpha_values=alpha_values,
            selection=selection,
            n_runs=n_runs,
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
        self.output_path_label.hide()
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.running"))
        self.study_progress.show()
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
        selected = study.selection
        for index, key in enumerate(
            ("traces", "phase", "traces", "convergence", "fi", "alpha", "variability", "summary")
        ):
            self.result_tabs.setTabVisible(index, getattr(selected, key))
        if selected.traces:
            self._display_trace_comparison(study)
            self._display_isi_sequence(study)
        else:
            self.trace_dt_combo.setVisible(False)
        if selected.phase:
            self._display_phase_study(study)
        if selected.convergence:
            self._display_convergence(study)
        if selected.fi:
            self._display_fi_curves(study)
        if selected.alpha:
            self._display_alpha_sweep(study)
        if selected.variability:
            self._display_variability(study)
        if selected.summary:
            self._display_summary(study)
        self.output_path_label.setText(
            self._translator.tr("gui.single_neuron.output_path", path=f"…/{Path(run_dir).name}")
        )
        self.output_path_label.setToolTip(run_dir)
        self.output_path_label.show()
        self.study_progress.hide()
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.completed"))
        self.result_tabs.setCurrentWidget(
            self.summary_tab
            if selected.summary
            else next(
                tab
                for key, tab in (
                    ("traces", self.trace_tab),
                    ("phase", self.phase_tab),
                    ("convergence", self.convergence_tab),
                    ("fi", self.fi_tab),
                    ("alpha", self.alpha_tab),
                    ("variability", self.variability_tab),
                )
                if getattr(selected, key)
            )
        )

    def _on_study_failed(self, traceback_text: str) -> None:
        self.study_progress.hide()
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.error"))
        QMessageBox.critical(
            self,
            self._translator.tr("gui.dialogs.error_title"),
            traceback_text,
        )

    def _on_study_cancelled(self) -> None:
        self.study_progress.hide()
        self.status_label.setText(self._translator.tr("gui.single_neuron.status.cancelled"))

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._set_running(False)

    def _set_running(self, running: bool) -> None:
        self.trace_button.setEnabled(not running)
        self.study_button.setEnabled(not running)
        self.open_button.setEnabled(not running)
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
            self.comparison_dt_spin,
            self.method_combo,
            self.reset_combo,
            self.comparison_reset_combo,
            self.composition_s_spin,
            self.alpha_spin,
            self.implicit_rtol_spin,
            self.implicit_atol_v_spin,
            self.implicit_atol_u_spin,
            self.method_list,
            self.dt_values_edit,
            self.current_range_edit,
            self.alpha_values_edit,
            self.runs_spin,
            self.seed_spin,
            self.noise_std_spin,
            self.noise_interval_spin,
            self.comparison_s_spin,
            self.comparison_alpha_spin,
            self.comparison_implicit_rtol_spin,
            self.comparison_implicit_atol_v_spin,
            self.comparison_implicit_atol_u_spin,
            *self.output_checks.values(),
        ):
            widget.setEnabled(not running)
        if not running:
            self._update_study_controls()
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
        if config.method_id == "implicit_euler" and trace.simulated_steps:
            self.trace_metrics_label.setText(
                self.trace_metrics_label.text() + " &nbsp; " + tr(
                    "gui.single_neuron.implicit_work",
                    attempts=self._format_number(
                        trace.implicit_step_attempts / trace.simulated_steps, 4
                    ),
                )
            )
        self._plot_traces(((config.method_id, trace),), config)

    def _display_trace_comparison(self, study: SingleNeuronStudy) -> None:
        traces = tuple((item.method_id, item.trace) for item in study.comparison_traces)
        if not traces:
            self.trace_dt_combo.setVisible(False)
            self._display_trace(study.trace, study.config)
            return
        dt_values = tuple(dict.fromkeys(
            item.dt_ms if item.dt_ms is not None else study.config.dt_ms
            for item in study.comparison_traces
        ))
        self.trace_dt_combo.blockSignals(True)
        self.trace_dt_combo.clear()
        for dt_ms in dt_values:
            self.trace_dt_combo.addItem(
                self._translator.tr("gui.single_neuron.trace_dt_option", dt=f"{dt_ms:g}"),
                dt_ms,
            )
        target = study.config.dt_ms if study.config.dt_ms in dt_values else min(dt_values)
        self._set_combo_data(self.trace_dt_combo, target)
        self.trace_dt_combo.blockSignals(False)
        self.trace_dt_combo.setVisible(len(dt_values) > 1)
        self._show_selected_trace_dt()

    def _show_selected_trace_dt(self) -> None:
        study = self._last_study
        if study is None or not study.comparison_traces:
            return
        dt_ms = float(self.trace_dt_combo.currentData() or study.config.dt_ms)
        traces = tuple(
            (item.method_id, item.trace)
            for item in study.comparison_traces
            if (item.dt_ms if item.dt_ms is not None else study.config.dt_ms) == dt_ms
        )
        self.trace_metrics_label.setText(
            self._translator.tr(
                "gui.single_neuron.comparison_context",
                count=len(traces),
                dt=self._format_number(dt_ms, 6),
                current=self._format_number(study.config.input_current, 6),
                reset=self._translator.tr(
                    f"gui.single_neuron.reset_modes.{study.config.reset_mode}"
                ),
            )
        )
        self._plot_traces(traces, study.config)

    def _display_isi_sequence(self, study: SingleNeuronStudy) -> None:
        values = tuple(dict.fromkeys(
            item.dt_ms if item.dt_ms is not None else study.config.dt_ms
            for item in study.comparison_traces
        )) or (study.config.dt_ms,)
        self.isi_dt_combo.blockSignals(True)
        self.isi_dt_combo.clear()
        for dt_ms in values:
            self.isi_dt_combo.addItem(
                self._translator.tr("gui.single_neuron.trace_dt_option", dt=f"{dt_ms:g}"),
                dt_ms,
            )
        self._set_combo_data(self.isi_dt_combo, self.trace_dt_combo.currentData() or values[0])
        self.isi_dt_combo.blockSignals(False)
        self.isi_dt_combo.setVisible(len(values) > 1)
        self._show_selected_isi_dt()

    def _show_selected_isi_dt(self) -> None:
        study = self._last_study
        if study is None or not study.selection.traces:
            return
        dt_ms = float(self.isi_dt_combo.currentData() or study.config.dt_ms)
        traces = tuple(
            (item.method_id, item.trace)
            for item in study.comparison_traces
            if (item.dt_ms if item.dt_ms is not None else study.config.dt_ms) == dt_ms
        )
        if not traces and study.trace is not None:
            traces = ((study.trace_config or study.config).method_id, study.trace)
        reference = (
            study.convergence.reference_period_ms
            if study.convergence is not None else float("nan")
        )
        reference_source = "reference" if np.isfinite(reference) else "estimate"
        if not np.isfinite(reference):
            source = (
                min(
                    study.comparison_traces,
                    key=lambda item: item.dt_ms or study.config.dt_ms,
                ).trace
                if study.comparison_traces else study.trace
            )
            times = source.spike_times_ms[source.spike_times_ms >= study.config.burn_in_ms]
            reference = float(np.median(np.diff(times))) if times.size >= 2 else float("nan")
        self.isi_plot.clear()
        self._isi_items.clear()
        self.isi_context_label.setText(
            self._translator.tr(
                "gui.single_neuron.isi_context",
                period=self._format_number(reference, 7),
                source=self._translator.tr(f"gui.single_neuron.isi_reference.{reference_source}"),
            )
        )
        hover_series = []
        for index, (method_id, trace) in enumerate(traces):
            times = trace.spike_times_ms[trace.spike_times_ms >= study.config.burn_in_ms]
            if times.size < 2 or trace.diverged or not np.isfinite(reference):
                continue
            x = np.arange(1, times.size, dtype=np.float64)
            y = np.diff(times) - reference
            name = self._method_name(method_id)
            item = self.isi_plot.plot(
                x,
                y,
                pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.6),
                symbol="o",
                symbolSize=5,
                name=self._method_name(method_id),
            )
            hover_series.append(self._hover_entry(name, x, y, item))
            self._isi_items[method_id] = item
        if np.isfinite(reference):
            self.isi_plot.addLine(y=0.0, pen=pg.mkPen("#aaaaaa", style=Qt.PenStyle.DashLine))
        self._set_hover_series(self.isi_plot, hover_series)
        self._rebuild_isi_method_controls(tuple(self._isi_items))
        self._set_plot_labels()
        self._enable_exports(self.isi_plot)

    def _rebuild_isi_method_controls(self, method_ids: tuple[str, ...]) -> None:
        self.isi_method_menu.clear()
        self._isi_method_actions.clear()
        for method_id in method_ids:
            action = self.isi_method_menu.addAction(self._method_name(method_id))
            action.setToolTip(self._method_name(method_id))
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda visible, value=method_id: self._isi_items[value].setVisible(visible)
            )
            self._isi_method_actions[method_id] = action
        self.isi_method_button.setVisible(len(method_ids) > 1)

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
            time_ms, voltage_mv, recovery, spike_times_ms = observed_trace_arrays(
                trace, config.burn_in_ms
            )
            voltage_item = self.voltage_plot.plot(
                time_ms,
                voltage_mv,
                pen=pg.mkPen(color, width=1.5),
                name=self._method_name(method_id),
            )
            recovery_item = self.recovery_plot.plot(
                time_ms,
                recovery,
                pen=pg.mkPen(color, width=1.3),
            )
            items: list[Any] = [voltage_item, recovery_item]
            if spike_times_ms.size:
                spike_item = self.voltage_plot.plot(
                    spike_times_ms,
                    np.full(spike_times_ms.shape, config.neuron.threshold_mv),
                    pen=None,
                    symbol="t1",
                    symbolSize=6,
                    symbolBrush=color,
                    symbolPen=None,
                )
                items.append(spike_item)
            self._trace_items[method_id] = items
            voltage_series.append(
                self._hover_entry(name, time_ms, voltage_mv, voltage_item)
            )
            recovery_series.append(
                self._hover_entry(name, time_ms, recovery, recovery_item)
            )
        self._set_hover_series(self.voltage_plot, voltage_series)
        self._set_hover_series(self.recovery_plot, recovery_series)
        self._rebuild_trace_method_controls(tuple(method_id for method_id, _ in traces))
        self._set_plot_labels()
        self._enable_exports(self.voltage_plot, self.recovery_plot)

    def _display_phase_study(self, study: SingleNeuronStudy) -> None:
        values = tuple(dict.fromkeys(
            item.dt_ms if item.dt_ms is not None else study.config.dt_ms
            for item in study.comparison_traces
        ))
        self.phase_dt_combo.blockSignals(True)
        self.phase_dt_combo.clear()
        for dt_ms in values:
            self.phase_dt_combo.addItem(
                self._translator.tr("gui.single_neuron.trace_dt_option", dt=f"{dt_ms:g}"),
                dt_ms,
            )
        if values:
            preferred = (study.trace_config or study.config).dt_ms
            self._set_combo_data(
                self.phase_dt_combo, preferred if preferred in values else min(values)
            )
        self.phase_dt_combo.blockSignals(False)
        self.phase_dt_combo.setVisible(len(values) > 1)
        self._show_selected_phase_dt()

    def _show_selected_phase_dt(self) -> None:
        study = self._last_study
        if study is None or not study.selection.phase:
            return
        if study.comparison_traces:
            dt_ms = float(self.phase_dt_combo.currentData() or study.config.dt_ms)
            traces = tuple(
                (item.method_id, item.trace)
                for item in study.comparison_traces
                if (item.dt_ms if item.dt_ms is not None else study.config.dt_ms) == dt_ms
            )
            config = study.config
        else:
            config = study.trace_config or study.config
            traces = ((config.method_id, study.trace),) if study.trace is not None else ()
        self._plot_phase_traces(traces, config)

    def _plot_phase_traces(
        self,
        traces: tuple[tuple[str, SingleNeuronTrace], ...],
        config: SingleNeuronConfig,
    ) -> None:
        self.phase_plot.clear()
        self._phase_items.clear()
        self._phase_limits = None
        hover_series = []
        voltage_bounds = []
        recovery_bounds = []
        for index, (method_id, trace) in enumerate(traces):
            _, voltage, recovery, _ = observed_trace_arrays(trace, config.burn_in_ms)
            valid = np.isfinite(voltage) & np.isfinite(recovery)
            voltage = voltage[valid]
            recovery = recovery[valid]
            if not voltage.size:
                continue
            # A reset is a discontinuity; never draw a continuous arc across it.
            connected = np.r_[np.diff(voltage) > -20.0, False]
            name = self._method_name(method_id)
            item = self.phase_plot.plot(
                voltage,
                recovery,
                connect=connected,
                pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.4),
                name=self._method_name(method_id),
            )
            self._phase_items[method_id] = item
            hover_series.append(self._hover_entry(name, voltage, recovery, item))
            voltage_bounds.extend((float(np.min(voltage)), float(np.max(voltage))))
            recovery_bounds.extend((float(np.min(recovery)), float(np.max(recovery))))
        self._set_hover_series(self.phase_plot, hover_series)
        self._rebuild_phase_method_controls(tuple(self._phase_items))
        if voltage_bounds:
            if config.noise_std == 0.0:
                grid = np.linspace(min(-100.0, min(voltage_bounds)),
                                   max(40.0, max(voltage_bounds)), 300)
                self.phase_plot.plot(
                    grid,
                    0.04 * grid**2 + 5 * grid + 140 + config.input_current,
                    pen=pg.mkPen("#f6d32d", style=Qt.PenStyle.DashLine),
                    name=self._translator.tr("gui.single_neuron.legend.v_nullcline"),
                )
                self.phase_plot.plot(
                    grid,
                    config.neuron.b * grid,
                    pen=pg.mkPen("#dc8add", style=Qt.PenStyle.DotLine),
                    name=self._translator.tr("gui.single_neuron.legend.u_nullcline"),
                )
            # The v nullcline rises far above the actual orbit. Fit the view to
            # the measured state so it does not flatten the useful trajectory.
            pad = max(3.0, 0.12 * (max(recovery_bounds) - min(recovery_bounds)))
            self._phase_limits = (
                min(min(voltage_bounds), config.neuron.c) - 5.0,
                max(max(voltage_bounds), config.neuron.threshold_mv) + 5.0,
                min(recovery_bounds) - pad,
                max(recovery_bounds) + pad,
            )
            self._fit_plot(self.phase_plot)
        self._set_plot_labels()
        self._enable_exports(self.phase_plot)

    def _rebuild_phase_method_controls(self, method_ids: tuple[str, ...]) -> None:
        self.phase_method_menu.clear()
        self._phase_method_actions.clear()
        for method_id in method_ids:
            action = self.phase_method_menu.addAction(self._method_name(method_id))
            action.setToolTip(self._method_name(method_id))
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda visible, value=method_id: self._phase_items[value].setVisible(visible)
            )
            self._phase_method_actions[method_id] = action
        self.phase_method_button.setVisible(len(method_ids) > 1)

    def _rebuild_trace_method_controls(self, method_ids: tuple[str, ...]) -> None:
        self.trace_method_menu.clear()
        self._trace_method_actions.clear()
        for method_id in method_ids:
            action = self.trace_method_menu.addAction(self._method_name(method_id))
            action.setToolTip(self._method_name(method_id))
            action.setCheckable(True)
            action.setChecked(True)
            action.toggled.connect(
                lambda visible, value=method_id: self._set_trace_method_visible(value, visible)
            )
            self._trace_method_actions[method_id] = action
        self.trace_method_button.setVisible(len(method_ids) > 1)

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
                name=self._method_name(series.method_id),
            )
            hover_series.append(
                self._hover_entry(name, x[valid], y[valid], item, log_x=True, log_y=True)
            )
            if np.count_nonzero(valid) >= 2:
                valid_series.append((x[valid], y[valid]))
        self._add_convergence_guides(valid_series)
        self._set_hover_series(self.convergence_plot, hover_series)
        self._set_plot_labels()
        self._enable_exports(self.convergence_plot)

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
            valid = np.isfinite(curve.rates_hz)
            if curve.valid_counts is not None:
                valid &= curve.valid_counts > 0
            else:
                valid &= ~curve.diverged
            name = self._method_name(curve.method_id)
            color = _PLOT_COLORS[index % len(_PLOT_COLORS)]
            item = self.fi_plot.plot(
                curve.currents[valid],
                curve.rates_hz[valid],
                pen=pg.mkPen(color, width=1.8),
                symbol="o",
                symbolSize=4,
                name=self._method_name(curve.method_id),
            )
            hover_series.append(
                self._hover_entry(name, curve.currents[valid], curve.rates_hz[valid], item)
            )
            if curve.rates_sd_hz is not None and study.n_runs > 1:
                self.fi_plot.addItem(
                    pg.ErrorBarItem(
                        x=curve.currents[valid],
                        y=curve.rates_hz[valid],
                        height=2.0 * curve.rates_sd_hz[valid],
                        beam=0.25,
                        pen=pg.mkPen(color, width=1.2),
                    )
                )
        self._set_hover_series(self.fi_plot, hover_series)
        self._set_plot_labels()
        self._enable_exports(self.fi_plot)

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
        self._enable_exports(self.alpha_plot)

    def _display_variability(self, study: SingleNeuronStudy) -> None:
        self.variability_plot.clear()
        hover_series: list[dict[str, Any]] = []
        summaries = []
        method_ids = tuple(dict.fromkeys(point.method_id for point in study.trial_points))
        for index, method_id in enumerate(method_ids):
            points = [point for point in study.trial_points if point.method_id == method_id]
            seeds = np.asarray([point.seed for point in points], dtype=np.float64)
            rates = np.asarray(
                [point.firing_rate_hz if not point.diverged else float("nan") for point in points],
                dtype=np.float64,
            )
            valid = np.isfinite(rates)
            name = self._method_name(method_id)
            item = self.variability_plot.plot(
                seeds[valid],
                rates[valid],
                pen=pg.mkPen(_PLOT_COLORS[index % len(_PLOT_COLORS)], width=1.5),
                symbol="o",
                symbolSize=7,
                name=self._method_name(method_id),
            )
            hover_series.append(self._hover_entry(name, seeds[valid], rates[valid], item))
            if np.count_nonzero(valid):
                mean = np.mean(rates[valid])
                sd = np.std(rates[valid], ddof=1) if np.count_nonzero(valid) > 1 else 0.0
                summaries.append(f"{name}: {mean:.3g} ± {sd:.2g} Hz (n={np.count_nonzero(valid)})")
        self.variability_summary_label.setText("   |   ".join(summaries))
        self._set_hover_series(self.variability_plot, hover_series)
        self._set_plot_labels()
        self._enable_exports(self.variability_plot)

    def _display_summary(self, study: SingleNeuronStudy) -> None:
        self._retranslate_summary_header()
        self._set_summary_reference(study)
        self.summary_table.setRowCount(len(study.convergence.series))
        control_by_method = {series.method_id: series for series in study.reset_control.series}
        for row, series in enumerate(study.convergence.series):
            fine = min(series.points, key=lambda point: point.dt_ms)
            control_series = control_by_method[series.method_id]
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
                self._format_number(control_series.observed_period_order, 4),
                f"{fine.spike_count} / {study.convergence.reference_spike_count}",
                self._format_number(fine.period_error_ms, 5),
                self._format_number(fine.drift_ms_per_s, 5),
                self._format_number(fine.nanoseconds_per_step, 5),
                status,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 7 and np.isfinite(fine.implicit_attempts_per_grid_step):
                    item.setToolTip(self._translator.tr(
                        "gui.single_neuron.implicit_work_tip",
                        attempts=self._format_number(
                            fine.implicit_attempts_per_grid_step, 4
                        ),
                    ))
                self.summary_table.setItem(row, column, item)

    def _set_summary_reference(self, study: SingleNeuronStudy) -> None:
        self.summary_reference_label.setText(
            self._translator.tr(
                "gui.single_neuron.summary_reference",
                period=self._format_number(study.convergence.reference_period_ms, 7),
                difference=self._format_number(
                    study.convergence.reference_uncertainty_ms, 4
                ),
                dt=self._format_number(study.convergence.reference_dt_ms, 5),
            )
        )

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
            (self.phase_plot, self.phase_hover_label),
            (self.isi_plot, self.isi_hover_label),
            (self.convergence_plot, self.convergence_hover_label),
            (self.fi_plot, self.fi_hover_label),
            (self.alpha_plot, self.alpha_hover_label),
            (self.variability_plot, self.variability_hover_label),
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
        self.summary_help_hint.setVisible(self._help_mode)
        header = self.summary_table.horizontalHeader()
        header.setSectionsClickable(self._help_mode)
        header.setCursor(
            Qt.CursorShape.PointingHandCursor
            if self._help_mode
            else Qt.CursorShape.ArrowCursor
        )
        for label in (
            self.trace_hover_label,
            self.phase_hover_label,
            self.isi_hover_label,
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
            self.phase_plot,
            self.isi_plot,
            self.convergence_plot,
            self.fi_plot,
            self.alpha_plot,
        ):
            plot.updateGeometry()
            plot.getViewBox().update()
