from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from snnlab.gui.widgets.help_dialog import HelpDialog


class EvaluationParameterLabel(QWidget):
    """Localized evaluation label with contextual study-mode help.

    Локализованная подпись параметра evaluation с контекстной справкой режима
    изучения.
    """

    def __init__(self, *, help_topic: str, parent=None) -> None:
        super().__init__(parent)
        self.help_topic = str(help_topic)
        self._translator = None
        self.text_label = QLabel()
        self.help_button = QToolButton()
        self.help_button.setText("?")
        self.help_button.setFixedSize(22, 22)
        self.help_button.setVisible(False)
        self.help_button.clicked.connect(self._open_help)

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
        self._translator = translator
        if translator is None:
            self.help_button.setToolTip(self.help_topic)
            return
        try:
            topic = translator.help_topic(self.help_topic)
        except (KeyError, TypeError):
            self.help_button.setToolTip(self.help_topic)
            return
        short = str(topic.get("short", ""))
        long_text = str(topic.get("long", ""))
        tooltip = short
        if long_text:
            tooltip = f"{short}\n\n{long_text}" if short else long_text
        self.help_button.setToolTip(tooltip)

    def _open_help(self) -> None:
        if self._translator is None:
            return
        HelpDialog(self._translator.help_topic(self.help_topic), parent=self).exec()


class EvaluationView(QWidget):
    """
    Configures and visualizes classifier evaluation with interactive diagnostics.

    The layout uses splitters and scroll areas so the tab remains usable on
    smaller screens. Plots expose hover readouts for exact values instead of
    forcing users to infer numbers visually.

    Настраивает и визуализирует evaluation классификатора с интерактивными
    диагностиками.

    Layout использует splitter-ы и области прокрутки, поэтому вкладка остаётся
    пригодной на небольших экранах. Графики показывают точные значения по hover,
    а не заставляют пользователя угадывать числа по картинке.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._translator = None
        self._help_mode = False
        self._architecture = "dci"
        self._last_result: Any | None = None
        self._last_confusion: np.ndarray | None = None
        self._last_class_counts: np.ndarray | None = None
        self._last_prediction_counts: np.ndarray | None = None
        self._last_per_class_accuracy: np.ndarray | None = None
        self._last_margin_hist: tuple[np.ndarray, np.ndarray] | None = None
        self._last_best_hist: tuple[np.ndarray, np.ndarray] | None = None
        self._last_response: np.ndarray | None = None
        self._last_response_order: np.ndarray | None = None
        self._mouse_proxies: list[Any] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.settings_group = QGroupBox()
        form = QFormLayout(self.settings_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.assignment_samples = QSpinBox()
        self.assignment_samples.setRange(1, 1_000_000)
        self.assignment_samples.setValue(500)
        self.test_samples = QSpinBox()
        self.test_samples.setRange(1, 1_000_000)
        self.test_samples.setValue(200)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(52)

        self.assignment_policy = QComboBox()
        self.homeostasis_mode = QComboBox()
        self.network_state_mode = QComboBox()
        self.prediction_rule = QComboBox()

        self.min_best_response = QDoubleSpinBox()
        self.min_best_response.setRange(0.0, 1_000_000.0)
        self.min_best_response.setDecimals(4)
        self.min_best_response.setSingleStep(0.01)
        self.min_best_response.setValue(0.0)

        self.min_absolute_margin = QDoubleSpinBox()
        self.min_absolute_margin.setRange(0.0, 1_000_000.0)
        self.min_absolute_margin.setDecimals(4)
        self.min_absolute_margin.setSingleStep(0.01)
        self.min_absolute_margin.setValue(0.0)

        self.min_relative_margin = QDoubleSpinBox()
        self.min_relative_margin.setRange(0.0, 1.0)
        self.min_relative_margin.setDecimals(3)
        self.min_relative_margin.setSingleStep(0.05)
        self.min_relative_margin.setValue(0.10)

        self.top_k_per_class = QSpinBox()
        self.top_k_per_class.setRange(0, 1_000_000)
        self.top_k_per_class.setValue(20)
        self.top_k_per_class.setSpecialValueText("∞")

        self.assignment_label = EvaluationParameterLabel(help_topic="evaluation.assignment_samples")
        self.test_label = EvaluationParameterLabel(help_topic="evaluation.test_samples")
        self.seed_label = EvaluationParameterLabel(help_topic="evaluation.seed")
        self.assignment_policy_label = EvaluationParameterLabel(
            help_topic="evaluation.assignment_policy"
        )
        self.homeostasis_label = EvaluationParameterLabel(help_topic="evaluation.homeostasis_mode")
        self.state_label = EvaluationParameterLabel(help_topic="evaluation.network_state_mode")
        self.rule_label = EvaluationParameterLabel(help_topic="evaluation.prediction_rule")
        self.min_best_label = EvaluationParameterLabel(help_topic="evaluation.min_best_response")
        self.min_absolute_margin_label = EvaluationParameterLabel(
            help_topic="evaluation.min_absolute_margin"
        )
        self.min_relative_margin_label = EvaluationParameterLabel(
            help_topic="evaluation.min_relative_margin"
        )
        self.top_k_label = EvaluationParameterLabel(help_topic="evaluation.top_k_per_class")
        self._parameter_labels = (
            self.assignment_label,
            self.test_label,
            self.seed_label,
            self.assignment_policy_label,
            self.homeostasis_label,
            self.state_label,
            self.rule_label,
            self.min_best_label,
            self.min_absolute_margin_label,
            self.min_relative_margin_label,
            self.top_k_label,
        )

        form.addRow(self.assignment_label, self.assignment_samples)
        form.addRow(self.test_label, self.test_samples)
        form.addRow(self.seed_label, self.seed)
        form.addRow(self.assignment_policy_label, self.assignment_policy)
        form.addRow(self.homeostasis_label, self.homeostasis_mode)
        form.addRow(self.state_label, self.network_state_mode)
        form.addRow(self.rule_label, self.prediction_rule)
        form.addRow(self.min_best_label, self.min_best_response)
        form.addRow(self.min_absolute_margin_label, self.min_absolute_margin)
        form.addRow(self.min_relative_margin_label, self.min_relative_margin)
        form.addRow(self.top_k_label, self.top_k_per_class)

        # EN: The protocol area and plots are separated by a vertical splitter.
        #     Users can shrink the protocol settings and give most of the screen
        #     to the plots during analysis or presentations.
        # RU: Протокол и графики разделены вертикальным splitter-ом.
        #     Пользователь может сжать настройки протокола и отдать большую
        #     часть экрана графикам во время анализа или демонстрации.
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(self.main_splitter, 1)

        self.protocol_panel = QWidget()
        protocol_layout = QVBoxLayout(self.protocol_panel)
        protocol_layout.setContentsMargins(0, 0, 0, 0)
        protocol_layout.setSpacing(6)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setWidget(self.settings_group)
        protocol_layout.addWidget(self.settings_scroll, 1)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        protocol_layout.addWidget(self.summary_label)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        protocol_layout.addWidget(self.details_label)

        self.result_tabs = QTabWidget()
        self.overview_tab = QWidget()
        self.diagnostics_tab = QWidget()
        self.response_tab = QWidget()
        self.result_tabs.addTab(self.overview_tab, "")
        self.result_tabs.addTab(self.diagnostics_tab, "")
        self.result_tabs.addTab(self.response_tab, "")

        self.main_splitter.addWidget(self.protocol_panel)
        self.main_splitter.addWidget(self.result_tabs)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([240, 760])
        self.result_tabs.currentChanged.connect(
            lambda *_: QTimer.singleShot(0, self.refresh_layout)
        )

        self._build_overview_tab()
        self._build_diagnostics_tab()
        self._build_response_tab()
        self._install_hover_handlers()
        self.set_architecture("dci")

    def _build_overview_tab(self) -> None:
        layout = QVBoxLayout(self.overview_tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.overview_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.overview_splitter, 1)

        self.confusion_panel = self._plot_panel()
        self.confusion_plot = pg.PlotWidget()
        self.confusion_plot.setAspectLocked(True)
        self.confusion_plot.getViewBox().invertY(True)
        self.confusion_plot.setLabel("bottom", "predicted")
        self.confusion_plot.setLabel("left", "true")
        self.confusion_image = pg.ImageItem(axisOrder="row-major")
        self.confusion_plot.addItem(self.confusion_image)
        self.confusion_info = QLabel()
        self.confusion_info.setWordWrap(True)
        self.confusion_info.setVisible(False)
        self.confusion_panel.layout().addWidget(self.confusion_plot, 1)
        self.confusion_panel.layout().addWidget(self.confusion_info)
        self.overview_splitter.addWidget(self.confusion_panel)

        self.class_panel = self._plot_panel()
        self.class_plot = pg.PlotWidget()
        self.class_plot.showGrid(x=True, y=True, alpha=0.2)
        self.class_info = QLabel()
        self.class_info.setWordWrap(True)
        self.class_info.setVisible(False)
        self.class_panel.layout().addWidget(self.class_plot, 1)
        self.class_panel.layout().addWidget(self.class_info)
        self.overview_splitter.addWidget(self.class_panel)
        self.overview_splitter.setChildrenCollapsible(False)
        self.overview_splitter.setHandleWidth(6)
        self.overview_splitter.setStretchFactor(0, 1)
        self.overview_splitter.setStretchFactor(1, 1)
        self.overview_splitter.setSizes([600, 800])

    def _build_diagnostics_tab(self) -> None:
        layout = QVBoxLayout(self.diagnostics_tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.diagnostics_splitter = QSplitter(Qt.Orientation.Vertical)
        self.diagnostics_top = QSplitter(Qt.Orientation.Horizontal)
        self.diagnostics_bottom = QSplitter(Qt.Orientation.Horizontal)
        self.diagnostics_splitter.addWidget(self.diagnostics_top)
        self.diagnostics_splitter.addWidget(self.diagnostics_bottom)
        layout.addWidget(self.diagnostics_splitter, 1)

        self.prediction_plot, self.prediction_info = self._add_diagnostic_plot(self.diagnostics_top)
        self.per_class_accuracy_plot, self.per_class_info = self._add_diagnostic_plot(
            self.diagnostics_top
        )
        self.margin_plot, self.margin_info = self._add_diagnostic_plot(self.diagnostics_bottom)
        self.best_response_plot, self.best_response_info = self._add_diagnostic_plot(
            self.diagnostics_bottom
        )
        for splitter in (self.diagnostics_splitter, self.diagnostics_top, self.diagnostics_bottom):
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(6)
        self.diagnostics_splitter.setStretchFactor(0, 1)
        self.diagnostics_splitter.setStretchFactor(1, 1)
        self.diagnostics_top.setStretchFactor(0, 1)
        self.diagnostics_top.setStretchFactor(1, 1)
        self.diagnostics_bottom.setStretchFactor(0, 1)
        self.diagnostics_bottom.setStretchFactor(1, 1)
        self.diagnostics_splitter.setSizes([420, 420])

    def _build_response_tab(self) -> None:
        layout = QVBoxLayout(self.response_tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.response_heatmap_plot = pg.PlotWidget()
        self.response_heatmap_plot.setLabel("bottom", "E neuron order")
        self.response_heatmap_plot.setLabel("left", "class")
        self.response_heatmap_plot.getViewBox().invertY(True)
        self.response_heatmap_image = pg.ImageItem(axisOrder="row-major")
        self.response_heatmap_plot.addItem(self.response_heatmap_image)
        self.response_help_label = QLabel()
        self.response_help_label.setWordWrap(True)
        self.response_help_label.setVisible(False)
        layout.addWidget(self.response_heatmap_plot, 1)
        layout.addWidget(self.response_help_label)

    @staticmethod
    def _plot_panel() -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        return panel

    def _add_diagnostic_plot(self, splitter: QSplitter) -> tuple[pg.PlotWidget, QLabel]:
        panel = self._plot_panel()
        plot = pg.PlotWidget()
        plot.showGrid(x=True, y=True, alpha=0.2)
        info = QLabel()
        info.setWordWrap(True)
        panel.layout().addWidget(plot, 1)
        panel.layout().addWidget(info)
        splitter.addWidget(panel)
        return plot, info

    def set_help_mode(self, enabled: bool) -> None:
        """Shows static explanatory captions only in learning mode.

        Показывает статические поясняющие подписи только в режиме изучения.
        Hover-значения остаются доступными независимо от режима.
        """
        self._help_mode = bool(enabled)
        for label in self._parameter_labels:
            label.set_help_mode(self._help_mode)
        self._refresh_static_infos()

    def _set_static_info(self, label: QLabel, key: str) -> None:
        if self._help_mode:
            label.setText(self._tr(key))
            label.setVisible(True)
        else:
            label.clear()
            label.setVisible(False)

    @staticmethod
    def _set_hover_info(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setVisible(True)

    def _refresh_static_infos(self) -> None:
        if self._last_result is None:
            for label in (
                self.confusion_info,
                self.class_info,
                self.prediction_info,
                self.per_class_info,
                self.margin_info,
                self.best_response_info,
                self.response_help_label,
            ):
                label.clear()
                label.setVisible(False)
            return
        self._set_static_info(self.confusion_info, "gui.evaluation.confusion_tip")
        self._set_static_info(self.class_info, "gui.evaluation.class_assignment_tip")
        self._set_static_info(self.prediction_info, "gui.evaluation.prediction_distribution_tip")
        self._set_static_info(self.per_class_info, "gui.evaluation.per_class_accuracy_tip")
        self._set_static_info(self.margin_info, "gui.evaluation.margin_tip")
        self._set_static_info(self.best_response_info, "gui.evaluation.best_response_tip")
        if hasattr(self._last_result, "assignment"):
            self._set_static_info(self.response_help_label, "gui.evaluation.response_heatmap_help")
        else:
            self._set_static_info(
                self.response_help_label, "gui.evaluation.response_heatmap_reservoir"
            )

    def set_architecture(self, architecture: str) -> None:
        self._architecture = str(architecture)
        is_dci = self._architecture == "dci"
        self.assignment_label.setVisible(is_dci)
        self.assignment_samples.setVisible(is_dci)
        self.assignment_policy_label.setVisible(is_dci)
        self.assignment_policy.setVisible(is_dci)
        self.homeostasis_label.setVisible(is_dci)
        self.homeostasis_mode.setVisible(is_dci)
        self.state_label.setVisible(is_dci)
        self.network_state_mode.setVisible(is_dci)
        self.rule_label.setVisible(is_dci)
        self.prediction_rule.setVisible(is_dci)
        for label, widget in (
            (self.min_best_label, self.min_best_response),
            (self.min_absolute_margin_label, self.min_absolute_margin),
            (self.min_relative_margin_label, self.min_relative_margin),
            (self.top_k_label, self.top_k_per_class),
        ):
            label.setVisible(is_dci)
            widget.setVisible(is_dci)
        self._refresh_texts()

    def retranslate(self, translator) -> None:
        self._translator = translator
        for label in self._parameter_labels:
            label.retranslate_help(translator)
        self._refresh_texts()
        if self._last_result is not None:
            self.display_result(self._last_result)

    def refresh_layout(self) -> None:
        """Refreshes visible splitter and pyqtgraph viewport geometry.

        Обновляет геометрию видимых splitter-ов и pyqtgraph viewport после
        переключения вкладки или изменения состояния главного окна.
        """
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        for splitter in (
            self.main_splitter,
            self.overview_splitter,
            self.diagnostics_splitter,
            self.diagnostics_top,
            self.diagnostics_bottom,
        ):
            splitter.refresh()
            splitter.updateGeometry()
        for plot in self.findChildren(pg.PlotWidget):
            plot.updateGeometry()
            plot.viewport().update()
            plot.update()

    def parameters(self) -> dict[str, Any]:
        return {
            "assignment_samples": int(self.assignment_samples.value()),
            "test_samples": int(self.test_samples.value()),
            "seed": int(self.seed.value()),
            "assignment_policy": str(self.assignment_policy.currentData() or "full_train_pool"),
            "homeostasis_mode": str(self.homeostasis_mode.currentData() or "frozen"),
            "network_state_mode": str(self.network_state_mode.currentData() or "fresh_continuous"),
            "prediction_rule": str(self.prediction_rule.currentData() or "balanced_topk"),
            "min_best_response": float(self.min_best_response.value()),
            "min_absolute_margin": float(self.min_absolute_margin.value()),
            "min_relative_margin": float(self.min_relative_margin.value()),
            "top_k_per_class": int(self.top_k_per_class.value()),
        }

    def set_parameters(self, values: dict[str, Any]) -> None:
        """Applies known evaluation parameters from a portable config.

        Применяет известные evaluation-параметры из переносимой конфигурации.
        """
        editors = {
            "assignment_samples": self.assignment_samples,
            "test_samples": self.test_samples,
            "seed": self.seed,
            "assignment_policy": self.assignment_policy,
            "homeostasis_mode": self.homeostasis_mode,
            "network_state_mode": self.network_state_mode,
            "prediction_rule": self.prediction_rule,
            "min_best_response": self.min_best_response,
            "min_absolute_margin": self.min_absolute_margin,
            "min_relative_margin": self.min_relative_margin,
            "top_k_per_class": self.top_k_per_class,
        }
        for key, value in values.items():
            editor = editors.get(key)
            if editor is None:
                continue
            editor.blockSignals(True)
            try:
                if isinstance(editor, QSpinBox):
                    editor.setValue(int(value))
                elif isinstance(editor, QDoubleSpinBox):
                    editor.setValue(float(value))
                elif isinstance(editor, QComboBox):
                    index = editor.findData(value)
                    if index >= 0:
                        editor.setCurrentIndex(index)
            finally:
                editor.blockSignals(False)

    def reset(self) -> None:
        self._last_result = None
        self._last_confusion = None
        self._last_class_counts = None
        self._last_prediction_counts = None
        self._last_per_class_accuracy = None
        self._last_margin_hist = None
        self._last_best_hist = None
        self._last_response = None
        self._last_response_order = None
        self.confusion_image.clear()
        self.response_heatmap_image.clear()
        for plot in (
            self.class_plot,
            self.prediction_plot,
            self.per_class_accuracy_plot,
            self.margin_plot,
            self.best_response_plot,
        ):
            plot.clear()
        self.summary_label.setText(self._tr("gui.evaluation.waiting"))
        self.details_label.clear()
        self._refresh_static_infos()

    def display_result(self, result: Any) -> None:
        self._last_result = result
        self._plot_confusion(result)
        self._plot_class_assignment(result)
        self._plot_diagnostics(result)
        self._plot_response_heatmap(result)
        self._update_summary(result)

    def _plot_confusion(self, result: Any) -> None:
        confusion = np.asarray(result.confusion_matrix, dtype=np.float64)
        self._last_confusion = confusion
        self.confusion_image.setImage(confusion, autoLevels=True)
        self.confusion_plot.autoRange()
        self._set_static_info(self.confusion_info, "gui.evaluation.confusion_tip")

    def _plot_class_assignment(self, result: Any) -> None:
        self.class_plot.clear()
        self._last_class_counts = None
        if hasattr(result, "assignment"):
            counts = np.asarray(result.assignment.class_counts, dtype=np.float64)
            self._last_class_counts = counts
            x = np.arange(len(counts), dtype=np.float64)
            bars = pg.BarGraphItem(x=x, height=counts, width=0.75, brush=pg.mkBrush(190, 190, 190))
            self.class_plot.addItem(bars)
        self.class_plot.setLabel("bottom", self._tr("gui.evaluation.class_id"))
        self.class_plot.setLabel("left", self._tr("gui.evaluation.assigned_neurons"))
        self._set_static_info(self.class_info, "gui.evaluation.class_assignment_tip")

    def _plot_diagnostics(self, result: Any) -> None:
        for plot in (
            self.prediction_plot,
            self.per_class_accuracy_plot,
            self.margin_plot,
            self.best_response_plot,
        ):
            plot.clear()
        self._last_prediction_counts = None
        self._last_per_class_accuracy = None
        self._last_margin_hist = None
        self._last_best_hist = None

        if not hasattr(result, "diagnostics"):
            return

        d = result.diagnostics
        x = np.arange(len(d.prediction_counts), dtype=np.float64)
        self._last_prediction_counts = np.asarray(d.prediction_counts, dtype=np.float64)
        pred_bars = pg.BarGraphItem(
            x=x, height=self._last_prediction_counts, width=0.75, brush=pg.mkBrush(180, 180, 180)
        )
        self.prediction_plot.addItem(pred_bars)
        self.prediction_plot.setTitle(self._tr("gui.evaluation.prediction_distribution"))
        self.prediction_plot.setLabel("bottom", self._tr("gui.evaluation.class_id"))
        self.prediction_plot.setLabel("left", self._tr("gui.evaluation.predicted_samples"))
        self._set_static_info(self.prediction_info, "gui.evaluation.prediction_distribution_tip")

        self._last_per_class_accuracy = (
            np.nan_to_num(np.asarray(d.per_class_accuracy, dtype=np.float64), nan=0.0) * 100.0
        )
        acc_bars = pg.BarGraphItem(
            x=x, height=self._last_per_class_accuracy, width=0.75, brush=pg.mkBrush(140, 190, 255)
        )
        self.per_class_accuracy_plot.addItem(acc_bars)
        self.per_class_accuracy_plot.setTitle(self._tr("gui.evaluation.per_class_accuracy"))
        self.per_class_accuracy_plot.setLabel("bottom", self._tr("gui.evaluation.class_id"))
        self.per_class_accuracy_plot.setLabel(
            "left", self._tr("gui.evaluation.per_class_accuracy_percent")
        )
        self.per_class_accuracy_plot.setYRange(0, 100)
        self._set_static_info(self.per_class_info, "gui.evaluation.per_class_accuracy_tip")

        assigned_mask = result.assignment.neuron_labels >= 0
        margins = np.asarray(result.assignment.relative_margin[assigned_mask], dtype=np.float64)
        best = np.asarray(result.assignment.best_response[assigned_mask], dtype=np.float64)
        self._last_margin_hist = self._plot_histogram(
            self.margin_plot, margins, self._tr("gui.evaluation.relative_selectivity_margin")
        )
        self._last_best_hist = self._plot_histogram(
            self.best_response_plot, best, self._tr("gui.evaluation.best_response")
        )
        self._set_static_info(self.margin_info, "gui.evaluation.margin_tip")
        self._set_static_info(self.best_response_info, "gui.evaluation.best_response_tip")

    def _plot_response_heatmap(self, result: Any) -> None:
        self.response_heatmap_image.clear()
        self._last_response = None
        self._last_response_order = None
        if not hasattr(result, "assignment"):
            self._set_static_info(
                self.response_help_label, "gui.evaluation.response_heatmap_reservoir"
            )
            return

        response = np.asarray(result.assignment.class_mean_response, dtype=np.float64)
        order = np.asarray(result.diagnostics.sorted_neuron_indices, dtype=np.int64)
        if order.size == response.shape[1]:
            response = response[:, order]
            self._last_response_order = order
        else:
            self._last_response_order = np.arange(response.shape[1], dtype=np.int64)
        self._last_response = response
        self.response_heatmap_image.setImage(response, autoLevels=True)
        self.response_heatmap_plot.autoRange()
        self._set_static_info(self.response_help_label, "gui.evaluation.response_heatmap_help")

    def _plot_histogram(
        self, plot: pg.PlotWidget, values: np.ndarray, title: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        plot.setTitle(title)
        if values.size == 0:
            return None
        bins = min(40, max(5, int(np.sqrt(values.size))))
        counts, edges = np.histogram(values, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        if np.all(widths > 0):
            item = pg.BarGraphItem(
                x=centers, height=counts, width=widths * 0.9, brush=pg.mkBrush(180, 180, 180)
            )
            plot.addItem(item)
        return counts.astype(float), edges.astype(float)

    def _update_summary(self, result: Any) -> None:
        if hasattr(result, "assigned_fraction"):
            self.summary_label.setText(
                self._tr(
                    "gui.evaluation.summary_dci",
                    accuracy=f"{100.0 * float(result.accuracy):.2f}",
                    assigned=f"{100.0 * float(result.assigned_fraction):.2f}",
                    accepted_assignment=f"{100.0 * float(result.accepted_fraction_assignment):.2f}",
                    accepted_test=f"{100.0 * float(result.accepted_fraction_test):.2f}",
                )
            )
            d = result.diagnostics
            self.details_label.setText(
                self._tr(
                    "gui.evaluation.diagnostics_summary",
                    unassigned=str(int(d.unassigned_count)),
                    silent=str(int(d.silent_count)),
                    quality_rejected=str(int(d.quality_rejected_count)),
                    top_k_pruned=str(int(d.top_k_pruned_count)),
                    median_relative_margin=f"{float(d.median_relative_margin):.4g}",
                    unclassified=str(int(d.unclassified_count)),
                )
            )
        else:
            self.summary_label.setText(
                self._tr(
                    "gui.evaluation.summary_reservoir",
                    accuracy=f"{100.0 * float(result.accuracy):.2f}",
                )
            )
            self.details_label.clear()

    def _refresh_texts(self) -> None:
        self.settings_group.setTitle(self._tr("gui.evaluation.settings"))
        self.assignment_label.setText(self._tr("gui.evaluation.assignment_samples"))
        self.test_label.setText(self._tr("gui.evaluation.test_samples"))
        self.seed_label.setText(self._tr("gui.evaluation.seed"))
        self.assignment_policy_label.setText(self._tr("gui.evaluation.assignment_policy"))
        self.homeostasis_label.setText(self._tr("gui.evaluation.homeostasis_mode"))
        self.state_label.setText(self._tr("gui.evaluation.network_state_mode"))
        self.rule_label.setText(self._tr("gui.evaluation.prediction_rule"))
        self.min_best_label.setText(self._tr("gui.evaluation.min_best_response"))
        self.min_absolute_margin_label.setText(self._tr("gui.evaluation.min_absolute_margin"))
        self.min_relative_margin_label.setText(self._tr("gui.evaluation.min_relative_margin"))
        self.top_k_label.setText(self._tr("gui.evaluation.top_k_per_class"))
        self.result_tabs.setTabText(0, self._tr("gui.evaluation.overview_tab"))
        self.result_tabs.setTabText(1, self._tr("gui.evaluation.diagnostics_tab"))
        self.result_tabs.setTabText(2, self._tr("gui.evaluation.response_tab"))
        self.confusion_plot.setTitle(self._tr("gui.evaluation.confusion_matrix"))
        self.class_plot.setTitle(self._tr("gui.evaluation.class_assignment"))
        self.prediction_plot.setTitle(self._tr("gui.evaluation.prediction_distribution"))
        self.per_class_accuracy_plot.setTitle(self._tr("gui.evaluation.per_class_accuracy"))
        self.response_heatmap_plot.setTitle(self._tr("gui.evaluation.response_heatmap"))

        current_assignment_policy = self.assignment_policy.currentData()
        current_homeo = self.homeostasis_mode.currentData()
        current_state = self.network_state_mode.currentData()
        current_rule = self.prediction_rule.currentData()
        self.assignment_policy.clear()
        self.assignment_policy.addItem(
            self._tr("gui.evaluation.assignment_full_pool"), "full_train_pool"
        )
        self.assignment_policy.addItem(
            self._tr("gui.evaluation.assignment_exclude_training"), "exclude_training"
        )
        self.assignment_policy.addItem(
            self._tr("gui.evaluation.assignment_training_subset"), "training_subset"
        )
        self.homeostasis_mode.clear()
        self.homeostasis_mode.addItem(self._tr("gui.evaluation.homeostasis_frozen"), "frozen")
        self.homeostasis_mode.addItem(self._tr("gui.evaluation.homeostasis_zero"), "zero")
        self.network_state_mode.clear()
        self.network_state_mode.addItem(self._tr("gui.evaluation.state_fresh"), "fresh_continuous")
        self.network_state_mode.addItem(
            self._tr("gui.evaluation.state_trained"), "trained_continuous"
        )
        self.prediction_rule.clear()
        self.prediction_rule.addItem(self._tr("gui.evaluation.rule_balanced"), "balanced_topk")
        self.prediction_rule.addItem(self._tr("gui.evaluation.rule_mean"), "mean_response")
        self.prediction_rule.addItem(self._tr("gui.evaluation.rule_sum"), "sum_response")
        for combo, value in (
            (self.assignment_policy, current_assignment_policy),
            (self.homeostasis_mode, current_homeo),
            (self.network_state_mode, current_state),
            (self.prediction_rule, current_rule),
        ):
            index = combo.findData(value)
            combo.setCurrentIndex(index if index >= 0 else 0)
        if self._last_result is None:
            self.summary_label.setText(self._tr("gui.evaluation.waiting"))
            self.details_label.clear()
            self._refresh_static_infos()

    def _install_hover_handlers(self) -> None:
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.confusion_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_confusion_hover,
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.class_plot.scene().sigMouseMoved, rateLimit=30, slot=self._handle_class_hover
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.prediction_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_prediction_hover,
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.per_class_accuracy_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_per_class_hover,
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.margin_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=lambda e: self._handle_hist_hover(
                    e, self.margin_plot, self._last_margin_hist, self.margin_info, "margin"
                ),
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.best_response_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=lambda e: self._handle_hist_hover(
                    e,
                    self.best_response_plot,
                    self._last_best_hist,
                    self.best_response_info,
                    "best response",
                ),
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.response_heatmap_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_response_hover,
            )
        )

    def _handle_confusion_hover(self, event: Any) -> None:
        pos = self._plot_pos(event, self.confusion_plot)
        if pos is None or self._last_confusion is None:
            return
        predicted = int(np.floor(pos.x()))
        true = int(np.floor(pos.y()))
        matrix = self._last_confusion
        if 0 <= true < matrix.shape[0] and 0 <= predicted < matrix.shape[1]:
            count = int(matrix[true, predicted])
            total = int(np.sum(matrix[true]))
            percent = 100.0 * count / total if total else 0.0
            self._set_hover_info(
                self.confusion_info,
                f"true={true} | predicted={predicted} | count={count} | row share={percent:.1f}%",
            )

    def _handle_class_hover(self, event: Any) -> None:
        pos = self._plot_pos(event, self.class_plot)
        if pos is None or self._last_class_counts is None:
            return
        cls = int(round(pos.x()))
        if 0 <= cls < self._last_class_counts.size:
            count = int(self._last_class_counts[cls])
            total = int(np.sum(self._last_class_counts))
            percent = 100.0 * count / max(total, 1)
            self._set_hover_info(
                self.class_info,
                f"class={cls} | assigned E={count} | share of assigned={percent:.1f}%",
            )

    def _handle_prediction_hover(self, event: Any) -> None:
        pos = self._plot_pos(event, self.prediction_plot)
        if pos is None or self._last_prediction_counts is None:
            return
        cls = int(round(pos.x()))
        if 0 <= cls < self._last_prediction_counts.size:
            count = int(self._last_prediction_counts[cls])
            total = int(np.sum(self._last_prediction_counts))
            percent = 100.0 * count / max(total, 1)
            self._set_hover_info(
                self.prediction_info,
                f"predicted class={cls} | samples={count} | share={percent:.1f}%",
            )

    def _handle_per_class_hover(self, event: Any) -> None:
        pos = self._plot_pos(event, self.per_class_accuracy_plot)
        if pos is None or self._last_per_class_accuracy is None:
            return
        cls = int(round(pos.x()))
        if 0 <= cls < self._last_per_class_accuracy.size:
            self._set_hover_info(
                self.per_class_info,
                f"class={cls} | accuracy={float(self._last_per_class_accuracy[cls]):.2f}%",
            )

    def _handle_hist_hover(
        self,
        event: Any,
        plot: pg.PlotWidget,
        hist: tuple[np.ndarray, np.ndarray] | None,
        label: QLabel,
        name: str,
    ) -> None:
        pos = self._plot_pos(event, plot)
        if pos is None or hist is None:
            return
        counts, edges = hist
        bin_index = int(np.searchsorted(edges, pos.x(), side="right") - 1)
        if 0 <= bin_index < counts.size:
            self._set_hover_info(
                label,
                f"{name}: {edges[bin_index]:.4g} ... {edges[bin_index + 1]:.4g} | count={int(counts[bin_index])}",
            )

    def _handle_response_hover(self, event: Any) -> None:
        pos = self._plot_pos(event, self.response_heatmap_plot)
        if pos is None or self._last_response is None:
            return
        class_id = int(np.floor(pos.y()))
        sorted_pos = int(np.floor(pos.x()))
        response = self._last_response
        if 0 <= class_id < response.shape[0] and 0 <= sorted_pos < response.shape[1]:
            neuron_id = (
                int(self._last_response_order[sorted_pos])
                if self._last_response_order is not None
                else sorted_pos
            )
            value = float(response[class_id, sorted_pos])
            suffix = ""
            if self._last_result is not None and hasattr(self._last_result, "assignment"):
                assignment = self._last_result.assignment
                assigned = int(assignment.neuron_labels[neuron_id])
                margin = float(assignment.selectivity_margin[neuron_id])
                relative = float(assignment.relative_margin[neuron_id])
                best = float(assignment.best_response[neuron_id])
                suffix = (
                    f" | assigned={assigned} | best={best:.4g}"
                    f" | margin={margin:.4g} | relative={relative:.3f}"
                )
            self._set_hover_info(
                self.response_help_label,
                f"class={class_id} | E={neuron_id} | mean response={value:.4g}{suffix}",
            )

    @staticmethod
    def _plot_pos(event: Any, plot: pg.PlotWidget):
        pos = event[0] if isinstance(event, (tuple, list)) else event
        if not plot.sceneBoundingRect().contains(pos):
            return None
        return plot.plotItem.vb.mapSceneToView(pos)

    def _tr(self, key: str, **kwargs: Any) -> str:
        if self._translator is None:
            return key.format(**kwargs) if kwargs else key
        return self._translator.tr(key, **kwargs)
