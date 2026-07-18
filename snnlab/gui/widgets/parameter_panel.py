from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from snnlab.gui.widgets.help_dialog import HelpDialog
from snnlab.i18n import Translator


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    key: str
    label_key: str
    kind: str
    default: Any
    group: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    decimals: int = 6
    choices: tuple[tuple[str, str], ...] = ()
    help_topic: str | None = None


class ParameterRow(QWidget):
    """
    Combines a localized label, editor, and optional contextual-help button.

    Объединяет локализованную метку, editor и кнопку контекстной справки.
    """

    def __init__(
        self,
        *,
        spec: ParameterSpec,
        editor: QWidget,
        translator: Translator,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.spec = spec
        self.editor = editor
        self.translator = translator
        self.label = QLabel()
        self.help_button = QToolButton()
        self.help_button.setText("?")
        self.help_button.setFixedSize(22, 22)
        self.help_button.setVisible(False)
        self.help_button.clicked.connect(self._open_help)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QWidget()
        from PySide6.QtWidgets import QHBoxLayout

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.help_button)

        layout.addWidget(header)
        layout.addWidget(editor)
        self.retranslate()

    def set_help_mode(self, enabled: bool) -> None:
        self.help_button.setVisible(bool(enabled and self.spec.help_topic))

    def retranslate(self) -> None:
        self.label.setText(self.translator.tr(self.spec.label_key))
        if self.spec.help_topic:
            try:
                topic = self.translator.help_topic(self.spec.help_topic)
            except (KeyError, TypeError):
                self.help_button.setToolTip(self.spec.help_topic)
            else:
                tooltip = str(topic.get("short", ""))
                long_text = str(topic.get("long", ""))
                if long_text:
                    tooltip = f"{tooltip}\n\n{long_text}" if tooltip else long_text
                self.help_button.setToolTip(tooltip)

    def _open_help(self) -> None:
        if not self.spec.help_topic:
            return
        topic = self.translator.help_topic(self.spec.help_topic)
        HelpDialog(topic, parent=self).exec()


class ParameterPanel(QScrollArea):
    """
    Builds architecture-dependent parameter editors from declarative specs.

    Создаёт архитектурно-зависимые editors параметров из декларативных specs.
    """

    values_changed = Signal()

    def __init__(self, translator: Translator, parent=None) -> None:
        super().__init__(parent)
        self.translator = translator
        self._architecture = "dci"
        self._help_mode = False
        self._rows: dict[str, ParameterRow] = {}
        self._specs: dict[str, ParameterSpec] = {}
        self._group_boxes: dict[str, QGroupBox] = {}
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.addStretch(1)
        self.setWidget(self._content)
        self.setWidgetResizable(True)
        self.setMinimumWidth(330)

    def set_architecture(self, architecture: str) -> None:
        self._architecture = architecture
        specs = dci_parameter_specs() if architecture == "dci" else reservoir_parameter_specs()
        self._rebuild(specs)

    def set_help_mode(self, enabled: bool) -> None:
        self._help_mode = bool(enabled)
        for row in self._rows.values():
            row.set_help_mode(self._help_mode)

    def values(self) -> dict[str, Any]:
        result: dict[str, Any] = {"architecture": self._architecture}
        for key, row in self._rows.items():
            editor = row.editor
            if isinstance(editor, QSpinBox):
                result[key] = editor.value()
            elif isinstance(editor, QDoubleSpinBox):
                result[key] = editor.value()
            elif isinstance(editor, QCheckBox):
                result[key] = editor.isChecked()
            elif isinstance(editor, QComboBox):
                result[key] = editor.currentData()
            else:
                raise TypeError(f"Unsupported editor for {key!r}: {type(editor)!r}")
        return result

    def set_values(self, values: dict[str, Any]) -> None:
        """
        Applies known values without rebuilding the parameter schema.

        Применяет известные значения без перестроения схемы параметров.
        """
        for key, value in values.items():
            row = self._rows.get(key)
            if row is None:
                continue
            editor = row.editor
            editor.blockSignals(True)
            try:
                if isinstance(editor, QSpinBox):
                    editor.setValue(int(value))
                elif isinstance(editor, QDoubleSpinBox):
                    editor.setValue(float(value))
                elif isinstance(editor, QCheckBox):
                    editor.setChecked(bool(value))
                elif isinstance(editor, QComboBox):
                    index = editor.findData(value)
                    if index >= 0:
                        editor.setCurrentIndex(index)
            finally:
                editor.blockSignals(False)

    def retranslate(self) -> None:
        for row in self._rows.values():
            row.retranslate()
        for group_key, box in self._group_boxes.items():
            box.setTitle(self.translator.tr(f"gui.groups.{group_key}"))
        for key, row in self._rows.items():
            if isinstance(row.editor, QComboBox):
                spec = self._specs[key]
                current = row.editor.currentData()
                row.editor.blockSignals(True)
                row.editor.clear()
                for value, label_key in spec.choices:
                    row.editor.addItem(self.translator.tr(label_key), value)
                index = row.editor.findData(current)
                row.editor.setCurrentIndex(max(0, index))
                row.editor.blockSignals(False)

    def _rebuild(self, specs: list[ParameterSpec]) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._rows.clear()
        self._specs = {spec.key: spec for spec in specs}
        self._group_boxes.clear()

        group_order: list[str] = []
        for spec in specs:
            if spec.group not in group_order:
                group_order.append(spec.group)

        group_layouts: dict[str, QVBoxLayout] = {}
        for group in group_order:
            box = QGroupBox(self.translator.tr(f"gui.groups.{group}"))
            layout = QVBoxLayout(box)
            self._content_layout.addWidget(box)
            self._group_boxes[group] = box
            group_layouts[group] = layout

        for spec in specs:
            editor = self._make_editor(spec)
            row = ParameterRow(spec=spec, editor=editor, translator=self.translator)
            row.set_help_mode(self._help_mode)
            self._rows[spec.key] = row
            group_layouts[spec.group].addWidget(row)

        self._content_layout.addStretch(1)

        if self._architecture == "reservoir":
            dataset_row = self._rows.get("dataset")
            if dataset_row is not None and isinstance(dataset_row.editor, QComboBox):
                dataset_row.editor.currentIndexChanged.connect(
                    lambda *_: self._apply_reservoir_dataset_preset()
                )

    def _apply_reservoir_dataset_preset(self) -> None:
        """
        Applies conservative architecture defaults when the reservoir dataset changes.

        The legacy notebook used one weight scale for both Iris and MNIST. With
        784 Poisson inputs that setting drives the MNIST reservoir into near-total
        spike saturation, while it is reasonable for the much smaller Iris encoder.

        Применяет консервативные архитектурные значения при смене reservoir-датасета.

        В старом блокноте одна шкала весов использовалась и для Iris, и для MNIST.
        При 784 Poisson-входах она вводит MNIST-reservoir почти в полное насыщение,
        хотя для компактного Iris-encoder остаётся приемлемой.
        """
        if self._architecture != "reservoir":
            return
        dataset_row = self._rows.get("dataset")
        if dataset_row is None or not isinstance(dataset_row.editor, QComboBox):
            return
        dataset = str(dataset_row.editor.currentData())
        presets = {
            "iris": {
                "n_reservoir": 150,
                "input_scale": 10.0,
                "recurrent_scale": 4.0,
                "select_k": 40,
            },
            "mnist": {
                "n_reservoir": 800,
                "input_scale": 2.0,
                "recurrent_scale": 1.0,
                "select_k": 400,
            },
        }
        values = presets.get(dataset)
        if values is None:
            return
        self.set_values(values)
        self.values_changed.emit()

    def _make_editor(self, spec: ParameterSpec) -> QWidget:
        if spec.kind == "int":
            editor = QSpinBox()
            editor.setRange(
                int(spec.minimum if spec.minimum is not None else -2_000_000_000),
                int(spec.maximum if spec.maximum is not None else 2_000_000_000),
            )
            editor.setSingleStep(int(spec.step or 1))
            editor.setValue(int(spec.default))
            editor.valueChanged.connect(lambda *_: self.values_changed.emit())
            return editor

        if spec.kind == "float":
            editor = QDoubleSpinBox()
            editor.setDecimals(spec.decimals)
            editor.setRange(
                float(spec.minimum if spec.minimum is not None else -1e12),
                float(spec.maximum if spec.maximum is not None else 1e12),
            )
            editor.setSingleStep(float(spec.step or 0.1))
            editor.setValue(float(spec.default))
            editor.valueChanged.connect(lambda *_: self.values_changed.emit())
            return editor

        if spec.kind == "bool":
            editor = QCheckBox()
            editor.setChecked(bool(spec.default))
            editor.toggled.connect(lambda *_: self.values_changed.emit())
            return editor

        if spec.kind == "choice":
            editor = QComboBox()
            for value, label_key in spec.choices:
                editor.addItem(self.translator.tr(label_key), value)
            index = editor.findData(spec.default)
            editor.setCurrentIndex(max(0, index))
            editor.currentIndexChanged.connect(lambda *_: self.values_changed.emit())
            return editor

        raise ValueError(f"Unsupported parameter kind: {spec.kind!r}")


def _method_choices() -> tuple[tuple[str, str], ...]:
    ids = (
        "explicit_euler",
        "semi_euler",
        "semi_implicit_euler",
        "implicit_euler",
        "midpoint",
        "semi_midpoint",
        "semi_implicit_midpoint",
    )
    return tuple((method_id, f"components.numerical_methods.{method_id}") for method_id in ids)


def _common_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            "seed",
            "gui.params.seed",
            "int",
            52,
            "run",
            0,
            2_000_000_000,
            1,
            help_topic="experiment.seed",
        ),
        ParameterSpec(
            "samples_per_epoch",
            "gui.params.samples_per_epoch",
            "int",
            300,
            "run",
            1,
            100_000,
            1,
            help_topic="experiment.samples_per_epoch",
        ),
        ParameterSpec(
            "epochs",
            "gui.params.epochs",
            "int",
            1,
            "run",
            1,
            10_000,
            1,
            help_topic="experiment.epochs",
        ),
        ParameterSpec(
            "checkpoint_every",
            "gui.params.checkpoint_every",
            "int",
            50,
            "run",
            0,
            100_000,
            1,
            help_topic="runtime.checkpoint_every",
        ),
        ParameterSpec(
            "live_frame_every_steps",
            "gui.params.live_frame_every_steps",
            "int",
            50,
            "run",
            1,
            100_000,
            1,
            help_topic="runtime.live_frame_every_steps",
        ),
        ParameterSpec(
            "train_pool_size",
            "gui.params.train_pool_size",
            "int",
            1000,
            "data",
            10,
            60_000,
            10,
            help_topic="dataset.train_pool_size",
        ),
        ParameterSpec(
            "test_pool_size",
            "gui.params.test_pool_size",
            "int",
            200,
            "data",
            10,
            10_000,
            10,
            help_topic="dataset.test_pool_size",
        ),
    ]


def dci_parameter_specs() -> list[ParameterSpec]:
    return _common_specs() + [
        ParameterSpec(
            "n_exc", "gui.params.n_exc", "int", 400, "network", 2, 20_000, 1, help_topic="dci.n_exc"
        ),
        ParameterSpec(
            "dt_ms",
            "gui.params.dt_ms",
            "float",
            0.1,
            "simulation",
            0.001,
            10.0,
            0.01,
            4,
            help_topic="simulation.dt_ms",
        ),
        ParameterSpec(
            "stimulus_ms",
            "gui.params.stimulus_ms",
            "float",
            350.0,
            "simulation",
            1.0,
            10000.0,
            10.0,
            2,
            help_topic="dci.stimulus_ms",
        ),
        ParameterSpec(
            "rest_ms",
            "gui.params.rest_ms",
            "float",
            150.0,
            "simulation",
            0.0,
            10000.0,
            10.0,
            2,
            help_topic="dci.rest_ms",
        ),
        ParameterSpec(
            "exc_method",
            "gui.params.exc_method",
            "choice",
            "explicit_euler",
            "simulation",
            choices=_method_choices(),
            help_topic="dci.exc_method",
        ),
        ParameterSpec(
            "inh_method",
            "gui.params.inh_method",
            "choice",
            "explicit_euler",
            "simulation",
            choices=_method_choices(),
            help_topic="dci.inh_method",
        ),
        ParameterSpec(
            "input_gain",
            "gui.params.input_gain",
            "float",
            0.60,
            "network",
            0.0,
            100.0,
            0.05,
            4,
            help_topic="dci.input_gain",
        ),
        ParameterSpec(
            "weight_exc_inh",
            "gui.params.weight_exc_inh",
            "float",
            0.30,
            "network",
            0.0,
            100.0,
            0.05,
            4,
            help_topic="dci.weight_exc_inh",
        ),
        ParameterSpec(
            "target_total_inhibition",
            "gui.params.target_total_inhibition",
            "float",
            20.0,
            "network",
            0.0,
            1000.0,
            1.0,
            4,
            help_topic="dci.target_total_inhibition",
        ),
        ParameterSpec(
            "base_max_rate_hz",
            "gui.params.base_max_rate_hz",
            "float",
            63.75,
            "presentation",
            0.0,
            5000.0,
            1.0,
            2,
            help_topic="dci.base_max_rate_hz",
        ),
        ParameterSpec(
            "rate_increment_hz",
            "gui.params.rate_increment_hz",
            "float",
            32.0,
            "presentation",
            0.0,
            5000.0,
            1.0,
            2,
            help_topic="dci.rate_increment_hz",
        ),
        ParameterSpec(
            "min_exc_spikes",
            "gui.params.min_exc_spikes",
            "int",
            5,
            "presentation",
            0,
            1_000_000,
            1,
            help_topic="dci.min_exc_spikes",
        ),
        ParameterSpec(
            "max_attempts",
            "gui.params.max_attempts",
            "int",
            5,
            "presentation",
            1,
            100,
            1,
            help_topic="dci.max_attempts",
        ),
        ParameterSpec(
            "tau_pre_ms",
            "gui.params.tau_pre_ms",
            "float",
            20.0,
            "learning",
            0.001,
            10000.0,
            1.0,
            3,
            help_topic="dci.tau_pre_ms",
        ),
        ParameterSpec(
            "eta",
            "gui.params.eta",
            "float",
            0.00003,
            "learning",
            0.0,
            1.0,
            0.00001,
            8,
            help_topic="dci.eta",
        ),
        ParameterSpec(
            "x_target",
            "gui.params.x_target",
            "float",
            0.40,
            "learning",
            -100.0,
            100.0,
            0.05,
            4,
            help_topic="dci.x_target",
        ),
        ParameterSpec(
            "mu",
            "gui.params.mu",
            "float",
            0.20,
            "learning",
            0.0,
            10.0,
            0.05,
            4,
            help_topic="dci.mu",
        ),
        ParameterSpec(
            "target_spikes_per_sample",
            "gui.params.target_spikes_per_sample",
            "float",
            0.25,
            "homeostasis",
            0.0,
            10000.0,
            0.05,
            4,
            help_topic="dci.target_spikes_per_sample",
        ),
        ParameterSpec(
            "homeo_learning_rate",
            "gui.params.homeo_learning_rate",
            "float",
            0.005,
            "homeostasis",
            0.0,
            100.0,
            0.001,
            6,
            help_topic="dci.homeo_learning_rate",
        ),
        ParameterSpec(
            "homeo_max_current",
            "gui.params.homeo_max_current",
            "float",
            8.0,
            "homeostasis",
            0.0,
            10000.0,
            0.5,
            3,
            help_topic="dci.homeo_max_current",
        ),
        ParameterSpec(
            "exc_a",
            "gui.params.exc_a",
            "float",
            0.02,
            "neuron_exc",
            0.0,
            100.0,
            0.01,
            4,
            help_topic="izhikevich.a",
        ),
        ParameterSpec(
            "exc_b",
            "gui.params.exc_b",
            "float",
            0.20,
            "neuron_exc",
            -100.0,
            100.0,
            0.05,
            4,
            help_topic="izhikevich.b",
        ),
        ParameterSpec(
            "exc_c",
            "gui.params.exc_c",
            "float",
            -65.0,
            "neuron_exc",
            -500.0,
            500.0,
            1.0,
            3,
            help_topic="izhikevich.c",
        ),
        ParameterSpec(
            "exc_d",
            "gui.params.exc_d",
            "float",
            8.0,
            "neuron_exc",
            -500.0,
            500.0,
            1.0,
            3,
            help_topic="izhikevich.d",
        ),
        ParameterSpec(
            "inh_a",
            "gui.params.inh_a",
            "float",
            0.10,
            "neuron_inh",
            0.0,
            100.0,
            0.01,
            4,
            help_topic="izhikevich.a",
        ),
        ParameterSpec(
            "inh_b",
            "gui.params.inh_b",
            "float",
            0.20,
            "neuron_inh",
            -100.0,
            100.0,
            0.05,
            4,
            help_topic="izhikevich.b",
        ),
        ParameterSpec(
            "inh_c",
            "gui.params.inh_c",
            "float",
            -65.0,
            "neuron_inh",
            -500.0,
            500.0,
            1.0,
            3,
            help_topic="izhikevich.c",
        ),
        ParameterSpec(
            "inh_d",
            "gui.params.inh_d",
            "float",
            2.0,
            "neuron_inh",
            -500.0,
            500.0,
            1.0,
            3,
            help_topic="izhikevich.d",
        ),
    ]


def reservoir_parameter_specs() -> list[ParameterSpec]:
    return _common_specs() + [
        ParameterSpec(
            "dataset",
            "gui.params.dataset",
            "choice",
            "iris",
            "data",
            choices=(("iris", "gui.datasets.iris"), ("mnist", "gui.datasets.mnist")),
            help_topic="dataset.kind",
        ),
        ParameterSpec(
            "iris_neurons_per_feature",
            "gui.params.iris_neurons_per_feature",
            "int",
            8,
            "data",
            1,
            100,
            1,
            help_topic="reservoir.iris_neurons_per_feature",
        ),
        ParameterSpec(
            "n_reservoir",
            "gui.params.n_reservoir",
            "int",
            150,
            "network",
            1,
            100_000,
            1,
            help_topic="reservoir.n_reservoir",
        ),
        ParameterSpec(
            "dt_ms",
            "gui.params.dt_ms",
            "float",
            0.5,
            "simulation",
            0.001,
            10.0,
            0.05,
            4,
            help_topic="simulation.dt_ms",
        ),
        ParameterSpec(
            "simulation_ms",
            "gui.params.simulation_ms",
            "float",
            100.0,
            "simulation",
            1.0,
            10000.0,
            10.0,
            2,
            help_topic="reservoir.simulation_ms",
        ),
        ParameterSpec(
            "max_rate_hz",
            "gui.params.max_rate_hz",
            "float",
            100.0,
            "simulation",
            0.0,
            5000.0,
            5.0,
            2,
            help_topic="reservoir.max_rate_hz",
        ),
        ParameterSpec(
            "numerical_method",
            "gui.params.numerical_method",
            "choice",
            "semi_euler",
            "simulation",
            choices=_method_choices(),
            help_topic="reservoir.numerical_method",
        ),
        ParameterSpec(
            "input_density",
            "gui.params.input_density",
            "float",
            0.20,
            "network",
            0.0,
            1.0,
            0.01,
            4,
            help_topic="reservoir.input_density",
        ),
        ParameterSpec(
            "recurrent_density",
            "gui.params.recurrent_density",
            "float",
            0.08,
            "network",
            0.0,
            1.0,
            0.01,
            4,
            help_topic="reservoir.recurrent_density",
        ),
        ParameterSpec(
            "excitatory_ratio",
            "gui.params.excitatory_ratio",
            "float",
            0.80,
            "network",
            0.001,
            1.0,
            0.05,
            4,
            help_topic="reservoir.excitatory_ratio",
        ),
        ParameterSpec(
            "input_scale",
            "gui.params.input_scale",
            "float",
            10.0,
            "network",
            0.0,
            10000.0,
            0.5,
            3,
            help_topic="reservoir.input_scale",
        ),
        ParameterSpec(
            "recurrent_scale",
            "gui.params.recurrent_scale",
            "float",
            4.0,
            "network",
            0.0,
            10000.0,
            0.5,
            3,
            help_topic="reservoir.recurrent_scale",
        ),
        ParameterSpec(
            "bias_current",
            "gui.params.bias_current",
            "float",
            2.0,
            "network",
            -10000.0,
            10000.0,
            0.5,
            3,
            help_topic="reservoir.bias_current",
        ),
        ParameterSpec(
            "tau_syn_ms",
            "gui.params.tau_syn_ms",
            "float",
            8.0,
            "network",
            0.001,
            10000.0,
            0.5,
            3,
            help_topic="reservoir.tau_syn_ms",
        ),
        ParameterSpec(
            "readout",
            "gui.params.readout",
            "choice",
            "ridge",
            "readout",
            choices=(
                ("ridge", "components.readouts.ridge"),
                ("logreg", "components.readouts.logreg"),
                ("linear_svm", "components.readouts.linear_svm"),
                ("rbf_svm", "components.readouts.rbf_svm"),
            ),
            help_topic="reservoir.readout",
        ),
        ParameterSpec(
            "use_feature_selection",
            "gui.params.use_feature_selection",
            "bool",
            True,
            "readout",
            help_topic="reservoir.use_feature_selection",
        ),
        ParameterSpec(
            "select_k",
            "gui.params.select_k",
            "int",
            40,
            "readout",
            1,
            100_000,
            1,
            help_topic="reservoir.select_k",
        ),
        ParameterSpec(
            "neuron_a",
            "gui.params.neuron_a",
            "float",
            0.02,
            "neuron",
            0.0,
            100.0,
            0.01,
            4,
            help_topic="izhikevich.a",
        ),
        ParameterSpec(
            "neuron_b",
            "gui.params.neuron_b",
            "float",
            0.20,
            "neuron",
            -100.0,
            100.0,
            0.05,
            4,
            help_topic="izhikevich.b",
        ),
        ParameterSpec(
            "neuron_c",
            "gui.params.neuron_c",
            "float",
            -65.0,
            "neuron",
            -500.0,
            500.0,
            1.0,
            3,
            help_topic="izhikevich.c",
        ),
        ParameterSpec(
            "neuron_d",
            "gui.params.neuron_d",
            "float",
            8.0,
            "neuron",
            -500.0,
            500.0,
            1.0,
            3,
            help_topic="izhikevich.d",
        ),
    ]
