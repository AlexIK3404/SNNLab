from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class TrainingView(QWidget):
    """
    Visualizes architecture-specific learning state.

    DCI exposes trainable Input -> E synapses, so the view renders receptive
    fields. Reservoir / LSM-like models keep recurrent weights fixed; for them
    the useful training objects are collected spike-count features and the
    external readout fitted on top of those features.

    Визуализирует архитектурно-зависимое состояние обучения.

    DCI содержит обучаемые связи Input -> E, поэтому вкладка показывает
    receptive fields. В Reservoir / LSM-like рекуррентные веса фиксированы;
    полезными объектами обучения являются собранные spike-count признаки и
    внешний readout, обучаемый поверх этих признаков.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._architecture = "dci"
        self._translator = None
        self._help_mode = False

        # DCI state / Состояние DCI.
        self._last_weights: np.ndarray | None = None
        self._last_indices: np.ndarray | None = None
        self._last_position: int | None = None
        self._last_w_min: float | None = None
        self._last_w_max: float | None = None
        self._last_render_indices: np.ndarray | None = None
        self._last_rf_side: int | None = None
        self._last_rf_columns: int | None = None
        self._last_rf_gap: int = 1

        # Reservoir state / Состояние Reservoir.
        self._reservoir_class_sums: dict[int, np.ndarray] = {}
        self._reservoir_class_counts: dict[int, int] = {}
        self._reservoir_current_feature: np.ndarray | None = None
        self._reservoir_current_label: int | None = None
        self._reservoir_current_position: int | None = None
        self._reservoir_readout_fitted = False
        self._reservoir_order: np.ndarray | None = None
        self._reservoir_class_labels: np.ndarray | None = None
        self._reservoir_display_matrix: np.ndarray | None = None

        self._mouse_proxies: list[Any] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setVisible(False)
        self.fixed_scale_checkbox = QCheckBox()
        self.fixed_scale_checkbox.setChecked(False)
        self.fixed_scale_checkbox.toggled.connect(lambda *_: self._rerender_dci())
        header.addWidget(self.info_label, 1)
        header.addWidget(self.fixed_scale_checkbox, 0)
        root.addLayout(header)

        # DCI receptive fields / Receptive fields DCI.
        self.rf_plot = pg.PlotWidget(title="Receptive fields")
        self.rf_plot.setAspectLocked(True)
        self.rf_plot.getViewBox().invertY(True)
        self.rf_plot.hideAxis("bottom")
        self.rf_plot.hideAxis("left")
        self.rf_image = pg.ImageItem(axisOrder="row-major")
        self.rf_plot.addItem(self.rf_image)
        root.addWidget(self.rf_plot, 1)

        # Reservoir feature diagnostics / Диагностика признаков Reservoir.
        self.reservoir_splitter = QSplitter(Qt.Orientation.Vertical)
        self.reservoir_splitter.setChildrenCollapsible(False)
        self.reservoir_splitter.setHandleWidth(6)

        self.reservoir_class_plot = pg.PlotWidget(title="Reservoir class means")
        self.reservoir_class_plot.showGrid(x=True, y=True, alpha=0.12)
        self.reservoir_class_plot.getViewBox().invertY(True)
        self.reservoir_class_image = pg.ImageItem(axisOrder="row-major")
        self.reservoir_class_plot.addItem(self.reservoir_class_image)
        self.reservoir_splitter.addWidget(self.reservoir_class_plot)

        self.reservoir_current_plot = pg.PlotWidget(title="Current reservoir feature")
        self.reservoir_current_plot.showGrid(x=True, y=True, alpha=0.18)
        self.reservoir_current_curve = self.reservoir_current_plot.plot(
            pen=pg.mkPen(80, 170, 255, width=1.5)
        )
        self.reservoir_splitter.addWidget(self.reservoir_current_plot)
        self.reservoir_splitter.setStretchFactor(0, 2)
        self.reservoir_splitter.setStretchFactor(1, 1)
        self.reservoir_splitter.setSizes([520, 260])
        root.addWidget(self.reservoir_splitter, 1)

        self.hover_label = QLabel()
        self.hover_label.setWordWrap(True)
        self.hover_label.setVisible(False)
        root.addWidget(self.hover_label)

        self.footer_label = QLabel()
        self.footer_label.setWordWrap(True)
        self.footer_label.setVisible(False)
        root.addWidget(self.footer_label)

        self._mouse_proxies.append(
            pg.SignalProxy(
                self.rf_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_rf_hover,
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.reservoir_class_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_reservoir_class_hover,
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.reservoir_current_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_reservoir_current_hover,
            )
        )
        self.set_architecture("dci")

    def refresh_layout(self) -> None:
        """Refreshes visible plot viewports after tab/window changes.

        Обновляет видимые viewport графиков после переключения вкладки или
        изменения состояния главного окна.
        """
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.reservoir_splitter.refresh()
        self.reservoir_splitter.updateGeometry()
        for plot in (
            self.rf_plot,
            self.reservoir_class_plot,
            self.reservoir_current_plot,
        ):
            plot.updateGeometry()
            plot.viewport().update()
            plot.update()

    def set_help_mode(self, enabled: bool) -> None:
        """Shows explanatory text only in study mode.

        Показывает поясняющий текст только в режиме изучения.
        """
        self._help_mode = bool(enabled)
        self._refresh_texts()

    @staticmethod
    def _set_hover_info(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setVisible(True)

    def set_architecture(self, architecture: str) -> None:
        """Switches between DCI and Reservoir diagnostics.

        Переключает диагностику DCI и Reservoir.
        """
        self._architecture = str(architecture)
        is_dci = self._architecture == "dci"
        self.rf_plot.setVisible(is_dci)
        self.fixed_scale_checkbox.setVisible(is_dci)
        self.reservoir_splitter.setVisible(not is_dci)
        self.hover_label.clear()
        self.hover_label.setVisible(False)
        self._refresh_texts()
        if is_dci:
            self._rerender_dci()
        else:
            self._rerender_reservoir()

    def retranslate(self, translator) -> None:
        """Refreshes localized labels after a runtime language switch.

        Обновляет локализованные подписи после переключения языка.
        """
        self._translator = translator
        self._refresh_texts()
        if self._architecture == "reservoir":
            self._rerender_reservoir()

    def reset(self) -> None:
        """Clears visible learning state. / Очищает видимое состояние обучения."""
        self._last_weights = None
        self._last_indices = None
        self._last_position = None
        self._last_w_min = None
        self._last_w_max = None
        self._last_render_indices = None
        self._last_rf_side = None
        self._last_rf_columns = None
        self.rf_image.clear()

        self._reservoir_class_sums.clear()
        self._reservoir_class_counts.clear()
        self._reservoir_current_feature = None
        self._reservoir_current_label = None
        self._reservoir_current_position = None
        self._reservoir_readout_fitted = False
        self._reservoir_order = None
        self._reservoir_class_labels = None
        self._reservoir_display_matrix = None
        self.reservoir_class_image.clear()
        self.reservoir_current_curve.setData([])
        self.reservoir_class_plot.getAxis("left").setTicks([])

        self.hover_label.clear()
        self.hover_label.setVisible(False)
        self._refresh_texts()

    def handle_learning_snapshot(self, payload: dict[str, Any]) -> None:
        """Consumes a copied DCI learning snapshot emitted by the trainer.

        Принимает копию DCI learning snapshot, отправленную trainer-ом.
        """
        if str(payload.get("architecture", "dci")) != "dci":
            return

        raw_weights = payload.get("w_input_exc")
        if raw_weights is None:
            return
        weights = np.asarray(raw_weights, dtype=np.float64)
        if weights.ndim != 2 or weights.size == 0:
            return

        self._last_weights = weights.copy()
        indices = payload.get("neuron_indices")
        self._last_indices = None if indices is None else np.asarray(indices, dtype=np.int64).copy()
        self._last_position = int(payload.get("position", 0))
        self._last_w_min = float(payload.get("w_min", np.nan))
        self._last_w_max = float(payload.get("w_max", np.nan))
        self._rerender_dci()

    def handle_sample_end(self, payload: dict[str, Any]) -> None:
        """Accumulates reservoir features from completed samples.

        Накапливает reservoir-признаки завершённых sample.
        """
        if self._architecture != "reservoir":
            return
        raw_counts = payload.get("spike_counts")
        if raw_counts is None:
            return
        spike_counts = np.asarray(raw_counts, dtype=np.float64).reshape(-1)
        if spike_counts.size == 0:
            return
        n_steps = max(1, int(payload.get("n_steps", 1)))
        feature = spike_counts / float(n_steps)
        label = int(payload.get("label", -1))

        existing = self._reservoir_class_sums.get(label)
        if existing is None or existing.shape != feature.shape:
            self._reservoir_class_sums[label] = np.zeros_like(feature)
            self._reservoir_class_counts[label] = 0
        self._reservoir_class_sums[label] += feature
        self._reservoir_class_counts[label] += 1
        self._reservoir_current_feature = feature.copy()
        self._reservoir_current_label = label
        self._reservoir_current_position = int(payload.get("position", 0))
        self._rerender_reservoir()

    def update_from_session(self, session: Any) -> None:
        """Displays the current model after session creation or loading.

        Сразу отображает текущую модель после создания или загрузки session.
        """
        self.set_architecture(session.kind)
        if session.kind == "dci":
            model = session.engine.state.model
            weights = np.asarray(model.connectivity.w_input_exc, dtype=np.float64).copy()
            self._last_weights = weights
            self._last_indices = None
            self._last_position = int(session.position)
            self._last_w_min = float(model.stdp_cfg.w_min)
            self._last_w_max = float(model.stdp_cfg.w_max)
            self._rerender_dci()
            return

        state = session.engine.state
        self._reservoir_class_sums.clear()
        self._reservoir_class_counts.clear()
        for feature, label in zip(state.features, state.labels, strict=True):
            values = np.asarray(feature, dtype=np.float64).reshape(-1)
            class_id = int(label)
            existing = self._reservoir_class_sums.get(class_id)
            if existing is None or existing.shape != values.shape:
                self._reservoir_class_sums[class_id] = np.zeros_like(values)
                self._reservoir_class_counts[class_id] = 0
            self._reservoir_class_sums[class_id] += values
            self._reservoir_class_counts[class_id] += 1

        if state.features:
            self._reservoir_current_feature = (
                np.asarray(state.features[-1], dtype=np.float64).reshape(-1).copy()
            )
            self._reservoir_current_label = int(state.labels[-1])
            self._reservoir_current_position = int(session.position)
        else:
            self._reservoir_current_feature = None
            self._reservoir_current_label = None
            self._reservoir_current_position = int(session.position)
        self._reservoir_readout_fitted = state.readout_model is not None
        self._rerender_reservoir()

    # ------------------------------------------------------------------
    # DCI rendering / Отрисовка DCI
    # ------------------------------------------------------------------
    def _rerender_dci(self) -> None:
        if self._architecture != "dci" or self._last_weights is None:
            return

        weights = self._last_weights
        n_input, n_exc = weights.shape
        side = int(round(math.sqrt(n_input)))
        if side * side != n_input:
            self.rf_image.clear()
            self._set_dci_footer("gui.training.non_square_input", n_input=n_input)
            return

        max_fields = min(100, n_exc)
        if self._last_indices is not None and self._last_indices.size:
            indices = self._last_indices[:max_fields]
        else:
            # EN: Stable evenly-spaced indices prevent visual jumping when
            #     activity rankings change.
            # RU: Стабильные равномерные индексы не дают мозаике прыгать при
            #     изменении рейтинга активности.
            indices = np.unique(np.linspace(0, n_exc - 1, num=max_fields, dtype=np.int64))

        columns = min(10, len(indices))
        gap = 1
        self._last_render_indices = indices.copy()
        self._last_rf_side = side
        self._last_rf_columns = columns
        self._last_rf_gap = gap
        fields = weights[:, indices].T.reshape(len(indices), side, side)
        mosaic = self._build_mosaic(fields, columns=columns, gap=gap)

        if self.fixed_scale_checkbox.isChecked():
            low = self._last_w_min if self._last_w_min is not None else float(np.nanmin(weights))
            high = self._last_w_max if self._last_w_max is not None else float(np.nanmax(weights))
            scale_mode_key = "gui.training.scale_fixed"
        else:
            finite = weights[np.isfinite(weights)]
            low = float(np.min(finite)) if finite.size else 0.0
            high = float(np.percentile(finite, 99.5)) if finite.size else 1.0
            scale_mode_key = "gui.training.scale_snapshot"

        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high) or high <= low:
            high = low + 1e-12

        self.rf_image.setImage(mosaic, autoLevels=False, levels=(low, high))
        self.rf_plot.autoRange()
        self._set_dci_footer(
            "gui.training.footer",
            position=self._last_position if self._last_position is not None else "-",
            shown=len(indices),
            total=n_exc,
            scale=self._tr(scale_mode_key),
        )

    @staticmethod
    def _build_mosaic(fields: np.ndarray, *, columns: int, gap: int) -> np.ndarray:
        """Packs [N, H, W] fields into a 2D mosaic with NaN separators.

        Упаковывает поля [N, H, W] в 2D-мозаику с NaN-разделителями.
        """
        n_fields, height, width = fields.shape
        columns = max(1, int(columns))
        rows = int(math.ceil(n_fields / columns))
        out_h = rows * height + max(0, rows - 1) * gap
        out_w = columns * width + max(0, columns - 1) * gap
        mosaic = np.full((out_h, out_w), np.nan, dtype=np.float64)

        for index, field in enumerate(fields):
            row = index // columns
            col = index % columns
            y0 = row * (height + gap)
            x0 = col * (width + gap)
            mosaic[y0 : y0 + height, x0 : x0 + width] = field
        return mosaic

    # ------------------------------------------------------------------
    # Reservoir rendering / Отрисовка Reservoir
    # ------------------------------------------------------------------
    def _rerender_reservoir(self) -> None:
        if self._architecture != "reservoir":
            return
        if not self._reservoir_class_sums:
            self.reservoir_class_image.clear()
            self.reservoir_current_curve.setData([])
            self._set_reservoir_footer(waiting=True)
            return

        labels = np.asarray(sorted(self._reservoir_class_sums), dtype=np.int64)
        means = np.vstack(
            [
                self._reservoir_class_sums[int(label)]
                / max(1, self._reservoir_class_counts[int(label)])
                for label in labels
            ]
        )
        if means.ndim != 2 or means.size == 0:
            return

        # EN: Group neurons by their preferred class and sort within each group
        #     by response strength. The x-axis remains an explicit "display
        #     order"; hover reveals the original neuron index.
        # RU: Группируем нейроны по предпочитаемому классу и сортируем внутри
        #     группы по силе отклика. Ось X остаётся явным "порядком показа";
        #     hover показывает исходный индекс нейрона.
        preferred_row = np.argmax(means, axis=0)
        best_response = np.max(means, axis=0)
        order = np.lexsort((-best_response, preferred_row)).astype(np.int64)
        display = means[:, order]
        self._reservoir_order = order
        self._reservoir_class_labels = labels
        self._reservoir_display_matrix = display

        finite = display[np.isfinite(display)]
        low = 0.0
        high = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        if not np.isfinite(high) or high <= low:
            high = low + 1e-12
        self.reservoir_class_image.setImage(
            display,
            autoLevels=False,
            levels=(low, high),
        )
        self.reservoir_class_plot.getAxis("left").setTicks(
            [[(float(i), str(int(label))) for i, label in enumerate(labels)]]
        )
        self.reservoir_class_plot.setXRange(0.0, float(display.shape[1]), padding=0.01)
        self.reservoir_class_plot.setYRange(0.0, float(display.shape[0]), padding=0.03)

        if self._reservoir_current_feature is not None:
            current = self._reservoir_current_feature
            if current.size == order.size:
                shown = current[order]
            else:
                shown = current
            self.reservoir_current_curve.setData(np.arange(shown.size, dtype=np.float64), shown)
            self.reservoir_current_plot.enableAutoRange(axis="y", enable=True)

        self._set_reservoir_footer(waiting=False)

    def _set_reservoir_footer(self, *, waiting: bool) -> None:
        if waiting:
            self.footer_label.setText(self._tr("gui.training.reservoir_waiting"))
            self.footer_label.setVisible(True)
            return

        total_samples = int(sum(self._reservoir_class_counts.values()))
        n_classes = int(len(self._reservoir_class_counts))
        if self._reservoir_display_matrix is None:
            active_features = 0
            total_features = 0
        else:
            total_features = int(self._reservoir_display_matrix.shape[1])
            active_features = int(
                np.count_nonzero(np.max(self._reservoir_display_matrix, axis=0) > 0.0)
            )
        readout_key = (
            "gui.training.reservoir_readout_fitted"
            if self._reservoir_readout_fitted
            else "gui.training.reservoir_readout_not_fitted"
        )
        self.footer_label.setText(
            self._tr(
                "gui.training.reservoir_footer",
                samples=total_samples,
                classes=n_classes,
                active=active_features,
                total=total_features,
                readout=self._tr(readout_key),
            )
        )
        self.footer_label.setVisible(True)

    # ------------------------------------------------------------------
    # Hover / Наведение курсора
    # ------------------------------------------------------------------
    def _handle_rf_hover(self, event: Any) -> None:
        """Shows exact receptive-field tile/pixel information.

        Показывает точную информацию о плитке/pixel receptive field.
        """
        if self._last_weights is None or self._last_render_indices is None:
            return
        pos = event[0] if isinstance(event, (tuple, list)) else event
        if not self.rf_plot.sceneBoundingRect().contains(pos):
            return
        point = self.rf_plot.plotItem.vb.mapSceneToView(pos)
        side = int(self._last_rf_side or 0)
        columns = int(self._last_rf_columns or 0)
        gap = int(self._last_rf_gap)
        if side <= 0 or columns <= 0:
            return
        x = int(np.floor(point.x()))
        y = int(np.floor(point.y()))
        tile_w = side + gap
        tile_h = side + gap
        col = x // tile_w
        row = y // tile_h
        px = x - col * tile_w
        py = y - row * tile_h
        tile_index = row * columns + col
        if px < 0 or py < 0 or px >= side or py >= side:
            return
        if tile_index < 0 or tile_index >= self._last_render_indices.size:
            return
        neuron_id = int(self._last_render_indices[tile_index])
        pixel_index = py * side + px
        value = float(self._last_weights[pixel_index, neuron_id])
        self._set_hover_info(
            self.hover_label,
            f"E={neuron_id} | tile={tile_index} | pixel=({px}, {py}) | "
            f"input index={pixel_index} | weight={value:.6g}",
        )

    def _handle_reservoir_class_hover(self, event: Any) -> None:
        if (
            self._reservoir_display_matrix is None
            or self._reservoir_order is None
            or self._reservoir_class_labels is None
        ):
            return
        pos = event[0] if isinstance(event, (tuple, list)) else event
        if not self.reservoir_class_plot.sceneBoundingRect().contains(pos):
            return
        point = self.reservoir_class_plot.plotItem.vb.mapSceneToView(pos)
        column = int(np.floor(point.x()))
        row = int(np.floor(point.y()))
        matrix = self._reservoir_display_matrix
        if not (0 <= row < matrix.shape[0] and 0 <= column < matrix.shape[1]):
            return
        neuron = int(self._reservoir_order[column])
        label = int(self._reservoir_class_labels[row])
        response = float(matrix[row, column])
        self._set_hover_info(
            self.hover_label,
            self._tr(
                "gui.training.reservoir_class_hover",
                display=column,
                neuron=neuron,
                label=label,
                response=f"{response:.6g}",
                samples=self._reservoir_class_counts.get(label, 0),
            ),
        )

    def _handle_reservoir_current_hover(self, event: Any) -> None:
        if self._reservoir_current_feature is None:
            return
        pos = event[0] if isinstance(event, (tuple, list)) else event
        if not self.reservoir_current_plot.sceneBoundingRect().contains(pos):
            return
        point = self.reservoir_current_plot.plotItem.vb.mapSceneToView(pos)
        display_index = int(round(point.x()))
        order = self._reservoir_order
        if order is not None and 0 <= display_index < order.size:
            neuron = int(order[display_index])
        else:
            neuron = display_index
        if not (0 <= neuron < self._reservoir_current_feature.size):
            return
        value = float(self._reservoir_current_feature[neuron])
        self._set_hover_info(
            self.hover_label,
            self._tr(
                "gui.training.reservoir_current_hover",
                position=self._reservoir_current_position
                if self._reservoir_current_position is not None
                else "-",
                label=self._reservoir_current_label
                if self._reservoir_current_label is not None
                else "-",
                display=display_index,
                neuron=neuron,
                value=f"{value:.6g}",
            ),
        )

    # ------------------------------------------------------------------
    # Text / Текст
    # ------------------------------------------------------------------
    def _refresh_texts(self) -> None:
        if self._architecture == "dci":
            if self._help_mode:
                self.info_label.setText(self._tr("gui.training.dci_info"))
                self.info_label.setVisible(True)
            else:
                self.info_label.clear()
                self.info_label.setVisible(False)
            self.fixed_scale_checkbox.setText(self._tr("gui.training.fixed_scale"))
            self.rf_plot.setTitle(self._tr("gui.training.receptive_fields"))
            if self._last_weights is None:
                self._set_dci_footer("gui.training.waiting")
        else:
            if self._help_mode:
                self.info_label.setText(self._tr("gui.training.reservoir_info"))
                self.info_label.setVisible(True)
            else:
                self.info_label.clear()
                self.info_label.setVisible(False)
            self.reservoir_class_plot.setTitle(self._tr("gui.training.reservoir_class_activity"))
            self.reservoir_current_plot.setTitle(self._tr("gui.training.reservoir_current_feature"))
            self.reservoir_class_plot.setLabel(
                "bottom", self._tr("gui.training.reservoir_display_order")
            )
            self.reservoir_class_plot.setLabel("left", self._tr("gui.training.class_label"))
            self.reservoir_current_plot.setLabel(
                "bottom", self._tr("gui.training.reservoir_display_order")
            )
            self.reservoir_current_plot.setLabel("left", self._tr("gui.training.feature_value"))
            self._set_reservoir_footer(waiting=not bool(self._reservoir_class_sums))

    def _set_dci_footer(self, key: str, **kwargs: Any) -> None:
        # EN: Detailed DCI footer is explanatory material and stays tied to
        #     study mode. Reservoir status is operational and always visible.
        # RU: Подробный DCI-footer является учебным пояснением и показывается
        #     только в режиме изучения. Reservoir-status рабочий и виден всегда.
        if self._help_mode:
            self.footer_label.setText(self._tr(key, **kwargs))
            self.footer_label.setVisible(True)
        else:
            self.footer_label.clear()
            self.footer_label.setVisible(False)

    def _tr(self, key: str, **kwargs: Any) -> str:
        if self._translator is None:
            return key.format(**kwargs) if kwargs else key
        return self._translator.tr(key, **kwargs)
