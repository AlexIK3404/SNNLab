from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget


class LiveExperimentView(QWidget):
    """
    Renders architecture-aware live SNN activity.

    DCI and Reservoir expose different dynamic objects. DCI keeps two coupled
    E/I populations and a continuous presentation state. Reservoir samples are
    simulated from a fresh state and are summarized by one recurrent population.
    The GUI therefore uses separate plotting semantics instead of forcing both
    architectures into the same labels and scales.

    Отображает архитектурно-зависимую live-активность SNN.

    DCI и Reservoir имеют разные динамические объекты. DCI содержит две связанные
    E/I-популяции и непрерывное состояние предъявления. Каждый reservoir-sample
    моделируется из свежего состояния и описывается одной рекуррентной популяцией.
    Поэтому GUI использует разные смыслы графиков, а не натягивает обе архитектуры
    на одинаковые подписи и шкалы.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._translator = None
        self._help_mode = False
        self._architecture = "dci"
        self._frame_index = 0
        self._max_raster_points = 25_000

        self._reservoir_neuron_signs: np.ndarray | None = None
        self._reservoir_dt_ms = 1.0
        self._reservoir_n_steps = 1
        self._reservoir_n_neurons = 0
        self._reservoir_raster_sample: int | None = None
        self._reservoir_total_raster_events = 0
        self._reservoir_displayed_raster_events = 0

        self._raster_x: deque[float] = deque()
        self._raster_y: deque[float] = deque()
        self._raster_brush: deque[Any] = deque()
        self._raster_population: deque[str] = deque()
        self._raster_local_neuron: deque[int] = deque()
        self._raster_sample: deque[Any] = deque()
        self._raster_phase: deque[str] = deque()
        self._raster_attempt: deque[Any] = deque()
        self._raster_step: deque[Any] = deque()

        self._sample_positions: deque[int] = deque(maxlen=300)
        self._spikes: deque[float] = deque(maxlen=300)
        self._active: deque[float] = deque(maxlen=300)
        self._sync: deque[float] = deque(maxlen=300)
        self._sample_payloads: deque[dict[str, Any]] = deque(maxlen=300)

        self._state_level_low: float | None = None
        self._state_level_high: float | None = None
        self._raster_separator: Any | None = None
        self._raster_e_label: Any | None = None
        self._raster_i_label: Any | None = None
        self._mouse_proxies: list[Any] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # EN: Splitters make the workspace responsive and user-resizable.
        # RU: Splitter-ы делают рабочую область адаптивной и позволяют вручную
        #     менять размеры графиков.
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.bottom_splitter)
        root.addWidget(self.main_splitter, 1)

        self.input_panel = self._panel()
        self.input_info = QLabel()
        self.input_info.setWordWrap(True)
        self.input_info.setVisible(False)
        self.input_plot = pg.PlotWidget(title="Input")
        self.input_plot.setAspectLocked(False)
        self.input_image = pg.ImageItem(axisOrder="row-major")
        self.input_plot.addItem(self.input_image)
        self.input_curve = self.input_plot.plot()
        self.input_curve.hide()
        self.input_panel.layout().addWidget(self.input_plot, 1)
        self.input_panel.layout().addWidget(self.input_info)
        self.top_splitter.addWidget(self.input_panel)

        self.raster_panel = self._panel()
        self.raster_info = QLabel()
        self.raster_info.setWordWrap(True)
        self.raster_info.setVisible(False)
        self.raster_plot = pg.PlotWidget(title="Live raster")
        self.raster_plot.showGrid(x=True, y=True, alpha=0.15)
        self.raster_scatter = pg.ScatterPlotItem(size=4, pen=None)
        self.raster_plot.addItem(self.raster_scatter)
        self.raster_plot.setLabel("bottom", "frame")
        self.raster_plot.setLabel("left", "neuron")
        self.raster_legend = self.raster_plot.addLegend(offset=(8, 8))
        self._raster_e_legend = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(80, 170, 255), pen=None)
        self._raster_i_legend = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush(255, 120, 100), pen=None
        )
        self.raster_legend.addItem(self._raster_e_legend, "Excitatory")
        self.raster_legend.addItem(self._raster_i_legend, "Inhibitory")
        self.raster_panel.layout().addWidget(self.raster_plot, 1)
        self.raster_panel.layout().addWidget(self.raster_info)
        self.top_splitter.addWidget(self.raster_panel)

        self.state_panel = self._panel()
        self.state_info = QLabel()
        self.state_info.setWordWrap(True)
        self.state_info.setVisible(False)
        self.state_plot = pg.PlotWidget(title="Population state")
        self.state_plot.setAspectLocked(True)
        self.state_plot.getViewBox().invertY(True)
        self.state_plot.hideAxis("bottom")
        self.state_plot.hideAxis("left")
        self.state_image = pg.ImageItem(axisOrder="row-major")
        self.state_plot.addItem(self.state_image)
        self.state_panel.layout().addWidget(self.state_plot, 1)
        self.state_panel.layout().addWidget(self.state_info)
        self.bottom_splitter.addWidget(self.state_panel)

        self.activity_panel = self._panel()
        self.activity_info = QLabel()
        self.activity_info.setWordWrap(True)
        self.activity_info.setVisible(False)
        self.activity_plot = pg.PlotWidget(title="Sample activity")
        self.activity_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spike_curve = self.activity_plot.plot(
            name="spikes", pen=pg.mkPen(80, 170, 255, width=2)
        )
        self.active_curve = self.activity_plot.plot(
            name="active", pen=pg.mkPen(110, 220, 130, width=2)
        )
        self.sync_curve = self.activity_plot.plot(
            name="max sync", pen=pg.mkPen(255, 190, 80, width=1)
        )
        self.activity_legend = self.activity_plot.addLegend(offset=(8, 8))
        self.activity_plot.setLabel("bottom", "sample")
        self.activity_panel.layout().addWidget(self.activity_plot, 1)
        self.activity_panel.layout().addWidget(self.activity_info)
        self.bottom_splitter.addWidget(self.activity_panel)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        for splitter in (self.main_splitter, self.top_splitter, self.bottom_splitter):
            splitter.setChildrenCollapsible(False)
            splitter.setHandleWidth(6)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 2)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 2)
        self.top_splitter.setSizes([350, 900])
        self.bottom_splitter.setSizes([450, 800])
        self.main_splitter.setSizes([520, 380])

        self._install_hover_handlers()
        self.set_architecture("dci")

    @staticmethod
    def _panel() -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        return panel

    def set_architecture(self, architecture: str) -> None:
        """Switches plot semantics for DCI or Reservoir.

        Переключает смысл графиков для DCI или Reservoir.
        """
        architecture = str(architecture)
        if architecture == self._architecture:
            self._refresh_plot_mode()
            return
        self._architecture = architecture
        self.reset()
        self._refresh_plot_mode()

    def update_from_session(self, session: Any) -> None:
        """Reads architecture metadata without mutating the runtime model.

        Читает метаданные архитектуры, не изменяя runtime-модель.
        """
        self.set_architecture(session.kind)
        if session.kind == "reservoir":
            network = session.engine.state.network
            cfg = session.engine.cfg
            self._reservoir_neuron_signs = np.asarray(network.neuron_signs, dtype=np.float32).copy()
            self._reservoir_dt_ms = float(cfg.dt_ms)
            self._reservoir_n_steps = int(cfg.n_steps)
            self._reservoir_n_neurons = int(cfg.n_reservoir)
        self._refresh_plot_mode()

    def refresh_layout(self) -> None:
        """Refreshes visible splitters and plot viewports after tab/window changes.

        Обновляет splitter-ы и viewport графиков после переключения вкладки или
        изменения состояния главного окна.
        """
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        for splitter in (self.main_splitter, self.top_splitter, self.bottom_splitter):
            splitter.refresh()
            splitter.updateGeometry()
        for plot in self.findChildren(pg.PlotWidget):
            plot.updateGeometry()
            plot.viewport().update()
            plot.update()

    def set_help_mode(self, enabled: bool) -> None:
        self._help_mode = bool(enabled)
        self._refresh_static_info()

    def _set_static_info(self, label: QLabel, key: str) -> None:
        if self._help_mode and self._translator is not None:
            label.setText(self._translator.tr(key))
            label.setVisible(True)
        else:
            label.clear()
            label.setVisible(False)

    @staticmethod
    def _set_hover_info(label: QLabel, text: str) -> None:
        label.setText(text)
        label.setVisible(True)

    def _refresh_static_info(self) -> None:
        self._set_static_info(self.input_info, "gui.live.input_tip")
        if self._architecture == "reservoir":
            self._set_static_info(self.raster_info, "gui.live.reservoir_raster_tip")
            self._set_static_info(self.state_info, "gui.live.reservoir_state_tip")
            self._set_static_info(self.activity_info, "gui.live.reservoir_activity_tip")
        else:
            self._set_static_info(self.raster_info, "gui.live.raster_tip")
            self._set_static_info(self.state_info, "gui.live.population_state_map_tip")
            self._set_static_info(self.activity_info, "gui.live.activity_tip")

    def retranslate(self, translator) -> None:
        self._translator = translator
        self.input_plot.setTitle(translator.tr("gui.live.input"))
        self.raster_plot.setLabel("left", translator.tr("gui.live.neuron"))
        self.activity_plot.setLabel("bottom", translator.tr("gui.live.sample"))
        self._refresh_plot_mode()
        self._refresh_static_info()

    def _refresh_plot_mode(self) -> None:
        if self._translator is None:
            return
        if self._architecture == "reservoir":
            self.raster_plot.setTitle(self._translator.tr("gui.live.reservoir_raster"))
            self.raster_plot.setLabel("bottom", self._translator.tr("gui.live.time_ms"))
            self.state_plot.setTitle(self._translator.tr("gui.live.reservoir_feature_state"))
            self.activity_plot.setTitle(self._translator.tr("gui.live.reservoir_sample_activity"))
            self.activity_plot.setLabel("left", "%")
            self.sync_curve.hide()
            self._refresh_activity_legend()
            self.activity_plot.setYRange(0.0, 100.0, padding=0.03)
        else:
            self.raster_plot.setTitle(self._translator.tr("gui.live.raster"))
            self.raster_plot.setLabel("bottom", self._translator.tr("gui.live.frame"))
            self.state_plot.setTitle(self._translator.tr("gui.live.population_state_map"))
            self.activity_plot.setTitle(self._translator.tr("gui.live.sample_activity"))
            self.activity_plot.setLabel("left", "")
            self.sync_curve.show()
            self._refresh_activity_legend()
            self.activity_plot.enableAutoRange(axis="y", enable=True)
        self._refresh_static_info()

    def _refresh_activity_legend(self) -> None:
        """Refreshes architecture-specific activity curve labels.

        Обновляет архитектурно-зависимые подписи кривых активности.
        """
        if self._translator is None:
            return
        self.activity_legend.clear()
        if self._architecture == "reservoir":
            self.activity_legend.addItem(
                self.spike_curve,
                self._translator.tr("gui.live.spike_occupancy"),
            )
            self.activity_legend.addItem(
                self.active_curve,
                self._translator.tr("gui.live.active_fraction"),
            )
        else:
            self.activity_legend.addItem(
                self.spike_curve,
                self._translator.tr("gui.live.exc_spikes"),
            )
            self.activity_legend.addItem(
                self.active_curve,
                self._translator.tr("gui.live.active_exc"),
            )
            self.activity_legend.addItem(
                self.sync_curve,
                self._translator.tr("gui.live.max_sync"),
            )

    def reset(self) -> None:
        self._frame_index = 0
        self._reservoir_raster_sample = None
        self._reservoir_total_raster_events = 0
        self._reservoir_displayed_raster_events = 0
        self._clear_raster_data()
        self._clear_raster_guides()
        self._sample_positions.clear()
        self._spikes.clear()
        self._active.clear()
        self._sync.clear()
        self._sample_payloads.clear()
        self._state_level_low = None
        self._state_level_high = None
        self.spike_curve.setData([])
        self.active_curve.setData([])
        self.sync_curve.setData([])
        self.state_image.clear()
        self.status_label.clear()
        self.status_label.setStyleSheet("")
        for label in (self.raster_info, self.activity_info, self.state_info):
            label.clear()
            label.setVisible(False)
        self._refresh_static_info()

    def _clear_raster_data(self) -> None:
        for sequence in (
            self._raster_x,
            self._raster_y,
            self._raster_brush,
            self._raster_population,
            self._raster_local_neuron,
            self._raster_sample,
            self._raster_phase,
            self._raster_attempt,
            self._raster_step,
        ):
            sequence.clear()
        self.raster_scatter.setData([])

    def _clear_raster_guides(self) -> None:
        for item_name in (
            "_raster_separator",
            "_raster_e_label",
            "_raster_i_label",
        ):
            item = getattr(self, item_name)
            if item is not None:
                try:
                    self.raster_plot.removeItem(item)
                except Exception:
                    pass
                setattr(self, item_name, None)

    def set_input_sample(self, sample: np.ndarray) -> None:
        values = np.asarray(sample)
        if values.ndim == 2:
            self.input_curve.hide()
            self.input_image.show()
            self.input_image.setImage(values.astype(np.float64), autoLevels=True)
            self.input_plot.getViewBox().invertY(True)
            self.input_plot.setAspectLocked(True)
        else:
            flat = values.reshape(-1).astype(np.float64)
            side = int(round(np.sqrt(flat.size)))
            if side * side == flat.size:
                self.input_curve.hide()
                self.input_image.show()
                self.input_image.setImage(flat.reshape(side, side), autoLevels=True)
                self.input_plot.getViewBox().invertY(True)
                self.input_plot.setAspectLocked(True)
            else:
                self.input_image.hide()
                self.input_curve.show()
                self.input_plot.getViewBox().invertY(False)
                self.input_curve.setData(np.arange(flat.size), flat)
                self.input_plot.setAspectLocked(False)

    def handle_simulation_frame(self, payload: dict[str, Any]) -> None:
        self._frame_index += 1
        phase = str(payload.get("phase", ""))
        attempt = payload.get("attempt", "")
        step = payload.get("step", "")
        sample_position = payload.get("position", "")

        if self._architecture == "reservoir":
            sample_int = int(sample_position) if sample_position not in ("", None) else None
            if sample_int is not None and sample_int != self._reservoir_raster_sample:
                self._clear_raster_data()
                self._reservoir_raster_sample = sample_int
            frame_x = float(payload.get("time_ms", payload.get("step", self._frame_index)))
        else:
            frame_x = float(self._frame_index)

        if "exc_spikes" in payload:
            exc_spikes_raw = np.asarray(payload["exc_spikes"], dtype=bool)
            inh_spikes_raw = np.asarray(payload["inh_spikes"], dtype=bool)
            exc = np.flatnonzero(exc_spikes_raw)
            inh = np.flatnonzero(inh_spikes_raw)
            n_exc = int(exc_spikes_raw.size)
            self._ensure_raster_population_guides(n_exc)
            self._append_raster(
                frame_x,
                exc,
                offset=0,
                brush=pg.mkBrush(80, 170, 255),
                population="E",
                sample=sample_position,
                phase=phase,
                attempt=attempt,
                step=step,
            )
            self._append_raster(
                frame_x,
                inh,
                offset=n_exc,
                brush=pg.mkBrush(255, 120, 100),
                population="I",
                sample=sample_position,
                phase=phase,
                attempt=attempt,
                step=step,
            )
            if "exc_v" in payload:
                self._set_population_state_map(np.asarray(payload["exc_v"], dtype=np.float64))
        elif "spikes" in payload:
            raw = np.asarray(payload["spikes"], dtype=bool).reshape(-1)
            indices = np.flatnonzero(raw)
            self._append_reservoir_raster_points(
                x_values=np.full(indices.size, frame_x, dtype=np.float64),
                neuron_indices=indices,
                sample=sample_position,
                steps=np.full(indices.size, int(step or 0), dtype=np.int64),
            )
            if "membrane_v" in payload:
                self._set_population_state_map(np.asarray(payload["membrane_v"], dtype=np.float64))

        self._trim_raster()
        self._update_raster_scatter()

        total_steps = payload.get("total_steps", "")
        epoch = payload.get("epoch")
        n_epochs = payload.get("n_epochs")
        epoch_text = f" | epoch={epoch}/{n_epochs}" if epoch is not None else ""
        self.status_label.setText(
            f"sample={payload.get('position', '-')}"
            f"{epoch_text} | phase={phase} | attempt={attempt} | step={step}/{total_steps}"
        )

    def handle_sample_end(self, payload: dict[str, Any]) -> None:
        position = int(payload.get("position", 0))
        self._sample_positions.append(position)
        # EN: Keep only scalar diagnostics for hover history. Full spike logs
        #     can be hundreds of kilobytes per sample and belong only to the
        #     current raster render.
        # RU: Для hover-history сохраняем только scalar-диагностику. Полные
        #     spike log могут занимать сотни килобайт на sample и нужны только
        #     для текущего raster.
        compact_payload = {
            key: value for key, value in payload.items() if key not in {"spike_log", "spike_counts"}
        }
        self._sample_payloads.append(compact_payload)

        if self._architecture == "reservoir":
            self._handle_reservoir_sample_end(payload)
        else:
            spikes = float(payload.get("exc_spikes", payload.get("spikes", 0)))
            active = float(payload.get("active_exc", payload.get("active", 0)))
            sync = float(payload.get("max_sync_exc", 0))
            self._spikes.append(spikes)
            self._active.append(active)
            self._sync.append(sync)
            self._update_activity_curves()

    def _handle_reservoir_sample_end(self, payload: dict[str, Any]) -> None:
        n_reservoir = max(
            1,
            int(payload.get("n_reservoir", self._reservoir_n_neurons or 1)),
        )
        n_steps = max(1, int(payload.get("n_steps", self._reservoir_n_steps or 1)))
        self._reservoir_n_neurons = n_reservoir
        self._reservoir_n_steps = n_steps
        simulation_ms = float(payload.get("simulation_ms", n_steps * self._reservoir_dt_ms))
        if n_steps > 0:
            self._reservoir_dt_ms = simulation_ms / n_steps

        occupancy = float(
            payload.get(
                "spike_occupancy",
                float(payload.get("spikes", 0)) / max(1, n_reservoir * n_steps),
            )
        )
        active_fraction = float(payload.get("active", 0)) / n_reservoir
        self._spikes.append(100.0 * occupancy)
        self._active.append(100.0 * active_fraction)
        self._sync.append(float(payload.get("mean_rate_hz", 0.0)))
        self._update_activity_curves()

        spike_counts = payload.get("spike_counts")
        if spike_counts is not None:
            self._set_population_state_map(np.asarray(spike_counts, dtype=np.float64))

        spike_log = payload.get("spike_log")
        if spike_log is not None:
            self._render_reservoir_spike_log(
                np.asarray(spike_log, dtype=bool),
                sample=int(payload.get("position", 0)),
            )

        if occupancy >= 0.25:
            warning = self._tr(
                "gui.live.reservoir_saturation_warning",
                occupancy=f"{100.0 * occupancy:.1f}",
                rate=f"{float(payload.get('mean_rate_hz', 0.0)):.1f}",
            )
            self.status_label.setText(warning)
            self.status_label.setStyleSheet("color: #ff8a65; font-weight: 600;")
        else:
            self.status_label.setStyleSheet("")
            self.status_label.setText(
                self._tr(
                    "gui.live.reservoir_sample_status",
                    position=payload.get("position", "-"),
                    spikes=payload.get("spikes", 0),
                    active=payload.get("active", 0),
                    total=n_reservoir,
                    occupancy=f"{100.0 * occupancy:.2f}",
                    rate=f"{float(payload.get('mean_rate_hz', 0.0)):.1f}",
                )
            )

    def _update_activity_curves(self) -> None:
        x = np.asarray(self._sample_positions, dtype=np.float64)
        self.spike_curve.setData(x, np.asarray(self._spikes, dtype=np.float64))
        self.active_curve.setData(x, np.asarray(self._active, dtype=np.float64))
        self.sync_curve.setData(x, np.asarray(self._sync, dtype=np.float64))

    def _render_reservoir_spike_log(self, spike_log: np.ndarray, *, sample: int) -> None:
        if spike_log.ndim != 2:
            return
        step_indices, neuron_indices = np.nonzero(spike_log)
        self._reservoir_total_raster_events = int(step_indices.size)
        if step_indices.size > self._max_raster_points:
            keep = np.linspace(
                0,
                step_indices.size - 1,
                num=self._max_raster_points,
                dtype=np.int64,
            )
            step_indices = step_indices[keep]
            neuron_indices = neuron_indices[keep]
        self._reservoir_displayed_raster_events = int(step_indices.size)

        self._clear_raster_data()
        self._reservoir_raster_sample = int(sample)
        x_values = (step_indices.astype(np.float64) + 1.0) * self._reservoir_dt_ms
        self._append_reservoir_raster_points(
            x_values=x_values,
            neuron_indices=neuron_indices,
            sample=sample,
            steps=step_indices + 1,
        )
        self._update_raster_scatter()
        if self._reservoir_n_neurons > 0:
            self.raster_plot.setYRange(-1.0, float(self._reservoir_n_neurons), padding=0.01)
        self.raster_plot.setXRange(
            0.0,
            max(self._reservoir_dt_ms, self._reservoir_n_steps * self._reservoir_dt_ms),
            padding=0.01,
        )

    def _append_reservoir_raster_points(
        self,
        *,
        x_values: np.ndarray,
        neuron_indices: np.ndarray,
        sample: Any,
        steps: np.ndarray,
    ) -> None:
        indices = np.asarray(neuron_indices, dtype=np.int64).reshape(-1)
        xs = np.asarray(x_values, dtype=np.float64).reshape(-1)
        steps = np.asarray(steps, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return
        if xs.size != indices.size or steps.size != indices.size:
            raise ValueError("Reservoir raster arrays must have equal length")

        signs = self._reservoir_neuron_signs
        for x_value, neuron, step_value in zip(xs, indices, steps, strict=True):
            inhibitory = bool(
                signs is not None and 0 <= int(neuron) < signs.size and signs[int(neuron)] < 0.0
            )
            self._raster_x.append(float(x_value))
            self._raster_y.append(float(neuron))
            self._raster_brush.append(
                pg.mkBrush(255, 120, 100) if inhibitory else pg.mkBrush(80, 170, 255)
            )
            self._raster_population.append("Reservoir-I" if inhibitory else "Reservoir-E")
            self._raster_local_neuron.append(int(neuron))
            self._raster_sample.append(sample)
            self._raster_phase.append("sample")
            self._raster_attempt.append("")
            self._raster_step.append(int(step_value))

    def _append_raster(
        self,
        frame_x: float,
        indices: np.ndarray,
        *,
        offset: int,
        brush: Any,
        population: str,
        sample: Any,
        phase: str,
        attempt: Any,
        step: Any,
    ) -> None:
        for index in indices:
            neuron = int(index)
            self._raster_x.append(frame_x)
            self._raster_y.append(float(neuron + offset))
            self._raster_brush.append(brush)
            self._raster_population.append(population)
            self._raster_local_neuron.append(neuron)
            self._raster_sample.append(sample)
            self._raster_phase.append(phase)
            self._raster_attempt.append(attempt)
            self._raster_step.append(step)

    def _trim_raster(self) -> None:
        sequences = (
            self._raster_x,
            self._raster_y,
            self._raster_brush,
            self._raster_population,
            self._raster_local_neuron,
            self._raster_sample,
            self._raster_phase,
            self._raster_attempt,
            self._raster_step,
        )
        while len(self._raster_x) > self._max_raster_points:
            for sequence in sequences:
                sequence.popleft()

    def _update_raster_scatter(self) -> None:
        self.raster_scatter.setData(
            x=np.fromiter(self._raster_x, dtype=np.float64),
            y=np.fromiter(self._raster_y, dtype=np.float64),
            brush=list(self._raster_brush),
        )

    def _ensure_raster_population_guides(self, n_exc: int) -> None:
        if self._raster_separator is not None:
            return
        self._raster_separator = pg.InfiniteLine(
            pos=float(n_exc) - 0.5,
            angle=0,
            pen=pg.mkPen(
                200,
                200,
                200,
                width=1,
                style=Qt.PenStyle.DashLine,
            ),
        )
        self.raster_plot.addItem(self._raster_separator)
        self._raster_e_label = pg.TextItem("E", color=(80, 170, 255), anchor=(0, 1))
        self._raster_i_label = pg.TextItem("I", color=(255, 120, 100), anchor=(0, 0))
        self._raster_e_label.setPos(0, max(0, n_exc - 2))
        self._raster_i_label.setPos(0, n_exc + 2)
        self.raster_plot.addItem(self._raster_e_label)
        self.raster_plot.addItem(self._raster_i_label)

    def _set_population_state_map(self, values: np.ndarray) -> None:
        """Packs a 1D population state into a near-square index map.

        Упаковывает одномерное состояние популяции в почти квадратную индексную
        карту. Соседство на экране — только раскладка индексов, а не топология сети.
        """
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return
        columns = int(np.ceil(np.sqrt(flat.size)))
        rows = int(np.ceil(flat.size / columns))
        padded = np.full(rows * columns, np.nan, dtype=np.float64)
        padded[: flat.size] = flat
        image = padded.reshape(rows, columns)
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return
        low = float(np.percentile(finite, 2.0))
        high = float(np.percentile(finite, 98.0))
        if high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
        if high <= low:
            high = low + 1e-9
        alpha = 0.20
        if self._state_level_low is None or self._state_level_high is None:
            self._state_level_low = low
            self._state_level_high = high
        else:
            self._state_level_low = (1.0 - alpha) * self._state_level_low + alpha * low
            self._state_level_high = (1.0 - alpha) * self._state_level_high + alpha * high
        self.state_image.setImage(
            image,
            autoLevels=False,
            levels=(self._state_level_low, self._state_level_high),
        )

    def _install_hover_handlers(self) -> None:
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.raster_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_raster_hover,
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.activity_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_activity_hover,
            )
        )
        self._mouse_proxies.append(
            pg.SignalProxy(
                self.state_plot.scene().sigMouseMoved,
                rateLimit=30,
                slot=self._handle_state_hover,
            )
        )

    def _handle_raster_hover(self, event: Any) -> None:
        pos = event[0] if isinstance(event, (tuple, list)) else event
        if not self.raster_plot.sceneBoundingRect().contains(pos) or not self._raster_x:
            return
        point = self.raster_plot.plotItem.vb.mapSceneToView(pos)
        x = np.fromiter(self._raster_x, dtype=np.float64)
        y = np.fromiter(self._raster_y, dtype=np.float64)
        index = self._nearest_index(self.raster_plot, x, y, point.x(), point.y())
        if index is None:
            return
        text = (
            f"population={list(self._raster_population)[index]} | "
            f"neuron={list(self._raster_local_neuron)[index]} | "
            f"x={x[index]:.3f} | sample={list(self._raster_sample)[index]} | "
            f"phase={list(self._raster_phase)[index]} | "
            f"attempt={list(self._raster_attempt)[index]} | "
            f"step={list(self._raster_step)[index]}"
        )
        if self._architecture == "reservoir" and self._reservoir_total_raster_events:
            text += (
                f" | displayed={self._reservoir_displayed_raster_events}/"
                f"{self._reservoir_total_raster_events} spikes"
            )
        self._set_hover_info(self.raster_info, text)

    def _handle_activity_hover(self, event: Any) -> None:
        pos = event[0] if isinstance(event, (tuple, list)) else event
        if not self.activity_plot.sceneBoundingRect().contains(pos) or not self._sample_positions:
            return
        point = self.activity_plot.plotItem.vb.mapSceneToView(pos)
        samples = np.asarray(self._sample_positions, dtype=np.float64)
        index = int(np.argmin(np.abs(samples - point.x())))
        payload = list(self._sample_payloads)[index]
        if self._architecture == "reservoir":
            self._set_hover_info(
                self.activity_info,
                f"sample={payload.get('position')} | label={payload.get('label', '-')} | "
                f"spikes={payload.get('spikes', '-')} | active={payload.get('active', '-')} | "
                f"occupancy={100.0 * float(payload.get('spike_occupancy', 0.0)):.3f}% | "
                f"mean rate={float(payload.get('mean_rate_hz', 0.0)):.3f} Hz | "
                f"E/I spikes={payload.get('excitatory_spikes', '-')}/"
                f"{payload.get('inhibitory_spikes', '-')}",
            )
        else:
            self._set_hover_info(
                self.activity_info,
                f"sample={payload.get('position')} | label={payload.get('label', '-')} | "
                f"accepted={payload.get('accepted', '-')} | attempts={payload.get('attempts', '-')} | "
                f"E spikes={payload.get('exc_spikes', payload.get('spikes', '-'))} | "
                f"active={payload.get('active_exc', payload.get('active', '-'))} | "
                f"max sync={payload.get('max_sync_exc', '-')}",
            )

    def _handle_state_hover(self, event: Any) -> None:
        pos = event[0] if isinstance(event, (tuple, list)) else event
        if not self.state_plot.sceneBoundingRect().contains(pos):
            return
        point = self.state_plot.plotItem.vb.mapSceneToView(pos)
        x = int(np.floor(point.x()))
        y = int(np.floor(point.y()))
        image = self.state_image.image
        if image is None:
            return
        arr = np.asarray(image)
        if 0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]:
            value = arr[y, x]
            if np.isfinite(value):
                neuron = y * arr.shape[1] + x
                value_name = "spike count" if self._architecture == "reservoir" else "v"
                self._set_hover_info(
                    self.state_info,
                    f"neuron index={neuron} | {value_name}={float(value):.6g}",
                )

    @staticmethod
    def _nearest_index(
        plot: pg.PlotWidget,
        x: np.ndarray,
        y: np.ndarray,
        px: float,
        py: float,
    ) -> int | None:
        if x.size == 0:
            return None
        (x_min, x_max), (y_min, y_max) = plot.plotItem.vb.viewRange()
        sx = max(abs(x_max - x_min), 1e-9)
        sy = max(abs(y_max - y_min), 1e-9)
        dist = ((x - px) / sx) ** 2 + ((y - py) / sy) ** 2
        idx = int(np.argmin(dist))
        if dist[idx] > 0.0006:
            return None
        return idx

    def _tr(self, key: str, **kwargs: Any) -> str:
        if self._translator is None:
            return key.format(**kwargs) if kwargs else key
        return self._translator.tr(key, **kwargs)
