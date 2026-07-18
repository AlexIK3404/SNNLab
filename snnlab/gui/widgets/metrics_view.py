from __future__ import annotations

from collections import deque
from typing import Any

from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class MetricsView(QWidget):
    """
    Shows recent sample-level metrics in a compact table.

    Показывает последние sample-level метрики в компактной таблице.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: deque[dict[str, Any]] = deque(maxlen=250)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "sample",
                "label",
                "accepted",
                "attempts",
                "spikes",
                "active",
                "sync",
                "rate Hz",
                "homeo max",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

    def retranslate(self, translator) -> None:
        """
        Refreshes table headers for the selected locale.

        Обновляет заголовки таблицы для выбранной локали.
        """
        self.table.setHorizontalHeaderLabels(
            [
                translator.tr("gui.metrics.sample"),
                translator.tr("gui.metrics.label"),
                translator.tr("gui.metrics.accepted"),
                translator.tr("gui.metrics.attempts"),
                translator.tr("gui.metrics.spikes"),
                translator.tr("gui.metrics.active"),
                translator.tr("gui.metrics.sync"),
                translator.tr("gui.metrics.rate_hz"),
                translator.tr("gui.metrics.homeo_max"),
            ]
        )

    def reset(self) -> None:
        self._rows.clear()
        self.table.setRowCount(0)

    def append_sample(self, payload: dict[str, Any]) -> None:
        self._rows.append(dict(payload))
        rows = list(self._rows)
        self.table.setRowCount(len(rows))
        for row_index, item in enumerate(rows):
            values = (
                item.get("position", ""),
                item.get("label", ""),
                item.get("accepted", ""),
                item.get("attempts", ""),
                item.get("exc_spikes", item.get("spikes", "")),
                item.get("active_exc", item.get("active", "")),
                item.get("max_sync_exc", ""),
                item.get("final_rate_hz", ""),
                item.get("homeo_max", ""),
            )
            for column, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.5g}"
                else:
                    text = str(value)
                self.table.setItem(row_index, column, QTableWidgetItem(text))
        self.table.scrollToBottom()
