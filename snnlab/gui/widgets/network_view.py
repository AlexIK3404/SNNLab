from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QVBoxLayout, QWidget


class NetworkView(QWidget):
    """
    Displays a lightweight architecture diagram for presentation and inspection.

    Показывает лёгкую схему архитектуры для демонстрации и анализа.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self._architecture = "dci"
        self._translator = None
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(self.view.renderHints())
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)
        self.set_architecture("dci")

    def set_architecture(self, architecture: str) -> None:
        self._architecture = architecture
        self.scene.clear()

        if architecture == "dci":
            self._draw_dci()
        else:
            self._draw_reservoir()

        # EN: Add a small margin around the diagram before fitting it into view.
        # RU: Добавляем небольшой отступ вокруг схемы перед масштабированием.
        bounds = self.scene.itemsBoundingRect().adjusted(-30, -30, 30, 30)
        self.scene.setSceneRect(bounds)
        self.view.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)

    def retranslate(self, translator) -> None:
        """
        Redraws architecture labels after a locale switch.

        Перерисовывает подписи архитектуры после переключения локали.
        """
        self._translator = translator
        self.set_architecture(self._architecture)

    def _tr(self, key: str, fallback: str) -> str:
        if self._translator is None:
            return fallback
        value = self._translator.tr(key)
        return fallback if value == key else value

    def _node(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        color: str,
    ):
        """
        Draws a rounded node with horizontally centered multiline text.

        Рисует скруглённый узел с центрированным многострочным текстом.
        """
        path = QPainterPath()
        path.addRoundedRect(x, y, w, h, 12, 12)

        rect = self.scene.addPath(
            path,
            QPen(QColor("#d0d5df"), 1.5),
            QBrush(QColor(color)),
        )

        label = self.scene.addText(text)
        label.setDefaultTextColor(QColor("#ffffff"))
        label.setTextWidth(max(20.0, w - 20.0))

        text_option = label.document().defaultTextOption()
        text_option.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.document().setDefaultTextOption(text_option)

        bounds = label.boundingRect()
        label.setPos(x + 10.0, y + (h - bounds.height()) / 2.0)

        return rect

    def _arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#8fb7e8",
    ) -> None:
        """
        Draws a straight directed connection.

        Рисует прямую направленную связь.
        """
        pen = QPen(QColor(color), 3)
        self.scene.addLine(x1, y1, x2, y2, pen)
        self._arrow_head(QPointF(x2, y2), QPointF(x1, y1), color)

    def _curved_arrow(
        self,
        path: QPainterPath,
        *,
        previous_point: QPointF,
        color: str,
    ) -> None:
        """
        Draws a directed curved connection using a painter path.

        Рисует направленную криволинейную связь через painter path.
        """
        self.scene.addPath(path, QPen(QColor(color), 3))
        self._arrow_head(path.currentPosition(), previous_point, color)

    def _arrow_head(
        self,
        tip: QPointF,
        previous_point: QPointF,
        color: str,
    ) -> None:
        """
        Adds an arrow head aligned with the final segment direction.

        Добавляет наконечник стрелки по направлению последнего сегмента.
        """
        dx = tip.x() - previous_point.x()
        dy = tip.y() - previous_point.y()

        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return

        angle = math.atan2(dy, dx)
        length = 12.0
        spread = math.radians(28.0)

        p1 = QPointF(
            tip.x() - length * math.cos(angle - spread),
            tip.y() - length * math.sin(angle - spread),
        )
        p2 = QPointF(
            tip.x() - length * math.cos(angle + spread),
            tip.y() - length * math.sin(angle + spread),
        )

        polygon = QPolygonF([tip, p1, p2])
        self.scene.addPolygon(
            polygon,
            QPen(QColor(color), 1.0),
            QBrush(QColor(color)),
        )

    def _draw_dci(self) -> None:
        self._node(
            20,
            120,
            150,
            70,
            self._tr("gui.network.input", "Input / Encoder"),
            "#34506f",
        )
        self._node(
            250,
            50,
            160,
            70,
            self._tr("gui.network.excitatory", "Excitatory E"),
            "#2f6f78",
        )
        self._node(
            250,
            220,
            160,
            70,
            self._tr("gui.network.inhibitory", "Inhibitory I"),
            "#7a3e48",
        )

        self._arrow(170, 155, 250, 85)
        self._arrow(330, 120, 330, 220)

        inhibition_path = QPainterPath(QPointF(410, 255))
        inhibition_path.cubicTo(500, 255, 500, 85, 410, 85)
        self._curved_arrow(
            inhibition_path,
            previous_point=QPointF(500, 85),
            color="#e08a7b",
        )

        rules = self.scene.addText(
            self._tr(
                "gui.network.dci_rules",
                "Input → E\nEᵢ → Iᵢ\nIᵢ → Eⱼ, j ≠ i",
            )
        )
        rules.setDefaultTextColor(QColor("#d0d5df"))
        rules.setPos(500, 120)

    def _draw_reservoir(self) -> None:
        # EN: Use a symmetric left-to-right layout and a real curved self-loop.
        #     The previous U-shaped polyline looked detached from the reservoir.
        # RU: Используем симметричную схему слева направо и настоящую кривую
        #     петлю. Предыдущая П-образная линия выглядела оторванной от узла.
        self._node(
            30,
            130,
            170,
            70,
            self._tr("gui.network.input", "Input / Encoder"),
            "#34506f",
        )
        self._node(
            290,
            95,
            260,
            140,
            self._tr(
                "gui.network.reservoir",
                "Spiking Reservoir\nFixed recurrent weights",
            ),
            "#426a54",
        )
        self._node(
            650,
            130,
            180,
            70,
            self._tr("gui.network.readout", "Readout"),
            "#6a4f7b",
        )

        self._arrow(200, 165, 290, 165)
        self._arrow(550, 165, 650, 165)

        loop = QPainterPath(QPointF(355, 95))
        loop.cubicTo(355, 20, 485, 20, 485, 95)
        self._curved_arrow(
            loop,
            previous_point=QPointF(485, 20),
            color="#78c69a",
        )

        loop_label = self.scene.addText(self._tr("gui.network.recurrent", "Recurrent"))
        loop_label.setDefaultTextColor(QColor("#9ee0bd"))
        loop_bounds = loop_label.boundingRect()
        loop_label.setPos(420 - loop_bounds.width() / 2.0, 25)
