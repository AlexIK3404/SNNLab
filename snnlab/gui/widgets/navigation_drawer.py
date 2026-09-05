from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class NavigationDrawer(QWidget):
    """Collapsible workspace navigation for successive ablation levels."""

    workspace_selected = Signal(str)

    def __init__(self, translator, parent=None) -> None:
        super().__init__(parent)
        self._translator = translator
        self.heading = QLabel()
        self.heading.setStyleSheet("font-size: 13pt; font-weight: 700; padding: 6px 4px;")
        self.levels_label = QLabel()
        self.levels_label.setStyleSheet("color: #9aa4b5; padding: 4px;")
        self.workspace_list = QListWidget()
        self.workspace_list.setObjectName("workspace_navigation")
        self.workspace_list.setSpacing(3)

        for workspace_id in ("single_neuron", "network"):
            item = QListWidgetItem(workspace_id)
            item.setData(Qt.ItemDataRole.UserRole, workspace_id)
            self.workspace_list.addItem(item)

        self.roadmap_label = QLabel()
        self.roadmap_label.setWordWrap(True)
        self.roadmap_label.setStyleSheet(
            "QLabel { color: #9aa4b5; background: #1d2129; border-radius: 6px; padding: 8px; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.heading)
        layout.addWidget(self.levels_label)
        layout.addWidget(self.workspace_list, 1)
        layout.addWidget(self.roadmap_label)

        self.workspace_list.itemClicked.connect(self._item_clicked)
        self.retranslate(translator)
        self.set_current("network")

    def _item_clicked(self, item: QListWidgetItem) -> None:
        workspace_id = item.data(Qt.ItemDataRole.UserRole)
        if workspace_id:
            self.workspace_selected.emit(str(workspace_id))

    def set_current(self, workspace_id: str) -> None:
        for row in range(self.workspace_list.count()):
            item = self.workspace_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == workspace_id:
                self.workspace_list.setCurrentItem(item)
                return

    def retranslate(self, translator) -> None:
        self._translator = translator
        self.heading.setText(translator.tr("gui.navigation.title"))
        self.levels_label.setText(translator.tr("gui.navigation.levels"))
        for row in range(self.workspace_list.count()):
            item = self.workspace_list.item(row)
            workspace_id = str(item.data(Qt.ItemDataRole.UserRole))
            item.setText(translator.tr(f"gui.navigation.items.{workspace_id}"))
        self.roadmap_label.setText(translator.tr("gui.navigation.roadmap"))
