from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class NavigationDrawer(QWidget):
    """Collapsible navigation between SNNLab experiment sections."""

    workspace_selected = Signal(str)

    def __init__(self, translator, parent=None) -> None:
        super().__init__(parent)
        self._translator = translator
        self.workspace_list = QListWidget()
        self.workspace_list.setObjectName("workspace_navigation")
        self.workspace_list.setSpacing(3)

        for workspace_id in ("single_neuron", "network"):
            item = QListWidgetItem(workspace_id)
            item.setData(Qt.ItemDataRole.UserRole, workspace_id)
            self.workspace_list.addItem(item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.workspace_list, 1)

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
        for row in range(self.workspace_list.count()):
            item = self.workspace_list.item(row)
            workspace_id = str(item.data(Qt.ItemDataRole.UserRole))
            item.setText(translator.tr(f"gui.navigation.items.{workspace_id}"))
