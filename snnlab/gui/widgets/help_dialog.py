from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout


class HelpDialog(QDialog):
    """
    Displays structured localized help for one parameter or concept.

    Показывает структурированную локализованную справку по параметру или термину.
    """

    def __init__(self, topic: dict[str, Any], *, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(str(topic.get("title", "Help")))
        self.resize(520, 260)

        layout = QVBoxLayout(self)
        title = QLabel(f"<b>{topic.get('title', '')}</b>")
        title.setWordWrap(True)
        short = QLabel(str(topic.get("short", "")))
        short.setWordWrap(True)
        long_text = QLabel(str(topic.get("long", "")))
        long_text.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(short)
        layout.addWidget(long_text)

        affects = topic.get("affects") or []
        if affects:
            affected = QLabel("<br>".join(f"• {item}" for item in affects))
            affected.setWordWrap(True)
            layout.addWidget(affected)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
