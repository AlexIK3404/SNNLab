from __future__ import annotations

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background: #17191f;
    color: #e8eaf0;
    font-size: 10pt;
}
QToolBar {
    background: #20232b;
    border: 0;
    spacing: 6px;
    padding: 6px;
}
QDockWidget::title {
    background: #252933;
    padding: 7px;
    font-weight: 600;
}
QGroupBox {
    border: 1px solid #343947;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton, QToolButton {
    background: #2b303b;
    border: 1px solid #3d4452;
    border-radius: 5px;
    padding: 6px 10px;
}
QPushButton:hover, QToolButton:hover {
    background: #343b49;
}
QPushButton:pressed, QToolButton:pressed {
    background: #222731;
}
QPushButton:disabled, QToolButton:disabled {
    color: #737987;
    background: #23262d;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #222630;
    border: 1px solid #3b4250;
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 22px;
}
QComboBox QAbstractItemView {
    background: #222630;
    selection-background-color: #3f5f8f;
}
QTabWidget::pane {
    border: 1px solid #303542;
}
QTabBar::tab {
    background: #222630;
    padding: 8px 14px;
    border: 1px solid #303542;
}
QTabBar::tab:selected {
    background: #303746;
}
QScrollArea {
    border: 0;
}
QTableWidget {
    background: #1d2027;
    gridline-color: #303542;
}
QHeaderView::section {
    background: #282d37;
    color: #eef0f5;
    padding: 5px;
    border: 0;
}
QTextEdit, QPlainTextEdit {
    background: #111318;
    color: #d8dbe3;
    border: 1px solid #303542;
}
QProgressBar {
    border: 1px solid #3b4250;
    border-radius: 4px;
    text-align: center;
    background: #222630;
}
QProgressBar::chunk {
    background: #3f78b8;
}
QStatusBar {
    background: #20232b;
}
"""
