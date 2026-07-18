from __future__ import annotations

import sys


def main() -> int:
    """
    Starts the SNNLab desktop application.

    Запускает desktop-приложение SNNLab.
    """
    try:
        import pyqtgraph as pg
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "GUI dependencies are missing. Install them with: "
            "python -m pip install -e '.[gui,mnist]'"
        ) from exc

    from snnlab.gui.main_window import MainWindow
    from snnlab.gui.styles import DARK_STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("SNNLab")
    app.setStyleSheet(DARK_STYLESHEET)

    # EN: Global pyqtgraph options keep plots consistent with the application theme.
    # RU: Глобальные настройки pyqtgraph согласуют графики с темой приложения.
    pg.setConfigOption("background", "#17191f")
    pg.setConfigOption("foreground", "#d9dde7")
    pg.setConfigOptions(antialias=True)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
