from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from snnlab.core.events import FrameworkEvent


class QtEventObserver(QObject):
    """
    Bridges framework events from worker threads into the Qt event loop.

    Передаёт события фреймворка из worker-потоков в event loop Qt.
    """

    event_received = Signal(object)

    def handle(self, event: FrameworkEvent) -> None:
        # EN: Qt queued signals keep all widget updates on the GUI thread.
        # RU: Queued-сигналы Qt оставляют все обновления виджетов в GUI-потоке.
        self.event_received.emit(event)
