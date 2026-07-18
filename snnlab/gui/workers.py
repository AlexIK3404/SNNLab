from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot


class ExperimentWorker(QObject):
    """
    Executes session creation and long-running operations outside the GUI thread.

    Выполняет создание session и длительные операции вне GUI-потока.
    """

    session_ready = Signal(object)
    result_ready = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] | None = None,
        session: Any | None = None,
        operation: str = "run",
        operation_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._session_factory = session_factory
        self._session = session
        self._operation = operation
        self._operation_args = dict(operation_args or {})

    @Slot()
    def run(self) -> None:
        try:
            session = self._session
            if session is None:
                if self._session_factory is None:
                    raise RuntimeError("No session or session_factory was provided")
                session = self._session_factory()

            self.session_ready.emit(session)

            if self._operation == "load":
                result = session
            elif self._operation == "run":
                result = session.run()
            elif self._operation == "fit_readout":
                result = session.fit_readout()
            elif self._operation == "evaluate":
                result = session.evaluate(self._operation_args)
            elif self._operation == "save_model":
                result = session.save_model_snapshot()
            else:
                raise ValueError(f"Unknown worker operation: {self._operation!r}")

            self.result_ready.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()
