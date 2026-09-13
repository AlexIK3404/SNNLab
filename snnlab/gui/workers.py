from __future__ import annotations

import traceback
from collections.abc import Callable
from threading import Event
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from snnlab.experiments.single_neuron import (
    SingleNeuronConfig,
    StudyCancelled,
    run_single_neuron_study,
)
from snnlab.runtime.single_neuron_io import save_single_neuron_study


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


class SingleNeuronStudyWorker(QObject):
    """Runs and persists the complete single-neuron ablation outside the GUI thread."""

    result_ready = Signal(object, str)
    progress = Signal(str, int, int)
    failed = Signal(str)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self,
        *,
        config: SingleNeuronConfig,
        method_ids: tuple[str, ...],
        dt_values_ms: tuple[float, ...],
        currents: tuple[float, ...],
        alpha_values: tuple[float, ...],
        runs_root: str = "runs/gui/single_neuron",
    ) -> None:
        super().__init__()
        self._config = config
        self._method_ids = method_ids
        self._dt_values_ms = dt_values_ms
        self._currents = currents
        self._alpha_values = alpha_values
        self._runs_root = runs_root
        self._cancel_event = Event()

    def request_cancel(self) -> None:
        """Set a thread-safe flag checked at simulation boundaries and long loops."""

        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = run_single_neuron_study(
                self._config,
                method_ids=self._method_ids,
                dt_values_ms=self._dt_values_ms,
                currents=self._currents,
                alpha_values=self._alpha_values,
                cancel_callback=self._cancel_event.is_set,
                progress_callback=self.progress.emit,
            )
            run_dir = save_single_neuron_study(result, root=self._runs_root)
            self.result_ready.emit(result, str(run_dir))
        except StudyCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.finished.emit()
