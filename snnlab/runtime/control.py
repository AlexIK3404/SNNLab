from __future__ import annotations

from threading import Condition


class RunControl:
    """
    Provides thread-safe pause, resume, and stop requests.

    Trainers call wait_if_paused() only at checkpoint-safe boundaries, normally
    after a complete sample. A GUI can therefore control a worker thread without
    interrupting membrane or synaptic updates mid-step.

    Предоставляет потокобезопасные запросы pause, resume и stop.

    Trainer вызывает wait_if_paused() только на безопасных границах, обычно
    после полного sample. Поэтому GUI может управлять worker-потоком, не обрывая
    мембранные или синаптические обновления посреди шага.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._paused = False
        self._stop_requested = False

    @property
    def pause_requested(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def stop_requested(self) -> bool:
        with self._condition:
            return self._stop_requested

    def request_pause(self) -> None:
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._paused = False
            self._condition.notify_all()

    def reset(self) -> None:
        with self._condition:
            self._paused = False
            self._stop_requested = False
            self._condition.notify_all()

    def wait_if_paused(self) -> bool:
        """
        Blocks at a safe point until resumed or stopped.

        Returns False when a stop request should terminate the run.

        Блокируется на безопасной точке до resume или stop.

        Возвращает False, если запрос stop должен завершить запуск.
        """
        with self._condition:
            while self._paused and not self._stop_requested:
                self._condition.wait()
            return not self._stop_requested
