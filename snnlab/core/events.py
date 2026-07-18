from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol

from snnlab.i18n import Translator


@dataclass(frozen=True, slots=True)
class FrameworkEvent:
    """
    Represents a framework event consumed by loggers, live plots, or GUI code.

    Представляет событие фреймворка для логгеров, live-графиков или GUI.
    """

    name: str
    source: str
    payload: dict[str, Any]
    timestamp: float

    @classmethod
    def create(
        cls, name: str, source: str, payload: dict[str, Any] | None = None
    ) -> FrameworkEvent:
        return cls(name=name, source=source, payload=payload or {}, timestamp=monotonic())


class EventObserver(Protocol):
    def handle(self, event: FrameworkEvent) -> None: ...


class NullObserver:
    """
    Discards all events.

    Игнорирует все события.
    """

    def handle(self, event: FrameworkEvent) -> None:
        return


class CompositeObserver:
    """
    Fans one event out to multiple observers.

    Передаёт одно событие нескольким observers.
    """

    def __init__(self, *observers: EventObserver):
        self._observers = tuple(observers)

    def handle(self, event: FrameworkEvent) -> None:
        for observer in self._observers:
            observer.handle(event)


class ConsoleObserver:
    """
    Prints localized progress messages.

    Печатает локализованные сообщения о прогрессе.
    """

    def __init__(self, translator: Translator | None = None, sample_every: int = 1):
        self.translator = translator or Translator("en")
        self.sample_every = max(1, int(sample_every))

    def handle(self, event: FrameworkEvent) -> None:
        p = event.payload
        if event.name == "run_start":
            print(self.translator.tr("run.start", architecture=p.get("architecture", event.source)))
        elif event.name == "run_end":
            print(self.translator.tr("run.end", position=p.get("position", 0)))
        elif event.name == "paused":
            print(self.translator.tr("run.paused", position=p.get("position", 0)))
        elif event.name == "resumed":
            print(self.translator.tr("run.resumed"))
        elif event.name == "stopped":
            print(self.translator.tr("run.stopped", position=p.get("position", 0)))
        elif event.name == "checkpoint_saved":
            print(self.translator.tr("run.checkpoint_saved", path=p.get("path", "")))
        elif event.name == "stage_start":
            print(self.translator.tr("stage.start", stage=p.get("stage", "")))
        elif event.name == "stage_end":
            print(self.translator.tr("stage.end", stage=p.get("stage", "")))
        elif event.name == "sample_end":
            position = int(p.get("position", 0))
            if position % self.sample_every != 0 and position != int(
                p.get("total_samples", position)
            ):
                return
            print(
                self.translator.tr(
                    "run.sample",
                    position=position,
                    total=p.get("total_samples", "?"),
                    label=p.get("label", "-"),
                    spikes=p.get("spikes", p.get("exc_spikes", 0)),
                    active=p.get("active", p.get("active_exc", 0)),
                )
            )
