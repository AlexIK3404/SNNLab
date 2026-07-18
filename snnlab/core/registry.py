from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """
    Describes a framework component using stable, locale-independent identifiers.

    GUI code should render localized names through display_name_key instead of
    persisting translated text in experiment configurations.

    Описывает компонент фреймворка через стабильные идентификаторы,
    не зависящие от локали.

    GUI должен получать локализованное имя через display_name_key, а не сохранять
    переведённый текст в конфигурациях эксперимента.
    """

    component_id: str
    category: str
    display_name_key: str
    supported_backends: tuple[str, ...] = ("python_cpu",)
    supported_neuron_models: tuple[str, ...] = ()
    experimental: bool = False


class ComponentRegistry(Generic[T]):
    """
    Stores components by stable identifier and exposes metadata for future GUI lists.

    Хранит компоненты по стабильному идентификатору и отдаёт metadata для будущих списков GUI.
    """

    def __init__(self, category: str) -> None:
        self.category = category
        self._items: dict[str, T] = {}
        self._descriptors: dict[str, ComponentDescriptor] = {}

    def register(
        self,
        component_id: str,
        item: T,
        *,
        display_name_key: str,
        supported_backends: Iterable[str] = ("python_cpu",),
        supported_neuron_models: Iterable[str] = (),
        experimental: bool = False,
    ) -> None:
        """
        Registers one component and rejects accidental identifier replacement.

        Регистрирует один компонент и запрещает случайную перезапись идентификатора.
        """
        if component_id in self._items:
            raise KeyError(f"Component {component_id!r} is already registered in {self.category!r}")
        self._items[component_id] = item
        self._descriptors[component_id] = ComponentDescriptor(
            component_id=component_id,
            category=self.category,
            display_name_key=display_name_key,
            supported_backends=tuple(supported_backends),
            supported_neuron_models=tuple(supported_neuron_models),
            experimental=experimental,
        )

    def get(self, component_id: str) -> T:
        """
        Returns a registered component by stable identifier.

        Возвращает зарегистрированный компонент по стабильному идентификатору.
        """
        try:
            return self._items[component_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._items))
            raise KeyError(
                f"Unknown {self.category} component {component_id!r}. Available: {available}"
            ) from exc

    def descriptor(self, component_id: str) -> ComponentDescriptor:
        return self._descriptors[component_id]

    def descriptors(self) -> tuple[ComponentDescriptor, ...]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, component_id: object) -> bool:
        return component_id in self._items


# EN: Separate registries keep stable configuration IDs independent from Python class names.
# RU: Раздельные реестры отделяют стабильные ID конфигураций от имён Python-классов.
neuron_models: ComponentRegistry[Any] = ComponentRegistry("neuron_model")
numerical_methods: ComponentRegistry[Any] = ComponentRegistry("numerical_method")
architectures: ComponentRegistry[Any] = ComponentRegistry("architecture")
readout_models: ComponentRegistry[Any] = ComponentRegistry("readout")
tasks: ComponentRegistry[Any] = ComponentRegistry("task")
backends: ComponentRegistry[Any] = ComponentRegistry("backend")
