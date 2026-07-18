from __future__ import annotations

from dataclasses import dataclass

from snnlab.core.numerical_methods import available_methods, get_stepper
from snnlab.core.registry import (
    architectures,
    backends,
    neuron_models,
    numerical_methods,
    readout_models,
    tasks,
)


@dataclass(frozen=True, slots=True)
class BuiltinComponent:
    """
    Minimal marker used for components whose full implementation lives elsewhere.

    Минимальный marker для компонентов, полная реализация которых находится в другом модуле.
    """

    component_id: str


def register_builtin_components() -> None:
    """
    Registers Stage-1 components exactly once.

    Регистрирует компоненты Этапа 1 ровно один раз.
    """
    if "python_cpu" not in backends:
        backends.register(
            "python_cpu",
            BuiltinComponent("python_cpu"),
            display_name_key="components.backends.python_cpu",
        )

    if "izhikevich" not in neuron_models:
        neuron_models.register(
            "izhikevich",
            BuiltinComponent("izhikevich"),
            display_name_key="components.neuron_models.izhikevich",
        )

    for method_id in available_methods():
        if method_id not in numerical_methods:
            numerical_methods.register(
                method_id,
                get_stepper(method_id),
                display_name_key=f"components.numerical_methods.{method_id}",
                supported_neuron_models=("izhikevich",),
            )

    if "reservoir" not in architectures:
        architectures.register(
            "reservoir",
            BuiltinComponent("reservoir"),
            display_name_key="components.architectures.reservoir",
            supported_neuron_models=("izhikevich",),
        )
    if "dci" not in architectures:
        architectures.register(
            "dci",
            BuiltinComponent("dci"),
            display_name_key="components.architectures.dci",
            supported_neuron_models=("izhikevich",),
            experimental=True,
        )

    for readout_id in ("ridge", "logreg", "linear_svm", "rbf_svm"):
        if readout_id not in readout_models:
            readout_models.register(
                readout_id,
                BuiltinComponent(readout_id),
                display_name_key=f"components.readouts.{readout_id}",
            )

    if "classification" not in tasks:
        tasks.register(
            "classification",
            BuiltinComponent("classification"),
            display_name_key="components.tasks.classification",
        )


register_builtin_components()
