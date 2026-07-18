"""Core components and stable registries / Ядро компонентов и стабильные реестры."""

from snnlab.core.catalog import register_builtin_components
from snnlab.core.registry import (
    ComponentDescriptor,
    ComponentRegistry,
    architectures,
    backends,
    neuron_models,
    numerical_methods,
    readout_models,
    tasks,
)

__all__ = [
    "ComponentDescriptor",
    "ComponentRegistry",
    "architectures",
    "backends",
    "neuron_models",
    "numerical_methods",
    "readout_models",
    "register_builtin_components",
    "tasks",
]
