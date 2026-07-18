# Architectures

Architecture implementations live here. They own network topology, mutable neuron/synapse state, simulation steps, and architecture-specific learning rules.

Здесь находятся реализации архитектур. Они отвечают за топологию сети, изменяемое состояние нейронов/синапсов, шаг симуляции и архитектурно-зависимые правила обучения.

## Current modules

- `dci.py` — Diehl–Cook-like competitive SNN with Izhikevich E/I populations, Input→E STDP, lateral inhibition, and homeostasis.
- `reservoir.py` — LSM-like fixed recurrent spiking reservoir used as a feature dynamical system with an external readout.

Keep GUI, file-dialog, and experiment-history concerns outside this package.