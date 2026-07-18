from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReservoirConfig:
    """
    Configures the fixed recurrent Reservoir / LSM-like architecture.

    The reservoir weights remain fixed. Spike-count features are extracted
    from reservoir activity and an external readout is fitted afterwards.

    Настраивает фиксированную рекуррентную Reservoir / LSM-like архитектуру.

    Веса резервуара остаются фиксированными. Из спайковой активности
    извлекаются признаки, после чего обучается внешний readout.
    """

    seed: int = 52
    neuron_model_id: str = "izhikevich"
    n_input: int = 784
    n_reservoir: int = 800

    dt_ms: float = 0.5
    simulation_ms: float = 100.0
    max_rate_hz: float = 100.0

    input_density: float = 0.20
    recurrent_density: float = 0.08
    excitatory_ratio: float = 0.80

    # EN: Safe defaults match the default 784-input MNIST configuration.
    #     Iris presets intentionally use the stronger 10/4 scales.
    # RU: Безопасные defaults соответствуют стандартному MNIST-входу из 784
    #     нейронов. Iris-preset намеренно использует более сильные 10/4.
    input_scale: float = 2.0
    recurrent_scale: float = 1.0
    bias_current: float = 2.0
    tau_syn_ms: float = 8.0

    neuron_a: float = 0.02
    neuron_b: float = 0.20
    neuron_c: float = -65.0
    neuron_d: float = 8.0
    v_peak: float = 30.0

    numerical_method: str = "semi_euler"

    readout: str = "ridge"
    use_feature_selection: bool = True
    select_k: int = 400

    @property
    def n_steps(self) -> int:
        """
        Returns the number of integration steps.

        Возвращает число шагов интегрирования.
        """
        return max(1, int(round(self.simulation_ms / self.dt_ms)))

    def __post_init__(self) -> None:
        if self.neuron_model_id != "izhikevich":
            raise ValueError(
                "Stage 1 reservoir currently implements only neuron_model_id='izhikevich'"
            )
        if self.n_input <= 0 or self.n_reservoir <= 0:
            raise ValueError("n_input and n_reservoir must be positive")
        if self.dt_ms <= 0 or self.simulation_ms <= 0:
            raise ValueError("dt_ms and simulation_ms must be positive")
        if not 0.0 < self.excitatory_ratio <= 1.0:
            raise ValueError("excitatory_ratio must be in (0, 1]")
        if not 0.0 <= self.input_density <= 1.0:
            raise ValueError("input_density must be in [0, 1]")
        if not 0.0 <= self.recurrent_density <= 1.0:
            raise ValueError("recurrent_density must be in [0, 1]")
