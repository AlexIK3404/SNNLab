from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DCIConfig:
    """
    Configures the current Diehl-Cook-like architecture with Izhikevich neurons.

    The default topology is Input -> E, one-to-one E -> I, and
    all-to-all-except-paired I -> E. Stable numerical-method identifiers are
    stored per population so the framework can later expose them in the GUI.

    Настраивает текущую Diehl-Cook-like архитектуру с нейронами Ижикевича.

    Топология по умолчанию: Input -> E, one-to-one E -> I и
    all-to-all-except-paired I -> E. Для каждой популяции хранится стабильный
    идентификатор численного метода, чтобы позже вывести выбор в GUI.
    """

    seed: int = 52
    exc_neuron_model_id: str = "izhikevich"
    inh_neuron_model_id: str = "izhikevich"
    n_input: int = 28 * 28
    n_exc: int = 400
    n_inh: int = 400

    dt_ms: float = 0.1
    stimulus_ms: float = 350.0
    rest_ms: float = 150.0

    exc_a: float = 0.02
    exc_b: float = 0.20
    exc_c: float = -65.0
    exc_d: float = 8.0

    inh_a: float = 0.10
    inh_b: float = 0.20
    inh_c: float = -65.0
    inh_d: float = 2.0

    v_peak: float = 30.0
    e_exc: float = 0.0
    e_inh: float = -80.0

    tau_g_exc_ms: float = 5.0
    tau_g_inh_ms: float = 10.0
    refractory_exc_ms: float = 5.0
    refractory_inh_ms: float = 2.0

    input_weight_sum: float = 1.0

    exc_numerical_method: str = "explicit_euler"
    inh_numerical_method: str = "explicit_euler"

    @property
    def stimulus_steps(self) -> int:
        return max(1, int(round(self.stimulus_ms / self.dt_ms)))

    @property
    def rest_steps(self) -> int:
        return max(1, int(round(self.rest_ms / self.dt_ms)))

    def __post_init__(self) -> None:
        if self.exc_neuron_model_id != "izhikevich" or self.inh_neuron_model_id != "izhikevich":
            raise ValueError("Stage 1 DCI currently implements only Izhikevich E/I populations")
        if self.n_exc != self.n_inh:
            raise ValueError("Current one-to-one E -> I topology requires n_exc == n_inh")
        if self.n_input <= 0 or self.n_exc <= 0:
            raise ValueError("Population sizes must be positive")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")


@dataclass(frozen=True, slots=True)
class DCIDynamicsConfig:
    """
    Configures input drive and E/I coupling strengths.

    Настраивает силу входного сигнала и связи между E/I-популяциями.
    """

    input_gain: float = 0.60
    weight_exc_inh: float = 0.30
    weight_inh_exc: float = 20.0 / 399.0
    bias_exc: float = 0.0
    bias_inh: float = 0.0
    numerical_v_min: float = -200.0
    numerical_v_max: float = 200.0


@dataclass(frozen=True, slots=True)
class DCIPresentationConfig:
    """
    Configures repeated Poisson presentation when activity is too low.

    Настраивает повторное Poisson-предъявление при слишком низкой активности.
    """

    base_max_rate_hz: float = 63.75
    rate_increment_hz: float = 32.0
    min_exc_spikes: int = 5
    max_attempts: int = 5


@dataclass(frozen=True, slots=True)
class DCISTDPConfig:
    """
    Configures the current Input -> E STDP rule.

    Настраивает текущее правило STDP для Input -> E.
    """

    tau_pre_ms: float = 20.0
    eta: float = 0.00003
    x_target: float = 0.40
    mu: float = 0.20
    w_min: float = 0.0
    w_max: float = 0.01


@dataclass(frozen=True, slots=True)
class DCIHomeostasisConfig:
    """
    Configures sample-level target-rate homeostasis.

    The current is updated once after the final presentation attempt:
        current += learning_rate * (spikes_per_sample - target)

    Настраивает target-rate гомеостаз на уровне одного sample.

    Ток обновляется один раз после финальной попытки предъявления:
        current += learning_rate * (spikes_per_sample - target)
    """

    tau_ms: float = 1_000_000_000.0
    target_spikes_per_sample: float = 0.25
    learning_rate: float = 0.005
    min_current: float = 0.0
    max_current: float = 8.0


def scaled_inh_exc_weight(*, n_exc: int, target_total_inhibition: float) -> float:
    """
    Converts desired total inhibition into one I -> E connection weight.

    Преобразует желаемое суммарное торможение в вес одной связи I -> E.
    """
    if n_exc <= 1:
        raise ValueError("n_exc must be greater than 1")
    return target_total_inhibition / (n_exc - 1)


def make_dci_dynamics(
    cfg: DCIConfig,
    *,
    target_total_inhibition: float,
    weight_exc_inh: float = 0.30,
    input_gain: float = 0.60,
) -> DCIDynamicsConfig:
    """
    Builds DCI dynamics with inhibition scaled by E-population size.

    Создаёт параметры DCI с торможением, масштабированным по размеру E-популяции.
    """
    return DCIDynamicsConfig(
        input_gain=input_gain,
        weight_exc_inh=weight_exc_inh,
        weight_inh_exc=scaled_inh_exc_weight(
            n_exc=cfg.n_exc,
            target_total_inhibition=target_total_inhibition,
        ),
    )
