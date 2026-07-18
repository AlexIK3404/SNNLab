from __future__ import annotations

from typing import Protocol

import numpy as np


class SpikeEncoder(Protocol):
    """
    Protocol for encoders that generate spike trains from one sample.

    Протокол энкодеров, генерирующих spike train из одного sample.
    """

    @property
    def output_size(self) -> int: ...

    def sample_spikes(
        self,
        x: np.ndarray,
        *,
        n_steps: int,
        dt_ms: float,
        rng: np.random.Generator,
    ) -> np.ndarray: ...


class PoissonEncoder:
    """
    Encodes normalized values as independent Poisson spike trains.

    Кодирует нормированные значения как независимые Poisson spike train.
    """

    def __init__(self, n_features: int, max_rate_hz: float = 100.0):
        self.n_features = int(n_features)
        self.max_rate_hz = float(max_rate_hz)

    @property
    def output_size(self) -> int:
        return self.n_features

    def sample_spikes(
        self,
        x: np.ndarray,
        *,
        n_steps: int,
        dt_ms: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64).reshape(-1)
        if values.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {values.size}")

        rates_hz = np.clip(values, 0.0, 1.0) * self.max_rate_hz
        probabilities = rates_hz * dt_ms / 1000.0
        if np.any(probabilities > 1.0):
            raise ValueError("Spike probability exceeded 1. Reduce dt_ms or max_rate_hz")

        return rng.random((n_steps, self.n_features)) < probabilities[None, :]


class GaussianPopulationEncoder:
    """
    Encodes normalized tabular features with Gaussian population coding.

    Кодирует нормированные табличные признаки гауссовым популяционным кодированием.
    """

    def __init__(
        self,
        *,
        n_features: int,
        n_per_feature: int = 8,
        max_rate_hz: float = 100.0,
        sigma_scale: float = 1.5,
    ):
        self.n_features = int(n_features)
        self.n_per_feature = int(n_per_feature)
        self.max_rate_hz = float(max_rate_hz)
        self.centers = np.linspace(0.0, 1.0, self.n_per_feature)
        self.sigma = sigma_scale / max(self.n_per_feature - 1, 1)

    @property
    def output_size(self) -> int:
        return self.n_features * self.n_per_feature

    def encode_rates(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64).reshape(-1)
        if values.size != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {values.size}")

        rates = np.zeros(self.output_size, dtype=np.float64)
        for feature_index, value in enumerate(values):
            start = feature_index * self.n_per_feature
            diff = value - self.centers
            rates[start : start + self.n_per_feature] = self.max_rate_hz * np.exp(
                -0.5 * (diff / self.sigma) ** 2
            )
        return rates

    def sample_spikes(
        self,
        x: np.ndarray,
        *,
        n_steps: int,
        dt_ms: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        rates_hz = self.encode_rates(x)
        probabilities = rates_hz * dt_ms / 1000.0
        if np.any(probabilities > 1.0):
            raise ValueError("Spike probability exceeded 1")
        return rng.random((n_steps, self.output_size)) < probabilities[None, :]
