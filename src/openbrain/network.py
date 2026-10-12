"""Recurrent networks of spiking neurons and their spike recordings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openbrain.neurons import Izhikevich
from openbrain.plasticity import STDP


@dataclass
class SpikeRecord:
    times: np.ndarray  # ms of each spike
    ids: np.ndarray  # neuron index of each spike
    n: int
    duration_ms: int

    def count(self, neuron: int) -> int:
        return int(np.count_nonzero(self.ids == neuron))

    def rates(self) -> np.ndarray:
        """Mean firing rate of each neuron, in Hz."""
        return np.bincount(self.ids, minlength=self.n) * 1000.0 / self.duration_ms


class Network:
    """Neurons wired by a weight matrix W[post, pre], with optional noise and STDP."""

    def __init__(
        self,
        neurons: Izhikevich,
        weights: np.ndarray,
        noise_std: np.ndarray | None = None,
        plasticity: STDP | None = None,
        seed: int | None = None,
    ):
        if weights.shape != (neurons.n, neurons.n):
            raise ValueError(f"weights must be {neurons.n}x{neurons.n}, got {weights.shape}")
        self.neurons = neurons
        self.weights = weights
        self.noise_std = np.zeros(neurons.n) if noise_std is None else noise_std
        self.plasticity = plasticity
        self.rng = np.random.default_rng(seed)

    def run(self, ms: int, external: np.ndarray | None = None) -> SpikeRecord:
        n = self.neurons.n
        drive = np.zeros(n) if external is None else external
        fired = np.zeros(n, dtype=bool)
        times, ids = [], []
        for t in range(ms):
            current = drive + self.noise_std * self.rng.standard_normal(n)
            current += self.weights[:, fired].sum(axis=1)
            fired = self.neurons.step(current)
            if self.plasticity is not None:
                self.plasticity.update(self.weights, fired)
            spiking = np.flatnonzero(fired)
            times.append(np.full(spiking.size, t))
            ids.append(spiking)
        return SpikeRecord(np.concatenate(times), np.concatenate(ids), n, ms)
