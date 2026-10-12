"""Izhikevich (2003) spiking neurons, vectorised over a population."""

from __future__ import annotations

import numpy as np

SPIKE_PEAK = 30.0

# Parameters (a, b, c, d) for the cell classes in Izhikevich 2003, Fig. 2.
PRESETS: dict[str, dict[str, float]] = {
    "RS": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},  # regular spiking
    "IB": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},  # intrinsically bursting
    "CH": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},  # chattering
    "FS": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},  # fast spiking
    "LTS": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},  # low-threshold spiking
    "TC": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 0.05},  # thalamo-cortical
    "RZ": {"a": 0.1, "b": 0.26, "c": -65.0, "d": 2.0},  # resonator
}


class Izhikevich:
    """A population of Izhikevich neurons advanced in 1 ms steps."""

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
        self.a, self.b, self.c, self.d = (np.asarray(p, dtype=float) for p in (a, b, c, d))
        self.n = self.a.size
        self.v = np.full(self.n, -65.0)
        self.u = self.b * self.v

    @classmethod
    def from_preset(cls, name: str, n: int) -> Izhikevich:
        p = PRESETS[name]
        return cls(*(np.full(n, p[k]) for k in "abcd"))

    def step(self, current: np.ndarray) -> np.ndarray:
        """Advance 1 ms under input `current`; return a boolean mask of neurons that spiked."""
        for _ in range(2):  # two 0.5 ms Euler substeps for numerical stability
            self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + current)
        self.u += self.a * (self.b * self.v - self.u)
        fired = self.v >= SPIKE_PEAK
        self.v[fired] = self.c[fired]
        self.u[fired] += self.d[fired]
        return fired
