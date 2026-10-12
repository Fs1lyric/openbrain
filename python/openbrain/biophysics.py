"""Biophysical (conductance-based) neurons, simulated by the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openbrain import _core


@dataclass(frozen=True)
class HHResult:
    t: np.ndarray  # ms at the end of each step
    v: np.ndarray  # membrane voltage, mV
    spike_times: np.ndarray  # ms


class HHNeuron:
    """A single-compartment Hodgkin-Huxley neuron.

    Any parameter from `HHNeuron().params` can be overridden, e.g. `HHNeuron(g_na=0)` to model
    sodium channels blocked by tetrodotoxin. Units: mV, ms, uA/cm^2, mS/cm^2, uF/cm^2, degC.
    """

    def __init__(self, **overrides: float):
        defaults = _core.hh_defaults()
        unknown = sorted(set(overrides) - set(defaults))
        if unknown:
            raise TypeError(f"unknown HH parameters: {', '.join(unknown)}")
        self.params: dict[str, float] = {**defaults, **overrides}

    def simulate(self, current: np.ndarray, dt: float = 0.01) -> HHResult:
        """Inject `current[i]` (uA/cm^2) during step `i` of length `dt` ms, starting from rest."""
        current = np.asarray(current, dtype=float)
        v, spikes = _core.simulate_hh(self.params, current.tolist(), dt)
        t = np.arange(1, current.size + 1) * dt
        return HHResult(t=t, v=np.asarray(v), spike_times=np.asarray(spikes))
