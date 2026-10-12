"""Pair-based spike-timing-dependent plasticity (STDP)."""

from __future__ import annotations

import numpy as np


class STDP:
    """Strengthens pre->post synapses when pre fires first, weakens them otherwise.

    Weights are indexed W[post, pre]. Only entries where `mask` is True ever change,
    and they stay within [0, w_max].
    """

    def __init__(
        self,
        n: int,
        mask: np.ndarray,
        w_max: float,
        a_plus: float = 0.1,
        a_minus: float = 0.12,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        self.mask = mask
        self.w_max = w_max
        self.a_plus, self.a_minus = a_plus, a_minus
        self.decay_plus = np.exp(-1.0 / tau_plus)
        self.decay_minus = np.exp(-1.0 / tau_minus)
        self.pre_trace = np.zeros(n)
        self.post_trace = np.zeros(n)

    def update(self, weights: np.ndarray, spikes: np.ndarray) -> None:
        """Apply one 1 ms step of STDP in place, given this step's spikes."""
        self.pre_trace *= self.decay_plus
        self.post_trace *= self.decay_minus
        delta = np.zeros_like(weights)
        delta[spikes, :] += self.a_plus * self.pre_trace  # post fired after earlier pre spikes
        delta[:, spikes] -= self.a_minus * self.post_trace[:, None]  # pre fired after post
        weights += delta * self.mask
        np.clip(weights, 0.0, self.w_max, out=weights, where=self.mask)
        self.pre_trace[spikes] += 1.0
        self.post_trace[spikes] += 1.0
