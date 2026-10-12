"""The 1000-neuron cortical network from Izhikevich (2003), "Simple Model of Spiking Neurons"."""

from __future__ import annotations

import numpy as np

from openbrain.network import Network
from openbrain.neurons import Izhikevich
from openbrain.plasticity import STDP

MAX_EXCITATORY_WEIGHT = 10.0


def build_cortex(
    n_exc: int = 800, n_inh: int = 200, seed: int | None = None, plastic: bool = False
) -> Network:
    """Randomly wired excitatory (RS..CH) and inhibitory (FS..LTS) cells under thalamic noise."""
    rng = np.random.default_rng(seed)
    re, ri = rng.random(n_exc), rng.random(n_inh)
    neurons = Izhikevich(
        a=np.concatenate([np.full(n_exc, 0.02), 0.02 + 0.08 * ri]),
        b=np.concatenate([np.full(n_exc, 0.2), 0.25 - 0.05 * ri]),
        c=np.concatenate([-65 + 15 * re**2, np.full(n_inh, -65.0)]),
        d=np.concatenate([8 - 6 * re**2, np.full(n_inh, 2.0)]),
    )
    n = n_exc + n_inh
    weights = np.hstack([0.5 * rng.random((n, n_exc)), -rng.random((n, n_inh))])
    noise = np.concatenate([np.full(n_exc, 5.0), np.full(n_inh, 2.0)])
    plasticity = None
    if plastic:
        mask = np.zeros((n, n), dtype=bool)
        mask[:, :n_exc] = True
        np.fill_diagonal(mask, False)
        plasticity = STDP(n, mask, w_max=MAX_EXCITATORY_WEIGHT, a_plus=0.01, a_minus=0.012)
    return Network(neurons, weights, noise_std=noise, plasticity=plasticity, seed=seed)
