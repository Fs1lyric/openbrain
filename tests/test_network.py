import numpy as np

from openbrain.cortex import build_cortex
from openbrain.network import Network
from openbrain.neurons import Izhikevich


def test_excitatory_synapse_drives_postsynaptic_neuron():
    weights = np.array([[0.0, 0.0], [30.0, 0.0]])  # W[post, pre]: 0 -> 1
    net = Network(Izhikevich.from_preset("RS", n=2), weights)
    record = net.run(500, external=np.array([10.0, 0.0]))
    assert record.count(0) > 0
    assert record.count(1) > 0


def test_no_synapse_means_no_postsynaptic_activity():
    net = Network(Izhikevich.from_preset("RS", n=2), np.zeros((2, 2)))
    record = net.run(500, external=np.array([10.0, 0.0]))
    assert record.count(1) == 0


def test_runs_are_reproducible_with_a_seed():
    a = build_cortex(n_exc=80, n_inh=20, seed=7).run(300)
    b = build_cortex(n_exc=80, n_inh=20, seed=7).run(300)
    assert np.array_equal(a.times, b.times)
    assert np.array_equal(a.ids, b.ids)


def population_fano(record) -> float:
    per_ms = np.bincount(record.times, minlength=record.duration_ms)
    return per_ms.var() / per_ms.mean()


def test_cortex_fires_at_biological_rates():
    rates = build_cortex(seed=1).run(1000).rates()
    assert 1 < rates[:800].mean() < 30
    assert 1 < rates[800:].mean() < 30


def test_recurrent_wiring_synchronises_the_population():
    wired = build_cortex(seed=1)
    unwired = build_cortex(seed=1)
    unwired.weights[:] = 0
    assert population_fano(wired.run(1000)) > 3  # bursty, rhythmic population activity
    assert population_fano(unwired.run(1000)) < 1.5  # independent, Poisson-like neurons


def test_stdp_changes_only_excitatory_weights():
    net = build_cortex(n_exc=80, n_inh=20, seed=3, plastic=True)
    before = net.weights.copy()
    net.run(500)
    assert not np.array_equal(net.weights[:, :80], before[:, :80])
    assert np.array_equal(net.weights[:, 80:], before[:, 80:])
    assert net.weights[:, :80].min() >= 0
