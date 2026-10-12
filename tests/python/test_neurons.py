import numpy as np

from openbrain.neurons import PRESETS, Izhikevich


def drive(pop: Izhikevich, current: float, ms: int) -> int:
    spikes = 0
    for _ in range(ms):
        spikes += int(pop.step(np.full(pop.n, current)).sum())
    return spikes


def test_resting_neuron_stays_silent_near_rest():
    pop = Izhikevich.from_preset("RS", n=1)
    assert drive(pop, 0.0, 1000) == 0
    assert -75 < pop.v[0] < -60


def test_regular_spiking_fires_tonically_under_constant_input():
    pop = Izhikevich.from_preset("RS", n=1)
    assert 5 <= drive(pop, 10.0, 1000) <= 50


def test_fast_spiking_outpaces_regular_spiking():
    rs = drive(Izhikevich.from_preset("RS", n=1), 10.0, 1000)
    fs = drive(Izhikevich.from_preset("FS", n=1), 10.0, 1000)
    assert fs > 2 * rs


def test_spike_resets_membrane_to_c():
    pop = Izhikevich.from_preset("RS", n=1)
    for _ in range(1000):
        if pop.step(np.array([10.0]))[0]:
            assert pop.v[0] == PRESETS["RS"]["c"]
            return
    raise AssertionError("neuron never spiked")
