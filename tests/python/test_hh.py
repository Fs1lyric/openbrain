import numpy as np
import pytest

from openbrain import HHNeuron


def test_simulate_returns_aligned_time_voltage_and_spikes():
    result = HHNeuron().simulate(np.full(110_000, 10.0), dt=0.01)
    assert result.v.shape == result.t.shape == (110_000,)
    assert result.t[1] - result.t[0] == pytest.approx(0.01)
    sustained = result.spike_times[result.spike_times >= 300]
    assert 60 < sustained.size / 0.8 < 80  # textbook ~68 Hz at 10 uA/cm^2


def test_parameters_can_be_overridden_to_model_a_drug():
    ttx = HHNeuron(g_na=0.0)  # tetrodotoxin blocks sodium channels
    assert ttx.simulate(np.full(20_000, 20.0)).spike_times.size == 0


def test_unknown_parameter_is_rejected():
    with pytest.raises(TypeError, match="g_naa"):
        HHNeuron(g_naa=1.0)


def test_defaults_come_from_the_rust_core():
    assert HHNeuron().params["g_na"] == 120.0
    assert HHNeuron(temperature=20.0).params["temperature"] == 20.0
