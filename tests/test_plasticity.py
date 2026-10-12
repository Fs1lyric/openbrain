import numpy as np

from openbrain.plasticity import STDP


def run_pairing(pre_first: bool) -> np.ndarray:
    weights = np.full((2, 2), 1.0)
    np.fill_diagonal(weights, 0.0)
    stdp = STDP(n=2, mask=weights > 0, w_max=5.0)
    first, second = (0, 1) if pre_first else (1, 0)
    for t in range(20):
        spikes = np.zeros(2, dtype=bool)
        if t == 0:
            spikes[first] = True
        if t == 5:
            spikes[second] = True
        stdp.update(weights, spikes)
    return weights


def test_pre_before_post_potentiates():
    w = run_pairing(pre_first=True)
    assert w[1, 0] > 1.0  # 0 fired before 1: 0 -> 1 strengthens
    assert w[0, 1] < 1.0  # 1 fired after 0: 1 -> 0 weakens


def test_weights_stay_within_bounds_and_mask():
    weights = np.array([[0.0, 4.99], [0.01, 0.0]])
    stdp = STDP(n=2, mask=weights > 0, w_max=5.0, a_plus=1.0, a_minus=1.0)
    for t in range(50):
        stdp.update(weights, np.array([t % 2 == 0, t % 2 == 1]))
        assert weights.min() >= 0.0
        assert weights.max() <= 5.0
        assert weights[0, 0] == 0.0 and weights[1, 1] == 0.0
