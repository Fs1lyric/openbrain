"""Load real wiring diagrams (e.g. the C. elegans connectome) from CSV edge lists."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from openbrain.network import Network
from openbrain.neurons import Izhikevich

REQUIRED_COLUMNS = {"pre", "post", "weight"}


def load_edges(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Read a `pre,post,weight` CSV into neuron names and a W[post, pre] matrix.

    Names are ordered by first appearance; repeated edges are summed.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if not REQUIRED_COLUMNS <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns pre, post, weight; got {reader.fieldnames}")
        edges = [(row["pre"], row["post"], float(row["weight"])) for row in reader]
    index: dict[str, int] = {}
    for pre, post, _ in edges:
        index.setdefault(pre, len(index))
        index.setdefault(post, len(index))
    weights = np.zeros((len(index), len(index)))
    for pre, post, w in edges:
        weights[index[post], index[pre]] += w
    return list(index), weights


def build_network(
    path: str | Path, preset: str = "RS", scale: float = 1.0, seed: int | None = None
) -> tuple[list[str], Network]:
    """Wire one neuron per connectome node, with synaptic weights multiplied by `scale`."""
    names, weights = load_edges(path)
    neurons = Izhikevich.from_preset(preset, n=len(names))
    return names, Network(neurons, weights * scale, seed=seed)
