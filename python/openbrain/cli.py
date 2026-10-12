"""Command-line entry point: `openbrain cortex` and `openbrain connectome`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from openbrain.connectome import build_network
from openbrain.cortex import build_cortex
from openbrain.network import SpikeRecord


def save_raster(record: SpikeRecord, path: str, n_exc: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    exc = record.ids < n_exc
    ax.scatter(record.times[exc], record.ids[exc], s=0.5, c="tab:blue", label="excitatory")
    ax.scatter(record.times[~exc], record.ids[~exc], s=0.5, c="tab:red", label="inhibitory")
    ax.set(xlabel="time (ms)", ylabel="neuron", xlim=(0, record.duration_ms), ylim=(0, record.n))
    ax.legend(loc="upper right", markerscale=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_cortex(args: argparse.Namespace) -> int:
    net = build_cortex(args.exc, args.inh, seed=args.seed, plastic=args.plastic)
    record = net.run(args.ms)
    rates = record.rates()
    print(f"simulated {record.n} neurons for {args.ms} ms")
    print(f"excitatory: {rates[: args.exc].mean():.1f} Hz")
    print(f"inhibitory: {rates[args.exc :].mean():.1f} Hz")
    if args.raster:
        save_raster(record, args.raster, args.exc)
        print(f"raster written to {args.raster}")
    return 0


def run_connectome(args: argparse.Namespace) -> int:
    names, net = build_network(args.edges, preset=args.preset, scale=args.scale, seed=args.seed)
    index = {name: i for i, name in enumerate(names)}
    unknown = [s for s in args.stimulate if s not in index]
    if unknown:
        print(f"unknown neurons: {', '.join(unknown)}")
        return 2
    drive = np.zeros(len(names))
    drive[[index[s] for s in args.stimulate]] = args.current
    rates = net.run(args.ms, external=drive).rates()
    print(f"simulated {len(names)} neurons for {args.ms} ms")
    for i in np.argsort(-rates)[: args.top]:
        if rates[i] > 0:
            print(f"{names[i]} {rates[i]:.1f} Hz")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openbrain", description="Spiking brain simulator.")
    sub = parser.add_subparsers(dest="command", required=True)

    cortex = sub.add_parser("cortex", help="run the Izhikevich 2003 cortical network")
    cortex.add_argument("--exc", type=int, default=800, help="excitatory neurons")
    cortex.add_argument("--inh", type=int, default=200, help="inhibitory neurons")
    cortex.add_argument("--ms", type=int, default=1000, help="simulated time")
    cortex.add_argument("--seed", type=int, default=0)
    cortex.add_argument("--plastic", action="store_true", help="enable STDP learning")
    cortex.add_argument("--raster", help="write a spike raster PNG here")
    cortex.set_defaults(func=run_cortex)

    conn = sub.add_parser("connectome", help="simulate a wiring diagram from a CSV edge list")
    conn.add_argument("edges", help="CSV with columns pre, post, weight")
    conn.add_argument("--stimulate", action="append", default=[], help="neuron to drive")
    conn.add_argument("--current", type=float, default=10.0, help="drive current")
    conn.add_argument("--scale", type=float, default=1.0, help="synaptic weight multiplier")
    conn.add_argument("--preset", default="RS", help="neuron type for every node")
    conn.add_argument("--ms", type=int, default=1000)
    conn.add_argument("--seed", type=int, default=0)
    conn.add_argument("--top", type=int, default=20, help="how many neurons to list")
    conn.set_defaults(func=run_connectome)

    args = parser.parse_args(argv)
    return args.func(args)
