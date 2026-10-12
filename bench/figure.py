"""Build docs/hero.png from the benchmark and XOR results.

Usage (from bench/): uv run python figure.py
"""

import json
import re

import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
SOMA_POINTS = 14

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)


def swc_points(path):
    rows = [line.split() for line in open(path) if line.strip() and not line.startswith("#")]
    ids = {int(r[0]): i for i, r in enumerate(rows)}
    xyz = np.array([[float(v) for v in r[2:5]] for r in rows])
    parent = [ids.get(int(r[6]), -1) for r in rows]
    return xyz, parent


def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", color=INK, fontsize=10, fontweight="bold")


def morphology(ax, xor):
    xyz, parent = swc_points("../data/neuromorpho/n419.CNG.swc")
    for i, p in enumerate(parent):
        if p >= 0:
            ax.plot(
                *zip(xyz[p, :2], xyz[i, :2], strict=True),
                color=NEUTRAL,
                lw=0.6,
                solid_capstyle="round",
            )
    sites = np.array(xor["sites"])
    ax.scatter(*xyz[sites, :2].T, s=40, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    title(ax, "A", "Real CA1 neuron, 16 synapse sites (placement 2)")
    ax.text(
        0.0,
        -0.02,
        "n419 · NeuroMorpho · 4,229 compartments",
        transform=ax.transAxes,
        color=INK_2,
        va="top",
    )


def traces(ax, xor):
    t = (np.arange(len(xor["traces"][0])) + 1) * xor["dt"]
    series = [
        (xor["traces"][0], NEUTRAL, "no input (00)"),
        (xor["traces"][3], ORANGE, "both inputs (11)"),
        (xor["traces"][1], BLUE, "one input (10, 01)"),
    ]
    window = (t > 10) & (t <= 60)
    v = np.array(xor["traces"])
    gap = v[1][window].mean() - v[3][window].mean()
    for (v, color, label), dy in zip(series, [-0.6, 0.9, -0.2], strict=True):
        ax.plot(t, v, color=color, lw=2)
        ax.text(t[-1] + 1, v[-1] + dy, label, color=INK_2, va="center")
    ax.text(
        35,
        -57.5,
        f"mean over shaded window:\none input sits {gap:.1f} mV above both",
        color=INK_2,
        ha="left",
    )
    ax.axvspan(10, 60, color=GRID, alpha=0.35, lw=0)
    ax.set_xlim(0, 85)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("soma voltage (mV)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    title(ax, "B", "It learns XOR by gradient descent (34 steps, 3 s)")


def capacity(ax):
    rows = [dict(re.findall(r"(\w+)=(\S+)", line)) for line in open("xor_capacity.txt")]
    tree_rows = sorted((float(r["best_gap"]), r["seed"]) for r in rows if r["mode"] == "tree")
    tree = [g for g, _ in tree_rows]
    soma = next(float(r["best_gap"]) for r in rows if r["mode"] == "soma")
    point = next(float(r["best_gap"]) for r in rows if r["mode"] == "point")
    y = np.arange(len(tree))
    ax.hlines(y, 0, tree, color=BLUE, lw=2)
    ax.scatter(tree, y, s=40, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for x, label in [(soma, "same synapses on soma"), (point, "point neuron, same area")]:
        ax.axvline(x, color=INK_2, lw=1, ls=(0, (3, 3)))
        ax.text(
            x,
            len(tree) - 0.3,
            f" {label}\n {x:.2f} mV",
            color=INK_2,
            va="bottom",
            ha="left" if x == point else "right",
        )
    ax.set_yticks(y, [f"random placement {s}" for _, s in tree_rows])
    ax.set_ylim(-0.7, len(tree) + 1.3)
    ax.set_xlim(0, max(tree) * 1.1)
    ax.set_xlabel("best XOR margin after 300 steps (mV)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    title(ax, "C", "Dendritic placement beats the soma control 8/8")


def benchmark(ax, bench):
    rows = bench["rows"]
    names = ["openbrain", "NEURON", "Jaxley"]
    colors = [BLUE, ORANGE, AQUA]
    groups = [("forward", "forward_ms"), ("forward + gradient", "gradient_ms")]
    width = 0.26
    for g, (_label, key) in enumerate(groups):
        for k, (row, color) in enumerate(zip(rows, colors, strict=True)):
            x = g + (k - 1) * (width + 0.02)
            value = row[key]
            if value is None:
                ax.text(x, 3, "no\ngradients", ha="center", va="bottom", color=INK_2, fontsize=8)
                continue
            ax.bar(x, value, width, color=color, edgecolor=SURFACE, linewidth=2)
            ax.text(x, value + 3, f"{value:.0f}", ha="center", va="bottom", color=INK)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 1], [g[0] for g in groups])
    ax.set_ylabel("wall time, ms (median of 21)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, names, frameon=False, loc="upper left", labelcolor=INK_2)
    errs = ", ".join(f"{n} {r['max_err_mV'] * 1e3:.0f}" for n, r in zip(names, rows, strict=True))
    ax.text(
        0.0,
        -0.2,
        f"max error vs converged NEURON (µV): {errs}",
        transform=ax.transAxes,
        color=INK_2,
        fontsize=8,
    )
    title(ax, "D", "Same cell, same inputs: 60 ms simulated on CPU")


def main():
    xor = json.load(open("xor_tree_seed2.json"))
    bench = json.load(open("results.json"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    morphology(axes[0, 0], xor)
    traces(axes[0, 1], xor)
    capacity(axes[1, 0])
    benchmark(axes[1, 1], bench)
    fig.savefig("../docs/hero.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
