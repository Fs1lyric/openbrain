"""Passive cable on real reconstructions: openbrain vs NEURON.

Run from the repo root:

    cargo build --release --example cable_bench
    uv sync --group bench
    uv run python bench/cable_vs_neuron.py bench/data/neuromorpho/*.swc

openbrain is driven through `examples/cable_bench.rs`, which also prints the geometry it built
(one cylinder per SWC point). NEURON is run on two models of each cell:

* `matched`: one NEURON section (nseg = 1) per openbrain compartment, with the same L and diam.
  Each child attaches at its parent's centre node (x = 0.5), and its Ra is scaled so that the
  resistance from its centre to its 0-end equals openbrain's half(child) + half(parent). This
  makes NEURON's linear system identical to openbrain's, so it isolates solver numerics.
* `import3d`: the SWC loaded with NEURON's own Import3d and discretised with the usual d_lambda
  rule (0.1 lambda at 100 Hz). This is how a NEURON user would model the cell, so differences here
  measure openbrain's geometry conventions, not its solver.

Passive properties: cm 1 uF/cm^2, g_pas 1e-4 S/cm^2, e_pas -70 mV, Ra 100 ohm cm. 0.1 nA is
injected at the soma. Steady state in NEURON comes from `Impedance` at 0 Hz; the transient is
100 ms of fixed-step backward Euler at dt 0.025 ms (NEURON's default method, same as openbrain).
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from neuron import h

CM, G_PAS, E_PAS, RA = 1.0, 1e-4, -70.0, 100.0
I_NA = 0.1
DT = 0.025
T_STOP = 100.0
REPEATS = 5
EXAMPLE = Path(__file__).resolve().parent.parent / "target/release/examples/cable_bench"

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")


def median_time(run) -> tuple[float, object]:
    times, result = [], None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def run_openbrain(swc: Path) -> dict:
    out = subprocess.run(
        [str(EXAMPLE), str(swc), str(REPEATS)], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout)


def clear_neuron() -> None:
    for sec in list(h.allsec()):
        h.delete_section(sec=sec)


def half_resistance(length_um: float, diameter_um: float) -> float:
    """Axial resistance of half a cylinder, ohm (same formula as Cable::new)."""
    radius_cm = diameter_um * 0.5e-4
    return RA * length_um * 0.5e-4 / (math.pi * radius_cm**2)


def build_matched(ob: dict):
    """One nseg=1 section per openbrain compartment; returns the soma section and all sections."""
    parent, length, diam = ob["parent"], ob["length_um"], ob["diameter_um"]
    secs = []
    for i, (p, L, d) in enumerate(zip(parent, length, diam, strict=True)):
        sec = h.Section(name=f"c{i}")
        sec.L, sec.diam, sec.nseg = L, d, 1
        if p < 0:
            sec.Ra = RA
        else:
            half_child = half_resistance(L, d)
            sec.Ra = RA * (half_child + half_resistance(length[p], diam[p])) / half_child
            sec.connect(secs[p](0.5), 0)
        secs.append(sec)
    return secs[0], secs


def build_import3d(swc: Path):
    reader = h.Import3d_SWC_read()
    reader.input(str(swc))
    h.Import3d_GUI(reader, False).instantiate(None)
    secs = list(h.allsec())
    for sec in secs:
        sec.Ra = RA
        sec.cm = CM
        lam = 1e5 * math.sqrt(sec.diam / (4 * math.pi * 100.0 * sec.Ra * sec.cm))
        sec.nseg = int((sec.L / (0.1 * lam) + 0.9) / 2) * 2 + 1
    return h.soma[0], secs


def measure_neuron(soma, secs) -> dict:
    for sec in secs:
        sec.cm = CM
        sec.insert("pas")
        for seg in sec:
            seg.pas.g = G_PAS
            seg.pas.e = E_PAS
    seg = soma(0.5)

    imp = h.Impedance()
    imp.loc(0.5, sec=soma)

    def steady():
        imp.compute(0, 1)
        return imp.input(0.5, sec=soma)

    t_steady, r_in = median_time(steady)

    stim = h.IClamp(seg)
    stim.delay, stim.dur, stim.amp = 0.0, 1e9, I_NA
    h.secondorder = 0
    h.cvode.active(0)
    h.dt = DT
    h.steps_per_ms = 1.0 / DT
    pc = h.ParallelContext()
    pc.set_maxstep(10)
    v = h.Vector().record(seg._ref_v)

    def transient():
        h.finitialize(E_PAS)
        pc.psolve(T_STOP)
        return np.array(v)

    t_transient, trace = median_time(transient)
    return {
        "n_sections": len(secs),
        "n_segments": sum(sec.nseg for sec in secs),
        "soma_area_um2": sum(s.area() for s in soma),
        "total_area_um2": sum(s.area() for sec in secs for s in sec),
        "r_in_mohm": r_in,
        "soma_v_ss_mv": E_PAS + r_in * I_NA,
        "time_steady_s": t_steady,
        "time_transient_s": t_transient,
        "soma_trace_mv": trace[1:],  # drop t = 0 to match openbrain's v after each step
    }


def compare(swc: Path) -> dict:
    ob = run_openbrain(swc)
    ob_trace = np.asarray(ob["soma_trace_mv"])
    length, diam = np.asarray(ob["length_um"]), np.asarray(ob["diameter_um"])
    row = {
        "cell": swc.stem.removesuffix(".CNG"),
        "openbrain": {
            "n_compartments": ob["n_compartments"],
            "soma_area_um2": math.pi * diam[0] * length[0],
            "total_area_um2": float(np.sum(math.pi * diam * length)),
            **{k: ob[k] for k in ("r_in_mohm", "soma_v_ss_mv", "time_build_s")},
            **{k: ob[k] for k in ("time_steady_s", "time_transient_s")},
        },
    }
    for label, build in (("matched", lambda: build_matched(ob)), ("import3d", lambda: None)):
        clear_neuron()
        start = time.perf_counter()
        soma, secs = build() if label == "matched" else build_import3d(swc)
        build_s = time.perf_counter() - start
        res = measure_neuron(soma, secs)
        trace = res.pop("soma_trace_mv")
        n = min(len(trace), len(ob_trace))
        res["time_build_s"] = build_s
        res["rms_transient_vs_openbrain_mv"] = float(
            np.sqrt(np.mean((trace[:n] - ob_trace[:n]) ** 2))
        )
        res["max_abs_transient_vs_openbrain_mv"] = float(np.max(np.abs(trace[:n] - ob_trace[:n])))
        res["r_in_rel_diff_openbrain_vs_this"] = ob["r_in_mohm"] / res["r_in_mohm"] - 1.0
        row[f"neuron_{label}"] = res
    clear_neuron()
    return row


def main() -> None:
    swcs = [Path(p) for p in sys.argv[1:]]
    if not swcs:
        sys.exit("usage: cable_vs_neuron.py file.swc [...]")
    if not EXAMPLE.exists():
        sys.exit("build the openbrain example first: cargo build --release --example cable_bench")
    report = {"neuron_version": h.nrnversion(5), "cells": [compare(p) for p in swcs]}
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
