"""Single-compartment Hodgkin-Huxley: openbrain vs NEURON's built-in `hh` mechanism.

Run from the repo root: `uv sync --group bench && uv run python bench/hh_vs_neuron.py`.

Both simulators get HhParams::default() (6.3 degC) and a constant 10 uA/cm^2 for 1000 ms from rest.
NEURON's soma is a cylinder with L = diam and area 100 um^2, so the IClamp amplitude in nA is the
openbrain current density times area. NEURON runs twice: with its default 1 mV rate lookup table
(`usetable_hh = 1`) and with exact rates (`usetable_hh = 0`). A NEURON CVODE run with tight
tolerances (exact rates) serves as the reference for spike times; openbrain at dt 0.001 ms is
reported against it as an independent check of that reference. NEURON fixed-step runs use backward
Euler (its default) and, as a further variant, Crank-Nicolson (`secondorder = 2`).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time

import numpy as np
from neuron import h

import openbrain

T_STOP = 1000.0
I_DENSITY = 10.0  # uA/cm^2
AREA_UM2 = 100.0
REPEATS = 5
THRESHOLD = 0.0

h.load_file("stdrun.hoc")


def crossings(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Upward crossings of THRESHOLD, linearly interpolated (same rule as openbrain)."""
    idx = np.nonzero((v[:-1] < THRESHOLD) & (v[1:] >= THRESHOLD))[0]
    frac = (THRESHOLD - v[idx]) / (v[idx + 1] - v[idx])
    return t[idx] + frac * (t[idx + 1] - t[idx])


def median_time(run) -> tuple[float, object]:
    times, result = [], None
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def build_neuron_soma():
    soma = h.Section(name="soma")
    soma.L = soma.diam = math.sqrt(AREA_UM2 / math.pi)
    soma.nseg = 1
    soma.cm = 1.0
    soma.insert("hh")
    seg = soma(0.5)
    seg.hh.gnabar = 0.120  # S/cm^2
    seg.hh.gkbar = 0.036
    seg.hh.gl = 0.0003
    seg.hh.el = -54.387
    seg.ena = 50.0
    seg.ek = -77.0
    h.celsius = 6.3
    stim = h.IClamp(seg)
    stim.delay = 0.0
    stim.dur = T_STOP
    stim.amp = I_DENSITY * seg.area() * 1e-5  # uA/cm^2 * um^2 -> nA
    return soma, seg, stim


# (label, usetable_hh, secondorder): NEURON's default is the 1 mV lookup table with backward Euler.
NEURON_VARIANTS = [
    ("neuron_default", 1, 0),
    ("neuron_exact_rates", 0, 0),
    ("neuron_exact_rates_cn", 0, 2),
]


def run_neuron(seg, dt: float, usetable: int, secondorder: int):
    """Median wall time of a fixed-step run (finitialize + ParallelContext.psolve, which loops
    in C) and of the same run through stdrun's hoc-level continuerun.

    Returns (psolve time, continuerun time, (t, v) of the last run)."""
    h.usetable_hh = usetable
    h.secondorder = secondorder
    h.cvode.active(0)
    h.dt = dt
    h.steps_per_ms = 1.0 / dt
    pc = h.ParallelContext()
    pc.set_maxstep(10)
    v = h.Vector().record(seg._ref_v)
    t = h.Vector().record(h._ref_t)

    def run(solve):
        h.finitialize(-65.0)
        solve(T_STOP)
        return np.array(t), np.array(v)

    time_psolve, trace = median_time(lambda: run(pc.psolve))
    time_continuerun, _ = median_time(lambda: run(h.continuerun))
    h.secondorder = 0
    return time_psolve, time_continuerun, trace


def neuron_reference(seg) -> np.ndarray:
    """Spike times from NEURON's variable-step CVODE with exact rates and tight tolerances."""
    h.usetable_hh = 0
    h.cvode.active(1)
    h.cvode.atol(1e-9)
    h.cvode.rtol(1e-9)
    spikes = h.Vector()
    nc = h.NetCon(seg._ref_v, None, sec=seg.sec)
    nc.threshold = THRESHOLD
    nc.record(spikes)
    h.finitialize(-65.0)
    h.continuerun(T_STOP)
    h.cvode.active(0)
    del nc
    return np.array(spikes)


def run_openbrain(dt: float):
    neuron = openbrain.HHNeuron()
    current = np.full(round(T_STOP / dt), I_DENSITY)
    return median_time(lambda: neuron.simulate(current, dt))


def spike_stats(spikes: np.ndarray, ref: np.ndarray) -> dict:
    n = min(10, len(spikes), len(ref))
    diffs = spikes[:n] - ref[:n]
    return {
        "count": len(spikes),
        "first10_ms": np.round(spikes[:10], 4).tolist(),
        "first10_minus_ref_ms": np.round(diffs, 4).tolist(),
        "max_abs_first10_vs_ref_ms": float(np.max(np.abs(diffs))) if n else None,
        "last_minus_ref_ms": float(spikes[-1] - ref[len(spikes) - 1])
        if len(spikes) <= len(ref) and len(spikes)
        else None,
    }


def main() -> None:
    soma, seg, _stim = build_neuron_soma()
    ref = neuron_reference(seg)
    report = {
        "neuron_version": h.nrnversion(5),
        "soma_area_um2": seg.area(),
        "reference_cvode_spike_count": len(ref),
        "reference_first10_ms": np.round(ref[:10], 4).tolist(),
        "runs": [],
    }
    # Independent check of the reference: openbrain's exponential integrator at a tiny step.
    _, fine = run_openbrain(0.001)
    report["openbrain_dt0.001_vs_reference"] = spike_stats(np.asarray(fine.spike_times), ref)
    for dt in (0.025, 0.01):
        ob_time, ob = run_openbrain(dt)
        ob_spikes = np.asarray(ob.spike_times)
        row = {"dt_ms": dt, "openbrain": {"time_s": ob_time, **spike_stats(ob_spikes, ref)}}
        for label, usetable, secondorder in NEURON_VARIANTS:
            nrn_time, nrn_time_hoc, (t, v) = run_neuron(seg, dt, usetable, secondorder)
            # NEURON records t = 0 too; openbrain's v[i] is the voltage at (i + 1) * dt.
            n = min(len(ob.v), len(v) - 1)
            assert np.allclose(t[1 : n + 1], ob.t[:n], atol=1e-6), "time grids differ"
            dv = ob.v[:n] - v[1 : n + 1]
            early = ob.t[:n] <= 50.0
            nrn_spikes = crossings(t, v)
            ob_vs_nrn = ob_spikes[: min(10, len(ob_spikes), len(nrn_spikes))]
            row[label] = {
                "time_s": nrn_time,
                "time_continuerun_s": nrn_time_hoc,
                **spike_stats(nrn_spikes, ref),
                "rms_v_diff_vs_openbrain_mv": float(np.sqrt(np.mean(dv**2))),
                "rms_v_diff_vs_openbrain_first50ms_mv": float(np.sqrt(np.mean(dv[early] ** 2))),
                "openbrain_minus_neuron_first10_ms": np.round(
                    ob_vs_nrn - nrn_spikes[: len(ob_vs_nrn)], 4
                ).tolist(),
            }
        report["runs"].append(row)
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
