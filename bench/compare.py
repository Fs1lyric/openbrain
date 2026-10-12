"""Run the openbrain benchmark workload in NEURON and Jaxley and compare speed and accuracy.

Every simulator gets the identical cable: one cylinder per SWC point, exported by
`cargo run --release --example bench`. The reference is NEURON on that same cable with 9 segments
per cylinder and dt = 0.0025 ms (Crank-Nicolson), converged far below the errors reported.

Usage: uv run python compare.py openbrain.json results.json
"""

import json
import sys
import time

import numpy as np

REPEATS = 21


def median_seconds(f):
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def pulse(t, onset, tau_rise, tau_decay):
    """Peak-normalised double exponential, as in openbrain's `double_exponential`."""
    t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    norm = 1.0 / (np.exp(-t_peak / tau_decay) - np.exp(-t_peak / tau_rise))
    s = t - onset
    return np.where(s >= 0, norm * (np.exp(-s / tau_decay) - np.exp(-s / tau_rise)), 0.0)


def neuron_run(w, nseg, dt, second_order, timed):
    from neuron import h

    h.load_file("stdrun.hoc")
    secs = []
    for i, (parent, length, diam) in enumerate(
        zip(w["parent"], w["length_um"], w["diameter_um"], strict=True)
    ):
        sec = h.Section(name=f"c{i}")
        sec.L, sec.diam, sec.nseg = length, diam, nseg
        sec.Ra, sec.cm = w["ri"], w["cm"]
        sec.insert("pas")
        for seg in sec:
            seg.pas.g, seg.pas.e = w["g_leak"], w["e_leak"]
        if parent >= 0:
            sec.connect(secs[parent](1), 0)
        secs.append(sec)
    t_end = w["steps"] * w["dt"]
    t_play = np.arange(0.0, t_end + 1.0, 0.005)
    keep = []
    for site, onset, amp in zip(w["sites"], w["onsets_ms"], w["amplitudes"], strict=True):
        clamp = h.IClamp(secs[site](0.5))
        clamp.delay, clamp.dur = 0, 1e9
        tvec = h.Vector(t_play)
        ivec = h.Vector(amp * pulse(t_play, onset, w["tau_rise"], w["tau_decay"]))
        ivec.play(clamp._ref_amp, tvec, 1)
        keep += [clamp, tvec, ivec]
    soma = h.Vector().record(secs[0](0.5)._ref_v)
    tvec = h.Vector().record(h._ref_t)
    h.dt, h.steps_per_ms = dt, 1.0 / dt
    h.secondorder = second_order
    h.celsius = 6.3

    def run():
        h.finitialize(w["e_leak"])
        h.continuerun(t_end)

    seconds = median_seconds(run) if timed else (run(), None)[1]
    run()
    t, v = np.array(tvec), np.array(soma)
    for sec in secs:
        h.delete_section(sec=sec)
    return t, v, seconds


def unbranched_paths(parent):
    """Split the compartment tree into unbranched chains (Jaxley branches)."""
    n = len(parent)
    children = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    branches, branch_parent, branch_of = [], [], {}
    for i in range(n):
        p = parent[i]
        if p >= 0 and len(children[p]) == 1 and p != 0:
            b = branch_of[p]
            branches[b].append(i)
        else:
            b = len(branches)
            branches.append([i])
            branch_parent.append(-1 if p < 0 else branch_of[p])
        branch_of[i] = b
    return branches, branch_parent, branch_of


def jaxley_run(w):
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    from jaxley.channels import Leak

    jax.config.update("jax_enable_x64", True)
    branches, branch_parent, branch_of = unbranched_paths(w["parent"])
    comp = jx.Compartment()
    cell = jx.Cell([jx.Branch(comp, ncomp=len(b)) for b in branches], parents=branch_parent)
    order = [i for b in branches for i in b]
    cell.set("length", np.array(w["length_um"])[order])
    cell.set("radius", np.array(w["diameter_um"])[order] / 2)
    cell.set("axial_resistivity", w["ri"])
    cell.set("capacitance", w["cm"])
    cell.insert(Leak())
    cell.set("Leak_gLeak", w["g_leak"])
    cell.set("Leak_eLeak", w["e_leak"])
    cell.set("v", w["e_leak"])
    cell.branch(0).comp(0).record("v", verbose=False)

    dt, steps = w["dt"], w["steps"]
    t = np.arange(steps + 1) * dt
    location = [(branch_of[s], branches[branch_of[s]].index(s)) for s in w["sites"]]
    waves = jnp.array([pulse(t, onset, w["tau_rise"], w["tau_decay"]) for onset in w["onsets_ms"]])
    lo, hi = w["window"]

    def simulate(amplitudes):
        stimuli = None
        for (b, c), wave, amp in zip(location, waves, amplitudes, strict=True):
            stimuli = cell.branch(b).comp(c).data_stimulate(amp * wave, stimuli, verbose=False)
        return jx.integrate(cell, delta_t=dt, t_max=steps * dt, data_stimuli=stimuli)[0]

    forward = jax.jit(simulate)
    loss = jax.jit(jax.value_and_grad(lambda a: jnp.mean(simulate(a)[lo + 1 : hi + 1])))
    amps = jnp.array(w["amplitudes"])
    start = time.perf_counter()
    forward(amps).block_until_ready()
    compile_forward = time.perf_counter() - start
    start = time.perf_counter()
    jax.block_until_ready(loss(amps))
    compile_grad = time.perf_counter() - start
    t_forward = median_seconds(lambda: forward(amps).block_until_ready())
    t_grad = median_seconds(lambda: jax.block_until_ready(loss(amps)))
    v = np.array(forward(amps))
    value, grad = loss(amps)
    return {
        "t": t[: len(v)],
        "v": v,
        "t_forward": t_forward,
        "t_grad": t_grad,
        "compile_forward": compile_forward,
        "compile_grad": compile_grad,
        "loss": float(value),
        "grad": np.array(grad),
        "compartments": len(order),
    }


def main():
    w = json.load(open(sys.argv[1]))
    dt, steps = w["dt"], w["steps"]
    t_ours = (np.arange(steps) + 1) * dt
    v_ours = np.array(w["soma"])

    t_ref, v_ref, _ = neuron_run(w, nseg=9, dt=0.0025, second_order=2, timed=False)

    def error(t, v):
        diff = np.interp(t_ours, t, v) - np.interp(t_ours, t_ref, v_ref)
        return float(np.abs(diff).max()), float(np.sqrt(np.mean(diff**2)))

    t_n, v_n, neuron_seconds = neuron_run(w, nseg=1, dt=dt, second_order=0, timed=True)
    jx_result = jaxley_run(w)

    lo, hi = w["window"]
    # finite-difference gradient on the reference, to check both gradients against it
    ref_mean = np.mean(np.interp(t_ours[lo:hi], t_ref, v_ref))
    peak = float((v_ref - w["e_leak"]).max())
    rows = [
        {
            "simulator": "openbrain (Rust, 1 thread)",
            "compartments": len(w["parent"]),
            "forward_ms": w["t_forward"] * 1e3,
            "gradient_ms": w["t_grad"] * 1e3,
            "max_err_mV": error(t_ours, v_ours)[0],
            "rms_err_mV": error(t_ours, v_ours)[1],
            "grad": list(w["grad"]),
        },
        {
            "simulator": f"NEURON {__import__('neuron').__version__}",
            "compartments": len(w["parent"]),
            "forward_ms": neuron_seconds * 1e3,
            "gradient_ms": None,
            "max_err_mV": error(t_n, v_n)[0],
            "rms_err_mV": error(t_n, v_n)[1],
        },
        {
            "simulator": f"Jaxley {__import__('jaxley').__version__} (CPU, JIT)",
            "compartments": jx_result["compartments"],
            "forward_ms": jx_result["t_forward"] * 1e3,
            "gradient_ms": jx_result["t_grad"] * 1e3,
            "compile_forward_s": jx_result["compile_forward"],
            "compile_grad_s": jx_result["compile_grad"],
            "max_err_mV": error(jx_result["t"], jx_result["v"])[0],
            "rms_err_mV": error(jx_result["t"], jx_result["v"])[1],
            "grad": list(jx_result["grad"]),
        },
    ]
    g_ours, g_jx = np.array(w["grad"]), jx_result["grad"]
    summary = {
        "cell": "n419 (CA1 pyramidal, NeuroMorpho)",
        "simulated_ms": steps * dt,
        "dt_ms": dt,
        "reference": "NEURON, 9 segments per cylinder, dt 0.0025 ms, Crank-Nicolson",
        "reference_peak_depolarisation_mV": peak,
        "reference_mean_soma_mV": ref_mean,
        "gradient_rel_diff_openbrain_vs_jaxley": float(
            np.linalg.norm(g_ours - g_jx) / np.linalg.norm(g_jx)
        ),
        "rows": rows,
        "traces": {
            "t": t_ours.tolist(),
            "openbrain": v_ours.tolist(),
            "reference": np.interp(t_ours, t_ref, v_ref).tolist(),
            "neuron": np.interp(t_ours, t_n, v_n).tolist(),
            "jaxley": np.interp(t_ours, jx_result["t"], jx_result["v"]).tolist(),
        },
    }
    json.dump(summary, open(sys.argv[2], "w"), indent=1)
    print(f"reference peak depolarisation {peak:.2f} mV")
    print(
        f"{'simulator':34} {'comps':>6} {'fwd ms':>8} {'grad ms':>8} {'max err':>9} {'rms err':>9}"
    )
    for r in rows:
        g = "-" if r["gradient_ms"] is None else f"{r['gradient_ms']:.1f}"
        print(
            f"{r['simulator']:34} {r['compartments']:6d} {r['forward_ms']:8.1f} {g:>8} "
            f"{r['max_err_mV']:9.2e} {r['rms_err_mV']:9.2e}"
        )
    print(
        "gradient rel. diff openbrain vs Jaxley:", summary["gradient_rel_diff_openbrain_vs_jaxley"]
    )
    print(
        f"Jaxley JIT compile: forward {jx_result['compile_forward']:.1f}s, "
        f"gradient {jx_result['compile_grad']:.1f}s"
    )


if __name__ == "__main__":
    main()
