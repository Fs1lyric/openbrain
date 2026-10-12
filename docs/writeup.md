# Training a real neuron's synapses through the cable equations

*Nabhan Lyric Chowdhury · openbrain · October 2026*

![Hero figure](hero.png)

## Summary

Asking what one neuron can compute usually means simulating it in NEURON and then fitting a deep
network that mimics it, which costs GPU-years for a few dozen cells (Aizenbud et al. 2025). openbrain
skips the surrogate. It is a Rust cable simulator with exact discrete-adjoint gradients, so a
reconstructed neuron's own synaptic weights can be trained directly. Three results:

1. **Pilot.** A real CA1 pyramidal cell (NeuroMorpho n419, 4,229 compartments, 96 synapses) learns
   XOR in 34–98 Adam steps and 9–23 CPU-seconds, on every seed tried (6/6).
2. **Control.** With the same synapses and the same budget, dendritic placements beat a soma-placed
   control in 8/8 random placements (best margin 1.27–2.81 mV vs 1.19 mV). A point neuron of matched
   membrane area reaches 1.65 mV, which 5/8 placements beat. Dendrites help on this task; they are
   not required for it.
3. **Benchmark.** On an identical cable, the forward simulation runs 1.4× faster than NEURON 9.0.2,
   and forward plus gradient runs 2.4× faster than Jaxley 0.14.0 on CPU. Errors against a converged
   reference stay within 15 µV for a 2.6 mV response.

## Model

**Cable.** Each SWC point becomes a cylinder from its parent point, and the soma a cylinder whose
length equals its diameter (same area as a sphere). The membrane is passive (C_m 1 µF/cm²,
g_leak 10⁻⁴ S/cm², E_leak −70 mV, R_a 150 Ω·cm). Each backward-Euler step is one O(n) Hines
solve on the tree.

**Synapses.** AMPA (τ 0.2/2 ms), NMDA (2/50 ms, Mg²⁺ block of Jahr & Stevens 1990) and GABA_A
(0.5/8 ms, E −80 mV) are conductances with peak-normalised double-exponential time courses. The
NMDA block is evaluated at the previous step's voltage, so every step stays one linear tree solve.
A current-based synapse was added for the benchmark.

**Gradients.** The loss depends on the soma voltage. Its exact gradient with respect to every
weight comes from the discrete adjoint of the time-stepping scheme. The system matrix is
symmetric, so each backward step is also one Hines solve, and a gradient costs about two forward
simulations whatever the number of weights. Tests check the adjoint against central finite
differences to a relative error of 10⁻⁶ for every synapse type.

## XOR task

Inputs A and B each drive AMPA, NMDA and GABA_A synapses at 16 shared dendritic sites (96
weights, w = w₀·e^θ to keep them positive). An input is a 3-spike burst at 10, 12 and 14 ms. The
readout is the mean soma voltage over 10–60 ms. XOR counts as solved when the lowest XOR-true
pattern (10, 01) sits at least 1 mV above the highest XOR-false pattern (00, 11).

The loss is a softplus on each true/false pair: softplus(β(Δ* − (m_true − m_false))), with
β = 2/mV and Δ* = 3 mV. An earlier version used squared error against fixed voltage targets. It
stalled at a 0.91 mV gap after 1,500 steps, because most of its gradient went into pulling pattern
11 toward an unreachable −70 mV. Pattern 00 has no input, so a passive cell sits exactly at rest
and it needs no simulation.

## The control, and a hypothesis that failed

The plan was to show that XOR needs dendrites. At steady state that holds for a single
compartment: with conductances g_i at reversals E_i, V₁₁ > min(V₁₀, V₀₁) whenever both
single-input responses depolarise, so pattern 11 cannot sit below both (`tests/xor.rs` checks
this). The test built on that argument failed: a point neuron learned the mean-voltage XOR in 66
steps. Over time the argument breaks down. The early GABA_A dips of two inputs add almost
linearly, while their slower excitation saturates, so the 11 response averages lower.

So the control became a capacity measurement: the best margin reached in 300 steps, with the
stopping threshold out of reach. A second mistake turned up here. Setting Δ* = 20 mV saturates
the softplus, and the loss then rewards raising every pattern, including 11. Measured with the
default Δ* = 3 mV:

| Configuration | Best XOR margin (mV) |
| --- | --- |
| Real tree, 8 random placements | 1.27, 1.30, 1.34, 1.76, 1.94, 2.53, 2.59, 2.81 (median 1.85) |
| Same synapses, all on the soma of the real tree | 1.19 |
| Point neuron, same membrane area | 1.65 |

Dendritic placement beats the soma control every time. It beats the point neuron by up to 1.7×,
and loses to it in 3 of 8 placements. So *where* synapses sit matters about as much as having
dendrites at all. That is the question the Neuron Compute Atlas will measure across many cells
([research plan](research/plan.md)).

## Benchmark

All three simulators run the same cable: openbrain's 4,229 cylinders exported verbatim to NEURON
(one section per cylinder) and Jaxley (one branch per unbranched chain). Sixteen dendritic sites
each receive a current pulse (τ 1/5 ms, 0.02 nA) at 10 + 2k ms. Sixty milliseconds are
simulated at dt = 0.1 ms with backward Euler. The reference is NEURON on the same cable with 9
segments per cylinder and dt = 0.0025 ms (Crank–Nicolson). The gradient is d(mean soma
voltage)/d(16 amplitudes). Times are medians of 21 runs on one machine (16 cores). openbrain and
NEURON use one thread; JAX picks its own threading, and its 1–2 s compile is excluded.
Jaxley's gradient time varied between 154 and 182 ms across runs; the table shows the run in the
figure.

| Simulator | Forward | Forward + gradient | Max error | RMS error |
| --- | --- | --- | --- | --- |
| openbrain | 33 ms | 63 ms | 15 µV | 8 µV |
| NEURON 9.0.2 | 48 ms | — | 7 µV | 3 µV |
| Jaxley 0.14.0 | 40 ms | 154 ms | 15 µV | 7 µV |

The two gradients agree to 0.2% (relative L2). NEURON is about twice as accurate at equal dt.
openbrain applies a step's input at the start of the step rather than the end, an O(dt) timing
offset that is the likely cause. That is the next fix.

## Limitations

- XOR in a single neuron is known (Gidon et al. 2020; Aizenbud et al. 2026). The contribution here
  is cost: no surrogate network, seconds on a CPU.
- Membranes are passive, so there are no dendritic spikes beyond NMDA's voltage dependence.
  Active channels in the dendrites will change the capacity numbers.
- The control uses one cell and eight placements, and margins come from a fixed 300-step budget,
  so they are lower bounds on capacity.
- Jaxley is designed for GPUs and many cells in parallel; this CPU comparison does not test that.

## Reproduce

Needs Rust, [mise](https://mise.jdx.dev) and [uv](https://docs.astral.sh/uv/). Download
`n419.CNG.swc` from [NeuroMorpho.Org](https://neuromorpho.org) into `data/neuromorpho/`.

```sh
cargo test --release                                                # 25 tests
cargo run --release --example xor -- data/neuromorpho/n419.CNG.swc --seed 2
cargo run --release --example xor -- data/neuromorpho/n419.CNG.swc --mode point --iters 300 --margin 100
cargo run --release --example bench -- data/neuromorpho/n419.CNG.swc bench/openbrain.json
cd bench && uv sync --python 3.12 && uv run python compare.py openbrain.json results.json
uv run python figure.py                                             # writes docs/hero.png
```

## References

- Aizenbud, Yoeli, Beniaguev, de Kock, London, Segev. What makes human cortical pyramidal neurons
  functionally complex. bioRxiv 2024.12.17.628883 (2025).
- Aizenbud et al. What can a neuron compute. bioRxiv 2026.06.08.730984 (2026).
- Beniaguev, Segev, London. Single cortical neurons as deep artificial neural networks. *Neuron* (2021).
- Deistler et al. Jaxley: differentiable simulation enables large-scale training of detailed
  biophysical models of neural dynamics. *Nature Methods* (2025).
- Gidon et al. Dendritic action potentials and computation in human layer 2/3 cortical neurons.
  *Science* (2020).
- Hines. Efficient computation of branched nerve equations. *Int. J. Bio-Medical Computing* (1984).
- Jahr, Stevens. Voltage dependence of NMDA-activated macroscopic conductances predicted by
  single-channel kinetics. *J. Neurosci.* (1990).
