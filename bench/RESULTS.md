# openbrain vs NEURON: accuracy and speed

Measured 2026-10-12. Raw output: `bench/results/hh_vs_neuron.json`, `bench/results/cable_vs_neuron.json`.

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 5700G (8 cores / 16 threads, boost to 4.67 GHz) |
| RAM | 7.1 GiB total (about 1.3 GiB free while benchmarking; other desktop apps running) |
| OS | Linux 7.2.5 (Arch-based desktop, not an isolated benchmark host) |
| Toolchain | rustc 1.99.0, Python 3.12.14, uv 0.13.0, NEURON 9.0.2 (PyPI wheel) |

All runs are single-threaded. Times are the median of 5 runs, imports and model construction
excluded unless stated. This is a shared desktop with frequency scaling on, so treat differences
under about 20% as noise.

## Commands

```sh
cargo test                       # 15 passed (7 cable, 8 hh)
uv sync && uv run pytest         # 20 passed
cargo build --release --example cable_bench
uv sync --group bench            # adds NEURON
uv run --group bench python bench/hh_vs_neuron.py > bench/results/hh_vs_neuron.json
uv run --group bench python bench/cable_vs_neuron.py \
    bench/data/neuromorpho/n419.CNG.swc \
    bench/data/neuromorpho/Ctrl-Cell3-40x.CNG.swc \
    bench/data/neuromorpho/Purkinje-slice-ageP43-6.CNG.swc > bench/results/cable_vs_neuron.json
```

The SWC files come from NeuroMorpho.Org (`https://neuromorpho.org/dableFiles/<archive>/CNG%20version/<name>.CNG.swc`)
and are gitignored under `bench/data/`:

| Cell | Archive | Species / region | SWC points |
|---|---|---|---|
| n419 | Turner | rat hippocampus CA1 pyramidal (dendrites complete, no axon) | 4229 |
| Ctrl-Cell3-40x | Prince | mouse somatosensory cortex L5 pyramidal (dendrites moderate) | 491 |
| Purkinje-slice-ageP43-6 | Dusart | mouse cerebellar Purkinje cell (dendrites complete) | 4156 |

## 1. Single-compartment Hodgkin-Huxley (`bench/hh_vs_neuron.py`)

Setup: `HhParams::default()` (6.3 °C) in both. NEURON uses its built-in `hh` mechanism on a
cylinder soma of area 100 µm², driven by an IClamp of 0.01 nA (= 10 µA/cm²) for 1000 ms from
v = -65 mV. NEURON variants:

* default: `usetable_hh = 1` (rates from a 1 mV lookup table), backward Euler. This is what you
  get out of the box.
* exact BE: `usetable_hh = 0`, backward Euler.
* exact CN: `usetable_hh = 0`, `secondorder = 2` (Crank-Nicolson).

Reference: NEURON CVODE with exact rates, atol = rtol = 1e-9, spike times from a NetCon at 0 mV
(69 spikes; first at 1.9025 ms). As an independent check, openbrain at dt = 0.001 ms lands within
0.0015 ms of every one of the first 10 reference spikes and 0.0007 ms of the last one, so the two
simulators agree on the converged answer.

Spike error = spike time minus reference spike time. Timing for NEURON is `finitialize` +
`ParallelContext.psolve` (stepping loop in C); the stdrun `continuerun` time is also listed
because that is what most NEURON scripts call. openbrain timing is `HHNeuron.simulate`, including
the Python/Rust list conversion.

| dt (ms) | Simulator | Spikes | Max abs error, first 10 spikes (ms) | Error of last spike (ms) | RMS V vs openbrain, 0–1000 ms (mV) | RMS V vs openbrain, 0–50 ms (mV) | Time (ms) |
|---|---|---|---|---|---|---|---|
| 0.025 | openbrain | 69 | 0.043 | 0.31 | | | 5.3 |
| 0.025 | NEURON default | 68 | 0.45 | 3.2 (spike 68) | 28.0 | 3.59 | 24.2 (continuerun 99.0) |
| 0.025 | NEURON exact BE | 68 | 0.62 | 4.4 (spike 68) | 30.8 | 4.75 | 25.6 (101) |
| 0.025 | NEURON exact CN | 69 | 0.021 | 0.15 | 3.10 | 0.19 | 26.9 (102) |
| 0.01 | openbrain | 69 | 0.0064 | 0.049 | | | 14.9 |
| 0.01 | NEURON default | 69 | 0.077 | 0.53 | 8.71 | 0.79 | 60.9 (249) |
| 0.01 | NEURON exact BE | 69 | 0.24 | 1.77 | 21.7 | 2.00 | 64.2 (256) |
| 0.01 | NEURON exact CN | 69 | 0.0030 | 0.023 | 0.50 | 0.030 | 68.6 (267) |

openbrain minus NEURON for the first 10 spikes (ms), straight from the JSON:

* dt 0.025, NEURON default: -0.018, -0.064, -0.107, -0.150, -0.193, -0.236, -0.280, -0.323, -0.366, -0.409
* dt 0.025, NEURON exact CN: 0.000, 0.003, 0.005, 0.008, 0.010, 0.012, 0.015, 0.017, 0.019, 0.022
* dt 0.01, NEURON default: -0.007, -0.015, -0.022, -0.029, -0.036, -0.043, -0.050, -0.057, -0.064, -0.071
* dt 0.01, NEURON exact CN: 0.000, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.003, 0.003, 0.004

Reading:

* Accuracy: openbrain (Rush-Larsen gates plus an exponential voltage update) is about 10x closer
  to the reference than NEURON's default backward Euler at the same dt, and gets the spike count
  right at dt 0.025 where NEURON BE drops one spike. NEURON with Crank-Nicolson (`secondorder = 2`)
  is about 2x more accurate than openbrain at the same dt. So openbrain is not the most accurate
  fixed-step option; it beats NEURON's default settings only.
* The full-run RMS voltage numbers are dominated by spike-phase drift (a 0.4 ms shift of a 100 mV
  spike produces tens of mV of RMS), so the 0–50 ms column is the more useful waveform comparison.
* Speed: for one compartment openbrain is about 4–4.5x faster than NEURON's C stepping loop
  (psolve) and 17–19x faster than `continuerun`. This mostly measures per-step overhead in
  NEURON's general-purpose machinery; with one compartment there is very little real work.

## 2. Passive cable on real morphologies (`bench/cable_vs_neuron.py`)

Setup: cm 1 µF/cm², g_pas 1e-4 S/cm², e_pas -70 mV, Ra 100 Ω·cm, 0.1 nA injected at the soma
compartment. openbrain is driven by `examples/cable_bench.rs` (parse SWC, `Cable::new`,
`steady_state`, then 4000 backward Euler steps of 0.025 ms). NEURON gets two models of each cell:

* matched: one `nseg = 1` section per openbrain compartment with the same L and diam; children
  attach at the parent's centre node and each child's Ra is scaled so the centre-to-centre
  resistance equals openbrain's `half(child) + half(parent)`. The linear system is then the same
  as openbrain's, so this isolates the solver.
* import3d: NEURON's own SWC import with d_lambda = 0.1 at 100 Hz, i.e. how a NEURON user would
  model the cell. This tests openbrain's geometry conventions against the field standard.

NEURON steady state comes from `Impedance.compute(0)` (input resistance at DC); openbrain's from
its direct Hines solve. These are not the same amount of work (Impedance computes transfer
impedances for the whole tree), so compare the steady-state timings only loosely.

### Accuracy

| Cell | Model | Compartments (NEURON segments) | Soma area (µm²) | Total area (µm²) | R_in (MΩ) | Soma V_ss (mV) | R_in openbrain vs this | Transient RMS vs openbrain (mV) |
|---|---|---|---|---|---|---|---|---|
| n419 (rat CA1) | openbrain | 4229 | 518.4 | 25070 | 58.771 | -64.123 | | |
| | NEURON matched | 4229 | 518.4 | 25070 | 58.771 | -64.123 | -1.2e-12 | 1.4e-11 |
| | NEURON import3d | 780 | 134.9 | 25350 | 57.832 | -64.217 | +1.6% | 0.090 |
| Ctrl-Cell3-40x (mouse L5) | openbrain | 491 | 852.4 | 6979 | 148.41 | -55.159 | | |
| | NEURON matched | 491 | 852.4 | 6979 | 148.41 | -55.159 | 1.8e-11 | 5.1e-11 |
| | NEURON import3d | 108 | 852.4 | 5919 | 174.86 | -52.514 | **-15.1%** | 2.44 |
| Purkinje-slice-ageP43-6 | openbrain | 4156 | 1104.3 | 27995 | 51.733 | -64.827 | | |
| | NEURON matched | 4156 | 1104.3 | 27995 | 51.733 | -64.827 | 5.9e-12 | 1.7e-11 |
| | NEURON import3d | 764 | 1104.3 | 27770 | 49.692 | -65.031 | +4.1% | 0.21 |

"Soma area" for openbrain is the root compartment only; see the discrepancies section for the
extra soma membrane it adds.

### Speed (median of 5, seconds)

| Cell | openbrain steady solve | openbrain 100 ms transient | NEURON matched transient | NEURON import3d transient | NEURON matched Impedance | NEURON import3d Impedance |
|---|---|---|---|---|---|---|
| n419 | 5.0e-5 | 0.222 | 0.236 | 0.046 | 0.252 | 0.0066 |
| Ctrl-Cell3-40x | 5.1e-6 | 0.023 | 0.029 | 0.0078 | 0.0040 | 0.00033 |
| Purkinje | 4.0e-5 | 0.182 | 0.228 | 0.048 | 0.244 | 0.012 |

Model construction (median for openbrain, single run for NEURON): openbrain parse + build
1.3–1.7 ms for the large cells; NEURON matched 25–27 ms, import3d 77–93 ms.

Reading:

* Solver accuracy: on the same linear system openbrain and NEURON agree to 1e-11 mV over the whole
  transient and to 1e-11 relative in input resistance. The Hines solver and backward Euler step are
  correct.
* Speed on the same system: openbrain's transient is 6–20% faster than NEURON on the matched
  model. That model is unfavourable to NEURON (see discrepancies: it carries an extra zero-area
  node per section), so the honest summary is "about the same speed per compartment".
* Speed in practice: NEURON on its own d_lambda model is 3–5x faster than openbrain, because it
  uses 4.5–5.5x fewer compartments. openbrain has no way to coarsen the discretisation: it always
  uses one compartment per SWC point.
* Geometry: openbrain's input resistance differs from NEURON's standard model by +1.6%, +4.1% and
  -15.1%. The large miss on the L5 cell comes from soma handling (below), not the solver.

## Discrepancies and caveats

1. **openbrain double-counts soma membrane on NeuroMorpho CNG files.** The SWC parser makes the root
   point a cylinder with L = diam (sphere-equivalent area), then turns every other soma point
   (type 1) into a further cylinder. CNG files use a 3-point soma, so two extra cylinders of
   length r and diameter 2r are added, which together equal the whole soma area again: +852 µm² on
   Ctrl-Cell3-40x (soma area x2) and +1104 µm² on the Purkinje cell. n419 has a 14-point soma
   contour; openbrain treats its 13 non-root points as 642 µm² of extra cylinders, while NEURON's
   Import3d turns the contour into a 135 µm² soma. This is the main source of the R_in differences
   versus the import3d model.
2. **Cylinders versus frusta.** openbrain uses the child point's diameter over each segment;
   NEURON's Import3d keeps the 3D points and integrates frusta. Total membrane area differs by
   -1%, +18% and +1% (openbrain vs import3d), mostly from item 1. openbrain also makes cylinders
   for dendrite segments that start at the soma centre, so part of each sits inside the soma.
3. **Branch-point resistance.** In openbrain each child is joined to its parent's centre by
   `half(child) + half(parent)`. A standard NEURON model joins children to the parent's 1-end, so
   siblings share the parent half-resistance. The matched model reproduces openbrain's topology
   exactly by attaching at x = 0.5 and rescaling Ra per section; it is therefore not a "natural"
   NEURON model, and every section also carries an unused zero-area 1-end node, adding one
   node per section to the system NEURON solves. That likely inflates NEURON's matched-model timings.
4. **Discretisation.** openbrain cannot refine or coarsen compartments; NEURON's d_lambda rule
   picked 108–780 segments where openbrain uses 491–4229. Some long SWC segments may also be
   under-resolved in openbrain (no check against lambda).
5. **HH integrators differ.** openbrain uses Rush-Larsen for the gates and an exponential update
   for V; NEURON uses cnexp gates staggered half a step from V, with backward Euler or
   Crank-Nicolson for V. Both converge to the same CVODE reference. NEURON's default rate table
   (`usetable_hh = 1`) happens to be closer to the reference than exact rates under backward Euler
   because its error partly cancels the integrator's.
6. **What the HH timing measures.** One compartment is almost pure per-step overhead. The 4–4.5x
   speedup should not be read as a statement about multicompartment or network performance.
7. **Timing noise.** Shared desktop with ~1.3 GiB free RAM and frequency scaling enabled; single
   process, no CPU pinning. The steady-state rows compare a direct solve with NEURON's `Impedance`
   class, which does more work.
8. **Not tested:** active channels on morphologies, CVODE performance, multithreading, networks. Also,
   the Ctrl-Cell3-40x reconstruction itself is listed as "Dendrites Moderate" (incomplete) on
   NeuroMorpho.
