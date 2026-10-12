# Research plan: openbrain

## Vision

A differentiable neuron simulator, with a Rust core and a Python API, for asking **what neurons compute
and how disease, drugs and gases change it**, and for plugging into the software neuroscientists
already use.

The flagship question: **why are neurons shaped the way they are?** Design optimal neurons by gradient
descent, map the frontier of computing power against biological cost, then place thousands of real
neurons on it: healthy, diseased, drugged, and starved of oxygen.

## Architecture

```
            ┌──────────────── Python API · CLI · notebooks ────────────────┐
 Interop    │ SWC · NeuroML import/export · NWB recordings · SONATA ·      │
            │ NEURON cross-validation · NeuroMorpho / Allen fetchers       │
            ├──────────────────────────────────────────────────────────────┤
 Studies    │ Capacity (TCS) · Inverse design · Atlas · Disease reserve    │
            ├──────────────────────────────────────────────────────────────┤
 Perturb    │ morphology operators (pruning, spine loss, atrophy)          │
            │ channelopathies (e.g. SCN1A, KCNQ2) · drugs · gases          │
            ├──────────────────────────────────────────────────────────────┤
 Rust core  │ cable solver (Hines) · discrete adjoint · ion channels ·     │
            │ synapses (AMPA/NMDA/GABA) · reaction-diffusion chemistry     │
            └──────────────────────────────────────────────────────────────┘
```

### Rust core
- Multicompartment cable equation solved with the Hines method, plus its discrete adjoint for exact
  gradients (see `literature.md`, gap 4).
- Pluggable ion channels and synapses. Every parameter can be differentiated.
- **Chemistry engine (reaction-diffusion):** tracks ion concentrations (Na⁺, K⁺, Ca²⁺, Cl⁻) per
  compartment, so reversal potentials follow Nernst. It covers Ca²⁺ buffering, the ATP-dependent Na⁺/K⁺
  pump and its energy cost, and user-defined reaction networks (mass-action kinetics) that diffuse along
  dendrites.

### Gases (through mechanisms with literature parameters)
| Gas | Mechanism | Lets you simulate |
| --- | --- | --- |
| O₂ | supply → ATP production → Na⁺/K⁺ pump | hypoxia, ischemic stroke, anoxic depolarisation |
| CO₂ | pH → proton modulation of NMDA receptors and channels | hypercapnia, acidosis |
| Xenon, N₂O | NMDA receptor block | anaesthesia, neuroprotection |
| Isoflurane, sevoflurane | GABA_A potentiation, K2P leak activation | general anaesthesia |
| NO | diffusible messenger produced downstream of NMDA Ca²⁺ | retrograde signalling |

### Disease and pharmacology library
- **Morphology:** dendritic pruning, spine loss, atrophy. Real disease-model cells come from NeuroMorpho.
  The API's `experiment_condition` field already lists 5xFAD and CK-p25 Alzheimer models, Huntington's
  disease, traumatic brain injury, CTE, chronic stress, and drug exposures (checked 2026-10-12).
- **Channelopathies:** conductance and kinetic shifts from the literature (e.g. SCN1A loss-of-function in
  interneurons, as in Dravet syndrome).
- **Drugs:** receptor and channel blocks as parameter overrides (TTX, 4-AP, memantine, ketamine), as the
  HH API already does with `HHNeuron(g_na=0)`.

### Interop
SWC and NeuroML in and out, NWB for loading experimental recordings, SONATA for networks, and the
same cell runnable in NEURON for cross-validation. Researchers bring their existing cells and data, and
can take results back to their own tools.

## Research questions

- **Q1. Frontier (flagship).** Inverse-design dendrites for maximum Task Capacity Score (TCS) under
  membrane-area (energy) and wiring-length costs. Do real neurons sit on the compute-cost frontier?
  This extends Stiefel & Sejnowski 2007 (optimisation over 2 simple tasks) to gradient-based design
  checked against thousands of real cells.
- **Q2. Computational reserve in disease.** How much capacity do Alzheimer's-model, Huntington's, TBI and
  CTE neurons lose compared with controls from the same lab and region? Does topology predict
  resilience? This extends Kirch & Gollo 2021, who measured firing rate and dynamic range under pruning,
  not computational capacity.
- **Q3. Gases and drugs.** Do anaesthetic gases reduce single-neuron computational capacity, and in what
  order (xenon vs isoflurane)? How fast does capacity collapse under hypoxia, and do some cell classes
  hold out longer?
- **Q4. Validity (gates everything).** TCS agrees with FCI rankings (Aizenbud 2025); NMDA removal drops
  TCS; the chemistry reproduces anoxic depolarisation timing measured in brain slices.

## Milestones (each gates the next)

1. **Pilot.** Cable solver + adjoint (matches finite differences to 1e-6); one real neuron learns XOR in
   under 1 CPU-minute. *Passed 2026-10-12:* CA1 cell n419 (4,229 compartments, 96 synapses) solves XOR
   on seeds 1–6 in 34–98 Adam steps, 9–23 CPU-s each (`cargo run --release --example xor`).
2. **Perturbation API + interop v1.** SWC, NeuroML import, NeuroMorpho fetcher (with disease metadata),
   drug and channelopathy overrides on multicompartment cells.
3. **Chemistry core.** Ion concentrations, Nernst, Na⁺/K⁺ pump with ATP, O₂ supply. Validate against
   anoxic depolarisation onset in slices and against NEURON's RxD on shared test cases.
4. **Capacity validation (Q4).**
5. **Inverse design (Q1).**
6. **Atlas + disease reserve (Q2).** 1,000+ cells, local cores or cloud.
7. **Gases study (Q3).**
8. **Release.** Docs site, dataset, interactive 3D atlas, write-up.

## Honest limits

- The simulator turns known mechanisms into predictions. It cannot discover new pharmacology; its gas and
  drug results are only as good as the published parameters behind them, and it must cite them.
- Single-cell results do not equal brain-level effects (consciousness, cognition). The write-up must
  not overclaim.
- Scope is large. Milestones 1–4 are a strong project on their own; each later milestone adds a result,
  not a requirement.
- Novelty rests on targeted searches. Do a deeper check before making public claims.

## Compute

Target: at most 1 CPU-hour per cell for the full task ladder. Local 16 cores for development; remote
Claude Code sessions for tests and small sweeps (measure their resources first); spot CPU instances
for the atlas if needed. No GPU dependency.
