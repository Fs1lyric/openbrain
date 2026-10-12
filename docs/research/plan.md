# Research plan: the Neuron Compute Atlas

## Question

Across thousands of real reconstructed neurons, spanning species, brain regions and cell classes, what
determines how much a single neuron can compute? Specifically: once dendritic size is accounted for,
does the *structure* of the tree still predict computational capacity?

## Contribution

1. **A cheap, simulator-native capacity measure: the Task Capacity Score (TCS).** Train the neuron's own
   synaptic weights, through the biophysical simulation, on a ladder of tasks of increasing difficulty,
   and report the hardest level it reliably learns. No surrogate DNN is trained.
2. **A Hines-structured discrete adjoint in Rust.** Exact gradients of a loss on the somatic voltage with
   respect to every synaptic weight, for about the cost of one extra simulation, on CPU.
3. **The atlas.** TCS for thousands of NeuroMorpho / Allen cells, released as an open dataset with an
   interactive 3D web viewer.
4. **The answer** to the size-versus-structure question, written up arXiv-style.

## Hypotheses (pre-registered here before any atlas data exists)

- **H1 (validity):** TCS ranks neurons like FCI does. On cells overlapping Aizenbud 2025 we expect human
  > rat, human L2/3 most complex, and Spearman ρ > 0.6 with published FCI where models can be matched.
- **H2 (ablation):** removing NMDA receptors lowers TCS sharply for every cell class.
- **H3 (replication at scale):** dendritic surface area is the strongest single predictor of TCS across
  all classes.
- **H4 (new):** after controlling for surface area, topological features (branch-order distribution,
  topological morphology descriptor entropy, electrotonic compartmentalisation) explain significant
  residual variance, and cell classes (pyramidal, Purkinje, interneuron, granule) differ in capacity
  per µm² of membrane.

A negative result on H4 is still a result: "size is all you need" across 1,000+ real cells would be news.

## Method sketch

- **Cells:** NeuroMorpho SWC files, filtered for complete dendrites and quality; Allen Cell Types for
  human/mouse cells with matched data.
- **Biophysics (uniform across cells, as in FCI):** passive dendrites with fixed specific membrane
  properties, AMPA + NMDA (Mg²⁺ block) synapses, Hodgkin–Huxley soma. Active dendritic channels are a
  later ablation.
- **Inputs:** k input channels, each a fixed random cluster of synapses on the tree. A task input is a
  pattern of channels firing brief bursts.
- **Task ladder:** random Boolean functions of dimension n = 2…12 (following Agrawal 2025's phase
  transition), n-bit parity, and temporal-order tasks.
- **Training:** gradient descent on synaptic weights via the adjoint; loss on somatic spike or peak
  depolarisation; several random seeds.
- **Score:** TCS = the critical dimension n* where P(learned) crosses 0.5.

## Milestones (each one gates the next)

1. **Pilot (go/no-go).** Cable solver + adjoint verified against finite differences; one real NeuroMorpho
   neuron learns XOR. *Go* if one cell × one task trains in under 1 CPU-minute and the gradient matches
   finite differences to 1e-6 relative error.
2. **Validation.** H1 and H2 on a small cell set, plus checks against analytical cable theory and against
   NEURON.
3. **Budget check.** Measure CPU-hours per cell for the full ladder; extrapolate to 1,000 cells. Run locally
   (16 cores) if feasible, otherwise in the cloud (see below).
4. **Atlas run.** 1,000+ cells, H3 and H4 analysis.
5. **Release.** Dataset, web atlas, write-up.

## Compute

Target: at most 1 CPU-hour per cell, so 1,000 cells ≈ 1,000 CPU-hours (about 3 days on 16 local
cores). Remote Claude Code sessions in Anthropic's cloud can run tests and smaller sweeps; first measure
their cores and RAM. If the full atlas outgrows both, use spot CPU instances. The design is CPU-parallel
across cells, with no GPU dependency.

## Risks

- **TCS and FCI measure different things** (learnability vs I/O mimicry). If H1 fails, report it, and
  treat TCS as a complementary measure rather than a replacement.
- **Uniform biophysics across cells is a simplification** (FCI makes the same choice). Cell-specific
  channel models from the Allen database are a follow-up.
- **Reconstruction quality varies across labs**, so filter for quality and control for source lab in
  the analysis.
- **Novelty is based on a targeted search.** Do a deeper check before any public claims.
