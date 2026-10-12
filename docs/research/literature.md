# Literature review: how much can a single neuron compute?

Reviewed 2026-10-12. This is a targeted search (Consensus, web, and full text of the two closest papers),
not a systematic review. Re-check before making public originality claims.

## Closest prior work

| Work | Neurons | Model | Capacity measure | Cost |
| --- | --- | --- | --- | --- |
| Beniaguev, Segev & London 2021, *Neuron* | 1 rat L5 pyramidal (plus ablations) | NEURON, full biophysics | depth of a temporal conv net that mimics the cell's I/O; needed 5–8 layers | very high (GPU training per cell) |
| Aizenbud, Yoeli, …, Segev 2025, bioRxiv 2024.12.17.628883 / *PNAS* | 24: 12 human + 12 rat pyramidal, L2/3–L6 | NEURON, passive dendrites + AMPA/NMDA synapses | Functional Complexity Index (FCI): how badly a fixed 3-layer TCN mimics the cell | **~2.3 single-GPU years** for 24 cells, plus a CPU cluster for 12,000 × 10 s simulations per cell |
| Agrawal & Buice 2025, *NeurIPS* | 2 synthetic 7-branch trees ("basal" vs "apical") | abstract sigmoid subunits, no biophysics | Boolean function learnability; phase transition at a critical input dimension | low |
| Aizenbud et al. 2026, bioRxiv 2026.06.08.730984 ("What can a neuron compute", TwinProp) | 1 rat L5 pyramidal | NEURON model trained through a DNN digital twin | task accuracy: images, audio, XOR, 10-bit parity, random Boolean | high (needs the DNN twin first) |
| Masoli et al. 2024, *Commun. Biol.* | human vs mouse Purkinje | biophysical | number of storable input patterns (51 vs 8) | moderate |
| Deistler et al. 2025, *Nature Methods* (Jaxley) | tooling | differentiable multicompartment simulator, JAX/GPU | n/a: trains biophysical models by gradient descent | GPU |
| Jones & Kording 2024, arXiv 2407.04025 | tooling | PyTorch ODE neuron models, adjoint via torchdiffeq | n/a: gradient fitting of dendritic channel densities | GPU |

## What is established

- One neuron is more than a perceptron: dendritic NMDA and voltage-gated nonlinearities let a single
  cell solve XOR, parity, and image tasks (Beniaguev 2021; Aizenbud 2026; Gidon et al. 2020, *Science*).
- Size dominates complexity. Total dendritic surface area is the strongest single predictor of FCI
  (Aizenbud 2025), and arbor size best predicts the critical Boolean dimension (Agrawal 2025).
- Removing NMDA receptors collapses complexity (Beniaguev 2021; Aizenbud 2025; Aizenbud 2026).
- Human pyramidal cells are more complex than rat ones (FCI 0.38 vs 0.22), and human L2/3 cells are the
  most complex, unlike in rat, where L5 cells are (Aizenbud 2025).

## The gap

1. **Scale and breadth.** The largest biophysical comparison is 24 cells, all cortical pyramidal.
   The FCI authors explicitly list as future work: hippocampal CA1/CA3, cerebellar Purkinje cells,
   other species (e.g. primates), and active dendritic channels. NeuroMorpho.Org holds about 150,000
   reconstructions across 78 species, 1,330 cell types and 381 brain regions (release 8.1).
2. **Cost of the measure.** FCI's own limitations section names "the computational cost of generating
   large I/O datasets … and training the respective neural networks" as the main challenge, and notes
   that FCI values are relative to the chosen DNN architecture.
3. **No cheap, simulator-native measure.** We found no 2025–2026 measure of single-neuron capacity that
   avoids training a surrogate DNN per cell.
4. **No Hines-structured adjoint.** Differentiable neuron work uses generic autodiff (JAX, PyTorch,
   torchdiffeq). We found no published derivation of a discrete adjoint that reuses the Hines tree
   solve (a backward sweep over the dendritic tree).
5. **The residual beyond size.** Both size findings come from either 24 pyramidal cells or 2 synthetic
   trees. Whether branching structure predicts capacity *after controlling for size*, across real
   cell classes, is open.

## Sources

- Beniaguev, Segev, London. Single cortical neurons as deep artificial neural networks. *Neuron* 2021. https://www.sciencedirect.com/science/article/pii/S0896627321005018
- Aizenbud, Yoeli, Beniaguev, de Kock, London, Segev. What makes human cortical pyramidal neurons functionally complex. bioRxiv 2024.12.17.628883. https://www.biorxiv.org/content/10.1101/2024.12.17.628883
- Published version: Dendritic morphology and synaptic nonlinearities enhance functional complexity in human cortical neurons. *PNAS*. https://www.pnas.org/doi/10.1073/pnas.2533168123
- Agrawal, Buice. Bounds on the computational complexity of neurons due to dendritic morphology. *NeurIPS* 2025. https://proceedings.neurips.cc/paper_files/paper/2025/file/d9e56606b1bd3095fbb4241c4e15e420-Paper-Conference.pdf
- Aizenbud et al. What can a neuron compute. bioRxiv 2026. https://sciety.org/articles/activity/10.64898/2026.06.08.730984
- Masoli et al. Human Purkinje cells outperform mouse Purkinje cells in dendritic complexity and computational capacity. *Commun. Biol.* 2024. https://consensus.app/papers/details/eda32b36ec695113947b48816abc7cb9/
- Deistler et al. Jaxley: differentiable simulation enables large-scale training of detailed biophysical models of neural dynamics. *Nature Methods* 2025. https://consensus.app/papers/details/02673c29f55052e7b18c162021970f00/
- Jones, Kording. Efficient optimization of ODE neuron models using gradient descent. arXiv 2407.04025. https://arxiv.org/abs/2407.04025
- NeuroMorpho.Org scale (release 8.1): https://www.biorxiv.org/content/10.1101/2021.12.17.473026v1.full
- Allen Cell Types Database (human + mouse morphologies, electrophysiology, biophysical models): https://brain-map.org/support/documentation/cell-types-database-physiology-and-morphology

## Added 2026-10-12: inverse design, disease, chemistry

- **Stiefel & Sejnowski 2007**, *J Neurophysiol* 98:513–526, "Mapping function onto neuronal
  morphology". Optimised dendritic structures (in NEURON, probably evolutionary) for two tasks: linear
  summation, and EPSP temporal order. The resulting shapes resembled nucleus laminaris and pyramidal
  cells. This is the closest precedent for inverse design. Small scale, no gradients, and no comparison
  against large real datasets. https://papers.cnl.salk.edu/PDFs/Mapping%20Function%20onto%20Neuronal%20Morphology%202007-3947.pdf
- **Kirch & Gollo 2021**, *Sci Rep*, "Single-neuron dynamical effects of dendritic pruning implicated in
  aging and neurodegeneration: towards a measure of neuronal reserve". Pruned NeuroMorpho cells; firing
  rate, energy efficiency and dynamic range drop; resilience depends on topology. Closest precedent for
  Q2, but it does not measure computational capacity. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7809359/
- **Dhupia et al. 2015**, *Front Cell Neurosci*: progressive pruning of CA1 models changes resonance
  and impedance maps. https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2014.00456/pdf
- **Cuntz group:** atrophy after denervation is partly homeostatic (granule cells retune excitability).
  This is a counterpoint to assuming atrophy is purely pathological.
- **NeuroMorpho API** (`/api/neuron/fields/experiment_condition`) exposes disease-model conditions: 5xFAD,
  CK-p25, human APP overexpression, Huntington's disease, TBI, CTE, chronic stress, drug exposures.
- **Chemistry precedents to validate against:** NEURON's RxD module, and STEPS (stochastic
  reaction-diffusion).
