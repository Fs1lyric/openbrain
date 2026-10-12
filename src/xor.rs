//! XOR as a test of single-neuron computation: two inputs, A and B, each drive AMPA, NMDA and
//! GABA_A synapses at a shared set of sites. Training adjusts every synaptic weight so that the mean
//! soma voltage is higher for exactly one active input (10, 01) than for none or both (00, 11).

use crate::cable::{Cable, Geometry};
use crate::learn::{Problem, Synapse, SynapseKind, double_exponential};

pub const DT: f64 = 0.1;
pub const STEPS: usize = 600; // 60 ms
/// Steps whose soma voltage is scored (10-60 ms).
pub const WINDOW: std::ops::Range<usize> = 100..600;
const BURST: [usize; 3] = [100, 120, 140];
const KINDS: [SynapseKind; 3] = [SynapseKind::Ampa, SynapseKind::Nmda, SynapseKind::GabaA];
/// Inputs (A, B) of each simulated pattern; 00 has no input and sits exactly at rest.
pub const PATTERNS: [(bool, bool); 3] = [(true, false), (false, true), (true, true)];

#[derive(Clone, Debug)]
pub struct Config {
    /// mV between the lowest XOR-true and highest XOR-false pattern that counts as solved.
    pub margin: f64,
    /// The loss keeps pushing past `margin` so the solution is not knife-edge.
    pub target_gap: f64,
    /// 1/mV, of the softplus pair loss.
    pub sharpness: f64,
    pub max_iters: usize,
    /// Initial weight, µS; weights are `w0 * exp(theta)` so they stay positive.
    pub w0: f64,
    pub learning_rate: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            margin: 1.0,
            target_gap: 3.0,
            sharpness: 2.0,
            max_iters: 1500,
            w0: 0.001,
            learning_rate: 0.1,
        }
    }
}

pub struct Progress<'a> {
    pub iter: usize,
    pub loss: f64,
    /// Mean soma mV over `WINDOW` for patterns [00 10 01 11].
    pub means: &'a [f64; 4],
    pub gap: f64,
}

pub struct Training {
    pub solved_at: Option<usize>,
    pub iterations: usize,
    pub best_gap: f64,
    /// Means [00 10 01 11] at the last iteration.
    pub means: [f64; 4],
    pub weights: Vec<f64>,
}

/// One compartment with the same total membrane area as `geom`: the point-neuron control.
pub fn point_geometry(geom: &Geometry) -> Geometry {
    let area: f64 = geom
        .length_um
        .iter()
        .zip(&geom.diameter_um)
        .map(|(l, d)| l * d)
        .sum();
    // a cylinder with length equal to its diameter has lateral area pi d^2
    let d = area.sqrt();
    Geometry {
        parent: vec![None],
        length_um: vec![d],
        diameter_um: vec![d],
    }
}

/// `count` compartments drawn (with replacement) from `first..n`, reproducibly from `seed`.
pub fn random_sites(n: usize, first: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            first + (state % (n - first) as u64) as usize
        })
        .collect()
}

/// Every input gets every synapse type at every site.
pub fn synapses(sites: &[usize]) -> (Vec<Synapse>, Vec<usize>) {
    let mut synapses = Vec::new();
    let mut owner = Vec::new();
    for input in 0..2 {
        for &compartment in sites {
            for kind in KINDS {
                synapses.push(Synapse { compartment, kind });
                owner.push(input);
            }
        }
    }
    (synapses, owner)
}

fn waveform(kind: SynapseKind) -> Vec<f64> {
    match kind {
        SynapseKind::Ampa => double_exponential(&BURST, 0.2, 2.0, DT, STEPS),
        SynapseKind::Nmda => double_exponential(&BURST, 2.0, 50.0, DT, STEPS),
        SynapseKind::GabaA => double_exponential(&BURST, 0.5, 8.0, DT, STEPS),
        SynapseKind::Current => double_exponential(&BURST, 1.0, 5.0, DT, STEPS),
    }
}

/// The three non-trivial patterns as simulation problems, in `PATTERNS` order.
pub fn problems<'a>(cable: &'a Cable, sites: &[usize]) -> Vec<Problem<'a>> {
    let (synapses, owner) = synapses(sites);
    PATTERNS
        .iter()
        .map(|&(a, b)| {
            let drive = synapses
                .iter()
                .zip(&owner)
                .map(|(s, &o)| {
                    if [a, b][o] {
                        waveform(s.kind)
                    } else {
                        vec![0.0; STEPS]
                    }
                })
                .collect();
            Problem::new(cable, synapses.clone(), drive, DT).expect("valid sites")
        })
        .collect()
}

/// Train the weights with Adam on a softplus pair loss over the patterns' mean soma voltages.
pub fn train(
    cable: &Cable,
    sites: &[usize],
    config: &Config,
    mut on_progress: impl FnMut(&Progress),
) -> Training {
    let problems = problems(cable, sites);
    let n_weights = 2 * sites.len() * KINDS.len();
    let mut theta = vec![0.0f64; n_weights];
    let (mut m, mut v) = (vec![0.0; n_weights], vec![0.0; n_weights]);
    let (b1, b2) = (0.9, 0.999);
    let scale = 1.0 / WINDOW.len() as f64;
    let mut best_gap = f64::NEG_INFINITY;
    let mut means = [cable.e_leak; 4];
    let mut w = vec![config.w0; n_weights];
    for iter in 1..=config.max_iters {
        w = theta.iter().map(|t| config.w0 * t.exp()).collect();
        // each evaluation's gradient is d(mean soma voltage over WINDOW)/dw
        let evals: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = problems
                .iter()
                .map(|problem| {
                    let w = &w;
                    scope.spawn(move || {
                        problem.evaluate(w, |k, _| {
                            if WINDOW.contains(&k) {
                                (0.0, scale)
                            } else {
                                (0.0, 0.0)
                            }
                        })
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (mean, e) in means[1..].iter_mut().zip(&evals) {
            *mean = e.soma[WINDOW].iter().sum::<f64>() * scale;
        }
        // softplus pair loss on every (XOR-true, XOR-false) pair
        let mut loss = 0.0;
        let mut d_means = [0.0; 4];
        for t in [1, 2] {
            for f in [0, 3] {
                let z = config.sharpness * (config.target_gap - (means[t] - means[f]));
                loss += z.max(0.0) + (-z.abs()).exp().ln_1p();
                let sigmoid = 1.0 / (1.0 + (-z).exp());
                d_means[t] -= sigmoid * config.sharpness;
                d_means[f] += sigmoid * config.sharpness;
            }
        }
        let gap = means[1].min(means[2]) - means[0].max(means[3]);
        best_gap = best_gap.max(gap);
        on_progress(&Progress {
            iter,
            loss,
            means: &means,
            gap,
        });
        if gap >= config.margin {
            return Training {
                solved_at: Some(iter),
                iterations: iter,
                best_gap,
                means,
                weights: w,
            };
        }
        let mut grad = vec![0.0; n_weights];
        for (e, d) in evals.iter().zip(&d_means[1..]) {
            grad.iter_mut().zip(&e.grad).for_each(|(a, b)| *a += d * b);
        }
        let t = iter as i32;
        for i in 0..n_weights {
            let g = grad[i] * w[i]; // chain rule through exp
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let (mh, vh) = (m[i] / (1.0 - b1.powi(t)), v[i] / (1.0 - b2.powi(t)));
            theta[i] -= config.learning_rate * mh / (vh.sqrt() + 1e-12);
        }
    }
    Training {
        solved_at: None,
        iterations: config.max_iters,
        best_gap,
        means,
        weights: w,
    }
}
