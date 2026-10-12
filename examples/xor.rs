//! Pilot: can a real reconstructed neuron learn XOR by gradient descent on its synaptic weights?
//!
//! Usage: cargo run --release --example xor -- path/to/neuron.swc [seed]

use std::time::Instant;

use openbrain_core::cable::{Cable, Passive};
use openbrain_core::learn::{Problem, Synapse, SynapseKind, double_exponential};
use openbrain_core::swc;

const DT: f64 = 0.1;
const STEPS: usize = 600; // 60 ms
const WINDOW: std::ops::Range<usize> = 100..600; // score 10-60 ms
const BURST: [usize; 3] = [100, 120, 140];
const SITES_PER_INPUT: usize = 8;
const KINDS: [SynapseKind; 3] = [SynapseKind::Ampa, SynapseKind::Nmda, SynapseKind::GabaA];
const MARGIN: f64 = 1.0; // mV between the lowest XOR-true and highest XOR-false pattern
const TARGET_GAP: f64 = 3.0; // the loss keeps pushing past MARGIN so the solution is not knife-edge
const SHARPNESS: f64 = 2.0; // 1/mV, of the softplus pair loss
const MAX_ITERS: i32 = 1500;

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: xor <neuron.swc> [seed]");
    let seed: u64 = args.get(2).map_or(1, |s| s.parse().expect("seed"));
    let geom = swc::parse(&std::fs::read_to_string(path).expect("read swc")).expect("parse swc");
    let passive = Passive {
        cm: 1.0,
        g_leak: 1e-4,
        e_leak: -70.0,
        ri: 150.0,
    };
    let cable = Cable::new(&geom, passive).expect("cable");
    let soma_points = geom.diameter_um.iter().take_while(|_| true).count().min(14);

    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut synapses = Vec::new();
    let mut owner = Vec::new(); // which input (0 = A, 1 = B) drives each synapse
    // a shared pool of dendritic sites; every input may use every synapse type at every site
    let sites: Vec<usize> = (0..2 * SITES_PER_INPUT)
        .map(|_| soma_points + rng.below(cable.len() - soma_points))
        .collect();
    for input in 0..2 {
        for &site in &sites {
            for kind in KINDS {
                synapses.push(Synapse {
                    compartment: site,
                    kind,
                });
                owner.push(input);
            }
        }
    }
    let waveform = |kind: SynapseKind| match kind {
        SynapseKind::Ampa => double_exponential(&BURST, 0.2, 2.0, DT, STEPS),
        SynapseKind::Nmda => double_exponential(&BURST, 2.0, 50.0, DT, STEPS),
        SynapseKind::GabaA => double_exponential(&BURST, 0.5, 8.0, DT, STEPS),
    };
    // 00 has no input, so a passive cell sits exactly at rest and needs no simulation
    let patterns = [(true, false), (false, true), (true, true)];
    let problems: Vec<Problem> = patterns
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
            Problem::new(&cable, synapses.clone(), drive, DT).unwrap()
        })
        .collect();

    // weights w = w0 * exp(theta) stay positive; Adam on theta
    let w0 = 0.001;
    let mut theta = vec![0.0f64; synapses.len()];
    let (mut m, mut v) = (vec![0.0; theta.len()], vec![0.0; theta.len()]);
    let (lr, b1, b2) = (0.1, 0.9, 0.999);
    let start = Instant::now();
    println!("{} compartments, {} synapses", cable.len(), synapses.len());
    for iter in 1..=MAX_ITERS {
        let w: Vec<f64> = theta.iter().map(|t| w0 * t.exp()).collect();
        let scale = 1.0 / WINDOW.len() as f64;
        // each evaluation's "loss" is the pattern's mean soma voltage over WINDOW, and its gradient
        // is d(mean)/dw; the classification loss below is a function of those means alone
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
        let mean = |e: &openbrain_core::learn::Evaluation| {
            e.soma[WINDOW].iter().sum::<f64>() / WINDOW.len() as f64
        };
        // means in order [00 10 01 11]
        let means = [
            passive.e_leak,
            mean(&evals[0]),
            mean(&evals[1]),
            mean(&evals[2]),
        ];
        // softplus pair loss on every (XOR-true, XOR-false) pair: penalise gaps below TARGET_GAP
        let mut loss = 0.0;
        let mut d_means = [0.0; 4];
        for t in [1, 2] {
            for f in [0, 3] {
                let z = SHARPNESS * (TARGET_GAP - (means[t] - means[f]));
                loss += z.max(0.0) + (-z.abs()).exp().ln_1p();
                let sigmoid = 1.0 / (1.0 + (-z).exp());
                d_means[t] -= sigmoid * SHARPNESS;
                d_means[f] += sigmoid * SHARPNESS;
            }
        }
        let mut grad = vec![0.0; w.len()];
        for (e, d) in evals.iter().zip(&d_means[1..]) {
            grad.iter_mut().zip(&e.grad).for_each(|(a, b)| *a += d * b);
        }
        let lowest_true = means[1].min(means[2]);
        let highest_false = means[0].max(means[3]);
        let gap = lowest_true - highest_false;
        let solved = gap >= MARGIN;
        if iter % 25 == 0 || solved {
            println!(
                "iter {iter:4}  loss {loss:7.3}  mean soma mV [00 10 01 11] = {:?}  gap {gap:5.2} mV  {:.1}s",
                means
                    .iter()
                    .map(|x| (x * 10.0).round() / 10.0)
                    .collect::<Vec<_>>(),
                start.elapsed().as_secs_f64()
            );
        }
        if solved {
            println!(
                "SOLVED XOR in {iter} iterations, {:.1}s wall",
                start.elapsed().as_secs_f64()
            );
            return;
        }
        for i in 0..theta.len() {
            let g = grad[i] * w[i]; // chain rule through exp
            m[i] = b1 * m[i] + (1.0 - b1) * g;
            v[i] = b2 * v[i] + (1.0 - b2) * g * g;
            let (mh, vh) = (m[i] / (1.0 - b1.powi(iter)), v[i] / (1.0 - b2.powi(iter)));
            theta[i] -= lr * mh / (vh.sqrt() + 1e-12);
        }
    }
    println!(
        "NOT solved after {MAX_ITERS} iterations, {:.1}s wall",
        start.elapsed().as_secs_f64()
    );
}
