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
const ON_TARGET: f64 = -60.0;
const MARGIN: f64 = 1.0; // mV between the lowest XOR-true and highest XOR-false pattern

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
    let patterns = [(false, false), (true, false), (false, true), (true, true)];
    let problems: Vec<(Problem, f64)> = patterns
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
            let target = if a ^ b { ON_TARGET } else { passive.e_leak };
            (
                Problem::new(&cable, synapses.clone(), drive, DT).unwrap(),
                target,
            )
        })
        .collect();

    // weights w = w0 * exp(theta) stay positive; Adam on theta
    let w0 = 0.001;
    let mut theta = vec![0.0f64; synapses.len()];
    let (mut m, mut v) = (vec![0.0; theta.len()], vec![0.0; theta.len()]);
    let (lr, b1, b2) = (0.1, 0.9, 0.999);
    let start = Instant::now();
    println!("{} compartments, {} synapses", cable.len(), synapses.len());
    for iter in 1..=1500 {
        let w: Vec<f64> = theta.iter().map(|t| w0 * t.exp()).collect();
        let scale = 1.0 / WINDOW.len() as f64;
        let evals: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = problems
                .iter()
                .map(|(problem, target)| {
                    let w = &w;
                    scope.spawn(move || {
                        problem.evaluate(w, |k, vs| {
                            if WINDOW.contains(&k) {
                                let e = vs - target;
                                (scale * e * e, scale * 2.0 * e)
                            } else {
                                (0.0, 0.0)
                            }
                        })
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let loss: f64 = evals.iter().map(|e| e.loss).sum();
        let mut grad = vec![0.0; w.len()];
        for e in &evals {
            grad.iter_mut().zip(&e.grad).for_each(|(a, b)| *a += b);
        }
        let means: Vec<f64> = evals
            .iter()
            .map(|e| e.soma[WINDOW].iter().sum::<f64>() / WINDOW.len() as f64)
            .collect();
        let lowest_true = means[1].min(means[2]);
        let highest_false = means[0].max(means[3]);
        let gap = lowest_true - highest_false;
        let correct = if gap >= MARGIN { 4 } else { 3 };
        if iter % 250 == 0 || correct == 4 {
            println!(
                "iter {iter:3}  loss {loss:8.3}  mean soma mV [00 10 01 11] = {:?}  gap {gap:5.2} mV  {:.1}s",
                means
                    .iter()
                    .map(|x| (x * 10.0).round() / 10.0)
                    .collect::<Vec<_>>(),
                start.elapsed().as_secs_f64()
            );
        }
        if correct == 4 {
            println!(
                "SOLVED XOR in {iter} iterations, {:.1}s",
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
        "NOT solved after 1500 iterations, {:.1}s",
        start.elapsed().as_secs_f64()
    );
}
