//! Benchmark workload, shared with `bench/compare.py`: a passive cell driven by current pulses at
//! 16 dendritic sites. Times one forward simulation and one forward+adjoint gradient of the mean
//! soma voltage with respect to the 16 input amplitudes, and writes the geometry, inputs, soma
//! trace, gradient and timings as JSON so NEURON and Jaxley can run the identical problem.
//!
//! Usage: cargo run --release --example bench -- neuron.swc out.json

use std::time::Instant;

use openbrain_core::cable::{Cable, Passive};
use openbrain_core::learn::{Problem, Synapse, SynapseKind, double_exponential};
use openbrain_core::swc;
use openbrain_core::xor;

const PASSIVE: Passive = Passive {
    cm: 1.0,
    g_leak: 1e-4,
    e_leak: -70.0,
    ri: 150.0,
};
const DT: f64 = 0.1;
const STEPS: usize = 600;
const WINDOW: std::ops::Range<usize> = 100..600;
const SITES: usize = 16;
const SOMA_POINTS: usize = 14;
const TAU_RISE: f64 = 1.0;
const TAU_DECAY: f64 = 5.0;
const REPEATS: usize = 21;

fn median_seconds(mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[REPEATS / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, out) = (&args[1], &args[2]);
    let geom = swc::parse(&std::fs::read_to_string(path).expect("read swc")).expect("parse swc");
    let cable = Cable::new(&geom, PASSIVE).expect("cable");
    let sites = xor::random_sites(cable.len(), SOMA_POINTS, SITES, 1);
    // site k receives one pulse at 10 + 2k ms
    let onsets: Vec<usize> = (0..SITES).map(|k| 100 + 20 * k).collect();
    let synapses: Vec<Synapse> = sites
        .iter()
        .map(|&compartment| Synapse {
            compartment,
            kind: SynapseKind::Current,
        })
        .collect();
    let drive = onsets
        .iter()
        .map(|&k| double_exponential(&[k], TAU_RISE, TAU_DECAY, DT, STEPS))
        .collect();
    let problem = Problem::new(&cable, synapses, drive, DT).expect("problem");
    let amplitudes = vec![0.02; SITES]; // nA
    let scale = 1.0 / WINDOW.len() as f64;
    let mean_loss = |k: usize, v: f64| {
        if WINDOW.contains(&k) {
            (scale * v, scale)
        } else {
            (0.0, 0.0)
        }
    };

    let soma = problem.soma_trace(&amplitudes);
    let (loss, grad) = problem.loss_and_grad(&amplitudes, mean_loss);
    let t_forward = median_seconds(|| {
        std::hint::black_box(problem.soma_trace(&amplitudes));
    });
    let t_grad = median_seconds(|| {
        std::hint::black_box(problem.loss_and_grad(&amplitudes, mean_loss));
    });
    println!(
        "{} compartments: forward {:.1} ms, forward+gradient {:.1} ms, mean soma {loss:.4} mV",
        cable.len(),
        t_forward * 1e3,
        t_grad * 1e3
    );
    let json = format!(
        concat!(
            "{{\"parent\":{:?},\"length_um\":{:?},\"diameter_um\":{:?},",
            "\"cm\":{},\"g_leak\":{},\"e_leak\":{},\"ri\":{},",
            "\"dt\":{},\"steps\":{},\"window\":[{},{}],\"tau_rise\":{},\"tau_decay\":{},",
            "\"sites\":{:?},\"onsets_ms\":{:?},\"amplitudes\":{:?},",
            "\"soma\":{:?},\"loss\":{},\"grad\":{:?},\"t_forward\":{},\"t_grad\":{}}}"
        ),
        geom.parent
            .iter()
            .map(|p| p.map_or(-1, |p| p as i64))
            .collect::<Vec<_>>(),
        geom.length_um,
        geom.diameter_um,
        PASSIVE.cm,
        PASSIVE.g_leak,
        PASSIVE.e_leak,
        PASSIVE.ri,
        DT,
        STEPS,
        WINDOW.start,
        WINDOW.end,
        TAU_RISE,
        TAU_DECAY,
        sites,
        onsets.iter().map(|&k| k as f64 * DT).collect::<Vec<_>>(),
        amplitudes,
        soma,
        loss,
        grad,
        t_forward,
        t_grad
    );
    std::fs::write(out, json).expect("write json");
}
