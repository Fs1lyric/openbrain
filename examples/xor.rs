//! Pilot: can a real reconstructed neuron learn XOR by gradient descent on its synaptic weights?
//!
//! Usage: cargo run --release --example xor -- neuron.swc [--seed N] [--mode tree|point|soma]
//!        [--iters N] [--margin mV] [--target mV] [--out weights.json]
//!
//! `tree` places synapses on random dendritic sites. The controls keep the same synapses but put
//! them all on one compartment: `point` collapses the cell into a single compartment with the same
//! membrane area, `soma` keeps the real tree and moves every synapse onto the soma.

use std::time::Instant;

use openbrain_core::cable::{Cable, Passive};
use openbrain_core::swc;
use openbrain_core::xor::{self, Config};

const SITES: usize = 16;
/// The first SWC points of a NeuroMorpho cell describe the soma; dendritic sites come after them.
const SOMA_POINTS: usize = 14;
pub const PASSIVE: Passive = Passive {
    cm: 1.0,
    g_leak: 1e-4,
    e_leak: -70.0,
    ri: 150.0,
};

fn flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter().position(|a| a == name).map_or(default, |i| {
        args[i + 1]
            .parse()
            .ok()
            .unwrap_or_else(|| panic!("bad {name}"))
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: xor <neuron.swc> [flags]");
    let seed: u64 = flag(&args, "--seed", 1);
    let mode: String = flag(&args, "--mode", "tree".to_string());
    let defaults = Config::default();
    let config = Config {
        max_iters: flag(&args, "--iters", defaults.max_iters),
        margin: flag(&args, "--margin", defaults.margin),
        target_gap: flag(&args, "--target", defaults.target_gap),
        ..defaults
    };
    let out: Option<String> = args
        .iter()
        .position(|a| a == "--out")
        .map(|i| args[i + 1].clone());

    let geom = swc::parse(&std::fs::read_to_string(path).expect("read swc")).expect("parse swc");
    let tree = Cable::new(&geom, PASSIVE).expect("cable");
    let point = Cable::new(&xor::point_geometry(&geom), PASSIVE).expect("cable");
    let dendritic = xor::random_sites(tree.len(), SOMA_POINTS, SITES, seed);
    let (cable, sites) = match mode.as_str() {
        "tree" => (&tree, dendritic.clone()),
        "point" => (&point, vec![0; SITES]),
        "soma" => (&tree, vec![0; SITES]),
        other => panic!("unknown mode {other}"),
    };
    println!(
        "{mode}: {} compartments, {} synapses",
        cable.len(),
        6 * SITES
    );

    let start = Instant::now();
    let result = xor::train(cable, &sites, &config, |p| {
        if p.iter % 25 == 0 {
            println!(
                "iter {:4}  loss {:7.3}  mean soma mV [00 10 01 11] = {:?}  gap {:5.2} mV  {:.1}s",
                p.iter,
                p.loss,
                p.means
                    .iter()
                    .map(|x| (x * 10.0).round() / 10.0)
                    .collect::<Vec<_>>(),
                p.gap,
                start.elapsed().as_secs_f64()
            );
        }
    });
    let elapsed = start.elapsed().as_secs_f64();
    match result.solved_at {
        Some(iter) => println!("SOLVED XOR in {iter} iterations, {elapsed:.1}s wall"),
        None => println!(
            "NOT solved after {} iterations, {elapsed:.1}s wall",
            result.iterations
        ),
    }
    println!(
        "RESULT mode={mode} seed={seed} solved_at={} best_gap={:.3} seconds={elapsed:.2}",
        result.solved_at.map_or("none".into(), |i| i.to_string()),
        result.best_gap
    );
    if let Some(out) = out {
        // soma traces of all four patterns with the trained weights, for plotting
        let problems = xor::problems(cable, &sites);
        let traces: Vec<Vec<f64>> = std::iter::once(vec![PASSIVE.e_leak; xor::STEPS])
            .chain(problems.iter().map(|p| p.soma_trace(&result.weights)))
            .collect();
        let json = format!(
            "{{\"mode\":\"{mode}\",\"seed\":{seed},\"dt\":{},\"sites\":{:?},\"best_gap\":{},\"traces\":{:?}}}",
            xor::DT,
            sites,
            result.best_gap,
            traces
        );
        std::fs::write(&out, json).expect("write --out");
    }
}
