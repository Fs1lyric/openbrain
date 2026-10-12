//! Benchmark the passive cable on one SWC morphology; prints a JSON object on stdout.
//!
//! Usage: cargo run --release --example cable_bench -- <file.swc> [repeats]
//!
//! Injects 0.1 nA into the soma (compartment 0) with cm 1 µF/cm², g_leak 1e-4 S/cm², e_leak -70 mV,
//! Ra 100 Ω·cm. Reports the steady-state soma voltage, input resistance, a 100 ms transient at
//! dt 0.025 ms, the geometry (so other simulators can rebuild the same compartments) and the median
//! wall-clock time of each phase over `repeats` runs.

use std::time::Instant;

use openbrain_core::cable::{Cable, Passive};
use openbrain_core::swc;

const PASSIVE: Passive = Passive {
    cm: 1.0,
    g_leak: 1e-4,
    e_leak: -70.0,
    ri: 100.0,
};
const I_SOMA_NA: f64 = 0.1;
const DT_MS: f64 = 0.025;
const T_STOP_MS: f64 = 100.0;

fn median_seconds(repeats: usize, mut run: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn json_array<T: std::fmt::Display>(values: impl IntoIterator<Item = T>) -> String {
    let items: Vec<String> = values.into_iter().map(|v| v.to_string()).collect();
    format!("[{}]", items.join(","))
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: cable_bench <file.swc> [repeats]")?;
    let repeats: usize = match args.next() {
        Some(r) => r.parse().map_err(|e| format!("repeats: {e}"))?,
        None => 5,
    };
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;

    let mut geom = None;
    let t_build = median_seconds(repeats, || {
        geom = Some(swc::parse(&text).map(|g| (Cable::new(&g, PASSIVE), g)));
    });
    let (cable, geom) = geom.expect("ran at least once")?;
    let cable = cable?;
    let n = cable.len();

    let mut current = vec![0.0; n];
    current[0] = I_SOMA_NA;

    let mut v_ss = Vec::new();
    let t_steady = median_seconds(repeats, || v_ss = cable.steady_state(&current));
    let soma_ss = v_ss[0];

    let steps = (T_STOP_MS / DT_MS).round() as usize;
    let mut trace = Vec::with_capacity(steps);
    let t_transient = median_seconds(repeats, || {
        let mut v = vec![PASSIVE.e_leak; n];
        trace.clear();
        for _ in 0..steps {
            cable.step(&mut v, &current, DT_MS);
            trace.push(v[0]);
        }
    });

    let parent = geom.parent.iter().map(|p| p.map_or(-1, |p| p as i64));
    println!(
        "{{\"n_compartments\":{n},\"soma_v_ss_mv\":{soma_ss},\"r_in_mohm\":{},\
         \"time_build_s\":{t_build},\"time_steady_s\":{t_steady},\"time_transient_s\":{t_transient},\
         \"dt_ms\":{DT_MS},\"soma_trace_mv\":{},\"parent\":{},\"length_um\":{},\"diameter_um\":{}}}",
        (soma_ss - PASSIVE.e_leak) / I_SOMA_NA,
        json_array(&trace),
        json_array(parent),
        json_array(&geom.length_um),
        json_array(&geom.diameter_um),
    );
    Ok(())
}
