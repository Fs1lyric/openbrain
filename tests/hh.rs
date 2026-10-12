//! Acceptance tests: the Hodgkin-Huxley neuron must reproduce the textbook 1952 behaviour.

use openbrain_core::hh::{HhParams, simulate};

const DT: f64 = 0.01;
const SETTLE_MS: f64 = 300.0;

fn rate_hz(params: &HhParams<f64>, current: f64, ms: f64) -> f64 {
    let steps = (ms / DT) as usize;
    let trace = simulate(params, &vec![current; steps], DT);
    // skip onset transients (near threshold HH fires a few spikes then falls silent)
    let late = trace
        .spike_times
        .iter()
        .filter(|&&t| t >= SETTLE_MS)
        .count();
    late as f64 * 1000.0 / (ms - SETTLE_MS)
}

#[test]
fn rests_at_minus_65_mv_without_input() {
    let trace = simulate(&HhParams::default(), &vec![0.0; 50_000], DT);
    assert!(trace.spike_times.is_empty());
    for v in trace.v {
        assert!((v + 65.0).abs() < 0.5, "drifted to {v} mV");
    }
}

#[test]
fn action_potential_overshoots_zero_and_stays_below_sodium_reversal() {
    let trace = simulate(&HhParams::default(), &vec![10.0; 5_000], DT);
    let peak = trace.v.iter().cloned().fold(f64::MIN, f64::max);
    assert!((20.0..50.0).contains(&peak), "peak {peak} mV");
}

#[test]
fn fires_near_68_hz_at_10_ua_per_cm2() {
    let rate = rate_hz(&HhParams::default(), 10.0, 1100.0);
    assert!((60.0..80.0).contains(&rate), "rate {rate} Hz");
}

#[test]
fn weak_input_gives_no_sustained_firing() {
    assert_eq!(rate_hz(&HhParams::default(), 5.0, 600.0), 0.0);
}

#[test]
fn type_ii_excitability_onset_rate_jumps_from_zero() {
    let params = HhParams::default();
    let first_nonzero = (0..40)
        .map(|k| rate_hz(&params, 4.0 + 0.25 * k as f64, 600.0))
        .find(|&r| r > 0.0)
        .expect("never fired repetitively");
    assert!(first_nonzero > 40.0, "onset rate {first_nonzero} Hz");
}

#[test]
fn spike_times_converge_as_dt_shrinks() {
    let coarse = simulate(&HhParams::default(), &vec![10.0; 10_000], 0.02);
    let fine = simulate(&HhParams::default(), &vec![10.0; 40_000], 0.005);
    for (a, b) in coarse.spike_times.iter().zip(&fine.spike_times).take(5) {
        assert!((a - b).abs() < 0.2, "{a} vs {b} ms");
    }
}

#[test]
fn warmer_neurons_fire_faster() {
    let cold = rate_hz(&HhParams::default(), 10.0, 1100.0);
    let warm = rate_hz(
        &HhParams {
            temperature: 16.3,
            ..HhParams::default()
        },
        10.0,
        1100.0,
    );
    assert!(warm > 1.5 * cold, "cold {cold} Hz, warm {warm} Hz");
}

#[test]
fn blocking_sodium_channels_abolishes_spikes() {
    let ttx = HhParams {
        g_na: 0.0,
        ..HhParams::default()
    };
    let trace = simulate(&ttx, &vec![20.0; 50_000], DT);
    assert!(trace.spike_times.is_empty());
}
