//! Single-compartment Hodgkin-Huxley (1952) neuron.
//!
//! Units: mV, ms, µA/cm², mS/cm², µF/cm², °C. Generic over the number type `T` so the same
//! solver can later run on dual numbers to differentiate traces with respect to parameters.

use num_traits::Float;

/// Voltage at which an upward crossing counts as a spike.
pub const SPIKE_THRESHOLD_MV: f64 = 0.0;
/// Temperature the original squid-axon rate constants were measured at.
const REFERENCE_TEMPERATURE: f64 = 6.3;
const Q10: f64 = 3.0;

fn k<T: Float>(x: f64) -> T {
    T::from(x).expect("constant representable in T")
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HhParams<T> {
    pub g_na: T,
    pub g_k: T,
    pub g_leak: T,
    pub e_na: T,
    pub e_k: T,
    pub e_leak: T,
    pub c_m: T,
    pub temperature: T,
    pub v_init: T,
}

impl<T: Float> Default for HhParams<T> {
    fn default() -> Self {
        Self {
            g_na: k(120.0),
            g_k: k(36.0),
            g_leak: k(0.3),
            e_na: k(50.0),
            e_k: k(-77.0),
            e_leak: k(-54.387),
            c_m: k(1.0),
            temperature: k(REFERENCE_TEMPERATURE),
            v_init: k(-65.0),
        }
    }
}

/// `x / (1 - exp(-x / s))`, with its limit `s` at the removable singularity x = 0.
fn exprel<T: Float>(x: T, s: T) -> T {
    if x.abs() < k(1e-6) {
        s
    } else {
        x / (T::one() - (-x / s).exp())
    }
}

/// Opening (alpha) and closing (beta) rates, in 1/ms at the reference temperature.
fn rates_m<T: Float>(v: T) -> (T, T) {
    (
        k::<T>(0.1) * exprel(v + k(40.0), k(10.0)),
        k::<T>(4.0) * (-(v + k(65.0)) / k(18.0)).exp(),
    )
}

fn rates_h<T: Float>(v: T) -> (T, T) {
    (
        k::<T>(0.07) * (-(v + k(65.0)) / k(20.0)).exp(),
        T::one() / (T::one() + (-(v + k(35.0)) / k(10.0)).exp()),
    )
}

fn rates_n<T: Float>(v: T) -> (T, T) {
    (
        k::<T>(0.01) * exprel(v + k(55.0), k(10.0)),
        k::<T>(0.125) * (-(v + k(65.0)) / k(80.0)).exp(),
    )
}

fn steady_state<T: Float>((alpha, beta): (T, T)) -> T {
    alpha / (alpha + beta)
}

/// Rush-Larsen step: exact for a gate under fixed voltage over `dt`.
fn advance_gate<T: Float>(x: T, (alpha, beta): (T, T), phi: T, dt: T) -> T {
    let total = (alpha + beta) * phi;
    let x_inf = alpha / (alpha + beta);
    x_inf + (x - x_inf) * (-dt * total).exp()
}

#[derive(Clone, Copy, Debug)]
pub struct HhState<T> {
    pub v: T,
    pub m: T,
    pub h: T,
    pub n: T,
}

impl<T: Float> HhState<T> {
    pub fn resting(params: &HhParams<T>) -> Self {
        let v = params.v_init;
        Self {
            v,
            m: steady_state(rates_m(v)),
            h: steady_state(rates_h(v)),
            n: steady_state(rates_n(v)),
        }
    }

    /// Advance by `dt` ms under injected current `i_ext`.
    pub fn step(&mut self, params: &HhParams<T>, i_ext: T, dt: T) {
        let phi = k::<T>(Q10).powf((params.temperature - k(REFERENCE_TEMPERATURE)) / k(10.0));
        self.m = advance_gate(self.m, rates_m(self.v), phi, dt);
        self.h = advance_gate(self.h, rates_h(self.v), phi, dt);
        self.n = advance_gate(self.n, rates_n(self.v), phi, dt);
        // With gates frozen the membrane is linear in v: relax exponentially towards v_inf.
        let g_na = params.g_na * self.m.powi(3) * self.h;
        let g_k = params.g_k * self.n.powi(4);
        let g_total = g_na + g_k + params.g_leak;
        let v_inf = (g_na * params.e_na + g_k * params.e_k + params.g_leak * params.e_leak + i_ext)
            / g_total;
        self.v = v_inf + (self.v - v_inf) * (-dt * g_total / params.c_m).exp();
    }
}

#[derive(Clone, Debug)]
pub struct Trace<T> {
    /// Membrane voltage after each step, mV.
    pub v: Vec<T>,
    /// Upward threshold crossings, ms, linearly interpolated within the step.
    pub spike_times: Vec<f64>,
}

/// Simulate one neuron from rest, injecting `current[i]` during step `i`.
pub fn simulate<T: Float>(params: &HhParams<T>, current: &[T], dt: T) -> Trace<T> {
    let mut state = HhState::resting(params);
    let mut v = Vec::with_capacity(current.len());
    let mut spike_times = Vec::new();
    let threshold: T = k(SPIKE_THRESHOLD_MV);
    let dt_ms = dt.to_f64().expect("dt is finite");
    for (i, &i_ext) in current.iter().enumerate() {
        let before = state.v;
        state.step(params, i_ext, dt);
        if before < threshold && state.v >= threshold {
            let frac = ((threshold - before) / (state.v - before))
                .to_f64()
                .unwrap_or(0.0);
            spike_times.push((i as f64 + frac) * dt_ms);
        }
        v.push(state.v);
    }
    Trace { v, spike_times }
}
