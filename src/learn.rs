//! Synapses on a passive dendritic tree, and exact gradients of a soma-voltage loss with respect to
//! every synaptic weight, computed by the discrete adjoint of the backward-Euler Hines scheme.
//!
//! Each step solves `A(v^n) v^{n+1} = b(v^n)`, where synaptic conductances `w_s * s_s[n] * B(v^n)`
//! enter the diagonal of `A` and the right-hand side `b`. NMDA's magnesium block `B` is evaluated at
//! the previous voltage, so every step is one linear tree solve. The adjoint walks backwards through
//! the same steps, and because `A` is symmetric each backward step is also one Hines solve.

use crate::cable::Cable;

const MAGNESIUM_MM: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SynapseKind {
    Ampa,
    Nmda,
    GabaA,
}

impl SynapseKind {
    fn reversal(self) -> f64 {
        match self {
            Self::Ampa | Self::Nmda => 0.0,
            Self::GabaA => -80.0,
        }
    }

    /// Voltage dependence of the conductance and its derivative (Jahr & Stevens 1990 for NMDA).
    fn block(self, v: f64) -> (f64, f64) {
        match self {
            Self::Nmda => {
                let e = (-0.062 * v).exp() * MAGNESIUM_MM / 3.57;
                let b = 1.0 / (1.0 + e);
                (b, b * b * 0.062 * e)
            }
            _ => (1.0, 0.0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Synapse {
    pub compartment: usize,
    pub kind: SynapseKind,
}

/// Peak-normalised double-exponential conductance waveform for presynaptic spikes at `spike_steps`.
pub fn double_exponential(
    spike_steps: &[usize],
    tau_rise: f64,
    tau_decay: f64,
    dt: f64,
    steps: usize,
) -> Vec<f64> {
    let t_peak = tau_rise * tau_decay / (tau_decay - tau_rise) * (tau_decay / tau_rise).ln();
    let norm = 1.0 / ((-t_peak / tau_decay).exp() - (-t_peak / tau_rise).exp());
    let mut s = vec![0.0; steps];
    for &k in spike_steps {
        for (n, value) in s.iter_mut().enumerate().skip(k) {
            let t = (n - k) as f64 * dt;
            *value += norm * ((-t / tau_decay).exp() - (-t / tau_rise).exp());
        }
    }
    s
}

/// A cell, its synapses and their input waveforms: everything except the weights.
pub struct Problem<'a> {
    cable: &'a Cable,
    synapses: Vec<Synapse>,
    drive: Vec<Vec<f64>>,
    dt: f64,
    steps: usize,
}

impl<'a> Problem<'a> {
    pub fn new(
        cable: &'a Cable,
        synapses: Vec<Synapse>,
        drive: Vec<Vec<f64>>,
        dt: f64,
    ) -> Result<Self, String> {
        if synapses.len() != drive.len() {
            return Err("one drive waveform per synapse".into());
        }
        if let Some(s) = synapses.iter().find(|s| s.compartment >= cable.len()) {
            return Err(format!("synapse on missing compartment {}", s.compartment));
        }
        let steps = drive.first().map_or(0, Vec::len);
        if drive.iter().any(|d| d.len() != steps) {
            return Err("drive waveforms differ in length".into());
        }
        Ok(Self {
            cable,
            synapses,
            drive,
            dt,
            steps,
        })
    }

    /// Voltages of every compartment at steps 0..=steps, starting from rest.
    fn trajectory(&self, weights: &[f64]) -> Vec<Vec<f64>> {
        let n = self.cable.len();
        let mut traj = Vec::with_capacity(self.steps + 1);
        traj.push(vec![self.cable.e_leak; n]);
        let mut extra = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        for step in 0..self.steps {
            let v = &traj[step];
            extra.fill(0.0);
            for i in 0..n {
                rhs[i] =
                    self.cable.cap[i] / self.dt * v[i] + self.cable.g_leak[i] * self.cable.e_leak;
            }
            for (s, syn) in self.synapses.iter().enumerate() {
                let c = syn.compartment;
                let g = weights[s] * self.drive[s][step] * syn.kind.block(v[c]).0;
                extra[c] += g;
                rhs[c] += g * syn.kind.reversal();
            }
            let mut next = vec![0.0; n];
            self.cable.solve(self.dt, &extra, &mut rhs, &mut next);
            traj.push(next);
        }
        traj
    }

    /// Soma (compartment 0) voltage after each step.
    pub fn soma_trace(&self, weights: &[f64]) -> Vec<f64> {
        self.trajectory(weights)
            .iter()
            .skip(1)
            .map(|v| v[0])
            .collect()
    }

    /// `sum_k loss(k, v_soma after step k)`; `loss` returns (value, d value / d v_soma).
    pub fn loss(&self, weights: &[f64], loss: impl Fn(usize, f64) -> (f64, f64)) -> f64 {
        self.soma_trace(weights)
            .iter()
            .enumerate()
            .map(|(k, &v)| loss(k, v).0)
            .sum()
    }

    /// The loss and its exact gradient with respect to every weight.
    pub fn loss_and_grad(
        &self,
        weights: &[f64],
        loss: impl Fn(usize, f64) -> (f64, f64),
    ) -> (f64, Vec<f64>) {
        let n = self.cable.len();
        let traj = self.trajectory(weights);
        let mut total = 0.0;
        let mut grad = vec![0.0; weights.len()];
        // adjoint = dL/dv^{step+1}, built up from the end of the simulation backwards
        let mut adjoint = vec![0.0; n];
        let (value, d_soma) = loss(self.steps - 1, traj[self.steps][0]);
        total += value;
        adjoint[0] = d_soma;
        let mut extra = vec![0.0; n];
        let mut mu = vec![0.0; n];
        for step in (0..self.steps).rev() {
            let (v, v_next) = (&traj[step], &traj[step + 1]);
            extra.fill(0.0);
            for (s, syn) in self.synapses.iter().enumerate() {
                extra[syn.compartment] +=
                    weights[s] * self.drive[s][step] * syn.kind.block(v[syn.compartment]).0;
            }
            let mut rhs = adjoint.clone();
            self.cable.solve(self.dt, &extra, &mut rhs, &mut mu);
            // d/dv^step of mu · (b(v^step) - A(v^step) v^{step+1})
            for i in 0..n {
                adjoint[i] = self.cable.cap[i] / self.dt * mu[i];
            }
            for (s, syn) in self.synapses.iter().enumerate() {
                let c = syn.compartment;
                let (block, d_block) = syn.kind.block(v[c]);
                let driving_force = syn.kind.reversal() - v_next[c];
                grad[s] += mu[c] * self.drive[s][step] * block * driving_force;
                adjoint[c] += mu[c] * weights[s] * self.drive[s][step] * d_block * driving_force;
            }
            if step > 0 {
                let (value, d_soma) = loss(step - 1, v[0]);
                total += value;
                adjoint[0] += d_soma;
            }
        }
        (total, grad)
    }
}
