//! Passive multicompartment cable, advanced with backward Euler and solved with the Hines method.
//!
//! Units: µm geometry, mV, ms, nA, µS, nF. Membrane properties per area: µF/cm², S/cm²; axial Ω·cm.

use std::f64::consts::PI;

/// A tree of cylindrical compartments. `parent[i] < i` for every non-root compartment.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub parent: Vec<Option<usize>>,
    pub length_um: Vec<f64>,
    pub diameter_um: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct Passive {
    pub cm: f64,
    pub g_leak: f64,
    pub e_leak: f64,
    pub ri: f64,
}

#[derive(Clone, Debug)]
pub struct Cable {
    pub parent: Vec<Option<usize>>,
    /// Membrane capacitance per compartment, nF.
    pub cap: Vec<f64>,
    /// Leak conductance per compartment, µS.
    pub g_leak: Vec<f64>,
    /// Axial conductance to the parent, µS (0 for the root).
    pub g_axial: Vec<f64>,
    pub e_leak: f64,
}

impl Cable {
    pub fn new(geom: &Geometry, passive: Passive) -> Result<Self, String> {
        let n = geom.parent.len();
        if geom.length_um.len() != n || geom.diameter_um.len() != n {
            return Err("geometry vectors differ in length".into());
        }
        let half_resistance = |i: usize| {
            let radius_cm = geom.diameter_um[i] * 0.5e-4;
            passive.ri * geom.length_um[i] * 0.5e-4 / (PI * radius_cm * radius_cm)
        };
        let mut g_axial = vec![0.0; n];
        for (i, parent) in geom.parent.iter().enumerate() {
            match *parent {
                Some(p) if p < i => g_axial[i] = 1e6 / (half_resistance(i) + half_resistance(p)),
                Some(p) => return Err(format!("compartment {i} listed before its parent {p}")),
                None if i != 0 => return Err(format!("compartment {i} is a second root")),
                None => {}
            }
        }
        let area_cm2: Vec<f64> = (0..n)
            .map(|i| PI * geom.diameter_um[i] * geom.length_um[i] * 1e-8)
            .collect();
        Ok(Self {
            parent: geom.parent.clone(),
            cap: area_cm2.iter().map(|a| passive.cm * a * 1e3).collect(),
            g_leak: area_cm2.iter().map(|a| passive.g_leak * a * 1e6).collect(),
            g_axial,
            e_leak: passive.e_leak,
        })
    }

    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Advance `v` by `dt` ms with injected currents `current` (nA), backward Euler.
    pub fn step(&self, v: &mut [f64], current: &[f64], dt: f64) {
        let mut rhs: Vec<f64> = (0..self.len())
            .map(|i| self.cap[i] / dt * v[i] + self.g_leak[i] * self.e_leak + current[i])
            .collect();
        self.solve(dt, &vec![0.0; self.len()], &mut rhs, v);
    }

    /// Solve `(C/dt + G_leak + G_axial + diag(extra_diag)) out = rhs` in O(n). The matrix is
    /// symmetric, so the same call also solves adjoint (transposed) systems. `rhs` is overwritten.
    pub fn solve(&self, dt: f64, extra_diag: &[f64], rhs: &mut [f64], out: &mut [f64]) {
        let base = self.system_diag(dt);
        let mut scratch = vec![0.0; self.len()];
        self.solve_prepared(&base, extra_diag, rhs, out, &mut scratch);
    }

    /// The diagonal `C/dt + G_leak + G_axial`, which stays fixed while `dt` does.
    pub fn system_diag(&self, dt: f64) -> Vec<f64> {
        let mut diag: Vec<f64> = (0..self.len())
            .map(|i| self.cap[i] / dt + self.g_leak[i])
            .collect();
        self.add_axial(&mut diag);
        diag
    }

    /// `solve` without allocating: `base` from `system_diag`, `scratch` of length `len()`.
    pub fn solve_prepared(
        &self,
        base: &[f64],
        extra_diag: &[f64],
        rhs: &mut [f64],
        out: &mut [f64],
        scratch: &mut [f64],
    ) {
        for i in 0..self.len() {
            scratch[i] = base[i] + extra_diag[i];
        }
        self.hines_solve(scratch, rhs, out);
    }

    /// Voltages once transients have died out under constant `current`.
    pub fn steady_state(&self, current: &[f64]) -> Vec<f64> {
        let mut diag = self.g_leak.clone();
        let mut rhs: Vec<f64> = (0..self.len())
            .map(|i| self.g_leak[i] * self.e_leak + current[i])
            .collect();
        self.add_axial(&mut diag);
        let mut v = vec![0.0; self.len()];
        self.hines_solve(&mut diag, &mut rhs, &mut v);
        v
    }

    /// The steady-state system as a dense matrix and right-hand side (for testing the solver).
    pub fn steady_state_system(&self, current: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = self.len();
        let mut diag = self.g_leak.clone();
        self.add_axial(&mut diag);
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = diag[i];
            if let Some(p) = self.parent[i] {
                a[i][p] = -self.g_axial[i];
                a[p][i] = -self.g_axial[i];
            }
        }
        let rhs = (0..n)
            .map(|i| self.g_leak[i] * self.e_leak + current[i])
            .collect();
        (a, rhs)
    }

    fn add_axial(&self, diag: &mut [f64]) {
        for i in 1..self.len() {
            if let Some(p) = self.parent[i] {
                diag[i] += self.g_axial[i];
                diag[p] += self.g_axial[i];
            }
        }
    }

    /// Solve the symmetric tree system (diag, off-diagonal -g_axial) in O(n): eliminate leaves
    /// into parents, then substitute from the root outwards.
    fn hines_solve(&self, diag: &mut [f64], rhs: &mut [f64], out: &mut [f64]) {
        for i in (1..self.len()).rev() {
            if let Some(p) = self.parent[i] {
                let f = -self.g_axial[i] / diag[i];
                diag[p] += f * self.g_axial[i];
                rhs[p] -= f * rhs[i];
            }
        }
        for i in 0..self.len() {
            out[i] = match self.parent[i] {
                Some(p) => (rhs[i] + self.g_axial[i] * out[p]) / diag[i],
                None => rhs[i] / diag[i],
            };
        }
    }
}
