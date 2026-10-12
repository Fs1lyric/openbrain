//! Python bindings, exposed as `openbrain._core`.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::hh::{self, HhParams};

type Field = (&'static str, fn(&mut HhParams<f64>) -> &mut f64);

const HH_FIELDS: [Field; 9] = [
    ("g_na", |p| &mut p.g_na),
    ("g_k", |p| &mut p.g_k),
    ("g_leak", |p| &mut p.g_leak),
    ("e_na", |p| &mut p.e_na),
    ("e_k", |p| &mut p.e_k),
    ("e_leak", |p| &mut p.e_leak),
    ("c_m", |p| &mut p.c_m),
    ("temperature", |p| &mut p.temperature),
    ("v_init", |p| &mut p.v_init),
];

/// Default Hodgkin-Huxley parameters, keyed by name.
#[pyfunction]
fn hh_defaults() -> HashMap<&'static str, f64> {
    let mut params = HhParams::default();
    HH_FIELDS
        .iter()
        .map(|(name, field)| (*name, *field(&mut params)))
        .collect()
}

/// Simulate one HH neuron; returns (voltage per step, spike times in ms).
#[pyfunction]
fn simulate_hh(
    py: Python<'_>,
    params: HashMap<String, f64>,
    current: Vec<f64>,
    dt: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if !(dt > 0.0 && dt.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "dt must be positive, got {dt}"
        )));
    }
    let mut hh_params = HhParams::default();
    for (name, value) in params {
        let (_, field) = HH_FIELDS
            .iter()
            .find(|(field_name, _)| *field_name == name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown HH parameter {name:?}")))?;
        *field(&mut hh_params) = value;
    }
    let trace = py.detach(|| hh::simulate(&hh_params, &current, dt));
    Ok((trace.v, trace.spike_times))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hh_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_hh, m)?)?;
    Ok(())
}
