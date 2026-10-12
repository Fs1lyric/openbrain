//! The XOR task and its point-neuron control.
//!
//! At steady state a single compartment mixes conductances supralinearly, so 11 would sit above 10
//! and 01. Scored on the *mean* soma voltage over a 50 ms window, though, a point neuron still learns
//! XOR: the fast GABA_A dip of two inputs sums almost linearly while their slower excitation
//! saturates. So this readout alone does not show that dendrites are needed; the tests pin that down.

use openbrain_core::cable::{Cable, Geometry, Passive};
use openbrain_core::xor::{self, Config};

const PASSIVE: Passive = Passive {
    cm: 1.0,
    g_leak: 1e-4,
    e_leak: -70.0,
    ri: 150.0,
};

/// Soma plus two long thin dendrites, one per input.
fn two_branch_tree() -> Geometry {
    let mut parent = vec![None];
    let mut length_um = vec![20.0];
    let mut diameter_um = vec![20.0];
    for _ in 0..2 {
        let mut prev = 0;
        for _ in 0..20 {
            parent.push(Some(prev));
            prev = parent.len() - 1;
            length_um.push(20.0);
            diameter_um.push(1.0);
        }
    }
    Geometry {
        parent,
        length_um,
        diameter_um,
    }
}

fn config() -> Config {
    Config {
        max_iters: 300,
        ..Config::default()
    }
}

#[test]
fn point_neuron_also_learns_mean_voltage_xor() {
    let tree = two_branch_tree();
    let point = Cable::new(&xor::point_geometry(&tree), PASSIVE).unwrap();
    assert_eq!(point.len(), 1);
    let result = xor::train(&point, &[0; 8], &config(), |_| {});
    assert!(result.solved_at.is_some(), "best gap {}", result.best_gap);
}

#[test]
fn constant_synaptic_drive_cannot_make_a_point_neuron_xor() {
    // the steady-state argument: V11 > min(V10, V01) whenever both depolarise
    let (gl, el) = (1.0, -70.0);
    let mix = |gs: &[(f64, f64)]| {
        let (g, ge) = gs
            .iter()
            .fold((gl, gl * el), |(g, ge), &(gi, ei)| (g + gi, ge + gi * ei));
        ge / g
    };
    let a = [(0.8, 0.0), (0.5, -80.0)];
    let b = [(0.3, 0.0), (2.0, -80.0)];
    let (v10, v01) = (mix(&a), mix(&b));
    let v11 = mix(&[a, b].concat());
    assert!(v10 > el && v01 > el);
    assert!(v11 > v10.min(v01));
}

#[test]
fn same_synapses_on_a_dendritic_tree_learn_xor() {
    let cable = Cable::new(&two_branch_tree(), PASSIVE).unwrap();
    let sites: Vec<usize> = (0..8).map(|i| 5 + 5 * i).collect();
    let result = xor::train(&cable, &sites, &config(), |_| {});
    assert!(result.solved_at.is_some(), "best gap {}", result.best_gap);
}

#[test]
fn point_geometry_preserves_membrane_area() {
    let tree = two_branch_tree();
    let area = |g: &Geometry| -> f64 {
        g.length_um
            .iter()
            .zip(&g.diameter_um)
            .map(|(l, d)| l * d)
            .sum()
    };
    let point = xor::point_geometry(&tree);
    assert!((area(&point) - area(&tree)).abs() < 1e-9 * area(&tree));
}
