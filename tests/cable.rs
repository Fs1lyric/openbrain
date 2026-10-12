//! The passive cable solver must match analytical cable theory and a dense linear solve.

use openbrain_core::cable::{Cable, Geometry, Passive};
use openbrain_core::swc;

const PASSIVE: Passive = Passive {
    cm: 1.0,
    g_leak: 1e-4,
    e_leak: -70.0,
    ri: 100.0,
};

/// Unbranched cylinder of `n` compartments, each `dx` µm long.
fn cylinder(n: usize, dx: f64, diameter: f64) -> Geometry {
    Geometry {
        parent: (0..n).map(|i| i.checked_sub(1)).collect(),
        length_um: vec![dx; n],
        diameter_um: vec![diameter; n],
    }
}

fn length_constant_um(d_um: f64) -> f64 {
    let rm = 1.0 / PASSIVE.g_leak; // Ω·cm²
    (rm * d_um * 1e-4 / (4.0 * PASSIVE.ri)).sqrt() * 1e4
}

#[test]
fn steady_state_attenuation_matches_sealed_end_cable_theory() {
    let (n, dx, d) = (200, 5.0, 2.0);
    let cable = Cable::new(&cylinder(n, dx, d), PASSIVE).unwrap();
    let mut current = vec![0.0; n];
    current[0] = 0.1; // nA
    let v = cable.steady_state(&current);
    let lambda = length_constant_um(d);
    let len = n as f64 * dx;
    let dv0 = v[0] - PASSIVE.e_leak;
    for k in [20, 60, 100, 150, 199] {
        let x = (k as f64 + 0.5) * dx - 0.5 * dx;
        let expected = ((len - x) / lambda).cosh() / (len / lambda).cosh();
        let got = (v[k] - PASSIVE.e_leak) / dv0;
        assert!((got - expected).abs() < 0.01, "x={x}: {got} vs {expected}");
    }
}

#[test]
fn input_resistance_matches_cable_theory() {
    let (n, dx, d) = (400, 2.5, 2.0);
    let cable = Cable::new(&cylinder(n, dx, d), PASSIVE).unwrap();
    let mut current = vec![0.0; n];
    current[0] = 0.1;
    let r_in = (cable.steady_state(&current)[0] - PASSIVE.e_leak) / 0.1; // MΩ
    let lambda_cm = length_constant_um(d) * 1e-4;
    let r_axial = 4.0 * PASSIVE.ri / (std::f64::consts::PI * (d * 1e-4f64).powi(2)); // Ω/cm
    let expected = r_axial * lambda_cm / (n as f64 * dx * 1e-4 / lambda_cm).tanh() * 1e-6;
    assert!(
        (r_in - expected).abs() / expected < 0.02,
        "{r_in} vs {expected} MΩ"
    );
}

#[test]
fn isopotential_compartment_charges_with_membrane_time_constant() {
    let cable = Cable::new(&cylinder(1, 20.0, 20.0), PASSIVE).unwrap();
    let mut v = vec![PASSIVE.e_leak];
    let dt = 0.001;
    let tau = PASSIVE.cm / PASSIVE.g_leak * 1e-3; // ms
    let i_ss = 0.01;
    for _ in 0..(tau / dt).round() as usize {
        cable.step(&mut v, &[i_ss], dt);
    }
    let v_inf = cable.steady_state(&[i_ss])[0];
    let frac = (v[0] - PASSIVE.e_leak) / (v_inf - PASSIVE.e_leak);
    assert!(
        (frac - (1.0 - (-1.0f64).exp())).abs() < 1e-3,
        "charged to {frac}"
    );
}

#[test]
fn hines_solve_matches_dense_solve_on_a_branched_tree() {
    // root, two branches, one of which forks again
    let geom = Geometry {
        parent: vec![None, Some(0), Some(1), Some(0), Some(3), Some(3), Some(5)],
        length_um: vec![15.0, 40.0, 30.0, 25.0, 50.0, 10.0, 60.0],
        diameter_um: vec![15.0, 2.0, 1.5, 3.0, 1.0, 2.0, 0.8],
    };
    let cable = Cable::new(&geom, PASSIVE).unwrap();
    let current = [0.05, 0.0, -0.02, 0.0, 0.03, 0.0, 0.01];
    let fast = cable.steady_state(&current);
    let (matrix, rhs) = cable.steady_state_system(&current);
    let dense = dense_solve(matrix, rhs);
    for (a, b) in fast.iter().zip(&dense) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn rejects_children_listed_before_parents() {
    let geom = Geometry {
        parent: vec![None, Some(2), Some(0)],
        length_um: vec![10.0; 3],
        diameter_um: vec![1.0; 3],
    };
    assert!(Cable::new(&geom, PASSIVE).is_err());
}

#[test]
fn swc_becomes_soma_plus_segment_compartments() {
    let text = "# comment\n1 1 0 0 0 5 -1\n2 3 10 0 0 1 1\n3 3 10 20 0 0.5 2\n4 3 -10 0 0 1 1\n";
    let geom = swc::parse(text).unwrap();
    assert_eq!(geom.parent, vec![None, Some(0), Some(1), Some(0)]);
    assert_eq!(geom.diameter_um, vec![10.0, 2.0, 1.0, 2.0]);
    assert!((geom.length_um[2] - 20.0).abs() < 1e-12);
    assert!((geom.length_um[0] - 10.0).abs() < 1e-12); // soma: cylinder with length = diameter
}

#[test]
fn swc_rejects_unknown_parent() {
    assert!(swc::parse("1 1 0 0 0 5 -1\n2 3 1 0 0 1 7\n").is_err());
}

#[allow(clippy::needless_range_loop)] // textbook Gaussian elimination reads clearer with indices
fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|k| a[i][k] * x[k]).sum::<f64>()) / a[i][i];
    }
    x
}
