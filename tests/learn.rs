//! Exact gradients through the cable simulation must match finite differences.

use openbrain_core::cable::{Cable, Geometry, Passive};
use openbrain_core::learn::{Problem, Synapse, SynapseKind, double_exponential};

const PASSIVE: Passive = Passive {
    cm: 1.0,
    g_leak: 1e-4,
    e_leak: -70.0,
    ri: 100.0,
};

fn tree() -> Cable {
    let geom = Geometry {
        parent: vec![
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(0),
            Some(4),
            Some(4),
            Some(6),
        ],
        length_um: vec![20.0, 50.0, 50.0, 50.0, 40.0, 60.0, 30.0, 80.0],
        diameter_um: vec![20.0, 2.0, 1.5, 1.0, 2.5, 0.8, 1.5, 0.7],
    };
    Cable::new(&geom, PASSIVE).unwrap()
}

fn problem(cable: &Cable) -> Problem<'_> {
    let steps = 400;
    let dt = 0.1;
    let synapses = vec![
        Synapse {
            compartment: 3,
            kind: SynapseKind::Ampa,
        },
        Synapse {
            compartment: 3,
            kind: SynapseKind::Nmda,
        },
        Synapse {
            compartment: 5,
            kind: SynapseKind::Nmda,
        },
        Synapse {
            compartment: 7,
            kind: SynapseKind::Ampa,
        },
        Synapse {
            compartment: 2,
            kind: SynapseKind::GabaA,
        },
    ];
    let drive = vec![
        double_exponential(&[50, 120], 0.2, 2.0, dt, steps),
        double_exponential(&[50, 120], 2.0, 50.0, dt, steps),
        double_exponential(&[80], 2.0, 50.0, dt, steps),
        double_exponential(&[60, 65, 70], 0.2, 2.0, dt, steps),
        double_exponential(&[100], 0.5, 8.0, dt, steps),
    ];
    Problem::new(cable, synapses, drive, dt).unwrap()
}

/// Squared error of the soma voltage against a -60 mV target, summed over time.
fn loss(_step: usize, v_soma: f64) -> (f64, f64) {
    let e = v_soma + 60.0;
    (e * e, 2.0 * e)
}

#[test]
fn gradient_matches_central_finite_differences() {
    let cable = tree();
    let p = problem(&cable);
    let w = vec![0.004, 0.006, 0.01, 0.003, 0.002];
    let (_, grad) = p.loss_and_grad(&w, loss);
    for i in 0..w.len() {
        let h = 1e-7;
        let (mut up, mut down) = (w.clone(), w.clone());
        up[i] += h;
        down[i] -= h;
        let fd = (p.loss(&up, loss) - p.loss(&down, loss)) / (2.0 * h);
        let rel = (grad[i] - fd).abs() / fd.abs().max(1e-12);
        assert!(
            rel < 1e-6,
            "weight {i}: adjoint {} vs finite difference {fd} (rel {rel})",
            grad[i]
        );
    }
}

#[test]
fn zero_weights_leave_the_cell_at_rest() {
    let cable = tree();
    let p = problem(&cable);
    let trace = p.soma_trace(&[0.0; 5]);
    assert!(trace.iter().all(|v| (v - PASSIVE.e_leak).abs() < 1e-9));
}

#[test]
fn nmda_is_supralinear_where_ampa_is_not() {
    // NMDA's magnesium block makes clustered input add up to more than the sum of its parts, once a
    // single input is strong enough (~4 mV) to start relieving the block but not to saturate it.
    let cable = tree();
    let peak = |kind, spikes: &[usize]| {
        let drive = vec![double_exponential(spikes, 2.0, 50.0, 0.1, 1500)];
        let syn = vec![Synapse {
            compartment: 3,
            kind,
        }];
        let p = Problem::new(&cable, syn, drive, 0.1).unwrap();
        p.soma_trace(&[0.004])
            .iter()
            .cloned()
            .fold(f64::MIN, f64::max)
            - PASSIVE.e_leak
    };
    let ratio = |kind| peak(kind, &[100, 101, 102, 103]) / (4.0 * peak(kind, &[100]));
    assert!(
        ratio(SynapseKind::Nmda) > 1.2,
        "NMDA ratio {}",
        ratio(SynapseKind::Nmda)
    );
    assert!(
        ratio(SynapseKind::Ampa) < 1.0,
        "AMPA ratio {}",
        ratio(SynapseKind::Ampa)
    );
}

#[test]
fn rejects_synapse_on_missing_compartment() {
    let cable = tree();
    let syn = vec![Synapse {
        compartment: 99,
        kind: SynapseKind::Ampa,
    }];
    assert!(Problem::new(&cable, syn, vec![vec![0.0; 10]], 0.1).is_err());
}

#[test]
fn current_synapse_matches_direct_current_injection() {
    let cable = tree();
    let (dt, steps) = (0.1, 300);
    let wave = double_exponential(&[20, 90], 1.0, 5.0, dt, steps);
    let syn = vec![Synapse {
        compartment: 5,
        kind: SynapseKind::Current,
    }];
    let p = Problem::new(&cable, syn, vec![wave.clone()], dt).unwrap();
    let trace = p.soma_trace(&[0.03]);
    let mut v = vec![PASSIVE.e_leak; cable.len()];
    let mut current = vec![0.0; cable.len()];
    for (step, &s) in wave.iter().enumerate() {
        current[5] = 0.03 * s;
        cable.step(&mut v, &current, dt);
        assert!((v[0] - trace[step]).abs() < 1e-12, "step {step}");
    }
}

#[test]
fn current_synapse_gradients_match_finite_differences() {
    let cable = tree();
    let (dt, steps) = (0.1, 300);
    let synapses = vec![
        Synapse {
            compartment: 7,
            kind: SynapseKind::Current,
        },
        Synapse {
            compartment: 3,
            kind: SynapseKind::Nmda,
        },
        Synapse {
            compartment: 5,
            kind: SynapseKind::Current,
        },
    ];
    let drive = vec![
        double_exponential(&[20], 1.0, 5.0, dt, steps),
        double_exponential(&[40], 2.0, 50.0, dt, steps),
        double_exponential(&[60, 70], 1.0, 5.0, dt, steps),
    ];
    let p = Problem::new(&cable, synapses, drive, dt).unwrap();
    let w = vec![0.02, 0.006, -0.01];
    let (_, grad) = p.loss_and_grad(&w, loss);
    for i in 0..w.len() {
        let h = 1e-7;
        let (mut up, mut down) = (w.clone(), w.clone());
        up[i] += h;
        down[i] -= h;
        let fd = (p.loss(&up, loss) - p.loss(&down, loss)) / (2.0 * h);
        let rel = (grad[i] - fd).abs() / fd.abs().max(1e-12);
        assert!(rel < 1e-6, "weight {i}: adjoint {} vs fd {fd}", grad[i]);
    }
}
