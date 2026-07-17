use super::qr::QrSolver;
use super::{
    build_smoothed_phase, center_to_reference, deramp_phase, estimate_scla,
    smooth_neighbor_envelope, Stage7Inputs,
};

#[test]
fn pivoted_qr_recovers_scaled_coefficients() {
    let design = vec![
        1.0, 1000.0, -2.0, 1.0, 2000.0, -1.0, 1.0, 3000.0, 1.0, 1.0, 5000.0, 4.0,
    ];
    let solver = QrSolver::factor(&design, 4, 3).unwrap();
    let expected = [2.0, -0.003, 5.0];
    let observed = solver.solve_with(|row| {
        (0..3)
            .map(|column| design[row * 3 + column] * expected[column])
            .sum()
    });
    for (left, right) in observed.iter().zip(expected) {
        assert!((left - right).abs() < 1.0e-11);
    }
}

#[test]
fn pivoted_qr_rejects_rank_deficiency() {
    let design = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
    assert!(QrSolver::factor(&design, 3, 2)
        .unwrap_err()
        .contains("rank deficient"));
}

#[test]
fn stage7_recovers_scla_and_preserves_f64_k() {
    let n_ps = 2;
    let n_ifg = 5;
    let day = [0.0, 10.0, 25.0, 45.0, 70.0];
    let baseline = [-20.0, 5.0, 0.0, 30.0, 70.0, -20.0, 5.0, 0.0, 30.0, 70.0];
    let k = [0.012345678901, -0.02];
    let c = [0.4, -0.8];
    let velocity = [0.002, -0.001];
    let mut phase = vec![0.0; n_ps * n_ifg];
    for row in 0..n_ps {
        for column in 0..n_ifg {
            phase[row * n_ifg + column] = k[row] * baseline[row * n_ifg + column]
                + c[row]
                + velocity[row] * (day[column] - day[2]);
        }
    }
    let output = estimate_scla(&Stage7Inputs {
        ph_proc: &phase,
        bperp_mat: &baseline,
        n_ps,
        n_ifg,
        unwrap_indices: &[0, 1, 2, 3, 4],
        solve_indices: &[0, 1, 3, 4],
        day: &day,
        master_index: 2,
        ifg_std: &[1.0; 5],
    })
    .unwrap();

    for row in 0..n_ps {
        assert!((output.k_ps_uw[row] - k[row]).abs() < 1.0e-10);
        assert!((output.c_ps_uw[row] as f64 - c[row]).abs() < 2.0e-6);
    }
    assert_eq!(output.ph_scla.len(), n_ps * n_ifg);
    assert_eq!(output.ifg_vcm.len(), n_ifg * n_ifg);
}

#[test]
fn stage7_handles_constant_acquisition_intervals_without_biasing_scla() {
    let n_ps = 2;
    let n_ifg = 5;
    let day = [0.0, 12.0, 24.0, 36.0, 48.0];
    let baseline = [-20.0, 5.0, 0.0, 31.0, 68.0, -20.0, 5.0, 0.0, 31.0, 68.0];
    let k = [0.012345678901, -0.02];
    let c = [0.4, -0.8];
    let velocity = [0.002, -0.001];
    let mut phase = vec![0.0; n_ps * n_ifg];
    for row in 0..n_ps {
        for column in 0..n_ifg {
            phase[row * n_ifg + column] = k[row] * baseline[row * n_ifg + column]
                + c[row]
                + velocity[row] * (day[column] - day[2]);
        }
    }

    let output = estimate_scla(&Stage7Inputs {
        ph_proc: &phase,
        bperp_mat: &baseline,
        n_ps,
        n_ifg,
        unwrap_indices: &[0, 1, 2, 3, 4],
        solve_indices: &[0, 1, 3, 4],
        day: &day,
        master_index: 2,
        ifg_std: &[1.0; 5],
    })
    .unwrap();

    for row in 0..n_ps {
        assert!((output.k_ps_uw[row] - k[row]).abs() < 1.0e-10);
        assert!((output.c_ps_uw[row] as f64 - c[row]).abs() < 2.0e-6);
    }
}

#[test]
fn smoothing_clamps_without_downcasting_k() {
    let k = [10.0000000001, 1.0000000002, 2.0000000003];
    let c = [5.0_f32, 0.0, 2.0];
    let edges = [[0, 1], [1, 2], [0, 2]];
    let (k_out, c_out) = smooth_neighbor_envelope(&k, &c, &edges).unwrap();
    assert_eq!(k_out, vec![2.0000000003; 3]);
    assert_eq!(c_out, vec![2.0; 3]);

    let phase = build_smoothed_phase(&k_out, &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 2).unwrap();
    assert_eq!(phase.len(), 6);
}

#[test]
fn phase_preparation_deramps_and_centers_with_nans() {
    let xy = [
        0.0, 0.0, 1000.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0, 2000.0, 0.0, 0.0, 2000.0, 2000.0,
        2000.0,
    ];
    let mut phase = vec![0.0; 7 * 2];
    for row in 0..7 {
        let x = xy[row * 2] / 1000.0;
        let y = xy[row * 2 + 1] / 1000.0;
        phase[row * 2] = 2.0 * x - 3.0 * y + 4.0;
        phase[row * 2 + 1] = -x + 0.5 * y - 2.0;
    }
    phase[3] = f64::NAN;
    let deramped = deramp_phase(&phase, &xy, 7, 2).unwrap();
    for (index, value) in deramped.phase.iter().enumerate() {
        if index == 3 {
            assert!(value.is_nan());
        } else {
            assert!(value.abs() < 1.0e-11);
        }
    }

    let centered = center_to_reference(&[1.0, 3.0, 5.0, f64::NAN], 2, 2, &[0, 1]).unwrap();
    assert_eq!(centered[0], -2.0);
    assert_eq!(centered[2], 2.0);
    assert_eq!(centered[1], 0.0);
    assert!(centered[3].is_nan());
}
