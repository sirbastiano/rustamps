use super::spatial::gaussian_low_pass;
use super::temporal::{gaussian_weights, high_pass_into};
use super::{estimate_scn, ScnConfig, ScnInputs};

#[test]
fn temporal_weights_exclude_master_and_survive_underflow() {
    let weights = gaussian_weights(&[0.0, 1.0e9], Some(0), 1.0e-3).unwrap();
    assert_eq!(weights, vec![0.0, 1.0, 0.0, 1.0]);
    let mut output = vec![0.0; 2];
    high_pass_into(&[2.0, 5.0], &weights, &mut output);
    assert_eq!(output, vec![-3.0, 0.0]);
}

#[test]
fn spatial_hash_matches_strict_radius_brute_force() {
    let xy = [0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 8.0, 0.0];
    let phase = [0.0_f32, 10.0, 100.0, 1000.0];
    let observed = gaussian_low_pass(&phase, &xy, 4, 1, 1.0).unwrap();
    let mut expected = Vec::new();
    for point in 0..4 {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for neighbor in 0..4 {
            let distance = xy[neighbor * 2] - xy[point * 2];
            if distance * distance < 16.0 {
                let weight = (-(distance * distance) / 2.0).exp();
                numerator += weight * phase[neighbor] as f64;
                denominator += weight;
            }
        }
        expected.push(numerator / denominator);
    }
    let reference = expected[0];
    for value in expected.iter_mut() {
        *value -= reference;
    }
    for (left, right) in observed.iter().zip(expected) {
        assert!((left - right).abs() < 1.0e-12);
    }
}

#[test]
fn scn_preserves_output_contract_and_reference_rules() {
    let n_ps = 4;
    let n_ifg = 4;
    let xy = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0];
    let phase = [
        1.0,
        0.0,
        2.0,
        5.0,
        2.0,
        0.0,
        f32::NAN,
        8.0,
        0.0,
        0.0,
        1.0,
        4.0,
        3.0,
        0.0,
        0.5,
        10.0,
    ];
    let scla = vec![0.25_f32; n_ps * n_ifg];
    let constant = [0.1_f32, 0.2, 0.3, 0.4];
    let mut scla_ramp = vec![0.0; n_ps * n_ifg];
    for row in 0..n_ps {
        scla_ramp[row * n_ifg + 3] = row as f64 * 0.1;
    }
    let output = estimate_scn(
        &ScnInputs {
            ph_uw: &phase,
            xy: &xy,
            day: &[0.0, 5.0, 11.0, 20.0],
            n_ps,
            n_ifg,
            ph_scla: Some(&scla),
            c_ps_uw: Some(&constant),
            scla_ramp: Some(&scla_ramp),
        },
        &ScnConfig {
            master_index: 1,
            unwrap_indices: &[0, 1, 3],
            deramp_indices: &[3],
            time_window: 8.0,
            wavelength: 1.0,
        },
    )
    .unwrap();

    assert_eq!(output.n_unwrap, 3);
    assert_eq!(output.n_deramp, 1);
    assert_eq!(output.ph_hpt.len(), n_ps * 3);
    assert_eq!(output.ph_ramp.len(), n_ps);
    assert_eq!(output.ph_scn_slave.len(), n_ps * n_ifg);
    assert!(output.ph_hpt.iter().all(|value| value.is_finite()));
    assert!(output.ph_scn_slave.iter().all(|value| value.is_finite()));
    assert!(output.ph_scn_slave[..n_ifg]
        .iter()
        .all(|value| *value == 0.0));
    for row in output.ph_scn_slave.chunks_exact(n_ifg) {
        assert_eq!(row[1], 0.0);
        assert_eq!(row[2], 0.0);
    }
}

#[test]
fn selected_deramp_recovers_the_fitted_plane() {
    let xy = [0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 2.0, 3.0];
    let mut phase = vec![0.0_f32; 4 * 3];
    for row in 0..4 {
        let x = xy[row * 2];
        let y = xy[row * 2 + 1];
        phase[row * 3] = row as f32 * 0.2;
        phase[row * 3 + 2] = (2.0 + 3.0 * x - 4.0 * y) as f32;
    }
    let output = estimate_scn(
        &ScnInputs {
            ph_uw: &phase,
            xy: &xy,
            day: &[0.0, 5.0, 10.0],
            n_ps: 4,
            n_ifg: 3,
            ph_scla: None,
            c_ps_uw: None,
            scla_ramp: None,
        },
        &ScnConfig {
            master_index: 1,
            unwrap_indices: &[0, 1, 2],
            deramp_indices: &[2],
            time_window: 5.0,
            wavelength: 1.0,
        },
    )
    .unwrap();

    for row in 0..4 {
        let expected = 2.0 + 3.0 * xy[row * 2] - 4.0 * xy[row * 2 + 1];
        assert!((output.ph_ramp[row] - expected).abs() < 1.0e-12);
    }
}

#[test]
fn requested_deramp_rejects_collinear_coordinates() {
    let error = estimate_scn(
        &ScnInputs {
            ph_uw: &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            xy: &[0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
            day: &[0.0, 1.0],
            n_ps: 3,
            n_ifg: 2,
            ph_scla: None,
            c_ps_uw: None,
            scla_ramp: None,
        },
        &ScnConfig {
            master_index: 0,
            unwrap_indices: &[0, 1],
            deramp_indices: &[1],
            time_window: 1.0,
            wavelength: 1.0,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("rank deficient"));
}
