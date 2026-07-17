use num_complex::Complex32;

use super::*;

#[test]
fn interferogram_sets_and_geometry_are_zero_based() {
    let sets = unwrap_ifg_sets(7, 3, &[1, 5], false).unwrap();
    assert_eq!(sets.unwrap_indices, [0, 2, 3, 4, 6]);
    assert_eq!(sets.solve_indices, [0, 2, 4, 6]);

    let geometry = single_master_ifg_geometry(4, 1).unwrap();
    assert_eq!(geometry.unwrap_indices, [0, 2, 3]);
    assert_eq!(geometry.ifgday_pairs, [[1, 0], [1, 2], [1, 3]]);
}

#[test]
fn grid_helpers_preserve_row_major_buffers_and_matlab_extraction_order() {
    let phase = [
        Complex32::new(1.0, 2.0),
        Complex32::new(0.5, -0.5),
        Complex32::new(3.0, -1.0),
        Complex32::new(-2.0, 0.25),
    ];
    let accumulated = grid_accumulate(&phase, 2, 2, &[1, 1], 3).unwrap();
    assert_eq!(accumulated[2], Complex32::new(4.0, 1.0));
    assert_eq!(accumulated[3], Complex32::new(-1.5, -0.25));

    let grid = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mask = [true, false, true, false, true, false];
    assert_eq!(
        extract_grid_values(&grid, &mask, 2, 3).unwrap(),
        [1.0, 5.0, 3.0]
    );
    let indices = ps_grid_indices(&mask, 2, 3, &[[0, 0], [1, 1], [0, 1]]).unwrap();
    assert_eq!(indices, [Some(0), Some(1), None]);
}

#[test]
fn cost_offsets_match_the_native_kernel_sign_contract() {
    let mut rowcost = vec![0_i16; 16];
    let mut colcost = vec![0_i16; 12];
    for chunk in rowcost
        .chunks_exact_mut(4)
        .chain(colcost.chunks_exact_mut(4))
    {
        chunk[1] = 1;
        chunk[2] = 32000;
        chunk[3] = -32000;
    }
    let (rows, cols) = prepare_cost_offsets(&CostOffsetInputs {
        rowcost_base: &rowcost,
        colcost_base: &colcost,
        rowix: &[1.0, -2.0, 0.0, f64::NAN],
        colix: &[2.0, -1.0, f64::NAN],
        row_shape: (2, 2),
        col_shape: (3, 1),
        wrapped_space_uw: &[1.25, -0.5],
        dph_smooth: &[0.25, 0.75],
        nshortcycle: 200.0,
    })
    .unwrap();
    assert_eq!([rows[0], rows[4], rows[8], rows[12]], [-32, -40, 0, 0]);
    assert_eq!([cols[0], cols[4], cols[8]], [-40, -32, 0]);
}

#[test]
fn reconstruction_keeps_missing_grid_assignments_as_nan() {
    let phase = [0.2_f32, 1.1, 3.4, -0.7];
    let wrapped = [
        Complex32::from_polar(1.0, 0.4),
        Complex32::from_polar(1.0, 1.3),
        Complex32::new(1.0, 0.0),
        Complex32::new(1.0, 0.0),
    ];
    let output = reconstruct_ps_phase(
        &phase,
        2,
        2,
        &[Some(0), None],
        &wrapped,
        Some(&[0.1, 0.2, 9.0, 9.0]),
    )
    .unwrap();
    assert!((output[0] - 0.5).abs() < 1.0e-6);
    assert!((output[1] - 1.5).abs() < 1.0e-6);
    assert!(output[2].is_nan() && output[3].is_nan());
}

#[test]
fn look_angle_estimation_matches_the_extension_reference_vector() {
    let day = [-24.0, -12.0, 18.0, 36.0];
    let bperp = [45.0, -15.0, 30.0, 75.0];
    let slopes = [0.015, -0.02, 0.0];
    let noise = [
        [0.02, -0.01, 0.03, -0.02],
        [0.01, 0.02, -0.02, 0.01],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let mut phase = Vec::new();
    for row in 0..3 {
        for column in 0..4 {
            phase.push(Complex32::from_polar(
                1.0,
                (slopes[row] * bperp[column] + noise[row][column]) as f32,
            ));
        }
    }
    let output = estimate_la_error_single_master(&phase, 3, 4, &day, &bperp, 2.0).unwrap();
    assert!((output[0] - 0.015132001).abs() < 1.0e-6);
    assert!((output[1] + 0.0201039).abs() < 1.0e-6);
    assert_eq!(output[2], 0.0);
}

#[test]
fn pure_grid_unwrap_honors_defo_shelf_and_reports_final_objective() {
    let input = GridUnwrapInputs {
        ifgw: &[Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)],
        rowcost: &[],
        colcost: &[-950, 1, 0, 1],
        nrow: 1,
        ncol: 2,
    };
    let output = unwrap_grid(
        &input,
        GridUnwrapConfig {
            parallel: false,
            ..Default::default()
        },
    )
    .unwrap();
    let label_delta = ((output.ifguw[1] - output.ifguw[0]) / std::f32::consts::TAU).round();
    assert_eq!(label_delta, 5.0);
    assert_eq!(output.post_label_flow_objective, 1251);
}

#[test]
fn profiled_grid_unwrap_preserves_output_and_reports_finite_phases() {
    let input = GridUnwrapInputs {
        ifgw: &[Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)],
        rowcost: &[],
        colcost: &[-950, 1, 0, 1],
        nrow: 1,
        ncol: 2,
    };
    let config = GridUnwrapConfig {
        parallel: false,
        ..Default::default()
    };

    let expected = unwrap_grid(&input, config).unwrap();
    let (observed, timings) = unwrap_grid_profiled(&input, config).unwrap();

    assert_eq!(observed, expected);
    for seconds in [
        timings.decode_sec,
        timings.initial_flow_sec,
        timings.initial_label_sec,
        timings.post_flow_sec,
        timings.final_label_sec,
        timings.msd_sec,
    ] {
        assert!(seconds.is_finite() && seconds >= 0.0);
    }
}
