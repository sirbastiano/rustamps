use num_complex::{Complex32, Complex64};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::f64::consts::PI;

mod stage3_clap_stack;
mod stage3_native;
mod stage4_edge_stats_native;
mod stage4_native;
mod stage5_native;
mod stage6_component_shift;
#[cfg(test)]
mod stage6_component_shift_tests;
mod stage6_cut;
mod stage6_cut_graph;
mod stage6_flow;
#[allow(dead_code)]
mod stage6_incr_cost;
mod stage6_la_native;
mod stage6_label_flow;
#[cfg(test)]
mod stage6_label_flow_tests;
mod stage6_local_cycles;
#[allow(dead_code)]
mod stage6_mst;
#[allow(dead_code)]
mod stage6_mst_flow;
#[cfg(test)]
mod stage6_mst_flow_tests;
mod stage6_native;
#[cfg(test)]
mod stage6_native_flow_tests;
mod stage6_patch;
#[allow(dead_code)]
mod stage6_residual;
#[cfg(test)]
mod stage6_residual_tests;
#[allow(dead_code)]
mod stage6_residual_view;
#[cfg(test)]
mod stage6_residual_view_tests;
#[allow(dead_code)]
mod stage6_residue;
mod stage6_route;
mod stage6_smooth_native;
#[allow(dead_code)]
mod stage6_tree_compact;
#[cfg(test)]
mod stage6_tree_compact_tests;
#[allow(dead_code)]
mod stage6_tree_cycle;
#[cfg(test)]
mod stage6_tree_cycle_tests;
#[allow(dead_code)]
mod stage6_tree_path;
#[cfg(test)]
mod stage6_tree_path_tests;
#[cfg(test)]
mod stage6_tree_search_tests;
mod stage7_deramp_native;
mod stage7_native;
mod stage7_scla_native;
mod stage8_native;
mod weighted_fit_native;

const QUARTER_PI: f64 = PI / 4.0;
const QUARTER_PI_F32: f32 = std::f32::consts::PI / 4.0;
const STAGE8_NOISE_SCALE: f32 = 0.5;

struct RowData {
    valid_cols: Vec<usize>,
    cpx: Vec<Complex64>,
    bp: Vec<f64>,
    weighting: Vec<f64>,
    wb: Vec<f64>,
    den_lin: f64,
    bperp_range: f64,
    denom: f64,
    n_col: usize,
}

struct RowDataSingle {
    valid_cols: Vec<usize>,
    cpx: Vec<Complex32>,
    bp: Vec<f32>,
    weighting: Vec<f32>,
    wb: Vec<f32>,
    den_lin: f32,
    bperp_range: f32,
    denom: f32,
    n_col: usize,
}

struct RefinedRow {
    k: f64,
    c: f64,
    coh: f64,
    residual: Vec<Complex32>,
}

struct Stage7Outputs {
    k_ps_uw: Vec<f64>,
    c_ps_uw: Vec<f32>,
    ph_scla: Vec<f32>,
    ifg_vcm: Vec<f64>,
    mean_v: Vec<f32>,
    m: Vec<f32>,
    ph_ramp: Vec<f64>,
}

struct Stage7RowResult {
    k: f64,
    c: f32,
    ph_scla_row: Vec<f32>,
    mean_v: f32,
    intercept: f32,
    slope: f32,
}

fn build_pool(thread_count: usize) -> PyResult<Option<ThreadPool>> {
    if thread_count <= 1 {
        return Ok(None);
    }
    ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .map(Some)
        .map_err(|err| PyValueError::new_err(format!("failed to build stage-2 thread pool: {err}")))
}

fn trial_values(n_trial_wraps: f64) -> Vec<f64> {
    let trial_n = (8.0 * n_trial_wraps).ceil() as i64;
    (-trial_n..=trial_n).map(|value| value as f64).collect()
}

fn parse_indices(values: &[i64], upper_bound: usize, label: &str) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(values.len());
    for &value in values {
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "{label} entries must be non-negative"
            )));
        }
        let idx = value as usize;
        if idx >= upper_bound {
            return Err(PyValueError::new_err(format!(
                "{label} entry {idx} exceeds width {upper_bound}"
            )));
        }
        out.push(idx);
    }
    Ok(out)
}

fn solve_linear_system(mut matrix: Vec<f64>, mut rhs: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for pivot_col in 0..n {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col * n + pivot_col].abs();
        for row in (pivot_col + 1)..n {
            let value_abs = matrix[row * n + pivot_col].abs();
            if value_abs > pivot_abs {
                pivot_row = row;
                pivot_abs = value_abs;
            }
        }
        if pivot_abs <= 1.0e-12 {
            return None;
        }
        if pivot_row != pivot_col {
            for col in 0..n {
                matrix.swap(pivot_col * n + col, pivot_row * n + col);
            }
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot = matrix[pivot_col * n + pivot_col];
        for row in (pivot_col + 1)..n {
            let factor = matrix[row * n + pivot_col] / pivot;
            if factor == 0.0 {
                continue;
            }
            matrix[row * n + pivot_col] = 0.0;
            for col in (pivot_col + 1)..n {
                matrix[row * n + col] -= factor * matrix[pivot_col * n + col];
            }
            rhs[row] -= factor * rhs[pivot_col];
        }
    }

    let mut out = vec![0.0; n];
    for row in (0..n).rev() {
        let mut value = rhs[row];
        for col in (row + 1)..n {
            value -= matrix[row * n + col] * out[col];
        }
        let diag = matrix[row * n + row];
        if diag.abs() <= 1.0e-12 {
            return None;
        }
        out[row] = value / diag;
    }
    Some(out)
}

fn invert_small_matrix(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inverse = vec![0.0; n * n];
    for col in 0..n {
        let mut basis = vec![0.0; n];
        basis[col] = 1.0;
        let solution = solve_linear_system(matrix.to_vec(), basis, n)?;
        for row in 0..n {
            inverse[row * n + col] = solution[row];
        }
    }
    Some(inverse)
}

fn invert_small_matrix_with_jitter(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut jitter = 0.0_f64;
    loop {
        let mut adjusted = matrix.to_vec();
        for diag in 0..n {
            adjusted[diag * n + diag] += jitter;
        }
        if let Some(inverse) = invert_small_matrix(&adjusted, n) {
            return inverse;
        }
        jitter = if jitter == 0.0 {
            1.0e-10
        } else {
            jitter * 10.0
        };
        if jitter > 1.0e-3 {
            let mut identity = vec![0.0; n * n];
            for diag in 0..n {
                identity[diag * n + diag] = 1.0;
            }
            return identity;
        }
    }
}

fn design_gram(design: &[f64], n_obs: usize, n_coeff: usize) -> Vec<f64> {
    let mut gram = vec![0.0; n_coeff * n_coeff];
    for row in 0..n_obs {
        let row_slice = &design[row * n_coeff..(row + 1) * n_coeff];
        for left in 0..n_coeff {
            for right in 0..n_coeff {
                gram[left * n_coeff + right] += row_slice[left] * row_slice[right];
            }
        }
    }
    gram
}

fn design_rhs(design: &[f64], y: &[f64], n_obs: usize, n_coeff: usize) -> Vec<f64> {
    let mut rhs = vec![0.0; n_coeff];
    for row in 0..n_obs {
        let y_value = y[row];
        let row_slice = &design[row * n_coeff..(row + 1) * n_coeff];
        for coeff_ix in 0..n_coeff {
            rhs[coeff_ix] += row_slice[coeff_ix] * y_value;
        }
    }
    rhs
}

fn mat_vec(matrix: &[f64], vector: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for row in 0..n {
        let mut accum = 0.0;
        for col in 0..n {
            accum += matrix[row * n + col] * vector[col];
        }
        out[row] = accum;
    }
    out
}

fn stage7_outputs(
    ph_proc: &[f64],
    ph_mean_v: &[f64],
    bperp_mat: &[f64],
    n_ps: usize,
    n_ifg: usize,
    unwrap_ix: &[usize],
    solve_ix: &[usize],
    day: &[f64],
    master_ix: usize,
    ifg_std: &[f64],
    threads: usize,
) -> PyResult<Stage7Outputs> {
    if unwrap_ix.len() < 2 {
        return Err(PyValueError::new_err(
            "stage7_scla requires at least two unwrap indices",
        ));
    }
    if solve_ix.len() < 2 {
        return Err(PyValueError::new_err(
            "stage7_scla requires at least two solve indices",
        ));
    }
    if master_ix == 0 || master_ix > n_ifg {
        return Err(PyValueError::new_err(
            "master_ix must be 1-based within the interferogram width",
        ));
    }
    if day.len() != n_ifg || ifg_std.len() != n_ifg {
        return Err(PyValueError::new_err(
            "day and ifg_std must match the interferogram width",
        ));
    }

    let unwrap_obs = unwrap_ix.len() - 1;
    let coest_mean_vel = unwrap_ix.len() >= 4;
    let seq_coeff = if coest_mean_vel { 3 } else { 2 };
    let mut mean_bperp = vec![0.0; unwrap_obs];
    for obs_ix in 0..unwrap_obs {
        let left_col = unwrap_ix[obs_ix];
        let right_col = unwrap_ix[obs_ix + 1];
        let mut accum = 0.0;
        for row_ix in 0..n_ps {
            accum += bperp_mat[row_ix * n_ifg + right_col] - bperp_mat[row_ix * n_ifg + left_col];
        }
        mean_bperp[obs_ix] = accum / n_ps as f64;
    }

    let mut design_seq = vec![0.0; unwrap_obs * seq_coeff];
    for obs_ix in 0..unwrap_obs {
        let day_diff = day[unwrap_ix[obs_ix + 1]] - day[unwrap_ix[obs_ix]];
        let row_offset = obs_ix * seq_coeff;
        design_seq[row_offset] = 1.0;
        design_seq[row_offset + 1] = mean_bperp[obs_ix];
        if coest_mean_vel {
            design_seq[row_offset + 2] = day_diff;
        }
    }
    let inv_seq = invert_small_matrix_with_jitter(
        &design_gram(&design_seq, unwrap_obs, seq_coeff),
        seq_coeff,
    );

    let master_zero = day[master_ix - 1];
    let mut solve_design = Vec::new();
    let mut solve_scales = Vec::new();
    let inv_c = if coest_mean_vel {
        solve_design = vec![0.0; solve_ix.len() * 2];
        solve_scales = vec![1.0; solve_ix.len()];
        for (obs_ix, &ifg_ix) in solve_ix.iter().enumerate() {
            let scale = if ifg_std[ifg_ix] > 0.0 {
                ifg_std[ifg_ix] * PI / 180.0
            } else {
                1.0
            };
            solve_scales[obs_ix] = scale;
            let row_offset = obs_ix * 2;
            solve_design[row_offset] = 1.0 / scale;
            solve_design[row_offset + 1] = (day[ifg_ix] - master_zero) / scale;
        }
        Some(invert_small_matrix_with_jitter(
            &design_gram(&solve_design, solve_ix.len(), 2),
            2,
        ))
    } else {
        None
    };

    let time_diff: Vec<f64> = day.iter().map(|&value| value - master_zero).collect();
    let weights_mv: Vec<f64> = ifg_std
        .iter()
        .map(|&std| {
            if std > 0.0 {
                1.0 / ((std * PI / 180.0) * (std * PI / 180.0))
            } else {
                0.0
            }
        })
        .collect();
    let s0: f64 = weights_mv.iter().sum();
    let s1: f64 = weights_mv
        .iter()
        .zip(time_diff.iter())
        .map(|(&w, &t)| w * t)
        .sum();
    let s2: f64 = weights_mv
        .iter()
        .zip(time_diff.iter())
        .map(|(&w, &t)| w * t * t)
        .sum();
    let det = s0 * s2 - s1 * s1;

    let ifg_var: Vec<f64> = ifg_std
        .iter()
        .map(|&std| (std * PI / 180.0) * (std * PI / 180.0))
        .collect();
    let mut ifg_vcm = vec![0.0; n_ifg * n_ifg];
    for diag_ix in 0..n_ifg {
        ifg_vcm[diag_ix * n_ifg + diag_ix] = ifg_var[diag_ix];
    }

    let pool = build_pool(threads)?;
    let row_results = {
        let compute = || {
            (0..n_ps)
                .into_par_iter()
                .map(|row_ix| {
                    let row_offset = row_ix * n_ifg;
                    let mut ph_seq = vec![0.0; unwrap_obs];
                    for obs_ix in 0..unwrap_obs {
                        ph_seq[obs_ix] = ph_proc[row_offset + unwrap_ix[obs_ix + 1]]
                            - ph_proc[row_offset + unwrap_ix[obs_ix]];
                    }
                    let coeff_seq = mat_vec(
                        &inv_seq,
                        &design_rhs(&design_seq, &ph_seq, unwrap_obs, seq_coeff),
                        seq_coeff,
                    );
                    let k = coeff_seq[1];

                    let mut ph_scla_row = vec![0.0_f32; n_ifg];
                    for ifg_ix in 0..n_ifg {
                        ph_scla_row[ifg_ix] = (k * bperp_mat[row_offset + ifg_ix]) as f32;
                    }

                    let c = if let Some(inv_c_ref) = inv_c.as_ref() {
                        let mut resid_weighted = vec![0.0; solve_ix.len()];
                        for (obs_ix, &ifg_ix) in solve_ix.iter().enumerate() {
                            let resid = ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                            resid_weighted[obs_ix] = resid / solve_scales[obs_ix];
                        }
                        mat_vec(
                            inv_c_ref,
                            &design_rhs(&solve_design, &resid_weighted, solve_ix.len(), 2),
                            2,
                        )[0] as f32
                    } else {
                        let mut accum = 0.0;
                        for &ifg_ix in solve_ix {
                            accum += ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                        }
                        (accum / solve_ix.len() as f64) as f32
                    };

                    let mut wy0 = 0.0;
                    let mut wy1 = 0.0;
                    for ifg_ix in 0..n_ifg {
                        let y = ph_mean_v[row_offset + ifg_ix];
                        let weight = weights_mv[ifg_ix];
                        wy0 += y * weight;
                        wy1 += y * weight * time_diff[ifg_ix];
                    }
                    let (intercept, slope) = if det.abs() <= 1.0e-12 {
                        let base = if s0 != 0.0 { wy0 / s0 } else { 0.0 };
                        (base as f32, 0.0_f32)
                    } else {
                        (
                            ((wy0 * s2 - wy1 * s1) / det) as f32,
                            ((wy1 * s0 - wy0 * s1) / det) as f32,
                        )
                    };

                    Stage7RowResult {
                        k,
                        c,
                        ph_scla_row,
                        mean_v: slope,
                        intercept,
                        slope,
                    }
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_ps)
                .map(|row_ix| {
                    let row_offset = row_ix * n_ifg;
                    let mut ph_seq = vec![0.0; unwrap_obs];
                    for obs_ix in 0..unwrap_obs {
                        ph_seq[obs_ix] = ph_proc[row_offset + unwrap_ix[obs_ix + 1]]
                            - ph_proc[row_offset + unwrap_ix[obs_ix]];
                    }
                    let coeff_seq = mat_vec(
                        &inv_seq,
                        &design_rhs(&design_seq, &ph_seq, unwrap_obs, seq_coeff),
                        seq_coeff,
                    );
                    let k = coeff_seq[1];

                    let mut ph_scla_row = vec![0.0_f32; n_ifg];
                    for ifg_ix in 0..n_ifg {
                        ph_scla_row[ifg_ix] = (k * bperp_mat[row_offset + ifg_ix]) as f32;
                    }

                    let c = if let Some(inv_c_ref) = inv_c.as_ref() {
                        let mut resid_weighted = vec![0.0; solve_ix.len()];
                        for (obs_ix, &ifg_ix) in solve_ix.iter().enumerate() {
                            let resid = ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                            resid_weighted[obs_ix] = resid / solve_scales[obs_ix];
                        }
                        mat_vec(
                            inv_c_ref,
                            &design_rhs(&solve_design, &resid_weighted, solve_ix.len(), 2),
                            2,
                        )[0] as f32
                    } else {
                        let mut accum = 0.0;
                        for &ifg_ix in solve_ix {
                            accum += ph_proc[row_offset + ifg_ix] - ph_scla_row[ifg_ix] as f64;
                        }
                        (accum / solve_ix.len() as f64) as f32
                    };

                    let mut wy0 = 0.0;
                    let mut wy1 = 0.0;
                    for ifg_ix in 0..n_ifg {
                        let y = ph_mean_v[row_offset + ifg_ix];
                        let weight = weights_mv[ifg_ix];
                        wy0 += y * weight;
                        wy1 += y * weight * time_diff[ifg_ix];
                    }
                    let (intercept, slope) = if det.abs() <= 1.0e-12 {
                        let base = if s0 != 0.0 { wy0 / s0 } else { 0.0 };
                        (base as f32, 0.0_f32)
                    } else {
                        (
                            ((wy0 * s2 - wy1 * s1) / det) as f32,
                            ((wy1 * s0 - wy0 * s1) / det) as f32,
                        )
                    };

                    Stage7RowResult {
                        k,
                        c,
                        ph_scla_row,
                        mean_v: slope,
                        intercept,
                        slope,
                    }
                })
                .collect::<Vec<_>>(),
        }
    };

    let k_ps_uw: Vec<f64> = row_results.iter().map(|row| row.k).collect();
    let c_ps_uw: Vec<f32> = row_results.iter().map(|row| row.c).collect();
    let mean_v: Vec<f32> = row_results.iter().map(|row| row.mean_v).collect();
    let mut ph_scla = vec![0.0_f32; n_ps * n_ifg];
    let mut m = vec![0.0_f32; 2 * n_ps];
    for (row_ix, row) in row_results.iter().enumerate() {
        ph_scla[row_ix * n_ifg..(row_ix + 1) * n_ifg].copy_from_slice(&row.ph_scla_row);
        m[row_ix] = row.intercept;
        m[n_ps + row_ix] = row.slope;
    }

    Ok(Stage7Outputs {
        k_ps_uw,
        c_ps_uw,
        ph_scla,
        ifg_vcm,
        mean_v,
        m,
        ph_ramp: vec![0.0; n_ps * n_ifg],
    })
}

fn argmax_first(values: &[f64]) -> usize {
    let mut best_ix = 0usize;
    let mut best_value = values.first().copied().unwrap_or(f64::NEG_INFINITY);
    for (idx, &value) in values.iter().enumerate().skip(1) {
        if value > best_value {
            best_ix = idx;
            best_value = value;
        }
    }
    best_ix
}

const STAGE2_TOPOFIT_NEAR_MAX_COH_TOL: f64 = 2.0e-4;

fn near_max_trial_indices(coh_trial: &[f64]) -> Vec<usize> {
    if coh_trial.len() <= 1 {
        return vec![0];
    }

    let mut local_max = vec![false; coh_trial.len()];
    local_max[0] = coh_trial[0] >= coh_trial[1];
    local_max[coh_trial.len() - 1] =
        coh_trial[coh_trial.len() - 1] >= coh_trial[coh_trial.len() - 2];
    if coh_trial.len() > 2 {
        for idx in 1..coh_trial.len() - 1 {
            local_max[idx] =
                coh_trial[idx] >= coh_trial[idx - 1] && coh_trial[idx] >= coh_trial[idx + 1];
        }
    }

    let max_coh = coh_trial.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut candidate_ix = local_max
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_local_max)| {
            if is_local_max && coh_trial[idx] >= max_coh - STAGE2_TOPOFIT_NEAR_MAX_COH_TOL {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if candidate_ix.is_empty() {
        candidate_ix.push(argmax_first(coh_trial));
    }
    candidate_ix
}

fn select_candidate(
    candidate_ix: &[usize],
    candidate_coh: &[f64],
    _refined_coh: &[f64],
    _trial_count: usize,
) -> usize {
    if candidate_ix.is_empty() {
        return 0;
    }

    let coarse_best_local = argmax_first(candidate_coh);
    candidate_ix[coarse_best_local]
}

fn collect_row(cpx_row: &[Complex64], bp_row: &[f64]) -> RowData {
    let mut valid_cols = Vec::with_capacity(cpx_row.len());
    let mut cpx = Vec::with_capacity(cpx_row.len());
    let mut bp = Vec::with_capacity(cpx_row.len());
    let mut weighting = Vec::with_capacity(cpx_row.len());
    let mut wb = Vec::with_capacity(cpx_row.len());

    let mut denom = 0.0;
    let mut den_lin = 0.0;
    let mut bperp_min = 0.0;
    let mut bperp_max = 0.0;
    let mut first_valid = true;

    for (col, (&value, &bp_value)) in cpx_row.iter().zip(bp_row.iter()).enumerate() {
        if value == Complex64::new(0.0, 0.0) {
            continue;
        }
        if first_valid {
            bperp_min = bp_value;
            bperp_max = bp_value;
            first_valid = false;
        } else {
            bperp_min = bperp_min.min(bp_value);
            bperp_max = bperp_max.max(bp_value);
        }

        let weight = value.norm();
        let wb_value = weight * bp_value;

        valid_cols.push(col);
        cpx.push(value);
        bp.push(bp_value);
        weighting.push(weight);
        wb.push(wb_value);
        denom += weight;
        den_lin += wb_value * wb_value;
    }

    if denom == 0.0 {
        denom = 1.0;
    }
    if den_lin == 0.0 {
        den_lin = 1.0;
    }

    let mut bperp_range = if valid_cols.is_empty() {
        1.0
    } else {
        bperp_max - bperp_min
    };
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }

    RowData {
        valid_cols,
        cpx,
        bp,
        weighting,
        wb,
        den_lin,
        bperp_range,
        denom,
        n_col: cpx_row.len(),
    }
}

fn collect_row_single(cpx_row: &[Complex32], bp_row: &[f32]) -> RowDataSingle {
    let mut valid_cols = Vec::with_capacity(cpx_row.len());
    let mut cpx = Vec::with_capacity(cpx_row.len());
    let mut bp = Vec::with_capacity(cpx_row.len());
    let mut weighting = Vec::with_capacity(cpx_row.len());
    let mut wb = Vec::with_capacity(cpx_row.len());

    let mut denom = 0.0_f32;
    let mut den_lin = 0.0_f32;
    let mut bperp_min = 0.0_f32;
    let mut bperp_max = 0.0_f32;
    let mut first_valid = true;

    for (col, (&value, &bp_value)) in cpx_row.iter().zip(bp_row.iter()).enumerate() {
        if value == Complex32::new(0.0, 0.0) {
            continue;
        }
        if first_valid {
            bperp_min = bp_value;
            bperp_max = bp_value;
            first_valid = false;
        } else {
            bperp_min = bperp_min.min(bp_value);
            bperp_max = bperp_max.max(bp_value);
        }

        let weight = value.norm();
        let wb_value = weight * bp_value;

        valid_cols.push(col);
        cpx.push(value);
        bp.push(bp_value);
        weighting.push(weight);
        wb.push(wb_value);
        denom += weight;
        den_lin += wb_value * wb_value;
    }

    if denom == 0.0 {
        denom = 1.0;
    }
    if den_lin == 0.0 {
        den_lin = 1.0;
    }

    let mut bperp_range = if valid_cols.is_empty() {
        1.0
    } else {
        bperp_max - bperp_min
    };
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }

    RowDataSingle {
        valid_cols,
        cpx,
        bp,
        weighting,
        wb,
        den_lin,
        bperp_range,
        denom,
        n_col: cpx_row.len(),
    }
}

fn coherence_trials_generic(row: &RowData, trial_mult: &[f64]) -> Vec<f64> {
    let mut coh_trial = vec![0.0; trial_mult.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        for idx in 0..row.cpx.len() {
            let phase = (row.bp[idx] / row.bperp_range) * QUARTER_PI * trial_value;
            let (sn, cs) = phase.sin_cos();
            let ph_re = row.cpx[idx].re;
            let ph_im = row.cpx[idx].im;
            sum_re += (ph_re * cs) + (ph_im * sn);
            sum_im += (ph_im * cs) - (ph_re * sn);
        }
        coh_trial[trial_ix] = sum_re.hypot(sum_im) / row.denom;
    }
    coh_trial
}

fn coherence_trials_generic_single(row: &RowDataSingle, trial_mult: &[f32]) -> Vec<f32> {
    let mut coh_trial = vec![0.0_f32; trial_mult.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let mut phaser_sum = Complex32::new(0.0, 0.0);
        for idx in 0..row.cpx.len() {
            let phase = (row.bp[idx] / row.bperp_range) * QUARTER_PI_F32 * trial_value;
            let (sn, cs) = phase.sin_cos();
            phaser_sum += row.cpx[idx] * Complex32::new(cs, -sn);
        }
        coh_trial[trial_ix] = phaser_sum.norm() / row.denom;
    }
    coh_trial
}

fn coherence_trials_row_invariant(
    row: &RowData,
    basis: &[Complex64],
    trial_count: usize,
) -> Vec<f64> {
    let mut coh_trial = vec![0.0; trial_count];
    for trial_ix in 0..trial_count {
        let basis_row = &basis[trial_ix * row.n_col..(trial_ix + 1) * row.n_col];
        let mut phaser_sum = Complex64::new(0.0, 0.0);
        for (idx, &col) in row.valid_cols.iter().enumerate() {
            phaser_sum += row.cpx[idx] * basis_row[col];
        }
        coh_trial[trial_ix] = phaser_sum.norm() / row.denom;
    }
    coh_trial
}

fn refine_candidate(row: &RowData, coarse_k0: f64, store_phase: bool) -> RefinedRow {
    let mut offset = Complex64::new(0.0, 0.0);
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        offset += row.cpx[idx] * Complex64::new(cs, -sn);
    }

    let offset_conj = offset.conj();
    let mut mopt_num = 0.0;
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex64::new(cs, -sn);
        let angle = (res * offset_conj).arg();
        mopt_num += row.wb[idx] * (row.weighting[idx] * angle);
    }

    let k = coarse_k0 + (mopt_num / row.den_lin);
    let mut mean_phase_residual = Complex64::new(0.0, 0.0);
    let mut denom2 = 0.0;
    let mut residual = if store_phase {
        vec![Complex32::new(0.0, 0.0); row.n_col]
    } else {
        Vec::new()
    };

    for (idx, &col) in row.valid_cols.iter().enumerate() {
        let phase = k * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex64::new(cs, -sn);
        mean_phase_residual += res;
        denom2 += res.norm();
        if store_phase {
            residual[col] = Complex32::new(res.re as f32, res.im as f32);
        }
    }

    if denom2 == 0.0 {
        denom2 = 1.0;
    }

    RefinedRow {
        k,
        c: mean_phase_residual.arg(),
        coh: mean_phase_residual.norm() / denom2,
        residual,
    }
}

fn refine_candidate_single(row: &RowDataSingle, coarse_k0: f32, store_phase: bool) -> RefinedRow {
    let mut offset = Complex32::new(0.0, 0.0);
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        offset += row.cpx[idx] * Complex32::new(cs, -sn);
    }

    let offset_conj = offset.conj();
    let mut mopt_num = 0.0_f32;
    for idx in 0..row.cpx.len() {
        let phase = coarse_k0 * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex32::new(cs, -sn);
        let angle = (res * offset_conj).arg();
        mopt_num += row.wb[idx] * (row.weighting[idx] * angle);
    }

    let k = coarse_k0 + (mopt_num / row.den_lin);
    let mut mean_phase_residual = Complex32::new(0.0, 0.0);
    let mut denom2 = 0.0_f32;
    let mut residual = if store_phase {
        vec![Complex32::new(0.0, 0.0); row.n_col]
    } else {
        Vec::new()
    };

    for (idx, &col) in row.valid_cols.iter().enumerate() {
        let phase = k * row.bp[idx];
        let (sn, cs) = phase.sin_cos();
        let res = row.cpx[idx] * Complex32::new(cs, -sn);
        mean_phase_residual += res;
        denom2 += res.norm();
        if store_phase {
            residual[col] = res;
        }
    }

    if denom2 == 0.0 {
        denom2 = 1.0;
    }

    RefinedRow {
        k: k as f64,
        c: mean_phase_residual.arg() as f64,
        coh: (mean_phase_residual.norm() / denom2) as f64,
        residual,
    }
}

fn solve_row_generic(
    cpx_row: &[Complex64],
    bp_row: &[f64],
    trial_mult: &[f64],
    store_phase: bool,
) -> RefinedRow {
    let row = collect_row(cpx_row, bp_row);
    if row.cpx.is_empty() {
        return RefinedRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx_row.len()],
        };
    }

    let coh_trial = coherence_trials_generic(&row, trial_mult);
    solve_row_from_trials(&row, trial_mult, &coh_trial, store_phase)
}

fn solve_row_generic_single(
    cpx_row: &[Complex32],
    bp_row: &[f32],
    trial_mult: &[f32],
    store_phase: bool,
) -> RefinedRow {
    let row = collect_row_single(cpx_row, bp_row);
    if row.cpx.is_empty() {
        return RefinedRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx_row.len()],
        };
    }

    let coh_trial = coherence_trials_generic_single(&row, trial_mult);
    let coh_trial_f64: Vec<f64> = coh_trial.iter().map(|&value| value as f64).collect();
    let candidate_ix = near_max_trial_indices(&coh_trial_f64);
    if candidate_ix.len() == 1 {
        let coarse_k0 = QUARTER_PI_F32 / row.bperp_range * trial_mult[candidate_ix[0]];
        return refine_candidate_single(&row, coarse_k0, store_phase);
    }

    let mut refined = Vec::with_capacity(candidate_ix.len());
    let mut candidate_coh = Vec::with_capacity(candidate_ix.len());
    for &trial_ix in &candidate_ix {
        let coarse_k0 = QUARTER_PI_F32 / row.bperp_range * trial_mult[trial_ix];
        refined.push(refine_candidate_single(&row, coarse_k0, store_phase));
        candidate_coh.push(coh_trial_f64[trial_ix]);
    }
    let refined_coh = refined.iter().map(|row| row.coh).collect::<Vec<_>>();
    let selected_trial_ix = select_candidate(
        &candidate_ix,
        &candidate_coh,
        &refined_coh,
        trial_mult.len(),
    );
    let selected_local_ix = candidate_ix
        .iter()
        .position(|&trial_ix| trial_ix == selected_trial_ix)
        .unwrap_or(0);
    refined.remove(selected_local_ix)
}

fn solve_row_row_invariant(
    cpx_row: &[Complex64],
    bp_vec: &[f64],
    trial_mult: &[f64],
    basis: &[Complex64],
    store_phase: bool,
) -> RefinedRow {
    let row = collect_row(cpx_row, bp_vec);
    if row.cpx.is_empty() {
        return RefinedRow {
            k: f64::NAN,
            c: f64::NAN,
            coh: f64::NAN,
            residual: vec![Complex32::new(0.0, 0.0); cpx_row.len()],
        };
    }

    let coh_trial = coherence_trials_row_invariant(&row, basis, trial_mult.len());
    solve_row_from_trials(&row, trial_mult, &coh_trial, store_phase)
}

fn solve_row_from_trials(
    row: &RowData,
    trial_mult: &[f64],
    coh_trial: &[f64],
    store_phase: bool,
) -> RefinedRow {
    let candidate_ix = near_max_trial_indices(coh_trial);
    if candidate_ix.len() == 1 {
        let coarse_k0 = QUARTER_PI / row.bperp_range * trial_mult[candidate_ix[0]];
        return refine_candidate(row, coarse_k0, store_phase);
    }

    let mut refined = Vec::with_capacity(candidate_ix.len());
    let mut candidate_coh = Vec::with_capacity(candidate_ix.len());
    for &trial_ix in &candidate_ix {
        let coarse_k0 = QUARTER_PI / row.bperp_range * trial_mult[trial_ix];
        refined.push(refine_candidate(row, coarse_k0, store_phase));
        candidate_coh.push(coh_trial[trial_ix]);
    }
    let refined_coh = refined.iter().map(|row| row.coh).collect::<Vec<_>>();
    let selected_trial_ix = select_candidate(
        &candidate_ix,
        &candidate_coh,
        &refined_coh,
        trial_mult.len(),
    );
    let selected_local_ix = candidate_ix
        .iter()
        .position(|&trial_ix| trial_ix == selected_trial_ix)
        .unwrap_or(0);
    refined.remove(selected_local_ix)
}

fn row_invariant_basis(bp_vec: &[f64], trial_mult: &[f64]) -> Vec<Complex64> {
    let mut bperp_min = bp_vec[0];
    let mut bperp_max = bp_vec[0];
    for &value in bp_vec.iter().skip(1) {
        bperp_min = bperp_min.min(value);
        bperp_max = bperp_max.max(value);
    }
    let mut bperp_range = bperp_max - bperp_min;
    if bperp_range == 0.0 {
        bperp_range = 1.0;
    }

    let mut basis = vec![Complex64::new(0.0, 0.0); trial_mult.len() * bp_vec.len()];
    for (trial_ix, &trial_value) in trial_mult.iter().enumerate() {
        let basis_row = &mut basis[trial_ix * bp_vec.len()..(trial_ix + 1) * bp_vec.len()];
        for (col, &bp_value) in bp_vec.iter().enumerate() {
            let phase = (bp_value / bperp_range) * QUARTER_PI * trial_value;
            let (sn, cs) = phase.sin_cos();
            basis_row[col] = Complex64::new(cs, -sn);
        }
    }
    basis
}

#[pyfunction]
fn accumulate_weighted_grid<'py>(
    py: Python<'py>,
    ph_weight: PyReadonlyArray2<Complex32>,
    grid_lin: PyReadonlyArray1<i64>,
    n_i: usize,
    n_j: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray3<Complex32>>> {
    let ph_view = ph_weight.as_array();
    let grid_view = grid_lin.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph_weight must be a 2-D matrix"));
    }
    if grid_view.len() != ph_view.shape()[0] {
        return Err(PyValueError::new_err(
            "grid_lin length must match ph_weight row count",
        ));
    }

    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_weight must be C-contiguous"))?;
    let grid_slice = grid_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("grid_lin must be contiguous"))?;
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    let grid_size = n_i * n_j;
    let pool = build_pool(threads)?;

    let columns = py.detach(move || {
        let compute = || {
            (0..n_ifg)
                .into_par_iter()
                .map(|ifg_ix| {
                    let mut column = vec![Complex32::new(0.0, 0.0); grid_size];
                    for row_ix in 0..n_ps {
                        let grid_ix = grid_slice[row_ix];
                        if grid_ix >= 0 && (grid_ix as usize) < grid_size {
                            column[grid_ix as usize] += ph_slice[row_ix * n_ifg + ifg_ix];
                        }
                    }
                    column
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_ifg)
                .map(|ifg_ix| {
                    let mut column = vec![Complex32::new(0.0, 0.0); grid_size];
                    for row_ix in 0..n_ps {
                        let grid_ix = grid_slice[row_ix];
                        if grid_ix >= 0 && (grid_ix as usize) < grid_size {
                            column[grid_ix as usize] += ph_slice[row_ix * n_ifg + ifg_ix];
                        }
                    }
                    column
                })
                .collect::<Vec<_>>(),
        }
    });

    let mut out = vec![Complex32::new(0.0, 0.0); grid_size * n_ifg];
    for ifg_ix in 0..n_ifg {
        for grid_ix in 0..grid_size {
            out[grid_ix * n_ifg + ifg_ix] = columns[ifg_ix][grid_ix];
        }
    }

    let array = Array3::from_shape_vec((n_i, n_j, n_ifg), out)
        .map_err(|err| PyValueError::new_err(format!("failed to build grid output: {err}")))?;
    Ok(array.into_pyarray(py))
}

fn ps_topofit_batch_generic_f64_impl<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp: PyReadonlyArray2<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp.as_array();
    if cpx_view.ndim() != 2 || bp_view.ndim() != 2 || cpx_view.shape() != bp_view.shape() {
        return Err(PyValueError::new_err(
            "ps_topofit batch expects cpxphase and bperp with matching 2-D shapes",
        ));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be C-contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    let trial_mult = trial_values(n_trial_wraps);
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>(),
        }
    });

    let k_values: Vec<f64> = rows.iter().map(|row| row.k).collect();
    let c_values: Vec<f64> = rows.iter().map(|row| row.c).collect();
    let coh_values: Vec<f64> = rows.iter().map(|row| row.coh).collect();
    let residual: Vec<Complex32> = rows.into_iter().flat_map(|row| row.residual).collect();

    let residual_array = Array2::from_shape_vec((n_row, n_col), residual).map_err(|err| {
        PyValueError::new_err(format!("failed to build topofit residual output: {err}"))
    })?;
    Ok((
        Array1::from_vec(k_values).into_pyarray(py),
        Array1::from_vec(c_values).into_pyarray(py),
        Array1::from_vec(coh_values).into_pyarray(py),
        residual_array.into_pyarray(py),
    ))
}

#[pyfunction]
fn ps_topofit_batch_generic_f32<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex32>,
    bperp: PyReadonlyArray2<f32>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp.as_array();
    if cpx_view.ndim() != 2 || bp_view.ndim() != 2 || cpx_view.shape() != bp_view.shape() {
        return Err(PyValueError::new_err(
            "ps_topofit batch expects cpxphase and bperp with matching 2-D shapes",
        ));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be C-contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    let trial_mult: Vec<f32> = trial_values(n_trial_wraps)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic_single(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    let bp_row = &bp_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_generic_single(cpx_row, bp_row, &trial_mult, true)
                })
                .collect::<Vec<_>>(),
        }
    });

    let k_values: Vec<f64> = rows.iter().map(|row| row.k).collect();
    let c_values: Vec<f64> = rows.iter().map(|row| row.c).collect();
    let coh_values: Vec<f64> = rows.iter().map(|row| row.coh).collect();
    let residual: Vec<Complex32> = rows.into_iter().flat_map(|row| row.residual).collect();

    let residual_array = Array2::from_shape_vec((n_row, n_col), residual).map_err(|err| {
        PyValueError::new_err(format!("failed to build topofit residual output: {err}"))
    })?;
    Ok((
        Array1::from_vec(k_values).into_pyarray(py),
        Array1::from_vec(c_values).into_pyarray(py),
        Array1::from_vec(coh_values).into_pyarray(py),
        residual_array.into_pyarray(py),
    ))
}

#[pyfunction]
fn ps_topofit_batch_generic<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp: PyReadonlyArray2<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    ps_topofit_batch_generic_f64_impl(py, cpxphase, bperp, n_trial_wraps, threads)
}

#[pyfunction]
fn ps_topofit_batch_row_invariant<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp_vec: PyReadonlyArray1<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex32>>,
)> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp_vec.as_array();
    if cpx_view.ndim() != 2 {
        return Err(PyValueError::new_err("cpxphase must be a 2-D matrix"));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp vector must be contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    if bp_view.len() != n_col {
        return Err(PyValueError::new_err(
            "row-invariant bperp vector length must match cpxphase width",
        ));
    }

    let trial_mult = trial_values(n_trial_wraps);
    let basis = row_invariant_basis(bp_slice, &trial_mult);
    let pool = build_pool(threads)?;

    let rows = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, true)
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, true)
                })
                .collect::<Vec<_>>(),
        }
    });

    let k_values: Vec<f64> = rows.iter().map(|row| row.k).collect();
    let c_values: Vec<f64> = rows.iter().map(|row| row.c).collect();
    let coh_values: Vec<f64> = rows.iter().map(|row| row.coh).collect();
    let residual: Vec<Complex32> = rows.into_iter().flat_map(|row| row.residual).collect();

    let residual_array = Array2::from_shape_vec((n_row, n_col), residual).map_err(|err| {
        PyValueError::new_err(format!(
            "failed to build row-invariant residual output: {err}"
        ))
    })?;
    Ok((
        Array1::from_vec(k_values).into_pyarray(py),
        Array1::from_vec(c_values).into_pyarray(py),
        Array1::from_vec(coh_values).into_pyarray(py),
        residual_array.into_pyarray(py),
    ))
}

#[pyfunction]
fn ps_topofit_coh_row_invariant<'py>(
    py: Python<'py>,
    cpxphase: PyReadonlyArray2<Complex64>,
    bperp_vec: PyReadonlyArray1<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cpx_view = cpxphase.as_array();
    let bp_view = bperp_vec.as_array();
    if cpx_view.ndim() != 2 {
        return Err(PyValueError::new_err("cpxphase must be a 2-D matrix"));
    }

    let cpx_slice = cpx_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("cpxphase must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp vector must be contiguous"))?;
    let n_row = cpx_view.shape()[0];
    let n_col = cpx_view.shape()[1];
    if bp_view.len() != n_col {
        return Err(PyValueError::new_err(
            "row-invariant bperp vector length must match cpxphase width",
        ));
    }

    let trial_mult = trial_values(n_trial_wraps);
    let basis = row_invariant_basis(bp_slice, &trial_mult);
    let pool = build_pool(threads)?;

    let coh_values = py.detach(move || {
        let compute = || {
            (0..n_row)
                .into_par_iter()
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, false).coh
                })
                .collect::<Vec<_>>()
        };
        match pool {
            Some(pool) => pool.install(compute),
            None => (0..n_row)
                .map(|row_ix| {
                    let cpx_row = &cpx_slice[row_ix * n_col..(row_ix + 1) * n_col];
                    solve_row_row_invariant(cpx_row, bp_slice, &trial_mult, &basis, false).coh
                })
                .collect::<Vec<_>>(),
        }
    });

    Ok(Array1::from_vec(coh_values).into_pyarray(py))
}

#[pyfunction]
fn histogram_with_centers<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f64>,
    centers: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let value_view = values.as_array();
    let center_view = centers.as_array();
    let value_slice = value_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("values must be contiguous"))?;
    let center_slice = center_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("centers must be contiguous"))?;
    let mut out = vec![0.0_f64; center_slice.len()];

    if center_slice.is_empty() {
        return Ok(Array1::from_vec(out).into_pyarray(py));
    }
    if center_slice.len() == 1 {
        out[0] = value_slice.len() as f64;
        return Ok(Array1::from_vec(out).into_pyarray(py));
    }

    let diffs: Vec<f64> = center_slice
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let max_abs_center = center_slice
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    let equal_spacing = diffs
        .iter()
        .all(|&diff| (diff - diffs[0]).abs() <= f64::EPSILON * (1.0_f64).max(max_abs_center));
    if equal_spacing {
        let d = if center_slice.len() < 3 {
            1.0_f64
        } else {
            (center_slice[center_slice.len() - 1] - center_slice[0])
                / ((center_slice.len() - 1) as f64)
        };
        let cutoff0 = (center_slice[0] + center_slice[1]) / 2.0;
        let max_bin = (center_slice.len() - 1) as f64;
        for &value in value_slice {
            if !value.is_finite() {
                continue;
            }
            let assignment = 1.0 + ((value - cutoff0) / d).ceil().clamp(0.0, max_bin);
            out[(assignment as usize) - 1] += 1.0;
        }
        return Ok(Array1::from_vec(out).into_pyarray(py));
    }

    let mids: Vec<f64> = center_slice
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) / 2.0)
        .collect();
    for &value in value_slice {
        if !value.is_finite() {
            continue;
        }
        let mut lo = 0usize;
        let mut hi = center_slice.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if mids[mid] < value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[lo] += 1.0;
    }

    Ok(Array1::from_vec(out).into_pyarray(py))
}

#[pyfunction]
fn stage2_clap_filter_kernel<'py>(
    py: Python<'py>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let _ = threads;
    let alpha = 2.5_f64;
    let std = (7.0_f64 - 1.0) / (2.0 * alpha);
    let center = (7.0_f64 - 1.0) / 2.0;
    let mut g = [0.0_f64; 7];
    for (idx, value) in g.iter_mut().enumerate() {
        let x = (idx as f64 - center) / std;
        *value = (-0.5 * x * x).exp();
    }

    let mut out = Vec::with_capacity(49);
    for row in 0..7 {
        for col in 0..7 {
            out.push(g[row] * g[col]);
        }
    }

    Ok(Array2::from_shape_vec((7, 7), out)
        .map_err(|err| {
            PyValueError::new_err(format!("failed to build stage2 clap filter kernel: {err}"))
        })?
        .into_pyarray(py))
}

#[pyfunction]
fn stage2_ph_weight_block<'py>(
    py: Python<'py>,
    ph_nm: PyReadonlyArray2<Complex32>,
    bperp: PyReadonlyArray2<f64>,
    k_ps: PyReadonlyArray1<f64>,
    weighting: PyReadonlyArray1<f64>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let _ = threads;
    let ph_view = ph_nm.as_array();
    let bp_view = bperp.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage2_ph_weight_block expects a 2-D ph_nm matrix",
        ));
    }
    if bp_view.shape() != ph_view.shape() {
        return Err(PyValueError::new_err(
            "stage2_ph_weight_block expects bperp shape to match ph_nm",
        ));
    }
    let n_row = ph_view.shape()[0];
    let n_col = ph_view.shape()[1];
    let k_view = k_ps.as_array();
    let weight_view = weighting.as_array();
    if k_view.len() != n_row || weight_view.len() != n_row {
        return Err(PyValueError::new_err(
            "stage2_ph_weight_block expects k_ps and weighting length to match ph_nm rows",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_nm must be C-contiguous"))?;
    let bp_slice = bp_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be C-contiguous"))?;
    let k_slice = k_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("k_ps must be contiguous"))?;
    let weight_slice = weight_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("weighting must be contiguous"))?;

    let mut out = Vec::with_capacity(n_row * n_col);
    for row in 0..n_row {
        let k = k_slice[row];
        let weight = weight_slice[row];
        for col in 0..n_col {
            let idx = row * n_col + col;
            let angle = bp_slice[idx] * k;
            let (sin_v, cos_v) = angle.sin_cos();
            let value = ph_slice[idx];
            let real = (f64::from(value.re) * cos_v + f64::from(value.im) * sin_v) * weight;
            let imag = (f64::from(value.im) * cos_v - f64::from(value.re) * sin_v) * weight;
            out.push(Complex32::new(real as f32, imag as f32));
        }
    }

    Ok(Array2::from_shape_vec((n_row, n_col), out)
        .map_err(|err| {
            PyValueError::new_err(format!("failed to build stage2 ph_weight output: {err}"))
        })?
        .into_pyarray(py))
}

#[pyfunction]
fn stage2_grid_indices<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<f32>,
    grid_size: f32,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let _ = threads;
    let xy_view = xy.as_array();
    if xy_view.ndim() != 2 || xy_view.shape()[1] < 3 {
        return Err(PyValueError::new_err(
            "stage2_grid_indices expects an xy matrix with at least 3 columns",
        ));
    }
    let n_row = xy_view.shape()[0];
    if n_row == 0 {
        return Err(PyValueError::new_err(
            "stage2_grid_indices expects at least one xy row",
        ));
    }
    if !grid_size.is_finite() || grid_size == 0.0 {
        return Err(PyValueError::new_err(
            "stage2_grid_indices expects a finite non-zero grid_size",
        ));
    }
    let n_col = xy_view.shape()[1];
    let xy_slice = xy_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("xy must be C-contiguous"))?;

    let mut x_min = xy_slice[1];
    let mut y_min = xy_slice[2];
    for row in 1..n_row {
        let base = row * n_col;
        x_min = x_min.min(xy_slice[base + 1]);
        y_min = y_min.min(xy_slice[base + 2]);
    }

    let eps = 1e-6_f32;
    let mut grid_i = Vec::with_capacity(n_row);
    let mut grid_j = Vec::with_capacity(n_row);
    let mut max_i = i64::MIN;
    let mut max_j = i64::MIN;
    for row in 0..n_row {
        let base = row * n_col;
        let i = ((xy_slice[base + 2] - y_min + eps) / grid_size).ceil() as i64;
        let j = ((xy_slice[base + 1] - x_min + eps) / grid_size).ceil() as i64;
        max_i = max_i.max(i);
        max_j = max_j.max(j);
        grid_i.push(i);
        grid_j.push(j);
    }

    if max_i > 1 {
        for value in &mut grid_i {
            if *value == max_i {
                *value = max_i - 1;
            }
        }
    }
    if max_j > 1 {
        for value in &mut grid_j {
            if *value == max_j {
                *value = max_j - 1;
            }
        }
    }

    let mut out = Vec::with_capacity(n_row * 2);
    for row in 0..n_row {
        out.push(grid_i[row].max(1) as f32);
        out.push(grid_j[row].max(1) as f32);
    }

    Ok(Array2::from_shape_vec((n_row, 2), out)
        .map_err(|err| {
            PyValueError::new_err(format!("failed to build stage2 grid indices output: {err}"))
        })?
        .into_pyarray(py))
}

#[pyfunction]
fn stage2_normalize_complex<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<Complex32>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    let _ = threads;
    let view = values.as_array();
    if view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage2_normalize_complex expects a 2-D complex matrix",
        ));
    }
    let n_row = view.shape()[0];
    let n_col = view.shape()[1];
    let slice = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("values must be C-contiguous"))?;

    let mut out = Vec::with_capacity(n_row * n_col);
    for &value in slice {
        let mag = (value.re * value.re + value.im * value.im).sqrt();
        if mag != 0.0 {
            out.push(Complex32::new(value.re / mag, value.im / mag));
        } else {
            out.push(value);
        }
    }

    Ok(Array2::from_shape_vec((n_row, n_col), out)
        .map_err(|err| {
            PyValueError::new_err(format!("failed to build stage2 normalized output: {err}"))
        })?
        .into_pyarray(py))
}

#[pyfunction]
fn stage2_normalize_phase_matrix<'py>(
    py: Python<'py>,
    ph_nm: PyReadonlyArray2<Complex32>,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = threads;
    let view = ph_nm.as_array();
    if view.ndim() != 2 {
        return Err(PyValueError::new_err(
            "stage2_normalize_phase_matrix expects a 2-D complex matrix",
        ));
    }
    let n_row = view.shape()[0];
    let n_col = view.shape()[1];
    let slice = view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_nm must be C-contiguous"))?;

    let mut ph_out = Vec::with_capacity(n_row * n_col);
    let mut amp_out = Vec::with_capacity(n_row * n_col);
    for &value in slice {
        let mut amp = (value.re * value.re + value.im * value.im).sqrt();
        if amp == 0.0 {
            amp = 1.0;
        }
        amp_out.push(amp);
        ph_out.push(Complex32::new(value.re / amp, value.im / amp));
    }

    let dict = PyDict::new(py);
    dict.set_item(
        "ph",
        Array2::from_shape_vec((n_row, n_col), ph_out)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to build stage2 normalized phase output: {err}"
                ))
            })?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "amp",
        Array2::from_shape_vec((n_row, n_col), amp_out)
            .map_err(|err| {
                PyValueError::new_err(format!("failed to build stage2 amplitude output: {err}"))
            })?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[pymodule]
fn _stage2_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accumulate_weighted_grid, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_batch_generic, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_batch_generic_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_batch_row_invariant, m)?)?;
    m.add_function(wrap_pyfunction!(ps_topofit_coh_row_invariant, m)?)?;
    m.add_function(wrap_pyfunction!(histogram_with_centers, m)?)?;
    m.add_function(wrap_pyfunction!(stage2_clap_filter_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(stage2_ph_weight_block, m)?)?;
    m.add_function(wrap_pyfunction!(stage2_grid_indices, m)?)?;
    m.add_function(wrap_pyfunction!(stage2_normalize_complex, m)?)?;
    m.add_function(wrap_pyfunction!(stage2_normalize_phase_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(
        weighted_fit_native::weighted_affine_fit,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        weighted_fit_native::weighted_slope_fit_real,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        weighted_fit_native::weighted_slope_fit_complex,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage3_native::stage3_select_ifg_index, m)?)?;
    m.add_function(wrap_pyfunction!(stage3_native::stage3_clap_filt_patch, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage3_clap_stack::stage3_clap_filt_patch_stack,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage3_native::stage3_clap_filt_grid, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage3_native::stage3_clap_filt_grid_stack,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage3_native::stage3_wrap_filt, m)?)?;
    m.add_function(wrap_pyfunction!(stage3_native::stage3_wrap_filt_global, m)?)?;
    m.add_function(wrap_pyfunction!(stage3_native::stage3_coh_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(stage4_native::stage4_duplicate_keep, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage4_native::stage4_adjacent_component_keep,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage4_native::stage4_weed_ifg_index, m)?)?;
    m.add_function(wrap_pyfunction!(stage4_native::stage4_phase_correction, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage4_edge_stats_native::stage4_edge_stats,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage5_native::stage5_ifg_std, m)?)?;
    m.add_function(wrap_pyfunction!(stage5_native::stage5_duplicate_keep, m)?)?;
    m.add_function(wrap_pyfunction!(stage5_native::stage5_rc2_correction, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage5_native::stage5_format_merged_rc2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage5_native::stage5_patch_keep_mask, m)?)?;
    m.add_function(wrap_pyfunction!(stage6_native::stage6_unwrap_grid, m)?)?;
    m.add_function(wrap_pyfunction!(stage6_native::stage6_unwrap_ifg_sets, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage6_native::stage6_single_master_ifg_geometry,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage6_native::stage6_grid_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage6_native::stage6_extract_grid_values,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stage6_native::stage6_prepare_cost_offsets,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stage6_native::stage6_reconstruct_ps_phase,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage6_native::stage6_ps_grid_indices, m)?)?;
    m.add_function(wrap_pyfunction!(stage6_native::stage6_select_ifgw, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage6_la_native::stage6_estimate_la_error_single_master,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stage6_smooth_native::stage6_smooth_3d_full_single_master,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stage7_native::stage7_mean_velocity_fit,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stage7_native::stage7_center_to_reference,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stage7_deramp_native::stage7_deramp_unwrapped_phase,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage7_native::stage7_scla_smooth, m)?)?;
    m.add_function(wrap_pyfunction!(stage7_scla_native::stage7_scla, m)?)?;
    m.add_function(wrap_pyfunction!(stage7_scla_native::stage7_scla_parity, m)?)?;
    m.add_function(wrap_pyfunction!(
        stage8_native::stage8_weighted_lstsq_diagonal,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        stage8_native::stage8_weighted_lstsq_full,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(stage8_native::stage8_edge_noise, m)?)?;
    Ok(())
}
