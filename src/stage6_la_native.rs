use num_complex::{Complex32, Complex64};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::f64::consts::PI;

fn insert_master_ix(day: &[f64]) -> usize {
    let mut best_ix: Option<usize> = None;
    for (ix, value) in day.iter().enumerate() {
        if *value > 0.0 {
            match best_ix {
                Some(prev) if day[prev] <= *value => {}
                _ => best_ix = Some(ix),
            }
        }
    }
    best_ix.unwrap_or_else(|| day.len().saturating_sub(1))
}

fn temp_value(row: &[Complex32], insert_ix: usize, mean_abs: f64, col: usize) -> Complex64 {
    if col < insert_ix {
        Complex64::new(f64::from(row[col].re), f64::from(row[col].im))
    } else if col == insert_ix {
        Complex64::new(mean_abs, 0.0)
    } else {
        Complex64::new(f64::from(row[col - 1].re), f64::from(row[col - 1].im))
    }
}

fn estimate_row(
    row: &[Complex32],
    insert_ix: usize,
    selected_ix: &[usize],
    bperp_diff: &[f64],
    safe_range: f64,
    trial_mult: &[i32],
    trial_phase: &[f64],
) -> (f32, f32) {
    if bperp_diff.is_empty() || trial_mult.is_empty() {
        return (0.0, 0.0);
    }
    let mean_abs = row
        .iter()
        .map(|value| f64::from((value.re.mul_add(value.re, value.im * value.im)).sqrt()))
        .sum::<f64>()
        / row.len().max(1) as f64;

    let mut cpx_full = Vec::with_capacity(row.len());
    for col in 0..row.len() {
        let right = temp_value(row, insert_ix, mean_abs, col + 1);
        let left = temp_value(row, insert_ix, mean_abs, col);
        let mut cpx = right * left.conj();
        let amp = cpx.norm();
        if amp != 0.0 {
            cpx /= amp;
        } else {
            cpx = Complex64::new(0.0, 0.0);
        }
        cpx_full.push(cpx);
    }
    let selected: Vec<Complex64> = selected_ix.iter().map(|ix| cpx_full[*ix]).collect();

    let denom = selected.iter().map(|value| value.norm()).sum::<f64>();
    if denom == 0.0 {
        return (0.0, 0.0);
    }

    let mut row_trial = Vec::with_capacity(trial_mult.len());
    for trial in trial_mult {
        let mut phaser_sum = Complex64::new(0.0, 0.0);
        for (cpx, phase) in selected.iter().zip(trial_phase) {
            let angle = -phase * f64::from(*trial);
            phaser_sum += *cpx * Complex64::new(angle.cos(), angle.sin());
        }
        row_trial.push(phaser_sum.norm() / denom);
    }

    let (coh_max_ix, coh_max) = row_trial
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, 0.0));

    let mut peak_start_ix = 0_usize;
    for ix in 0..coh_max_ix {
        if row_trial[ix + 1] - row_trial[ix] < 0.0 {
            peak_start_ix = ix + 1;
        }
    }
    let mut peak_end_ix = row_trial.len() - 1;
    for ix in coh_max_ix..(row_trial.len() - 1) {
        if row_trial[ix + 1] - row_trial[ix] > 0.0 {
            peak_end_ix = ix;
            break;
        }
    }
    let next_peak = row_trial
        .iter()
        .enumerate()
        .filter_map(|(ix, value)| {
            if ix >= peak_start_ix && ix <= peak_end_ix {
                None
            } else {
                Some(*value)
            }
        })
        .fold(0.0_f64, f64::max);
    if coh_max - next_peak <= 0.1 {
        return (0.0, 0.0);
    }

    let k0 = (PI / 4.0 / safe_range) * f64::from(trial_mult[coh_max_ix]);
    let mut offset_phase = Complex64::new(0.0, 0.0);
    let mut resphase = Vec::with_capacity(selected.len());
    for (cpx, bp) in selected.iter().zip(bperp_diff) {
        let angle = -(k0 * *bp);
        let value = *cpx * Complex64::new(angle.cos(), angle.sin());
        offset_phase += value;
        resphase.push(value);
    }

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for ((cpx, bp), res) in selected.iter().zip(bperp_diff).zip(resphase) {
        let residual = res * offset_phase.conj();
        let resphase_angle = residual.im.atan2(residual.re);
        let weight = cpx.norm();
        let wb = weight * *bp;
        den += wb * wb;
        num += wb * (weight * resphase_angle);
    }
    let kval = k0 + if den != 0.0 { num / den } else { 0.0 };
    let mut phase_sum = Complex64::new(0.0, 0.0);
    let mut phase_abs_sum = 0.0_f64;
    for (cpx, bp) in selected.iter().zip(bperp_diff) {
        let angle = -(kval * *bp);
        let value = *cpx * Complex64::new(angle.cos(), angle.sin());
        phase_sum += value;
        phase_abs_sum += value.norm();
    }
    let coh = if phase_abs_sum != 0.0 {
        phase_sum.norm() / phase_abs_sum
    } else {
        0.0
    };
    (kval as f32, coh as f32)
}

#[pyfunction(signature = (dph_space, day, bperp, n_trial_wraps = 200.0, threads = 0))]
pub fn stage6_estimate_la_error_single_master<'py>(
    py: Python<'py>,
    dph_space: PyReadonlyArray2<Complex32>,
    day: PyReadonlyArray1<f64>,
    bperp: PyReadonlyArray1<f64>,
    n_trial_wraps: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let _ = threads;
    let dph = dph_space.as_array();
    let day_view = day.as_array();
    let bperp_view = bperp.as_array();
    let shape = dph.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "stage6_estimate_la_error expects a 2-D dph_space array",
        ));
    }
    let n_edge = shape[0];
    let n_ifg = shape[1];
    if n_ifg == 0 || day_view.len() != n_ifg || bperp_view.len() != n_ifg {
        return Err(PyValueError::new_err(
            "stage6_estimate_la_error expects day/bperp aligned with dph_space columns",
        ));
    }
    if n_edge == 0 {
        return Ok(Vec::<f32>::new().into_pyarray(py));
    }

    let day_vec = day_view.to_vec();
    let bperp_vec = bperp_view.to_vec();
    let insert_ix = insert_master_ix(&day_vec);
    let mut bperp_master = Vec::with_capacity(n_ifg + 1);
    bperp_master.extend_from_slice(&bperp_vec[..insert_ix]);
    bperp_master.push(0.0);
    bperp_master.extend_from_slice(&bperp_vec[insert_ix..]);
    let mut bperp_diff_full = Vec::with_capacity(n_ifg);
    for ix in 0..n_ifg {
        bperp_diff_full.push(bperp_master[ix + 1] - bperp_master[ix]);
    }
    let bperp_range_orig = bperp_vec.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - bperp_vec.iter().copied().fold(f64::INFINITY, f64::min);
    let bperp_range_full = bperp_diff_full
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        - bperp_diff_full
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
    let mut n_trial_wraps_sub = n_trial_wraps;
    if bperp_range_orig != 0.0 {
        n_trial_wraps_sub *= bperp_range_full / bperp_range_orig;
    }
    let selected_ix: Vec<usize> = bperp_diff_full
        .iter()
        .enumerate()
        .filter_map(|(ix, value)| if *value != 0.0 { Some(ix) } else { None })
        .collect();
    let bperp_diff: Vec<f64> = selected_ix.iter().map(|ix| bperp_diff_full[*ix]).collect();
    let safe_range = bperp_range_full.max(1.0e-12);
    let trial_limit = (8.0 * n_trial_wraps_sub).ceil() as i32;
    let trial_mult: Vec<i32> = (-trial_limit..=trial_limit).collect();
    let trial_phase: Vec<f64> = bperp_diff
        .iter()
        .map(|value| value / safe_range * PI / 4.0)
        .collect();

    let mut out = Vec::with_capacity(n_edge);
    for edge in 0..n_edge {
        let row_full: Vec<Complex32> = dph.row(edge).to_vec();
        let (kval, coh) = estimate_row(
            &row_full,
            insert_ix,
            &selected_ix,
            &bperp_diff,
            safe_range,
            &trial_mult,
            &trial_phase,
        );
        out.push(if coh < 0.31 { 0.0 } else { kval });
    }
    Ok(out.into_pyarray(py))
}
