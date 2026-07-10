use num_complex::{Complex32, Complex64};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

fn wrap_pi(value: f64) -> f64 {
    (value + PI).rem_euclid(TWO_PI) - PI
}

fn close_master_ix(day: &[f64]) -> Vec<usize> {
    if day.is_empty() {
        return Vec::new();
    }
    let mut best_ix: Option<usize> = None;
    for (ix, value) in day.iter().enumerate() {
        if *value > 0.0 {
            match best_ix {
                Some(prev) if day[prev] <= *value => {}
                _ => best_ix = Some(ix),
            }
        }
    }
    let insert_ix = best_ix.unwrap_or(day.len() - 1);
    if insert_ix > 0 {
        vec![insert_ix - 1, insert_ix]
    } else {
        vec![insert_ix]
    }
}

fn weighted_affine_intercept(time_diff: &[f64], y: &[f64], weight: &[f64]) -> f64 {
    let mut s0 = 0.0_f64;
    let mut s1 = 0.0_f64;
    let mut s2 = 0.0_f64;
    let mut wy0 = 0.0_f64;
    let mut wy1 = 0.0_f64;
    for ((t, val), w) in time_diff.iter().zip(y).zip(weight) {
        s0 += *w;
        s1 += *w * *t;
        s2 += *w * *t * *t;
        wy0 += *val * *w;
        wy1 += *val * *w * *t;
    }
    let det = s0 * s2 - s1 * s1;
    if det == 0.0 {
        if s0 != 0.0 {
            wy0 / s0
        } else {
            0.0
        }
    } else {
        (wy0 * s2 - wy1 * s1) / det
    }
}

fn angle32(value: Complex32) -> f32 {
    value.im.atan2(value.re)
}

fn smooth_row(
    row: &[Complex32],
    day: &[f64],
    time_win: f64,
    close_ix: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let n_ifg = day.len();
    let time_win_f = time_win.max(1.0e-6);
    let row64: Vec<Complex64> = row
        .iter()
        .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect();
    let row_angle: Vec<f64> = row64.iter().map(|value| value.im.atan2(value.re)).collect();
    let mut dph_smooth = vec![Complex64::new(0.0, 0.0); n_ifg];

    for i1 in 0..n_ifg {
        let time_diff: Vec<f64> = day.iter().map(|value| day[i1] - *value).collect();
        let mut weight: Vec<f64> = time_diff
            .iter()
            .map(|value| (-(value * value) / (2.0 * time_win_f * time_win_f)).exp())
            .collect();
        let weight_sum = weight.iter().sum::<f64>().max(1.0e-12);
        for value in &mut weight {
            *value /= weight_sum;
        }
        let mut dph_mean = Complex64::new(0.0, 0.0);
        for (value, w) in row64.iter().zip(&weight) {
            dph_mean += *value * *w;
        }
        let mean_angle = dph_mean.im.atan2(dph_mean.re);
        let mut dph_mean_adj = Vec::with_capacity(n_ifg);
        for (angle, td) in row_angle.iter().zip(&time_diff) {
            let mut adjusted = wrap_pi(*angle - mean_angle);
            if (adjusted + PI).abs() <= 2.0e-7 && *td > 0.0 {
                adjusted = PI;
            }
            dph_mean_adj.push(adjusted);
        }
        let m0 = weighted_affine_intercept(&time_diff, &dph_mean_adj, &weight);
        dph_smooth[i1] = dph_mean * Complex64::new(m0.cos(), m0.sin());
    }

    let mut dph_noise = Vec::with_capacity(n_ifg);
    for (value, smooth) in row64.iter().zip(&dph_smooth) {
        let noise = *value * smooth.conj();
        dph_noise.push(noise.im.atan2(noise.re) as f32);
    }

    let smooth32: Vec<Complex32> = dph_smooth
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect();
    let mut smooth_uw = vec![0.0_f32; n_ifg];
    if n_ifg > 0 {
        smooth_uw[0] = angle32(smooth32[0]);
        for ix in 1..n_ifg {
            let delta = smooth32[ix] * smooth32[ix - 1].conj();
            smooth_uw[ix] = smooth_uw[ix - 1] + angle32(delta);
        }
        if !close_ix.is_empty() {
            let close_mean =
                close_ix.iter().map(|ix| smooth_uw[*ix]).sum::<f32>() / close_ix.len() as f32;
            let wrapped_close = close_mean.sin().atan2(close_mean.cos());
            let adjust = close_mean - wrapped_close;
            for value in &mut smooth_uw {
                *value -= adjust;
            }
        }
    }

    (smooth_uw, dph_noise)
}

#[pyfunction(signature = (dph_space, day, time_win = 36.0, threads = 0))]
pub fn stage6_smooth_3d_full_single_master<'py>(
    py: Python<'py>,
    dph_space: PyReadonlyArray2<Complex32>,
    day: PyReadonlyArray1<f64>,
    time_win: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = threads;
    let dph = dph_space.as_array();
    let day_view = day.as_array();
    let shape = dph.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "stage6_smooth_3d_full_single_master expects a 2-D dph_space array",
        ));
    }
    let n_edge = shape[0];
    let n_ifg = shape[1];
    if n_ifg == 0 || day_view.len() != n_ifg {
        return Err(PyValueError::new_err(
            "stage6_smooth_3d_full_single_master expects day aligned with dph_space columns",
        ));
    }
    let day_vec = day_view.to_vec();
    let close_ix = close_master_ix(&day_vec);
    let mut smooth = Vec::with_capacity(n_edge * n_ifg);
    let mut noise = Vec::with_capacity(n_edge * n_ifg);
    for edge in 0..n_edge {
        let row = dph.row(edge).to_vec();
        let (smooth_row, noise_row) = smooth_row(&row, &day_vec, time_win, &close_ix);
        smooth.extend(smooth_row);
        noise.extend(noise_row);
    }

    let smooth_array = Array2::from_shape_vec((n_edge, n_ifg), smooth)
        .map_err(|_| PyValueError::new_err("stage6 smoothing output shape construction failed"))?;
    let noise_array = Array2::from_shape_vec((n_edge, n_ifg), noise)
        .map_err(|_| PyValueError::new_err("stage6 smoothing output shape construction failed"))?;
    let dict = PyDict::new(py);
    dict.set_item("dph_smooth_uw", smooth_array.into_pyarray(py))?;
    dict.set_item("dph_noise", noise_array.into_pyarray(py))?;
    Ok(dict)
}
