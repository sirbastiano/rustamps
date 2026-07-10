use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let mut best = pivot;
        for row in (pivot + 1)..3 {
            if a[row][pivot].abs() > a[best][pivot].abs() {
                best = row;
            }
        }
        if a[best][pivot].abs() <= 1.0e-12 {
            return None;
        }
        if best != pivot {
            a.swap(best, pivot);
            b.swap(best, pivot);
        }
        let div = a[pivot][pivot];
        for col in pivot..3 {
            a[pivot][col] /= div;
        }
        b[pivot] /= div;
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for col in pivot..3 {
                a[row][col] -= factor * a[pivot][col];
            }
            b[row] -= factor * b[pivot];
        }
    }
    Some(b)
}

fn fit_plane(design: &[[f64; 3]], values: &[f64], valid: Option<&[bool]>) -> Option<[f64; 3]> {
    let mut gram = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    let mut count = 0_usize;
    for (row_ix, row) in design.iter().enumerate() {
        if valid.is_some_and(|mask| !mask[row_ix]) {
            continue;
        }
        count += 1;
        let y = values[row_ix];
        for i in 0..3 {
            rhs[i] += row[i] * y;
            for j in 0..3 {
                gram[i][j] += row[i] * row[j];
            }
        }
    }
    if count < 3 {
        return None;
    }
    solve3(gram, rhs)
}

#[pyfunction]
pub fn stage7_deramp_unwrapped_phase<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<f64>,
    ph_all: PyReadonlyArray2<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let xy_view = xy.as_array();
    let ph_view = ph_all.as_array();
    if xy_view.ndim() != 2 || xy_view.shape()[1] < 3 {
        return Err(PyValueError::new_err(
            "stage7_deramp_unwrapped_phase xy must have shape (n_ps, >=3)",
        ));
    }
    if ph_view.ndim() != 2 || ph_view.shape()[0] != xy_view.shape()[0] {
        return Err(PyValueError::new_err(
            "stage7_deramp_unwrapped_phase phase rows must match xy rows",
        ));
    }
    let xy_slice = xy_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("xy must be C-contiguous"))?;
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph_all must be C-contiguous"))?;
    let n_ps = ph_view.shape()[0];
    let n_col = ph_view.shape()[1];
    let xy_width = xy_view.shape()[1];
    let design: Vec<[f64; 3]> = (0..n_ps)
        .map(|row| {
            [
                xy_slice[row * xy_width + 1] / 1000.0,
                xy_slice[row * xy_width + 2] / 1000.0,
                1.0,
            ]
        })
        .collect();

    let has_nan = ph_slice.iter().any(|value| value.is_nan());
    let mut ph_out = ph_slice.to_vec();
    let mut ph_ramp = if has_nan {
        vec![f64::NAN; n_ps * n_col]
    } else {
        vec![0.0_f64; n_ps * n_col]
    };

    for col in 0..n_col {
        let values: Vec<f64> = (0..n_ps).map(|row| ph_slice[row * n_col + col]).collect();
        let valid_mask;
        let valid = if has_nan {
            valid_mask = values
                .iter()
                .map(|value| !value.is_nan())
                .collect::<Vec<_>>();
            if valid_mask.iter().filter(|value| **value).count() <= 5 {
                continue;
            }
            Some(valid_mask.as_slice())
        } else {
            None
        };
        let Some(coeffs) = fit_plane(&design, &values, valid) else {
            continue;
        };
        for row in 0..n_ps {
            let ramp = design[row][0] * coeffs[0] + design[row][1] * coeffs[1] + coeffs[2];
            ph_ramp[row * n_col + col] = ramp;
            if valid.is_none_or(|mask| mask[row]) {
                ph_out[row * n_col + col] = ph_slice[row * n_col + col] - ramp;
            }
        }
    }

    let ph_out_arr = Array2::from_shape_vec((n_ps, n_col), ph_out)
        .map_err(|_| PyValueError::new_err("stage7 deramp output shape construction failed"))?;
    let ph_ramp_arr = Array2::from_shape_vec((n_ps, n_col), ph_ramp)
        .map_err(|_| PyValueError::new_err("stage7 deramp output shape construction failed"))?;
    let dict = PyDict::new(py);
    dict.set_item("ph_out", ph_out_arr.into_pyarray(py))?;
    dict.set_item("ph_ramp", ph_ramp_arr.into_pyarray(py))?;
    Ok(dict)
}
