use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyfunction]
pub fn stage5_duplicate_keep<'py>(
    py: Python<'py>,
    lonlat: PyReadonlyArray2<f64>,
    coh_ps: PyReadonlyArray1<f64>,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let _ = threads;
    let lonlat_view = lonlat.as_array();
    let coh_view = coh_ps.as_array();
    if lonlat_view.ndim() != 2 || lonlat_view.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "lonlat must be a 2-D matrix with two columns",
        ));
    }
    let n = lonlat_view.shape()[0];
    if coh_view.len() != n {
        return Err(PyValueError::new_err(
            "coh_ps length must match lonlat rows",
        ));
    }
    let lonlat_slice = lonlat_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("lonlat must be C-contiguous"))?;
    let coh_slice = coh_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coh_ps must be contiguous"))?;

    let mut keep = vec![true; n];
    let mut groups: HashMap<(u64, u64), usize> = HashMap::with_capacity(n);
    for row in 0..n {
        let key = (
            lonlat_slice[row * 2].to_bits(),
            lonlat_slice[row * 2 + 1].to_bits(),
        );
        if let Some(&best) = groups.get(&key) {
            if coh_slice[row] > coh_slice[best] {
                keep[best] = false;
                groups.insert(key, row);
            } else {
                keep[row] = false;
            }
        } else {
            groups.insert(key, row);
        }
    }

    Ok(keep.into_pyarray(py))
}

#[pyfunction]
pub fn stage5_patch_keep_mask<'py>(
    py: Python<'py>,
    ij_cols: PyReadonlyArray2<i64>,
    merged_ij_cols: PyReadonlyArray2<i64>,
    merged_indices: PyReadonlyArray1<i64>,
    patch_bounds: Option<PyReadonlyArray1<i64>>,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = threads;
    let ij_view = ij_cols.as_array();
    let merged_view = merged_ij_cols.as_array();
    let merged_indices_view = merged_indices.as_array();
    if ij_view.ndim() != 2 || ij_view.shape()[1] < 2 {
        return Err(PyValueError::new_err(
            "stage5_patch_keep_mask expects ij_cols with at least two columns",
        ));
    }
    if merged_view.ndim() != 2 || (merged_view.shape()[0] > 0 && merged_view.shape()[1] < 2) {
        return Err(PyValueError::new_err(
            "stage5_patch_keep_mask expects merged_ij_cols with at least two columns",
        ));
    }
    if merged_indices_view.len() != merged_view.shape()[0] {
        return Err(PyValueError::new_err(
            "stage5_patch_keep_mask expects merged_indices length to match merged_ij_cols rows",
        ));
    }
    let bounds_vec = if let Some(bounds) = patch_bounds {
        let bounds_view = bounds.as_array();
        if bounds_view.len() < 4 {
            return Err(PyValueError::new_err(
                "stage5_patch_keep_mask patch_bounds must contain four values",
            ));
        }
        let bounds_slice = bounds_view
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("patch_bounds must be contiguous"))?;
        Some([
            bounds_slice[0],
            bounds_slice[1],
            bounds_slice[2],
            bounds_slice[3],
        ])
    } else {
        None
    };

    let n_row = ij_view.shape()[0];
    let mut keep_patch = vec![true; n_row];
    let mut remove_ix = Vec::<i64>::new();
    for row in 0..n_row {
        let col = ij_view[[row, 0]];
        let patch_row = ij_view[[row, 1]];
        let in_bounds = if let Some([row_min, row_max, col_min, col_max]) = bounds_vec {
            col >= col_min - 1
                && col <= col_max - 1
                && patch_row >= row_min - 1
                && patch_row <= row_max - 1
        } else {
            true
        };

        let mut merged_index = None;
        for merged_row in 0..merged_view.shape()[0] {
            if merged_view[[merged_row, 0]] == col && merged_view[[merged_row, 1]] == patch_row {
                merged_index = Some(merged_indices_view[merged_row]);
                break;
            }
        }
        if in_bounds {
            if let Some(index) = merged_index {
                remove_ix.push(index);
            }
            keep_patch[row] = true;
        } else {
            keep_patch[row] = merged_index.is_none();
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("keep_patch", keep_patch.into_pyarray(py))?;
    dict.set_item("remove_ix", remove_ix.into_pyarray(py))?;
    Ok(dict)
}
