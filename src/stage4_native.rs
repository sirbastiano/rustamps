use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn stage4_duplicate_keep<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<f64>,
    coh: PyReadonlyArray1<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let xy_view = xy.as_array();
    let coh_view = coh.as_array();
    if xy_view.ndim() != 2 || xy_view.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "xy must be a 2-D matrix with two columns",
        ));
    }
    let n = xy_view.shape()[0];
    if coh_view.len() != n {
        return Err(PyValueError::new_err("coh length must match xy rows"));
    }
    let xy_slice = xy_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("xy must be C-contiguous"))?;
    let coh_slice = coh_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coh must be contiguous"))?;

    let mut keep = vec![true; n];
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let x = xy_slice[i * 2];
        let y = xy_slice[i * 2 + 1];
        let mut best = i;
        let mut duplicates = Vec::new();
        duplicates.push(i);
        for j in (i + 1)..n {
            if xy_slice[j * 2] == x && xy_slice[j * 2 + 1] == y {
                duplicates.push(j);
                if coh_slice[j] > coh_slice[best] {
                    best = j;
                }
            }
        }
        if duplicates.len() <= 1 {
            continue;
        }
        for ix in duplicates {
            keep[ix] = ix == best;
        }
    }

    Ok(keep.into_pyarray(py))
}

#[pyfunction]
pub fn stage4_adjacent_component_keep<'py>(
    py: Python<'py>,
    ij_cols23: PyReadonlyArray2<i64>,
    coh: PyReadonlyArray1<f64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let ij_view = ij_cols23.as_array();
    let coh_view = coh.as_array();
    if ij_view.ndim() != 2 || ij_view.shape()[1] != 2 {
        return Err(PyValueError::new_err(
            "ij_cols23 must be a 2-D matrix with two columns",
        ));
    }
    let n_ps = ij_view.shape()[0];
    if coh_view.len() != n_ps {
        return Err(PyValueError::new_err("coh length must match ij rows"));
    }
    if n_ps == 0 {
        return Ok(Vec::<bool>::new().into_pyarray(py));
    }
    let ij_slice = ij_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ij_cols23 must be C-contiguous"))?;
    let coh_slice = coh_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("coh must be contiguous"))?;

    let mut min_r = ij_slice[0];
    let mut min_c = ij_slice[1];
    for row in 0..n_ps {
        min_r = min_r.min(ij_slice[row * 2]);
        min_c = min_c.min(ij_slice[row * 2 + 1]);
    }

    let mut shifted = Vec::with_capacity(n_ps);
    let mut max_r = 0usize;
    let mut max_c = 0usize;
    for row in 0..n_ps {
        let r = (ij_slice[row * 2] + 2 - min_r) as usize;
        let c = (ij_slice[row * 2 + 1] + 2 - min_c) as usize;
        max_r = max_r.max(r);
        max_c = max_c.max(c);
        shifted.push((r, c));
    }

    let n_r = max_r + 2;
    let n_c = max_c + 2;
    let mut neigh_ix = vec![0usize; n_r * n_c];
    let cell = |r: usize, c: usize, n_c_val: usize| -> usize { r * n_c_val + c };

    for (idx, &(r, c)) in shifted.iter().enumerate() {
        for rr in (r - 1)..=(r + 1) {
            for cc in (c - 1)..=(c + 1) {
                if rr == r && cc == c {
                    continue;
                }
                let pos = cell(rr, cc, n_c);
                if neigh_ix[pos] == 0 {
                    neigh_ix[pos] = idx + 1;
                }
            }
        }
    }

    let mut neigh_ps = vec![Vec::<usize>::new(); n_ps + 1];
    for (idx, &(r, c)) in shifted.iter().enumerate() {
        let owner = neigh_ix[cell(r, c, n_c)];
        if owner != 0 {
            neigh_ps[owner].push(idx + 1);
        }
    }

    let mut keep = vec![true; n_ps];
    for i in 1..=n_ps {
        if neigh_ps[i].is_empty() {
            continue;
        }
        let mut same_ps = vec![i];
        let mut pos = 0usize;
        while pos < same_ps.len() {
            let ps_i = same_ps[pos];
            if !neigh_ps[ps_i].is_empty() {
                let linked = std::mem::take(&mut neigh_ps[ps_i]);
                same_ps.extend(linked);
            }
            pos += 1;
        }

        same_ps.sort_unstable();
        same_ps.dedup();
        let mut best = same_ps[0];
        let mut best_coh = coh_slice[best - 1];
        for &candidate in same_ps.iter().skip(1) {
            let candidate_coh = coh_slice[candidate - 1];
            if candidate_coh > best_coh {
                best = candidate;
                best_coh = candidate_coh;
            }
        }
        for candidate in same_ps {
            if candidate != best {
                keep[candidate - 1] = false;
            }
        }
    }

    Ok(keep.into_pyarray(py))
}

#[pyfunction]
pub fn stage4_weed_ifg_index<'py>(
    py: Python<'py>,
    n_ifg: i64,
    drop_ifg_index: PyReadonlyArray1<i64>,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if n_ifg < 0 {
        return Err(PyValueError::new_err("n_ifg must be non-negative"));
    }
    let drop_view = drop_ifg_index.as_array();
    let drop_slice = drop_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("drop_ifg_index must be contiguous"))?;

    let mut out = Vec::with_capacity(n_ifg as usize);
    'ifg_loop: for ifg in 1..=n_ifg {
        for &drop in drop_slice {
            if ifg == drop {
                continue 'ifg_loop;
            }
        }
        out.push(ifg as f64);
    }

    Ok(out.into_pyarray(py))
}

fn normalize_complex(value: Complex64) -> Complex64 {
    let mag = value.norm();
    if mag == 0.0 {
        Complex64::new(0.0, 0.0)
    } else {
        value / mag
    }
}

#[pyfunction]
pub fn stage4_phase_correction<'py>(
    py: Python<'py>,
    ph2: PyReadonlyArray2<Complex64>,
    ix_weed: PyReadonlyArray1<bool>,
    k_ps: PyReadonlyArray1<f64>,
    c_ps: PyReadonlyArray1<f64>,
    bperp: PyReadonlyArray1<f64>,
    small_baseline: bool,
    master_ix: i64,
    _threads: usize,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let ph_view = ph2.as_array();
    if ph_view.ndim() != 2 {
        return Err(PyValueError::new_err("ph2 must be a 2-D complex matrix"));
    }
    let n_ps = ph_view.shape()[0];
    let n_ifg = ph_view.shape()[1];
    let ix_view = ix_weed.as_array();
    let k_view = k_ps.as_array();
    let c_view = c_ps.as_array();
    let b_view = bperp.as_array();
    if ix_view.len() != n_ps || k_view.len() != n_ps || c_view.len() != n_ps {
        return Err(PyValueError::new_err(
            "ix_weed, k_ps, and c_ps lengths must match ph2 rows",
        ));
    }
    if b_view.len() != n_ifg {
        return Err(PyValueError::new_err("bperp length must match ph2 columns"));
    }
    if !small_baseline && (master_ix < 1 || master_ix as usize > n_ifg) {
        return Err(PyValueError::new_err(
            "master_ix must be a valid 1-based ph2 column",
        ));
    }
    let ph_slice = ph_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ph2 must be C-contiguous"))?;
    let ix_slice = ix_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("ix_weed must be contiguous"))?;
    let k_slice = k_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("k_ps must be contiguous"))?;
    let c_slice = c_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("c_ps must be contiguous"))?;
    let b_slice = b_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("bperp must be contiguous"))?;

    let n_keep = ix_slice.iter().filter(|&&keep| keep).count();
    let mut out = Vec::with_capacity(n_keep * n_ifg);
    for row in 0..n_ps {
        if !ix_slice[row] {
            continue;
        }
        let k = k_slice[row];
        for col in 0..n_ifg {
            let angle = -(k * b_slice[col]);
            let ramp = Complex64::new(angle.cos(), angle.sin());
            let value = ph_slice[row * n_ifg + col] * ramp;
            out.push(normalize_complex(normalize_complex(value)));
        }
        if !small_baseline {
            let col = (master_ix - 1) as usize;
            let c = c_slice[row];
            let row_start = out.len() - n_ifg;
            out[row_start + col] = Complex64::new(c.cos(), c.sin());
        }
    }

    Ok(Array2::from_shape_vec((n_keep, n_ifg), out)
        .map_err(|err| {
            PyValueError::new_err(format!(
                "failed to build stage4 phase correction output: {err}"
            ))
        })?
        .into_pyarray(py))
}
