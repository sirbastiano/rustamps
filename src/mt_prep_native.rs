use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs;
use std::path::PathBuf;

#[path = "mt_prep_native_io.rs"]
mod mt_prep_native_io;
#[path = "mt_prep_native_write.rs"]
mod mt_prep_native_write;

use self::mt_prep_native_io::{
    dataset_shape, diff_files, ranges, read_be_c64_amplitude, read_be_f32_raster, resolve_master,
    sorted_files,
};
use self::mt_prep_native_write::write_patch;

pub(crate) fn err(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}

#[pyfunction]
#[pyo3(signature = (dataset_root, master_date=None, amp_dispersion=0.4, range_patches=1, azimuth_patches=1, range_overlap=50, azimuth_overlap=50))]
pub fn mt_prep_prepare_snap_inputs<'py>(
    py: Python<'py>,
    dataset_root: &str,
    master_date: Option<String>,
    amp_dispersion: f64,
    range_patches: usize,
    azimuth_patches: usize,
    range_overlap: usize,
    azimuth_overlap: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let root = PathBuf::from(dataset_root);
    let (width, length) = dataset_shape(&root)?;
    let cells = width * length;
    let master = resolve_master(&root, master_date.as_deref())?;
    let rslc_files = sorted_files(&root.join("rslc"), ".rslc")?;
    if rslc_files.is_empty() {
        return Err(err(format!(
            "No rslc/*.rslc files found under {}",
            root.display()
        )));
    }
    let diffs = diff_files(&root, &master)?;

    let mut sum_amp = vec![0.0_f64; cells];
    let mut sum_sq = vec![0.0_f64; cells];
    let mut has_low_amp = vec![false; cells];
    for path in &rslc_files {
        let amplitudes = read_be_c64_amplitude(path, cells)?;
        let mut calibration_sum = 0.0_f64;
        let mut calibration_count = 0usize;
        for amp in &amplitudes {
            if *amp > 0.001 {
                calibration_sum += *amp;
                calibration_count += 1;
            }
        }
        let calibration = if calibration_count == 0 {
            0.0
        } else {
            calibration_sum / calibration_count as f64
        };
        for (idx, raw_amp) in amplitudes.into_iter().enumerate() {
            let amp = raw_amp / calibration;
            if amp <= 0.00005 {
                has_low_amp[idx] = true;
                sum_amp[idx] = 0.0;
            } else {
                sum_amp[idx] += amp;
                sum_sq[idx] += amp * amp;
            }
        }
    }
    let count = rslc_files.len() as f64;
    let lon = read_be_f32_raster(&root.join("geo").join(format!("{master}.lon")), cells)?;
    let lat = read_be_f32_raster(&root.join("geo").join(format!("{master}.lat")), cells)?;
    let hgt = read_be_f32_raster(&root.join("geo").join("elevation_dem.rdc"), cells)?;
    let mut da = vec![0.0_f32; cells];
    let mut normalized_amp_sum = vec![0.0_f32; cells];
    let mut selected = Vec::new();
    for idx in 0..cells {
        let dispersion = (count * sum_sq[idx] / (sum_amp[idx] * sum_amp[idx]) - 1.0)
            .max(0.0)
            .sqrt();
        da[idx] = dispersion as f32;
        normalized_amp_sum[idx] = sum_amp[idx] as f32;
        if lon[idx].is_finite()
            && lat[idx].is_finite()
            && hgt[idx].is_finite()
            && !has_low_amp[idx]
            && dispersion.is_finite()
            && sum_amp[idx] > 0.0
            && dispersion < amp_dispersion
        {
            selected.push((idx / width, idx % width));
        }
    }

    let col_ranges = ranges(width, range_patches, range_overlap)?;
    let row_ranges = ranges(length, azimuth_patches, azimuth_overlap)?;
    let patch_rows = PyList::empty(py);
    let mut names = Vec::new();
    let mut candidate_count = 0usize;
    let mut patch_index = 1usize;
    for (col_range, col_noover) in &col_ranges {
        for (row_range, row_noover) in &row_ranges {
            let patch_name = format!("PATCH_{patch_index}");
            let bounds = (col_range.0, col_range.1, row_range.0, row_range.1);
            let noover = (col_noover.0, col_noover.1, row_noover.0, row_noover.1);
            let n = write_patch(
                &root.join(&patch_name),
                bounds,
                noover,
                &selected,
                (&lon, &lat, &hgt, &da, &normalized_amp_sum),
                &diffs,
                cells,
                width,
            )?;
            if n > 0 {
                names.push(patch_name.clone());
                candidate_count += n;
                let row = PyDict::new(py);
                row.set_item("patch", patch_name)?;
                row.set_item("candidates", n)?;
                row.set_item("bounds", bounds)?;
                row.set_item("noover", noover)?;
                patch_rows.append(row)?;
            }
            patch_index += 1;
        }
    }
    if names.is_empty() {
        return Err(err(
            "No candidates passed the amplitude-dispersion threshold",
        ));
    }
    fs::write(root.join("patch.list"), format!("{}\n", names.join("\n")))
        .map_err(|e| err(format!("failed to write patch.list: {e}")))?;
    let out = PyDict::new(py);
    out.set_item("dataset_root", root.to_string_lossy().to_string())?;
    out.set_item("patch_count", names.len())?;
    out.set_item("candidate_count", candidate_count)?;
    out.set_item("patch_rows", patch_rows)?;
    Ok(out)
}
