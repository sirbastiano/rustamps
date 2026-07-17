mod discover;
mod stats;
mod write;

#[cfg(test)]
mod tests;

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

use crate::{read_mat, write_mat, MatArray, MatFile, MatValue, StageTransaction};

#[derive(Clone, Copy, Debug)]
pub struct SnapPrepOptions<'a> {
    pub master_date: Option<&'a str>,
    pub amp_dispersion: f64,
    pub range_patches: usize,
    pub azimuth_patches: usize,
    pub range_overlap: usize,
    pub azimuth_overlap: usize,
    pub force: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PatchSummary {
    pub patch: String,
    pub candidates: usize,
    pub bounds: [usize; 4],
    pub noover: [usize; 4],
}

#[derive(Clone, Debug, Serialize)]
pub struct SnapPrepSummary {
    pub dataset_root: PathBuf,
    pub patch_count: usize,
    pub candidate_count: usize,
    pub patches: Vec<PatchSummary>,
}

#[derive(Debug, Error)]
pub enum SnapPrepError {
    #[error("SNAP preparation I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid SNAP dataset: {0}")]
    Invalid(String),
    #[error("SNAP preparation transaction failed: {0}")]
    Transaction(#[from] crate::TransactionError),
}

pub fn prepare_snap(
    root: impl AsRef<Path>,
    options: SnapPrepOptions<'_>,
) -> Result<SnapPrepSummary, SnapPrepError> {
    validate_options(&options)?;
    let root = root.as_ref().canonicalize()?;
    let discovered = discover::discover(&root, options.master_date)?;
    let candidates = stats::candidate_statistics(
        &discovered.rslc,
        &discovered.lon,
        &discovered.lat,
        &discovered.height,
        discovered.cells(),
        discovered.width,
        options.amp_dispersion,
    )?;
    let grid = discover::patch_grid(
        discovered.width,
        discovered.length,
        options.range_patches,
        options.azimuth_patches,
        options.range_overlap,
        options.azimuth_overlap,
    )?;
    let transaction = StageTransaction::begin(&root, "mt-prep")?;
    write_sensor_params(
        &root,
        &transaction.path("parms.mat"),
        discovered.heading,
        discovered.wavelength,
    )?;
    let mut patches = Vec::new();
    for (index, bounds, noover) in grid {
        let name = format!("PATCH_{index}");
        let count = write::write_patch(
            &transaction.path(&name),
            bounds,
            noover,
            &candidates,
            &discovered.diff,
            discovered.width,
            discovered.length,
        )?;
        if count > 0 {
            patches.push(PatchSummary {
                patch: name,
                candidates: count,
                bounds,
                noover,
            });
        }
    }
    if patches.is_empty() {
        return Err(SnapPrepError::Invalid(
            "no candidates passed the amplitude-dispersion threshold".to_owned(),
        ));
    }
    let list = patches
        .iter()
        .map(|row| row.patch.as_str())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(transaction.path("patch.list"), list)?;
    if !options.force && patches.iter().any(|row| root.join(&row.patch).exists()) {
        return Err(SnapPrepError::Invalid(
            "PATCH_* already exists; pass --force to replace prepared inputs".to_owned(),
        ));
    }
    let mut names = patches
        .iter()
        .map(|row| row.patch.as_str())
        .collect::<Vec<_>>();
    names.push("patch.list");
    names.push("parms.mat");
    transaction.commit(&names, "patch.list")?;
    Ok(SnapPrepSummary {
        dataset_root: root,
        patch_count: patches.len(),
        candidate_count: patches.iter().map(|row| row.candidates).sum(),
        patches,
    })
}

fn write_sensor_params(
    root: &Path,
    output: &Path,
    heading: f64,
    wavelength: f64,
) -> Result<(), SnapPrepError> {
    let source = root.join("parms.mat");
    let mut params = if source.is_file() {
        read_mat(source).map_err(|error| SnapPrepError::Invalid(error.to_string()))?
    } else {
        MatFile::new()
    };
    for (key, value) in [("heading", heading), ("lambda", wavelength)] {
        params.insert(
            key.to_owned(),
            MatValue::F64(MatArray {
                shape: vec![1, 1],
                values: vec![value],
            }),
        );
    }
    write_mat(output, &params).map_err(|error| SnapPrepError::Invalid(error.to_string()))
}

fn validate_options(options: &SnapPrepOptions<'_>) -> Result<(), SnapPrepError> {
    if !options.amp_dispersion.is_finite() || options.amp_dispersion < 0.0 {
        return Err(SnapPrepError::Invalid(
            "amp_dispersion must be finite and non-negative".to_owned(),
        ));
    }
    if options.range_patches == 0 || options.azimuth_patches == 0 {
        return Err(SnapPrepError::Invalid(
            "patch counts must be positive".to_owned(),
        ));
    }
    Ok(())
}
