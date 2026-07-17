use std::path::{Path, PathBuf};

use pystamps_io::{read_mat, DatasetError, MatError};
use pystamps_pipeline::config::ToleranceConfig;
use thiserror::Error;

use crate::artifacts::artifact_paths;
use crate::value_compare::compare_file;
use crate::{FileComparison, VerificationReport};

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("failed to read verification artifact {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: MatError,
    },
    #[error("failed to discover verification artifacts: {0}")]
    Dataset(#[from] DatasetError),
    #[error("verification root does not exist: {0}")]
    MissingRoot(String),
    #[error("invalid verification tolerance: {0}")]
    InvalidTolerance(String),
    #[error("verification stage must be between 1 and 8, found {0}")]
    InvalidStage(u8),
}

pub fn verify_paths(
    run: &Path,
    golden: &Path,
    tolerance: &ToleranceConfig,
) -> Result<VerificationReport, VerifyError> {
    verify_paths_with_scope(run, golden, tolerance, None, false)
}

pub fn verify_paths_through_stage(
    run: &Path,
    golden: &Path,
    tolerance: &ToleranceConfig,
    through_stage: u8,
) -> Result<VerificationReport, VerifyError> {
    if !(1..=8).contains(&through_stage) {
        return Err(VerifyError::InvalidStage(through_stage));
    }
    verify_paths_with_scope(run, golden, tolerance, Some(through_stage), false)
}

pub fn verify_paths_with_scope(
    run: &Path,
    golden: &Path,
    tolerance: &ToleranceConfig,
    through_stage: Option<u8>,
    final_products_only: bool,
) -> Result<VerificationReport, VerifyError> {
    if through_stage.is_some_and(|stage| !(1..=8).contains(&stage)) {
        return Err(VerifyError::InvalidStage(through_stage.unwrap()));
    }
    tolerance
        .validate()
        .map_err(VerifyError::InvalidTolerance)?;
    if !run.exists() {
        return Err(VerifyError::MissingRoot(run.display().to_string()));
    }
    if !golden.exists() {
        return Err(VerifyError::MissingRoot(golden.display().to_string()));
    }
    let artifacts = artifact_paths(golden, through_stage, final_products_only)?;
    if artifacts.is_empty() {
        return Ok(VerificationReport {
            comparisons: vec![failure("<dataset>", "no golden artifacts found")],
        });
    }
    let mut report = VerificationReport::default();
    for relative in artifacts {
        let golden_path = golden.join(&relative);
        let run_path = run.join(&relative);
        let display = relative.to_string_lossy();
        if !run_path.exists() {
            report
                .comparisons
                .push(failure(&display, "missing run artifact"));
            continue;
        }
        let observed = read(&run_path)?;
        let expected = read(&golden_path)?;
        report
            .comparisons
            .push(compare_file(&display, &observed, &expected, tolerance));
    }
    Ok(report)
}

fn read(path: &Path) -> Result<pystamps_io::MatFile, VerifyError> {
    read_mat(path).map_err(|source| VerifyError::Read {
        path: path.to_path_buf(),
        source,
    })
}

fn failure(path: &str, message: &str) -> FileComparison {
    FileComparison {
        path: path.to_owned(),
        ok: false,
        message: message.to_owned(),
        failing_key: None,
        max_abs: None,
        outliers: Vec::new(),
    }
}
