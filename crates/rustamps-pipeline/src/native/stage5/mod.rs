mod merge;
mod merged_write;
mod patch;
mod write;

#[cfg(test)]
mod tests;

use std::path::Path;

use crate::{PipelineError, RunConfig};

pub fn run_patch(path: &Path, config: &RunConfig) -> Result<String, PipelineError> {
    patch::run(path, config)
}

pub fn run_merged(path: &Path, config: &RunConfig) -> Result<String, PipelineError> {
    merge::run(path, config)
}
