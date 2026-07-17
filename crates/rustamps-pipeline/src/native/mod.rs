mod mat;
mod params;
mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod stage5;
mod stage6;
mod stage7;
mod stage8;
mod topology;

use std::path::Path;

use crate::{PipelineError, RunConfig, StageExecutor};

pub struct NativeExecutor;

impl StageExecutor for NativeExecutor {
    fn run_patch(
        &self,
        stage: u8,
        patch: &Path,
        config: &RunConfig,
    ) -> Result<String, PipelineError> {
        match stage {
            1 => stage1::run(patch, config),
            2 => stage2::run(patch, config),
            3 => stage3::run(patch, config),
            4 => stage4::run(patch, config),
            5 => stage5::run_patch(patch, config),
            _ => Err(unavailable(
                stage,
                "native patch-stage wiring is incomplete",
            )),
        }
    }

    fn run_merged(
        &self,
        stage: u8,
        root: &Path,
        config: &RunConfig,
    ) -> Result<String, PipelineError> {
        match stage {
            5 => stage5::run_merged(root, config),
            6 => stage6::run(root, config),
            7 => stage7::run(root, config),
            8 => stage8::run(root, config),
            _ => Err(unavailable(
                stage,
                "native merged-stage wiring is incomplete",
            )),
        }
    }
}

pub(super) fn failure(stage: u8, error: impl std::fmt::Display) -> PipelineError {
    PipelineError::Stage {
        stage,
        details: error.to_string(),
    }
}

fn unavailable(stage: u8, details: &str) -> PipelineError {
    PipelineError::Unavailable {
        stage,
        details: details.to_owned(),
    }
}
