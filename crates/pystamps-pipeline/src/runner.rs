use std::path::{Path, PathBuf};
use std::time::Instant;

use pystamps_io::discover_dataset;
use thiserror::Error;

use crate::invalidation::invalidate_downstream;
use crate::{PipelineReport, RunConfig, StageResult};

#[cfg(test)]
mod runner_tests;

const PATCH_ARTIFACTS: [&str; 5] = ["ps1.mat", "pm1.mat", "select1.mat", "weed1.mat", "ph2.mat"];
const MERGED_ARTIFACTS: [&str; 4] = ["ifgstd2.mat", "phuw2.mat", "scla2.mat", "scn2.mat"];

pub struct PipelineContext {
    pub dataset_root: PathBuf,
    pub config: RunConfig,
    pub start_step: u8,
    pub end_step: u8,
    pub dry_run: bool,
}

pub trait StageExecutor: Sync {
    fn run_patch(
        &self,
        stage: u8,
        patch: &Path,
        config: &RunConfig,
    ) -> Result<String, PipelineError>;
    fn run_merged(
        &self,
        stage: u8,
        root: &Path,
        config: &RunConfig,
    ) -> Result<String, PipelineError>;
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("invalid stage range {0}..{1}")]
    StageRange(u8, u8),
    #[error("dataset error: {0}")]
    Dataset(#[from] pystamps_io::DatasetError),
    #[error("stage {stage} is not available in the native runner: {details}")]
    Unavailable { stage: u8, details: String },
    #[error("stage {stage} failed: {details}")]
    Stage { stage: u8, details: String },
    #[error("runtime configuration failed: {0}")]
    Runtime(String),
}

pub fn run_pipeline(
    context: &PipelineContext,
    executor: &dyn StageExecutor,
) -> Result<PipelineReport, PipelineError> {
    let workers = context.config.runtime.cpu_workers;
    if workers == 0 {
        return run_pipeline_inner(context, executor);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|error| PipelineError::Runtime(error.to_string()))?;
    pool.install(|| run_pipeline_inner(context, executor))
}

fn run_pipeline_inner(
    context: &PipelineContext,
    executor: &dyn StageExecutor,
) -> Result<PipelineReport, PipelineError> {
    if context.start_step > context.end_step || context.end_step > 8 {
        return Err(PipelineError::StageRange(
            context.start_step,
            context.end_step,
        ));
    }
    let layout = discover_dataset(&context.dataset_root)?;
    let mut report = PipelineReport::default();
    for stage in context.start_step.max(1)..=context.end_step {
        if stage <= 5 {
            for patch in &layout.patches {
                let result = run_one(
                    context,
                    executor,
                    stage,
                    "patch",
                    patch,
                    PATCH_ARTIFACTS[(stage - 1) as usize],
                );
                let failed = result.status == "failed";
                report.results.push(result);
                if failed {
                    return Ok(report);
                }
            }
            if stage == 5 {
                let result = run_one(
                    context,
                    executor,
                    stage,
                    "merged",
                    &layout.root,
                    MERGED_ARTIFACTS[0],
                );
                let failed = result.status == "failed";
                report.results.push(result);
                if failed {
                    return Ok(report);
                }
            }
        } else {
            let result = run_one(
                context,
                executor,
                stage,
                "merged",
                &layout.root,
                MERGED_ARTIFACTS[(stage - 5) as usize],
            );
            let failed = result.status == "failed";
            report.results.push(result);
            if failed {
                return Ok(report);
            }
        }
    }
    Ok(report)
}

fn run_one(
    context: &PipelineContext,
    executor: &dyn StageExecutor,
    stage: u8,
    scope: &str,
    target: &Path,
    marker: &str,
) -> StageResult {
    let name = target
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_owned();
    if context.start_step == 0 && target.join(marker).exists() {
        return result(
            stage,
            scope,
            name,
            "skipped_existing",
            format!("{marker} present"),
            0.0,
        );
    }
    if context.dry_run {
        return result(
            stage,
            scope,
            name,
            "planned",
            format!("Would produce {marker}"),
            0.0,
        );
    }
    let started = Instant::now();
    if let Err(error) = invalidate_downstream(stage, scope, target, &context.dataset_root) {
        return result(
            stage,
            scope,
            name,
            "failed",
            format!("failed to invalidate downstream products: {error}"),
            started.elapsed().as_secs_f64(),
        );
    }
    let outcome = if scope == "patch" {
        executor.run_patch(stage, target, &context.config)
    } else {
        executor.run_merged(stage, target, &context.config)
    };
    match outcome {
        Ok(details) => result(
            stage,
            scope,
            name,
            "completed",
            details,
            started.elapsed().as_secs_f64(),
        ),
        Err(error) => result(
            stage,
            scope,
            name,
            "failed",
            error.to_string(),
            started.elapsed().as_secs_f64(),
        ),
    }
}

fn result(
    stage: u8,
    scope: &str,
    target: String,
    status: &str,
    details: String,
    duration_sec: f64,
) -> StageResult {
    StageResult {
        stage,
        scope: scope.to_owned(),
        target,
        status: status.to_owned(),
        details,
        duration_sec,
    }
}
