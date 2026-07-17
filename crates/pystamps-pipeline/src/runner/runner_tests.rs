use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{run_pipeline, PipelineContext, PipelineError, StageExecutor};
use crate::RunConfig;

struct Temp(PathBuf);

impl Temp {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("pystamps-runner-{nonce}"));
        fs::create_dir_all(root.join("PATCH_1")).unwrap();
        Self(root)
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct ThreadCountExecutor(AtomicUsize);

impl StageExecutor for ThreadCountExecutor {
    fn run_patch(
        &self,
        _stage: u8,
        _patch: &Path,
        _config: &RunConfig,
    ) -> Result<String, PipelineError> {
        self.0.store(rayon::current_num_threads(), Ordering::SeqCst);
        Ok("observed worker pool".to_owned())
    }

    fn run_merged(
        &self,
        _stage: u8,
        _root: &Path,
        _config: &RunConfig,
    ) -> Result<String, PipelineError> {
        unreachable!()
    }
}

#[test]
fn configured_cpu_workers_builds_the_pipeline_rayon_pool() {
    let temp = Temp::new();
    let mut config = RunConfig::default();
    config.runtime.cpu_workers = 3;
    let executor = ThreadCountExecutor(AtomicUsize::new(0));
    let report = run_pipeline(
        &PipelineContext {
            dataset_root: temp.0.clone(),
            config,
            start_step: 1,
            end_step: 1,
            dry_run: false,
        },
        &executor,
    )
    .unwrap();
    assert!(report.ok());
    assert_eq!(executor.0.load(Ordering::SeqCst), 3);
}
