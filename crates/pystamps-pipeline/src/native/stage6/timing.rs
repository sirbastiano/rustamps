use std::path::{Path, PathBuf};

use pystamps_io::atomic_write;
use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: u64 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfgTiming {
    pub solve_ordinal: usize,
    pub ifg_index: usize,
    pub resumed: bool,
    pub prepare_sec: f64,
    pub core_sec: f64,
    pub decode_sec: f64,
    pub initial_flow_sec: f64,
    pub initial_label_sec: f64,
    pub post_flow_sec: f64,
    pub final_label_sec: f64,
    pub msd_sec: f64,
    pub extract_sec: f64,
    pub total_sec: f64,
}

impl IfgTiming {
    pub fn resumed(solve_ordinal: usize, ifg_index: usize) -> Self {
        Self {
            solve_ordinal,
            ifg_index,
            resumed: true,
            prepare_sec: 0.0,
            core_sec: 0.0,
            decode_sec: 0.0,
            initial_flow_sec: 0.0,
            initial_label_sec: 0.0,
            post_flow_sec: 0.0,
            final_label_sec: 0.0,
            msd_sec: 0.0,
            extract_sec: 0.0,
            total_sec: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTimings {
    pub input_sec: f64,
    pub grid_sec: f64,
    pub interpolation_sec: f64,
    pub space_time_sec: f64,
    pub costs_sec: f64,
    pub solve_wall_sec: f64,
    pub solve_output_sec: f64,
    pub total_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage6TimingReport {
    pub schema_version: u64,
    pub input_fingerprint: u64,
    pub n_ps: usize,
    pub n_ifg: usize,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub requested_ifg_workers: usize,
    pub rayon_workers: usize,
    pub effective_ifg_workers: usize,
    pub resumed_ifgs: usize,
    pub solved_ifgs: usize,
    pub phases: PhaseTimings,
    pub interferograms: Vec<IfgTiming>,
}

pub fn report_path(root: &Path, fingerprint: u64) -> PathBuf {
    root.join(".pystamps-stage6")
        .join(format!("timing-v{SCHEMA_VERSION}-{fingerprint:013x}.json"))
}

pub fn write(root: &Path, report: &Stage6TimingReport) -> Result<PathBuf, String> {
    let path = report_path(root, report.input_fingerprint);
    let bytes = serde_json::to_vec_pretty(report).map_err(|error| error.to_string())?;
    atomic_write(&path, &bytes).map_err(|error| error.to_string())?;
    Ok(path)
}

pub fn new_report(
    input_fingerprint: u64,
    n_ps: usize,
    n_ifg: usize,
    grid_shape: (usize, usize),
    requested_ifg_workers: usize,
    phases: PhaseTimings,
    solve: super::output::SolveSummary,
) -> Stage6TimingReport {
    Stage6TimingReport {
        schema_version: SCHEMA_VERSION,
        input_fingerprint,
        n_ps,
        n_ifg,
        grid_rows: grid_shape.0,
        grid_cols: grid_shape.1,
        requested_ifg_workers,
        rayon_workers: solve.rayon_workers,
        effective_ifg_workers: solve.effective_ifg_workers,
        resumed_ifgs: solve.resumed_ifgs,
        solved_ifgs: solve.solved_ifgs,
        phases,
        interferograms: solve.ifg_timings,
    }
}
