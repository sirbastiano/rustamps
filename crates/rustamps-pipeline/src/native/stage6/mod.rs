mod cache_meta;
mod costs;
mod filter;
mod fingerprint;
mod grid;
mod grid_checkpoint;
mod input;
mod interp;
mod nearest;
mod output;
mod schedule;
mod solve;
mod solve_checkpoint;
mod space_time;
mod timing;
mod wrapped;

use std::path::Path;
use std::time::{Duration, Instant};

use crate::{PipelineError, RunConfig};

use super::failure;
use super::params::Params;

pub fn run(root: &Path, config: &RunConfig) -> Result<String, PipelineError> {
    let total_started = Instant::now();
    if config.runtime.stage6_solver != "native" {
        return Err(failure(
            6,
            "standalone Stage 6 supports only the native solver",
        ));
    }
    let started = Instant::now();
    let params = Params::load(root).map_err(|error| failure(6, error))?;
    let input = input::load(root, &params, config).map_err(|error| failure(6, error))?;
    let input_sec = report_phase("input", started.elapsed());
    let started = Instant::now();
    let grid = grid::load_or_build(root, &input).map_err(|error| failure(6, error))?;
    let grid_sec = report_phase("grid", started.elapsed());
    let started = Instant::now();
    let interpolation = interp::load_or_build(root, &grid).map_err(|error| failure(6, error))?;
    let interpolation_sec = report_phase("interpolation", started.elapsed());
    let started = Instant::now();
    let space_time = space_time::load_or_build(root, &input, &grid, &interpolation)
        .map_err(|error| failure(6, error))?;
    let space_time_sec = report_phase("space-time", started.elapsed());
    let started = Instant::now();
    let costs = costs::build(&grid, &interpolation, &space_time, input.unwrap.len());
    let costs_sec = report_phase("costs", started.elapsed());
    let started = Instant::now();
    let solve = output::unwrap_and_write(root, &input, &grid, &interpolation, &space_time, &costs)
        .map_err(|error| failure(6, error))?;
    let solve_output_sec = report_phase("solve/output", started.elapsed());
    let phases = timing::PhaseTimings {
        input_sec,
        grid_sec,
        interpolation_sec,
        space_time_sec,
        costs_sec,
        solve_wall_sec: solve.solve_wall_sec,
        solve_output_sec,
        total_sec: total_started.elapsed().as_secs_f64(),
    };
    let report = timing::new_report(
        input.fingerprint,
        input.n_ps,
        input.n_ifg,
        (grid.rows, grid.cols),
        input.options.ifg_workers,
        phases,
        solve,
    );
    let effective_ifg_workers = report.effective_ifg_workers;
    let timing_detail = match timing::write(root, &report) {
        Ok(path) => {
            eprintln!("Stage 6: timing report {}", path.display());
            path.display().to_string()
        }
        Err(error) => {
            eprintln!("Stage 6: warning: timing report was not written: {error}");
            "unavailable".to_owned()
        }
    };
    Ok(format!(
        "Stage 6 unwrapped {} PS across {} interferograms with {} IFG workers; timing {}",
        input.n_ps, input.n_ifg, effective_ifg_workers, timing_detail
    ))
}

fn report_phase(name: &str, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    eprintln!("Stage 6 timing: {name}={seconds:.3}s");
    seconds
}

#[cfg(test)]
mod resume_tests;
#[cfg(test)]
mod tests;
