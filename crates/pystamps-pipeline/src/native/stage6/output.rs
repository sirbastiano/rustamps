use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use pystamps_core::stage6::{ps_grid_indices, reconstruct_ps_phase};
use pystamps_io::{write_mat, StageTransaction};

use super::super::mat::{f32_array, f64_array};
use super::costs::Costs;
use super::grid::Grid;
use super::input::Input;
use super::interp::Interpolation;
use super::schedule;
use super::solve;
use super::solve_checkpoint::{self, Solution};
use super::space_time::SpaceTime;
use super::timing::IfgTiming;

pub struct SolveSummary {
    pub rayon_workers: usize,
    pub effective_ifg_workers: usize,
    pub resumed_ifgs: usize,
    pub solved_ifgs: usize,
    pub solve_wall_sec: f64,
    pub ifg_timings: Vec<IfgTiming>,
}

pub fn unwrap_and_write(
    root: &Path,
    input: &Input,
    grid: &Grid,
    interp: &Interpolation,
    space: &SpaceTime,
    costs: &Costs,
) -> Result<SolveSummary, String> {
    let count = input.unwrap.len();
    let mut grid_phase = vec![0.0_f32; grid.n_points * count];
    let mut selected_msd = vec![0.0_f64; count];
    let mut missing = Vec::new();
    let mut ifg_timings = Vec::with_capacity(count);
    for ordinal in 0..count {
        match solve_checkpoint::load(root, input, grid, ordinal) {
            Some(solution) => {
                scatter_solution(
                    &mut grid_phase,
                    &mut selected_msd,
                    count,
                    ordinal,
                    &solution,
                );
                ifg_timings.push(IfgTiming::resumed(ordinal + 1, input.unwrap[ordinal] + 1));
            }
            None => missing.push(ordinal),
        }
    }
    let resumed_ifgs = count - missing.len();
    let solved_ifgs = missing.len();
    let mut complete = resumed_ifgs;
    if complete > 0 {
        eprintln!("Stage 6: resumed {complete}/{count} interferogram solves");
    }
    let schedule = schedule::choose(
        input.options.ifg_workers,
        grid.rows,
        grid.cols,
        missing.len(),
    );
    let requested = if input.options.ifg_workers == 0 {
        "auto".to_owned()
    } else {
        input.options.ifg_workers.to_string()
    };
    eprintln!(
        "Stage 6: IFG scheduler requested={requested}, Rayon workers={}, effective workers={}, grid={}x{}",
        schedule.rayon_workers, schedule.effective_ifg_workers, grid.rows, grid.cols
    );
    let solve_started = Instant::now();
    for batch in missing.chunks(schedule.effective_ifg_workers.max(1)) {
        let batch_started = Instant::now();
        solve::run_batch(
            batch,
            input,
            grid,
            interp,
            space,
            costs,
            |ifg, values, msd, timing| {
                let solution = Solution { values, msd };
                solve_checkpoint::write(root, input, grid, ifg, &solution)?;
                scatter_solution(&mut grid_phase, &mut selected_msd, count, ifg, &solution);
                complete += 1;
                eprintln!(
                    "Stage 6: checkpointed {complete}/{count} (IFG {}): prepare={:.3}s core={:.3}s extract={:.3}s total={:.3}s; batch={:.2}s",
                    timing.ifg_index,
                    timing.prepare_sec,
                    timing.core_sec,
                    timing.extract_sec,
                    timing.total_sec,
                    batch_started.elapsed().as_secs_f64()
                );
                ifg_timings.push(timing);
                Ok(())
            },
        )?;
    }
    let solve_wall_sec = if solved_ifgs == 0 {
        0.0
    } else {
        solve_started.elapsed().as_secs_f64()
    };
    let ps_grid = ps_grid_indices(&grid.mask, grid.rows, grid.cols, &grid.coordinates)
        .map_err(|error| error.to_string())?;
    let restore = (0..input.n_ps)
        .flat_map(|row| {
            input
                .unwrap
                .iter()
                .map(move |&ifg| input.phase_restore[row * input.n_ifg + ifg])
        })
        .collect::<Vec<_>>();
    let selected = reconstruct_ps_phase(
        &grid_phase,
        grid.n_points,
        count,
        &ps_grid,
        &grid.phase_in,
        Some(&restore),
    )
    .map_err(|error| error.to_string())?;
    let mut phase = vec![0.0_f32; input.n_ps * input.n_ifg];
    let mut msd = vec![0.0_f32; input.n_ifg];
    for row in 0..input.n_ps {
        for (column, &ifg) in input.unwrap.iter().enumerate() {
            phase[row * input.n_ifg + ifg] = selected[row * count + column];
        }
    }
    for (column, &ifg) in input.unwrap.iter().enumerate() {
        msd[ifg] = selected_msd[column] as f32;
    }
    write_outputs(root, input, grid_phase, selected_msd, phase, msd)?;
    ifg_timings.sort_by_key(|timing| timing.solve_ordinal);
    Ok(SolveSummary {
        rayon_workers: schedule.rayon_workers,
        effective_ifg_workers: schedule.effective_ifg_workers,
        resumed_ifgs,
        solved_ifgs,
        solve_wall_sec,
        ifg_timings,
    })
}

fn scatter_solution(
    grid_phase: &mut [f32],
    selected_msd: &mut [f64],
    count: usize,
    ordinal: usize,
    solution: &Solution,
) {
    for (row, &value) in solution.values.iter().enumerate() {
        grid_phase[row * count + ordinal] = value;
    }
    selected_msd[ordinal] = solution.msd;
}

fn write_outputs(
    root: &Path,
    input: &Input,
    grid_phase: Vec<f32>,
    selected_msd: Vec<f64>,
    phase: Vec<f32>,
    msd: Vec<f32>,
) -> Result<(), String> {
    let transaction = StageTransaction::begin(root, "stage6").map_err(|error| error.to_string())?;
    let mut checkpoint = BTreeMap::new();
    checkpoint.insert(
        "ph_uw".to_owned(),
        f32_array(
            vec![grid_phase.len() / input.unwrap.len(), input.unwrap.len()],
            grid_phase,
        ),
    );
    checkpoint.insert(
        "msd".to_owned(),
        f64_array(vec![input.unwrap.len(), 1], selected_msd),
    );
    write_mat(transaction.path("uw_phaseuw.mat"), &checkpoint)
        .map_err(|error| error.to_string())?;
    let mut final_output = BTreeMap::new();
    final_output.insert(
        "ph_uw".to_owned(),
        f32_array(vec![input.n_ps, input.n_ifg], phase),
    );
    final_output.insert("msd".to_owned(), f32_array(vec![input.n_ifg, 1], msd));
    write_mat(transaction.path("phuw2.mat"), &final_output).map_err(|error| error.to_string())?;
    transaction
        .commit(&["uw_phaseuw.mat", "phuw2.mat"], "phuw2.mat")
        .map_err(|error| error.to_string())
}
