use std::sync::mpsc;
use std::time::Instant;

use pystamps_core::stage6::{
    extract_grid_values, prepare_cost_offsets, select_ifgw, unwrap_grid_profiled, CostOffsetInputs,
    GridUnwrapConfig, GridUnwrapInputs,
};

use super::costs::Costs;
use super::grid::Grid;
use super::input::Input;
use super::interp::Interpolation;
use super::space_time::SpaceTime;
use super::timing::IfgTiming;

type SolveResult = Result<(usize, Vec<f32>, f64, IfgTiming), String>;

pub fn run_batch(
    batch: &[usize],
    input: &Input,
    grid: &Grid,
    interp: &Interpolation,
    space: &SpaceTime,
    costs: &Costs,
    mut accept: impl FnMut(usize, Vec<f32>, f64, IfgTiming) -> Result<(), String> + Send,
) -> Result<(), String> {
    if batch.len() == 1 {
        let (ordinal, values, msd, timing) =
            solve_ifg(batch[0], input, grid, interp, space, costs)?;
        return accept(ordinal, values, msd, timing);
    }
    if input.options.custom_pool {
        return run_rayon_batch(batch, input, grid, interp, space, costs, &mut accept);
    }
    run_global_batch(batch, input, grid, interp, space, costs, &mut accept)
}

#[allow(clippy::too_many_arguments)]
fn run_rayon_batch(
    batch: &[usize],
    input: &Input,
    grid: &Grid,
    interp: &Interpolation,
    space: &SpaceTime,
    costs: &Costs,
    accept: &mut (impl FnMut(usize, Vec<f32>, f64, IfgTiming) -> Result<(), String> + Send),
) -> Result<(), String> {
    rayon::scope(|scope| {
        let (sender, receiver) = mpsc::sync_channel(batch.len());
        for &ordinal in batch {
            let sender = sender.clone();
            scope.spawn(move |_| {
                let result = solve_ifg(ordinal, input, grid, interp, space, costs);
                let _ = sender.send(result);
            });
        }
        drop(sender);
        receive_results(&receiver, batch.len(), accept)
    })
}

#[allow(clippy::too_many_arguments)]
fn run_global_batch(
    batch: &[usize],
    input: &Input,
    grid: &Grid,
    interp: &Interpolation,
    space: &SpaceTime,
    costs: &Costs,
    accept: &mut impl FnMut(usize, Vec<f32>, f64, IfgTiming) -> Result<(), String>,
) -> Result<(), String> {
    std::thread::scope(|scope| {
        let (sender, receiver) = mpsc::sync_channel(batch.len());
        for &ordinal in batch {
            let sender = sender.clone();
            scope.spawn(move || {
                let result = solve_ifg(ordinal, input, grid, interp, space, costs);
                let _ = sender.send(result);
            });
        }
        drop(sender);
        receive_results(&receiver, batch.len(), accept)
    })
}

fn receive_results(
    receiver: &mpsc::Receiver<SolveResult>,
    count: usize,
    accept: &mut impl FnMut(usize, Vec<f32>, f64, IfgTiming) -> Result<(), String>,
) -> Result<(), String> {
    let mut first_error = None;
    for _ in 0..count {
        match receiver.recv() {
            Ok(Ok((ordinal, values, msd, timing))) => {
                if let Err(error) = accept(ordinal, values, msd, timing) {
                    first_error.get_or_insert(error);
                }
            }
            Ok(Err(error)) => {
                first_error.get_or_insert(error);
            }
            Err(_) => {
                first_error.get_or_insert_with(|| {
                    "Stage 6 solve worker stopped without a result".to_owned()
                });
                break;
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn solve_ifg(
    ordinal: usize,
    input: &Input,
    grid: &Grid,
    interp: &Interpolation,
    space: &SpaceTime,
    costs: &Costs,
) -> Result<(usize, Vec<f32>, f64, IfgTiming), String> {
    let total_started = Instant::now();
    let prepare_started = Instant::now();
    let count = input.unwrap.len();
    let wrapped = (0..interp.edges.len())
        .map(|edge| {
            let value = space.unwrapped[edge * count + ordinal];
            value.sin().atan2(value.cos())
        })
        .collect::<Vec<_>>();
    let smooth = (0..interp.edges.len())
        .map(|edge| {
            let index = edge * count + ordinal;
            space.unwrapped[index] - space.noise[index]
        })
        .collect::<Vec<_>>();
    let (rowcost, colcost) = prepare_cost_offsets(&CostOffsetInputs {
        rowcost_base: &costs.row_base,
        colcost_base: &costs.col_base,
        rowix: &costs.row_indices,
        colix: &costs.col_indices,
        row_shape: (grid.rows.saturating_sub(1), grid.cols),
        col_shape: (grid.rows, grid.cols.saturating_sub(1)),
        wrapped_space_uw: &wrapped,
        dph_smooth: &smooth,
        nshortcycle: 200.0,
    })
    .map_err(|error| error.to_string())?;
    let wrapped_grid = select_ifgw(
        &grid.phase,
        grid.n_points,
        count,
        &interp.nearest,
        grid.rows,
        grid.cols,
        ordinal,
    )
    .map_err(|error| error.to_string())?;
    let prepare_sec = prepare_started.elapsed().as_secs_f64();
    let core_started = Instant::now();
    let (output, core_timing) = unwrap_grid_profiled(
        &GridUnwrapInputs {
            ifgw: &wrapped_grid,
            rowcost: &rowcost,
            colcost: &colcost,
            nrow: grid.rows,
            ncol: grid.cols,
        },
        GridUnwrapConfig {
            nshortcycle: 200.0,
            parallel: input.options.parallel,
            max_flow_passes: input.options.max_flow_passes,
        },
    )
    .map_err(|error| error.to_string())?;
    let core_sec = core_started.elapsed().as_secs_f64();
    let extract_started = Instant::now();
    let values = extract_grid_values(&output.ifguw, &grid.mask, grid.rows, grid.cols)
        .map_err(|error| error.to_string())?;
    let extract_sec = extract_started.elapsed().as_secs_f64();
    let timing = IfgTiming {
        solve_ordinal: ordinal + 1,
        ifg_index: input.unwrap[ordinal] + 1,
        resumed: false,
        prepare_sec,
        core_sec,
        decode_sec: core_timing.decode_sec,
        initial_flow_sec: core_timing.initial_flow_sec,
        initial_label_sec: core_timing.initial_label_sec,
        post_flow_sec: core_timing.post_flow_sec,
        final_label_sec: core_timing.final_label_sec,
        msd_sec: core_timing.msd_sec,
        extract_sec,
        total_sec: total_started.elapsed().as_secs_f64(),
    };
    Ok((ordinal, values, output.msd, timing))
}

#[cfg(test)]
mod tests {
    use super::{mpsc, receive_results};

    #[test]
    fn batch_drain_retains_success_after_a_peer_error() {
        let (sender, receiver) = mpsc::channel();
        sender.send(Err("first failed".to_owned())).unwrap();
        sender
            .send(Ok((
                3,
                vec![1.0, 2.0],
                4.0,
                super::IfgTiming::resumed(4, 7),
            )))
            .unwrap();
        drop(sender);
        let mut accepted = Vec::new();

        let error = receive_results(&receiver, 2, &mut |ordinal, values, msd, _| {
            accepted.push((ordinal, values, msd));
            Ok(())
        })
        .unwrap_err();

        assert_eq!(error, "first failed");
        assert_eq!(accepted, [(3, vec![1.0, 2.0], 4.0)]);
    }
}
