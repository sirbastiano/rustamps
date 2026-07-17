mod input;
mod output;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::path::Path;

use rustamps_core::stages::stage1::Matrix;
use rustamps_core::stages::stage4::{
    measure_stage4, weed_stage4, NativeStage4Options, Stage4Config, Stage4Input,
};

use crate::{PipelineError, RunConfig};

use super::failure;
use super::params::Params;

pub fn run(patch: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    execute(patch).map_err(|error| failure(4, error))
}

fn execute(patch: &Path) -> Result<String, String> {
    let params = Params::load(patch)?;
    let small = params.flag("small_baseline_flag", false)?;
    let loaded = input::load(patch, small)?;
    let weed_zero = params.flag("weed_zero_elevation", false)?;
    let weed_std = params.scalar("weed_standard_dev", if small { f64::INFINITY } else { 1.0 })?;
    let weed_max = params.scalar("weed_max_noise", f64::INFINITY)?;
    let (ifg_columns, ifg_index) = ifg_indices(&loaded, &params)?;
    let active = loaded
        .selection_keep
        .iter()
        .enumerate()
        .filter_map(|(row, &keep)| keep.then_some((row, loaded.selected_ix[row] - 1)))
        .collect::<Vec<_>>();
    let ij = Matrix {
        rows: active.len(),
        cols: 2,
        values: active
            .iter()
            .flat_map(|&(_, source)| [loaded.ij.row(source)[1], loaded.ij.row(source)[2]])
            .collect(),
    };
    let xy = Matrix {
        rows: active.len(),
        cols: 2,
        values: active
            .iter()
            .flat_map(|&(_, source)| [loaded.xy.row(source)[1], loaded.xy.row(source)[2]])
            .collect(),
    };
    let phase = Matrix {
        rows: active.len(),
        cols: loaded.n_ifg,
        values: active
            .iter()
            .flat_map(|&(_, source)| loaded.phase.row(source).iter().copied())
            .collect(),
    };
    let coherence = active
        .iter()
        .map(|&(row, _)| loaded.coherence[row])
        .collect::<Vec<_>>();
    let k_ps = active
        .iter()
        .map(|&(row, _)| loaded.k_ps[row])
        .collect::<Vec<_>>();
    let c_ps = active
        .iter()
        .map(|&(row, _)| loaded.c_ps[row])
        .collect::<Vec<_>>();
    let active_height = loaded.height.as_ref().map(|height| {
        active
            .iter()
            .map(|&(_, source)| height[source])
            .collect::<Vec<_>>()
    });
    let options = NativeStage4Options {
        weed_neighbours: params.flag("weed_neighbours", false)?,
        weed_zero_elevation: weed_zero,
        weed_standard_dev: weed_std,
        weed_max_noise: weed_max,
        weed_time_window: params.scalar("weed_time_win", 730.0)?,
        small_baseline: small,
        master_ix: loaded.master + 1,
        interferogram_indices: ifg_columns,
    };
    let measurements = measure_stage4(
        &ij,
        &xy,
        &coherence,
        active_height.as_deref(),
        &phase,
        &k_ps,
        &c_ps,
        &loaded.bperp,
        &loaded.day,
        &options,
    )
    .map_err(|error| error.to_string())?;
    let result = weed_stage4(
        &Stage4Input {
            selected_ix: loaded.selected_ix,
            selection_keep: loaded.selection_keep,
            height: loaded.height,
        },
        &measurements,
        &Stage4Config {
            weed_zero_elevation: weed_zero,
            weed_standard_dev: weed_std,
            weed_max_noise: weed_max,
        },
    )
    .map_err(|error| error.to_string())?;
    let retained = result.ix_weed.iter().filter(|&&keep| keep).count();
    let selected = result.ix_weed.len();
    output::write(patch, result, &ifg_index)?;
    Ok(format!(
        "Stage 4 retained {retained}/{selected} selected PS"
    ))
}

fn ifg_indices(loaded: &input::Loaded, params: &Params) -> Result<(Vec<usize>, Vec<f64>), String> {
    let dropped = params
        .indices("drop_ifg_index")?
        .into_iter()
        .collect::<BTreeSet<_>>();
    if dropped.iter().any(|&index| index >= loaded.n_ifg) {
        return Err("drop_ifg_index exceeds the interferogram count".to_owned());
    }
    let columns = (0..loaded.n_ifg)
        .filter(|index| !dropped.contains(index))
        .collect::<Vec<_>>();
    let one_based = columns.iter().map(|index| (index + 1) as f64).collect();
    Ok((columns, one_based))
}
