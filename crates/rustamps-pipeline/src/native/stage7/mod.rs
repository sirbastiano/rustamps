mod input;
mod output;
mod reference;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::path::Path;

use rustamps_core::stage7::{
    build_smoothed_phase, center_to_reference, deramp_phase, estimate_scla,
    smooth_neighbor_envelope, Stage7Inputs,
};
use rustamps_io::read_mat;

use crate::{PipelineError, RunConfig};

use super::mat::numeric_f64;
use super::params::Params;
use super::{failure, topology};

pub fn run(root: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    let params = Params::load(root).map_err(|error| failure(7, error))?;
    let scla_method = params
        .text("scla_method", "L2")
        .map_err(|error| failure(7, error))?;
    if !scla_method.eq_ignore_ascii_case("L2") {
        return Err(failure(
            7,
            format!("scla_method={scla_method} is unsupported; native Stage 7 supports L2 only"),
        ));
    }
    if params
        .flag("subtr_tropo", false)
        .map_err(|error| failure(7, error))?
    {
        return Err(failure(
            7,
            "subtr_tropo='y' is unsupported by native Stage 7",
        ));
    }
    if params
        .flag("small_baseline_flag", false)
        .map_err(|error| failure(7, error))?
    {
        return Err(failure(
            7,
            "small-baseline Stage 7 requires the three-pass workflow and is not yet supported",
        ));
    }
    reference::validate(&params).map_err(|error| failure(7, error))?;
    if root.join("aps2.mat").is_file() {
        return Err(failure(
            7,
            "aps2.mat atmosphere subtraction is unsupported by native Stage 7",
        ));
    }
    let scla_deramp = params
        .flag("scla_deramp", false)
        .map_err(|error| failure(7, error))?;
    if scla_deramp {
        validate_deramp_degree(root).map_err(|error| failure(7, error))?;
    }
    let loaded = input::load(root).map_err(|error| failure(7, error))?;
    let mut phase = loaded
        .ph_uw
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let ph_ramp = if scla_deramp {
        let result = deramp_phase(&phase, &loaded.xy, loaded.n_ps, loaded.n_ifg)
            .map_err(|error| failure(7, error))?;
        phase = result.phase;
        result.ramp
    } else {
        Vec::new()
    };
    let reference = reference::select(&loaded, &params).map_err(|error| failure(7, error))?;
    let phase = center_to_reference(&phase, loaded.n_ps, loaded.n_ifg, &reference)
        .map_err(|error| failure(7, error))?;
    let dropped = dropped_indices(&params, loaded.n_ifg).map_err(|error| failure(7, error))?;
    let unwrap_indices = (0..loaded.n_ifg)
        .filter(|index| !dropped.contains(index))
        .collect::<Vec<_>>();
    let solve_indices = unwrap_indices
        .iter()
        .copied()
        .filter(|&index| index != loaded.master_index)
        .collect::<Vec<_>>();
    let estimated = estimate_scla(&Stage7Inputs {
        ph_proc: &phase,
        bperp_mat: &loaded.bperp,
        n_ps: loaded.n_ps,
        n_ifg: loaded.n_ifg,
        unwrap_indices: &unwrap_indices,
        solve_indices: &solve_indices,
        day: &loaded.day,
        master_index: loaded.master_index,
        ifg_std: &loaded.ifg_std,
    })
    .map_err(|error| failure(7, error))?;
    let edges = topology::delaunay_edges(&loaded.xy).map_err(|error| failure(7, error))?;
    let (smooth_k, smooth_c) =
        smooth_neighbor_envelope(&estimated.k_ps_uw, &estimated.c_ps_uw, &edges)
            .map_err(|error| failure(7, error))?;
    let smooth_phase = build_smoothed_phase(&smooth_k, &loaded.bperp, loaded.n_ifg)
        .map_err(|error| failure(7, error))?;
    output::write(
        root,
        loaded.n_ps,
        loaded.n_ifg,
        estimated,
        &ph_ramp,
        smooth_k,
        smooth_c,
        smooth_phase,
    )
    .map_err(|error| failure(7, error))?;
    Ok(format!("Stage 7 estimated SCLA for {} PS", loaded.n_ps))
}

fn validate_deramp_degree(root: &Path) -> Result<(), String> {
    let path = root.join("deramp_degree.mat");
    if !path.is_file() {
        return Ok(());
    }
    let file = read_mat(path).map_err(|error| error.to_string())?;
    let degree = numeric_f64(&file, "degree")?;
    if degree.len() != 1 || degree[0] != 1.0 {
        return Err("native Stage 7 supports only deramp degree 1".to_owned());
    }
    Ok(())
}

fn dropped_indices(params: &Params, n_ifg: usize) -> Result<BTreeSet<usize>, String> {
    let mut dropped = params.indices("drop_ifg_index")?;
    dropped.extend(params.indices("scla_drop_index")?);
    if dropped.iter().any(|&index| index >= n_ifg) {
        return Err("Stage 7 drop index exceeds interferogram count".to_owned());
    }
    Ok(dropped.into_iter().collect())
}
