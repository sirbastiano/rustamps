mod input;
mod output;
mod reestimate;
mod threshold;

#[cfg(test)]
mod tests;

use std::path::Path;

use pystamps_core::stages::stage3::{initial_selection, Stage3Input};

use crate::{PipelineError, RunConfig};

use super::failure;
use super::params::Params;

pub fn run(patch: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    execute(patch).map_err(|error| failure(3, error))
}

fn execute(patch: &Path) -> Result<String, String> {
    let params = Params::load(patch)?;
    let gamma_stdev_reject = params.scalar("gamma_stdev_reject", 0.0)?;
    if gamma_stdev_reject != 0.0 {
        return Err(format!(
            "gamma_stdev_reject={gamma_stdev_reject} requests bootstrap rejection, which is not implemented by the native Stage 3"
        ));
    }
    let small = params.flag("small_baseline_flag", false)?;
    let data = input::load(patch, small)?;
    let threshold = threshold::ThresholdContext::build(&data, &params, small)?;
    let initial_threshold = threshold.calculate(&data, &data.coherence)?;
    let initial = initial_selection(&Stage3Input {
        coherence: data.coherence.clone(),
        k_ps: data.k_ps.clone(),
        c_ps: data.c_ps.clone(),
        amplitude_dispersion: data.amplitude_dispersion.clone(),
        phase_patch: data.phase_patch.clone(),
        phase_residual: data.phase_residual.clone(),
        coherence_threshold: initial_threshold.threshold,
    })
    .map_err(|error| error.to_string())?;
    let reestimate_requested = params.flag("quick_est_gamma_flag", true)?
        && params.flag("select_reest_gamma_flag", true)?;
    let selected_rows = initial
        .selected_ix
        .iter()
        .map(|index| index - 1)
        .collect::<Vec<_>>();
    let (selected, coefficients) = if reestimate_requested && !selected_rows.is_empty() {
        reestimate::run(patch, &data, &params, &threshold, &selected_rows, small)?
    } else {
        (initial, initial_threshold.linear_coefficients)
    };
    let (_, ifg_index) = reestimate::ifg_indices(&data, &params, small)?;
    let selected_count = selected.selected_ix.len();
    output::write(
        patch,
        selected,
        &output::Metadata {
            coefficients: &coefficients,
            clap_alpha: params.scalar("clap_alpha", 1.0)?,
            clap_beta: params.scalar("clap_beta", 0.3)?,
            window: params.scalar("clap_win", 32.0)?,
            maximum_random: threshold.maximum_random,
            gamma_stdev_reject,
            small_baseline: small,
            ifg_index: &ifg_index,
        },
    )?;
    Ok(format!("Stage 3 selected {selected_count} PS"))
}
