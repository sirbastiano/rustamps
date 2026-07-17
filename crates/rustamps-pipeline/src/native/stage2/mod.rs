mod input;
mod options;
mod output;
mod reference;

use std::path::Path;
use std::time::Instant;

use rustamps_core::stages::stage2::run_stage2_native_detailed;

use crate::{PipelineError, RunConfig};

use super::{failure, params::Params};

pub fn run(patch: &Path, _config: &RunConfig) -> Result<String, PipelineError> {
    let started = Instant::now();
    let params = Params::load(patch).map_err(|error| failure(2, error))?;
    if !params
        .flag("quick_est_gamma_flag", true)
        .map_err(|error| failure(2, error))?
    {
        return Err(failure(
            2,
            "quick_est_gamma_flag='n' requests the unavailable legacy full gamma estimator",
        ));
    }
    if params
        .flag("small_baseline_flag", false)
        .map_err(|error| failure(2, error))?
    {
        return Err(failure(
            2,
            "small-baseline Stage 2 is not supported by the native solver",
        ));
    }
    let loaded = input::load(patch).map_err(|error| failure(2, error))?;
    let options =
        options::LoadedOptions::load(&loaded, &params).map_err(|error| failure(2, error))?;
    let reference = match options.filter_weighting {
        options::FilterWeighting::Psquare => {
            reference::load_or_generate(patch, &loaded.nominal_bperp, options.config.n_trial_wraps)
                .map_err(|error| failure(2, error))?
        }
        options::FilterWeighting::SignalNoise => {
            reference::signal_noise_schema(&loaded.nominal_bperp)
        }
    };
    let wavelength = params
        .scalar("clap_low_pass_wavelength", 800.0)
        .map_err(|error| failure(2, error))?;
    let native_options = options.native(reference.model.clone(), wavelength);
    let result = run_stage2_native_detailed(&loaded.input, &options.config, native_options.clone())
        .map_err(|error| failure(2, error))?;
    output::write(
        patch,
        &loaded.input,
        &native_options,
        &reference.model,
        reference.bperp_fingerprint,
        &result,
    )
    .map_err(|error| failure(2, error))?;
    Ok(format!(
        "Stage 2 computed coherence for {} candidates in {:.3}s (random reference {})",
        loaded.input.phase.rows,
        started.elapsed().as_secs_f64(),
        match options.filter_weighting {
            options::FilterWeighting::SignalNoise => "bypassed for SNR",
            options::FilterWeighting::Psquare if reference.cache_hit => "cached",
            options::FilterWeighting::Psquare => "generated",
        }
    ))
}

#[cfg(test)]
mod snr_tests;
#[cfg(test)]
mod tests;
