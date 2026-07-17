use pystamps_core::stages::stage2::{
    butterworth_low_pass, NativeStage2Options, NativeWeighting, PsquareReference, Stage2Config,
};

use super::super::params::Params;
use super::input::Loaded;

const STAMPS_REFERENCE_RANGE_M: f64 = 830_000.0;

pub struct LoadedOptions {
    pub config: Stage2Config,
    pub grid_size: f32,
    pub clap_alpha: f64,
    pub clap_beta: f64,
    pub clap_window: usize,
    pub clap_padding: usize,
    pub low_pass_size: usize,
    pub filter_weighting: FilterWeighting,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterWeighting {
    Psquare,
    SignalNoise,
}

impl LoadedOptions {
    pub fn load(input: &Loaded, params: &Params) -> Result<Self, String> {
        let grid_size = params.scalar("filter_grid_size", 50.0)?;
        let clap_win = params.scalar("clap_win", 32.0)?;
        let clap_alpha = params.scalar("clap_alpha", 1.0)?;
        let clap_beta = params.scalar("clap_beta", 0.3)?;
        let wavelength = params.scalar("clap_low_pass_wavelength", 800.0)?;
        for (name, value) in [
            ("filter_grid_size", grid_size),
            ("clap_win", clap_win),
            ("clap_low_pass_wavelength", wavelength),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("parameter {name} must be finite and positive"));
            }
        }
        if !clap_alpha.is_finite() || !clap_beta.is_finite() {
            return Err("CLAP alpha and beta must be finite".to_owned());
        }
        let clap_window = (clap_win * 0.75).round() as usize;
        let clap_padding = (clap_win * 0.25).round() as usize;
        let low_pass_size = clap_window + clap_padding;
        if clap_window == 0 || clap_window % 2 != 0 || low_pass_size == 0 {
            return Err(format!(
                "clap_win={clap_win} produces unsupported window={clap_window}, padding={clap_padding}"
            ));
        }
        let max_iterations = integer_parameter(
            "gamma_max_iterations",
            params.scalar("gamma_max_iterations", 3.0)?,
        )?;
        if max_iterations == 0 {
            return Err("gamma_max_iterations must be positive".to_owned());
        }
        let convergence = params.scalar("gamma_change_convergence", 0.005)?;
        if !convergence.is_finite() || convergence < 0.0 {
            return Err("gamma_change_convergence must be finite and non-negative".to_owned());
        }
        let filter_weighting =
            parse_filter_weighting(&params.text("filter_weighting", "P-square")?)?;
        let n_trial_wraps = trial_wraps(input, params)?;
        Ok(Self {
            config: Stage2Config {
                convergence,
                max_iterations,
                n_trial_wraps,
            },
            grid_size: grid_size as f32,
            clap_alpha,
            clap_beta,
            clap_window,
            clap_padding,
            low_pass_size,
            filter_weighting,
        })
    }

    pub fn native(&self, reference: PsquareReference, wavelength: f64) -> NativeStage2Options {
        NativeStage2Options {
            grid_size: self.grid_size,
            clap_alpha: self.clap_alpha,
            clap_beta: self.clap_beta,
            clap_window: self.clap_window,
            clap_padding: self.clap_padding,
            low_pass: butterworth_low_pass(
                self.low_pass_size,
                f64::from(self.grid_size),
                wavelength,
            ),
            n_trial_wraps: self.config.n_trial_wraps,
            weighting: match self.filter_weighting {
                FilterWeighting::Psquare => NativeWeighting::Psquare(reference),
                FilterWeighting::SignalNoise => NativeWeighting::SignalNoise,
            },
        }
    }
}

fn parse_filter_weighting(value: &str) -> Result<FilterWeighting, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "p-square" | "psquare" => Ok(FilterWeighting::Psquare),
        "snr" | "signal-noise" | "signal_noise" | "signal noise" => {
            Ok(FilterWeighting::SignalNoise)
        }
        value => Err(format!("unsupported filter_weighting={value}")),
    }
}

pub fn trial_wraps(input: &Loaded, params: &Params) -> Result<f64, String> {
    let max_topo_err = params.scalar("max_topo_err", 20.0)?;
    let wavelength = params.scalar("lambda", f64::NAN)?;
    if !max_topo_err.is_finite() || max_topo_err < 0.0 {
        return Err("max_topo_err must be finite and non-negative".to_owned());
    }
    if !wavelength.is_finite() || wavelength <= 0.0 {
        return Err("lambda must be present, finite, and positive".to_owned());
    }
    let sine = input.mean_incidence.sin().abs();
    if !sine.is_finite() || sine <= 1e-12 {
        return Err("mean incidence produces a singular topographic scale".to_owned());
    }
    let low = input
        .nominal_bperp
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let high = input
        .nominal_bperp
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !low.is_finite() || !high.is_finite() {
        return Err("ps1.bperp contains no finite non-master baseline".to_owned());
    }
    let max_k = max_topo_err
        / (wavelength * STAMPS_REFERENCE_RANGE_M * sine / (4.0 * std::f64::consts::PI));
    Ok((high - low) * max_k / (2.0 * std::f64::consts::PI))
}

fn integer_parameter(name: &str, value: f64) -> Result<usize, String> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        Err(format!("parameter {name} must be a non-negative integer"))
    } else {
        Ok(value as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_filter_weighting, FilterWeighting};

    #[test]
    fn documented_snr_weighting_alias_is_accepted() {
        assert_eq!(
            parse_filter_weighting("SNR").unwrap(),
            FilterWeighting::SignalNoise
        );
    }
}
