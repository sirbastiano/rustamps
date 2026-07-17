use rustamps_core::stages::stage3::{
    coherence_threshold, da_bin_edges, CoherenceThresholdInput, CoherenceThresholdOutput,
    SelectMethod,
};

use super::super::params::Params;
use super::input::Initial;

pub(super) struct ThresholdContext {
    pub edges: Vec<f64>,
    pub dispersion: Vec<f64>,
    pub maximum_random: f64,
    pub method: SelectMethod,
    pub low_bins: usize,
}

impl ThresholdContext {
    pub fn build(data: &Initial, params: &Params, small: bool) -> Result<Self, String> {
        let method = match params
            .text("select_method", "DENSITY")?
            .to_ascii_uppercase()
            .as_str()
        {
            "DENSITY" => SelectMethod::Density,
            "PERCENT" => SelectMethod::Percent,
            value => return Err(format!("unsupported select_method {value}")),
        };
        let (edges, dispersion) = da_bin_edges(&data.amplitude_dispersion);
        let maximum_random = match method {
            SelectMethod::Percent => {
                params.scalar("percent_rand", if small { 1.0 } else { 20.0 })?
            }
            SelectMethod::Density => {
                let density = params.scalar("density_rand", if small { 2.0 } else { 20.0 })?;
                density * patch_area(data) / (edges.len() - 1).max(1) as f64
            }
        };
        if !maximum_random.is_finite() || maximum_random < 0.0 {
            return Err("selection random threshold must be finite and non-negative".to_owned());
        }
        Ok(Self {
            edges,
            dispersion,
            maximum_random,
            method,
            low_bins: if small { 15 } else { 31 },
        })
    }

    pub fn calculate(
        &self,
        data: &Initial,
        coherence: &[f64],
    ) -> Result<CoherenceThresholdOutput, String> {
        coherence_threshold(&CoherenceThresholdInput {
            coherence,
            amplitude_dispersion: &self.dispersion,
            dispersion_edges: &self.edges,
            coherence_bins: &data.coherence_bins,
            random_distribution: &data.random_distribution,
            low_coherence_bins: self.low_bins,
            maximum_random: self.maximum_random,
            method: self.method,
        })
        .map_err(|error| error.to_string())
    }
}

fn patch_area(data: &Initial) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for row in 0..data.xy.rows {
        let values = data.xy.row(row);
        min_x = min_x.min(values[1]);
        max_x = max_x.max(values[1]);
        min_y = min_y.min(values[2]);
        max_y = max_y.max(values[2]);
    }
    let area = (max_x - min_x) * (max_y - min_y) / 1_000_000.0;
    if area.is_finite() && area > 0.0 {
        area
    } else {
        1.0
    }
}
