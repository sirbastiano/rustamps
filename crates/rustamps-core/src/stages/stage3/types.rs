use crate::stages::stage1::{Complex32, Matrix};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq)]
pub struct Stage3Config {
    pub select_method: SelectMethod,
    pub reestimate: bool,
    pub gamma_stdev_reject: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectMethod {
    Density,
    Percent,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage3Input {
    pub coherence: Vec<f64>,
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub amplitude_dispersion: Vec<f64>,
    pub phase_patch: Matrix<Complex32>,
    pub phase_residual: Matrix<f32>,
    pub coherence_threshold: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReestimatedSelection {
    pub source_rows: Vec<usize>,
    pub coherence: Vec<f64>,
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub phase_patch: Matrix<Complex32>,
    pub phase_residual: Matrix<f32>,
    pub coherence_threshold: Vec<f64>,
    pub bperp_range: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage3Output {
    pub selected_ix: Vec<usize>,
    pub keep_ix: Vec<bool>,
    pub coherence: Vec<f64>,
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub phase_patch: Matrix<Complex32>,
    pub phase_residual: Matrix<f32>,
    pub coherence_threshold: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage3Error {
    InvalidInput(&'static str),
    ReestimateRequired(String),
}

impl Display for Stage3Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid Stage 3 input: {message}"),
            Self::ReestimateRequired(message) => {
                write!(f, "Stage 3 re-estimation failed: {message}")
            }
        }
    }
}

impl Error for Stage3Error {}
