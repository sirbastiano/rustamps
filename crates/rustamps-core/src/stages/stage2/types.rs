use crate::stages::stage1::{Complex32, Matrix};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Config {
    pub convergence: f64,
    pub max_iterations: usize,
    pub n_trial_wraps: f64,
}

impl Default for Stage2Config {
    fn default() -> Self {
        Self {
            convergence: 0.005,
            max_iterations: 3,
            n_trial_wraps: 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Input {
    pub phase: Matrix<Complex32>,
    pub bperp_mat: Matrix<f64>,
    pub xy: Matrix<f32>,
    pub amplitude_dispersion: Vec<f64>,
    pub master_ix: usize,
    pub small_baseline: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage2State {
    pub iteration: usize,
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub coherence: Vec<f64>,
    pub previous_coherence: Vec<f64>,
    pub weighting: Vec<f64>,
    pub previous_rms_change: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Iteration {
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub coherence: Vec<f64>,
    pub weighting: Vec<f64>,
    pub phase_residual: Matrix<f32>,
    pub phase_patch: Matrix<Complex32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage2Output {
    pub iterations: usize,
    pub k_ps: Vec<f64>,
    /// Topographic phase coefficients used to build the final filtered grid.
    pub filter_k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub coherence: Vec<f64>,
    /// Weights used to build the final filtered grid.
    pub weighting: Vec<f64>,
    pub phase_residual: Matrix<f32>,
    pub phase_patch: Matrix<Complex32>,
    pub gamma_rms: f64,
    pub gamma_change: f64,
}

pub trait Stage2Kernel {
    fn estimate(
        &mut self,
        input: &Stage2Input,
        state: &Stage2State,
    ) -> Result<Stage2Iteration, Stage2Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage2Error {
    InvalidInput(&'static str),
    Kernel(String),
}

impl Display for Stage2Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid Stage 2 input: {message}"),
            Self::Kernel(message) => write!(f, "Stage 2 kernel failed: {message}"),
        }
    }
}

impl Error for Stage2Error {}
