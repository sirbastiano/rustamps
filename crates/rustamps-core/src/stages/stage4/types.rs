use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq)]
pub struct Stage4Config {
    pub weed_zero_elevation: bool,
    pub weed_standard_dev: f64,
    pub weed_max_noise: f64,
}

impl Default for Stage4Config {
    fn default() -> Self {
        Self {
            weed_zero_elevation: false,
            weed_standard_dev: 1.0,
            weed_max_noise: f64::INFINITY,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage4Input {
    pub selected_ix: Vec<usize>,
    pub selection_keep: Vec<bool>,
    pub height: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage4Measurements {
    pub adjacency_keep: Vec<bool>,
    pub duplicate_keep: Vec<bool>,
    pub ps_std: Vec<f64>,
    pub ps_max: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage4Output {
    pub selected_ix: Vec<usize>,
    pub ix_weed: Vec<bool>,
    pub ix_weed2: Vec<bool>,
    pub ps_std: Vec<f32>,
    pub ps_max: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage4Error {
    InvalidInput(&'static str),
}

impl Display for Stage4Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid Stage 4 input: {message}"),
        }
    }
}

impl Error for Stage4Error {}
