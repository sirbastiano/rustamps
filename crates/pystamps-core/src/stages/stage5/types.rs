use crate::stages::stage1::{Complex32, Matrix};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq)]
pub struct Stage5PatchInput {
    pub ij: Matrix<f64>,
    pub lonlat: Matrix<f64>,
    pub phase: Matrix<Complex32>,
    pub k_ps: Vec<f64>,
    pub c_ps: Vec<f64>,
    pub coherence: Vec<f64>,
    pub phase_patch: Matrix<Complex32>,
    pub phase_residual: Matrix<f32>,
    pub retain: Vec<bool>,
    pub bperp_mat: Option<Matrix<f32>>,
    pub height: Option<Vec<f32>>,
    pub look_angle: Option<Vec<f64>>,
    pub amplitude_dispersion: Option<Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage5Row {
    pub ij: [f64; 3],
    pub lonlat: [f64; 2],
    pub phase: Vec<Complex32>,
    pub k_ps: f64,
    pub c_ps: f64,
    pub coherence: f64,
    pub phase_patch: Vec<Complex32>,
    pub phase_residual: Vec<f32>,
    pub bperp: Option<Vec<f32>>,
    pub height: Option<f32>,
    pub look_angle: Option<f64>,
    pub amplitude_dispersion: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoOverlapBounds {
    pub row_min: i64,
    pub row_max: i64,
    pub column_min: i64,
    pub column_max: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PromotedPatch {
    pub name: String,
    pub no_overlap: Option<NoOverlapBounds>,
    pub rows: Vec<Stage5Row>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage5Merged {
    pub rows: Vec<Stage5Row>,
    pub xy: Matrix<f32>,
    pub xy_origin: [f64; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage5Error {
    InvalidInput(&'static str),
    UnsupportedResampling,
}

impl Display for Stage5Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid Stage 5 input: {message}"),
            Self::UnsupportedResampling => write!(
                f,
                "nonzero merge_resample_size is unsupported without weighted patch resampling"
            ),
        }
    }
}

impl Error for Stage5Error {}
