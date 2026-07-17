use std::error::Error;
use std::fmt::{Display, Formatter};

pub use num_complex::Complex32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterShape {
    pub length: usize,
    pub width: usize,
}

impl RasterShape {
    pub fn cells(self) -> Result<usize, PrepError> {
        if self.length == 0 || self.width == 0 {
            return Err(PrepError::InvalidShape(self));
        }
        self.length
            .checked_mul(self.width)
            .ok_or(PrepError::InvalidShape(self))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtPrepOptions {
    pub amp_dispersion: f64,
    pub range_patches: usize,
    pub azimuth_patches: usize,
    pub range_overlap: usize,
    pub azimuth_overlap: usize,
}

impl Default for MtPrepOptions {
    fn default() -> Self {
        Self {
            amp_dispersion: 0.4,
            range_patches: 1,
            azimuth_patches: 1,
            range_overlap: 50,
            azimuth_overlap: 50,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SnapPrepInput<'a> {
    pub shape: RasterShape,
    pub rslc_amplitudes: &'a [&'a [f64]],
    pub diff_phase: &'a [&'a [Complex32]],
    pub lon: &'a [f32],
    pub lat: &'a [f32],
    pub height: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateStats {
    pub selected: Vec<bool>,
    pub amplitude_dispersion: Vec<f32>,
    pub normalized_amplitude_sum: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InclusiveRange {
    pub start: usize,
    pub end: usize,
}

impl InclusiveRange {
    pub fn contains(self, one_based: usize) -> bool {
        one_based >= self.start && one_based <= self.end
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchBounds {
    pub columns: InclusiveRange,
    pub rows: InclusiveRange,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatchCandidate {
    pub source_index: usize,
    pub row: usize,
    pub column: usize,
    pub lon: f32,
    pub lat: f32,
    pub height: f32,
    pub amplitude_dispersion: f32,
    pub phase: Vec<Complex32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedPatch {
    pub name: String,
    pub bounds: PatchBounds,
    pub no_overlap: PatchBounds,
    pub candidates: Vec<PatchCandidate>,
    pub mean_amplitude: Vec<f32>,
    pub mean_amplitude_shape: RasterShape,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MtPrepOutput {
    pub patches: Vec<PreparedPatch>,
    pub candidate_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrepError {
    InvalidShape(RasterShape),
    InvalidOption(&'static str),
    EmptyAcquisitionStack,
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NoCandidates,
}

impl Display for PrepError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape(shape) => write!(
                f,
                "invalid raster shape: length={} width={}",
                shape.length, shape.width
            ),
            Self::InvalidOption(name) => write!(f, "invalid mt_prep option: {name}"),
            Self::EmptyAcquisitionStack => write!(f, "at least one RSLC acquisition is required"),
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "{field} length mismatch: expected {expected}, found {actual}"
            ),
            Self::NoCandidates => {
                write!(f, "no candidates passed the amplitude-dispersion threshold")
            }
        }
    }
}

impl Error for PrepError {}
