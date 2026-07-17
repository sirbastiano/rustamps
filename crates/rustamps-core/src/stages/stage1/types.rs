use std::error::Error;
use std::fmt::{Display, Formatter};

pub use num_complex::Complex32;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn new(rows: usize, cols: usize, values: Vec<T>) -> Result<Self, Stage1Error> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(Stage1Error::InvalidMatrix("matrix size overflow"))?;
        if values.len() != expected {
            return Err(Stage1Error::LengthMismatch {
                field: "matrix values",
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { rows, cols, values })
    }

    pub fn row(&self, row: usize) -> &[T] {
        &self.values[row * self.cols..(row + 1) * self.cols]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage1Input {
    pub ij: Matrix<f64>,
    pub phase: Matrix<Complex32>,
    pub lonlat: Matrix<f64>,
    pub day_yyyymmdd: Vec<i32>,
    pub master_day_yyyymmdd: i32,
    pub bperp: Vec<f64>,
    pub per_pixel_bperp: Option<Matrix<f32>>,
    pub amplitude_dispersion: Option<Vec<f64>>,
    pub height: Option<Vec<f32>>,
    pub heading_deg: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage1Output {
    pub ij: Matrix<f64>,
    pub phase: Matrix<Complex32>,
    pub lonlat: Matrix<f64>,
    pub xy: Matrix<f32>,
    pub day: Vec<f64>,
    pub master_day: f64,
    pub master_ix: usize,
    pub bperp: Vec<f32>,
    pub bperp_mat: Matrix<f32>,
    pub sort_ix: Vec<usize>,
    pub ll0: [f64; 2],
    pub amplitude_dispersion: Option<Vec<f64>>,
    pub height: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stage1Error {
    InvalidDate(i32),
    DuplicateMaster,
    NoCandidates,
    InvalidMatrix(&'static str),
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl Display for Stage1Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDate(value) => write!(f, "invalid YYYYMMDD date: {value}"),
            Self::DuplicateMaster => {
                write!(f, "input contains duplicate master-day interferograms")
            }
            Self::NoCandidates => write!(f, "no candidates remain after NaN filtering"),
            Self::InvalidMatrix(message) => write!(f, "invalid Stage 1 matrix: {message}"),
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => write!(
                f,
                "{field} length mismatch: expected {expected}, found {actual}"
            ),
        }
    }
}

impl Error for Stage1Error {}

pub(crate) fn check_len(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), Stage1Error> {
    if actual == expected {
        Ok(())
    } else {
        Err(Stage1Error::LengthMismatch {
            field,
            expected,
            actual,
        })
    }
}
