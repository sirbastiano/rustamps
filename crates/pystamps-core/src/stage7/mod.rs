//! Pure-Rust Stage 7 spatially-correlated look-angle error estimation.
//!
//! Matrices are row-major. Interferogram and master indices are zero-based.

mod phase;
pub(crate) mod qr;
mod scla;
mod smooth;

use std::fmt;

pub use phase::{center_to_reference, deramp_phase, DerampOutputs};
pub use scla::estimate_scla;
pub use smooth::{build_smoothed_phase, smooth_neighbor_envelope};

#[derive(Clone, Debug)]
pub struct Stage7Inputs<'a> {
    pub ph_proc: &'a [f64],
    pub bperp_mat: &'a [f64],
    pub n_ps: usize,
    pub n_ifg: usize,
    pub unwrap_indices: &'a [usize],
    pub solve_indices: &'a [usize],
    pub day: &'a [f64],
    pub master_index: usize,
    pub ifg_std: &'a [f64],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage7Outputs {
    pub k_ps_uw: Vec<f64>,
    pub c_ps_uw: Vec<f32>,
    pub ph_scla: Vec<f32>,
    pub ifg_vcm: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage7Error(pub(crate) String);

impl Stage7Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Stage7Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Stage7Error {}

#[cfg(test)]
mod tests;
