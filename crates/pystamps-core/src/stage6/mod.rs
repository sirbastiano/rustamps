//! Pure-Rust Stage 6 phase-unwrapping kernels.
//!
//! Matrix buffers are row-major and public indices are zero-based. Functions
//! that reproduce MATLAB extraction order document that exception explicitly.

mod geometry;
mod grid;
mod la;
mod phase;
pub mod unwrap;

use std::fmt;

pub use geometry::{single_master_ifg_geometry, unwrap_ifg_sets, IfgSets, SingleMasterGeometry};
pub use grid::{extract_grid_values, grid_accumulate, ps_grid_indices, select_ifgw};
pub use la::estimate_la_error_single_master;
pub use phase::{prepare_cost_offsets, reconstruct_ps_phase, CostOffsetInputs};
pub use unwrap::{
    unwrap_grid, unwrap_grid_profiled, GridUnwrapConfig, GridUnwrapInputs, GridUnwrapOutput,
    GridUnwrapTimings,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6Error(String);

impl Stage6Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Stage6Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Stage6Error {}

pub(crate) fn checked_len(rows: usize, cols: usize, name: &str) -> Result<usize, Stage6Error> {
    rows.checked_mul(cols).ok_or_else(|| {
        Stage6Error::new(format!(
            "{name} dimensions overflow the platform index size"
        ))
    })
}

pub(crate) fn require_shape<T>(
    values: &[T],
    rows: usize,
    cols: usize,
    name: &str,
) -> Result<(), Stage6Error> {
    let expected = checked_len(rows, cols, name)?;
    if values.len() != expected {
        return Err(Stage6Error::new(format!(
            "{name} must contain {rows}x{cols} row-major values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
