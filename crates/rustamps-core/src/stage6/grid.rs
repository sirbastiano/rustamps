use num_complex::Complex32;

use super::{checked_len, require_shape, Stage6Error};

pub fn grid_accumulate(
    ph_in: &[Complex32],
    n_ps: usize,
    n_ifg: usize,
    grid_indices: &[usize],
    n_cells: usize,
) -> Result<Vec<Complex32>, Stage6Error> {
    require_shape(ph_in, n_ps, n_ifg, "ph_in")?;
    if grid_indices.len() != n_ps {
        return Err(Stage6Error::new(
            "grid_indices must contain one cell index per phase row",
        ));
    }
    if grid_indices.iter().any(|&index| index >= n_cells) {
        return Err(Stage6Error::new(
            "grid_indices contains an out-of-range cell",
        ));
    }
    let output_len = checked_len(n_cells, n_ifg, "grid output")?;
    let mut output = vec![Complex32::new(0.0, 0.0); output_len];
    for row in 0..n_ps {
        let cell = grid_indices[row];
        for column in 0..n_ifg {
            output[cell * n_ifg + column] += ph_in[row * n_ifg + column];
        }
    }
    Ok(output)
}

/// Extract active grid values in MATLAB column-major mask traversal order.
pub fn extract_grid_values(
    grid: &[f32],
    mask: &[bool],
    nrow: usize,
    ncol: usize,
) -> Result<Vec<f32>, Stage6Error> {
    require_shape(grid, nrow, ncol, "grid")?;
    require_shape(mask, nrow, ncol, "mask")?;
    let mut output = Vec::with_capacity(mask.iter().filter(|&&keep| keep).count());
    for col in 0..ncol {
        for row in 0..nrow {
            let index = row * ncol + col;
            if mask[index] {
                output.push(grid[index]);
            }
        }
    }
    Ok(output)
}

/// Map zero-based grid coordinates to column-major active-cell identifiers.
pub fn ps_grid_indices(
    mask: &[bool],
    nrow: usize,
    ncol: usize,
    grid_ij: &[[usize; 2]],
) -> Result<Vec<Option<usize>>, Stage6Error> {
    require_shape(mask, nrow, ncol, "mask")?;
    if grid_ij
        .iter()
        .any(|point| point[0] >= nrow || point[1] >= ncol)
    {
        return Err(Stage6Error::new(
            "grid_ij contains a coordinate outside the grid",
        ));
    }
    let mut active = vec![None; nrow * ncol];
    let mut next = 0_usize;
    for col in 0..ncol {
        for row in 0..nrow {
            let index = row * ncol + col;
            if mask[index] {
                active[index] = Some(next);
                next += 1;
            }
        }
    }
    Ok(grid_ij
        .iter()
        .map(|point| active[point[0] * ncol + point[1]])
        .collect())
}

pub fn select_ifgw(
    uw_ph: &[Complex32],
    n_grid_ps: usize,
    n_ifg: usize,
    z: &[usize],
    nrow: usize,
    ncol: usize,
    ifg_index: usize,
) -> Result<Vec<Complex32>, Stage6Error> {
    require_shape(uw_ph, n_grid_ps, n_ifg, "uw_ph")?;
    require_shape(z, nrow, ncol, "z")?;
    if ifg_index >= n_ifg {
        return Err(Stage6Error::new(
            "ifg_index must be within the phase columns",
        ));
    }
    if z.iter().any(|&index| index >= n_grid_ps) {
        return Err(Stage6Error::new(
            "z contains an out-of-range phase-row index",
        ));
    }
    Ok(z.iter()
        .map(|&index| uw_ph[index * n_ifg + ifg_index])
        .collect())
}
