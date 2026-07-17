use num_complex::Complex32;

use super::{checked_len, require_shape, Stage6Error};

#[derive(Clone, Debug)]
pub struct CostOffsetInputs<'a> {
    pub rowcost_base: &'a [i16],
    pub colcost_base: &'a [i16],
    pub rowix: &'a [f64],
    pub colix: &'a [f64],
    pub row_shape: (usize, usize),
    pub col_shape: (usize, usize),
    pub wrapped_space_uw: &'a [f32],
    pub dph_smooth: &'a [f32],
    pub nshortcycle: f64,
}

pub fn prepare_cost_offsets(
    input: &CostOffsetInputs<'_>,
) -> Result<(Vec<i16>, Vec<i16>), Stage6Error> {
    let (row_count, row_arcs) = input.row_shape;
    let (col_count, col_arcs) = input.col_shape;
    let row_cost_width = checked_len(row_arcs, 4, "rowcost_base width")?;
    let col_cost_width = checked_len(col_arcs, 4, "colcost_base width")?;
    require_shape(input.rowix, row_count, row_arcs, "rowix")?;
    require_shape(input.colix, col_count, col_arcs, "colix")?;
    require_shape(
        input.rowcost_base,
        row_count,
        row_cost_width,
        "rowcost_base",
    )?;
    require_shape(
        input.colcost_base,
        col_count,
        col_cost_width,
        "colcost_base",
    )?;
    if input.wrapped_space_uw.len() != input.dph_smooth.len() {
        return Err(Stage6Error::new(
            "wrapped_space_uw and dph_smooth must have matching lengths",
        ));
    }
    if !input.nshortcycle.is_finite() || input.nshortcycle <= 0.0 {
        return Err(Stage6Error::new("nshortcycle must be positive and finite"));
    }

    let mut rowcost = input.rowcost_base.to_vec();
    let mut colcost = input.colcost_base.to_vec();
    let n_edge = input.wrapped_space_uw.len();
    let scale = input.nshortcycle / std::f64::consts::TAU;
    fill_offsets(
        &mut rowcost,
        input.rowix,
        row_count,
        row_arcs,
        input.wrapped_space_uw,
        input.dph_smooth,
        n_edge,
        scale,
        -1,
    )?;
    fill_offsets(
        &mut colcost,
        input.colix,
        col_count,
        col_arcs,
        input.wrapped_space_uw,
        input.dph_smooth,
        n_edge,
        scale,
        1,
    )?;
    Ok((rowcost, colcost))
}

#[allow(clippy::too_many_arguments)]
fn fill_offsets(
    costs: &mut [i16],
    edge_indices: &[f64],
    rows: usize,
    arcs: usize,
    wrapped: &[f32],
    smooth: &[f32],
    n_edge: usize,
    scale: f64,
    output_sign: i16,
) -> Result<(), Stage6Error> {
    for row in 0..rows {
        for col in 0..arcs {
            let edge_value = edge_indices[row * arcs + col];
            if !edge_value.is_finite() || edge_value == 0.0 {
                continue;
            }
            let edge_index = edge_value.abs() as usize;
            if edge_index == 0 || edge_index > n_edge {
                return Err(Stage6Error::new(
                    "edge-index matrix references a missing phase edge",
                ));
            }
            let offset = (f64::from(wrapped[edge_index - 1]) - f64::from(smooth[edge_index - 1]))
                * edge_value.signum()
                * scale;
            costs[row * arcs * 4 + col * 4] =
                (offset.round_ties_even() as i16).saturating_mul(output_sign);
        }
    }
    Ok(())
}

pub fn reconstruct_ps_phase(
    ph_uw_grid: &[f32],
    n_grid_ps: usize,
    n_ifg: usize,
    ps_grid_indices: &[Option<usize>],
    ph_in: &[Complex32],
    phase_restore: Option<&[f32]>,
) -> Result<Vec<f32>, Stage6Error> {
    require_shape(ph_uw_grid, n_grid_ps, n_ifg, "ph_uw_grid")?;
    let n_ps = ps_grid_indices.len();
    require_shape(ph_in, n_ps, n_ifg, "ph_in")?;
    if phase_restore.is_some_and(|values| values.len() != n_ps * n_ifg) {
        return Err(Stage6Error::new("phase_restore must match ph_in"));
    }
    if ps_grid_indices
        .iter()
        .flatten()
        .any(|&index| index >= n_grid_ps)
    {
        return Err(Stage6Error::new(
            "ps_grid_indices references a missing grid row",
        ));
    }

    let mut output = vec![f32::NAN; n_ps * n_ifg];
    for row in 0..n_ps {
        let Some(grid_row) = ps_grid_indices[row] else {
            continue;
        };
        for column in 0..n_ifg {
            let index = row * n_ifg + column;
            let phase = ph_uw_grid[grid_row * n_ifg + column];
            let wrapped = ph_in[index];
            let real = wrapped.re * phase.cos() + wrapped.im * phase.sin();
            let imag = wrapped.im * phase.cos() - wrapped.re * phase.sin();
            let restore = phase_restore.map_or(0.0, |values| values[index]);
            output[index] = phase + imag.atan2(real) + restore;
        }
    }
    Ok(output)
}
