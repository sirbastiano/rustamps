use num_complex::Complex32;

use super::{checked_len, Stage6Error};

#[path = "stage6_component_shift.rs"]
mod component_shift;
#[path = "stage6_cut.rs"]
mod cut;
#[path = "stage6_cut_graph.rs"]
mod cut_graph;
#[path = "stage6_incr_cost.rs"]
mod incr_cost;
#[path = "stage6_label_flow.rs"]
mod label_flow;
#[path = "stage6_local_cycles.rs"]
mod local_cycles;
#[path = "stage6_mst.rs"]
mod mst;
#[path = "stage6_mst_flow.rs"]
mod mst_flow;
mod native;
#[path = "stage6_patch.rs"]
mod patch;
#[path = "stage6_residual.rs"]
mod residual;
#[cfg(test)]
#[path = "stage6_residual_tests.rs"]
mod residual_tests;
#[path = "stage6_residual_view.rs"]
mod residual_view;
#[path = "stage6_residue.rs"]
mod residue;
#[path = "stage6_tree_compact.rs"]
mod tree_compact;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridUnwrapConfig {
    pub nshortcycle: f32,
    pub parallel: bool,
    pub max_flow_passes: Option<usize>,
}

impl Default for GridUnwrapConfig {
    fn default() -> Self {
        Self {
            nshortcycle: 200.0,
            parallel: true,
            max_flow_passes: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GridUnwrapInputs<'a> {
    pub ifgw: &'a [Complex32],
    pub rowcost: &'a [i16],
    pub colcost: &'a [i16],
    pub nrow: usize,
    pub ncol: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GridUnwrapOutput {
    pub ifguw: Vec<f32>,
    pub msd: f64,
    pub flow_cycles: usize,
    pub flow_objective: i64,
    pub post_label_flow_cycles: usize,
    pub post_label_flow_objective: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GridUnwrapTimings {
    pub decode_sec: f64,
    pub initial_flow_sec: f64,
    pub initial_label_sec: f64,
    pub post_flow_sec: f64,
    pub final_label_sec: f64,
    pub msd_sec: f64,
}

pub fn unwrap_grid(
    input: &GridUnwrapInputs<'_>,
    config: GridUnwrapConfig,
) -> Result<GridUnwrapOutput, Stage6Error> {
    unwrap_grid_profiled(input, config).map(|(output, _timings)| output)
}

pub fn unwrap_grid_profiled(
    input: &GridUnwrapInputs<'_>,
    config: GridUnwrapConfig,
) -> Result<(GridUnwrapOutput, GridUnwrapTimings), Stage6Error> {
    if input.nrow == 0 || input.ncol == 0 {
        return Err(Stage6Error::new("unwrap grid must be non-empty"));
    }
    let cell_count = checked_len(input.nrow, input.ncol, "ifgw")?;
    if input.ifgw.len() != cell_count {
        return Err(Stage6Error::new("ifgw does not match nrow by ncol"));
    }
    let row_arcs = input.nrow.saturating_sub(1);
    let expected_row = checked_len(row_arcs, input.ncol, "rowcost")?
        .checked_mul(4)
        .ok_or_else(|| Stage6Error::new("rowcost dimensions overflow"))?;
    let expected_col = checked_len(input.nrow, input.ncol.saturating_sub(1), "colcost")?
        .checked_mul(4)
        .ok_or_else(|| Stage6Error::new("colcost dimensions overflow"))?;
    if input.rowcost.len() != expected_row || input.colcost.len() != expected_col {
        return Err(Stage6Error::new(
            "rowcost or colcost does not match the unwrap grid",
        ));
    }
    if !config.nshortcycle.is_finite() || config.nshortcycle <= 0.0 {
        return Err(Stage6Error::new("nshortcycle must be positive and finite"));
    }
    let result = native::unwrap_grid(
        input.ifgw,
        input.nrow,
        input.ncol,
        input.rowcost,
        input.colcost,
        config.nshortcycle,
        config.parallel,
        config.max_flow_passes,
    );
    let msd_started = std::time::Instant::now();
    let msd = native::neighbor_msd(&result.ifguw, input.nrow, input.ncol);
    let timings = GridUnwrapTimings {
        decode_sec: result.timings.decode_sec,
        initial_flow_sec: result.timings.initial_flow_sec,
        initial_label_sec: result.timings.initial_label_sec,
        post_flow_sec: result.timings.post_flow_sec,
        final_label_sec: result.timings.final_label_sec,
        msd_sec: msd_started.elapsed().as_secs_f64(),
    };
    Ok((
        GridUnwrapOutput {
            msd,
            ifguw: result.ifguw,
            flow_cycles: result.flow_cycles,
            flow_objective: result.flow_objective,
            post_label_flow_cycles: result.post_cycles,
            post_label_flow_objective: result.post_objective,
        },
        timings,
    ))
}
