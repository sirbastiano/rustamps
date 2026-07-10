use crate::stage6_native::{
    apply_edge_correction, defo_edge_cost, horizontal_index, vertical_index, EdgeDatum,
};

pub(crate) fn mst_scalar_weight(edge: EdgeDatum) -> i32 {
    let base = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        0,
    );
    let pos = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        1,
    ) - base;
    let neg = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        -1,
    ) - base;
    pos.min(neg).clamp(1, 1000) as i32
}

pub(crate) fn apply_mst_flows(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    rowflow: &[i32],
    colflow: &[i32],
) {
    for row in 0..nrow.saturating_sub(1) {
        for col in 0..ncol {
            let flow = rowflow[row * ncol + col];
            apply_edge_correction(&mut vertical[vertical_index(row, col, ncol)], -flow);
        }
    }
    for row in 0..nrow {
        for col in 0..ncol.saturating_sub(1) {
            let flow = colflow[row * (ncol - 1) + col];
            apply_edge_correction(&mut horizontal[horizontal_index(row, col, ncol)], flow);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(flow_sign: i32) -> EdgeDatum {
        EdgeDatum {
            cost: 1000,
            desired_delta: 0.0,
            offset: 0,
            dzmax: 32000,
            laycost: -32000,
            nshortcycle: 200,
            flow_sign,
            flow: 0,
        }
    }

    #[test]
    fn mst_scalar_weight_matches_snaphu_unit_defo_increment() {
        assert_eq!(mst_scalar_weight(edge(1)), 40);
    }

    #[test]
    fn mst_scalar_weight_clips_to_snaphu_range() {
        let mut low = edge(1);
        low.cost = 32000;
        assert_eq!(mst_scalar_weight(low), 1);

        let mut high = edge(1);
        high.cost = 1;
        assert_eq!(mst_scalar_weight(high), 1000);
    }

    #[test]
    fn apply_mst_flows_uses_snaphu_row_and_column_signs() {
        let nrow = 2;
        let ncol = 2;
        let mut horizontal = vec![Some(edge(1)); nrow * (ncol - 1)];
        let mut vertical = vec![Some(edge(-1)); (nrow - 1) * ncol];

        apply_mst_flows(
            &mut horizontal,
            &mut vertical,
            nrow,
            ncol,
            &[1, -2],
            &[3, -4],
        );

        let upper_left = vertical[vertical_index(0, 0, ncol)].unwrap();
        assert_eq!(upper_left.flow, 1);
        assert_eq!(upper_left.desired_delta.round() as i32, -1);

        let upper_right = vertical[vertical_index(0, 1, ncol)].unwrap();
        assert_eq!(upper_right.flow, -2);
        assert_eq!(upper_right.desired_delta.round() as i32, 2);

        let left = horizontal[horizontal_index(0, 0, ncol)].unwrap();
        assert_eq!(left.flow, 3);
        assert_eq!(left.desired_delta.round() as i32, 3);

        let lower_left = horizontal[horizontal_index(1, 0, ncol)].unwrap();
        assert_eq!(lower_left.flow, -4);
        assert_eq!(lower_left.desired_delta.round() as i32, -4);
    }
}
