use crate::stage6_incr_cost::defo_incremental_costs;
use crate::stage6_native::EdgeDatum;
use crate::stage6_residual::build_unit_residual_arcs;
use crate::stage6_residual_view::CompactResidualView;

fn edge(cost: i32, flow_sign: i32, flow: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
        desired_delta: 0.0,
        offset: 0,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign,
        flow,
    }
}

#[test]
fn compact_residual_view_matches_materialized_arcs_for_full_grid() {
    let nrow = 3;
    let ncol = 3;
    let horizontal = vec![Some(edge(1000, 1, 1)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(2000, -1, -1)); (nrow - 1) * ncol];
    let materialized = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let compact = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);

    assert_eq!(compact.node_count(), (nrow - 1) * (ncol - 1) + 1);
    assert_eq!(compact.arc_count(), materialized.len());
    for (index, expected) in materialized.iter().copied().enumerate() {
        assert_eq!(compact.arc(index), Some(expected));
    }
}

#[test]
fn compact_residual_view_skips_missing_edges() {
    let nrow = 2;
    let ncol = 2;
    let horizontal = vec![None, Some(edge(1000, 1, 0))];
    let vertical = vec![Some(edge(1000, -1, 0)), None];
    let compact = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);

    assert_eq!(compact.arc_count(), 8);
    assert!(compact.arc(0).is_none());
    assert!(compact.arc(1).is_none());
    assert!(compact.arc(2).is_some());
    assert!(compact.arc(3).is_some());
    assert!(compact.arc(4).is_some());
    assert!(compact.arc(5).is_some());
    assert!(compact.arc(6).is_none());
    assert!(compact.arc(7).is_none());
}

#[test]
fn compact_residual_view_scans_present_arcs_without_materializing() {
    let nrow = 3;
    let ncol = 3;
    let mut horizontal = vec![Some(edge(1000, 1, 1)); nrow * (ncol - 1)];
    let vertical = vec![Some(edge(2000, -1, -1)); (nrow - 1) * ncol];
    horizontal[2] = None;
    let materialized = build_unit_residual_arcs(&horizontal, &vertical, nrow, ncol);
    let compact = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    let mut scanned = 0_usize;
    let mut cost_sum = 0_i32;

    compact.for_each_arc(|_index, arc| {
        scanned += 1;
        cost_sum += arc.cost;
    });

    assert_eq!(scanned, materialized.len());
    assert_eq!(
        cost_sum,
        materialized.iter().map(|arc| arc.cost).sum::<i32>()
    );
}

#[test]
fn compact_residual_view_can_expose_larger_flow_steps() {
    let nrow = 2;
    let ncol = 2;
    let horizontal = vec![Some(edge(1000, 1, 0)), None];
    let vertical = vec![None, None];
    let compact = CompactResidualView::with_nflow(&horizontal, &vertical, nrow, ncol, 4);
    let arc = compact.arc(0).unwrap();

    assert_eq!(arc.correction_delta, 4);
    assert_eq!(
        arc.cost,
        i32::from(defo_incremental_costs(edge(1000, 1, 0), 4).pos)
    );
}
