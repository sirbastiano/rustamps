use super::native::EdgeDatum;
use super::residual::{find_negative_unit_cycle, saturate_residual_cycle, ResidualArc};
use super::residual_view::saturate_compact_residual_cycle_with_nflow;

#[test]
fn negative_cycle_is_saturated_and_counted_once() {
    let mut horizontal = [Some(EdgeDatum {
        cost: 1000,
        desired_delta: 0.0,
        offset: 1000,
        dzmax: 32000,
        laycost: -32000,
        nshortcycle: 200,
        flow_sign: 1,
        flow: 0,
    })];
    let mut vertical = [];
    let arcs = [ResidualArc {
        from: 0,
        to: 0,
        cost: -360,
        is_horizontal: true,
        edge_index: 0,
        correction_delta: -1,
    }];

    let increments = saturate_residual_cycle(&mut horizontal, &mut vertical, &arcs, &[0]);

    assert_eq!(increments, 5);
    assert_eq!(horizontal[0].unwrap().flow, -5);
}

#[test]
fn opposite_directions_of_one_edge_are_not_a_physical_cycle() {
    let edge = EdgeDatum {
        cost: 1000,
        desired_delta: 0.0,
        offset: 1000,
        dzmax: 10000,
        laycost: 1000,
        nshortcycle: 200,
        flow_sign: 1,
        flow: 0,
    };
    let mut horizontal = [Some(edge)];
    let mut vertical = [];
    let mut reverse = ResidualArc {
        from: 0,
        to: 1,
        cost: -1,
        is_horizontal: true,
        edge_index: 0,
        correction_delta: 1,
    };
    let forward = reverse;
    reverse.correction_delta = -1;
    let arcs = [forward, reverse];

    assert_eq!(
        saturate_residual_cycle(&mut horizontal, &mut vertical, &arcs, &[0, 1]),
        0
    );
    assert_eq!(horizontal[0].unwrap().flow, 0);
}

#[test]
fn compact_saturation_rejects_opposite_directions() {
    let mut horizontal = [Some(EdgeDatum {
        cost: 1000,
        desired_delta: 0.0,
        offset: 1000,
        dzmax: 10000,
        laycost: 1000,
        nshortcycle: 200,
        flow_sign: 1,
        flow: 0,
    })];
    let mut vertical = [];

    let increments = saturate_compact_residual_cycle_with_nflow(
        &mut horizontal,
        &mut vertical,
        1,
        2,
        &[0, 1],
        1,
    );

    assert_eq!(increments, 0);
    assert_eq!(horizontal[0].unwrap().flow, 0);
}

#[test]
fn degenerate_pair_does_not_hide_a_genuine_negative_cycle() {
    let arc = |from, to, cost, edge_index| ResidualArc {
        from,
        to,
        cost,
        is_horizontal: true,
        edge_index,
        correction_delta: 1,
    };
    let arcs = [
        arc(0, 1, -10, 0),
        arc(1, 0, 0, 0),
        arc(1, 2, 0, 1),
        arc(2, 0, 0, 2),
    ];

    let cycle = find_negative_unit_cycle(&arcs, 3).unwrap();

    assert_eq!(cycle.len(), 3);
    assert!(cycle.iter().all(|&index| index != 1));
}

#[test]
fn negative_opposing_pair_is_ignored_by_cycle_search() {
    let arc = |from, to, cost| ResidualArc {
        from,
        to,
        cost,
        is_horizontal: true,
        edge_index: 0,
        correction_delta: 1,
    };
    assert!(find_negative_unit_cycle(&[arc(0, 1, -10), arc(1, 0, 0)], 2).is_none());
}
