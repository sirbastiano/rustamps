use super::*;

fn edge(cost: i32, laycost: i32, flow_sign: i32) -> EdgeDatum {
    EdgeDatum {
        cost,
        desired_delta: 0.0,
        offset: 0,
        dzmax: 32000,
        laycost,
        nshortcycle: 200,
        flow_sign,
        flow: 0,
    }
}

#[test]
fn boundary_routes_use_defo_increment_costs() {
    let nrow = 2;
    let ncol = 3;
    let mut horizontal = vec![None; nrow * (ncol - 1)];
    let mut vertical = vec![None; (nrow - 1) * ncol];
    horizontal[horizontal_index(0, 0, ncol)] = Some(edge(1000, -32000, 1));
    vertical[vertical_index(0, 0, ncol)] = Some(edge(1, 1, -1));

    let routes = compute_boundary_routes_for_amount(&horizontal, &vertical, nrow, ncol, 1);

    assert_eq!(routes[plaquette_index(0, 0, ncol)], ROUTE_LEFT);
}
