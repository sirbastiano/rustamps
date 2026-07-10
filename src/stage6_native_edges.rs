#[derive(Clone, Copy)]
pub(crate) struct EdgeDatum {
    pub(crate) cost: i32,
    pub(crate) desired_delta: f32,
    pub(crate) offset: i32,
    pub(crate) dzmax: i32,
    pub(crate) laycost: i32,
    pub(crate) nshortcycle: i32,
    pub(crate) flow_sign: i32,
    pub(crate) flow: i32,
}

pub(crate) fn rounded_delta(edge: EdgeDatum) -> i32 {
    edge.desired_delta.round() as i32
}

pub(crate) fn horizontal_index(row: usize, col: usize, ncol: usize) -> usize {
    row * (ncol - 1) + col
}

pub(crate) fn vertical_index(row: usize, col: usize, ncol: usize) -> usize {
    row * ncol + col
}

pub(crate) fn edge_weight(cost: i32) -> f64 {
    1.0 / f64::from(cost.abs().max(1))
}

pub(crate) fn apply_edge_correction(edge: &mut Option<EdgeDatum>, delta: i32) {
    if let Some(datum) = edge {
        datum.desired_delta += delta as f32;
        datum.flow += datum.flow_sign * delta;
    }
}

pub(crate) fn defo_edge_cost(
    cost: i32,
    offset: i32,
    dzmax: i32,
    laycost: i32,
    nshortcycle: i32,
    flow: i32,
) -> i64 {
    if cost == 32000 {
        return 0;
    }
    let sigsq = cost.max(1) as i64;
    let dz = (i64::from(flow) * i64::from(nshortcycle) + i64::from(offset)).abs();
    let laycost_i64 = i64::from(laycost);
    let dzmax = if laycost == -32000 {
        32000
    } else {
        dzmax.max(0)
    };
    let dzmax = i64::from(dzmax);
    if dz > dzmax {
        let falloff_dz = dz - dzmax;
        (falloff_dz * falloff_dz) / (2 * sigsq) + laycost_i64
    } else {
        let mut cost = (dz * dz) / sigsq;
        if laycost != -32000 && cost > laycost_i64 {
            cost = laycost_i64;
        }
        cost
    }
}

pub(crate) fn edge_increment_cost(edge: EdgeDatum, desired_delta: i32) -> f64 {
    let flow_delta = edge.flow_sign * desired_delta;
    let before = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        edge.flow,
    );
    let after = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        edge.flow + flow_delta,
    );
    (after - before) as f64
}

pub(crate) fn edge_label_energy(edge: EdgeDatum, from_label: i32, to_label: i32) -> i64 {
    let label_delta = to_label - from_label;
    let flow = edge.flow + edge.flow_sign * (label_delta - rounded_delta(edge));
    defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        flow,
    )
}
