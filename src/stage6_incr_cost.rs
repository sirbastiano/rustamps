use crate::stage6_native::{defo_edge_cost, EdgeDatum};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct IncrementalCost {
    pub(crate) pos: i16,
    pub(crate) neg: i16,
}

fn scale_increment(value: i64, nflow: i32) -> i16 {
    let divisor = i64::from(nflow) * i64::from(nflow);
    let scaled = if value >= 0 {
        (value + divisor - 1) / divisor
    } else {
        value.div_euclid(divisor)
    };
    scaled.clamp(-32000, 32000) as i16
}

pub(crate) fn defo_incremental_costs(edge: EdgeDatum, nflow: i32) -> IncrementalCost {
    let nflow = nflow.abs().max(1);
    let base = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        edge.flow,
    );
    let pos = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        edge.flow + nflow,
    ) - base;
    let neg = defo_edge_cost(
        edge.cost,
        edge.offset,
        edge.dzmax,
        edge.laycost,
        edge.nshortcycle,
        edge.flow - nflow,
    ) - base;
    IncrementalCost {
        pos: scale_increment(pos, nflow),
        neg: scale_increment(neg, nflow),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(cost: i32, flow: i32) -> EdgeDatum {
        EdgeDatum {
            cost,
            desired_delta: 0.0,
            offset: 0,
            dzmax: 32000,
            laycost: -32000,
            nshortcycle: 200,
            flow_sign: 1,
            flow,
        }
    }

    fn shelf_edge(flow: i32) -> EdgeDatum {
        EdgeDatum {
            cost: 1,
            desired_delta: 0.0,
            offset: -1000,
            dzmax: 1000,
            laycost: 1,
            nshortcycle: 200,
            flow_sign: 1,
            flow,
        }
    }

    #[test]
    fn defo_incremental_costs_match_snaphu_unit_increments() {
        assert_eq!(
            defo_incremental_costs(edge(1000, 0), 1),
            IncrementalCost { pos: 40, neg: 40 }
        );
    }

    #[test]
    fn defo_incremental_costs_can_be_negative_around_existing_flow() {
        assert_eq!(
            defo_incremental_costs(edge(1000, 1), 1),
            IncrementalCost { pos: 120, neg: -40 }
        );
    }

    #[test]
    fn defo_incremental_costs_scale_by_flow_increment_squared() {
        assert_eq!(
            defo_incremental_costs(edge(1000, 0), 2),
            IncrementalCost { pos: 40, neg: 40 }
        );
    }

    #[test]
    fn defo_incremental_costs_clip_to_short_range() {
        assert_eq!(
            defo_incremental_costs(edge(1, 0), 1),
            IncrementalCost {
                pos: 32000,
                neg: 32000
            }
        );
    }

    #[test]
    fn negative_scaled_increment_uses_floor_division() {
        assert_eq!(defo_incremental_costs(shelf_edge(7), 2).neg, -1);
    }
}
