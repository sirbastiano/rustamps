use crate::stage6::unwrap::native::{apply_edge_correction, EdgeDatum};
use crate::stage6::unwrap::residual::{residual_arc_cost, ResidualArc};
use std::collections::HashSet;

pub(crate) struct CompactResidualView<'a> {
    horizontal: &'a [Option<EdgeDatum>],
    vertical: &'a [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    prn: usize,
    pcn: usize,
    horizontal_pairs: usize,
    arc_count: usize,
    node_count: usize,
    correction_step: i32,
}

impl<'a> CompactResidualView<'a> {
    pub(crate) fn new(
        horizontal: &'a [Option<EdgeDatum>],
        vertical: &'a [Option<EdgeDatum>],
        nrow: usize,
        ncol: usize,
    ) -> Self {
        Self::with_nflow(horizontal, vertical, nrow, ncol, 1)
    }

    pub(crate) fn with_nflow(
        horizontal: &'a [Option<EdgeDatum>],
        vertical: &'a [Option<EdgeDatum>],
        nrow: usize,
        ncol: usize,
        nflow: i32,
    ) -> Self {
        let prn = nrow.saturating_sub(1);
        let pcn = ncol.saturating_sub(1);
        let horizontal_pairs = nrow * pcn;
        Self {
            horizontal,
            vertical,
            nrow,
            ncol,
            prn,
            pcn,
            horizontal_pairs,
            arc_count: (horizontal_pairs + prn * ncol) * 2,
            node_count: prn * pcn + 1,
            correction_step: nflow.abs().max(1),
        }
    }

    #[inline(always)]
    pub(crate) fn node_count(&self) -> usize {
        self.node_count
    }

    #[inline(always)]
    pub(crate) fn arc_count(&self) -> usize {
        self.arc_count
    }

    #[inline(always)]
    pub(crate) fn arc(&self, arc_index: usize) -> Option<ResidualArc> {
        if arc_index >= self.arc_count {
            return None;
        }
        let pair = arc_index >> 1;
        let reverse = arc_index & 1 == 1;
        let (from, to, is_horizontal, edge_index, edge) = if pair < self.horizontal_pairs {
            self.horizontal_arc(pair)?
        } else {
            self.vertical_arc(pair - self.horizontal_pairs)?
        };
        let correction_delta = if reverse {
            -self.correction_step
        } else {
            self.correction_step
        };
        Some(ResidualArc {
            from: if reverse { to } else { from },
            to: if reverse { from } else { to },
            cost: residual_arc_cost(edge, correction_delta),
            is_horizontal,
            edge_index,
            correction_delta,
        })
    }

    #[inline(always)]
    pub(crate) fn endpoints(&self, arc_index: usize) -> Option<(usize, usize)> {
        if arc_index >= self.arc_count {
            return None;
        }
        let pair = arc_index >> 1;
        let reverse = arc_index & 1 == 1;
        let (from, to, _, _, _) = if pair < self.horizontal_pairs {
            self.horizontal_arc(pair)?
        } else {
            self.vertical_arc(pair - self.horizontal_pairs)?
        };
        Some(if reverse { (to, from) } else { (from, to) })
    }

    #[inline(always)]
    pub(crate) fn cost(&self, arc_index: usize) -> Option<i32> {
        if arc_index >= self.arc_count {
            return None;
        }
        let pair = arc_index >> 1;
        let edge = if pair < self.horizontal_pairs {
            self.horizontal.get(pair).and_then(|edge| *edge)?
        } else {
            self.vertical
                .get(pair - self.horizontal_pairs)
                .and_then(|edge| *edge)?
        };
        let correction = if arc_index & 1 == 1 {
            -self.correction_step
        } else {
            self.correction_step
        };
        Some(residual_arc_cost(edge, correction))
    }

    pub(crate) fn for_each_arc(&self, mut visit: impl FnMut(usize, ResidualArc)) {
        for index in 0..self.arc_count() {
            if let Some(arc) = self.arc(index) {
                visit(index, arc);
            }
        }
    }

    #[inline(always)]
    fn horizontal_arc(&self, pair: usize) -> Option<(usize, usize, bool, usize, EdgeDatum)> {
        if self.pcn == 0 {
            return None;
        }
        let row = pair / self.pcn;
        let col = pair - row * self.pcn;
        if row >= self.nrow {
            return None;
        }
        let edge_index = pair;
        let edge = self.horizontal.get(pair).and_then(|edge| *edge)?;
        let ground = self.node_count - 1;
        let (from, to) = if row == 0 {
            (ground, col)
        } else if row + 1 == self.nrow {
            ((row - 1) * self.pcn + col, ground)
        } else {
            ((row - 1) * self.pcn + col, row * self.pcn + col)
        };
        Some((from, to, true, edge_index, edge))
    }

    #[inline(always)]
    fn vertical_arc(&self, pair: usize) -> Option<(usize, usize, bool, usize, EdgeDatum)> {
        if self.prn == 0 || self.ncol == 0 {
            return None;
        }
        let row = pair / self.ncol;
        let col = pair - row * self.ncol;
        if row >= self.prn {
            return None;
        }
        let edge_index = pair;
        let edge = self.vertical.get(pair).and_then(|edge| *edge)?;
        let ground = self.node_count - 1;
        let (from, to) = if col == 0 {
            (row * self.pcn, ground)
        } else if col + 1 == self.ncol {
            (ground, row * self.pcn + col - 1)
        } else {
            (row * self.pcn + col, row * self.pcn + col - 1)
        };
        Some((from, to, false, edge_index, edge))
    }
}

pub(crate) fn apply_compact_residual_cycle(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    cycle: &[usize],
) {
    apply_compact_residual_cycle_with_nflow(horizontal, vertical, nrow, ncol, cycle, 1);
}

pub(crate) fn apply_compact_residual_cycle_with_nflow(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    cycle: &[usize],
    nflow: i32,
) {
    for &index in cycle {
        let arc = {
            let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
            view.arc(index)
        };
        let Some(arc) = arc else {
            continue;
        };
        let edge = if arc.is_horizontal {
            &mut horizontal[arc.edge_index]
        } else {
            &mut vertical[arc.edge_index]
        };
        apply_edge_correction(edge, arc.correction_delta);
    }
}

pub(crate) fn saturate_compact_residual_cycle_with_nflow(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    cycle: &[usize],
    nflow: i32,
) -> usize {
    let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
    let mut seen = HashSet::with_capacity(cycle.len());
    if cycle
        .iter()
        .any(|&index| view.arc(index).is_none() || !seen.insert(index >> 1))
    {
        return 0;
    }
    let mut increments = 0;
    loop {
        let cost = {
            let view = CompactResidualView::with_nflow(horizontal, vertical, nrow, ncol, nflow);
            cycle.iter().try_fold(0_i64, |sum, &index| {
                Some(sum + i64::from(view.cost(index)?))
            })
        };
        if cost.is_none_or(|value| value >= 0) {
            return increments;
        }
        apply_compact_residual_cycle_with_nflow(horizontal, vertical, nrow, ncol, cycle, nflow);
        increments += 1;
    }
}
