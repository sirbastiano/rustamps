use crate::stage6_native::{apply_edge_correction, horizontal_index, vertical_index, EdgeDatum};
use crate::stage6_residual::{residual_arc_cost, ResidualArc};

pub(crate) struct CompactResidualView<'a> {
    horizontal: &'a [Option<EdgeDatum>],
    vertical: &'a [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    horizontal_pairs: usize,
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
        Self {
            horizontal,
            vertical,
            nrow,
            ncol,
            horizontal_pairs: nrow * ncol.saturating_sub(1),
            correction_step: nflow.abs().max(1),
        }
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nrow.saturating_sub(1) * self.ncol.saturating_sub(1) + 1
    }

    pub(crate) fn arc_count(&self) -> usize {
        (self.horizontal_pairs + self.nrow.saturating_sub(1) * self.ncol) * 2
    }

    pub(crate) fn arc(&self, arc_index: usize) -> Option<ResidualArc> {
        if arc_index >= self.arc_count() {
            return None;
        }
        let pair = arc_index / 2;
        let reverse = arc_index % 2 == 1;
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

    pub(crate) fn for_each_arc(&self, mut visit: impl FnMut(usize, ResidualArc)) {
        for index in 0..self.arc_count() {
            if let Some(arc) = self.arc(index) {
                visit(index, arc);
            }
        }
    }

    fn horizontal_arc(&self, pair: usize) -> Option<(usize, usize, bool, usize, EdgeDatum)> {
        let pcn = self.ncol.saturating_sub(1);
        if pcn == 0 {
            return None;
        }
        let row = pair / pcn;
        let col = pair % pcn;
        if row >= self.nrow {
            return None;
        }
        let edge_index = horizontal_index(row, col, self.ncol);
        let edge = self.horizontal.get(edge_index).and_then(|edge| *edge)?;
        let ground = self.node_count() - 1;
        let (from, to) = if row == 0 {
            (ground, col)
        } else if row + 1 == self.nrow {
            ((row - 1) * pcn + col, ground)
        } else {
            ((row - 1) * pcn + col, row * pcn + col)
        };
        Some((from, to, true, edge_index, edge))
    }

    fn vertical_arc(&self, pair: usize) -> Option<(usize, usize, bool, usize, EdgeDatum)> {
        let prn = self.nrow.saturating_sub(1);
        let pcn = self.ncol.saturating_sub(1);
        if prn == 0 || self.ncol == 0 {
            return None;
        }
        let row = pair / self.ncol;
        let col = pair % self.ncol;
        if row >= prn {
            return None;
        }
        let edge_index = vertical_index(row, col, self.ncol);
        let edge = self.vertical.get(edge_index).and_then(|edge| *edge)?;
        let ground = self.node_count() - 1;
        let (from, to) = if col == 0 {
            (row * pcn, ground)
        } else if col + 1 == self.ncol {
            (ground, row * pcn + col - 1)
        } else {
            (row * pcn + col, row * pcn + col - 1)
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
