use crate::stage6_native::EdgeDatum;
use crate::stage6_residual::{
    apply_residual_cycle, build_unit_residual_arcs, residual_cycle_cost, ResidualArc,
};
use crate::stage6_tree_path::TreePathCosts;
use std::collections::VecDeque;

#[cfg(test)]
pub(crate) use crate::stage6_tree_compact::{
    compact_tree_edge_mask, find_negative_tree_cycle_compact, optimize_tree_cycles_compact,
    optimize_tree_cycles_compact_with_nflow, pivot_compact_tree_on_cycle,
    relax_compact_tree_by_reduced_cost, relax_compact_tree_by_reduced_cost_candidates,
    spanning_tree_arc_indices_compact, CompactTreeBasis,
};

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            rank: vec![0; count],
        }
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            self.parent[node] = self.find(self.parent[node]);
        }
        self.parent[node]
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let mut root_left = self.find(left);
        let mut root_right = self.find(right);
        if root_left == root_right {
            return false;
        }
        if self.rank[root_left] < self.rank[root_right] {
            std::mem::swap(&mut root_left, &mut root_right);
        }
        self.parent[root_right] = root_left;
        if self.rank[root_left] == self.rank[root_right] {
            self.rank[root_left] += 1;
        }
        true
    }
}

fn same_primal_edge(left: ResidualArc, right: ResidualArc) -> bool {
    left.is_horizontal == right.is_horizontal && left.edge_index == right.edge_index
}

fn reverse_arc_index(arcs: &[ResidualArc], index: usize) -> Option<usize> {
    let arc = arcs.get(index).copied()?;
    arcs.iter().position(|candidate| {
        candidate.from == arc.to
            && candidate.to == arc.from
            && candidate.correction_delta == -arc.correction_delta
            && same_primal_edge(*candidate, arc)
    })
}

pub(crate) fn spanning_tree_arc_indices(arcs: &[ResidualArc], node_count: usize) -> Vec<usize> {
    let mut dsu = DisjointSet::new(node_count);
    let mut tree = Vec::with_capacity(node_count.saturating_sub(1));
    for (index, arc) in arcs.iter().enumerate() {
        if arc.from >= node_count || arc.to >= node_count || arc.from == arc.to {
            continue;
        }
        if dsu.union(arc.from, arc.to) {
            tree.push(index);
            if tree.len() + 1 == node_count {
                break;
            }
        }
    }
    tree
}

fn tree_adjacency(
    arcs: &[ResidualArc],
    node_count: usize,
    tree_arc_indices: &[usize],
) -> Vec<Vec<(usize, usize)>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for &index in tree_arc_indices {
        let Some(arc) = arcs.get(index).copied() else {
            continue;
        };
        if arc.from >= node_count || arc.to >= node_count {
            continue;
        }
        adjacency[arc.from].push((arc.to, index));
        if let Some(reverse) = reverse_arc_index(arcs, index) {
            adjacency[arc.to].push((arc.from, reverse));
        }
    }
    adjacency
}

pub(crate) fn tree_cycle_for_arc(
    arcs: &[ResidualArc],
    node_count: usize,
    tree_arc_indices: &[usize],
    non_tree_arc_index: usize,
) -> Option<Vec<usize>> {
    let non_tree = arcs.get(non_tree_arc_index).copied()?;
    if non_tree.from >= node_count || non_tree.to >= node_count {
        return None;
    }
    let adjacency = tree_adjacency(arcs, node_count, tree_arc_indices);
    let start = non_tree.to;
    let goal = non_tree.from;
    let mut queue = VecDeque::from([start]);
    let mut seen = vec![false; node_count];
    let mut prev_node = vec![usize::MAX; node_count];
    let mut prev_arc = vec![usize::MAX; node_count];
    seen[start] = true;

    while let Some(node) = queue.pop_front() {
        if node == goal {
            break;
        }
        for &(next, arc_index) in &adjacency[node] {
            if !seen[next] {
                seen[next] = true;
                prev_node[next] = node;
                prev_arc[next] = arc_index;
                queue.push_back(next);
            }
        }
    }
    if !seen[goal] {
        return None;
    }

    let mut path = Vec::new();
    let mut node = goal;
    while node != start {
        let arc_index = prev_arc[node];
        if arc_index == usize::MAX {
            return None;
        }
        path.push(arc_index);
        node = prev_node[node];
    }
    path.reverse();

    let mut cycle = Vec::with_capacity(path.len() + 1);
    cycle.push(non_tree_arc_index);
    cycle.extend(path);
    Some(cycle)
}

pub(crate) fn tree_edge_mask(arcs: &[ResidualArc], tree_arc_indices: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; arcs.len()];
    for &index in tree_arc_indices {
        if index >= arcs.len() {
            continue;
        }
        mask[index] = true;
        let reverse = index ^ 1;
        if reverse < arcs.len() && same_primal_edge(arcs[index], arcs[reverse]) {
            mask[reverse] = true;
        }
    }
    mask
}

pub(crate) fn find_negative_tree_cycle(
    arcs: &[ResidualArc],
    node_count: usize,
    tree_arc_indices: &[usize],
) -> Option<Vec<usize>> {
    let tree_mask = tree_edge_mask(arcs, tree_arc_indices);
    let mut best = None;
    let mut best_cost = 0_i32;
    for index in 0..arcs.len() {
        if tree_mask[index] {
            continue;
        }
        let Some(cycle) = tree_cycle_for_arc(arcs, node_count, tree_arc_indices, index) else {
            continue;
        };
        let cost = residual_cycle_cost(arcs, &cycle);
        if cost < best_cost {
            best_cost = cost;
            best = Some(cycle);
        }
    }
    best
}

pub(crate) fn find_negative_tree_cycle_fast(
    arcs: &[ResidualArc],
    node_count: usize,
    tree_arc_indices: &[usize],
) -> Option<Vec<usize>> {
    let paths = TreePathCosts::new(arcs, node_count, tree_arc_indices)?;
    let tree_mask = tree_edge_mask(arcs, tree_arc_indices);
    let mut best_arc = None;
    let mut best_cost = 0_i64;
    for (index, arc) in arcs.iter().enumerate() {
        if tree_mask[index] {
            continue;
        }
        let Some(path_cost) = paths.path_cost(arc.to, arc.from) else {
            continue;
        };
        let cost = i64::from(arc.cost) + path_cost;
        if cost < best_cost {
            best_cost = cost;
            best_arc = Some(index);
        }
    }
    tree_cycle_for_arc(arcs, node_count, tree_arc_indices, best_arc?)
}

pub(crate) fn cancel_negative_tree_cycles_with_tree(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    tree_arc_indices: &[usize],
    max_cycles: usize,
) -> usize {
    let node_count = nrow.saturating_sub(1) * ncol.saturating_sub(1) + 1;
    let mut applied = 0;
    for _ in 0..max_cycles {
        let arcs = build_unit_residual_arcs(horizontal, vertical, nrow, ncol);
        let Some(cycle) = find_negative_tree_cycle_fast(&arcs, node_count, tree_arc_indices) else {
            break;
        };
        apply_residual_cycle(horizontal, vertical, &arcs, &cycle);
        applied += 1;
    }
    applied
}

pub(crate) fn optimize_tree_cycles(
    horizontal: &mut [Option<EdgeDatum>],
    vertical: &mut [Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    max_cycles: usize,
) -> usize {
    let node_count = nrow.saturating_sub(1) * ncol.saturating_sub(1) + 1;
    let arcs = build_unit_residual_arcs(horizontal, vertical, nrow, ncol);
    let tree = spanning_tree_arc_indices(&arcs, node_count);
    if tree.len() + 1 != node_count {
        return 0;
    }
    cancel_negative_tree_cycles_with_tree(horizontal, vertical, nrow, ncol, &tree, max_cycles)
}
