use crate::stage6_residual::ResidualArc;
use crate::stage6_residual_view::CompactResidualView;
use std::collections::VecDeque;

pub(crate) struct TreePathCosts {
    depth: Vec<usize>,
    parent: Vec<u32>,
    up_root_cost: Vec<i64>,
    down_root_cost: Vec<i64>,
    levels: usize,
    node_count: usize,
}

impl TreePathCosts {
    fn index(&self, level: usize, node: usize) -> usize {
        level * self.node_count + node
    }

    pub(crate) fn new(
        arcs: &[ResidualArc],
        node_count: usize,
        tree_arc_indices: &[usize],
    ) -> Option<Self> {
        if node_count == 0 || node_count > u32::MAX as usize {
            return None;
        }
        let mut adjacency = vec![Vec::<(usize, i64, i64)>::new(); node_count];
        for &index in tree_arc_indices {
            let arc = arcs.get(index).copied()?;
            let reverse = arcs.get(index ^ 1).copied()?;
            if arc.from >= node_count || arc.to >= node_count {
                return None;
            }
            if reverse.from != arc.to || reverse.to != arc.from {
                return None;
            }
            adjacency[arc.from].push((arc.to, i64::from(arc.cost), i64::from(reverse.cost)));
            adjacency[arc.to].push((arc.from, i64::from(reverse.cost), i64::from(arc.cost)));
        }

        Self::from_adjacency(adjacency)
    }

    pub(crate) fn new_compact(
        view: &CompactResidualView<'_>,
        tree_arc_indices: &[usize],
    ) -> Option<Self> {
        let node_count = view.node_count();
        if node_count == 0 || node_count > u32::MAX as usize {
            return None;
        }
        let mut adjacency = vec![Vec::<(usize, i64, i64)>::new(); node_count];
        for &index in tree_arc_indices {
            let arc = view.arc(index)?;
            let reverse = view.arc(index ^ 1)?;
            if reverse.from != arc.to || reverse.to != arc.from {
                return None;
            }
            adjacency[arc.from].push((arc.to, i64::from(arc.cost), i64::from(reverse.cost)));
            adjacency[arc.to].push((arc.from, i64::from(reverse.cost), i64::from(arc.cost)));
        }
        Self::from_adjacency(adjacency)
    }

    fn from_adjacency(adjacency: Vec<Vec<(usize, i64, i64)>>) -> Option<Self> {
        let node_count = adjacency.len();
        if node_count == 0 || node_count > u32::MAX as usize {
            return None;
        }
        let levels = (usize::BITS - node_count.saturating_sub(1).leading_zeros()) as usize + 1;
        let slots = levels * node_count;
        let mut parent = vec![u32::MAX; slots];
        let mut up_root_cost = vec![0_i64; node_count];
        let mut down_root_cost = vec![0_i64; node_count];
        let mut depth = vec![0_usize; node_count];
        let mut seen = vec![false; node_count];
        let mut queue = VecDeque::from([0_usize]);
        seen[0] = true;
        parent[0] = 0;

        while let Some(node) = queue.pop_front() {
            for &(next, forward_cost, reverse_cost) in &adjacency[node] {
                if seen[next] {
                    continue;
                }
                seen[next] = true;
                parent[next] = node as u32;
                up_root_cost[next] = up_root_cost[node] + reverse_cost;
                down_root_cost[next] = down_root_cost[node] + forward_cost;
                depth[next] = depth[node] + 1;
                queue.push_back(next);
            }
        }
        if seen.iter().any(|value| !*value) {
            return None;
        }

        for level in 1..levels {
            for node in 0..node_count {
                let prev = (level - 1) * node_count + node;
                let mid = parent[prev] as usize;
                let current = level * node_count + node;
                let prev_mid = (level - 1) * node_count + mid;
                parent[current] = parent[prev_mid];
            }
        }

        Some(Self {
            depth,
            parent,
            up_root_cost,
            down_root_cost,
            levels,
            node_count,
        })
    }

    pub(crate) fn table_shape(&self) -> (usize, usize) {
        (self.levels, self.node_count)
    }

    pub(crate) fn directed_cost_storage_len(&self) -> usize {
        self.up_root_cost.len() + self.down_root_cost.len()
    }

    fn raise(&self, mut node: usize, steps: usize) -> Option<usize> {
        if node >= self.depth.len() {
            return None;
        }
        for level in 0..self.levels {
            if ((steps >> level) & 1) == 1 {
                let index = self.index(level, node);
                node = self.parent[index] as usize;
            }
        }
        Some(node)
    }

    pub(crate) fn path_cost(&self, from: usize, to: usize) -> Option<i64> {
        if from >= self.depth.len() || to >= self.depth.len() {
            return None;
        }
        let mut left = from;
        let mut right = to;

        if self.depth[left] > self.depth[right] {
            left = self.raise(left, self.depth[left] - self.depth[right])?;
        } else if self.depth[right] > self.depth[left] {
            right = self.raise(right, self.depth[right] - self.depth[left])?;
        }

        let lca = if left == right {
            left
        } else {
            for level in (0..self.levels).rev() {
                let left_index = self.index(level, left);
                let right_index = self.index(level, right);
                if self.parent[left_index] != self.parent[right_index] {
                    left = self.parent[left_index] as usize;
                    right = self.parent[right_index] as usize;
                }
            }
            self.parent[left] as usize
        };

        Some(
            (self.up_root_cost[from] - self.up_root_cost[lca])
                + (self.down_root_cost[to] - self.down_root_cost[lca]),
        )
    }
}
