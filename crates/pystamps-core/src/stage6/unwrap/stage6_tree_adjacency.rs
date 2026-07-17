use crate::stage6::unwrap::residual_view::CompactResidualView;

#[derive(Clone, Copy)]
pub(super) struct TreeEntry {
    pub next: u32,
    pub down: u32,
    pub up: u32,
}

#[derive(Default)]
pub(super) struct TreeAdjacency {
    degrees: Vec<usize>,
    offsets: Vec<usize>,
    entries: Vec<TreeEntry>,
    tree_mask: Vec<bool>,
}

impl TreeAdjacency {
    #[cfg(test)]
    pub fn new(view: &CompactResidualView<'_>, tree_arc_indices: &[usize]) -> Option<Self> {
        let mut adjacency = Self::default();
        adjacency.rebuild(view, tree_arc_indices)?;
        Some(adjacency)
    }

    pub fn rebuild(
        &mut self,
        view: &CompactResidualView<'_>,
        tree_arc_indices: &[usize],
    ) -> Option<()> {
        let node_count = view.node_count();
        if node_count == 0 || node_count > u32::MAX as usize || view.arc_count() > u32::MAX as usize
        {
            return None;
        }
        self.degrees.resize(node_count, 0);
        self.degrees.fill(0);
        self.tree_mask.resize(view.arc_count(), false);
        self.tree_mask.fill(false);
        for &index in tree_arc_indices {
            let (from, to) = arc_pair(view, index)?;
            if from >= node_count || to >= node_count {
                return None;
            }
            self.degrees[from] = self.degrees[from].checked_add(1)?;
            self.degrees[to] = self.degrees[to].checked_add(1)?;
            self.tree_mask[index] = true;
            self.tree_mask[index ^ 1] = true;
        }
        self.offsets.resize(node_count + 1, 0);
        self.offsets[0] = 0;
        for node in 0..node_count {
            self.offsets[node + 1] = self.offsets[node].checked_add(self.degrees[node])?;
            self.degrees[node] = self.offsets[node];
        }
        let empty = TreeEntry {
            next: u32::MAX,
            down: u32::MAX,
            up: u32::MAX,
        };
        self.entries.resize(self.offsets[node_count], empty);
        self.entries.fill(empty);
        for &index in tree_arc_indices {
            let (from, to) = arc_pair(view, index)?;
            let reverse_index = index ^ 1;
            self.entries[self.degrees[from]] = TreeEntry {
                next: to as u32,
                down: index as u32,
                up: reverse_index as u32,
            };
            self.degrees[from] += 1;
            self.entries[self.degrees[to]] = TreeEntry {
                next: from as u32,
                down: reverse_index as u32,
                up: index as u32,
            };
            self.degrees[to] += 1;
        }
        Some(())
    }

    pub fn neighbors(&self, node: usize) -> &[TreeEntry] {
        &self.entries[self.offsets[node]..self.offsets[node + 1]]
    }

    pub fn is_tree_arc(&self, index: usize) -> bool {
        self.tree_mask.get(index).copied().unwrap_or(false)
    }

    #[cfg(test)]
    pub fn storage_ptrs(&self) -> [usize; 3] {
        [
            self.degrees.as_ptr() as usize,
            self.offsets.as_ptr() as usize,
            self.entries.as_ptr() as usize,
        ]
    }
}

fn arc_pair(view: &CompactResidualView<'_>, index: usize) -> Option<(usize, usize)> {
    view.endpoints(index)
}

#[cfg(test)]
mod tests {
    use crate::stage6::unwrap::native::EdgeDatum;
    use crate::stage6::unwrap::residual_view::CompactResidualView;

    use super::TreeAdjacency;

    #[test]
    fn csr_neighbors_preserve_tree_insertion_order_and_arc_orientation() {
        let edge = EdgeDatum {
            cost: 10,
            desired_delta: 0.0,
            offset: 0,
            dzmax: 100,
            laycost: 1_000,
            nshortcycle: 200,
            flow_sign: 1,
            flow: 0,
        };
        let horizontal = vec![Some(edge); 6];
        let vertical = vec![Some(edge); 6];
        let view = CompactResidualView::new(&horizontal, &vertical, 3, 3);
        let tree = [1, 2, 4, 6];
        let mut expected = vec![Vec::new(); view.node_count()];
        for &index in &tree {
            let arc = view.arc(index).unwrap();
            expected[arc.from].push((arc.to, index, index ^ 1));
            expected[arc.to].push((arc.from, index ^ 1, index));
        }

        let adjacency = TreeAdjacency::new(&view, &tree).unwrap();

        for (node, expected) in expected.iter().enumerate() {
            let observed = adjacency
                .neighbors(node)
                .iter()
                .map(|edge| (edge.next as usize, edge.down as usize, edge.up as usize))
                .collect::<Vec<_>>();
            assert_eq!(&observed, expected);
        }
    }
}
