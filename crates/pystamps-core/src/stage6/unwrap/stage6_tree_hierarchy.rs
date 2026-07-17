#[cfg(test)]
pub(super) fn chain_heads(parent: &[usize], order: &[usize]) -> Vec<u32> {
    let mut heads = Vec::new();
    chain_heads_into(parent, order, &mut heads, &mut Vec::new(), &mut Vec::new());
    heads
}

pub(super) fn chain_heads_into(
    parent: &[usize],
    order: &[usize],
    heads: &mut Vec<u32>,
    subtree_size: &mut Vec<usize>,
    heavy_child: &mut Vec<usize>,
) {
    subtree_size.resize(parent.len(), 1);
    subtree_size.fill(1);
    heavy_child.resize(parent.len(), usize::MAX);
    heavy_child.fill(usize::MAX);
    for &node in order.iter().skip(1).rev() {
        let parent_node = parent[node];
        subtree_size[parent_node] += subtree_size[node];
        let heavy = heavy_child[parent_node];
        if heavy == usize::MAX || subtree_size[node] > subtree_size[heavy] {
            heavy_child[parent_node] = node;
        }
    }
    heads.resize(parent.len(), 0);
    heads.fill(0);
    for &node in order.iter().skip(1) {
        let parent_node = parent[node];
        heads[node] = if heavy_child[parent_node] == node {
            heads[parent_node]
        } else {
            node as u32
        };
    }
}

#[inline(always)]
pub(super) fn lca(
    depth: &[usize],
    parent: &[usize],
    heads: &[u32],
    mut left: usize,
    mut right: usize,
) -> Option<usize> {
    if left >= parent.len() || right >= parent.len() {
        return None;
    }
    while heads[left] != heads[right] {
        let left_head = heads[left] as usize;
        let right_head = heads[right] as usize;
        if depth[left_head] > depth[right_head] {
            left = parent[left_head];
        } else {
            right = parent[right_head];
        }
    }
    Some(if depth[left] < depth[right] {
        left
    } else {
        right
    })
}

#[cfg(test)]
mod tests {
    use super::{chain_heads, lca};

    #[test]
    fn heavy_light_lca_matches_branching_tree_ancestry() {
        let parent = [0, 0, 0, 1, 1, 2, 3, 3, 5];
        let order = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let depth = [0, 1, 1, 2, 2, 2, 3, 3, 3];
        let heads = chain_heads(&parent, &order);

        assert_eq!(lca(&depth, &parent, &heads, 6, 7), Some(3));
        assert_eq!(lca(&depth, &parent, &heads, 4, 7), Some(1));
        assert_eq!(lca(&depth, &parent, &heads, 8, 6), Some(0));
        assert_eq!(lca(&depth, &parent, &heads, 8, 5), Some(5));
        assert_eq!(lca(&depth, &parent, &heads, 9, 0), None);
    }
}
