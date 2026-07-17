use crate::stage6::unwrap::native::EdgeDatum;
use crate::stage6::unwrap::residual_view::CompactResidualView;

use super::CompactTreeBasis;

fn edge() -> EdgeDatum {
    EdgeDatum {
        cost: 2,
        desired_delta: 0.0,
        offset: 0,
        dzmax: 100,
        laycost: 1_000,
        nshortcycle: 1,
        flow_sign: 1,
        flow: 0,
    }
}

fn topology(basis: &CompactTreeBasis) -> Vec<Vec<(u32, u32, u32)>> {
    (0..basis.node_count)
        .map(|node| {
            basis
                .adjacency
                .neighbors(node)
                .iter()
                .map(|entry| (entry.next, entry.down, entry.up))
                .collect()
        })
        .collect()
}

#[test]
fn rebuilt_workspace_matches_fresh_basis_after_pivot_and_cost_change() {
    let (nrow, ncol) = (3, 3);
    let mut horizontal = vec![Some(edge()); nrow * (ncol - 1)];
    let mut vertical = vec![Some(edge()); (nrow - 1) * ncol];
    let mut tree;
    let mut reused;
    {
        let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
        tree = super::super::spanning_tree_arc_indices_compact(&view);
        reused = CompactTreeBasis::new(&view, &tree).unwrap();
        reused.refresh_root_costs(&view).unwrap();
        let cycle = (0..view.arc_count())
            .filter(|&index| !reused.adjacency.is_tree_arc(index))
            .find_map(|index| reused.cycle_for_arc(&view, index))
            .unwrap();
        assert!(super::super::pivot_compact_tree_on_cycle(
            &view, &mut tree, &cycle
        ));
    }
    let parent_storage = reused.parent_base.as_ptr();
    let root_cost_storage = reused.up_root_cost.as_ptr();
    let adjacency_storage = reused.adjacency.storage_ptrs();
    for datum in horizontal.iter_mut().chain(vertical.iter_mut()).flatten() {
        datum.flow += 5;
    }

    let view = CompactResidualView::new(&horizontal, &vertical, nrow, ncol);
    reused.rebuild(&view, &tree).unwrap();
    reused.refresh_root_costs(&view).unwrap();
    let mut fresh = CompactTreeBasis::new(&view, &tree).unwrap();
    fresh.refresh_root_costs(&view).unwrap();

    assert_eq!(reused.parent_base.as_ptr(), parent_storage);
    assert_eq!(reused.up_root_cost.as_ptr(), root_cost_storage);
    assert_eq!(reused.adjacency.storage_ptrs(), adjacency_storage);
    assert_eq!(topology(&reused), topology(&fresh));
    assert_eq!(reused.depth, fresh.depth);
    assert_eq!(reused.parent_base, fresh.parent_base);
    assert_eq!(reused.chain_head, fresh.chain_head);
    assert_eq!(reused.up_arc, fresh.up_arc);
    assert_eq!(reused.down_arc, fresh.down_arc);
    assert_eq!(reused.order, fresh.order);
    assert_eq!(reused.up_root_cost, fresh.up_root_cost);
    assert_eq!(reused.down_root_cost, fresh.down_root_cost);
    assert_eq!(
        reused.negative_cycles(&view, 16, false),
        fresh.negative_cycles(&view, 16, false)
    );
}
