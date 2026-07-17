use crate::stage6_mst::mst_scalar_weight;
use crate::stage6_native::{horizontal_index, vertical_index, EdgeDatum};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct Action {
    is_row: bool,
    index: usize,
    coeff: i32,
}

#[derive(Clone, Copy)]
struct Prev {
    from: usize,
    action: Action,
}

fn apply_action(rowflow: &mut [i32], colflow: &mut [i32], action: Action, signed_amount: i32) {
    if action.is_row {
        rowflow[action.index] += action.coeff * signed_amount;
    } else {
        colflow[action.index] += action.coeff * signed_amount;
    }
}

fn visit_neighbors(
    node: usize,
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
    mut visit: impl FnMut(usize, i64, Action),
) {
    let prn = nrow.saturating_sub(1);
    let pcn = ncol.saturating_sub(1);
    let ground = prn * pcn;
    if node == ground {
        for row in 0..prn {
            for (col, edge_col, coeff) in [(0, 0, 1), (pcn - 1, ncol - 1, -1)] {
                if let Some(edge) = vertical[vertical_index(row, edge_col, ncol)] {
                    visit(
                        row * pcn + col,
                        i64::from(mst_scalar_weight(edge)),
                        Action {
                            is_row: true,
                            index: vertical_index(row, edge_col, ncol),
                            coeff,
                        },
                    );
                }
            }
        }
        for col in 0..pcn {
            for (row, edge_row, coeff) in [(0, 0, 1), (prn - 1, nrow - 1, -1)] {
                if let Some(edge) = horizontal[horizontal_index(edge_row, col, ncol)] {
                    visit(
                        row * pcn + col,
                        i64::from(mst_scalar_weight(edge)),
                        Action {
                            is_row: false,
                            index: horizontal_index(edge_row, col, ncol),
                            coeff,
                        },
                    );
                }
            }
        }
        return;
    }

    let row = node / pcn;
    let col = node % pcn;
    if col > 0 {
        if let Some(edge) = vertical[vertical_index(row, col, ncol)] {
            visit(
                node - 1,
                i64::from(mst_scalar_weight(edge)),
                Action {
                    is_row: true,
                    index: vertical_index(row, col, ncol),
                    coeff: -1,
                },
            );
        }
    } else if let Some(edge) = vertical[vertical_index(row, 0, ncol)] {
        visit(
            ground,
            i64::from(mst_scalar_weight(edge)),
            Action {
                is_row: true,
                index: vertical_index(row, 0, ncol),
                coeff: -1,
            },
        );
    }
    if col + 1 < pcn {
        if let Some(edge) = vertical[vertical_index(row, col + 1, ncol)] {
            visit(
                node + 1,
                i64::from(mst_scalar_weight(edge)),
                Action {
                    is_row: true,
                    index: vertical_index(row, col + 1, ncol),
                    coeff: 1,
                },
            );
        }
    } else if let Some(edge) = vertical[vertical_index(row, ncol - 1, ncol)] {
        visit(
            ground,
            i64::from(mst_scalar_weight(edge)),
            Action {
                is_row: true,
                index: vertical_index(row, ncol - 1, ncol),
                coeff: 1,
            },
        );
    }
    if row > 0 {
        if let Some(edge) = horizontal[horizontal_index(row, col, ncol)] {
            visit(
                node - pcn,
                i64::from(mst_scalar_weight(edge)),
                Action {
                    is_row: false,
                    index: horizontal_index(row, col, ncol),
                    coeff: -1,
                },
            );
        }
    } else if let Some(edge) = horizontal[horizontal_index(0, col, ncol)] {
        visit(
            ground,
            i64::from(mst_scalar_weight(edge)),
            Action {
                is_row: false,
                index: horizontal_index(0, col, ncol),
                coeff: -1,
            },
        );
    }
    if row + 1 < prn {
        if let Some(edge) = horizontal[horizontal_index(row + 1, col, ncol)] {
            visit(
                node + pcn,
                i64::from(mst_scalar_weight(edge)),
                Action {
                    is_row: false,
                    index: horizontal_index(row + 1, col, ncol),
                    coeff: 1,
                },
            );
        }
    } else if let Some(edge) = horizontal[horizontal_index(nrow - 1, col, ncol)] {
        visit(
            ground,
            i64::from(mst_scalar_weight(edge)),
            Action {
                is_row: false,
                index: horizontal_index(nrow - 1, col, ncol),
                coeff: 1,
            },
        );
    }
}

pub(crate) fn mst_initial_flows(
    residue: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> (Vec<i32>, Vec<i32>) {
    shortest_path_initial_flows(residue, horizontal, vertical, nrow, ncol)
}

pub(crate) fn shortest_path_initial_flows(
    residue: &[i32],
    horizontal: &[Option<EdgeDatum>],
    vertical: &[Option<EdgeDatum>],
    nrow: usize,
    ncol: usize,
) -> (Vec<i32>, Vec<i32>) {
    let prn = nrow.saturating_sub(1);
    let pcn = ncol.saturating_sub(1);
    let ground = prn * pcn;
    let node_count = ground + 1;
    let mut rowflow = vec![0; prn * ncol];
    let mut colflow = vec![0; nrow * pcn];
    if residue.iter().all(|value| *value == 0) {
        return (rowflow, colflow);
    }

    let mut charge = vec![0_i32; node_count];
    for (index, value) in residue.iter().copied().enumerate().take(ground) {
        charge[index] = value;
        charge[ground] -= value;
    }
    let terminals: Vec<bool> = charge.iter().map(|value| *value != 0).collect();
    let Some(root) = terminals.iter().position(|value| *value) else {
        return (rowflow, colflow);
    };
    let mut parent = vec![None::<Prev>; node_count];

    let mut dist = vec![i64::MAX; node_count];
    let mut prev = vec![None::<Prev>; node_count];
    let mut heap = BinaryHeap::new();
    dist[root] = 0;
    heap.push(Reverse((0_i64, root)));
    while let Some(Reverse((cost, node))) = heap.pop() {
        if cost != dist[node] {
            continue;
        }
        visit_neighbors(
            node,
            horizontal,
            vertical,
            nrow,
            ncol,
            |to, weight, action| {
                let next_cost = cost + weight;
                if next_cost < dist[to] {
                    dist[to] = next_cost;
                    prev[to] = Some(Prev { from: node, action });
                    heap.push(Reverse((next_cost, to)));
                }
            },
        );
    }

    for target in 0..node_count {
        if !terminals[target] || target == root || dist[target] == i64::MAX {
            continue;
        }
        let mut node = target;
        while node != root && parent[node].is_none() {
            let Some(link) = prev[node] else {
                break;
            };
            parent[node] = Some(link);
            node = link.from;
        }
    }

    let mut child_count = vec![0_u32; node_count];
    for link in parent.iter().flatten() {
        child_count[link.from] += 1;
    }
    let mut pending = Vec::new();
    for node in 0..node_count {
        if parent[node].is_some() && child_count[node] == 0 {
            pending.push(node);
        }
    }
    let mut subtree_charge = charge;
    while let Some(node) = pending.pop() {
        let Some(link) = parent[node] else {
            continue;
        };
        let amount = subtree_charge[node];
        apply_action(&mut rowflow, &mut colflow, link.action, -amount);
        subtree_charge[link.from] += amount;
        child_count[link.from] -= 1;
        if link.from != root && child_count[link.from] == 0 {
            pending.push(link.from);
        }
    }
    (rowflow, colflow)
}
