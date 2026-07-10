#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.stage6_residual_cycle_diagnostics import _has_negative_cycle, _residual_arcs


@dataclass(frozen=True)
class _Basis:
    parent: list[int]
    ancestors: list[list[int]]
    depth: list[int]
    up_arc: list[int]
    down_arc: list[int]
    up_root_cost: list[int]
    down_root_cost: list[int]
    tree_mask: list[bool]


def tree_candidate_summary_from_flows(
    rowcost: np.ndarray,
    colcost: np.ndarray,
    h_flow: np.ndarray,
    v_flow: np.ndarray,
    *,
    nshortcycle: int = 200,
    max_nodes: int = 20_000,
    cycle_check_max_nodes: int | None = None,
    max_remounts: int = 8,
) -> dict[str, Any]:
    rows = np.asarray(rowcost, dtype=np.int16)
    cols = np.asarray(colcost, dtype=np.int16)
    h = np.asarray(h_flow, dtype=np.int64)
    v = np.asarray(v_flow, dtype=np.int64)
    nrow = cols.shape[0]
    ncol = rows.shape[1] if rows.size else cols.shape[1] + 1
    node_count = max(nrow - 1, 0) * max(ncol - 1, 0) + 1
    if node_count > max_nodes:
        return {"status": "skipped", "node_count": int(node_count), "max_nodes": int(max_nodes)}

    arcs = _residual_arcs(rows, cols, h, v, nshortcycle)
    cycle_limit = max_nodes if cycle_check_max_nodes is None else cycle_check_max_nodes
    negative_cycle = None
    if node_count <= cycle_limit:
        negative_cycle, _last_cost = _has_negative_cycle(node_count, arcs)
    tree = _spanning_tree(arcs, node_count)
    retained = _find_negative_tree_cycle(arcs, node_count, tree)
    candidates = [index for index, (_start, _end, cost) in enumerate(arcs) if cost < 0]
    remounts = _reduced_cost_remounts(arcs, node_count, tree, candidates, max_remounts)
    remounted = _find_negative_tree_cycle(arcs, node_count, tree)
    return {
        "status": "ok",
        "node_count": int(node_count),
        "arc_count": int(len(arcs)),
        "negative_cycle": None if negative_cycle is None else bool(negative_cycle),
        "retained_tree_negative_cycle": bool(retained),
        "negative_arc_candidate_count": int(len(candidates)),
        "reduced_cost_remounts": int(remounts),
        "remounted_tree_negative_cycle": bool(remounted),
    }


def _spanning_tree(arcs: list[tuple[int, int, int]], node_count: int) -> list[int]:
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    out = []
    for index, (start, end, _cost) in enumerate(arcs):
        left = find(start)
        right = find(end)
        if left == right:
            continue
        parent[right] = left
        out.append(index)
        if len(out) + 1 == node_count:
            break
    return out


def _basis(arcs: list[tuple[int, int, int]], node_count: int, tree: list[int]) -> _Basis | None:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(node_count)]
    tree_mask = [False] * len(arcs)
    for index in tree:
        if index >= len(arcs):
            continue
        start, end, _cost = arcs[index]
        reverse = index ^ 1
        if reverse >= len(arcs) or arcs[reverse][0] != end or arcs[reverse][1] != start:
            return None
        tree_mask[index] = True
        tree_mask[reverse] = True
        adjacency[start].append((end, index))
        adjacency[end].append((start, reverse))

    parent = [-1] * node_count
    depth = [0] * node_count
    up_arc = [-1] * node_count
    down_arc = [-1] * node_count
    up_root_cost = [0] * node_count
    down_root_cost = [0] * node_count
    stack = [0]
    parent[0] = 0
    while stack:
        node = stack.pop()
        for next_node, arc_index in adjacency[node]:
            if parent[next_node] != -1:
                continue
            parent[next_node] = node
            depth[next_node] = depth[node] + 1
            down_arc[next_node] = arc_index
            up_arc[next_node] = arc_index ^ 1
            up_root_cost[next_node] = up_root_cost[node] + arcs[arc_index ^ 1][2]
            down_root_cost[next_node] = down_root_cost[node] + arcs[arc_index][2]
            stack.append(next_node)
    if any(value < 0 for value in parent):
        return None
    ancestors = _ancestor_table(parent)
    return _Basis(parent, ancestors, depth, up_arc, down_arc, up_root_cost, down_root_cost, tree_mask)


def _find_negative_tree_cycle(
    arcs: list[tuple[int, int, int]],
    node_count: int,
    tree: list[int],
) -> bool:
    basis = _basis(arcs, node_count, tree)
    if basis is None:
        return False
    for index, (start, end, cost) in enumerate(arcs):
        if basis.tree_mask[index]:
            continue
        cycle_cost = cost + _path_cost(arcs, basis, end, start)
        if cycle_cost < 0:
            return True
    return False


def _reduced_cost_remounts(
    arcs: list[tuple[int, int, int]],
    node_count: int,
    tree: list[int],
    candidates: list[int],
    max_remounts: int,
) -> int:
    applied = 0
    for _ in range(max_remounts):
        basis = _basis(arcs, node_count, tree)
        if basis is None:
            break
        best_gain = 0
        best: tuple[int, int] | None = None
        for index in candidates:
            if index >= len(arcs) or basis.tree_mask[index]:
                continue
            start, end, cost = arcs[index]
            if end == 0:
                continue
            gain = basis.down_root_cost[end] - basis.down_root_cost[start] - cost
            if gain <= best_gain or _lca(basis, end, start) == end:
                continue
            best_gain = gain
            best = (index, basis.down_arc[end] // 2)
        if best is None:
            break
        entering, leaving_pair = best
        for pos, index in enumerate(tree):
            if index // 2 == leaving_pair:
                tree[pos] = entering
                applied += 1
                break
        else:
            break
    return applied


def _path_cost(arcs: list[tuple[int, int, int]], basis: _Basis, start: int, end: int) -> int:
    del arcs
    root = _lca(basis, start, end)
    return (basis.up_root_cost[start] - basis.up_root_cost[root]) + (
        basis.down_root_cost[end] - basis.down_root_cost[root]
    )


def _lca(basis: _Basis, left: int, right: int) -> int:
    if basis.depth[left] > basis.depth[right]:
        left = _raise(basis, left, basis.depth[left] - basis.depth[right])
    elif basis.depth[right] > basis.depth[left]:
        right = _raise(basis, right, basis.depth[right] - basis.depth[left])
    if left == right:
        return left
    for level in range(len(basis.ancestors) - 1, -1, -1):
        if basis.ancestors[level][left] != basis.ancestors[level][right]:
            left = basis.ancestors[level][left]
            right = basis.ancestors[level][right]
    return basis.parent[left]


def _raise(basis: _Basis, node: int, steps: int) -> int:
    level = 0
    while steps:
        if steps & 1:
            node = basis.ancestors[level][node]
        steps >>= 1
        level += 1
    return node


def _ancestor_table(parent: list[int]) -> list[list[int]]:
    levels = max(len(parent).bit_length(), 1)
    table = [parent.copy()]
    for level in range(1, levels):
        previous = table[level - 1]
        table.append([previous[previous[node]] for node in range(len(parent))])
    return table
