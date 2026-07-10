from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from scripts.stage6_hf_flow_diagnostics import TWO_PI, _desired_flows, _edge_cost, _reshape_costs

INF_CAP = 1 << 58


class _Dinic:
    def __init__(self, nodes: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(nodes)]

    def add_arc(self, src: int, dst: int, cap: int) -> None:
        if cap <= 0:
            return
        rev_dst = len(self.graph[dst])
        rev_src = len(self.graph[src])
        self.graph[src].append([dst, rev_dst, int(cap)])
        self.graph[dst].append([src, rev_src, 0])

    def add_cut_edge(self, a: int, b: int, cap: int) -> None:
        self.add_arc(a, b, cap)
        self.add_arc(b, a, cap)

    def max_flow(self, source: int, sink: int) -> None:
        while True:
            level = [-1] * len(self.graph)
            queue: deque[int] = deque([source])
            level[source] = 0
            while queue:
                node = queue.popleft()
                for dst, _rev, cap in self.graph[node]:
                    if cap > 0 and level[dst] < 0:
                        level[dst] = level[node] + 1
                        queue.append(dst)
            if level[sink] < 0:
                return
            iters = [0] * len(self.graph)
            while self._dfs(source, sink, INF_CAP, level, iters) > 0:
                pass

    def _dfs(self, node: int, sink: int, flow: int, level: list[int], iters: list[int]) -> int:
        if node == sink:
            return flow
        while iters[node] < len(self.graph[node]):
            edge_ix = iters[node]
            dst, rev, cap = self.graph[node][edge_ix]
            if cap > 0 and level[node] + 1 == level[dst]:
                pushed = self._dfs(dst, sink, min(flow, cap), level, iters)
                if pushed > 0:
                    self.graph[node][edge_ix][2] -= pushed
                    self.graph[dst][rev][2] += pushed
                    return pushed
            iters[node] += 1
        return 0

    def reachable_from(self, source: int) -> list[bool]:
        seen = [False] * len(self.graph)
        queue: deque[int] = deque([source])
        seen[source] = True
        while queue:
            node = queue.popleft()
            for dst, _rev, cap in self.graph[node]:
                if cap > 0 and not seen[dst]:
                    seen[dst] = True
                    queue.append(dst)
        return seen


def replay_binary_cut_patch(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    patch: tuple[int, int, int, int],
    *,
    shift: int,
    oracle_mask: np.ndarray | None = None,
    nshortcycle: int = 200,
) -> dict[str, Any]:
    phase = np.angle(np.asarray(ifgw, dtype=np.complex64)).astype(np.float32)
    labels = np.rint((np.asarray(native, dtype=np.float32) - phase) / TWO_PI).astype(np.int64)
    rows, cols = _reshape_costs(rowcost, colcost, phase.shape)
    h_desired, v_desired = _desired_flows(phase)
    row0, col0, height, width = patch
    cells = height * width
    source = cells
    sink = cells + 1
    graph = _Dinic(cells + 2)
    stats = {"pair_terms": 0, "skipped_non_submodular_pairs": 0}

    if row0 == 0 and col0 == 0:
        _add_unary(graph, source, sink, 0, 0, INF_CAP)
    for row in range(row0, row0 + height):
        for col in range(max(col0 - 1, 0), min(col0 + width, phase.shape[1] - 1)):
            costs = [
                _h_energy(rows, cols, labels, h_desired, row, col, patch, shift, False, False, nshortcycle),
                _h_energy(rows, cols, labels, h_desired, row, col, patch, shift, True, False, nshortcycle),
                _h_energy(rows, cols, labels, h_desired, row, col, patch, shift, False, True, nshortcycle),
                _h_energy(rows, cols, labels, h_desired, row, col, patch, shift, True, True, nshortcycle),
            ]
            _add_edge_terms(graph, source, sink, (row, col), (row, col + 1), patch, width, costs, stats)
    for row in range(max(row0 - 1, 0), min(row0 + height, phase.shape[0] - 1)):
        for col in range(col0, col0 + width):
            costs = [
                _v_energy(rows, cols, labels, v_desired, row, col, patch, shift, False, False, nshortcycle),
                _v_energy(rows, cols, labels, v_desired, row, col, patch, shift, True, False, nshortcycle),
                _v_energy(rows, cols, labels, v_desired, row, col, patch, shift, False, True, nshortcycle),
                _v_energy(rows, cols, labels, v_desired, row, col, patch, shift, True, True, nshortcycle),
            ]
            _add_edge_terms(graph, source, sink, (row, col), (row + 1, col), patch, width, costs, stats)

    graph.max_flow(source, sink)
    selected = np.asarray([not seen for seen in graph.reachable_from(source)[:cells]], dtype=bool)
    selected = selected.reshape((height, width))
    before = _patch_energy(rows, cols, labels, h_desired, v_desired, patch, None, shift, nshortcycle)
    after = _patch_energy(rows, cols, labels, h_desired, v_desired, patch, selected, shift, nshortcycle)
    out = {
        "patch": [int(v) for v in patch],
        "shift": int(shift),
        "selected_count": int(selected.sum()),
        "selected_gain": int(before - after),
        "selected_mask": selected.tolist(),
        **stats,
    }
    if oracle_mask is not None:
        mask = np.asarray(oracle_mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError("oracle_mask shape must match patch height/width")
        oracle_after = _patch_energy(rows, cols, labels, h_desired, v_desired, patch, mask, shift, nshortcycle)
        out["oracle_count"] = int(mask.sum())
        out["oracle_gain"] = int(before - oracle_after)
    return out


def _add_unary(graph: _Dinic, source: int, sink: int, node: int, d0: int, d1: int) -> None:
    base = min(d0, d1)
    graph.add_arc(source, node, d1 - base)
    graph.add_arc(node, sink, d0 - base)


def _add_unary_delta(graph: _Dinic, source: int, sink: int, node: int, coeff: int) -> None:
    _add_unary(graph, source, sink, node, 0, coeff)


def _add_pair(
    graph: _Dinic,
    source: int,
    sink: int,
    left: int,
    right: int,
    costs: list[int],
    stats: dict[str, int],
) -> None:
    e00, e10, e01, e11 = costs
    weight = e10 + e01 - e00 - e11
    stats["pair_terms"] += 1
    if weight < 0:
        stats["skipped_non_submodular_pairs"] += 1
        return
    _add_unary_delta(graph, source, sink, left, 2 * (e10 - e00) - weight)
    _add_unary_delta(graph, source, sink, right, 2 * (e01 - e00) - weight)
    graph.add_cut_edge(left, right, weight)


def _add_edge_terms(
    graph: _Dinic,
    source: int,
    sink: int,
    a: tuple[int, int],
    b: tuple[int, int],
    patch: tuple[int, int, int, int],
    width: int,
    costs: list[int],
    stats: dict[str, int],
) -> None:
    row0, col0, height, _ = patch
    a_in = row0 <= a[0] < row0 + height and col0 <= a[1] < col0 + width
    b_in = row0 <= b[0] < row0 + height and col0 <= b[1] < col0 + width
    if a_in and b_in:
        _add_pair(graph, source, sink, _local_ix(a, row0, col0, width), _local_ix(b, row0, col0, width), costs, stats)
    elif a_in:
        _add_unary(graph, source, sink, _local_ix(a, row0, col0, width), 2 * costs[0], 2 * costs[1])
    elif b_in:
        _add_unary(graph, source, sink, _local_ix(b, row0, col0, width), 2 * costs[0], 2 * costs[2])


def _local_ix(cell: tuple[int, int], row0: int, col0: int, width: int) -> int:
    return (cell[0] - row0) * width + (cell[1] - col0)


def _cell_delta(mask: np.ndarray | None, cell: tuple[int, int], patch: tuple[int, int, int, int], shift: int) -> int:
    if mask is None:
        return 0
    row0, col0, height, width = patch
    row, col = cell
    if row0 <= row < row0 + height and col0 <= col < col0 + width and mask[row - row0, col - col0]:
        return shift
    return 0


def _h_energy(_rows, cols, labels, h_desired, row, col, patch, shift, left, right, nshortcycle) -> int:
    mask = np.asarray([[left, right]], dtype=bool)
    local_patch = (row, col, 1, 2)
    flow = labels[row, col + 1] + _cell_delta(mask, (row, col + 1), local_patch, shift)
    flow -= labels[row, col] + _cell_delta(mask, (row, col), local_patch, shift)
    flow -= h_desired[row, col]
    return int(_edge_cost(cols[row : row + 1, col : col + 1], np.asarray([[flow]]), nshortcycle)[0, 0])


def _v_energy(rows, _cols, labels, v_desired, row, col, patch, shift, upper, lower, nshortcycle) -> int:
    mask = np.asarray([[upper], [lower]], dtype=bool)
    local_patch = (row, col, 2, 1)
    flow = labels[row + 1, col] + _cell_delta(mask, (row + 1, col), local_patch, shift)
    flow -= labels[row, col] + _cell_delta(mask, (row, col), local_patch, shift)
    flow = -flow + v_desired[row, col]
    return int(_edge_cost(rows[row : row + 1, col : col + 1], np.asarray([[flow]]), nshortcycle)[0, 0])


def _patch_energy(rows, cols, labels, h_desired, v_desired, patch, mask, shift, nshortcycle) -> int:
    row0, col0, height, width = patch
    nrow, ncol = labels.shape
    total = 0
    for row in range(row0, row0 + height):
        for col in range(max(col0 - 1, 0), min(col0 + width, ncol - 1)):
            left = labels[row, col] + _cell_delta(mask, (row, col), patch, shift)
            right = labels[row, col + 1] + _cell_delta(mask, (row, col + 1), patch, shift)
            flow = right - left - h_desired[row, col]
            total += int(_edge_cost(cols[row : row + 1, col : col + 1], np.asarray([[flow]]), nshortcycle)[0, 0])
    for row in range(max(row0 - 1, 0), min(row0 + height, nrow - 1)):
        for col in range(col0, col0 + width):
            upper = labels[row, col] + _cell_delta(mask, (row, col), patch, shift)
            lower = labels[row + 1, col] + _cell_delta(mask, (row + 1, col), patch, shift)
            flow = -(lower - upper - v_desired[row, col])
            total += int(_edge_cost(rows[row : row + 1, col : col + 1], np.asarray([[flow]]), nshortcycle)[0, 0])
    return int(total)
