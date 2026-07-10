import importlib.util

import numpy as np
import pytest

from pystamps.kernels import run_stage6_unwrap_grid_kernel


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_honors_defo_dzmax_shelf() -> None:
    ifgw = np.ones((1, 2), dtype=np.complex64)
    rowcost = np.zeros((0, 8), dtype=np.int16)
    colcost = np.zeros((1, 4), dtype=np.int16)
    colcost[0] = np.asarray([-950, 1, 0, 1], dtype=np.int16)

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)

    assert int(labels[0, 1] - labels[0, 0]) == 5


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_accepts_objective_reducing_column_shift() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    rowcost = np.asarray([[[0, 5, 32000, -32000], [400, 2, 32000, -32000]]], dtype=np.int16)
    colcost = np.asarray([[[200, 5, 32000, -32000]], [[200, 20, 32000, -32000]]], dtype=np.int16)

    out = run_stage6_unwrap_grid_kernel(
        ifgw,
        rowcost.reshape((1, 8)),
        colcost.reshape((2, 4)),
        backend="native",
    )
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    np.testing.assert_array_equal(labels, np.asarray([[0, -1], [0, 1]], dtype=np.int32))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_accepts_objective_reducing_patch_shift() -> None:
    ifgw = np.ones((2, 3), dtype=np.complex64)
    rowcost = np.asarray(
        [[[800, 10, 32000, -32000], [0, 50, 32000, -32000], [600, 1, 32000, -32000]]],
        dtype=np.int16,
    )
    colcost = np.asarray(
        [
            [[-600, 10, 32000, -32000], [200, 1, 32000, -32000]],
            [[600, 1, 32000, -32000], [-800, 20, 32000, -32000]],
        ],
        dtype=np.int16,
    )

    out = run_stage6_unwrap_grid_kernel(
        ifgw,
        rowcost.reshape((1, 12)),
        colcost.reshape((2, 8)),
        backend="native",
    )
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    np.testing.assert_array_equal(labels, np.asarray([[0, 3, 2], [4, 1, 5]], dtype=np.int32))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_accepts_objective_reducing_region_shift() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:5, 1:3] = True
    mask[3, 3] = True

    ifgw = np.ones(mask.shape, dtype=np.complex64)
    rowcost = np.zeros((9, 10, 4), dtype=np.int16)
    colcost = np.zeros((10, 9, 4), dtype=np.int16)
    rowcost[..., 1:4] = np.asarray([50, 32000, -32000], dtype=np.int16)
    colcost[..., 1:4] = np.asarray([50, 32000, -32000], dtype=np.int16)

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1] - 1):
            left = mask[row, col]
            right = mask[row, col + 1]
            if left and right:
                colcost[row, col, 1] = 1
            elif not left and right:
                colcost[row, col] = np.asarray([-200, 2, 32000, -32000], dtype=np.int16)
            elif left and not right:
                colcost[row, col] = np.asarray([200, 2, 32000, -32000], dtype=np.int16)
    for row in range(mask.shape[0] - 1):
        for col in range(mask.shape[1]):
            upper = mask[row, col]
            lower = mask[row + 1, col]
            if upper and lower:
                rowcost[row, col, 1] = 1
            elif not upper and lower:
                rowcost[row, col] = np.asarray([200, 2, 32000, -32000], dtype=np.int16)
            elif upper and not lower:
                rowcost[row, col] = np.asarray([-200, 2, 32000, -32000], dtype=np.int16)

    out = run_stage6_unwrap_grid_kernel(
        ifgw,
        rowcost.reshape((9, 40)),
        colcost.reshape((10, 36)),
        backend="native",
    )
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    np.testing.assert_array_equal(labels, mask.astype(np.int32))
