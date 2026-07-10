from __future__ import annotations

import numpy as np

from pystamps.io.mat import read_mat, write_mat
from scripts.stage6_crop_fixture import crop_arrays, crop_bounds, crop_fixture


def test_crop_arrays_preserves_stage6_edge_contracts() -> None:
    nrow, ncol = 4, 5
    nzix = np.arange(nrow * ncol).reshape(nrow, ncol) % 2 == 0
    ifgw = (np.arange(nrow * ncol).reshape(nrow, ncol) + 1j).astype(np.complex64)
    snaphu = np.arange(nrow * ncol, dtype=np.float32).reshape(nrow, ncol)
    rowcost = np.arange((nrow - 1) * ncol * 4, dtype=np.int16).reshape(nrow - 1, ncol, 4)
    colcost = np.arange(nrow * (ncol - 1) * 4, dtype=np.int16).reshape(nrow, ncol - 1, 4)

    crop = crop_arrays(nzix, ifgw, rowcost, colcost, snaphu, 1, 4, 2, 5)

    np.testing.assert_array_equal(crop["nzix"], nzix[1:4, 2:5])
    np.testing.assert_array_equal(crop["ifgw"], ifgw[1:4, 2:5])
    np.testing.assert_array_equal(crop["snaphu"], snaphu[1:4, 2:5])
    np.testing.assert_array_equal(crop["rowcost"], rowcost[1:3, 2:5, :])
    np.testing.assert_array_equal(crop["colcost"], colcost[1:4, 2:4, :])


def test_crop_bounds_clamps_inclusive_bbox_with_margin() -> None:
    assert crop_bounds((4, 5), (0, 0, 0, 0), margin=1) == (0, 2, 0, 2)
    assert crop_bounds((4, 5), (2, 3, 3, 4), margin=2) == (0, 4, 1, 5)


def test_crop_fixture_writes_minimal_stage6_files(tmp_path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    nrow, ncol = 3, 4
    nzix = np.ones((nrow, ncol), dtype=bool)
    ifgw = (np.arange(nrow * ncol).reshape(nrow, ncol) + 2j).astype(np.complex64)
    snaphu = np.arange(nrow * ncol, dtype=np.float32).reshape(nrow, ncol)
    rowcost = np.arange((nrow - 1) * ncol * 4, dtype=np.int16).reshape(nrow - 1, ncol, 4)
    colcost = np.arange(nrow * (ncol - 1) * 4, dtype=np.int16).reshape(nrow, ncol - 1, 4)

    write_mat(source / "uw_grid.mat", {"nzix": nzix})
    ifgw.tofile(source / "snaphu.in")
    np.concatenate([rowcost.reshape(-1), colcost.reshape(-1)]).tofile(source / "snaphu.costinfile")
    snaphu.tofile(source / "snaphu.out")

    crop_fixture(source, dest, bbox=(1, 2, 1, 3), margin=0)

    written_nzix = np.asarray(read_mat(dest / "uw_grid.mat")["nzix"], dtype=bool)
    written_ifgw = np.fromfile(dest / "snaphu.in", dtype=np.complex64).reshape((2, 3))
    written_cost = np.fromfile(dest / "snaphu.costinfile", dtype=np.int16)
    written_snaphu = np.fromfile(dest / "snaphu.out", dtype=np.float32).reshape((2, 3))

    np.testing.assert_array_equal(written_nzix, nzix[1:3, 1:4])
    np.testing.assert_array_equal(written_ifgw, ifgw[1:3, 1:4])
    np.testing.assert_array_equal(written_cost[: 1 * 3 * 4].reshape(1, 3, 4), rowcost[1:2, 1:4])
    np.testing.assert_array_equal(written_cost[1 * 3 * 4 :].reshape(2, 2, 4), colcost[1:3, 1:3])
    np.testing.assert_array_equal(written_snaphu, snaphu[1:3, 1:4])
