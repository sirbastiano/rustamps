import importlib.util
import multiprocessing as mp
import os
import queue
from pathlib import Path

import numpy as np
import pytest

from pystamps.io.mat import read_mat
from pystamps.kernels import run_stage6_unwrap_grid_kernel
from scripts.stage6_hf_core import initial_defo_objective


pytestmark = pytest.mark.skipif(
    os.environ.get("PYSTAMPS_ENABLE_HF_FIXTURE_TESTS") != "1",
    reason="set PYSTAMPS_ENABLE_HF_FIXTURE_TESTS=1 to run the local HF SNAPHU fixture parity test",
)

# SNAPHU's float32 output carries tiny nonzero differences on otherwise flat
# neighbor edges; exact `diff != 0` makes the MSD denominator backend-dependent.
_MSD_ZERO_EPS = 5.0e-6
_MSD_ABS_TOL = 5.0e-3
_OBJECTIVE_REL_TOL = 1.0e-3
_MAX_FLOW_DIFF_EDGES = 2500


def _stable_dense_msd(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    diff1 = (arr[:-1, :] - arr[1:, :]).reshape(-1)
    diff1 = diff1[np.abs(diff1) > _MSD_ZERO_EPS]
    diff2 = (arr[:, :-1] - arr[:, 1:]).reshape(-1)
    diff2 = diff2[np.abs(diff2) > _MSD_ZERO_EPS]
    denom = diff1.size + diff2.size
    if denom == 0:
        return 0.0
    return float((np.sum(diff1.astype(np.float64) ** 2) + np.sum(diff2.astype(np.float64) ** 2)) / denom)


def _fixture_metrics(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    native: np.ndarray,
    snaphu: np.ndarray,
) -> tuple[float, float, int, int, int]:
    wrap_diff = np.angle(np.exp(1j * (native - snaphu)))
    label_diff = np.rint((native - snaphu) / (2.0 * np.pi)).astype(np.int16)
    changed_flow_edges = int(np.count_nonzero(label_diff[:, 1:] != label_diff[:, :-1])) + int(
        np.count_nonzero(label_diff[1:, :] != label_diff[:-1, :])
    )
    native_objective = initial_defo_objective(ifgw, rowcost, colcost, native)
    snaphu_objective = initial_defo_objective(ifgw, rowcost, colcost, snaphu)
    return (
        _stable_dense_msd(native),
        float(np.nanmax(np.abs(wrap_diff))),
        native_objective,
        snaphu_objective,
        changed_flow_edges,
    )


def _measure_native_fixture(root: str, outq: mp.Queue) -> None:
    dataset_root = Path(root)
    nzix = np.asarray(read_mat(dataset_root / "uw_grid.mat")["nzix"], dtype=bool)
    nrow, ncol = nzix.shape
    row_elems = (nrow - 1) * ncol * 4
    cost_raw = np.fromfile(dataset_root / "snaphu.costinfile", dtype=np.int16)
    rowcost = cost_raw[:row_elems].reshape((nrow - 1, ncol, 4))
    colcost = cost_raw[row_elems:].reshape((nrow, ncol - 1, 4))
    ifgw = np.fromfile(dataset_root / "snaphu.in", dtype=np.complex64).reshape((nrow, ncol))
    snaphu = np.fromfile(dataset_root / "snaphu.out", dtype=np.float32).reshape((nrow, ncol))

    observed = run_stage6_unwrap_grid_kernel(
        ifgw,
        rowcost.reshape((nrow - 1, ncol * 4)),
        colcost.reshape((nrow, (ncol - 1) * 4)),
        backend="native",
        nshortcycle=200.0,
    )
    ifguw = np.asarray(observed["ifguw"], dtype=np.float32)
    outq.put(_fixture_metrics(ifgw, rowcost, colcost, ifguw, snaphu))


def _run_native_fixture_with_timeout(root: Path) -> tuple[float, float, int, int, int]:
    methods = mp.get_all_start_methods()
    ctx = mp.get_context("fork" if "fork" in methods else methods[0])
    outq = ctx.Queue()
    proc = ctx.Process(target=_measure_native_fixture, args=(str(root), outq))
    proc.start()
    timeout = float(os.environ.get("PYSTAMPS_STAGE6_FIXTURE_TIMEOUT", "1200"))
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail(f"native Stage 6 fixture timed out after {timeout:g}s")
    if proc.exitcode != 0:
        pytest.fail(f"native Stage 6 fixture child failed with exit code {proc.exitcode}")
    try:
        return outq.get_nowait()
    except queue.Empty:
        pytest.fail("native Stage 6 fixture child produced no result")


def _default_fixture_root() -> Path:
    configured = os.environ.get("PYSTAMPS_STAGE6_FIXTURE_ROOT")
    if configured:
        return Path(configured)
    retained = Path("inputs_and_outputs/validation_runs/stage6_fixture_minimal")
    if retained.exists():
        return retained
    return Path("inputs_and_outputs/InSAR_dataset_test")


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_saved_hf_fixture_matches_snaphu_stable_dense_msd() -> None:
    root = _default_fixture_root()
    required = [root / "uw_grid.mat", root / "snaphu.in", root / "snaphu.costinfile", root / "snaphu.out"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing local HF SNAPHU fixture files: {missing}")

    nzix = np.asarray(read_mat(root / "uw_grid.mat")["nzix"], dtype=bool)
    nrow, ncol = nzix.shape
    row_elems = (nrow - 1) * ncol * 4
    col_elems = nrow * (ncol - 1) * 4
    cost_raw = np.fromfile(root / "snaphu.costinfile", dtype=np.int16)
    assert cost_raw.size == row_elems + col_elems

    rowcost = cost_raw[:row_elems].reshape((nrow - 1, ncol, 4))
    colcost = cost_raw[row_elems:].reshape((nrow, ncol - 1, 4))
    ifgw = np.fromfile(root / "snaphu.in", dtype=np.complex64).reshape((nrow, ncol))
    snaphu = np.fromfile(root / "snaphu.out", dtype=np.float32).reshape((nrow, ncol))

    configured_native_file = os.environ.get("PYSTAMPS_STAGE6_NATIVE_FILE")
    if configured_native_file:
        native_file = Path(configured_native_file)
        if not native_file.exists():
            pytest.fail(f"configured cached native fixture does not exist: {native_file}")
        ifguw = np.load(native_file)
        if ifguw.shape != snaphu.shape:
            pytest.fail(f"cached native fixture shape {ifguw.shape} does not match SNAPHU shape {snaphu.shape}")
        metrics = _fixture_metrics(ifgw, rowcost, colcost, ifguw, snaphu)
    else:
        metrics = _run_native_fixture_with_timeout(root)
    observed_msd, max_wrap_diff, native_objective, snaphu_objective, changed_flow_edges = metrics
    assert max_wrap_diff < 1e-4
    assert observed_msd == pytest.approx(_stable_dense_msd(snaphu), abs=_MSD_ABS_TOL)
    objective_rel_tol = float(os.environ.get("PYSTAMPS_STAGE6_OBJECTIVE_REL_TOL", str(_OBJECTIVE_REL_TOL)))
    max_objective_gap = int(np.ceil(abs(snaphu_objective) * objective_rel_tol))
    assert native_objective - snaphu_objective <= max_objective_gap
    max_flow_diff_edges = int(os.environ.get("PYSTAMPS_STAGE6_MAX_FLOW_DIFF_EDGES", str(_MAX_FLOW_DIFF_EDGES)))
    assert changed_flow_edges <= max_flow_diff_edges
