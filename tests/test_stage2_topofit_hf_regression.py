from __future__ import annotations

from pathlib import Path

import importlib.util

import numpy as np
import pytest

from pystamps.io.mat import read_mat
from pystamps.kernels import run_stage2_topofit_kernel
from pystamps.pipeline import ported


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_native_generic_topofit_matches_python_for_missing_ifgs() -> None:
    patch_dir = Path("inputs_and_outputs/InSAR_dataset_test/PATCH_1")
    if not (patch_dir / "pm1.mat").exists():
        pytest.skip("HF parity fixture not present")

    pm = read_mat(patch_dir / "pm1.mat")
    ps = read_mat(patch_dir / "ps1.mat")
    ph = read_mat(patch_dir / "ph1.mat")["ph"]
    bp = read_mat(patch_dir / "bp1.mat")
    n_ps = int(round(float(np.asarray(ps["n_ps"]).reshape(-1)[0])))
    master_ix = int(round(float(np.asarray(ps["master_ix"]).reshape(-1)[0])))
    ph = ported._as_ps_ifg_complex(ph, n_ps, "ph1.ph")
    no_master = np.arange(ph.shape[1]) != (master_ix - 1)
    ph_nm = ph[:, no_master].astype(np.complex64)
    amp = np.abs(ph_nm).astype(np.float32)
    amp[amp == 0] = 1.0
    ph_nm = np.divide(ph_nm, amp, out=np.zeros_like(ph_nm), where=amp != 0).astype(np.complex64)

    rows = np.asarray([31777, 16149, 9857, 5900, 30271], dtype=np.int64)
    ph_patch = np.asarray(pm["ph_patch"], dtype=np.complex64)
    psdph = (np.conjugate(ph_patch[rows, :]).astype(np.complex64) * ph_nm[rows, :]).astype(np.complex128)
    bperp_mat = np.asarray(bp["bperp_mat"], dtype=np.float64)
    if bperp_mat.shape[1] == ph.shape[1]:
        bperp_mat = bperp_mat[:, no_master]
    n_trial_wraps = float(np.asarray(pm["n_trial_wraps"]).reshape(-1)[0])
    expected = ported._ps_topofit_batch(
        psdph,
        bperp_mat[rows, :],
        n_trial_wraps,
        kernel_backend="python",
        native_threads=0,
    )
    observed = run_stage2_topofit_kernel(
        psdph,
        bperp_mat[rows, :],
        n_trial_wraps,
        backend="native",
    )

    np.testing.assert_allclose(observed[0], expected[0], atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected[2], atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(observed[3], expected[3], atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_topofit_keeps_interior_peak_when_endpoint_refines_higher() -> None:
    patch_dir = Path("inputs_and_outputs/InSAR_dataset_test/PATCH_1")
    if not (patch_dir / "pm1.mat").exists():
        pytest.skip("HF parity fixture not present")

    pm = read_mat(patch_dir / "pm1.mat")
    ps = read_mat(patch_dir / "ps1.mat")
    ph = read_mat(patch_dir / "ph1.mat")["ph"]
    bp = read_mat(patch_dir / "bp1.mat")
    n_ps = int(round(float(np.asarray(ps["n_ps"]).reshape(-1)[0])))
    master_ix = int(round(float(np.asarray(ps["master_ix"]).reshape(-1)[0])))
    ph = ported._as_ps_ifg_complex(ph, n_ps, "ph1.ph")
    no_master = np.arange(ph.shape[1]) != (master_ix - 1)
    ph_nm = ph[:, no_master].astype(np.complex64)
    amp = np.abs(ph_nm).astype(np.float32)
    amp[amp == 0] = 1.0
    ph_nm = np.divide(ph_nm, amp, out=np.zeros_like(ph_nm), where=amp != 0).astype(np.complex64)

    rows = np.asarray([40316, 44969], dtype=np.int64)
    ph_patch = np.asarray(pm["ph_patch"], dtype=np.complex64)
    psdph = (np.conjugate(ph_patch[rows, :]).astype(np.complex64) * ph_nm[rows, :]).astype(np.complex128)
    bperp_mat = np.asarray(bp["bperp_mat"], dtype=np.float64)
    if bperp_mat.shape[1] == ph.shape[1]:
        bperp_mat = bperp_mat[:, no_master]
    n_trial_wraps = float(np.asarray(pm["n_trial_wraps"]).reshape(-1)[0])
    expected_k = np.asarray(pm["K_ps"], dtype=np.float64).reshape(-1)[rows]
    expected_c = np.asarray(pm["C_ps"], dtype=np.float64).reshape(-1)[rows]
    expected_coh = np.asarray(pm["coh_ps"], dtype=np.float64).reshape(-1)[rows]

    observed = run_stage2_topofit_kernel(
        psdph,
        bperp_mat[rows, :],
        n_trial_wraps,
        backend="native",
    )

    np.testing.assert_allclose(observed[0], expected_k, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected_c, atol=5e-4, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected_coh, atol=5e-4, rtol=0.0)
