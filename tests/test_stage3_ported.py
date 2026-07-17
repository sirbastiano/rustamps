from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pystamps.io.mat import read_mat, write_mat
from pystamps.pipeline import ported


def test_stage3_reestimate_writes_reestimated_threshold_coeffs(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.asarray([0.2], dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.6], dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1], dtype=np.float64),
            "C_ps": np.asarray([0.0], dtype=np.float64),
            "ph_grid": np.zeros((2, 2, 2), dtype=np.complex64),
            "grid_ij": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "n_trial_wraps": np.asarray(1.0, dtype=np.float64),
            "low_pass": np.ones((1, 1), dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ph1.mat",
        {"ph": np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)},
    )
    write_mat(
        patch_dir / "bp1.mat",
        {"bperp_mat": np.asarray([[10.0, 20.0]], dtype=np.float64)},
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms, **kwargs: np.asarray([1.0, 2.0], dtype=np.float64))

    real_as_ps_dim = ported._as_ps_dim
    real_as_ps_ifg_complex = ported._as_ps_ifg_complex
    real_as_ps_matrix = ported._as_ps_matrix

    def fake_as_ps_dim(values, n_ps, n_dim, name):
        if name == "ps1.xy":
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        if name == "pm1.grid_ij":
            return np.asarray([[1.0, 1.0]], dtype=np.float64)
        return real_as_ps_dim(values, n_ps, n_dim, name)

    monkeypatch.setattr(ported, "_as_ps_dim", fake_as_ps_dim)

    def fake_as_ps_ifg_complex(values, n_ps, name):
        if name == "pm1.ph_patch":
            return np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64)
        if name == "ph1.ph":
            return np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
        return real_as_ps_ifg_complex(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_ifg_complex", fake_as_ps_ifg_complex)

    def fake_as_ps_matrix(values, n_ps, name):
        if name == "pm1.ph_res":
            return np.zeros((1, 2), dtype=np.float32)
        if name == "bp1.bperp_mat":
            return np.asarray([[10.0, 20.0]], dtype=np.float64)
        return real_as_ps_matrix(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_matrix", fake_as_ps_matrix)

    clap_calls: list[tuple[tuple[int, ...], str]] = []

    def fake_clap_stack_kernel(
        ph_stack: np.ndarray,
        *,
        alpha: float,
        beta: float,
        low_pass: np.ndarray,
        backend: str = "auto",
        threads: int = 0,
    ) -> np.ndarray:
        del alpha, beta, low_pass, threads
        vals = np.asarray([0.5 + 0.0j, 0.25 + 0.0j], dtype=np.complex128)
        out = np.zeros_like(ph_stack, dtype=np.complex128)
        out[0, 0, :] = vals
        clap_calls.append((ph_stack.shape, backend))
        return out

    monkeypatch.setattr(ported, "run_stage3_clap_filt_patch_stack_kernel", fake_clap_stack_kernel, raising=False)
    monkeypatch.setattr(
        ported,
        "run_stage3_clap_filt_patch_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("single-plane clap patch should not be used")),
    )
    monkeypatch.setattr(
        ported,
        "_clap_filt_patch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python clap patch should not be used")),
    )

    coeff_calls = {"count": 0}
    initial_coeffs = np.asarray([9.0, 8.0], dtype=np.float64)
    reestimated_coeffs = np.asarray([1.5, -0.2], dtype=np.float64)

    def fake_threshold(*args, **kwargs):
        coeff_calls["count"] += 1
        coeffs = initial_coeffs if coeff_calls["count"] == 1 else reestimated_coeffs
        return np.zeros(1, dtype=np.float64), coeffs

    monkeypatch.setattr(ported, "run_stage3_coh_threshold_kernel", fake_threshold)

    topofit_calls: list[tuple[np.ndarray, np.ndarray, float, str]] = []

    def fake_topofit_kernel(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "auto",
        threads: int = 0,
        cpu_fallback=None,
    ):
        topofit_calls.append((cpxphase.copy(), bperp.copy(), float(n_trial_wraps), backend))
        return (
            np.asarray([0.1], dtype=np.float64),
            np.asarray([0.2], dtype=np.float64),
            np.asarray([0.9], dtype=np.float64),
            np.ones((1, 2), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_kernel", fake_topofit_kernel)
    monkeypatch.setattr(
        ported,
        "_ps_topofit_single",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scalar topofit should not be used")),
    )

    result = ported.stage3_select_ps(patch_dir, backend="native")

    assert result == "Stage 3 selected 1 PS"
    assert len(topofit_calls) == 1
    cpxphase, bperp, n_trial_wraps, backend = topofit_calls[0]
    assert cpxphase.shape == (1, 2)
    assert bperp.shape == (1, 2)
    assert n_trial_wraps == 1.0
    assert backend == "native"
    assert clap_calls == [((1, 1, 2), "native")]
    payload = read_mat(patch_dir / "select1.mat")
    np.testing.assert_allclose(np.asarray(payload["K_ps2"], dtype=np.float64).reshape(-1), [0.1])
    np.testing.assert_allclose(np.asarray(payload["C_ps2"], dtype=np.float64).reshape(-1), [0.2])
    np.testing.assert_allclose(np.asarray(payload["coh_ps2"], dtype=np.float64).reshape(-1), [0.9])
    np.testing.assert_allclose(
        np.asarray(payload["coh_thresh_coeffs"], dtype=np.float64).reshape(-1),
        reestimated_coeffs,
        atol=0.0,
        rtol=0.0,
    )


def test_stage3_reestimate_skips_topofit_when_any_ifg_is_zero(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.asarray([0.2], dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.6], dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.asarray([[0.5 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1], dtype=np.float64),
            "C_ps": np.asarray([0.0], dtype=np.float64),
            "ph_grid": np.zeros((2, 2, 2), dtype=np.complex64),
            "grid_ij": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "n_trial_wraps": np.asarray(1.0, dtype=np.float64),
            "low_pass": np.ones((1, 1), dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ph1.mat",
        {"ph": np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)},
    )
    write_mat(
        patch_dir / "bp1.mat",
        {"bperp_mat": np.asarray([[10.0, 20.0]], dtype=np.float64)},
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms, **kwargs: np.asarray([1.0, 2.0], dtype=np.float64))

    real_as_ps_dim = ported._as_ps_dim
    real_as_ps_ifg_complex = ported._as_ps_ifg_complex
    real_as_ps_matrix = ported._as_ps_matrix

    def fake_as_ps_dim(values, n_ps, n_dim, name):
        if name == "ps1.xy":
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        if name == "pm1.grid_ij":
            return np.asarray([[1.0, 1.0]], dtype=np.float64)
        return real_as_ps_dim(values, n_ps, n_dim, name)

    monkeypatch.setattr(ported, "_as_ps_dim", fake_as_ps_dim)

    def fake_as_ps_ifg_complex(values, n_ps, name):
        if name == "pm1.ph_patch":
            return np.asarray([[0.5 + 0.0j, 0.0 + 0.0j]], dtype=np.complex64)
        if name == "ph1.ph":
            return np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
        return real_as_ps_ifg_complex(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_ifg_complex", fake_as_ps_ifg_complex)

    def fake_as_ps_matrix(values, n_ps, name):
        if name == "pm1.ph_res":
            return np.zeros((1, 2), dtype=np.float32)
        if name == "bp1.bperp_mat":
            return np.asarray([[10.0, 20.0]], dtype=np.float64)
        return real_as_ps_matrix(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_matrix", fake_as_ps_matrix)

    clap_calls = {"count": 0}

    def fake_clap_stack_kernel(
        ph_stack: np.ndarray,
        *,
        alpha: float,
        beta: float,
        low_pass: np.ndarray,
        backend: str = "auto",
        threads: int = 0,
    ) -> np.ndarray:
        del alpha, beta, low_pass, backend, threads
        vals = np.asarray([0.5 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        out = np.zeros_like(ph_stack, dtype=np.complex128)
        out[0, 0, :] = vals
        clap_calls["count"] += 1
        return out

    monkeypatch.setattr(ported, "run_stage3_clap_filt_patch_stack_kernel", fake_clap_stack_kernel)
    monkeypatch.setattr(
        ported,
        "run_stage3_coh_threshold_kernel",
        lambda *args, **kwargs: (np.zeros(1, dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)),
    )

    called = {"count": 0}

    def fake_topofit(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        called["count"] += 1
        return 0.15, 0.25, 0.85, np.ones(2, dtype=np.complex64)

    monkeypatch.setattr(ported, "_ps_topofit_single", fake_topofit)

    result = ported.stage3_select_ps(patch_dir)

    assert result == "Stage 3 selected 1 PS"
    assert called["count"] == 0
    payload = read_mat(patch_dir / "select1.mat")
    assert np.isnan(np.asarray(payload["K_ps2"], dtype=np.float64).reshape(-1)[0])
    assert np.isnan(np.asarray(payload["coh_ps2"], dtype=np.float64).reshape(-1)[0])


def test_stage3_reestimate_keep_ix_uses_strict_source_threshold(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0, dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0, 20.0], dtype=np.float64),
            "xy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.asarray([0.2], dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.6], dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1], dtype=np.float64),
            "C_ps": np.asarray([0.0], dtype=np.float64),
            "ph_grid": np.zeros((2, 2, 2), dtype=np.complex64),
            "grid_ij": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "n_trial_wraps": np.asarray(1.0, dtype=np.float64),
            "low_pass": np.ones((1, 1), dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ph1.mat",
        {"ph": np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)},
    )
    write_mat(
        patch_dir / "bp1.mat",
        {"bperp_mat": np.asarray([[10.0, 20.0]], dtype=np.float64)},
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms, **kwargs: np.asarray([1.0, 2.0], dtype=np.float64))

    real_as_ps_dim = ported._as_ps_dim
    real_as_ps_ifg_complex = ported._as_ps_ifg_complex
    real_as_ps_matrix = ported._as_ps_matrix

    def fake_as_ps_dim(values, n_ps, n_dim, name):
        if name == "ps1.xy":
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        if name == "pm1.grid_ij":
            return np.asarray([[1.0, 1.0]], dtype=np.float64)
        return real_as_ps_dim(values, n_ps, n_dim, name)

    monkeypatch.setattr(ported, "_as_ps_dim", fake_as_ps_dim)

    def fake_as_ps_ifg_complex(values, n_ps, name):
        if name == "pm1.ph_patch":
            return np.asarray([[0.5 + 0.0j, 0.25 + 0.0j]], dtype=np.complex64)
        if name == "ph1.ph":
            return np.asarray([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]], dtype=np.complex64)
        return real_as_ps_ifg_complex(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_ifg_complex", fake_as_ps_ifg_complex)

    def fake_as_ps_matrix(values, n_ps, name):
        if name == "pm1.ph_res":
            return np.zeros((1, 2), dtype=np.float32)
        if name == "bp1.bperp_mat":
            return np.asarray([[10.0, 20.0]], dtype=np.float64)
        return real_as_ps_matrix(values, n_ps, name)

    monkeypatch.setattr(ported, "_as_ps_matrix", fake_as_ps_matrix)

    clap_calls = {"count": 0}

    def fake_clap_stack_kernel(
        ph_stack: np.ndarray,
        *,
        alpha: float,
        beta: float,
        low_pass: np.ndarray,
        backend: str = "auto",
        threads: int = 0,
    ) -> np.ndarray:
        del alpha, beta, low_pass, backend, threads
        vals = np.asarray([0.5 + 0.0j, 0.25 + 0.0j], dtype=np.complex128)
        out = np.zeros_like(ph_stack, dtype=np.complex128)
        out[0, 0, :] = vals
        clap_calls["count"] += 1
        return out

    monkeypatch.setattr(ported, "run_stage3_clap_filt_patch_stack_kernel", fake_clap_stack_kernel)
    monkeypatch.setattr(
        ported,
        "run_stage3_coh_threshold_kernel",
        lambda *args, **kwargs: (np.asarray([0.5], dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)),
    )

    def fake_topofit_kernel(
        cpxphase: np.ndarray,
        bperp: np.ndarray,
        n_trial_wraps: float,
        *,
        backend: str = "auto",
        threads: int = 0,
        cpu_fallback=None,
    ):
        return (
            np.asarray([0.1], dtype=np.float64),
            np.asarray([0.2], dtype=np.float64),
            np.asarray([0.5000005], dtype=np.float64),
            np.ones((1, 2), dtype=np.complex64),
        )

    monkeypatch.setattr(ported, "run_stage2_topofit_kernel", fake_topofit_kernel)

    result = ported.stage3_select_ps(patch_dir)

    assert result == "Stage 3 selected 1 PS"
    payload = read_mat(patch_dir / "select1.mat")
    assert np.asarray(payload["keep_ix"]).reshape(-1)[0]


def test_stage3_oversample_zeroing_matches_matlab_first_plane_semantics() -> None:
    ph_bit = np.stack(
        (
            np.full((4, 4), 10.0 + 0.0j, dtype=np.complex64),
            np.full((4, 4), 20.0 + 0.0j, dtype=np.complex64),
        ),
        axis=2,
    )
    ps_bit_i = 2
    ps_bit_j = 2
    slc_osf = 2

    ph_bit[ps_bit_i - 1, ps_bit_j - 1, :] = 0
    rad = slc_osf - 1
    ii = np.arange(ps_bit_i - rad, ps_bit_i + rad + 1, dtype=np.int64)
    jj = np.arange(ps_bit_j - rad, ps_bit_j + rad + 1, dtype=np.int64)
    ii = ii[(ii > 0) & (ii <= ph_bit.shape[0])] - 1
    jj = jj[(jj > 0) & (jj <= ph_bit.shape[1])] - 1
    ph_bit[np.ix_(ii, jj, np.asarray([0], dtype=np.int64))] = 0

    np.testing.assert_allclose(ph_bit[:3, :3, 0], 0.0, atol=0.0, rtol=0.0)
    assert ph_bit[1, 1, 1] == 0.0
    assert ph_bit[0, 0, 1] == 20.0 + 0.0j


def test_clap_filt_patch_stack_returns_complex128_workspace() -> None:
    ph = np.ones((4, 4, 2), dtype=np.complex64)
    low_pass = np.ones((4, 4), dtype=np.float64)

    out = ported._clap_filt_patch_stack(ph, alpha=1.0, beta=0.3, low_pass=low_pass)

    assert out.dtype == np.complex128


def test_clap_filt_grid_stack_prepared_uses_dispatcher_out_buffer(monkeypatch) -> None:
    ph = np.ones((5, 5, 2), dtype=np.complex64)
    low_pass = np.ones((4, 4), dtype=np.float64)
    prepared = ported._prepare_clap_filt_grid_stack(ph.shape, n_win=4, n_pad=0, low_pass=low_pass)
    out = np.empty_like(ph)
    calls: list[tuple[tuple[int, ...], int, int, bool]] = []

    def fake_stack_kernel(
        ph_stack: np.ndarray,
        *,
        alpha: float,
        beta: float,
        n_win: int,
        n_pad: int,
        low_pass: np.ndarray,
        preserve_precision: bool,
        backend: str,
    ) -> np.ndarray:
        del alpha, beta, low_pass, backend
        calls.append((ph_stack.shape, n_win, n_pad, preserve_precision))
        result = np.empty(ph_stack.shape, dtype=np.complex64)
        result[:, :, 0] = 2.0 + 0.0j
        result[:, :, 1] = 3.0 + 0.0j
        return result

    monkeypatch.setattr(ported, "run_stage3_clap_filt_grid_stack_kernel", fake_stack_kernel)

    observed = ported._clap_filt_grid_stack_prepared(
        ph,
        alpha=1.0,
        beta=0.3,
        prepared=prepared,
        out=out,
        workers=1,
        preserve_precision=False,
    )

    assert observed is out
    assert calls == [((5, 5, 2), 4, 0, False)]
    np.testing.assert_allclose(out[:, :, 0], 2.0 + 0.0j)
    np.testing.assert_allclose(out[:, :, 1], 3.0 + 0.0j)


def test_wrap_filt_uses_dispatcher_default_padding(monkeypatch) -> None:
    ph = np.ones((4, 4), dtype=np.complex64)
    expected = np.full((4, 4), 2.0 + 0.0j, dtype=np.complex64)
    expected_low = np.full((4, 4), 3.0 + 0.0j, dtype=np.complex64)
    calls: list[tuple[int, float, int, str]] = []

    def fake_wrap_kernel(
        ph_grid: np.ndarray,
        *,
        n_win: int,
        alpha: float,
        n_pad: int,
        low_flag: str,
        backend: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        del ph_grid, backend
        calls.append((n_win, alpha, n_pad, low_flag))
        return expected, expected_low

    monkeypatch.setattr(ported, "run_stage3_wrap_filt_kernel", fake_wrap_kernel)

    observed, observed_low = ported._wrap_filt(ph, n_win=4, alpha=1.25, n_pad=None, low_flag="y")

    assert calls == [(4, 1.25, 1, "y")]
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(observed_low, expected_low)


def test_wrap_filt_global_uses_dispatcher_default_padding(monkeypatch) -> None:
    ph = np.ones((4, 4), dtype=np.complex64)
    expected = np.full((4, 4), 4.0 + 0.0j, dtype=np.complex64)
    calls: list[tuple[int, float, int, str]] = []

    def fake_wrap_global_kernel(
        ph_grid: np.ndarray,
        *,
        n_win: int,
        alpha: float,
        n_pad: int,
        low_flag: str,
        backend: str,
    ) -> tuple[np.ndarray, None]:
        del ph_grid, backend
        calls.append((n_win, alpha, n_pad, low_flag))
        return expected, None

    monkeypatch.setattr(ported, "run_stage3_wrap_filt_global_kernel", fake_wrap_global_kernel)

    observed, observed_low = ported._wrap_filt_global(ph, n_win=4, alpha=1.1, n_pad=None, low_flag="n")

    assert calls == [(4, 1.1, 1, "n")]
    np.testing.assert_array_equal(observed, expected)
    assert observed_low is None


@pytest.mark.parametrize(("reestimate_flag", "should_fail"), [("y", True), ("n", False)])
def test_stage3_missing_reestimate_inputs_fail_only_when_requested(
    tmp_path: Path,
    monkeypatch,
    reestimate_flag: str,
    should_fail: bool,
) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("PERCENT"),
            "percent_rand": np.asarray(0.0),
            "quick_est_gamma_flag": ported._matlab_char_row("y"),
            "select_reest_gamma_flag": ported._matlab_char_row(reestimate_flag),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(1.0),
            "n_ifg": np.asarray(3.0),
            "master_ix": np.asarray(1.0),
            "bperp": np.asarray([0.0, 10.0, 20.0]),
            "xy": np.asarray([[1.0, 0.0, 0.0]]),
        },
    )
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.asarray([0.9]),
            "coh_bins": np.arange(0.005, 1.0, 0.01),
            "Nr": np.ones(100),
            "ph_patch": np.ones((1, 2), dtype=np.complex64),
            "ph_res": np.zeros((1, 2), dtype=np.float32),
            "K_ps": np.asarray([0.1]),
            "C_ps": np.asarray([0.0]),
        },
    )
    monkeypatch.setattr(
        ported,
        "run_stage3_coh_threshold_kernel",
        lambda *args, **kwargs: (np.zeros(1), np.asarray([1.0, 0.0])),
    )
    monkeypatch.setattr(
        ported,
        "_ifg_index_for_selection",
        lambda *args, **kwargs: np.asarray([1.0, 2.0]),
    )
    monkeypatch.setattr(
        ported,
        "_as_ps_ifg_complex",
        lambda values, n_ps, name: np.asarray(values, dtype=np.complex64).reshape(n_ps, -1),
    )
    monkeypatch.setattr(
        ported,
        "_as_ps_matrix",
        lambda values, n_ps, name: np.asarray(values, dtype=np.float32).reshape(n_ps, -1),
    )

    if should_fail:
        with pytest.raises(ported.PortedStageError, match="bp1.mat is missing"):
            ported.stage3_select_ps(patch_dir)
        assert not (patch_dir / "select1.mat").exists()
    else:
        assert ported.stage3_select_ps(patch_dir) == "Stage 3 selected 1 PS"
        payload = read_mat(patch_dir / "select1.mat")
        np.testing.assert_allclose(np.asarray(payload["K_ps2"]).reshape(-1), [0.1])


def test_stage3_density_threshold_uses_matlab_da_bin_edges(tmp_path: Path, monkeypatch) -> None:
    patch_dir = tmp_path / "PATCH_1"
    patch_dir.mkdir()

    n_ps = 50000
    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    write_mat(
        patch_dir / "parms.mat",
        {
            "select_method": ported._matlab_char_row("DENSITY"),
            "density_rand": np.asarray(1.0, dtype=np.float64),
            "small_baseline_flag": ported._matlab_char_row("n"),
            "gamma_stdev_reject": np.asarray(0.0, dtype=np.float64),
            "clap_win": np.asarray(1.0, dtype=np.float64),
            "clap_alpha": np.asarray(1.0, dtype=np.float64),
            "clap_beta": np.asarray(0.3, dtype=np.float64),
            "slc_osf": np.asarray(1.0, dtype=np.float64),
        },
    )
    write_mat(
        patch_dir / "ps1.mat",
        {
            "n_ps": np.asarray(float(n_ps), dtype=np.float64),
            "master_ix": np.asarray(1.0, dtype=np.float64),
            "bperp": np.asarray([0.0, 10.0], dtype=np.float64),
            "xy": np.zeros((n_ps, 3), dtype=np.float64),
        },
    )
    write_mat(patch_dir / "da1.mat", {"D_A": np.arange(1, n_ps + 1, dtype=np.float64)})
    write_mat(
        patch_dir / "pm1.mat",
        {
            "coh_ps": np.zeros(n_ps, dtype=np.float64),
            "coh_bins": coh_bins,
            "Nr": np.ones(coh_bins.size, dtype=np.float64),
            "ph_patch": np.zeros((n_ps, 1), dtype=np.complex64),
            "ph_res": np.zeros((n_ps, 1), dtype=np.float32),
            "K_ps": np.zeros(n_ps, dtype=np.float64),
            "C_ps": np.zeros(n_ps, dtype=np.float64),
        },
    )

    monkeypatch.setattr(ported, "_ifg_index_for_selection", lambda ps, parms, **kwargs: np.asarray([1.0], dtype=np.float64))
    monkeypatch.setattr(
        ported,
        "_as_ps_ifg_complex",
        lambda values, n_ps_arg, name: np.zeros((n_ps_arg, 1), dtype=np.complex64),
    )
    monkeypatch.setattr(
        ported,
        "_as_ps_matrix",
        lambda values, n_ps_arg, name: np.zeros((n_ps_arg, 1), dtype=np.float32),
    )

    captured: dict[str, np.ndarray | str] = {}

    def fake_threshold(*args, backend: str = "python", **kwargs):
        captured["D_A_max"] = np.asarray(args[2], dtype=np.float64).copy()
        captured["histogram_backend"] = backend
        return np.ones(n_ps, dtype=np.float64), np.asarray([1.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(ported, "run_stage3_coh_threshold_kernel", fake_threshold)

    result = ported.stage3_select_ps(patch_dir, backend="native")

    assert result == "Stage 3 selected 0 PS"
    np.testing.assert_array_equal(
        captured["D_A_max"],
        np.asarray([0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0], dtype=np.float64),
    )
    assert captured["histogram_backend"] == "native"


def test_stage3_threshold_histogram_uses_requested_kernel_backend(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray, str]] = []

    def fake_histogram(values: np.ndarray, centers: np.ndarray, *, backend: str = "auto") -> np.ndarray:
        calls.append((np.asarray(values).copy(), np.asarray(centers).copy(), backend))
        return np.asarray([0.0, 0.0, 10.0, 1.0], dtype=np.float64)

    monkeypatch.setattr(ported, "run_stage2_histogram_kernel", fake_histogram)

    ported._coh_threshold_from_dist(
        coh_values=np.asarray([0.2, 0.4, 0.6], dtype=np.float64),
        D_A=np.asarray([0.2, 0.3, 0.4], dtype=np.float64),
        D_A_max=np.asarray([0.0, 1.0], dtype=np.float64),
        coh_bins=np.asarray([0.1, 0.3, 0.5, 0.7], dtype=np.float64),
        Nr_dist=np.ones(4, dtype=np.float64),
        low_coh_thresh=2,
        max_percent_rand=1.0,
        select_method="DENSITY",
        histogram_backend="native",
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], np.asarray([0.2, 0.4, 0.6], dtype=np.float64))
    np.testing.assert_allclose(calls[0][1], np.asarray([0.1, 0.3, 0.5, 0.7], dtype=np.float64))
    assert calls[0][2] == "native"


def test_stage3_saved_patch1_row_matches_oracle_residual_angles() -> None:
    patch_dir = Path("inputs_and_outputs/InSAR_dataset_test/PATCH_1")

    sel = read_mat(patch_dir / "select1.mat")
    ps = read_mat(patch_dir / "ps1.mat")
    ph1 = read_mat(patch_dir / "ph1.mat")
    bp1 = read_mat(patch_dir / "bp1.mat")
    pm1 = read_mat(patch_dir / "pm1.mat")

    row = 56350
    ix = np.asarray(sel["ix"], dtype=np.float64).reshape(-1).astype(np.int64)
    ps_idx = int(ix[row] - 1)

    ph_all = np.asarray(ph1["ph"], dtype=np.complex128)
    master_ix = int(round(float(np.asarray(ps["master_ix"], dtype=np.float64).reshape(-1)[0])))
    ph_work = ph_all[:, np.arange(ph_all.shape[1]) != (master_ix - 1)]
    ph_patch2 = np.asarray(sel["ph_patch2"], dtype=np.complex128)
    bperp_mat = np.asarray(bp1["bperp_mat"], dtype=np.float64)
    ifg_index_ix = np.asarray(sel["ifg_index"], dtype=np.float64).reshape(-1).astype(np.int64) - 1

    psdph = ph_work[ps_idx, :] * np.conj(ph_patch2[row, :])
    psdph = np.divide(psdph, np.abs(psdph), out=np.zeros_like(psdph), where=np.abs(psdph) != 0)
    _, _, _, phase_residual = ported._ps_topofit_single(
        psdph[ifg_index_ix].astype(np.complex64, copy=False),
        bperp_mat[ps_idx, :][ifg_index_ix],
        float(np.asarray(pm1["n_trial_wraps"], dtype=np.float64).reshape(-1)[0]),
    )

    observed = np.angle(phase_residual).astype(np.float32, copy=False)
    expected = np.asarray(sel["ph_res2"], dtype=np.float32)[row, ifg_index_ix]

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=5 * np.finfo(np.float32).eps)
