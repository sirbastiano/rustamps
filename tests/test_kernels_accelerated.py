import numpy as np
import pytest
import importlib.util
from scipy import signal

import pystamps.kernels.accelerated as accel
from pystamps.kernels import (
    BackendUnavailableError,
    describe_backend_matrix,
    run_stage4_edge_stats_kernel,
    run_stage2_clap_filter_kernel,
    run_stage2_grid_accumulate_kernel,
    run_stage2_grid_indices_kernel,
    run_stage2_histogram_kernel,
    run_stage2_normalize_complex_kernel,
    run_stage2_normalize_phase_matrix_kernel,
    run_stage2_ph_weight_block_kernel,
    run_stage2_topofit_coh_row_invariant_kernel,
    run_stage2_topofit_kernel,
    run_stage2_topofit_row_invariant_kernel,
    run_stage3_coh_threshold_kernel,
    run_stage3_clap_filt_grid_kernel,
    run_stage3_clap_filt_grid_stack_kernel,
    run_stage3_clap_filt_patch_kernel,
    run_stage3_select_ifg_index_kernel,
    run_stage3_wrap_filt_kernel,
    run_stage3_wrap_filt_global_kernel,
    run_stage4_duplicate_keep_kernel,
    run_stage4_adjacent_component_keep_kernel,
    run_stage4_phase_correction_kernel,
    run_stage5_format_merged_rc2_kernel,
    run_stage5_duplicate_keep_kernel,
    run_stage5_ifg_std_kernel,
    run_stage5_patch_keep_mask_kernel,
    run_stage5_rc2_correction_kernel,
    run_stage4_weed_ifg_index_kernel,
    run_stage6_extract_grid_values_kernel,
    run_stage6_grid_accumulate_kernel,
    run_stage6_prepare_cost_offsets_kernel,
    run_stage6_reconstruct_ps_phase_kernel,
    run_stage6_ps_grid_indices_kernel,
    run_stage6_select_ifgw_kernel,
    run_stage6_single_master_ifg_geometry_kernel,
    run_stage6_unwrap_ifg_sets_kernel,
    run_stage6_unwrap_grid_kernel,
    run_stage7_center_to_reference_kernel,
    run_stage7_scla_kernel,
    run_stage7_scla_smooth_kernel,
    run_stage8_edge_noise_kernel,
    run_stage8_weighted_lstsq_kernel,
)
from pystamps.kernels.registry import KernelRegistry
from pystamps.pipeline import ported


def _reference_clap_filt_patch(ph: np.ndarray, alpha: float, beta: float, low_pass: np.ndarray) -> np.ndarray:
    ph_arr = np.asarray(ph, dtype=np.complex128).copy()
    ph_arr[np.isnan(ph_arr)] = 0
    ph_fft = np.fft.fft2(ph_arr)
    h = np.abs(ph_fft)
    b = ported._clap_filter_kernel()
    h = np.fft.ifftshift(signal.convolve2d(np.fft.fftshift(h), b, mode="same", boundary="fill", fillvalue=0.0))
    mean_h = float(np.median(h))
    if mean_h != 0.0:
        h = h / mean_h
    h = np.power(h, float(alpha))
    h = h - 1.0
    h[h < 0.0] = 0.0
    g = h * float(beta) + np.asarray(low_pass, dtype=np.float64)
    return np.fft.ifft2(ph_fft * g)


def _reference_clap_filt_grid(
    ph: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int,
    low_pass: np.ndarray,
    preserve_precision: bool = False,
) -> np.ndarray:
    ph_arr = np.asarray(ph, dtype=np.complex128 if preserve_precision else np.complex64).copy()
    ph_arr[np.isnan(ph_arr)] = 0
    n_i, n_j = ph_arr.shape
    out = np.zeros((n_i, n_j), dtype=np.complex128)
    n_inc = max(1, int(n_win) // 4)
    n_win_i = int(np.ceil(n_i / float(n_inc)) - 3)
    n_win_j = int(np.ceil(n_j / float(n_inc)) - 3)
    if n_win_i <= 0 or n_win_j <= 0:
        return out.astype(np.complex128 if preserve_precision else np.complex64, copy=False)

    x = np.arange(0, int(n_win) // 2, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    wind_func = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0) + 1e-6
    ph_bit = np.zeros((int(n_win) + int(n_pad), int(n_win) + int(n_pad)), dtype=np.complex128)

    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + int(n_win)
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - int(n_win)
            wf = np.vstack((np.zeros((i_shift, int(n_win)), dtype=np.float64), wf[: int(n_win) - i_shift, :]))
        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + int(n_win)
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - int(n_win)
                wf2 = np.hstack((np.zeros((int(n_win), j_shift), dtype=np.float64), wf2[:, : int(n_win) - j_shift]))
            ph_bit.fill(0)
            ph_bit[: int(n_win), : int(n_win)] = ph_arr[i1:i2, j1:j2]
            ph_filt = _reference_clap_filt_patch(ph_bit, alpha=alpha, beta=beta, low_pass=low_pass)
            out[i1:i2, j1:j2] += ph_filt[: int(n_win), : int(n_win)] * wf2
    return out.astype(np.complex128 if preserve_precision else np.complex64, copy=False)


def test_weighted_affine_fit_native_matches_python_reference() -> None:
    time_diff = np.asarray([-2.0, 0.0, 3.0, 7.0], dtype=np.float64)
    y = np.asarray(
        [
            [3.0, 4.0, 6.0, 9.0],
            [-2.0, 0.0, 1.0, 5.0],
        ],
        dtype=np.float64,
    )
    weight = np.asarray([1.0, 4.0, 2.0, 0.5], dtype=np.float64)

    expected = ported._weighted_affine_fit(time_diff, y, weight)
    observed = accel.run_weighted_affine_fit_kernel(time_diff, y, weight, backend="native")

    np.testing.assert_allclose(observed[0], expected[0], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-12, rtol=0.0)


def test_weighted_slope_fit_native_matches_python_reference() -> None:
    x = np.asarray([-1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    y_real = np.asarray(
        [
            [1.0, 3.0, 5.0, 9.0],
            [-2.0, 4.0, 8.0, 16.0],
        ],
        dtype=np.float64,
    )
    y_complex = (y_real + 1j * np.flip(y_real, axis=1)).astype(np.complex128)
    weight = np.asarray([1.0, np.inf, 0.0, 2.0], dtype=np.float64)

    expected_real = ported._weighted_slope_fit(x, y_real, weight)
    observed_real = accel.run_weighted_slope_fit_kernel(x, y_real, weight, backend="native")
    expected_complex = ported._weighted_slope_fit(x, y_complex, weight)
    observed_complex = accel.run_weighted_slope_fit_kernel(x, y_complex, weight, backend="native")

    np.testing.assert_allclose(observed_real, expected_real, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(observed_complex, expected_complex, atol=1e-12, rtol=0.0)


def _reference_clap_filt_grid_stack(
    ph_stack: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int,
    low_pass: np.ndarray,
    preserve_precision: bool = False,
) -> np.ndarray:
    ph_arr = np.asarray(ph_stack)
    out_dtype = np.complex128 if preserve_precision else np.complex64
    out = np.empty(ph_arr.shape, dtype=out_dtype)
    for i_ifg in range(ph_arr.shape[2]):
        out[:, :, i_ifg] = _reference_clap_filt_grid(
            ph_arr[:, :, i_ifg],
            alpha=alpha,
            beta=beta,
            n_win=n_win,
            n_pad=n_pad,
            low_pass=low_pass,
            preserve_precision=preserve_precision,
        )
    return out


def _reference_wrap_filt(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    ph_arr = np.asarray(ph, dtype=np.complex64).copy()
    ph_arr[np.isnan(ph_arr)] = 0
    n_i, n_j = ph_arr.shape
    n_inc = max(1, int(np.floor(int(n_win) / 2.0)))
    n_win_blocks_i = int(np.ceil(n_i / n_inc) - 1)
    n_win_blocks_j = int(np.ceil(n_j / n_inc) - 1)
    out = np.zeros_like(ph_arr, dtype=np.complex64)
    want_low = str(low_flag).lower() == "y"
    out_low = np.zeros_like(ph_arr, dtype=np.complex64) if want_low else None

    x = np.arange(1, int(n_win) // 2 + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.concatenate((X, np.fliplr(X)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0).astype(np.float64)
    b = np.outer(ported._gausswin(7), ported._gausswin(7))
    ph_bit = np.zeros((int(n_win) + int(n_pad), int(n_win) + int(n_pad)), dtype=np.complex64)
    low_filter = None
    if want_low:
        g = ported._gausswin(int(n_win) + int(n_pad), alpha=16.0)
        low_filter = np.fft.ifftshift(np.outer(g, g))

    for ix1 in range(n_win_blocks_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + int(n_win)
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - int(n_win)
            wf = np.vstack((np.zeros((i_shift, int(n_win)), dtype=np.float64), wf[: int(n_win) - i_shift, :]))
        for ix2 in range(n_win_blocks_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + int(n_win)
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - int(n_win)
                wf2 = np.hstack((np.zeros((int(n_win), j_shift), dtype=np.float64), wf2[:, : int(n_win) - j_shift]))
            ph_bit.fill(0)
            ph_bit[: int(n_win), : int(n_win)] = ph_arr[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            h = np.abs(ph_fft)
            h = np.fft.ifftshift(signal.convolve2d(np.fft.fftshift(h), b, mode="same", boundary="fill", fillvalue=0.0))
            mean_h = float(np.median(h))
            if mean_h != 0.0:
                h = h / mean_h
            h = np.power(h, float(alpha))
            ph_filt = np.fft.ifft2(ph_fft * h)[: int(n_win), : int(n_win)] * wf2
            out[i1:i2, j1:j2] += ph_filt.astype(np.complex64)
            if out_low is not None and low_filter is not None:
                ph_filt_low = np.fft.ifft2(ph_fft * low_filter)[: int(n_win), : int(n_win)] * wf2
                out_low[i1:i2, j1:j2] += ph_filt_low.astype(np.complex64)

    ph_mag = np.abs(ph_arr).astype(np.float32)
    out = (ph_mag * np.exp(1j * np.angle(out))).astype(np.complex64)
    if out_low is not None:
        out_low = (ph_mag * np.exp(1j * np.angle(out_low))).astype(np.complex64)
    return out, out_low


def _reference_wrap_filt_global(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    ph_arr = np.asarray(ph, dtype=np.complex64).copy()
    ph_arr[np.isnan(ph_arr)] = 0
    n_i, n_j = ph_arr.shape
    n_inc = max(1, int(n_win) // 2)
    n_win_count_i = max(1, int(np.ceil(n_i / n_inc) - 1))
    n_win_count_j = max(1, int(np.ceil(n_j / n_inc) - 1))
    out = np.zeros((n_i, n_j), dtype=np.complex64)
    want_low = str(low_flag).lower() == "y"
    out_low = np.zeros((n_i, n_j), dtype=np.complex64) if want_low else None

    half = int(n_win) // 2
    x = np.arange(1, half + 1, dtype=np.float32)
    X, Y = np.meshgrid(x, x)
    wind_func = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0).astype(np.float32)
    b = np.outer(ported._gausswin(7), ported._gausswin(7)).astype(np.float32)
    ph_bit = np.zeros((int(n_win) + int(n_pad), int(n_win) + int(n_pad)), dtype=np.complex64)
    low_filter = None
    if want_low:
        g = ported._gausswin(int(n_win) + int(n_pad), alpha=16.0)
        low_filter = np.fft.ifftshift(np.outer(g, g))

    for ix1 in range(n_win_count_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + int(n_win)
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - int(n_win)
            wf = np.vstack((np.zeros((i_shift, int(n_win)), dtype=np.float32), wf[: int(n_win) - i_shift, :]))
        for ix2 in range(n_win_count_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + int(n_win)
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - int(n_win)
                wf2 = np.hstack((np.zeros((int(n_win), j_shift), dtype=np.float32), wf2[:, : int(n_win) - j_shift]))
            ph_bit.fill(0)
            ph_bit[: int(n_win), : int(n_win)] = ph_arr[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            h = np.abs(ph_fft)
            h = np.fft.ifftshift(signal.convolve2d(np.fft.fftshift(h), b, mode="same", boundary="fill", fillvalue=0.0))
            mean_h = float(np.median(h))
            if mean_h != 0.0:
                h = h / mean_h
            h = np.power(h, float(alpha))
            ph_filt = np.fft.ifft2(ph_fft * h)[: int(n_win), : int(n_win)] * wf2
            out[i1:i2, j1:j2] += ph_filt
            if out_low is not None and low_filter is not None:
                ph_filt_low = np.fft.ifft2(ph_fft * low_filter)[: int(n_win), : int(n_win)] * wf2
                out_low[i1:i2, j1:j2] += ph_filt_low

    magnitude = np.abs(ph_arr)
    out = (magnitude * np.exp(1j * np.angle(out))).astype(np.complex64)
    if out_low is not None:
        out_low = (magnitude * np.exp(1j * np.angle(out_low))).astype(np.complex64)
    return out, out_low


def _install_fake_stage78_native_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[object, ...]], object]:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(accel, "cpu_budget", lambda: 6)

    class _FakeNative:
        def stage4_edge_stats(
            self,
            ph_weed: np.ndarray,
            node_a: np.ndarray,
            node_b: np.ndarray,
            bperp: np.ndarray,
            day: np.ndarray,
            time_win: float,
            small_baseline: bool,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            calls.append(("stage4", int(threads), bool(small_baseline), tuple(np.asarray(ph_weed).shape)))
            n_node, n_ifg = np.asarray(ph_weed).shape
            assert np.asarray(node_a).shape == np.asarray(node_b).shape
            assert np.asarray(bperp).shape == (n_ifg,)
            if not bool(small_baseline):
                assert np.asarray(day).shape == (n_ifg,)
            return {
                "ps_std": np.full(n_node, 3.0, dtype=np.float64),
                "ps_max": np.full(n_node, 4.0, dtype=np.float64),
            }

        def stage7_scla_parity(
            self,
            ph_proc: np.ndarray,
            ph_mean_v: np.ndarray,
            bperp_mat: np.ndarray,
            unwrap_ix: np.ndarray,
            solve_ix: np.ndarray,
            day: np.ndarray,
            master_ix: int,
            ifg_std: np.ndarray,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            calls.append(("stage7", int(threads), tuple(np.asarray(ph_proc).shape)))
            n_ps, n_ifg = np.asarray(ph_proc).shape
            assert np.asarray(ph_mean_v).shape == (n_ps, n_ifg)
            assert np.asarray(bperp_mat).shape[0] == n_ps
            assert np.asarray(unwrap_ix).ndim == 1
            assert np.asarray(solve_ix).ndim == 1
            assert np.asarray(day).shape == (n_ifg,)
            assert np.asarray(ifg_std).shape == (n_ifg,)
            assert int(master_ix) == 1
            return {
                "K_ps_uw": np.full(n_ps, 11.0, dtype=np.float64),
                "C_ps_uw": np.full(n_ps, 12.0, dtype=np.float32),
                "ph_scla": np.full((n_ps, n_ifg), 13.0, dtype=np.float32),
                "ph_ramp": np.full((n_ps, n_ifg), 14.0, dtype=np.float64),
                "ifg_vcm": np.eye(n_ifg, dtype=np.float64),
                "mean_v": np.full(n_ps, 15.0, dtype=np.float32),
                "m": np.full((2, n_ps), 16.0, dtype=np.float32),
            }

        def stage7_scla_smooth(
            self,
            k_ps_uw: np.ndarray,
            c_ps_uw: np.ndarray,
            edges: np.ndarray,
            threads: int = 0,
        ) -> tuple[np.ndarray, np.ndarray]:
            calls.append(("stage7_smooth", int(threads), tuple(np.asarray(edges).shape)))
            n_ps = np.asarray(k_ps_uw).size
            assert np.asarray(c_ps_uw).shape == (n_ps,)
            return (
                np.full(n_ps, 21.0, dtype=np.float32),
                np.full(n_ps, 22.0, dtype=np.float32),
            )

        def stage8_edge_noise(
            self,
            uw_ph: np.ndarray,
            node_a: np.ndarray,
            node_b: np.ndarray,
            chunk_edges: int = 0,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            calls.append(("stage8", int(chunk_edges), int(threads), tuple(np.asarray(uw_ph).shape)))
            n_edge = np.asarray(node_a).size
            n_ifg = np.asarray(uw_ph).shape[1]
            assert np.asarray(node_b).shape == np.asarray(node_a).shape
            return {
                "dph_noise": np.full((n_edge, n_ifg), -1.0, dtype=np.float32),
                "dph_space_uw": np.full((n_edge, n_ifg), 2.0, dtype=np.float32),
            }

    native_mod = _FakeNative()
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: native_mod)
    return calls, native_mod


def test_stage4_kernel_cpu_small_baseline_zero_noise() -> None:
    ph_weed = np.ones((3, 3), dtype=np.complex128)
    out = run_stage4_edge_stats_kernel(
        ph_weed,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([], dtype=np.float64),
        time_win=30.0,
        small_baseline=True,
        backend="python",
    )

    np.testing.assert_allclose(out["ps_std"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out["ps_max"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)


def test_stage4_kernel_cpu_non_small_baseline_zero_noise() -> None:
    ph_weed = np.ones((3, 3), dtype=np.complex128)
    out = run_stage4_edge_stats_kernel(
        ph_weed,
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        time_win=30.0,
        small_baseline=False,
        backend="python",
    )

    np.testing.assert_allclose(out["ps_std"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(out["ps_max"], np.zeros(3, dtype=np.float64), atol=0.0, rtol=0.0)


def test_stage7_kernel_cpu_shapes() -> None:
    ph_proc = np.asarray([[0.0, 0.2, 0.4], [0.0, -0.1, 0.3]], dtype=np.float64)
    ph_mean_v = ph_proc.copy()
    b = np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]], dtype=np.float64)
    unwrap_ix = np.asarray([0, 1, 2], dtype=np.int64)
    solve_ix = np.asarray([1, 2], dtype=np.int64)
    day = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    ifg_std = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    out = run_stage7_scla_kernel(
        ph_proc,
        ph_mean_v,
        b,
        unwrap_ix,
        solve_ix,
        day,
        master_ix=1,
        ifg_std=ifg_std,
        backend="cpu",
        chunk_ps=1,
    )

    assert out["K_ps_uw"].shape == (2,)
    assert out["C_ps_uw"].shape == (2,)
    assert out["ph_scla"].shape == (2, 3)
    assert out["ph_ramp"].shape == (2, 3)
    assert out["ifg_vcm"].shape == (3, 3)
    assert out["mean_v"].shape == (2,)
    assert out["m"].shape == (2, 2)


def test_stage7_kernel_uses_master_dropped_solve_set_for_c_mean() -> None:
    ph_proc = np.asarray(
        [
            [0.0, 10.0, 20.0, 30.0],
            [0.0, -3.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    b = np.zeros_like(ph_proc)
    out = run_stage7_scla_kernel(
        ph_proc,
        ph_proc,
        b,
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
        np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        master_ix=1,
        ifg_std=np.ones(4, dtype=np.float64),
        backend="cpu",
        chunk_ps=0,
    )

    np.testing.assert_allclose(out["C_ps_uw"], np.asarray([25.0, 1.5], dtype=np.float32), atol=0.0, rtol=0.0)


def test_stage7_kernel_coest_mean_vel_threshold_uses_unwrap_count() -> None:
    ph_proc = np.asarray([[0.0, 9.0, 11.0, 13.0, 15.0]], dtype=np.float64)
    b = np.zeros_like(ph_proc)
    out = run_stage7_scla_kernel(
        ph_proc,
        ph_proc,
        b,
        np.asarray([1, 2, 3, 4], dtype=np.int64),
        np.asarray([2, 3, 4], dtype=np.int64),
        np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
        master_ix=1,
        ifg_std=np.ones(5, dtype=np.float64),
        backend="cpu",
        chunk_ps=0,
    )

    np.testing.assert_allclose(out["C_ps_uw"], np.asarray([7.0], dtype=np.float32), atol=1e-6, rtol=0.0)


def test_stage8_kernel_cpu_shapes() -> None:
    uw_ph = np.asarray([[1 + 0j, 1 + 0j], [1j, -1j], [1 + 1j, 1 - 1j]], dtype=np.complex64)
    node_a = np.asarray([0, 1], dtype=np.int64)
    node_b = np.asarray([1, 2], dtype=np.int64)
    out = run_stage8_edge_noise_kernel(uw_ph, node_a, node_b, backend="cpu")

    assert out["dph_noise"].shape == (2, 2)
    assert out["dph_space_uw"].shape == (2, 2)


def test_stage8_kernel_uses_forward_edge_orientation() -> None:
    uw_ph = np.asarray(
        [
            [1 + 0j, 1j],
            [1j, 1 + 0j],
        ],
        dtype=np.complex64,
    )
    out = run_stage8_edge_noise_kernel(
        uw_ph,
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        backend="cpu",
    )

    expected = np.angle(uw_ph[[1], :] * np.conj(uw_ph[[0], :])).astype(np.float32)

    np.testing.assert_allclose(out["dph_space_uw"], expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        out["dph_noise"],
        (expected - np.mean(expected, axis=1, keepdims=True)) * np.float32(0.5),
        atol=0.0,
        rtol=0.0,
    )


def test_gpu_backend_requires_cupy() -> None:
    if importlib.util.find_spec("cupy") is not None:
        out = run_stage8_edge_noise_kernel(
            np.asarray([[1 + 0j]], dtype=np.complex64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            backend="gpu",
        )
        assert out["dph_noise"].shape == (1, 1)
        return

    with pytest.raises(BackendUnavailableError, match="CuPy"):
        run_stage8_edge_noise_kernel(
            np.asarray([[1 + 0j]], dtype=np.complex64),
            np.asarray([0], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            backend="gpu",
        )


def test_stage2_grid_accumulate_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)

    ph_weight = np.asarray(
        [
            [1 + 1j, 2 + 0j],
            [3 + 0j, 4 + 1j],
            [5 - 1j, 6 + 2j],
        ],
        dtype=np.complex64,
    )
    grid_lin = np.asarray([0, 2, 0], dtype=np.int64)

    observed = run_stage2_grid_accumulate_kernel(ph_weight, grid_lin, 3, 1, backend="auto")
    expected = accel._stage2_grid_accumulate_cpu(ph_weight, grid_lin, 3, 1)

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)


def test_stage2_topofit_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)
    calls: list[str] = []

    def fake_cpu(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        calls.append("cpu")
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.ones(n_row, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    out = run_stage2_topofit_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float64),
        1.0,
        backend="auto",
        threads=5,
        cpu_fallback=fake_cpu,
    )

    assert calls == ["cpu"]
    assert out[0].shape == (2,)


def test_stage2_row_invariant_topofit_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)
    calls: list[str] = []

    def fake_cpu(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        calls.append("cpu")
        assert bperp.shape == (2, 3)
        n_row, n_col = cpxphase.shape
        return (
            np.zeros(n_row, dtype=np.float64),
            np.zeros(n_row, dtype=np.float64),
            np.ones(n_row, dtype=np.float64),
            np.ones((n_row, n_col), dtype=np.complex64),
        )

    out = run_stage2_topofit_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="auto",
        threads=5,
        cpu_fallback=fake_cpu,
    )

    assert calls == ["cpu"]
    assert out[0].shape == (2,)


def test_stage2_row_invariant_coh_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)
    calls: list[str] = []

    def fake_cpu(cpxphase: np.ndarray, bperp: np.ndarray, n_trial_wraps: float):
        calls.append("cpu")
        assert bperp.shape == (2, 3)
        return np.full(cpxphase.shape[0], 0.75, dtype=np.float64)

    out = run_stage2_topofit_coh_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="auto",
        threads=5,
        cpu_fallback=fake_cpu,
    )

    assert calls == ["cpu"]
    np.testing.assert_allclose(out, np.full(2, 0.75, dtype=np.float64))


def test_stage2_histogram_kernel_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: None)

    values = np.asarray([0.1, 0.4, 0.49, 0.8, np.nan], dtype=np.float64)
    centers = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)

    observed = run_stage2_histogram_kernel(values, centers, backend="auto")
    expected = accel._stage2_histogram_with_centers_cpu(values, centers)

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(observed, np.asarray([1.0, 2.0, 1.0], dtype=np.float64), atol=0.0, rtol=0.0)


def test_explicit_stage5_native_backend_errors_when_native_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accel, "stage5_native_available", lambda: False)

    with pytest.raises(BackendUnavailableError, match="native"):
        run_stage5_ifg_std_kernel(
            np.ones((1, 2), dtype=np.complex64),
            np.ones((1, 2), dtype=np.complex64),
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            backend="native",
        )


def test_stage2_histogram_kernel_equal_spacing_matches_octave_rule() -> None:
    centers = np.asarray([0.005, 0.015, 0.025, 0.035], dtype=np.float64)
    values = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)
    observed = accel._stage2_histogram_with_centers_cpu(values, centers)
    np.testing.assert_allclose(observed, np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float64), atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_histogram_kernel_native_equal_spacing_matches_cpu() -> None:
    centers = np.asarray([0.005, 0.015, 0.025, 0.035], dtype=np.float64)
    values = np.asarray([0.01, 0.02, 0.03], dtype=np.float64)
    expected = accel._stage2_histogram_with_centers_cpu(values, centers)
    observed = run_stage2_histogram_kernel(values, centers, backend="native")
    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)


def test_describe_backend_matrix_reports_registered_coverage() -> None:
    matrix = describe_backend_matrix()

    assert "providers" in matrix
    assert "kernels" in matrix
    assert matrix["providers"]["python"]["available"] is True
    assert matrix["providers"]["cuda"]["aliases"] == ["gpu"]
    assert matrix["kernels"]["stage2_topofit"]["baseline_backend"] == "python"
    assert matrix["kernels"]["stage4_edge_stats"]["baseline_backend"] == "python"
    assert "native" in matrix["kernels"]["stage2_topofit"]["supported_backends"]
    assert matrix["kernels"]["stage7_scla"]["baseline_backend"] == "python"
    assert "cuda" in matrix["kernels"]["stage8_edge_noise"]["supported_backends"]


def test_coverage_manifest_clears_unavailable_reason_for_available_provider() -> None:
    registry = KernelRegistry()
    registry.register_provider(
        "native",
        description="Native test backend",
        availability_probe=lambda: True,
        unavailable_reason="missing native backend",
    )

    provider = registry.coverage_manifest()["providers"]["native"]

    assert provider["available"] is True
    assert provider["unavailable_reason"] == ""


def test_describe_backend_matrix_reports_stage7_stage8_native_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_stage78_native_backends(monkeypatch)

    matrix = describe_backend_matrix()

    assert "native" in matrix["kernels"]["stage4_edge_stats"]["supported_backends"]
    assert "native" in matrix["kernels"]["stage7_scla"]["supported_backends"]
    assert "native" in matrix["kernels"]["stage7_scla_smooth"]["supported_backends"]
    assert "native" in matrix["kernels"]["stage8_edge_noise"]["supported_backends"]
    assert "native" in matrix["kernels"]["stage4_edge_stats"]["available_backends"]
    assert "native" in matrix["kernels"]["stage7_scla"]["available_backends"]
    assert "native" in matrix["kernels"]["stage7_scla_smooth"]["available_backends"]
    assert "native" in matrix["kernels"]["stage8_edge_noise"]["available_backends"]


def test_stage4_stage7_stage8_native_wrappers_dispatch_to_fake_native_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _native_mod = _install_fake_stage78_native_backends(monkeypatch)

    stage4 = run_stage4_edge_stats_kernel(
        np.ones((3, 3), dtype=np.complex128),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        time_win=30.0,
        small_baseline=False,
        backend="native",
    )
    stage7 = run_stage7_scla_kernel(
        np.asarray([[0.0, 0.2, 0.4], [0.0, -0.1, 0.3]], dtype=np.float64),
        np.asarray([[0.0, 0.2, 0.4], [0.0, -0.1, 0.3]], dtype=np.float64),
        np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]], dtype=np.float64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
        master_ix=1,
        ifg_std=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        backend="native",
        chunk_ps=4,
    )
    stage8 = run_stage8_edge_noise_kernel(
        np.asarray([[1 + 0j, 1 + 0j], [1j, -1j], [1 + 1j, 1 - 1j]], dtype=np.complex64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 2], dtype=np.int64),
        backend="native",
        chunk_edges=3,
    )

    assert calls == [("stage4", 6, False, (3, 3)), ("stage7", 6, (2, 3)), ("stage8", 3, 6, (3, 2))]
    np.testing.assert_allclose(stage4["ps_std"], np.full(3, 3.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage4["ps_max"], np.full(3, 4.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage7["K_ps_uw"], np.full(2, 11.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage7["ph_ramp"], np.full((2, 3), 14.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage8["dph_noise"], np.full((2, 2), -1.0, dtype=np.float32), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(stage8["dph_space_uw"], np.full((2, 2), 2.0, dtype=np.float32), atol=0.0, rtol=0.0)


def test_native_loader_retries_after_initial_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    class _FakeNative:
        def stage7_scla_parity(
            self,
            ph_proc: np.ndarray,
            ph_mean_v: np.ndarray,
            bperp_mat: np.ndarray,
            unwrap_ix: np.ndarray,
            solve_ix: np.ndarray,
            day: np.ndarray,
            master_ix: int,
            ifg_std: np.ndarray,
            threads: int = 0,
        ) -> dict[str, np.ndarray]:
            n_ps, n_ifg = np.asarray(ph_proc).shape
            return {
                "K_ps_uw": np.zeros(n_ps, dtype=np.float64),
                "C_ps_uw": np.zeros(n_ps, dtype=np.float32),
                "ph_scla": np.zeros((n_ps, n_ifg), dtype=np.float32),
                "ph_ramp": np.zeros((n_ps, n_ifg), dtype=np.float64),
                "ifg_vcm": np.eye(n_ifg, dtype=np.float64),
                "mean_v": np.zeros(n_ps, dtype=np.float32),
                "m": np.zeros((2, n_ps), dtype=np.float32),
            }

    def fake_import_module(name: str) -> object:
        assert name == "pystamps.kernels._stage2_native"
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ImportError("native extension not built yet")
        return _FakeNative()

    monkeypatch.setattr(accel.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(accel.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(accel, "_STAGE2_NATIVE_MODULE", None)
    monkeypatch.setattr(accel, "_STAGE2_NATIVE_IMPORT_ATTEMPTED", False)

    assert accel.stage7_native_available() is False
    assert accel.stage7_native_available() is True
    assert attempts["count"] == 2


def test_stage2_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeNative:
        def accumulate_weighted_grid(
            self,
            ph_weight: np.ndarray,
            grid_lin: np.ndarray,
            n_i: int,
            n_j: int,
            threads: int,
        ) -> np.ndarray:
            calls.append(f"grid:{threads}")
            return np.full((n_i, n_j, ph_weight.shape[1]), 7 + 0j, dtype=np.complex64)

        def ps_topofit_batch_generic(
            self,
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            calls.append(f"topofit:{threads}")
            n_row, n_col = cpxphase.shape
            return (
                np.full(n_row, 1.0, dtype=np.float64),
                np.full(n_row, 2.0, dtype=np.float64),
                np.full(n_row, 3.0, dtype=np.float64),
                np.full((n_row, n_col), 4 + 0j, dtype=np.complex64),
            )

        def ps_topofit_batch_generic_f32(
            self,
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            calls.append(f"topofit32:{threads}")
            n_row, n_col = cpxphase.shape
            return (
                np.full(n_row, 1.5, dtype=np.float64),
                np.full(n_row, 2.5, dtype=np.float64),
                np.full(n_row, 3.5, dtype=np.float64),
                np.full((n_row, n_col), 4.5 + 0j, dtype=np.complex64),
            )

        def ps_topofit_batch_row_invariant(
            self,
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            calls.append(f"row:{threads}")
            n_row, n_col = cpxphase.shape
            return (
                np.full(n_row, 5.0, dtype=np.float64),
                np.full(n_row, 6.0, dtype=np.float64),
                np.full(n_row, 7.0, dtype=np.float64),
                np.full((n_row, n_col), 8 + 0j, dtype=np.complex64),
            )

        def ps_topofit_coh_row_invariant(
            self,
            cpxphase: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> np.ndarray:
            calls.append(f"rowcoh:{threads}")
            return np.full(cpxphase.shape[0], 9.0, dtype=np.float64)

        def histogram_with_centers(
            self,
            values: np.ndarray,
            centers: np.ndarray,
        ) -> np.ndarray:
            calls.append("hist")
            return np.asarray([2.0, 1.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    grid = run_stage2_grid_accumulate_kernel(
        np.ones((2, 3), dtype=np.complex64),
        np.asarray([0, 1], dtype=np.int64),
        2,
        1,
        backend="native",
        threads=3,
    )
    topofit = run_stage2_topofit_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float64),
        1.0,
        backend="native",
        threads=4,
    )
    topofit_single = run_stage2_topofit_kernel(
        np.ones((2, 3), dtype=np.complex64),
        np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32),
        1.0,
        backend="native",
        threads=5,
    )
    topofit_row = run_stage2_topofit_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="native",
        threads=2,
    )
    coh_row = run_stage2_topofit_coh_row_invariant_kernel(
        np.ones((2, 3), dtype=np.complex128),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        1.0,
        backend="native",
        threads=6,
    )
    hist = run_stage2_histogram_kernel(
        np.asarray([0.1, 0.4, 0.7], dtype=np.float64),
        np.asarray([0.0, 0.6, 1.0], dtype=np.float64),
        backend="native",
    )

    assert calls == ["grid:3", "topofit:4", "topofit32:5", "row:2", "rowcoh:6", "hist"]
    np.testing.assert_allclose(grid, np.full((2, 1, 3), 7 + 0j, dtype=np.complex64))
    np.testing.assert_allclose(topofit[0], np.full(2, 1.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(topofit_single[0], np.full(2, 1.5, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(topofit_row[0], np.full(2, 5.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(coh_row, np.full(2, 9.0, dtype=np.float64), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(hist, np.asarray([2.0, 1.0, 0.0], dtype=np.float64))


def test_stage6_unwrap_grid_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, float, tuple[int, int]]] = []

    class _FakeNative:
        def stage6_unwrap_grid(
            self,
            ifgw: np.ndarray,
            rowcost: np.ndarray,
            colcost: np.ndarray,
            nshortcycle: float,
            threads: int,
        ) -> dict[str, np.ndarray | float]:
            calls.append((int(threads), float(nshortcycle), tuple(np.asarray(ifgw).shape)))
            assert np.asarray(rowcost).dtype == np.int16
            assert np.asarray(colcost).dtype == np.int16
            return {
                "ifguw": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "msd": 5.5,
            }

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = run_stage6_unwrap_grid_kernel(
        np.ones((2, 2), dtype=np.complex64),
        np.zeros((1, 8), dtype=np.int16),
        np.zeros((2, 4), dtype=np.int16),
        backend="native",
        nshortcycle=200.0,
        threads=3,
    )

    assert calls == [(3, 200.0, (2, 2))]
    np.testing.assert_array_equal(out["ifguw"], np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert out["msd"] == 5.5


def test_stage6_extract_grid_values_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []

    class _FakeNative:
        def stage6_extract_grid_values(
            self,
            ifguw: np.ndarray,
            nzix: np.ndarray,
            threads: int,
        ) -> np.ndarray:
            calls.append((int(threads), tuple(np.asarray(ifguw).shape)))
            assert np.asarray(nzix).dtype == np.bool_
            return np.asarray([9.0, 8.0, 7.0], dtype=np.float32)

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = run_stage6_extract_grid_values_kernel(
        np.ones((2, 3), dtype=np.float32),
        np.asarray([[True, False, True], [False, True, False]], dtype=bool),
        backend="native",
        threads=4,
    )

    assert calls == [(4, (2, 3))]
    np.testing.assert_array_equal(out, np.asarray([9.0, 8.0, 7.0], dtype=np.float32))


def test_stage6_estimate_la_error_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, float, tuple[int, int]]] = []

    class _FakeNative:
        def stage6_estimate_la_error_single_master(
            self,
            dph_space: np.ndarray,
            day: np.ndarray,
            bperp: np.ndarray,
            n_trial_wraps: float,
            threads: int,
        ) -> np.ndarray:
            calls.append((int(threads), float(n_trial_wraps), tuple(np.asarray(dph_space).shape)))
            assert np.asarray(dph_space).dtype == np.complex64
            assert np.asarray(day).dtype == np.float64
            assert np.asarray(bperp).dtype == np.float64
            return np.asarray([0.25, -0.5], dtype=np.float32)

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = accel.run_stage6_estimate_la_error_kernel(
        np.ones((2, 3), dtype=np.complex64),
        np.asarray([-12.0, 6.0, 18.0], dtype=np.float64),
        np.asarray([30.0, -10.0, 45.0], dtype=np.float64),
        2.5,
        backend="native",
        threads=4,
    )

    assert calls == [(4, 2.5, (2, 3))]
    np.testing.assert_array_equal(out, np.asarray([0.25, -0.5], dtype=np.float32))


def test_stage6_smooth_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, float, tuple[int, int]]] = []

    class _FakeNative:
        def stage6_smooth_3d_full_single_master(
            self,
            dph_space: np.ndarray,
            day: np.ndarray,
            time_win: float,
            threads: int,
        ) -> dict[str, np.ndarray]:
            calls.append((int(threads), float(time_win), tuple(np.asarray(dph_space).shape)))
            assert np.asarray(dph_space).dtype == np.complex64
            assert np.asarray(day).dtype == np.float64
            return {
                "dph_smooth_uw": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "dph_noise": np.asarray([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32),
            }

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = accel.run_stage6_smooth_3d_full_single_master_kernel(
        np.ones((2, 2), dtype=np.complex64),
        np.asarray([-10.0, 20.0], dtype=np.float64),
        36.0,
        backend="native",
        threads=5,
    )

    assert calls == [(5, 36.0, (2, 2))]
    np.testing.assert_array_equal(out[0], np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.testing.assert_array_equal(out[1], np.asarray([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32))


def test_stage5_ifg_std_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []

    class _FakeNative:
        def stage5_ifg_std(
            self,
            ph2: np.ndarray,
            ph_patch: np.ndarray,
            bperp: np.ndarray,
            k_ps: np.ndarray,
            c_ps: np.ndarray,
            threads: int,
        ) -> np.ndarray:
            calls.append((int(threads), tuple(np.asarray(ph2).shape)))
            assert np.asarray(ph_patch).shape == np.asarray(ph2).shape
            assert np.asarray(bperp).shape == np.asarray(ph2).shape
            assert np.asarray(k_ps).shape == (2,)
            assert np.asarray(c_ps).shape == (2,)
            return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = run_stage5_ifg_std_kernel(
        np.ones((2, 3), dtype=np.complex64),
        np.ones((2, 3), dtype=np.complex64),
        np.zeros((2, 3), dtype=np.float64),
        np.asarray([0.1, 0.2], dtype=np.float64),
        np.asarray([0.3, 0.4], dtype=np.float64),
        backend="native",
        threads=4,
    )

    assert calls == [(4, (2, 3))]
    np.testing.assert_array_equal(out, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_stage7_scla_smooth_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []

    class _FakeNative:
        def stage7_scla_smooth(
            self,
            k_ps_uw: np.ndarray,
            c_ps_uw: np.ndarray,
            edges: np.ndarray,
            threads: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            calls.append((int(threads), tuple(np.asarray(edges).shape)))
            return (
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([3.0, 4.0], dtype=np.float32),
            )

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    k_out, c_out = run_stage7_scla_smooth_kernel(
        np.asarray([10.0, 20.0], dtype=np.float32),
        np.asarray([30.0, 40.0], dtype=np.float32),
        np.asarray([[0, 1]], dtype=np.int64),
        backend="native",
        threads=5,
    )

    assert calls == [(5, (1, 2))]
    np.testing.assert_array_equal(k_out, np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(c_out, np.asarray([3.0, 4.0], dtype=np.float32))


def test_stage7_mean_velocity_fit_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[int, int], int]] = []

    class _FakeNative:
        def stage7_mean_velocity_fit(
            self,
            ph_mean_v: np.ndarray,
            day: np.ndarray,
            master_ix: int,
            ifg_std: np.ndarray,
            threads: int,
        ) -> np.ndarray:
            calls.append((int(threads), tuple(np.asarray(ph_mean_v).shape), int(master_ix)))
            assert np.asarray(ph_mean_v).dtype == np.float64
            assert np.asarray(day).dtype == np.float64
            assert np.asarray(ifg_std).dtype == np.float64
            return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = accel.run_stage7_mean_velocity_fit_kernel(
        np.ones((2, 3), dtype=np.float64),
        np.asarray([0.0, 5.0, 12.0], dtype=np.float64),
        master_ix=2,
        ifg_std=np.asarray([1.0, 2.0, 4.0], dtype=np.float64),
        backend="native",
        threads=6,
    )

    assert calls == [(6, (2, 3), 2)]
    np.testing.assert_array_equal(out, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_stage7_deramp_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    class _FakeNative:
        def stage7_deramp_unwrapped_phase(
            self,
            xy: np.ndarray,
            ph_all: np.ndarray,
            threads: int,
        ) -> dict[str, np.ndarray]:
            calls.append((int(threads), tuple(np.asarray(xy).shape), tuple(np.asarray(ph_all).shape)))
            assert np.asarray(xy).dtype == np.float64
            assert np.asarray(ph_all).dtype == np.float64
            return {
                "ph_out": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                "ph_ramp": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
            }

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = accel.run_stage7_deramp_unwrapped_phase_kernel(
        np.ones((2, 3), dtype=np.float64),
        np.ones((2, 2), dtype=np.float64),
        backend="native",
        threads=7,
    )

    assert calls == [(7, (2, 3), (2, 2))]
    np.testing.assert_array_equal(out[0], np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    np.testing.assert_array_equal(out[1], np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))


def test_stage8_weighted_lstsq_native_dispatch_uses_native_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    class _FakeNative:
        def stage8_weighted_lstsq_diagonal(
            self,
            design: np.ndarray,
            values: np.ndarray,
            variances: np.ndarray,
            threads: int,
        ) -> np.ndarray:
            calls.append((int(threads), tuple(np.asarray(design).shape), tuple(np.asarray(values).shape)))
            assert np.asarray(variances).shape == (3,)
            return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = run_stage8_weighted_lstsq_kernel(
        np.ones((3, 2), dtype=np.float64),
        np.ones((3, 2), dtype=np.float64),
        covariance=np.diag(np.asarray([1.0, 2.0, 3.0], dtype=np.float64)),
        backend="native",
        threads=7,
    )

    assert calls == [(7, (3, 2), (3, 2))]
    np.testing.assert_array_equal(out, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_native_kernels_match_python_reference() -> None:
    rng = np.random.default_rng(11)
    cpxphase = np.exp(1j * rng.normal(size=(6, 5))).astype(np.complex128)
    bperp = rng.normal(size=(6, 5)).astype(np.float64)
    bperp_row = np.tile(np.asarray([-120.0, -40.0, 0.0, 55.0, 90.0], dtype=np.float64), (6, 1))

    expected_grid = accel._stage2_grid_accumulate_cpu(cpxphase.astype(np.complex64), np.asarray([0, 1, 0, 1, 2, 2]), 3, 1)
    observed_grid = run_stage2_grid_accumulate_kernel(
        cpxphase.astype(np.complex64),
        np.asarray([0, 1, 0, 1, 2, 2], dtype=np.int64),
        3,
        1,
        backend="native",
    )
    np.testing.assert_allclose(observed_grid, expected_grid, atol=1e-6, rtol=0.0)

    expected = ported._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps=1.5)
    observed = run_stage2_topofit_kernel(cpxphase, bperp, 1.5, backend="native")
    np.testing.assert_allclose(observed[0], expected[0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected[2], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed[3], expected[3], atol=1e-5, rtol=0.0)

    expected_row = ported._ps_topofit_batch_row_invariant(cpxphase, bperp_row, n_trial_wraps=1.5)
    observed_row = run_stage2_topofit_row_invariant_kernel(cpxphase, bperp_row, 1.5, backend="native")
    np.testing.assert_allclose(observed_row[0], expected_row[0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed_row[1], expected_row[1], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed_row[2], expected_row[2], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed_row[3], expected_row[3], atol=1e-5, rtol=0.0)

    coh_row = run_stage2_topofit_coh_row_invariant_kernel(cpxphase, bperp_row, 1.5, backend="native")
    np.testing.assert_allclose(coh_row, expected_row[2], atol=1e-10, rtol=0.0)

    hist_expected = accel._stage2_histogram_with_centers_cpu(
        np.asarray([0.1, 0.49, 0.7, 0.9, np.nan], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    hist_observed = run_stage2_histogram_kernel(
        np.asarray([0.1, 0.49, 0.7, 0.9, np.nan], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        backend="native",
    )
    np.testing.assert_allclose(hist_observed, hist_expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(hist_observed, np.asarray([1.0, 2.0, 1.0], dtype=np.float64), atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_ph_weight_block_native_matches_python_reference() -> None:
    ph_nm = np.asarray(
        [
            [0.7 + 0.2j, -0.3 + 0.9j, 0.1 - 0.8j],
            [0.6 - 0.4j, -0.2 + 0.5j, -0.9 - 0.1j],
        ],
        dtype=np.complex64,
    )
    bperp = np.asarray(
        [
            [12345.678, -9876.543, 5432.1],
            [-22222.25, 11111.75, 3333.333],
        ],
        dtype=np.float64,
    )
    k_ps = np.asarray([0.000123456789, -0.000987654321], dtype=np.float64)
    weighting = np.asarray([0.25, 0.75], dtype=np.float64)

    expected = ported._stage2_ph_weight_block(ph_nm, bperp, k_ps, weighting)
    observed = run_stage2_ph_weight_block_kernel(ph_nm, bperp, k_ps, weighting, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=0.0)
    assert observed.dtype == np.complex64


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_grid_indices_native_matches_python_reference() -> None:
    xy = np.asarray(
        [
            [1.0, 100.0, 200.0],
            [2.0, 120.1, 245.5],
            [3.0, 159.9, 260.0],
            [4.0, 100.0, 200.0],
        ],
        dtype=np.float64,
    )

    expected = ported._stage2_grid_indices(xy, 30.0)
    observed = run_stage2_grid_indices_kernel(xy, 30.0, backend="native")

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    assert observed.dtype == np.float32


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_clap_filter_native_matches_python_reference() -> None:
    expected = ported._clap_filter_kernel()
    observed = run_stage2_clap_filter_kernel(backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=1e-15)
    assert observed.shape == (7, 7)
    assert observed.dtype == np.float64


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_normalize_complex_native_matches_python_reference() -> None:
    values = np.asarray(
        [
            [3.0 + 4.0j, 0.0 + 0.0j, -5.0 + 12.0j],
            [0.25 - 0.75j, -2.0 - 2.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    expected = values.copy()
    ported._normalize_complex_unit_magnitude_inplace(expected)

    observed = run_stage2_normalize_complex_kernel(values, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-7, rtol=0.0)
    assert observed.dtype == np.complex64


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_normalize_phase_matrix_native_matches_python_reference() -> None:
    ph_nm = np.asarray(
        [
            [3.0 + 4.0j, 0.0 + 0.0j, -5.0 + 12.0j],
            [0.25 - 0.75j, -2.0 - 2.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    expected_amp = np.abs(ph_nm).astype(np.float32)
    expected_amp[expected_amp == 0] = 1.0
    expected_ph = np.divide(ph_nm, expected_amp, out=np.zeros_like(ph_nm), where=expected_amp != 0).astype(np.complex64)

    observed = run_stage2_normalize_phase_matrix_kernel(ph_nm, backend="native")

    np.testing.assert_allclose(observed["ph"], expected_ph, atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(observed["amp"], expected_amp, atol=1e-6, rtol=0.0)
    assert observed["ph"].dtype == np.complex64
    assert observed["amp"].dtype == np.float32


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_unwraps_synthetic_ramp() -> None:
    phase = np.asarray(
        [
            [0.2, 2.8, 3.4],
            [0.4, 3.0, 3.6],
        ],
        dtype=np.float32,
    )
    ifgw = np.exp(1j * phase).astype(np.complex64)
    rowcost = np.zeros((1, 12), dtype=np.int16)
    colcost = np.zeros((2, 8), dtype=np.int16)
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")

    observed = np.asarray(out["ifguw"], dtype=np.float32)
    observed -= observed[0, 0] - phase[0, 0]
    np.testing.assert_allclose(observed, phase, atol=1e-5, rtol=0.0)
    assert float(out["msd"]) >= 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_uses_positive_laycost_arcs() -> None:
    phase = np.asarray([[2.8, 3.4]], dtype=np.float32)
    ifgw = np.exp(1j * phase).astype(np.complex64)
    rowcost = np.zeros((0, 8), dtype=np.int16)
    colcost = np.zeros((1, 4), dtype=np.int16)
    colcost[0, 1] = 1
    colcost[0, 2] = 32000
    colcost[0, 3] = 1

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    observed = np.asarray(out["ifguw"], dtype=np.float32)
    observed -= observed[0, 0] - phase[0, 0]

    np.testing.assert_allclose(observed, phase, atol=1e-5, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_routes_real_residues_around_expensive_arcs() -> None:
    pi = np.pi
    phase = np.asarray(
        [
            [-pi, -pi / 2.0, -pi, -pi],
            [pi / 2.0, 0.0, -pi / 2.0, 0.0],
            [pi / 2.0, 0.0, -pi / 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    ifgw = np.exp(1j * phase).astype(np.complex64)
    rowcost = np.zeros((2, 16), dtype=np.int16)
    colcost = np.zeros((3, 12), dtype=np.int16)
    rowcost[:, 1::4] = 1000
    colcost[:, 1::4] = 1000
    rowcost[:, 2::4] = 32000
    colcost[:, 2::4] = 32000
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000
    rowcost[0, 1 * 4 + 1] = 1
    rowcost[0, 2 * 4 + 1] = 1

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    observed = np.asarray(out["ifguw"], dtype=np.float32)
    direct_flow = np.rint((observed[0, 1:3] - observed[1, 1:3]) / (2.0 * np.pi)).astype(np.int32)

    np.testing.assert_array_equal(direct_flow, np.zeros(2, dtype=np.int32))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_uses_shortest_residue_route() -> None:
    phase = np.asarray(
        [
            [-3.0, -3.0, -3.0, -3.0, -3.0],
            [1.5, -1.5, -3.0, -1.5, 1.5],
            [1.5, -1.5, -3.0, -1.5, 1.5],
            [1.5, -1.5, -3.0, -1.5, 1.5],
        ],
        dtype=np.float32,
    )
    ifgw = np.exp(1j * phase).astype(np.complex64)
    rowcost = np.zeros((3, 20), dtype=np.int16)
    colcost = np.zeros((4, 16), dtype=np.int16)
    rowcost[:, 1::4] = 1
    colcost[:, 1::4] = 1
    rowcost[:, 2::4] = 32000
    colcost[:, 2::4] = 32000
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000
    rowcost[2, 1 * 4 + 1] = 1000
    rowcost[2, 2 * 4 + 1] = 1000
    rowcost[2, 3 * 4 + 1] = 1000
    colcost[1, 0 * 4 + 1] = 1000
    colcost[1, 3 * 4 + 1] = 1000
    colcost[2, 0 * 4 + 1] = 1000
    colcost[2, 3 * 4 + 1] = 1000

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    observed = np.asarray(out["ifguw"], dtype=np.float32)
    direct_top = np.rint((observed[0, 1:4] - observed[1, 1:4]) / (2.0 * np.pi)).astype(np.int32)
    direct_middle = np.rint((observed[1, 1:4] - observed[2, 1:4]) / (2.0 * np.pi)).astype(np.int32)
    expensive_top_boundary = np.rint((observed[0, 1:] - observed[0, :-1]) / (2.0 * np.pi)).astype(np.int32)
    cheap_deep_route = np.rint((observed[2, 1:4] - observed[3, 1:4]) / (2.0 * np.pi)).astype(np.int32)

    np.testing.assert_array_equal(direct_top, np.zeros(3, dtype=np.int32))
    np.testing.assert_array_equal(direct_middle, np.zeros(3, dtype=np.int32))
    np.testing.assert_array_equal(expensive_top_boundary, np.zeros(4, dtype=np.int32))
    assert np.count_nonzero(cheap_deep_route) > 0


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_uses_defo_cost_for_offset_only_inconsistent_loop() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    rowcost = np.zeros((1, 8), dtype=np.int16)
    colcost = np.zeros((2, 4), dtype=np.int16)
    rowcost[:, 1::4] = 1
    colcost[:, 1::4] = 1
    rowcost[:, 2::4] = 32000
    colcost[:, 2::4] = 32000
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000

    # Offset-derived cycle targets are intentionally inconsistent around the
    # plaquette. They must not become branch-balance residues, but they still
    # participate in the DEFO label cost.
    rowcost[0, 0] = 0
    rowcost[0, 4] = -1000
    colcost[0, 0] = 1000
    colcost[1, 0] = 0

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    np.testing.assert_array_equal(labels, np.asarray([[0, -2], [-2, -5]], dtype=np.int32))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_uses_defo_cost_for_offset_only_loop_error() -> None:
    ifgw = np.ones((2, 2), dtype=np.complex64)
    rowcost = np.zeros((1, 8), dtype=np.int16)
    colcost = np.zeros((2, 4), dtype=np.int16)
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000
    rowcost[:, 1::4] = np.asarray([[1, 5]], dtype=np.int16)
    colcost[:, 1::4] = np.asarray([[1], [2]], dtype=np.int16)
    rowcost[:, 2::4] = 32000
    colcost[:, 2::4] = 32000

    # Offset-derived cycle targets around the only plaquette are inconsistent.
    # They should not be routed as wrapped-phase residues, but they do affect
    # the final DEFO label objective.
    colcost[0, 0] = 600
    rowcost[0, 4] = -600
    colcost[1, 0] = -400
    rowcost[0, 0] = 0

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    np.testing.assert_array_equal(labels, np.asarray([[0, -2], [-1, -1]], dtype=np.int32))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_uses_defo_cost_for_offset_only_boundary_residue() -> None:
    ifgw = np.ones((2, 3), dtype=np.complex64)
    rowcost = np.zeros((1, 12), dtype=np.int16)
    colcost = np.zeros((2, 8), dtype=np.int16)
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000
    rowcost[:, 1::4] = np.asarray([[2, 50, 5]], dtype=np.int16)
    colcost[:, 1::4] = np.asarray([[3, 1], [2, 1]], dtype=np.int16)
    rowcost[:, 2::4] = 32000
    colcost[:, 2::4] = 32000

    # Offset-derived cycle targets create artificial curl. Wrapped phase is
    # residue-free, so branch balancing leaves residue routing alone while
    # DEFO label refinement can still move labels.
    colcost[0, 0] = -600
    colcost[0, 4] = 0
    colcost[1, 0] = 200
    colcost[1, 4] = 400
    rowcost[0, 0] = -200
    rowcost[0, 4] = 0
    rowcost[0, 8] = 0

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    expected = np.asarray([[0, 1, 0], [0, 0, -2]], dtype=np.int32)
    equivalent = np.asarray([[0, 1, 1], [0, 0, -2]], dtype=np.int32)
    assert labels.tolist() in (expected.tolist(), equivalent.tolist())


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_uses_defo_cost_for_offset_only_residue_pair() -> None:
    ifgw = np.ones((2, 3), dtype=np.complex64)
    rowcost = np.zeros((1, 12), dtype=np.int16)
    colcost = np.zeros((2, 8), dtype=np.int16)
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000
    rowcost[:, 1::4] = np.asarray([[3, 5, 5]], dtype=np.int16)
    colcost[:, 1::4] = np.asarray([[5, 50], [1, 1]], dtype=np.int16)
    rowcost[:, 2::4] = 32000
    colcost[:, 2::4] = 32000

    # Offset targets would form adjacent artificial residues. These are not
    # wrapped-phase residues, so branch balancing stays on the phase graph and
    # label refinement handles the cost tension.
    colcost[0, 0] = 0
    colcost[0, 4] = -200
    colcost[1, 0] = -200
    colcost[1, 4] = 400
    rowcost[0, 0] = -400
    rowcost[0, 4] = -600
    rowcost[0, 8] = 400

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    np.testing.assert_array_equal(labels, np.asarray([[0, 1, -4], [-2, -1, -3]], dtype=np.int32))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_native_unwrap_grid_accepts_patch_shift_for_separated_offset_targets() -> None:
    ifgw = np.ones((2, 4), dtype=np.complex64)
    rowcost = np.zeros((1, 16), dtype=np.int16)
    colcost = np.zeros((2, 12), dtype=np.int16)
    rowcost[:, 3::4] = -32000
    colcost[:, 3::4] = -32000
    rowcost[:, 1::4] = np.asarray([[1, 50, 50, 1]], dtype=np.int16)
    colcost[:, 1::4] = 1
    rowcost[:, 2::4] = 32000
    colcost[:, 2::4] = 32000

    # Desired horizontal jumps are top [1, 0, -1], bottom [0, 0, 0].
    # These offset targets are not treated as wrapped-phase residues during
    # branch balancing, but later DEFO objective refinement can still move a
    # bounded patch when it lowers the edge cost.
    colcost[0, 0] = 200
    colcost[0, 4] = 0
    colcost[0, 8] = -200

    out = run_stage6_unwrap_grid_kernel(ifgw, rowcost, colcost, backend="native")
    labels = np.rint(np.asarray(out["ifguw"], dtype=np.float32) / (2.0 * np.pi)).astype(np.int32)
    labels -= labels[0, 0]

    np.testing.assert_array_equal(labels, np.asarray([[0, -1, -1, 0], [0, 0, 0, 0]], dtype=np.int32))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_extract_grid_values_native_matches_python_reference() -> None:
    ifguw = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    nzix = np.asarray([[True, False, True], [False, True, False]], dtype=bool)

    expected = run_stage6_extract_grid_values_kernel(ifguw, nzix, backend="python")
    observed = run_stage6_extract_grid_values_kernel(ifguw, nzix, backend="native")

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_prepare_cost_offsets_native_matches_python_reference() -> None:
    rowcost_base = np.zeros((2, 8), dtype=np.int16)
    colcost_base = np.zeros((3, 4), dtype=np.int16)
    rowcost_base[:, 1::4] = np.asarray([[3, 5], [7, 11]], dtype=np.int16)
    rowcost_base[:, 2::4] = 32000
    rowcost_base[:, 3::4] = np.asarray([[-32000, 1], [-32000, -32000]], dtype=np.int16)
    colcost_base[:, 1::4] = np.asarray([[13], [17], [19]], dtype=np.int16)
    colcost_base[:, 2::4] = 32000
    colcost_base[:, 3::4] = np.asarray([[-32000], [-32000], [1]], dtype=np.int16)
    rowix = np.asarray([[1.0, -2.0], [0.0, np.nan]], dtype=np.float64)
    colix = np.asarray([[2.0], [-1.0], [np.nan]], dtype=np.float64)
    wrapped = np.asarray([1.25, -0.5], dtype=np.float32)
    smooth = np.asarray([0.25, 0.75], dtype=np.float32)

    expected_row, expected_col = run_stage6_prepare_cost_offsets_kernel(
        rowcost_base,
        colcost_base,
        rowix,
        colix,
        wrapped,
        smooth,
        nshortcycle=200.0,
        backend="python",
    )
    observed_row, observed_col = run_stage6_prepare_cost_offsets_kernel(
        rowcost_base,
        colcost_base,
        rowix,
        colix,
        wrapped,
        smooth,
        nshortcycle=200.0,
        backend="native",
    )

    np.testing.assert_array_equal(observed_row, expected_row)
    np.testing.assert_array_equal(observed_col, expected_col)
    np.testing.assert_array_equal(rowcost_base[:, 0::4], np.zeros((2, 2), dtype=np.int16))
    np.testing.assert_array_equal(colcost_base[:, 0::4], np.zeros((3, 1), dtype=np.int16))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_reconstruct_ps_phase_native_matches_python_reference() -> None:
    ph_uw_grid = np.asarray(
        [
            [0.2, 1.1, -2.4],
            [3.4, -0.7, 2.2],
            [-1.5, 0.6, 4.0],
        ],
        dtype=np.float32,
    )
    ps_grid_idx = np.asarray([1, 0, 3, 2], dtype=np.int64)
    phase_in = np.asarray(
        [
            [0.4, 1.3, -2.1],
            [2.0, -1.0, 0.5],
            [-1.2, 0.9, -2.0],
            [-2.5, 1.7, 2.8],
        ],
        dtype=np.float32,
    )
    ph_in = np.exp(1j * phase_in).astype(np.complex64)
    phase_restore = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [9.0, 9.0, 9.0],
            [-0.5, 0.0, 0.5],
            [1.0, -1.0, 0.25],
        ],
        dtype=np.float32,
    )

    expected = run_stage6_reconstruct_ps_phase_kernel(
        ph_uw_grid,
        ps_grid_idx,
        ph_in,
        phase_restore=phase_restore,
        backend="python",
    )
    observed = run_stage6_reconstruct_ps_phase_kernel(
        ph_uw_grid,
        ps_grid_idx,
        ph_in,
        phase_restore=phase_restore,
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=0.0, equal_nan=True)
    assert np.isnan(observed[1, :]).all()


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_ps_grid_indices_native_matches_python_reference() -> None:
    nzix = np.asarray(
        [
            [True, False, True],
            [False, True, True],
            [True, False, False],
        ],
        dtype=bool,
    )
    grid_ij = np.asarray(
        [
            [1, 1],
            [2, 1],
            [3, 1],
            [1, 2],
            [2, 2],
            [3, 3],
        ],
        dtype=np.int64,
    )

    gridix_flat = np.zeros(nzix.size, dtype=np.int64)
    nz_flat_f = np.flatnonzero(nzix.reshape(-1, order="F"))
    gridix_flat[nz_flat_f] = np.arange(1, int(np.count_nonzero(nzix)) + 1, dtype=np.int64)
    expected = gridix_flat.reshape(nzix.shape, order="F")[grid_ij[:, 0] - 1, grid_ij[:, 1] - 1]
    observed = run_stage6_ps_grid_indices_kernel(nzix, grid_ij, backend="native")

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_select_ifgw_native_matches_python_reference() -> None:
    uw_ph = np.asarray(
        [
            [1.0 + 0.0j, 0.5 + 0.5j],
            [0.0 + 1.0j, -0.5 + 0.25j],
            [-1.0 + 0.0j, 0.0 - 1.0j],
            [0.25 - 0.75j, 1.0 + 0.25j],
        ],
        dtype=np.complex64,
    )
    z = np.asarray([[1, 3, 2], [4, 1, 3]], dtype=np.int64)

    expected = np.asarray(uw_ph[z - 1, 1], dtype=np.complex64)
    observed = run_stage6_select_ifgw_kernel(uw_ph, z, 1, backend="native")

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_select_ifg_index_native_matches_python_reference() -> None:
    observed = run_stage3_select_ifg_index_kernel(
        n_ifg=6,
        master_ix=3,
        drop_ifg_index=np.asarray([2, 6], dtype=np.int64),
        small_baseline=False,
        backend="native",
    )

    np.testing.assert_array_equal(
        observed,
        np.asarray([1.0, 3.0, 4.0], dtype=np.float64),
    )

    sb_observed = run_stage3_select_ifg_index_kernel(
        n_ifg=6,
        master_ix=3,
        drop_ifg_index=np.asarray([2, 6], dtype=np.int64),
        small_baseline=True,
        backend="native",
    )
    np.testing.assert_array_equal(
        sb_observed,
        np.asarray([1.0, 3.0, 4.0, 5.0], dtype=np.float64),
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_coh_threshold_native_matches_python_reference() -> None:
    coh_values = np.asarray(
        [0.06, 0.12, 0.18, 0.31, 0.44, 0.57, 0.68, 0.73, 0.82, 0.91, 0.0, np.nan],
        dtype=np.float64,
    )
    d_a = np.asarray([0.1, 0.12, 0.16, 0.22, 0.28, 0.32, 0.38, 0.42, 0.48, 0.52, 0.58, 0.62], dtype=np.float64)
    d_a_max = np.asarray([0.0, 0.2, 0.4, 0.7], dtype=np.float64)
    coh_bins = np.arange(0.005, 1.0, 0.01, dtype=np.float64)
    nr_dist = np.linspace(1.0, 2.0, coh_bins.size, dtype=np.float64)

    expected_thresh, expected_coeffs = ported._coh_threshold_from_dist(
        coh_values=coh_values,
        D_A=d_a,
        D_A_max=d_a_max,
        coh_bins=coh_bins,
        Nr_dist=nr_dist,
        low_coh_thresh=31,
        max_percent_rand=3.0,
        select_method="DENSITY",
        histogram_backend="python",
    )
    observed_thresh, observed_coeffs = run_stage3_coh_threshold_kernel(
        coh_values,
        d_a,
        d_a_max,
        coh_bins,
        nr_dist,
        low_coh_thresh=31,
        max_percent_rand=3.0,
        select_method="DENSITY",
        backend="native",
    )

    np.testing.assert_allclose(observed_thresh, expected_thresh, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(observed_coeffs, expected_coeffs, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_clap_filt_patch_native_matches_python_reference() -> None:
    yy, xx = np.mgrid[0:8, 0:8]
    ph = (np.exp(1j * (0.2 * xx + 0.35 * yy)) * (1.0 + 0.05 * xx)).astype(np.complex128)
    ph[2, 3] = np.nan + 0j
    low_pass = np.full((8, 8), 0.15, dtype=np.float64)
    low_pass[0, 0] = 0.6

    expected = _reference_clap_filt_patch(ph, alpha=1.1, beta=0.25, low_pass=low_pass)
    observed = run_stage3_clap_filt_patch_kernel(ph, alpha=1.1, beta=0.25, low_pass=low_pass, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)
    assert observed.dtype == np.complex128


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_clap_filt_patch_native_matches_odd_sized_python_reference() -> None:
    yy, xx = np.mgrid[0:7, 0:9]
    ph = (np.exp(1j * (0.17 * xx - 0.29 * yy)) * (1.0 + 0.03 * yy)).astype(np.complex128)
    ph[4, 1] = np.nan + 0j
    low_pass = np.full((7, 9), 0.12, dtype=np.float64)
    low_pass[0, 0] = 0.5

    expected = _reference_clap_filt_patch(ph, alpha=1.3, beta=0.35, low_pass=low_pass)
    observed = run_stage3_clap_filt_patch_kernel(ph, alpha=1.3, beta=0.35, low_pass=low_pass, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_clap_filt_grid_native_matches_python_reference() -> None:
    yy, xx = np.mgrid[0:7, 0:8]
    ph = (np.exp(1j * (0.19 * xx - 0.11 * yy)) * (1.0 + 0.02 * xx + 0.03 * yy)).astype(np.complex64)
    ph[2, 5] = np.nan + 0j
    low_pass = np.full((6, 6), 0.08, dtype=np.float64)
    low_pass[0, 0] = 0.4

    expected = _reference_clap_filt_grid(
        ph,
        alpha=1.2,
        beta=0.27,
        n_win=4,
        n_pad=2,
        low_pass=low_pass,
    )
    observed = run_stage3_clap_filt_grid_kernel(
        ph,
        alpha=1.2,
        beta=0.27,
        n_win=4,
        n_pad=2,
        low_pass=low_pass,
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)
    assert observed.dtype == np.complex64


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_clap_filt_grid_stack_native_matches_python_reference() -> None:
    yy, xx = np.mgrid[0:7, 0:8]
    base = np.exp(1j * (0.13 * xx + 0.21 * yy)) * (1.0 + 0.01 * xx + 0.02 * yy)
    ph = np.stack(
        [
            base,
            np.conj(base) * (1.0 + 0.05 * yy),
            np.exp(1j * (0.31 * xx - 0.07 * yy)),
        ],
        axis=2,
    ).astype(np.complex64)
    ph[1, 6, 0] = np.nan + 0j
    ph[4, 2, 2] = np.nan + 0j
    low_pass = np.full((6, 6), 0.06, dtype=np.float64)
    low_pass[0, 0] = 0.35

    expected = _reference_clap_filt_grid_stack(
        ph,
        alpha=1.15,
        beta=0.22,
        n_win=4,
        n_pad=2,
        low_pass=low_pass,
    )
    observed = run_stage3_clap_filt_grid_stack_kernel(
        ph,
        alpha=1.15,
        beta=0.22,
        n_win=4,
        n_pad=2,
        low_pass=low_pass,
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)
    assert observed.dtype == np.complex64


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_wrap_filt_native_matches_python_reference_with_low_output() -> None:
    yy, xx = np.mgrid[0:7, 0:8]
    ph = (np.exp(1j * (0.23 * xx - 0.17 * yy)) * (1.0 + 0.04 * xx)).astype(np.complex64)
    ph[2, 4] = np.nan + 0j

    expected, expected_low = _reference_wrap_filt(ph, n_win=4, alpha=1.25, n_pad=2, low_flag="y")
    observed, observed_low = run_stage3_wrap_filt_kernel(
        ph,
        n_win=4,
        alpha=1.25,
        n_pad=2,
        low_flag="y",
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)
    assert observed.dtype == np.complex64
    assert observed_low is not None
    assert expected_low is not None
    np.testing.assert_allclose(observed_low, expected_low, atol=1e-5, rtol=1e-5)
    assert observed_low.dtype == np.complex64


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_wrap_filt_global_native_matches_python_reference_without_low_output() -> None:
    yy, xx = np.mgrid[0:8, 0:8]
    ph = (np.exp(1j * (0.18 * xx + 0.09 * yy)) * (1.0 + 0.03 * yy)).astype(np.complex64)
    ph[5, 1] = np.nan + 0j

    expected, expected_low = _reference_wrap_filt_global(ph, n_win=4, alpha=1.1, n_pad=1, low_flag="n")
    observed, observed_low = run_stage3_wrap_filt_global_kernel(
        ph,
        n_win=4,
        alpha=1.1,
        n_pad=1,
        low_flag="n",
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=1e-5)
    assert observed.dtype == np.complex64
    assert expected_low is None
    assert observed_low is None


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage4_duplicate_keep_native_matches_python_reference() -> None:
    xy = np.asarray(
        [
            [10.0, 20.0],
            [10.0, 20.0],
            [15.0, 25.0],
            [15.0, 25.0],
            [50.0, 60.0],
        ],
        dtype=np.float64,
    )
    coh = np.asarray([0.3, 0.8, 0.7, 0.1, 0.2], dtype=np.float64)

    expected = ported._dedup_lonlat_keep_highest_coh(xy, coh)
    observed = run_stage4_duplicate_keep_kernel(xy, coh, backend="native")

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage4_adjacent_component_keep_native_matches_python_reference() -> None:
    ij = np.asarray(
        [
            [10, 10],
            [10, 11],
            [11, 10],
            [30, 30],
            [31, 30],
            [60, 60],
        ],
        dtype=np.int64,
    )
    coh = np.asarray([0.5, 0.9, 0.7, 0.2, 0.4, 0.1], dtype=np.float64)

    expected = ported._adjacent_component_keep_mask(ij, coh)
    observed = run_stage4_adjacent_component_keep_kernel(ij, coh, backend="native")

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage4_weed_ifg_index_native_matches_python_reference() -> None:
    observed = run_stage4_weed_ifg_index_kernel(
        n_ifg=6,
        drop_ifg_index=np.asarray([2, 4], dtype=np.int64),
        backend="native",
    )

    np.testing.assert_array_equal(
        observed,
        np.asarray([1.0, 3.0, 5.0, 6.0], dtype=np.float64),
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-4 extension not available",
)
def test_stage4_phase_correction_native_matches_python_reference() -> None:
    ph2 = np.asarray(
        [
            [0.7 + 0.2j, -0.3 + 0.9j, 0.1 - 0.8j, 0.4 + 0.5j],
            [0.6 - 0.4j, -0.2 + 0.5j, -0.9 - 0.1j, 0.3 - 0.7j],
            [0.1 + 0.0j, 0.0 + 0.0j, -0.5 + 0.2j, 0.8 - 0.6j],
        ],
        dtype=np.complex128,
    )
    ix_weed = np.asarray([True, False, True], dtype=bool)
    k_ps = np.asarray([0.0012, -0.0034, 0.0045], dtype=np.float64)
    c_ps = np.asarray([0.2, -0.4, 0.7], dtype=np.float64)
    bperp = np.asarray([0.0, 12.5, -33.0, 44.0], dtype=np.float64)
    master_ix = 2

    expected = ph2[ix_weed, :] * np.exp(-1j * (k_ps[ix_weed][:, None] * bperp[None, :]))
    expected = np.divide(expected, np.abs(expected), out=np.zeros_like(expected), where=np.abs(expected) != 0)
    expected = np.divide(expected, np.abs(expected), out=np.zeros_like(expected), where=np.abs(expected) != 0)
    expected[:, master_ix - 1] = np.exp(1j * c_ps[ix_weed])

    observed = run_stage4_phase_correction_kernel(
        ph2,
        ix_weed,
        k_ps,
        c_ps,
        bperp,
        small_baseline=False,
        master_ix=master_ix,
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0.0)
    assert observed.dtype == np.complex128


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_unwrap_ifg_sets_native_matches_python_reference() -> None:
    observed = run_stage6_unwrap_ifg_sets_kernel(
        n_ifg=7,
        master_ix=4,
        drop_ifg_index=np.asarray([2, 6], dtype=np.int64),
        small_baseline=False,
        backend="native",
    )

    np.testing.assert_array_equal(observed["unwrap_ifg"], np.asarray([1, 3, 4, 5, 7], dtype=np.int64))
    np.testing.assert_array_equal(observed["solve_ifg"], np.asarray([1, 3, 5, 7], dtype=np.int64))

    sb_observed = run_stage6_unwrap_ifg_sets_kernel(
        n_ifg=5,
        master_ix=3,
        drop_ifg_index=np.asarray([4], dtype=np.int64),
        small_baseline=True,
        backend="native",
    )

    np.testing.assert_array_equal(sb_observed["unwrap_ifg"], np.asarray([1, 2, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(sb_observed["solve_ifg"], np.asarray([1, 2, 3, 5], dtype=np.int64))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_single_master_ifg_geometry_native_matches_python_reference() -> None:
    expected_unwrap, expected_ifgday = ported._build_single_master_ifg_geometry(n_ifg=6, master_ix=3)

    observed = run_stage6_single_master_ifg_geometry_kernel(
        n_ifg=6,
        master_ix=3,
        backend="native",
    )

    np.testing.assert_array_equal(observed["unwrap_ifg"], expected_unwrap)
    np.testing.assert_array_equal(observed["ifgday_ix"], expected_ifgday)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_grid_accumulate_native_matches_python_reference() -> None:
    ph_in = np.asarray(
        [
            [1.0 + 2.0j, 0.5 - 0.5j],
            [3.0 - 1.0j, -2.0 + 0.25j],
            [-1.0 + 0.0j, 1.5 + 1.0j],
            [0.25 + 0.75j, 0.0 - 1.0j],
        ],
        dtype=np.complex64,
    )
    grid_lin = np.asarray([2, 0, 2, 4], dtype=np.int64)
    group_lin, grouped_cols = ported._group_reduce_by_index(ph_in, grid_lin)
    expected = np.column_stack(
        [
            ported._accumulate_grid_column(group_lin, grouped_cols[:, i_ifg], 6)
            for i_ifg in range(ph_in.shape[1])
        ]
    ).astype(np.complex64)

    observed = run_stage6_grid_accumulate_kernel(ph_in, grid_lin, n_cells=6, backend="native")

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_estimate_la_error_native_matches_python_reference() -> None:
    day = np.asarray([-24.0, -12.0, 18.0, 36.0], dtype=np.float64)
    bperp = np.asarray([45.0, -15.0, 30.0, 75.0], dtype=np.float64)
    k_true = np.asarray([0.015, -0.02, 0.0], dtype=np.float64)
    phase = k_true[:, None] * bperp[None, :]
    phase += np.asarray([[0.02, -0.01, 0.03, -0.02], [0.01, 0.02, -0.02, 0.01], [0.0, 0.0, 0.0, 0.0]])
    dph_space = np.exp(1j * phase).astype(np.complex64)

    expected = ported._estimate_la_error_single_master(dph_space, day=day, bperp=bperp, n_trial_wraps=2.0)
    observed = accel.run_stage6_estimate_la_error_kernel(
        dph_space,
        day,
        bperp,
        2.0,
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage6_smooth_native_matches_python_reference() -> None:
    phase = np.asarray(
        [
            [0.2, -0.3, 0.5, -0.7],
            [-0.4, 0.1, -0.2, 0.6],
            [0.9, -0.8, 0.3, -0.1],
        ],
        dtype=np.float32,
    )
    dph_space = np.exp(1j * phase).astype(np.complex64)
    day = np.asarray([-18.0, -6.0, 12.0, 30.0], dtype=np.float64)

    expected = ported._smooth_3d_full_single_master(dph_space, day=day, time_win=24.0, chunk_edges=2)
    observed = accel.run_stage6_smooth_3d_full_single_master_kernel(dph_space, day, 24.0, backend="native")

    np.testing.assert_allclose(observed[0], expected[0], atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=2e-6, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage5_native_ifg_std_matches_python_reference() -> None:
    phase = np.asarray(
        [
            [0.2, -0.1, 0.4],
            [-0.3, 0.5, -0.2],
            [0.1, 0.3, -0.6],
        ],
        dtype=np.float32,
    )
    patch_phase = np.asarray(
        [
            [0.05, 0.0, -0.05],
            [-0.02, 0.04, 0.01],
            [0.0, -0.03, 0.02],
        ],
        dtype=np.float32,
    )
    bperp = np.asarray(
        [
            [10.0, 0.0, 20.0],
            [15.0, 0.0, 25.0],
            [12.0, 0.0, 22.0],
        ],
        dtype=np.float64,
    )
    k_ps = np.asarray([0.01, -0.02, 0.03], dtype=np.float64)
    c_ps = np.asarray([0.1, -0.2, 0.05], dtype=np.float64)

    ph2 = np.exp(1j * phase).astype(np.complex64)
    ph_patch = np.exp(1j * patch_phase).astype(np.complex64)
    expected = run_stage5_ifg_std_kernel(ph2, ph_patch, bperp, k_ps, c_ps, backend="python")
    observed = run_stage5_ifg_std_kernel(ph2, ph_patch, bperp, k_ps, c_ps, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-5, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage5_duplicate_keep_native_matches_python_reference() -> None:
    lonlat = np.asarray(
        [
            [13.0, 45.0],
            [13.0, 45.0],
            [13.5, 45.2],
            [14.0, 46.0],
            [13.5, 45.2],
            [15.0, 47.0],
        ],
        dtype=np.float64,
    )
    coh = np.asarray([0.2, 0.8, 0.5, 0.1, 0.9, 0.3], dtype=np.float64)

    expected = ported._dedup_lonlat_keep_highest_coh(lonlat, coh)
    observed = run_stage5_duplicate_keep_kernel(lonlat, coh, backend="native")

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage5_format_merged_rc2_native_matches_python_reference() -> None:
    rc2_all = np.asarray(
        [
            [3.0 + 4.0j, 0.0 + 0.0j, -2.0j],
            [1.0 - 1.0j, 2.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex64,
    )

    expected = ported._format_merged_rc2_payload(rc2_all)
    observed = run_stage5_format_merged_rc2_kernel(rc2_all, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage5_rc2_correction_native_matches_python_reference() -> None:
    ph2 = np.asarray(
        [
            [0.7 + 0.2j, -0.3 + 0.9j, 0.1 - 0.8j, 0.4 + 0.5j],
            [0.6 - 0.4j, -0.2 + 0.5j, -0.9 - 0.1j, 0.3 - 0.7j],
        ],
        dtype=np.complex64,
    )
    ph_patch = np.asarray(
        [
            [0.1 + 0.9j, 0.3 - 0.2j, -0.4 + 0.6j],
            [0.7 - 0.1j, -0.8 + 0.3j, 0.2 + 0.5j],
        ],
        dtype=np.complex64,
    )
    bperp = np.asarray([[11.0, -22.0, 33.0], [44.0, -55.0, 66.0]], dtype=np.float64)
    k_ps = np.asarray([0.0012, -0.0034], dtype=np.float64)
    c_ps = np.asarray([0.2, -0.4], dtype=np.float64)
    master_ix = 2

    bperp_full = np.concatenate(
        [
            bperp[:, : master_ix - 1],
            np.zeros((ph2.shape[0], 1), dtype=np.float64),
            bperp[:, master_ix - 1 :],
        ],
        axis=1,
    )
    expected_rc = ph2.astype(np.complex128) * np.exp(-1j * (k_ps[:, None] * bperp_full + c_ps[:, None]))
    expected_reref = np.concatenate(
        [
            ph_patch[:, : master_ix - 1],
            np.ones((ph2.shape[0], 1), dtype=np.complex64),
            ph_patch[:, master_ix - 1 :],
        ],
        axis=1,
    )

    observed = run_stage5_rc2_correction_kernel(
        ph2,
        ph_patch,
        bperp,
        k_ps,
        c_ps,
        small_baseline=False,
        master_ix=master_ix,
        backend="native",
    )

    np.testing.assert_allclose(observed["ph_rc"], expected_rc.astype(np.complex64), atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(observed["ph_reref"], expected_reref.astype(np.complex64), atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage5_patch_keep_mask_native_matches_python_reference() -> None:
    ij_cols = np.asarray(
        [
            [9, 9],
            [2, 2],
            [4, 2],
            [8, 8],
            [6, 3],
        ],
        dtype=np.int64,
    )
    merged_ij_cols = np.asarray([[9, 9], [4, 2], [8, 8]], dtype=np.int64)
    merged_indices = np.asarray([10, 11, 12], dtype=np.int64)
    patch_bounds = np.asarray([2, 4, 2, 6], dtype=np.int64)
    merged_index_by_key = {
        key: int(index)
        for key, index in zip(ported._row_keys(merged_ij_cols), merged_indices, strict=True)
    }
    expected_keep, expected_remove = ported._compute_patch_keep_mask(
        ij_cols=ij_cols,
        ij_keys=ported._row_keys(ij_cols),
        patch_bounds=tuple(int(v) for v in patch_bounds.tolist()),
        merged_index_by_key=merged_index_by_key,
    )

    observed = run_stage5_patch_keep_mask_kernel(
        ij_cols,
        merged_ij_cols,
        merged_indices,
        patch_bounds,
        backend="native",
    )

    np.testing.assert_array_equal(observed["keep_patch"], expected_keep)
    np.testing.assert_array_equal(observed["remove_ix"], np.asarray(expected_remove, dtype=np.int64))


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage4_stage7_stage8_native_kernels_match_python_reference() -> None:
    rng = np.random.default_rng(123)

    ph_weed = np.exp(1j * rng.normal(size=(8, 6))).astype(np.complex128)
    node_a = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    node_b = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    bperp = np.linspace(-50.0, 80.0, 6, dtype=np.float64)
    day = np.asarray([1.0, 5.0, 9.0, 15.0, 22.0, 30.0], dtype=np.float64)

    for small_baseline, day_vec in (
        (False, day),
        (True, np.asarray([], dtype=np.float64)),
    ):
        expected_stage4 = run_stage4_edge_stats_kernel(
            ph_weed,
            node_a,
            node_b,
            bperp,
            day_vec,
            time_win=30.0,
            small_baseline=small_baseline,
            backend="python",
        )
        observed_stage4 = run_stage4_edge_stats_kernel(
            ph_weed,
            node_a,
            node_b,
            bperp,
            day_vec,
            time_win=30.0,
            small_baseline=small_baseline,
            backend="native",
        )
        np.testing.assert_allclose(observed_stage4["ps_std"], expected_stage4["ps_std"], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(observed_stage4["ps_max"], expected_stage4["ps_max"], atol=1e-12, rtol=0.0)

    ph_proc = rng.normal(size=(5, 6))
    ph_mean_v = rng.normal(size=(5, 6))
    bperp_mat = rng.normal(size=(5, 6))
    unwrap_ix = np.arange(6, dtype=np.int64)
    solve_ix = np.asarray([0, 1, 3, 4, 5], dtype=np.int64)
    day_stage7 = np.cumsum(rng.uniform(1.0, 4.0, size=6))
    ifg_std = rng.uniform(0.5, 2.0, size=6)

    expected_stage7 = run_stage7_scla_kernel(
        ph_proc,
        ph_mean_v,
        bperp_mat,
        unwrap_ix,
        solve_ix,
        day_stage7,
        master_ix=3,
        ifg_std=ifg_std,
        backend="python",
    )
    observed_stage7 = run_stage7_scla_kernel(
        ph_proc,
        ph_mean_v,
        bperp_mat,
        unwrap_ix,
        solve_ix,
        day_stage7,
        master_ix=3,
        ifg_std=ifg_std,
        backend="native",
    )
    for key in ("K_ps_uw", "C_ps_uw", "ph_scla", "ph_ramp", "ifg_vcm", "mean_v", "m"):
        np.testing.assert_allclose(observed_stage7[key], expected_stage7[key], atol=1e-12, rtol=0.0)

    uw_ph = np.exp(1j * rng.normal(size=(8, 6))).astype(np.complex64)
    expected_stage8 = run_stage8_edge_noise_kernel(uw_ph, node_a, node_b, backend="python")
    observed_stage8 = run_stage8_edge_noise_kernel(uw_ph, node_a, node_b, backend="native")
    np.testing.assert_allclose(observed_stage8["dph_noise"], expected_stage8["dph_noise"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(observed_stage8["dph_space_uw"], expected_stage8["dph_space_uw"], atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage7_scla_smooth_native_matches_python_reference() -> None:
    k_ps_uw = np.asarray([10.0, 1.0, 2.0, -4.0, 3.0], dtype=np.float32)
    c_ps_uw = np.asarray([5.0, 0.0, 2.0, 8.0, -1.0], dtype=np.float32)
    edges = np.asarray([[0, 1], [1, 2], [0, 2], [2, 3], [3, 4], [99, 1], [1, 1]], dtype=np.int64)

    expected = run_stage7_scla_smooth_kernel(k_ps_uw, c_ps_uw, edges, backend="python")
    observed = run_stage7_scla_smooth_kernel(k_ps_uw, c_ps_uw, edges, backend="native")

    np.testing.assert_allclose(observed[0], expected[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage7_center_to_reference_native_matches_python_reference() -> None:
    ph = np.asarray(
        [
            [1.0, np.nan, 4.0],
            [3.0, 6.0, np.nan],
            [9.0, 8.0, 10.0],
        ],
        dtype=np.float64,
    )
    ref_ix = np.asarray([0, 1], dtype=np.int64)

    expected = ported._center_to_reference(ph, ref_ix)
    observed = run_stage7_center_to_reference_kernel(ph, ref_ix, backend="native")

    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0, equal_nan=True)
    np.testing.assert_allclose(
        run_stage7_center_to_reference_kernel(ph, np.asarray([], dtype=np.int64), backend="native"),
        ph,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage8_weighted_lstsq_native_matches_python_reference() -> None:
    design = np.asarray(
        [
            [1.0, -2.0],
            [1.0, 0.0],
            [1.0, 3.0],
            [1.0, 5.0],
        ],
        dtype=np.float64,
    )
    coeffs_true = np.asarray([[1.5, -2.0, 0.25], [0.5, 1.25, -0.75]], dtype=np.float64)
    values = design @ coeffs_true
    covariance = np.diag(np.asarray([1.0, 4.0, 9.0, 16.0], dtype=np.float64))

    expected = run_stage8_weighted_lstsq_kernel(design, values, covariance=covariance, backend="python")
    observed = run_stage8_weighted_lstsq_kernel(design, values, covariance=covariance, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage8_weighted_lstsq_native_matches_python_full_covariance() -> None:
    design = np.asarray(
        [
            [1.0, -2.0],
            [1.0, 0.0],
            [1.0, 3.0],
            [1.0, 5.0],
        ],
        dtype=np.float64,
    )
    coeffs_true = np.asarray([[1.5, -2.0, 0.25], [0.5, 1.25, -0.75]], dtype=np.float64)
    values = design @ coeffs_true
    chol = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.3, 1.5, 0.0, 0.0],
            [0.2, -0.1, 1.2, 0.0],
            [0.1, 0.2, -0.3, 1.8],
        ],
        dtype=np.float64,
    )
    covariance = chol @ chol.T

    expected = run_stage8_weighted_lstsq_kernel(design, values, covariance=covariance, backend="python")
    observed = run_stage8_weighted_lstsq_kernel(design, values, covariance=covariance, backend="native")

    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage7_mean_velocity_fit_native_matches_python_reference() -> None:
    ph_mean_v = np.asarray(
        [
            [3.0, 0.0, -1.0, 1.0],
            [-2.0, 0.0, 4.0, 7.0],
            [0.5, -0.5, 2.0, 3.5],
        ],
        dtype=np.float64,
    )
    day = np.asarray([8.0, 10.0, 13.0, 17.0], dtype=np.float64)
    ifg_std = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)

    expected = ported._stage7_mean_velocity_fit(ph_mean_v, day, master_ix=2, ifg_std=ifg_std)
    observed = accel.run_stage7_mean_velocity_fit_kernel(
        ph_mean_v,
        day,
        master_ix=2,
        ifg_std=ifg_std,
        backend="native",
    )

    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage7_deramp_native_matches_python_reference_with_nans() -> None:
    xy = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1000.0, 0.0],
            [3.0, 0.0, 1000.0],
            [4.0, 1000.0, 1000.0],
            [5.0, 2000.0, 0.0],
            [6.0, 0.0, 2000.0],
            [7.0, 2000.0, 2000.0],
        ],
        dtype=np.float64,
    )
    x_km = xy[:, 1] / 1000.0
    y_km = xy[:, 2] / 1000.0
    ph = np.column_stack(
        (
            1.5 * x_km + 0.75 * y_km + 2.0,
            -0.5 * x_km + 1.25 * y_km - 1.0,
        )
    )
    ph[0, 1] = np.nan

    ps = {"n_ps": np.asarray(float(xy.shape[0])), "xy": xy}
    expected = ported._deramp_unwrapped_phase(ps, ph)
    observed = accel.run_stage7_deramp_unwrapped_phase_kernel(xy, ph, backend="native")

    np.testing.assert_allclose(observed[0], expected[0], atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-10, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_native_generic_matches_python_single_precision() -> None:
    rng = np.random.default_rng(21)
    cpxphase = np.exp(1j * rng.normal(size=(5, 7))).astype(np.complex64)
    bperp = rng.normal(size=(5, 7)).astype(np.float32)

    expected = ported._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps=1.5)
    observed = run_stage2_topofit_kernel(cpxphase, bperp, 1.5, backend="native")

    f32_tol = 8 * float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(observed[0], expected[0], atol=f32_tol, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=f32_tol, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected[2], atol=f32_tol, rtol=0.0)
    np.testing.assert_allclose(observed[3], expected[3], atol=1e-6, rtol=0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage2_native_generic_matches_python_single_precision_near_max_selector_regression() -> None:
    cpxphase = np.asarray(
        [
            [
                (0.9982544183731079 - 0.05906030535697937j),
                (0.8873175978660583 + 0.4611586332321167j),
                (0.29619884490966797 + 0.9551264047622681j),
                (-0.9712775945663452 + 0.23794908821582794j),
                (-0.2978763282299042 - 0.9546045064926147j),
                (0.5448974967002869 + 0.8385025262832642j),
                (-0.792346715927124 + 0.6100711226463318j),
                (-0.9800438284873962 - 0.19878160953521729j),
                (-0.07463288307189941 + 0.9972109794616699j),
                (-0.48219722509384155 + 0.87606281042099j),
                (-0.9998970031738281 - 0.014354166574776173j),
                (0.9977385401725769 - 0.06721504032611847j),
                (-0.09485628455877304 + 0.9954910278320312j),
                (-0.3512025773525238 + 0.9362995624542236j),
                (0.8777415752410889 + 0.4791341722011566j),
                (0.9811630249023438 + 0.19318196177482605j),
                (-0.11611422151327133 - 0.9932358860969543j),
                (0.8924177885055542 + 0.45121023058891296j),
                (0.09596839547157288 + 0.9953843355178833j),
                (0.9769843220710754 + 0.21331138908863068j),
                (-0.021037235856056213 + 0.9997786283493042j),
                (-0.8792001605033875 + 0.47645264863967896j),
                (-0.1141195148229599 - 0.9934670925140381j),
                (0.3829892873764038 - 0.9237527847290039j),
                (-0.9088584780693054 - 0.41710495948791504j),
                (-0.010275483131408691 - 0.9999473094940186j),
                (-0.9824351072311401 + 0.18660499155521393j),
                (-0.17854323983192444 - 0.9839321374893188j),
                (0.1761409044265747 - 0.984364926815033j),
                (0.3011814057826996 - 0.9535667300224304j),
                (0.9816787242889404 - 0.19054442644119263j),
                (0.7752916216850281 - 0.6316033005714417j),
                (0.9490712285041809 - 0.315062016248703j),
                (0.03413497656583786 + 0.9994171261787415j),
                (0.1550622135400772 + 0.9879047274589539j),
                (0.8024176955223083 + 0.5967625975608826j),
                (0.11122504621744156 + 0.9937950968742371j),
                (0.6744176149368286 + 0.7383500933647156j),
                (-0.9529107213020325 - 0.3032509386539459j),
                (0.6581352949142456 - 0.7528997659683228j),
                (-0.4477686285972595 + 0.8941494822502136j),
                (0.9615651369094849 - 0.27457690238952637j),
                (-0.6289923787117004 - 0.7774114012718201j),
                (0.8930615782737732 - 0.4499346911907196j),
                (0.08623586595058441 + 0.9962747693061829j),
                (-0.9984840154647827 - 0.055043138563632965j),
                (0.58107590675354 - 0.8138492703437805j),
                (0.5663127899169922 + 0.8241905570030212j),
                (-0.7192481756210327 - 0.6947533488273621j),
                (0.3608364164829254 + 0.9326291680335999j),
                (0.15415889024734497 - 0.9880460500717163j),
                (0.9999878406524658 - 0.004955730866640806j),
                (-0.874171793460846 + 0.4856167435646057j),
                (-0.8551686406135559 - 0.5183498859405518j),
                (-0.7429763674736023 - 0.669317364692688j),
                (-0.8908007740974426 - 0.45439407229423523j),
                (-0.5406952500343323 + 0.8412185907363892j),
                (0.5624164342880249 - 0.8268542289733887j),
                (0.7658286690711975 - 0.6430449485778809j),
                (0.688852071762085 - 0.7249021530151367j),
                (-0.6209292411804199 - 0.7838664650917053j),
                (0.10671449452638626 - 0.9942896366119385j),
                (-0.7070095539093018 - 0.7072041630744934j),
                (0.2713291347026825 + 0.9624866843223572j),
                (0.3476226329803467 - 0.9376345276832581j),
                (0.1296563446521759 + 0.991558849811554j),
                (0.3715232312679291 + 0.9284237027168274j),
                (0.993870735168457 + 0.11054952442646027j),
                (0.8484829664230347 - 0.5292226672172546j),
                (-0.5058281421661377 - 0.8626341819763184j),
                (0.3247547447681427 - 0.9457983374595642j),
                (0.9948955774307251 - 0.10090969502925873j),
                (-0.30419793725013733 - 0.9526088833808899j),
                (0.9999990463256836 + 0.0013774563558399677j),
                (0.9995426535606384 - 0.03023890033364296j),
            ]
        ],
        dtype=np.complex64,
    )
    bperp = np.asarray(
        [
            [
                -354.87603759765625,
                -148.41204833984375,
                11.849981307983398,
                -143.1541290283203,
                -32.72481155395508,
                -25.145904541015625,
                -78.75508117675781,
                -264.9224548339844,
                -76.25702667236328,
                -51.60499954223633,
                -27.05449104309082,
                -60.93518829345703,
                -136.94781494140625,
                -36.86498260498047,
                -70.62962341308594,
                -105.08844757080078,
                -55.63603210449219,
                -184.31182861328125,
                -77.8426742553711,
                -59.77418899536133,
                -166.5838623046875,
                -227.78106689453125,
                -140.2783966064453,
                -80.2271728515625,
                -75.05155944824219,
                -99.26610565185547,
                -116.754150390625,
                -187.2490692138672,
                -142.78953552246094,
                -70.37977600097656,
                -94.35164642333984,
                -80.41920471191406,
                -104.22224426269531,
                -335.36883544921875,
                24.38962745666504,
                187.44175720214844,
                94.51117706298828,
                177.63790893554688,
                155.34625244140625,
                113.44317626953125,
                213.68711853027344,
                103.27958679199219,
                121.45433044433594,
                -64.13392639160156,
                -43.7103271484375,
                -86.97090148925781,
                -50.98783874511719,
                -162.53964233398438,
                -298.62713623046875,
                -383.17041015625,
                -103.06913757324219,
                50.384891510009766,
                -6.7172770500183105,
                -41.386409759521484,
                -65.24752044677734,
                -100.1313705444336,
                -9.320809364318848,
                45.66149139404297,
                2.6101229190826416,
                -203.9463653564453,
                -195.20115661621094,
                -32.57182693481445,
                65.24066162109375,
                -14.017067909240723,
                -36.665836334228516,
                49.27800369262695,
                149.13890075683594,
                230.8185577392578,
                268.7464599609375,
                203.97015380859375,
                179.36856079101562,
                143.6444549560547,
                121.03768920898438,
                110.94869995117188,
                -41.953399658203125,
            ]
        ],
        dtype=np.float32,
    )

    expected = ported._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps=0.725669801235199)
    observed = run_stage2_topofit_kernel(cpxphase, bperp, 0.725669801235199, backend="native")

    np.testing.assert_allclose(observed[0], expected[0], atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(observed[2], expected[2], atol=1e-7, rtol=0.0)
    np.testing.assert_allclose(observed[3], expected[3], atol=1e-6, rtol=0.0)
