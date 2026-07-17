from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import signal

from pystamps.config import ConfigError, normalize_stage2_kernel_backend
from pystamps.kernels.registry import DEFAULT_REGISTRY, KernelResolutionError
from pystamps.runtime.resources import cpu_budget

_STAGE8_NOISE_SCALE = np.float32(0.5)
_STAGE2_NATIVE_MODULE: Any | None = None
_STAGE2_NATIVE_IMPORT_ATTEMPTED = False


class BackendUnavailableError(RuntimeError):
    """Raised when a requested compute backend is not available."""


def _cupy() -> Any | None:
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:
        return None


def _cuda_available() -> bool:
    return _cupy() is not None


def _resolve_backend(backend: str) -> str:
    name = (backend or "auto").strip().lower()
    if name in {"auto"}:
        return "auto"
    if name in {"threads", "thread", "io", "processes", "process", "cpu", "python"}:
        return "python"
    if name in {"native"}:
        return "native"
    if name in {"gpu", "cuda"}:
        return "cuda"
    raise BackendUnavailableError(
        f"Unsupported kernel backend '{backend}'. Use: auto, python, native, or cuda"
    )


def _resolve_stage2_kernel_backend(backend: str) -> str:
    try:
        return normalize_stage2_kernel_backend(backend)
    except ConfigError as exc:
        raise BackendUnavailableError(str(exc)) from exc


def _to_numpy(arr: Any) -> np.ndarray:
    cp = _cupy()
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _cov_from_accumulators(sum_res: np.ndarray, sum_outer: np.ndarray, count: int) -> np.ndarray:
    k = int(sum_res.size)
    if k == 0:
        return np.empty((0, 0), dtype=np.float64)
    if count <= 1:
        return np.zeros((k, k), dtype=np.float64)
    mean = sum_res / float(count)
    cov = (sum_outer - float(count) * np.outer(mean, mean)) / float(count - 1)
    return cov.astype(np.float64)


def _auto_chunk_size(total_rows: int, width_hint: int, itemsize: int, target_bytes: int = 128 * 1024 * 1024) -> int:
    if total_rows <= 0:
        return 1
    width = max(1, int(width_hint))
    bytes_per_row = max(1, width * max(1, int(itemsize)))
    chunk = max(1, target_bytes // bytes_per_row)
    return max(1, min(int(total_rows), int(chunk)))


def _load_stage2_native_module() -> Any | None:
    global _STAGE2_NATIVE_IMPORT_ATTEMPTED, _STAGE2_NATIVE_MODULE
    if _STAGE2_NATIVE_MODULE is not None:
        return _STAGE2_NATIVE_MODULE
    if _STAGE2_NATIVE_IMPORT_ATTEMPTED:
        # Re-check after failed loads so same-process rebuilds become visible.
        importlib.invalidate_caches()
    _STAGE2_NATIVE_IMPORT_ATTEMPTED = True
    try:
        _STAGE2_NATIVE_MODULE = importlib.import_module("pystamps.kernels._stage2_native")
    except Exception:
        _STAGE2_NATIVE_MODULE = None
    return _STAGE2_NATIVE_MODULE


def _native_export(name: str) -> Any | None:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        return None
    fn = getattr(native_mod, str(name), None)
    return fn if callable(fn) else None


def _native_threads(threads: int = 0) -> int:
    requested = int(threads)
    if requested > 0:
        return requested
    return cpu_budget()


def stage2_native_available() -> bool:
    return _load_stage2_native_module() is not None


def stage2_ph_weight_block_native_available() -> bool:
    return _native_export("stage2_ph_weight_block") is not None


def stage2_grid_indices_native_available() -> bool:
    return _native_export("stage2_grid_indices") is not None


def stage2_clap_filter_kernel_native_available() -> bool:
    return _native_export("stage2_clap_filter_kernel") is not None


def stage2_normalize_complex_native_available() -> bool:
    return _native_export("stage2_normalize_complex") is not None


def stage2_normalize_phase_matrix_native_available() -> bool:
    return _native_export("stage2_normalize_phase_matrix") is not None


def stage4_native_available() -> bool:
    return _native_export("stage4_edge_stats") is not None


def stage4_duplicate_keep_native_available() -> bool:
    return _native_export("stage4_duplicate_keep") is not None


def stage4_adjacent_component_keep_native_available() -> bool:
    return _native_export("stage4_adjacent_component_keep") is not None


def stage4_weed_ifg_index_native_available() -> bool:
    return _native_export("stage4_weed_ifg_index") is not None


def stage4_phase_correction_native_available() -> bool:
    return _native_export("stage4_phase_correction") is not None


def stage3_select_ifg_index_native_available() -> bool:
    return _native_export("stage3_select_ifg_index") is not None


def stage3_clap_filt_patch_native_available() -> bool:
    return _native_export("stage3_clap_filt_patch") is not None


def stage3_clap_filt_patch_stack_native_available() -> bool:
    return _native_export("stage3_clap_filt_patch_stack") is not None


def stage3_clap_filt_grid_native_available() -> bool:
    return _native_export("stage3_clap_filt_grid") is not None


def stage3_clap_filt_grid_stack_native_available() -> bool:
    return _native_export("stage3_clap_filt_grid_stack") is not None


def stage3_wrap_filt_native_available() -> bool:
    return _native_export("stage3_wrap_filt") is not None


def stage3_wrap_filt_global_native_available() -> bool:
    return _native_export("stage3_wrap_filt_global") is not None


def stage3_coh_threshold_native_available() -> bool:
    return _native_export("stage3_coh_threshold") is not None


def stage5_native_available() -> bool:
    return _native_export("stage5_ifg_std") is not None


def stage5_format_merged_rc2_native_available() -> bool:
    return _native_export("stage5_format_merged_rc2") is not None


def stage5_duplicate_keep_native_available() -> bool:
    return _native_export("stage5_duplicate_keep") is not None


def stage5_patch_keep_mask_native_available() -> bool:
    return _native_export("stage5_patch_keep_mask") is not None


def stage5_rc2_correction_native_available() -> bool:
    return _native_export("stage5_rc2_correction") is not None


def stage6_native_available() -> bool:
    return _native_export("stage6_unwrap_grid") is not None


def stage6_extract_grid_values_native_available() -> bool:
    return _native_export("stage6_extract_grid_values") is not None


def stage6_prepare_cost_offsets_native_available() -> bool:
    return _native_export("stage6_prepare_cost_offsets") is not None


def stage6_reconstruct_ps_phase_native_available() -> bool:
    return _native_export("stage6_reconstruct_ps_phase") is not None


def stage6_ps_grid_indices_native_available() -> bool:
    return _native_export("stage6_ps_grid_indices") is not None


def stage6_select_ifgw_native_available() -> bool:
    return _native_export("stage6_select_ifgw") is not None


def stage6_grid_accumulate_native_available() -> bool:
    return _native_export("stage6_grid_accumulate") is not None


def stage6_unwrap_ifg_sets_native_available() -> bool:
    return _native_export("stage6_unwrap_ifg_sets") is not None


def stage6_single_master_ifg_geometry_native_available() -> bool:
    return _native_export("stage6_single_master_ifg_geometry") is not None


def stage6_estimate_la_error_native_available() -> bool:
    return _native_export("stage6_estimate_la_error_single_master") is not None


def stage6_smooth_3d_full_single_master_native_available() -> bool:
    return _native_export("stage6_smooth_3d_full_single_master") is not None


def stage7_native_available() -> bool:
    return _native_export("stage7_scla_parity") is not None


def stage7_mean_velocity_fit_native_available() -> bool:
    return _native_export("stage7_mean_velocity_fit") is not None


def stage7_deramp_unwrapped_phase_native_available() -> bool:
    return _native_export("stage7_deramp_unwrapped_phase") is not None


def stage7_center_to_reference_native_available() -> bool:
    return _native_export("stage7_center_to_reference") is not None


def stage7_scla_smooth_native_available() -> bool:
    return _native_export("stage7_scla_smooth") is not None


def stage8_native_available() -> bool:
    return _native_export("stage8_edge_noise") is not None


def stage8_weighted_lstsq_native_available() -> bool:
    return _native_export("stage8_weighted_lstsq_diagonal") is not None


def weighted_affine_fit_native_available() -> bool:
    return _native_export("weighted_affine_fit") is not None


def weighted_slope_fit_native_available() -> bool:
    return _native_export("weighted_slope_fit_real") is not None and _native_export("weighted_slope_fit_complex") is not None


def _resolve_stage2_kernel(
    name: str,
    backend: str,
    *,
    implementations: dict[str, Any] | None = None,
) -> Any:
    requested = _resolve_stage2_kernel_backend(backend)
    try:
        return DEFAULT_REGISTRY.resolve(
            name,
            requested="auto" if requested == "auto" else requested,
            fallback_order=("native", "python") if requested == "auto" else (),
            strict_requested=requested != "auto",
            implementations=implementations,
        )
    except KernelResolutionError as exc:
        raise BackendUnavailableError(str(exc)) from exc


def _resolve_generic_kernel(
    name: str,
    backend: str,
    *,
    auto_order: tuple[str, ...] = ("python",),
    explicit_fallbacks: dict[str, tuple[str, ...]] | None = None,
    implementations: dict[str, Any] | None = None,
) -> Any:
    requested = _resolve_backend(backend)
    fallback_map = explicit_fallbacks or {}
    try:
        if requested == "auto":
            return DEFAULT_REGISTRY.resolve(
                name,
                requested="auto",
                fallback_order=auto_order,
                implementations=implementations,
            )
        return DEFAULT_REGISTRY.resolve(
            name,
            requested=requested,
            fallback_order=fallback_map.get(requested, ()),
            strict_requested=requested not in fallback_map,
            implementations=implementations,
        )
    except KernelResolutionError as exc:
        raise BackendUnavailableError(str(exc)) from exc


def _ported_stage2_module() -> Any:
    from pystamps.pipeline import ported

    return ported


def _stage2_grid_accumulate_cpu(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    ph = np.asarray(ph_weight, dtype=np.complex64)
    grid = np.asarray(grid_lin, dtype=np.int64).reshape(-1)
    if out is None:
        grid_out = np.zeros((int(n_i), int(n_j), ph.shape[1]), dtype=np.complex64)
    else:
        grid_out = out
        grid_out.fill(0)
    for i_ifg in range(ph.shape[1]):
        flat = grid_out[:, :, i_ifg].reshape(-1)
        np.add.at(flat, grid, ph[:, i_ifg])
    return grid_out


def _stage2_grid_accumulate_python(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    threads: int = 0,
) -> np.ndarray:
    return _stage2_grid_accumulate_cpu(ph_weight, grid_lin, n_i, n_j)


def _stage2_grid_accumulate_native(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    threads: int = 0,
) -> np.ndarray:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    return np.asarray(
        native_mod.accumulate_weighted_grid(
            np.ascontiguousarray(ph_weight, dtype=np.complex64),
            np.ascontiguousarray(np.asarray(grid_lin, dtype=np.int64).reshape(-1)),
            int(n_i),
            int(n_j),
            max(1, int(threads)) if int(threads) > 0 else 1,
        ),
        dtype=np.complex64,
    )


def _stage2_grid_indices_python(xy: np.ndarray, grid_size: float, threads: int = 0) -> np.ndarray:
    del threads
    xy32 = np.asarray(xy, dtype=np.float32)
    x = xy32[:, 1]
    y = xy32[:, 2]
    grid_scale = np.float32(grid_size)
    eps = np.float32(1e-6)

    grid_i = np.ceil((y - np.min(y) + eps) / grid_scale).astype(np.int64)
    grid_j = np.ceil((x - np.min(x) + eps) / grid_scale).astype(np.int64)
    if np.max(grid_i) > 1:
        grid_i[grid_i == np.max(grid_i)] = np.max(grid_i) - 1
    if np.max(grid_j) > 1:
        grid_j[grid_j == np.max(grid_j)] = np.max(grid_j) - 1
    grid_i[grid_i < 1] = 1
    grid_j[grid_j < 1] = 1
    return np.column_stack((grid_i, grid_j)).astype(np.float32)


def _stage2_grid_indices_native(xy: np.ndarray, grid_size: float, threads: int = 0) -> np.ndarray:
    native_fn = _native_export("stage2_grid_indices")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage2_grid_indices but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(xy, dtype=np.float32)),
            np.float32(grid_size),
            int(threads),
        ),
        dtype=np.float32,
    )


def _stage2_clap_filter_kernel_python(threads: int = 0) -> np.ndarray:
    del threads
    alpha = 2.5
    std = (7 - 1) / (2.0 * alpha)
    center = (7 - 1) / 2.0
    x = (np.arange(7, dtype=np.float64) - center) / std
    g = np.exp(-0.5 * x * x)
    return np.outer(g, g).astype(np.float64)


def _stage2_clap_filter_kernel_native(threads: int = 0) -> np.ndarray:
    native_fn = _native_export("stage2_clap_filter_kernel")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage2_clap_filter_kernel but the compiled extension does not export it"
        )
    return np.asarray(native_fn(int(threads)), dtype=np.float64)


def _stage2_normalize_complex_python(
    values: np.ndarray,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    del threads
    out_arr = np.asarray(values)
    work_dtype = np.complex128 if preserve_precision else np.complex64
    if out_arr.dtype == work_dtype:
        work_arr = out_arr.copy()
    else:
        work_arr = out_arr.astype(work_dtype, copy=True)
    abs_arr = np.abs(work_arr).astype(np.float64 if preserve_precision else np.float32, copy=False)
    np.divide(work_arr, abs_arr, out=work_arr, where=abs_arr != 0)
    return work_arr.astype(out_arr.dtype, copy=False) if out_arr.dtype != work_dtype else work_arr


def _stage2_normalize_complex_native(
    values: np.ndarray,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    if preserve_precision:
        return _stage2_normalize_complex_python(values, preserve_precision, threads)
    native_fn = _native_export("stage2_normalize_complex")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage2_normalize_complex but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(values, dtype=np.complex64)),
            int(threads),
        ),
        dtype=np.complex64,
    )


def _stage2_normalize_phase_matrix_python(ph_nm: np.ndarray, threads: int = 0) -> dict[str, np.ndarray]:
    del threads
    ph = np.asarray(ph_nm, dtype=np.complex64)
    amp = np.abs(ph).astype(np.float32)
    amp[amp == 0] = 1.0
    ph_out = np.divide(ph, amp, out=np.zeros_like(ph), where=amp != 0).astype(np.complex64)
    return {"ph": ph_out, "amp": amp}


def _stage2_normalize_phase_matrix_native(ph_nm: np.ndarray, threads: int = 0) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage2_normalize_phase_matrix")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage2_normalize_phase_matrix but the compiled extension does not export it"
        )
    payload = native_fn(
        np.ascontiguousarray(np.asarray(ph_nm, dtype=np.complex64)),
        int(threads),
    )
    return {
        "ph": np.asarray(payload["ph"], dtype=np.complex64),
        "amp": np.asarray(payload["amp"], dtype=np.float32),
    }


def _stage2_ph_weight_block_python(
    ph_nm: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    weighting: np.ndarray,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    del threads
    ph_chunk = np.asarray(ph_nm, dtype=np.complex64)
    bp_chunk = np.asarray(bperp, dtype=np.float64)
    k_chunk = np.asarray(k_ps, dtype=np.float64).reshape(-1, 1)
    weight_chunk = np.asarray(weighting, dtype=np.float64).reshape(-1, 1)
    phase_ramp = np.exp(-1j * (bp_chunk * k_chunk))
    out = ph_chunk.astype(np.complex128) * phase_ramp
    out = out * weight_chunk
    if preserve_precision:
        return out
    return out.astype(np.complex64, copy=False)


def _stage2_ph_weight_block_native(
    ph_nm: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    weighting: np.ndarray,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    if preserve_precision:
        return _stage2_ph_weight_block_python(ph_nm, bperp, k_ps, weighting, preserve_precision, threads)
    native_fn = _native_export("stage2_ph_weight_block")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage2_ph_weight_block but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph_nm, dtype=np.complex64)),
            np.ascontiguousarray(np.asarray(bperp, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(k_ps, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(weighting, dtype=np.float64).reshape(-1)),
            int(threads),
        ),
        dtype=np.complex64,
    )


def _stage2_histogram_with_centers_cpu(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    bins = np.asarray(centers, dtype=np.float64).reshape(-1)
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if bins.size == 0:
        return np.asarray([], dtype=np.float64)
    if bins.size == 1:
        return np.asarray([float(samples.size)], dtype=np.float64)
    diffs = np.diff(bins)
    equal_spacing = bool(
        np.all(np.abs(diffs - diffs[0]) <= np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(bins)))))
    )
    if equal_spacing:
        d = 1.0 if bins.size < 3 else float((bins[-1] - bins[0]) / (bins.size - 1))
        cutoff0 = float((bins[0] + bins[1]) / 2.0)
        assignments = 1 + np.maximum(
            0.0,
            np.minimum(
                float(bins.size - 1),
                np.ceil((samples - cutoff0) / d),
            ),
        )
        return np.bincount(assignments.astype(np.int64) - 1, minlength=bins.size).astype(np.float64)
    mids = (bins[:-1] + bins[1:]) / 2.0
    assignments = np.searchsorted(mids, samples, side="left")
    assignments = np.clip(assignments, 0, bins.size - 1)
    return np.bincount(assignments, minlength=bins.size).astype(np.float64)


def _stage2_histogram_with_centers_python(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    return _stage2_histogram_with_centers_cpu(values, centers)


def _stage2_histogram_with_centers_native(
    values: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    return np.asarray(
        native_mod.histogram_with_centers(
            np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(centers, dtype=np.float64).reshape(-1)),
        ),
        dtype=np.float64,
    )


def _stage2_row_invariant_bperp_matrix(
    bperp: np.ndarray,
    n_row: int,
) -> tuple[np.ndarray, np.ndarray]:
    bp = np.asarray(bperp)
    real_dtype = np.float32 if bp.dtype == np.float32 else np.float64
    if bp.ndim == 1:
        bp_vec = np.ascontiguousarray(bp.reshape(-1), dtype=real_dtype)
        bp_mat = np.broadcast_to(bp_vec, (int(n_row), bp_vec.size)).astype(real_dtype, copy=True)
        return bp_vec, bp_mat
    if bp.ndim == 2:
        if bp.shape[0] not in {1, int(n_row)}:
            raise BackendUnavailableError(
                f"Row-invariant stage-2 bperp has incompatible shape {bp.shape} for n_row={n_row}"
            )
        bp_vec = np.ascontiguousarray(bp.reshape(-1) if bp.shape[0] == 1 else bp[0, :], dtype=real_dtype)
        if bp.shape[0] == int(n_row):
            return bp_vec, np.ascontiguousarray(bp, dtype=real_dtype)
        bp_mat = np.broadcast_to(bp_vec, (int(n_row), bp_vec.size)).astype(real_dtype, copy=True)
        return bp_vec, bp_mat
    raise BackendUnavailableError("Row-invariant stage-2 bperp must be a 1-D vector or 2-D matrix")


def _stage2_topofit_python(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _ported_stage2_module()._ps_topofit_batch_generic(cpxphase, bperp, n_trial_wraps)


def _stage2_topofit_native(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    bperp_arr = np.asarray(bperp)
    if bperp_arr.ndim == 1:
        return _stage2_topofit_row_invariant_native(cpxphase, bperp_arr, n_trial_wraps, threads)
    if bperp_arr.ndim == 2 and bperp_arr.shape[0] > 0:
        row0 = np.ascontiguousarray(bperp_arr[0], dtype=bperp_arr.dtype)
        if np.array_equal(bperp_arr, np.broadcast_to(row0, bperp_arr.shape)):
            return _stage2_topofit_row_invariant_native(cpxphase, row0, n_trial_wraps, threads)
    cpx_arr = np.asarray(cpxphase)
    use_single = cpx_arr.dtype == np.complex64 or bperp_arr.dtype == np.float32
    if use_single:
        K0, C0, coh0, phase_residual = native_mod.ps_topofit_batch_generic_f32(
            np.ascontiguousarray(cpx_arr, dtype=np.complex64),
            np.ascontiguousarray(bperp_arr, dtype=np.float32),
            float(n_trial_wraps),
            _native_threads(threads),
        )
    else:
        K0, C0, coh0, phase_residual = native_mod.ps_topofit_batch_generic(
            np.ascontiguousarray(cpx_arr, dtype=np.complex128),
            np.ascontiguousarray(bperp_arr, dtype=np.float64),
            float(n_trial_wraps),
            _native_threads(threads),
        )
    return (
        np.asarray(K0, dtype=np.float64),
        np.asarray(C0, dtype=np.float64),
        np.asarray(coh0, dtype=np.float64),
        np.asarray(phase_residual, dtype=np.complex64),
    )


def _stage2_topofit_row_invariant_python(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
    return _ported_stage2_module()._ps_topofit_batch_row_invariant(cpxphase, bperp_mat, n_trial_wraps)


def _stage2_topofit_row_invariant_native(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    bp_vec, _ = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
    K0, C0, coh0, phase_residual = native_mod.ps_topofit_batch_row_invariant(
        np.ascontiguousarray(cpxphase, dtype=np.complex128),
        np.ascontiguousarray(bp_vec, dtype=np.float64),
        float(n_trial_wraps),
        _native_threads(threads),
    )
    return (
        np.asarray(K0, dtype=np.float64),
        np.asarray(C0, dtype=np.float64),
        np.asarray(coh0, dtype=np.float64),
        np.asarray(phase_residual, dtype=np.complex64),
    )


def _stage2_topofit_coh_row_invariant_python(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> np.ndarray:
    _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
    return np.asarray(
        _ported_stage2_module()._ps_topofit_batch_row_invariant_coh(cpxphase, bperp_mat, n_trial_wraps),
        dtype=np.float64,
    )


def _stage2_topofit_coh_row_invariant_native(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> np.ndarray:
    native_mod = _load_stage2_native_module()
    if native_mod is None:
        raise BackendUnavailableError("Native stage-2 extension is unavailable")
    bp_vec, _ = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
    return np.asarray(
        native_mod.ps_topofit_coh_row_invariant(
            np.ascontiguousarray(cpxphase, dtype=np.complex128),
            np.ascontiguousarray(bp_vec, dtype=np.float64),
            float(n_trial_wraps),
            _native_threads(threads),
        ),
        dtype=np.float64,
    )


DEFAULT_REGISTRY.register_provider(
    "python",
    description="NumPy/Python baseline backend",
    aliases=("cpu",),
)
DEFAULT_REGISTRY.register_provider(
    "native",
    description="Compiled native backend",
    availability_probe=stage2_native_available,
    unavailable_reason="Native stage-2 extension is unavailable",
)
DEFAULT_REGISTRY.register_provider(
    "cuda",
    description="CuPy CUDA backend",
    aliases=("gpu",),
    availability_probe=_cuda_available,
    unavailable_reason="GPU backend requested but CuPy is not available",
)
DEFAULT_REGISTRY.register("stage2_grid_accumulate", python=_stage2_grid_accumulate_python, native=_stage2_grid_accumulate_native)
DEFAULT_REGISTRY.register(
    "stage2_grid_indices",
    python=_stage2_grid_indices_python,
    native=_stage2_grid_indices_native,
)
DEFAULT_REGISTRY.register(
    "stage2_clap_filter_kernel",
    python=_stage2_clap_filter_kernel_python,
    native=_stage2_clap_filter_kernel_native,
)
DEFAULT_REGISTRY.register(
    "stage2_normalize_complex",
    python=_stage2_normalize_complex_python,
    native=_stage2_normalize_complex_native,
)
DEFAULT_REGISTRY.register(
    "stage2_normalize_phase_matrix",
    python=_stage2_normalize_phase_matrix_python,
    native=_stage2_normalize_phase_matrix_native,
)
DEFAULT_REGISTRY.register(
    "stage2_ph_weight_block",
    python=_stage2_ph_weight_block_python,
    native=_stage2_ph_weight_block_native,
)
DEFAULT_REGISTRY.register("stage2_topofit", python=_stage2_topofit_python, native=_stage2_topofit_native)
DEFAULT_REGISTRY.register(
    "stage2_topofit_row_invariant",
    python=_stage2_topofit_row_invariant_python,
    native=_stage2_topofit_row_invariant_native,
)
DEFAULT_REGISTRY.register(
    "stage2_topofit_coh_row_invariant",
    python=_stage2_topofit_coh_row_invariant_python,
    native=_stage2_topofit_coh_row_invariant_native,
)
DEFAULT_REGISTRY.register(
    "stage2_histogram",
    python=_stage2_histogram_with_centers_python,
    native=_stage2_histogram_with_centers_native,
)


def _stage3_select_ifg_index_python(
    n_ifg: int,
    master_ix: int,
    drop_ifg_index: np.ndarray,
    small_baseline: bool,
    threads: int = 0,
) -> np.ndarray:
    del threads
    drop = set(int(v) for v in np.asarray(drop_ifg_index, dtype=np.int64).reshape(-1).tolist())
    ifg = [i for i in range(1, int(n_ifg) + 1) if i not in drop]
    if not bool(small_baseline):
        master = int(master_ix)
        ifg = [i for i in ifg if i != master]
        ifg = [i - 1 if i > master else i for i in ifg]
    return np.asarray(ifg, dtype=np.float64)


def _stage3_select_ifg_index_native(
    n_ifg: int,
    master_ix: int,
    drop_ifg_index: np.ndarray,
    small_baseline: bool,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage3_select_ifg_index")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage3_select_ifg_index but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            int(n_ifg),
            int(master_ix),
            np.ascontiguousarray(np.asarray(drop_ifg_index, dtype=np.int64).reshape(-1)),
            bool(small_baseline),
            int(threads),
        ),
        dtype=np.float64,
    )


def _stage3_clap_filt_patch_python(
    ph: np.ndarray,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    ph_arr = np.asarray(ph, dtype=np.complex128).copy()
    ph_arr[np.isnan(ph_arr)] = 0
    ph_fft = np.fft.fft2(ph_arr)
    h = np.abs(ph_fft)
    b = _stage2_clap_filter_kernel_python()
    h = np.fft.ifftshift(signal.convolve2d(np.fft.fftshift(h), b, mode="same", boundary="fill", fillvalue=0.0))
    mean_h = float(np.median(h))
    if mean_h != 0.0:
        h = h / mean_h
    h = np.power(h, float(alpha))
    h = h - 1.0
    h[h < 0.0] = 0.0
    g = h * float(beta) + np.asarray(low_pass, dtype=np.float64)
    return np.fft.ifft2(ph_fft * g)


def _stage3_clap_filt_patch_native(
    ph: np.ndarray,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage3_clap_filt_patch")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage3_clap_filt_patch but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph, dtype=np.complex128)),
            float(alpha),
            float(beta),
            np.ascontiguousarray(np.asarray(low_pass, dtype=np.float64)),
            int(threads),
        ),
        dtype=np.complex128,
    )


def _stage3_clap_filt_patch_stack_python(
    ph_stack: np.ndarray,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    ph_arr = np.asarray(ph_stack)
    if ph_arr.ndim != 3:
        raise ValueError("clap_filt_patch_stack expects a 3-D complex stack")
    out = np.empty(ph_arr.shape, dtype=np.complex128)
    for i_ifg in range(ph_arr.shape[2]):
        out[:, :, i_ifg] = _stage3_clap_filt_patch_python(
            ph_arr[:, :, i_ifg],
            alpha,
            beta,
            low_pass,
        )
    return out


def _stage3_clap_filt_patch_stack_native(
    ph_stack: np.ndarray,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage3_clap_filt_patch_stack")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage3_clap_filt_patch_stack but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph_stack, dtype=np.complex128)),
            float(alpha),
            float(beta),
            np.ascontiguousarray(np.asarray(low_pass, dtype=np.float64)),
            int(threads),
        ),
        dtype=np.complex128,
    )


def _stage3_clap_filt_grid_python(
    ph: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    del threads
    out_dtype = np.complex128 if preserve_precision else np.complex64
    ph_arr = np.asarray(ph, dtype=np.complex128 if preserve_precision else np.complex64).copy()
    if ph_arr.ndim != 2:
        raise ValueError("clap_filt_grid expects a 2-D complex grid")

    n_win_int = int(round(n_win))
    if n_win_int <= 0:
        n_win_int = 32
    n_pad_int = int(round(n_pad))
    n_i, n_j = ph_arr.shape
    ph_out = np.zeros((n_i, n_j), dtype=np.complex128)
    n_inc = max(1, n_win_int // 4)
    n_win_i = int(np.ceil(n_i / float(n_inc)) - 3)
    n_win_j = int(np.ceil(n_j / float(n_inc)) - 3)
    if n_win_i <= 0 or n_win_j <= 0:
        return ph_out.astype(out_dtype, copy=False)

    x = np.arange(0, n_win_int // 2, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="xy")
    wind_func = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0) + 1e-6

    ph_arr[np.isnan(ph_arr)] = 0
    n_win_ex = n_win_int + n_pad_int
    low_pass_use = (
        np.zeros((n_win_ex, n_win_ex), dtype=np.float64)
        if low_pass is None
        else np.asarray(low_pass, dtype=np.float64)
    )
    ph_bit = np.zeros((n_win_ex, n_win_ex), dtype=np.complex128)
    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win_int
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win_int
            wf = np.vstack((np.zeros((i_shift, n_win_int), dtype=np.float64), wf[: n_win_int - i_shift, :]))
        for ix2 in range(n_win_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win_int
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win_int
                wf2 = np.hstack((np.zeros((n_win_int, j_shift), dtype=np.float64), wf2[:, : n_win_int - j_shift]))
            ph_bit.fill(0)
            ph_bit[:n_win_int, :n_win_int] = ph_arr[i1:i2, j1:j2]
            ph_filt = _stage3_clap_filt_patch_python(
                ph_bit,
                alpha=alpha,
                beta=beta,
                low_pass=low_pass_use,
            )
            ph_out[i1:i2, j1:j2] += ph_filt[:n_win_int, :n_win_int] * wf2
    return ph_out.astype(out_dtype, copy=False)


def _stage3_clap_filt_grid_native(
    ph: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage3_clap_filt_grid")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage3_clap_filt_grid but the compiled extension does not export it"
        )
    n_win_int = int(round(n_win))
    if n_win_int <= 0:
        n_win_int = 32
    n_pad_int = int(round(n_pad))
    n_win_ex = n_win_int + n_pad_int
    low_pass_use = (
        np.zeros((n_win_ex, n_win_ex), dtype=np.float64)
        if low_pass is None
        else np.ascontiguousarray(np.asarray(low_pass, dtype=np.float64))
    )
    ph_dtype = np.complex128 if preserve_precision else np.complex64
    ph_use = np.ascontiguousarray(np.asarray(ph, dtype=ph_dtype).astype(np.complex128, copy=False))
    out = np.asarray(
        native_fn(
            ph_use,
            float(alpha),
            float(beta),
            int(n_win_int),
            int(n_pad_int),
            low_pass_use,
            int(threads),
        ),
        dtype=np.complex128,
    )
    return out.astype(np.complex128 if preserve_precision else np.complex64, copy=False)


def _stage3_clap_filt_grid_stack_python(
    ph_stack: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    ph_arr = np.asarray(ph_stack)
    if ph_arr.ndim != 3:
        raise ValueError("clap_filt_grid_stack expects a 3-D complex stack")
    out_dtype = np.complex128 if preserve_precision else np.complex64
    out = np.empty(ph_arr.shape, dtype=out_dtype)
    for i_ifg in range(ph_arr.shape[2]):
        out[:, :, i_ifg] = _stage3_clap_filt_grid_python(
            ph_arr[:, :, i_ifg],
            alpha=alpha,
            beta=beta,
            n_win=n_win,
            n_pad=n_pad,
            low_pass=low_pass,
            preserve_precision=preserve_precision,
            threads=threads,
        )
    return out


def _stage3_clap_filt_grid_stack_native(
    ph_stack: np.ndarray,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
    preserve_precision: bool = False,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage3_clap_filt_grid_stack")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage3_clap_filt_grid_stack but the compiled extension does not export it"
        )
    n_win_int = int(round(n_win))
    if n_win_int <= 0:
        n_win_int = 32
    n_pad_int = int(round(n_pad))
    n_win_ex = n_win_int + n_pad_int
    low_pass_use = (
        np.zeros((n_win_ex, n_win_ex), dtype=np.float64)
        if low_pass is None
        else np.ascontiguousarray(np.asarray(low_pass, dtype=np.float64))
    )
    ph_dtype = np.complex128 if preserve_precision else np.complex64
    ph_use = np.ascontiguousarray(np.asarray(ph_stack, dtype=ph_dtype).astype(np.complex128, copy=False))
    out = np.asarray(
        native_fn(
            ph_use,
            float(alpha),
            float(beta),
            int(n_win_int),
            int(n_pad_int),
            low_pass_use,
            int(threads),
        ),
        dtype=np.complex128,
    )
    return out.astype(np.complex128 if preserve_precision else np.complex64, copy=False)


def _stage3_wrap_filt_python(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    del threads
    ph_arr = np.asarray(ph, dtype=np.complex64).copy()
    if ph_arr.ndim != 2:
        raise ValueError("wrap_filt expects a 2-D complex grid")
    n_win_i = int(round(n_win))
    if n_win_i <= 1:
        raise ValueError("wrap_filt window must be > 1")
    n_pad_i = max(0, int(round(n_pad)))
    n_i, n_j = ph_arr.shape
    n_inc = max(1, int(np.floor(n_win_i / 2.0)))
    n_win_blocks_i = int(np.ceil(n_i / n_inc) - 1)
    n_win_blocks_j = int(np.ceil(n_j / n_inc) - 1)
    ph_out = np.zeros_like(ph_arr, dtype=np.complex64)
    want_low = str(low_flag).lower() == "y"
    ph_out_low = np.zeros_like(ph_arr, dtype=np.complex64) if want_low else None

    x = np.arange(1, n_win_i // 2 + 1, dtype=np.float64)
    X, Y = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.concatenate((X, np.fliplr(X)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0).astype(np.float64)

    ph_arr[np.isnan(ph_arr)] = 0
    gauss = _ported_stage2_module()._gausswin
    b = np.outer(gauss(7), gauss(7))
    ph_bit = np.zeros((n_win_i + n_pad_i, n_win_i + n_pad_i), dtype=np.complex64)
    low_filter = None
    if want_low:
        g = gauss(n_win_i + n_pad_i, alpha=16.0)
        low_filter = np.fft.ifftshift(np.outer(g, g))

    for ix1 in range(n_win_blocks_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win_i
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win_i
            wf = np.vstack((np.zeros((i_shift, n_win_i), dtype=np.float64), wf[: n_win_i - i_shift, :]))
        for ix2 in range(n_win_blocks_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win_i
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win_i
                wf2 = np.hstack((np.zeros((n_win_i, j_shift), dtype=np.float64), wf2[:, : n_win_i - j_shift]))
            ph_bit.fill(0)
            ph_bit[:n_win_i, :n_win_i] = ph_arr[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            h = np.abs(ph_fft)
            h = np.fft.ifftshift(signal.convolve2d(np.fft.fftshift(h), b, mode="same", boundary="fill", fillvalue=0.0))
            mean_h = float(np.median(h))
            if mean_h != 0.0:
                h = h / mean_h
            h = np.power(h, float(alpha))
            ph_filt = np.fft.ifft2(ph_fft * h)[:n_win_i, :n_win_i] * wf2
            ph_out[i1:i2, j1:j2] += ph_filt.astype(np.complex64)
            if ph_out_low is not None and low_filter is not None:
                ph_filt_low = np.fft.ifft2(ph_fft * low_filter)[:n_win_i, :n_win_i] * wf2
                ph_out_low[i1:i2, j1:j2] += ph_filt_low.astype(np.complex64)

    ph_mag = np.abs(ph_arr).astype(np.float32)
    ph_out = (ph_mag * np.exp(1j * np.angle(ph_out))).astype(np.complex64)
    if ph_out_low is not None:
        ph_out_low = (ph_mag * np.exp(1j * np.angle(ph_out_low))).astype(np.complex64)
    return ph_out, ph_out_low


def _stage3_wrap_filt_native(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    native_fn = _native_export("stage3_wrap_filt")
    if native_fn is None:
        raise BackendUnavailableError("Native backend requested for stage3_wrap_filt but the compiled extension does not export it")
    want_low = str(low_flag).lower() == "y"
    out, out_low = native_fn(
        np.ascontiguousarray(np.asarray(ph, dtype=np.complex64).astype(np.complex128, copy=False)),
        int(round(n_win)),
        float(alpha),
        max(0, int(round(n_pad))),
        bool(want_low),
        int(threads),
    )
    out_arr = np.asarray(out, dtype=np.complex64)
    low_arr = np.asarray(out_low, dtype=np.complex64) if want_low else None
    return out_arr, low_arr


def _stage3_wrap_filt_global_python(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    del threads
    ph_arr = np.asarray(ph, dtype=np.complex64).copy()
    if ph_arr.ndim != 2:
        raise ValueError("wrap_filt_global expects a 2-D complex grid")
    n_win_i = int(n_win)
    if n_win_i <= 0:
        raise ValueError("wrap_filt_global requires a positive window size")
    if n_win_i % 2 != 0:
        raise ValueError("wrap_filt_global requires an even window size")
    n_pad_i = max(0, int(n_pad))

    ph_arr[np.isnan(ph_arr)] = 0
    n_i, n_j = ph_arr.shape
    n_inc = max(1, n_win_i // 2)
    n_win_count_i = max(1, int(np.ceil(n_i / n_inc) - 1))
    n_win_count_j = max(1, int(np.ceil(n_j / n_inc) - 1))
    ph_out = np.zeros((n_i, n_j), dtype=np.complex64)
    want_low = str(low_flag).lower() == "y"
    ph_out_low = np.zeros((n_i, n_j), dtype=np.complex64) if want_low else None

    half = n_win_i // 2
    x = np.arange(1, half + 1, dtype=np.float32)
    X, Y = np.meshgrid(x, x)
    wind_func = np.concatenate((X + Y, np.fliplr(X + Y)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0).astype(np.float32)
    gauss = _ported_stage2_module()._gausswin
    b = np.outer(gauss(7), gauss(7)).astype(np.float32)
    ph_bit = np.zeros((n_win_i + n_pad_i, n_win_i + n_pad_i), dtype=np.complex64)
    low_filter = None
    if ph_out_low is not None:
        g = gauss(n_win_i + n_pad_i, alpha=16.0)
        low_filter = np.fft.ifftshift(np.outer(g, g))

    for ix1 in range(n_win_count_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win_i
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win_i
            wf = np.vstack((np.zeros((i_shift, n_win_i), dtype=np.float32), wf[: n_win_i - i_shift, :]))
        for ix2 in range(n_win_count_j):
            wf2 = wf.copy()
            j1 = ix2 * n_inc
            j2 = j1 + n_win_i
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win_i
                wf2 = np.hstack((np.zeros((n_win_i, j_shift), dtype=np.float32), wf2[:, : n_win_i - j_shift]))
            ph_bit.fill(0)
            ph_bit[:n_win_i, :n_win_i] = ph_arr[i1:i2, j1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            h = np.abs(ph_fft)
            h = np.fft.ifftshift(signal.convolve2d(np.fft.fftshift(h), b, mode="same", boundary="fill", fillvalue=0.0))
            mean_h = float(np.median(h))
            if mean_h != 0.0:
                h = h / mean_h
            h = np.power(h, float(alpha))
            ph_filt = np.fft.ifft2(ph_fft * h)[:n_win_i, :n_win_i] * wf2
            ph_out[i1:i2, j1:j2] += ph_filt
            if ph_out_low is not None and low_filter is not None:
                ph_filt_low = np.fft.ifft2(ph_fft * low_filter)[:n_win_i, :n_win_i] * wf2
                ph_out_low[i1:i2, j1:j2] += ph_filt_low

    magnitude = np.abs(ph_arr)
    ph_out = (magnitude * np.exp(1j * np.angle(ph_out))).astype(np.complex64)
    if ph_out_low is not None:
        ph_out_low = (magnitude * np.exp(1j * np.angle(ph_out_low))).astype(np.complex64)
    return ph_out, ph_out_low


def _stage3_wrap_filt_global_native(
    ph: np.ndarray,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    native_fn = _native_export("stage3_wrap_filt_global")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage3_wrap_filt_global but the compiled extension does not export it"
        )
    want_low = str(low_flag).lower() == "y"
    out, out_low = native_fn(
        np.ascontiguousarray(np.asarray(ph, dtype=np.complex64).astype(np.complex128, copy=False)),
        int(n_win),
        float(alpha),
        max(0, int(n_pad)),
        bool(want_low),
        int(threads),
    )
    out_arr = np.asarray(out, dtype=np.complex64)
    low_arr = np.asarray(out_low, dtype=np.complex64) if want_low else None
    return out_arr, low_arr


DEFAULT_REGISTRY.register(
    "stage3_select_ifg_index",
    python=_stage3_select_ifg_index_python,
    native=_stage3_select_ifg_index_native,
)
DEFAULT_REGISTRY.register(
    "stage3_clap_filt_patch",
    python=_stage3_clap_filt_patch_python,
    native=_stage3_clap_filt_patch_native,
)
DEFAULT_REGISTRY.register(
    "stage3_clap_filt_patch_stack",
    python=_stage3_clap_filt_patch_stack_python,
    native=_stage3_clap_filt_patch_stack_native,
)
DEFAULT_REGISTRY.register(
    "stage3_clap_filt_grid",
    python=_stage3_clap_filt_grid_python,
    native=_stage3_clap_filt_grid_native,
)
DEFAULT_REGISTRY.register(
    "stage3_clap_filt_grid_stack",
    python=_stage3_clap_filt_grid_stack_python,
    native=_stage3_clap_filt_grid_stack_native,
)
DEFAULT_REGISTRY.register(
    "stage3_wrap_filt",
    python=_stage3_wrap_filt_python,
    native=_stage3_wrap_filt_native,
)
DEFAULT_REGISTRY.register(
    "stage3_wrap_filt_global",
    python=_stage3_wrap_filt_global_python,
    native=_stage3_wrap_filt_global_native,
)


def _stage3_coh_threshold_python(
    coh_values: np.ndarray,
    d_a: np.ndarray,
    d_a_max: np.ndarray,
    coh_bins: np.ndarray,
    nr_dist: np.ndarray,
    low_coh_thresh: int,
    max_percent_rand: float,
    select_method: str,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del threads
    return _ported_stage2_module()._coh_threshold_from_dist(
        coh_values=coh_values,
        D_A=d_a,
        D_A_max=d_a_max,
        coh_bins=coh_bins,
        Nr_dist=nr_dist,
        low_coh_thresh=low_coh_thresh,
        max_percent_rand=max_percent_rand,
        select_method=select_method,
        histogram_backend="python",
    )


def _stage3_coh_threshold_native(
    coh_values: np.ndarray,
    d_a: np.ndarray,
    d_a_max: np.ndarray,
    coh_bins: np.ndarray,
    nr_dist: np.ndarray,
    low_coh_thresh: int,
    max_percent_rand: float,
    select_method: str,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    native_fn = _native_export("stage3_coh_threshold")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage3_coh_threshold but the compiled extension does not export it"
        )
    coh_thresh, coeffs = native_fn(
        np.ascontiguousarray(np.asarray(coh_values, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(d_a, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(d_a_max, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(coh_bins, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(nr_dist, dtype=np.float64).reshape(-1)),
        int(low_coh_thresh),
        float(max_percent_rand),
        str(select_method),
        int(threads),
    )
    return np.asarray(coh_thresh, dtype=np.float64), np.asarray(coeffs, dtype=np.float64)


DEFAULT_REGISTRY.register(
    "stage3_coh_threshold",
    python=_stage3_coh_threshold_python,
    native=_stage3_coh_threshold_native,
)


def run_stage2_grid_accumulate_kernel(
    ph_weight: np.ndarray,
    grid_lin: np.ndarray,
    n_i: int,
    n_j: int,
    *,
    backend: str = "auto",
    threads: int = 0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    resolved = _resolve_stage2_kernel("stage2_grid_accumulate", backend)
    result = resolved.fn(ph_weight, grid_lin, n_i, n_j, threads)
    if out is not None:
        out[...] = result
        return out
    return np.asarray(result, dtype=np.complex64)


def run_stage2_grid_indices_kernel(
    xy: np.ndarray,
    grid_size: float,
    *,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage2_grid_indices")
    if not stage2_grid_indices_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage2_grid_indices",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return np.asarray(resolved.fn(xy, grid_size, threads), dtype=np.float32)


def run_stage2_clap_filter_kernel(
    *,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage2_clap_filter_kernel")
    if not stage2_clap_filter_kernel_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage2_clap_filter_kernel",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return np.asarray(resolved.fn(threads), dtype=np.float64)


def run_stage2_normalize_complex_kernel(
    values: np.ndarray,
    *,
    preserve_precision: bool = False,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage2_normalize_complex")
    if not stage2_normalize_complex_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage2_normalize_complex",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return np.asarray(resolved.fn(values, preserve_precision, threads), dtype=np.asarray(values).dtype)


def run_stage2_normalize_phase_matrix_kernel(
    ph_nm: np.ndarray,
    *,
    backend: str = "auto",
    threads: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage2_normalize_phase_matrix")
    if not stage2_normalize_phase_matrix_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage2_normalize_phase_matrix",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph_nm, threads)


def run_stage2_ph_weight_block_kernel(
    ph_nm: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    weighting: np.ndarray,
    *,
    preserve_precision: bool = False,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage2_ph_weight_block")
    if not stage2_ph_weight_block_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage2_ph_weight_block",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph_nm, bperp, k_ps, weighting, preserve_precision, threads)


def run_stage2_topofit_kernel(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    backend: str = "auto",
    threads: int = 0,
    cpu_fallback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage2_topofit")
    if cpu_fallback is not None:
        implementations["python"] = lambda arr, bp, wraps, _threads=0: cpu_fallback(arr, bp, wraps)
    resolved = _resolve_stage2_kernel("stage2_topofit", backend, implementations=implementations)
    return resolved.fn(cpxphase, bperp, n_trial_wraps, threads)


def run_stage2_topofit_row_invariant_kernel(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    backend: str = "auto",
    threads: int = 0,
    cpu_fallback: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage2_topofit_row_invariant")
    if cpu_fallback is not None:
        _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
        implementations["python"] = lambda arr, _bp, wraps, _threads=0: cpu_fallback(arr, bperp_mat, wraps)
    resolved = _resolve_stage2_kernel("stage2_topofit_row_invariant", backend, implementations=implementations)
    return resolved.fn(cpxphase, bperp, n_trial_wraps, threads)


def run_stage2_topofit_coh_row_invariant_kernel(
    cpxphase: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    *,
    backend: str = "auto",
    threads: int = 0,
    cpu_fallback: Any | None = None,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage2_topofit_coh_row_invariant")
    if cpu_fallback is not None:
        _, bperp_mat = _stage2_row_invariant_bperp_matrix(bperp, np.asarray(cpxphase).shape[0])
        implementations["python"] = lambda arr, _bp, wraps, _threads=0: np.asarray(
            cpu_fallback(arr, bperp_mat, wraps), dtype=np.float64
        )
    resolved = _resolve_stage2_kernel("stage2_topofit_coh_row_invariant", backend, implementations=implementations)
    return np.asarray(resolved.fn(cpxphase, bperp, n_trial_wraps, threads), dtype=np.float64)


def run_stage2_histogram_kernel(
    values: np.ndarray,
    centers: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    finite_values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    centers_arr = np.asarray(centers, dtype=np.float64).reshape(-1)
    resolved = _resolve_stage2_kernel("stage2_histogram", backend)
    return np.asarray(resolved.fn(finite_values, centers_arr), dtype=np.float64)


def run_stage3_select_ifg_index_kernel(
    n_ifg: int,
    master_ix: int,
    drop_ifg_index: np.ndarray,
    small_baseline: bool,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage3_select_ifg_index")
    if not stage3_select_ifg_index_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_select_ifg_index",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(n_ifg, master_ix, drop_ifg_index, small_baseline, threads)


def run_stage3_clap_filt_patch_kernel(
    ph: np.ndarray,
    *,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage3_clap_filt_patch")
    if not stage3_clap_filt_patch_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_clap_filt_patch",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph, alpha, beta, low_pass, threads)


def run_stage3_clap_filt_patch_stack_kernel(
    ph_stack: np.ndarray,
    *,
    alpha: float,
    beta: float,
    low_pass: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage3_clap_filt_patch_stack")
    if not stage3_clap_filt_patch_stack_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_clap_filt_patch_stack",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph_stack, alpha, beta, low_pass, threads)


def run_stage3_clap_filt_grid_kernel(
    ph: np.ndarray,
    *,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
    preserve_precision: bool = False,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage3_clap_filt_grid")
    n_win_int = int(round(n_win))
    if not stage3_clap_filt_grid_native_available() or n_win_int % 2 != 0:
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_clap_filt_grid",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph, alpha, beta, n_win, n_pad, low_pass, preserve_precision, threads)


def run_stage3_clap_filt_grid_stack_kernel(
    ph_stack: np.ndarray,
    *,
    alpha: float,
    beta: float,
    n_win: int,
    n_pad: int = 0,
    low_pass: np.ndarray | None = None,
    preserve_precision: bool = False,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage3_clap_filt_grid_stack")
    n_win_int = int(round(n_win))
    if not stage3_clap_filt_grid_stack_native_available() or n_win_int % 2 != 0:
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_clap_filt_grid_stack",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph_stack, alpha, beta, n_win, n_pad, low_pass, preserve_precision, threads)


def run_stage3_wrap_filt_kernel(
    ph: np.ndarray,
    *,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str = "n",
    backend: str = "auto",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    implementations = DEFAULT_REGISTRY.implementations("stage3_wrap_filt")
    n_win_int = int(round(n_win))
    ph_shape = np.asarray(ph).shape
    native_supported_shape = len(ph_shape) == 2 and ph_shape[0] >= n_win_int and ph_shape[1] >= n_win_int
    if not stage3_wrap_filt_native_available() or n_win_int % 2 != 0 or not native_supported_shape:
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_wrap_filt",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph, n_win, alpha, n_pad, low_flag, threads)


def run_stage3_wrap_filt_global_kernel(
    ph: np.ndarray,
    *,
    n_win: int,
    alpha: float,
    n_pad: int,
    low_flag: str = "n",
    backend: str = "auto",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    implementations = DEFAULT_REGISTRY.implementations("stage3_wrap_filt_global")
    n_win_int = int(n_win)
    ph_shape = np.asarray(ph).shape
    native_supported_shape = len(ph_shape) == 2 and ph_shape[0] >= n_win_int and ph_shape[1] >= n_win_int
    if not stage3_wrap_filt_global_native_available() or n_win_int % 2 != 0 or not native_supported_shape:
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_wrap_filt_global",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph, n_win, alpha, n_pad, low_flag, threads)


def run_stage3_coh_threshold_kernel(
    coh_values: np.ndarray,
    d_a: np.ndarray,
    d_a_max: np.ndarray,
    coh_bins: np.ndarray,
    nr_dist: np.ndarray,
    low_coh_thresh: int,
    max_percent_rand: float,
    select_method: str,
    backend: str = "auto",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage3_coh_threshold")
    if not stage3_coh_threshold_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage3_coh_threshold",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(
        coh_values,
        d_a,
        d_a_max,
        coh_bins,
        nr_dist,
        low_coh_thresh,
        max_percent_rand,
        select_method,
        threads,
    )


def _stage4_edge_stats_core(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
) -> dict[str, np.ndarray]:
    ported = _ported_stage2_module()

    ph_arr = np.asarray(ph_weed, dtype=np.complex128)
    a = np.asarray(node_a, dtype=np.int64).reshape(-1)
    b = np.asarray(node_b, dtype=np.int64).reshape(-1)
    bperp_arr = np.asarray(bperp, dtype=np.float64).reshape(-1)
    day_arr = np.asarray(day, dtype=np.float64).reshape(-1)

    if ph_arr.ndim != 2:
        raise BackendUnavailableError("stage4_edge_stats expects a 2-D phase matrix")
    if a.shape != b.shape:
        raise BackendUnavailableError("stage4_edge_stats node arrays must have matching shapes")
    if ph_arr.shape[1] != bperp_arr.size:
        raise BackendUnavailableError("stage4_edge_stats bperp vector must match phase width")
    if not small_baseline and day_arr.size != ph_arr.shape[1]:
        raise BackendUnavailableError("stage4_edge_stats day vector must match phase width for non-small-baseline mode")

    n_node, n_use = ph_arr.shape
    ps_std = np.full(n_node, np.inf, dtype=np.float64)
    ps_max = np.full(n_node, np.inf, dtype=np.float64)
    if a.size == 0 or n_use == 0:
        return {"ps_std": ps_std, "ps_max": ps_max}

    dph_space = ph_arr[b, :] * np.conj(ph_arr[a, :])
    if not small_baseline:
        time_win_f = max(float(time_win), 1e-6)
        time_diff_all = day_arr[:, None] - day_arr[None, :]
        weight_all = np.exp(-(time_diff_all**2) / (2.0 * time_win_f**2))
        weight_sums = np.sum(weight_all, axis=1, keepdims=True)
        zero_rows = weight_sums[:, 0] <= 0
        if np.any(zero_rows):
            weight_all[zero_rows, :] = 1.0 / float(max(1, n_use))
            weight_sums = np.sum(weight_all, axis=1, keepdims=True)
        weight_all = weight_all / weight_sums
        diag_weights = np.diag(weight_all).copy()
        dph_smooth = dph_space @ weight_all.T
        dph_smooth2 = dph_smooth - (dph_space * diag_weights[None, :])

        for i1 in range(n_use):
            time_diff = time_diff_all[i1]
            weight = weight_all[i1]
            dph_mean = dph_smooth[:, i1].copy()
            dph_mean_adj = np.angle(dph_space * np.conj(dph_mean)[:, None])
            m0, m1 = ported._weighted_affine_fit(time_diff, dph_mean_adj, weight)
            detrended = dph_mean_adj - (m0[:, None] + m1[:, None] * time_diff[None, :])
            dph_mean_adj2 = np.angle(np.exp(1j * detrended))
            m20, _m21 = ported._weighted_affine_fit(time_diff, dph_mean_adj2, weight)
            dph_smooth[:, i1] = dph_mean * np.exp(1j * (m0 + m20))

        dph_noise = np.angle(dph_space * np.conj(dph_smooth)).astype(np.float64)
        dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2)).astype(np.float64)
        ddof_var = 1 if dph_noise2.shape[0] > 1 else 0
        ifg_var = np.var(dph_noise2, axis=0, ddof=ddof_var)
        w_ifg = np.divide(
            1.0,
            ifg_var,
            out=np.full_like(ifg_var, np.inf, dtype=np.float64),
            where=ifg_var != 0,
        )
        k_edge = ported._weighted_slope_fit(bperp_arr, dph_noise, w_ifg.astype(np.float64))
        dph_noise = dph_noise - k_edge[:, None] * bperp_arr[None, :]
        ddof = 1 if n_use > 1 else 0
        edge_std = np.std(dph_noise, axis=1, ddof=ddof)
        edge_max = np.max(np.abs(dph_noise), axis=1)
    else:
        ddof_var = 1 if dph_space.shape[0] > 1 else 0
        ifg_var = np.var(dph_space, axis=0, ddof=ddof_var)
        w_ifg = np.divide(
            1.0,
            ifg_var,
            out=np.full_like(ifg_var, np.inf, dtype=np.float64),
            where=ifg_var != 0,
        )
        k_edge = ported._weighted_slope_fit(bperp_arr, dph_space, w_ifg.astype(np.float64))
        dph_adj = dph_space - k_edge[:, None] * bperp_arr[None, :]
        ang = np.angle(dph_adj)
        ddof = 1 if n_use > 1 else 0
        edge_std = np.std(ang, axis=1, ddof=ddof)
        edge_max = np.max(np.abs(ang), axis=1)

    np.minimum.at(ps_std, a, edge_std)
    np.minimum.at(ps_std, b, edge_std)
    np.minimum.at(ps_max, a, edge_max)
    np.minimum.at(ps_max, b, edge_max)
    return {"ps_std": ps_std, "ps_max": ps_max}


def _stage4_edge_stats_python(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    return _stage4_edge_stats_core(ph_weed, node_a, node_b, bperp, day, time_win, small_baseline)


def _stage4_edge_stats_native(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage4_edge_stats")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage4_edge_stats but the compiled extension does not export it"
        )

    payload = native_fn(
        np.ascontiguousarray(np.asarray(ph_weed, dtype=np.complex128)),
        np.ascontiguousarray(np.asarray(node_a, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(node_b, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(bperp, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(day, dtype=np.float64).reshape(-1)),
        float(time_win),
        bool(small_baseline),
        _native_threads(threads),
    )
    return {
        "ps_std": np.asarray(payload["ps_std"], dtype=np.float64),
        "ps_max": np.asarray(payload["ps_max"], dtype=np.float64),
    }


def _stage7_scla_core(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
) -> dict[str, np.ndarray]:
    ported = _ported_stage2_module()

    ph_proc_arr = np.asarray(ph_proc, dtype=np.float64)
    ph_mean_v_arr = np.asarray(ph_mean_v, dtype=np.float64)
    bperp_arr = np.asarray(bperp_mat, dtype=np.float64)
    unwrap_ix_arr = np.asarray(unwrap_ix, dtype=np.int64).reshape(-1)
    solve_ix_arr = np.asarray(solve_ix, dtype=np.int64).reshape(-1)
    day_arr = np.asarray(day, dtype=np.float64).reshape(-1)
    ifg_std_arr = np.asarray(ifg_std, dtype=np.float64).reshape(-1)

    ph_seq = np.diff(ph_proc_arr[:, unwrap_ix_arr], axis=1)
    bperp_seq = np.diff(bperp_arr[:, unwrap_ix_arr], axis=1)
    day_seq = np.diff(day_arr[unwrap_ix_arr])
    coest_mean_vel = unwrap_ix_arr.size >= 4

    mean_bperp = np.mean(bperp_seq, axis=0)
    if coest_mean_vel:
        g_seq = np.column_stack((np.ones(day_seq.size, dtype=np.float64), mean_bperp, day_seq))
    else:
        g_seq = np.column_stack((np.ones(day_seq.size, dtype=np.float64), mean_bperp))
    coeffs_seq = ported._weighted_lstsq_shared_design(g_seq, ph_seq.T, cov=None)
    k_ps_uw = coeffs_seq[1, :].astype(np.float64)
    ph_scla = (k_ps_uw[:, None] * bperp_arr).astype(np.float32)

    ifg_vcm = np.diag((ifg_std_arr * np.pi / 180.0) ** 2).astype(np.float64)
    resid_full = ph_proc_arr[:, solve_ix_arr] - ph_scla[:, solve_ix_arr].astype(np.float64)
    if coest_mean_vel:
        g_c = np.column_stack(
            (
                np.ones(solve_ix_arr.size, dtype=np.float64),
                day_arr[solve_ix_arr] - day_arr[int(master_ix) - 1],
            )
        )
        coeffs_c = ported._weighted_lstsq_shared_design(g_c, resid_full.T, cov=ifg_vcm[np.ix_(solve_ix_arr, solve_ix_arr)])
        c_ps_uw = coeffs_c[0, :].astype(np.float32)
    else:
        c_ps_uw = np.mean(resid_full, axis=1).astype(np.float32)

    m = np.asarray(ported._stage7_mean_velocity_fit(ph_mean_v_arr, day_arr, master_ix, ifg_std_arr), dtype=np.float32)
    return {
        "K_ps_uw": k_ps_uw,
        "C_ps_uw": c_ps_uw,
        "ph_scla": ph_scla,
        "ph_ramp": np.zeros_like(ph_proc_arr, dtype=np.float64),
        "ifg_vcm": ifg_vcm,
        "mean_v": m[1, :].astype(np.float32),
        "m": m,
    }


def _stage7_scla_cpu(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    return _stage7_scla_core(ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std)


def _stage7_scla_gpu(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    cp = _cupy()
    if cp is None:
        raise BackendUnavailableError("GPU backend requested but CuPy is not available")
    return _stage7_scla_core(ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std)


def _stage7_scla_native(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage7_scla_parity")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage7_scla but the compiled extension does not export stage7_scla_parity"
        )

    payload = native_fn(
        np.ascontiguousarray(np.asarray(ph_proc, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(ph_mean_v, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(bperp_mat, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(unwrap_ix, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(solve_ix, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(day, dtype=np.float64).reshape(-1)),
        int(master_ix),
        np.ascontiguousarray(np.asarray(ifg_std, dtype=np.float64).reshape(-1)),
        _native_threads(),
    )
    return {
        "K_ps_uw": np.asarray(payload["K_ps_uw"], dtype=np.float64),
        "C_ps_uw": np.asarray(payload["C_ps_uw"], dtype=np.float32),
        "ph_scla": np.asarray(payload["ph_scla"], dtype=np.float32),
        "ph_ramp": np.asarray(payload["ph_ramp"], dtype=np.float64),
        "ifg_vcm": np.asarray(payload["ifg_vcm"], dtype=np.float64),
        "mean_v": np.asarray(payload["mean_v"], dtype=np.float32),
        "m": np.asarray(payload["m"], dtype=np.float32),
    }


def _stage7_scla_smooth_python(
    k_ps_uw: np.ndarray,
    c_ps_uw: np.ndarray,
    edges: np.ndarray,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del threads
    return _ported_stage2_module()._smooth_scla_neighbor_envelope(k_ps_uw, c_ps_uw, edges)


def _stage7_scla_smooth_native(
    k_ps_uw: np.ndarray,
    c_ps_uw: np.ndarray,
    edges: np.ndarray,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    native_fn = _native_export("stage7_scla_smooth")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage7_scla_smooth but the compiled extension does not export it"
        )
    k_out, c_out = native_fn(
        np.ascontiguousarray(np.asarray(k_ps_uw, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(c_ps_uw, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(edges, dtype=np.int64).reshape(-1, 2)),
        int(threads),
    )
    return np.asarray(k_out, dtype=np.float32), np.asarray(c_out, dtype=np.float32)


def _stage7_mean_velocity_fit_python(
    ph_mean_v: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return np.asarray(
        _ported_stage2_module()._stage7_mean_velocity_fit(ph_mean_v, day, int(master_ix), ifg_std),
        dtype=np.float32,
    )


def _stage7_mean_velocity_fit_native(
    ph_mean_v: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage7_mean_velocity_fit")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage7_mean_velocity_fit but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph_mean_v, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(day, dtype=np.float64).reshape(-1)),
            int(master_ix),
            np.ascontiguousarray(np.asarray(ifg_std, dtype=np.float64).reshape(-1)),
            int(threads),
        ),
        dtype=np.float32,
    )


def _stage7_deramp_unwrapped_phase_python(
    xy: np.ndarray,
    ph_all: np.ndarray,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del threads
    xy_arr = np.asarray(xy, dtype=np.float64)
    ps = {
        "n_ps": np.asarray(float(xy_arr.shape[0]), dtype=np.float64),
        "xy": xy_arr,
    }
    return _ported_stage2_module()._deramp_unwrapped_phase(ps, ph_all)


def _stage7_deramp_unwrapped_phase_native(
    xy: np.ndarray,
    ph_all: np.ndarray,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    native_fn = _native_export("stage7_deramp_unwrapped_phase")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage7_deramp_unwrapped_phase but the compiled extension does not export it"
        )
    payload = native_fn(
        np.ascontiguousarray(np.asarray(xy, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(ph_all, dtype=np.float64)),
        int(threads),
    )
    return np.asarray(payload["ph_out"], dtype=np.float64), np.asarray(payload["ph_ramp"], dtype=np.float64)


def _stage7_center_to_reference_python(
    ph: np.ndarray,
    ref_ix: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    ph_arr = np.asarray(ph, dtype=np.float64)
    ref_arr = np.asarray(ref_ix, dtype=np.int64).reshape(-1)
    if ref_arr.size == 0:
        return ph_arr
    ref_mean = np.nanmean(ph_arr[ref_arr, :], axis=0, keepdims=True)
    return ph_arr - ref_mean


def _stage7_center_to_reference_native(
    ph: np.ndarray,
    ref_ix: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage7_center_to_reference")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage7_center_to_reference but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(ref_ix, dtype=np.int64).reshape(-1)),
            int(threads),
        ),
        dtype=np.float64,
    )


def _stage8_edge_noise_cpu(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    ph = np.asarray(uw_ph, dtype=np.complex64)
    a = np.asarray(node_a, dtype=np.int64).reshape(-1)
    b = np.asarray(node_b, dtype=np.int64).reshape(-1)
    n_edge = int(a.size)
    n_ifg = ph.shape[1]

    if chunk_edges <= 0:
        chunk_edges = _auto_chunk_size(n_edge, max(1, n_ifg * 3), np.dtype(np.float32).itemsize)
    chunk_edges = max(1, int(chunk_edges))

    dph_space_uw = np.empty((n_edge, n_ifg), dtype=np.float32)
    dph_noise = np.empty((n_edge, n_ifg), dtype=np.float32)
    for start in range(0, n_edge, chunk_edges):
        end = min(start + chunk_edges, n_edge)
        dph_space = np.angle(ph[b[start:end], :] * np.conj(ph[a[start:end], :])).astype(np.float32)
        dph_space_uw[start:end, :] = dph_space
        dph_noise[start:end, :] = (
            (dph_space - np.mean(dph_space, axis=1, keepdims=True)) * _STAGE8_NOISE_SCALE
        ).astype(np.float32)
    return {"dph_noise": dph_noise, "dph_space_uw": dph_space_uw}


def _stage8_edge_noise_gpu(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    cp = _cupy()
    if cp is None:
        raise BackendUnavailableError("GPU backend requested but CuPy is not available")

    ph = cp.asarray(uw_ph, dtype=cp.complex64)
    a = np.asarray(node_a, dtype=np.int64).reshape(-1)
    b = np.asarray(node_b, dtype=np.int64).reshape(-1)
    n_edge = int(a.size)
    n_ifg = ph.shape[1]

    if chunk_edges <= 0:
        chunk_edges = _auto_chunk_size(n_edge, max(1, n_ifg * 6), np.dtype(np.float32).itemsize)
    chunk_edges = max(1, int(chunk_edges))

    dph_space_uw = np.empty((n_edge, n_ifg), dtype=np.float32)
    dph_noise = np.empty((n_edge, n_ifg), dtype=np.float32)
    for start in range(0, n_edge, chunk_edges):
        end = min(start + chunk_edges, n_edge)
        a_c = cp.asarray(a[start:end], dtype=cp.int64)
        b_c = cp.asarray(b[start:end], dtype=cp.int64)
        dph_space = cp.angle(ph[b_c, :] * cp.conj(ph[a_c, :])).astype(cp.float32)
        dph_space_np = _to_numpy(dph_space).astype(np.float32)
        dph_space_uw[start:end, :] = dph_space_np
        dph_noise[start:end, :] = (
            (dph_space_np - np.mean(dph_space_np, axis=1, keepdims=True)) * _STAGE8_NOISE_SCALE
        ).astype(np.float32)
    return {"dph_noise": dph_noise, "dph_space_uw": dph_space_uw}


def _stage8_edge_noise_native(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage8_edge_noise")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage8_edge_noise but the compiled extension does not export it"
        )

    payload = native_fn(
        np.ascontiguousarray(np.asarray(uw_ph, dtype=np.complex64)),
        np.ascontiguousarray(np.asarray(node_a, dtype=np.int64).reshape(-1)),
        np.ascontiguousarray(np.asarray(node_b, dtype=np.int64).reshape(-1)),
        int(chunk_edges),
        _native_threads(),
    )
    return {
        "dph_noise": np.asarray(payload["dph_noise"], dtype=np.float32),
        "dph_space_uw": np.asarray(payload["dph_space_uw"], dtype=np.float32),
    }


def _stage8_weighted_lstsq_python(
    design: np.ndarray,
    values: np.ndarray,
    covariance: np.ndarray | None = None,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return np.asarray(_ported_stage2_module()._weighted_lstsq_shared_design(design, values, cov=covariance), dtype=np.float64)


def _stage8_weighted_lstsq_native(
    design: np.ndarray,
    values: np.ndarray,
    covariance: np.ndarray | None = None,
    threads: int = 0,
) -> np.ndarray:
    diagonal_fn = _native_export("stage8_weighted_lstsq_diagonal")
    if diagonal_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage8_weighted_lstsq but the compiled extension does not export it"
        )
    design_arr = np.asarray(design, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    if covariance is None:
        variances = np.ones(design_arr.shape[0], dtype=np.float64)
    else:
        cov_arr = np.asarray(covariance, dtype=np.float64)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1] or cov_arr.shape[0] != design_arr.shape[0]:
            raise BackendUnavailableError("stage8_weighted_lstsq covariance has incompatible shape")
        if not np.allclose(cov_arr, np.diag(np.diag(cov_arr))):
            full_fn = _native_export("stage8_weighted_lstsq_full")
            if full_fn is None:
                raise BackendUnavailableError(
                    "Native stage8_weighted_lstsq full-covariance export is unavailable"
                )
            return np.asarray(
                full_fn(
                    np.ascontiguousarray(design_arr),
                    np.ascontiguousarray(values_arr),
                    np.ascontiguousarray(cov_arr),
                    int(threads),
                ),
                dtype=np.float64,
            )
        variances = np.diag(cov_arr).astype(np.float64, copy=True)
    return np.asarray(
        diagonal_fn(
            np.ascontiguousarray(design_arr),
            np.ascontiguousarray(values_arr),
            np.ascontiguousarray(variances),
            int(threads),
        ),
        dtype=np.float64,
    )


def _weighted_affine_fit_python(
    time_diff: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del threads
    return _ported_stage2_module()._weighted_affine_fit(time_diff, values, weights)


def _weighted_affine_fit_native(
    time_diff: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del threads
    native_fn = _native_export("weighted_affine_fit")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for weighted_affine_fit but the compiled extension does not export it"
        )
    payload = native_fn(
        np.ascontiguousarray(np.asarray(time_diff, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(values, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1)),
    )
    return np.asarray(payload["intercept"], dtype=np.float64), np.asarray(payload["slope"], dtype=np.float64)


def _weighted_slope_fit_python(
    x: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return np.asarray(_ported_stage2_module()._weighted_slope_fit(x, values, weights))


def _weighted_slope_fit_native(
    x: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    value_arr = np.asarray(values)
    weight_arr = np.ascontiguousarray(np.asarray(weights, dtype=np.float64).reshape(-1))
    if np.iscomplexobj(value_arr):
        native_fn = _native_export("weighted_slope_fit_complex")
        if native_fn is None:
            raise BackendUnavailableError(
                "Native backend requested for weighted_slope_fit but the complex export is unavailable"
            )
        y_arg = np.ascontiguousarray(value_arr.astype(np.complex128, copy=False))
    else:
        native_fn = _native_export("weighted_slope_fit_real")
        if native_fn is None:
            raise BackendUnavailableError(
                "Native backend requested for weighted_slope_fit but the real export is unavailable"
            )
        y_arg = np.ascontiguousarray(np.asarray(value_arr, dtype=np.float64))
    return np.asarray(native_fn(x_arr, y_arg, weight_arr))


def _stage5_ifg_std_python(
    ph2: np.ndarray,
    ph_patch: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    ph2_arr = np.asarray(ph2, dtype=np.complex64)
    ph_patch_arr = np.asarray(ph_patch, dtype=np.complex64)
    bperp_arr = np.asarray(bperp, dtype=np.float64)
    k_arr = np.asarray(k_ps, dtype=np.float64).reshape(-1)
    c_arr = np.asarray(c_ps, dtype=np.float64).reshape(-1)
    if ph2_arr.shape != ph_patch_arr.shape or ph2_arr.shape != bperp_arr.shape:
        raise ValueError("stage5_ifg_std expects ph2, ph_patch, and bperp with matching shapes")
    if ph2_arr.shape[0] != k_arr.size or ph2_arr.shape[0] != c_arr.size:
        raise ValueError("stage5_ifg_std expects k_ps and c_ps length to match ph2 rows")
    ph_diff = np.angle(
        ph2_arr.astype(np.complex128)
        * np.conj(ph_patch_arr.astype(np.complex128))
        * np.exp(-1j * (k_arr[:, None] * bperp_arr + c_arr[:, None]))
    )
    return (np.sqrt(np.sum(ph_diff**2, axis=0) / max(1, ph2_arr.shape[0])) * 180.0 / np.pi).astype(np.float32)


def _stage5_ifg_std_native(
    ph2: np.ndarray,
    ph_patch: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage5_ifg_std")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage5_ifg_std but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph2, dtype=np.complex64)),
            np.ascontiguousarray(np.asarray(ph_patch, dtype=np.complex64)),
            np.ascontiguousarray(np.asarray(bperp, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(k_ps, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(c_ps, dtype=np.float64).reshape(-1)),
            int(threads),
        ),
        dtype=np.float32,
    )


def _stage5_duplicate_keep_python(
    lonlat: np.ndarray,
    coh_ps: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return _ported_stage2_module()._dedup_lonlat_keep_highest_coh(
        np.asarray(lonlat, dtype=np.float64),
        np.asarray(coh_ps, dtype=np.float64).reshape(-1),
    )


def _stage5_duplicate_keep_native(
    lonlat: np.ndarray,
    coh_ps: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage5_duplicate_keep")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage5_duplicate_keep but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(lonlat, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(coh_ps, dtype=np.float64).reshape(-1)),
            int(threads),
        ),
        dtype=bool,
    )


def _stage6_unwrap_grid_native(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    nshortcycle: float = 200.0,
    threads: int = 0,
) -> dict[str, np.ndarray | float]:
    native_fn = _native_export("stage6_unwrap_grid")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_unwrap_grid but the compiled extension does not export it"
        )

    payload = native_fn(
        np.ascontiguousarray(np.asarray(ifgw, dtype=np.complex64)),
        np.ascontiguousarray(np.asarray(rowcost, dtype=np.int16)),
        np.ascontiguousarray(np.asarray(colcost, dtype=np.int16)),
        float(nshortcycle),
        int(threads),
    )
    out: dict[str, np.ndarray | float | int] = {
        "ifguw": np.asarray(payload["ifguw"], dtype=np.float32),
        "msd": float(payload["msd"]),
    }
    if "flow_cycles" in payload:
        out["flow_cycles"] = int(payload["flow_cycles"])
    if "flow_objective" in payload:
        out["flow_objective"] = int(payload["flow_objective"])
    if "post_label_flow_cycles" in payload:
        out["post_label_flow_cycles"] = int(payload["post_label_flow_cycles"])
    if "post_label_flow_objective" in payload:
        out["post_label_flow_objective"] = int(payload["post_label_flow_objective"])
    return out


def _stage6_extract_grid_values_python(
    ifguw: np.ndarray,
    nzix: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return np.asarray(_ported_stage2_module()._extract_grid_values_for_ps(ifguw, nzix), dtype=np.float32)


def _stage6_extract_grid_values_native(
    ifguw: np.ndarray,
    nzix: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage6_extract_grid_values")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_extract_grid_values but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ifguw, dtype=np.float32)),
            np.ascontiguousarray(np.asarray(nzix, dtype=bool)),
            int(threads),
        ),
        dtype=np.float32,
    )


def _stage6_prepare_cost_offsets_python(
    rowcost_base: np.ndarray,
    colcost_base: np.ndarray,
    rowix: np.ndarray,
    colix: np.ndarray,
    wrapped_space_uw: np.ndarray,
    dph_smooth: np.ndarray,
    nshortcycle: float = 200.0,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del threads
    rowcost = np.asarray(rowcost_base, dtype=np.int16).copy()
    colcost = np.asarray(colcost_base, dtype=np.int16).copy()
    rowix_arr = np.asarray(rowix, dtype=np.float64)
    colix_arr = np.asarray(colix, dtype=np.float64)
    wrapped_arr = np.asarray(wrapped_space_uw, dtype=np.float32).reshape(-1)
    smooth_arr = np.asarray(dph_smooth, dtype=np.float32).reshape(-1)
    if wrapped_arr.shape != smooth_arr.shape:
        raise BackendUnavailableError("stage6_prepare_cost_offsets phase vectors must have matching lengths")
    offset_cycle = (wrapped_arr - smooth_arr) / (2.0 * np.pi)
    nzrowix = np.abs(rowix_arr) > 0
    nzcolix = np.abs(colix_arr) > 0
    offgrid = np.zeros(rowix_arr.shape, dtype=np.int16)
    offgrid[nzrowix] = np.rint(
        offset_cycle[np.abs(rowix_arr[nzrowix]).astype(np.int64) - 1]
        * np.sign(rowix_arr[nzrowix])
        * float(nshortcycle)
    ).astype(np.int16)
    rowcost[:, 0::4] = -offgrid
    offgrid = np.zeros(colix_arr.shape, dtype=np.int16)
    offgrid[nzcolix] = np.rint(
        offset_cycle[np.abs(colix_arr[nzcolix]).astype(np.int64) - 1]
        * np.sign(colix_arr[nzcolix])
        * float(nshortcycle)
    ).astype(np.int16)
    colcost[:, 0::4] = offgrid
    return rowcost, colcost


def _stage6_prepare_cost_offsets_native(
    rowcost_base: np.ndarray,
    colcost_base: np.ndarray,
    rowix: np.ndarray,
    colix: np.ndarray,
    wrapped_space_uw: np.ndarray,
    dph_smooth: np.ndarray,
    nshortcycle: float = 200.0,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    native_fn = _native_export("stage6_prepare_cost_offsets")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_prepare_cost_offsets but the compiled extension does not export it"
        )
    rowcost, colcost = native_fn(
        np.ascontiguousarray(np.asarray(rowcost_base, dtype=np.int16)),
        np.ascontiguousarray(np.asarray(colcost_base, dtype=np.int16)),
        np.ascontiguousarray(np.asarray(rowix, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(colix, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(wrapped_space_uw, dtype=np.float32).reshape(-1)),
        np.ascontiguousarray(np.asarray(dph_smooth, dtype=np.float32).reshape(-1)),
        float(nshortcycle),
        int(threads),
    )
    return np.asarray(rowcost, dtype=np.int16), np.asarray(colcost, dtype=np.int16)


def _stage6_reconstruct_ps_phase_python(
    ph_uw_grid: np.ndarray,
    ps_grid_idx: np.ndarray,
    ph_in: np.ndarray,
    phase_restore: np.ndarray | None = None,
    threads: int = 0,
) -> np.ndarray:
    del threads
    grid_arr = np.asarray(ph_uw_grid, dtype=np.float32)
    idx_arr = np.asarray(ps_grid_idx, dtype=np.int64).reshape(-1)
    ph_in_arr = np.asarray(ph_in, dtype=np.complex64)
    out = np.full((idx_arr.size, grid_arr.shape[1]), np.nan, dtype=np.float32)
    valid = idx_arr > 0
    if np.any(valid):
        ph_uw_pix = grid_arr[idx_arr[valid] - 1, :].astype(np.float32)
        out[valid, :] = ph_uw_pix + np.angle(ph_in_arr[valid, :] * np.exp(-1j * ph_uw_pix)).astype(np.float32)
        if phase_restore is not None:
            out[valid, :] += np.asarray(phase_restore, dtype=np.float32)[valid, :]
    return out


def _stage6_reconstruct_ps_phase_native(
    ph_uw_grid: np.ndarray,
    ps_grid_idx: np.ndarray,
    ph_in: np.ndarray,
    phase_restore: np.ndarray | None = None,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage6_reconstruct_ps_phase")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_reconstruct_ps_phase but the compiled extension does not export it"
        )
    restore_arg = None
    if phase_restore is not None:
        restore_arg = np.ascontiguousarray(np.asarray(phase_restore, dtype=np.float32))
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph_uw_grid, dtype=np.float32)),
            np.ascontiguousarray(np.asarray(ps_grid_idx, dtype=np.int64).reshape(-1)),
            np.ascontiguousarray(np.asarray(ph_in, dtype=np.complex64)),
            restore_arg,
            int(threads),
        ),
        dtype=np.float32,
    )


def _stage6_ps_grid_indices_python(
    nzix: np.ndarray,
    grid_ij: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    nzix_arr = np.asarray(nzix, dtype=bool)
    grid_arr = np.asarray(grid_ij, dtype=np.int64)
    gridix_flat = np.zeros(nzix_arr.size, dtype=np.int64)
    nz_flat_f = np.flatnonzero(nzix_arr.reshape(-1, order="F"))
    gridix_flat[nz_flat_f] = np.arange(1, int(np.count_nonzero(nzix_arr)) + 1, dtype=np.int64)
    gridix = gridix_flat.reshape(nzix_arr.shape, order="F")
    return gridix[grid_arr[:, 0] - 1, grid_arr[:, 1] - 1]


def _stage6_ps_grid_indices_native(
    nzix: np.ndarray,
    grid_ij: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage6_ps_grid_indices")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_ps_grid_indices but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(nzix, dtype=bool)),
            np.ascontiguousarray(np.asarray(grid_ij, dtype=np.int64)),
            int(threads),
        ),
        dtype=np.int64,
    )


def _stage6_select_ifgw_python(
    uw_ph: np.ndarray,
    z: np.ndarray,
    ifg_ix: int,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return np.asarray(np.asarray(uw_ph, dtype=np.complex64)[np.asarray(z, dtype=np.int64) - 1, int(ifg_ix)], dtype=np.complex64)


def _stage6_select_ifgw_native(
    uw_ph: np.ndarray,
    z: np.ndarray,
    ifg_ix: int,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage6_select_ifgw")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_select_ifgw but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(uw_ph, dtype=np.complex64)),
            np.ascontiguousarray(np.asarray(z, dtype=np.int64)),
            int(ifg_ix),
            int(threads),
        ),
        dtype=np.complex64,
    )


def _stage6_grid_accumulate_python(
    ph_in: np.ndarray,
    grid_lin: np.ndarray,
    n_cells: int,
    threads: int = 0,
) -> np.ndarray:
    del threads
    ph_arr = np.asarray(ph_in, dtype=np.complex64)
    grid_arr = np.asarray(grid_lin, dtype=np.int64).reshape(-1)
    out = np.zeros((int(n_cells), ph_arr.shape[1]), dtype=np.complex64)
    np.add.at(out, grid_arr, ph_arr)
    return out


def _stage6_grid_accumulate_native(
    ph_in: np.ndarray,
    grid_lin: np.ndarray,
    n_cells: int,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage6_grid_accumulate")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_grid_accumulate but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph_in, dtype=np.complex64)),
            np.ascontiguousarray(np.asarray(grid_lin, dtype=np.int64).reshape(-1)),
            int(n_cells),
            int(threads),
        ),
        dtype=np.complex64,
    )


def _stage6_unwrap_ifg_sets_python(
    n_ifg: int,
    master_ix: int,
    drop_ifg_index: np.ndarray,
    small_baseline: bool,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    del threads
    drop = set(int(v) for v in np.asarray(drop_ifg_index, dtype=np.int64).reshape(-1).tolist())
    unwrap_ifg = np.asarray([i for i in range(1, int(n_ifg) + 1) if i not in drop], dtype=np.int64)
    if bool(small_baseline):
        solve_ifg = unwrap_ifg.copy()
    else:
        solve_ifg = unwrap_ifg[unwrap_ifg != int(master_ix)]
    return {"unwrap_ifg": unwrap_ifg, "solve_ifg": solve_ifg}


def _stage6_unwrap_ifg_sets_native(
    n_ifg: int,
    master_ix: int,
    drop_ifg_index: np.ndarray,
    small_baseline: bool,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage6_unwrap_ifg_sets")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_unwrap_ifg_sets but the compiled extension does not export it"
        )
    payload = native_fn(
        int(n_ifg),
        int(master_ix),
        np.ascontiguousarray(np.asarray(drop_ifg_index, dtype=np.int64).reshape(-1)),
        bool(small_baseline),
        int(threads),
    )
    return {
        "unwrap_ifg": np.asarray(payload["unwrap_ifg"], dtype=np.int64),
        "solve_ifg": np.asarray(payload["solve_ifg"], dtype=np.int64),
    }


def _stage6_single_master_ifg_geometry_python(
    n_ifg: int,
    master_ix: int,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    del threads
    unwrap_ifg = np.asarray([i for i in range(1, int(n_ifg) + 1) if i != int(master_ix)], dtype=np.int64)
    ifgday_ix = np.column_stack(
        (
            np.full(unwrap_ifg.size, int(master_ix), dtype=np.int64),
            unwrap_ifg,
        )
    ).astype(np.int64)
    return {"unwrap_ifg": unwrap_ifg, "ifgday_ix": ifgday_ix}


def _stage6_single_master_ifg_geometry_native(
    n_ifg: int,
    master_ix: int,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage6_single_master_ifg_geometry")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_single_master_ifg_geometry but the compiled extension does not export it"
        )
    payload = native_fn(int(n_ifg), int(master_ix), int(threads))
    return {
        "unwrap_ifg": np.asarray(payload["unwrap_ifg"], dtype=np.int64),
        "ifgday_ix": np.asarray(payload["ifgday_ix"], dtype=np.int64),
    }


def _stage6_estimate_la_error_python(
    dph_space: np.ndarray,
    day: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return np.asarray(
        _ported_stage2_module()._estimate_la_error_single_master(
            dph_space,
            day=day,
            bperp=bperp,
            n_trial_wraps=n_trial_wraps,
        ),
        dtype=np.float32,
    )


def _stage6_estimate_la_error_native(
    dph_space: np.ndarray,
    day: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage6_estimate_la_error_single_master")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_estimate_la_error but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(dph_space, dtype=np.complex64)),
            np.ascontiguousarray(np.asarray(day, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(bperp, dtype=np.float64).reshape(-1)),
            float(n_trial_wraps),
            int(threads),
        ),
        dtype=np.float32,
    )


def _stage6_smooth_3d_full_single_master_python(
    dph_space: np.ndarray,
    day: np.ndarray,
    time_win: float,
    chunk_edges: int = 32768,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del threads
    return _ported_stage2_module()._smooth_3d_full_single_master(
        dph_space,
        day=day,
        time_win=time_win,
        chunk_edges=chunk_edges,
    )


def _stage6_smooth_3d_full_single_master_native(
    dph_space: np.ndarray,
    day: np.ndarray,
    time_win: float,
    chunk_edges: int = 32768,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    del chunk_edges
    native_fn = _native_export("stage6_smooth_3d_full_single_master")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage6_smooth_3d_full_single_master but the compiled extension does not export it"
        )
    payload = native_fn(
        np.ascontiguousarray(np.asarray(dph_space, dtype=np.complex64)),
        np.ascontiguousarray(np.asarray(day, dtype=np.float64).reshape(-1)),
        float(time_win),
        int(threads),
    )
    return np.asarray(payload["dph_smooth_uw"], dtype=np.float32), np.asarray(payload["dph_noise"], dtype=np.float32)


DEFAULT_REGISTRY.register("stage7_scla", python=_stage7_scla_cpu, native=_stage7_scla_native, cuda=_stage7_scla_gpu)
DEFAULT_REGISTRY.register("stage7_scla_smooth", python=_stage7_scla_smooth_python, native=_stage7_scla_smooth_native)
DEFAULT_REGISTRY.register(
    "stage7_mean_velocity_fit",
    python=_stage7_mean_velocity_fit_python,
    native=_stage7_mean_velocity_fit_native,
)
DEFAULT_REGISTRY.register(
    "stage7_deramp_unwrapped_phase",
    python=_stage7_deramp_unwrapped_phase_python,
    native=_stage7_deramp_unwrapped_phase_native,
)
DEFAULT_REGISTRY.register(
    "stage7_center_to_reference",
    python=_stage7_center_to_reference_python,
    native=_stage7_center_to_reference_native,
)
DEFAULT_REGISTRY.register("stage4_edge_stats", python=_stage4_edge_stats_python, native=_stage4_edge_stats_native)


def _stage4_duplicate_keep_python(
    xy: np.ndarray,
    coh: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return _ported_stage2_module()._dedup_lonlat_keep_highest_coh(
        np.asarray(xy, dtype=np.float64),
        np.asarray(coh, dtype=np.float64).reshape(-1),
    )


def _stage4_duplicate_keep_native(
    xy: np.ndarray,
    coh: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage4_duplicate_keep")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage4_duplicate_keep but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(xy, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(coh, dtype=np.float64).reshape(-1)),
            int(threads),
        ),
        dtype=bool,
    )


DEFAULT_REGISTRY.register(
    "stage4_duplicate_keep",
    python=_stage4_duplicate_keep_python,
    native=_stage4_duplicate_keep_native,
)


def _stage4_adjacent_component_keep_python(
    ij_cols23: np.ndarray,
    coh: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return _ported_stage2_module()._adjacent_component_keep_mask(
        np.asarray(ij_cols23, dtype=np.int64),
        np.asarray(coh, dtype=np.float64).reshape(-1),
    )


def _stage4_adjacent_component_keep_native(
    ij_cols23: np.ndarray,
    coh: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage4_adjacent_component_keep")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage4_adjacent_component_keep but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ij_cols23, dtype=np.int64)),
            np.ascontiguousarray(np.asarray(coh, dtype=np.float64).reshape(-1)),
            int(threads),
        ),
        dtype=bool,
    )


DEFAULT_REGISTRY.register(
    "stage4_adjacent_component_keep",
    python=_stage4_adjacent_component_keep_python,
    native=_stage4_adjacent_component_keep_native,
)


def _stage4_weed_ifg_index_python(
    n_ifg: int,
    drop_ifg_index: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    drop = set(int(v) for v in np.asarray(drop_ifg_index, dtype=np.int64).reshape(-1).tolist())
    return np.asarray([i for i in range(1, int(n_ifg) + 1) if i not in drop], dtype=np.float64)


def _stage4_weed_ifg_index_native(
    n_ifg: int,
    drop_ifg_index: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage4_weed_ifg_index")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage4_weed_ifg_index but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            int(n_ifg),
            np.ascontiguousarray(np.asarray(drop_ifg_index, dtype=np.int64).reshape(-1)),
            int(threads),
        ),
        dtype=np.float64,
    )


def _stage4_phase_correction_python(
    ph2: np.ndarray,
    ix_weed: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    bperp: np.ndarray,
    small_baseline: bool,
    master_ix: int,
    threads: int = 0,
) -> np.ndarray:
    del threads
    ph_arr = np.asarray(ph2, dtype=np.complex128)
    keep = np.asarray(ix_weed, dtype=bool).reshape(-1)
    k = np.asarray(k_ps, dtype=np.float64).reshape(-1)
    c = np.asarray(c_ps, dtype=np.float64).reshape(-1)
    bp = np.asarray(bperp, dtype=np.float64).reshape(-1)
    out = ph_arr[keep, :] * np.exp(-1j * (k[keep][:, None] * bp[None, :]))
    out = np.divide(out, np.abs(out), out=np.zeros_like(out), where=np.abs(out) != 0)
    out = np.divide(out, np.abs(out), out=np.zeros_like(out), where=np.abs(out) != 0)
    if not small_baseline:
        out[:, int(master_ix) - 1] = np.exp(1j * c[keep])
    return out.astype(np.complex128, copy=False)


def _stage4_phase_correction_native(
    ph2: np.ndarray,
    ix_weed: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    bperp: np.ndarray,
    small_baseline: bool,
    master_ix: int,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage4_phase_correction")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage4_phase_correction but the compiled extension does not export it"
        )
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(ph2, dtype=np.complex128)),
            np.ascontiguousarray(np.asarray(ix_weed, dtype=bool).reshape(-1)),
            np.ascontiguousarray(np.asarray(k_ps, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(c_ps, dtype=np.float64).reshape(-1)),
            np.ascontiguousarray(np.asarray(bperp, dtype=np.float64).reshape(-1)),
            bool(small_baseline),
            int(master_ix),
            int(threads),
        ),
        dtype=np.complex128,
    )


DEFAULT_REGISTRY.register(
    "stage4_weed_ifg_index",
    python=_stage4_weed_ifg_index_python,
    native=_stage4_weed_ifg_index_native,
)
DEFAULT_REGISTRY.register(
    "stage4_phase_correction",
    python=_stage4_phase_correction_python,
    native=_stage4_phase_correction_native,
)
DEFAULT_REGISTRY.register("stage5_ifg_std", python=_stage5_ifg_std_python, native=_stage5_ifg_std_native)
DEFAULT_REGISTRY.register(
    "stage5_duplicate_keep",
    python=_stage5_duplicate_keep_python,
    native=_stage5_duplicate_keep_native,
)


def _stage5_rc2_correction_python(
    ph2: np.ndarray,
    ph_patch: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    small_baseline: bool,
    master_ix: int,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    del threads
    ph_arr = np.asarray(ph2, dtype=np.complex64)
    bp = np.asarray(bperp, dtype=np.float64)
    k = np.asarray(k_ps, dtype=np.float64).reshape(-1)
    c = np.asarray(c_ps, dtype=np.float64).reshape(-1)
    if small_baseline:
        ph_rc = ph_arr.astype(np.complex128) * np.exp(-1j * (k[:, None] * bp))
        return {"ph_rc": ph_rc.astype(np.complex64)}

    master = int(master_ix)
    bperp_full = np.concatenate(
        [
            bp[:, : master - 1],
            np.zeros((ph_arr.shape[0], 1), dtype=np.float64),
            bp[:, master - 1 :],
        ],
        axis=1,
    )
    ph_rc = ph_arr.astype(np.complex128) * np.exp(-1j * (k[:, None] * bperp_full + c[:, None]))
    patch = np.asarray(ph_patch, dtype=np.complex64)
    ph_reref = np.concatenate(
        [
            patch[:, : master - 1],
            np.ones((ph_arr.shape[0], 1), dtype=np.complex64),
            patch[:, master - 1 :],
        ],
        axis=1,
    )
    return {"ph_rc": ph_rc.astype(np.complex64), "ph_reref": ph_reref.astype(np.complex64)}


def _stage5_rc2_correction_native(
    ph2: np.ndarray,
    ph_patch: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    small_baseline: bool,
    master_ix: int,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage5_rc2_correction")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage5_rc2_correction but the compiled extension does not export it"
        )
    payload = native_fn(
        np.ascontiguousarray(np.asarray(ph2, dtype=np.complex64)),
        np.ascontiguousarray(np.asarray(ph_patch, dtype=np.complex64)),
        np.ascontiguousarray(np.asarray(bperp, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(k_ps, dtype=np.float64).reshape(-1)),
        np.ascontiguousarray(np.asarray(c_ps, dtype=np.float64).reshape(-1)),
        bool(small_baseline),
        int(master_ix),
        int(threads),
    )
    out = {"ph_rc": np.asarray(payload["ph_rc"], dtype=np.complex64)}
    if "ph_reref" in payload:
        out["ph_reref"] = np.asarray(payload["ph_reref"], dtype=np.complex64)
    return out


DEFAULT_REGISTRY.register(
    "stage5_rc2_correction",
    python=_stage5_rc2_correction_python,
    native=_stage5_rc2_correction_native,
)


def _stage5_format_merged_rc2_python(
    rc2_all: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    del threads
    return _ported_stage2_module()._format_merged_rc2_payload(rc2_all)


def _stage5_format_merged_rc2_native(
    rc2_all: np.ndarray,
    threads: int = 0,
) -> np.ndarray:
    native_fn = _native_export("stage5_format_merged_rc2")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage5_format_merged_rc2 but the compiled extension does not export it"
        )
    if not np.iscomplexobj(rc2_all):
        return _stage5_format_merged_rc2_python(rc2_all, threads)
    return np.asarray(
        native_fn(
            np.ascontiguousarray(np.asarray(rc2_all, dtype=np.complex64)),
            int(threads),
        ),
        dtype=np.complex64,
    )


DEFAULT_REGISTRY.register(
    "stage5_format_merged_rc2",
    python=_stage5_format_merged_rc2_python,
    native=_stage5_format_merged_rc2_native,
)


def _stage5_patch_keep_mask_python(
    ij_cols: np.ndarray,
    merged_ij_cols: np.ndarray,
    merged_indices: np.ndarray,
    patch_bounds: np.ndarray | None = None,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    del threads
    ij_arr = np.asarray(ij_cols, dtype=np.int64)
    merged_arr = np.asarray(merged_ij_cols, dtype=np.int64)
    merged_ix = np.asarray(merged_indices, dtype=np.int64).reshape(-1)
    bounds = None if patch_bounds is None else np.asarray(patch_bounds, dtype=np.int64).reshape(-1)
    keep_patch = np.ones(ij_arr.shape[0], dtype=bool)
    if bounds is not None and bounds.size >= 4:
        row_min, row_max, col_min, col_max = (int(v) for v in bounds[:4])
        keep_patch = (
            (ij_arr[:, 0] >= col_min - 1)
            & (ij_arr[:, 0] <= col_max - 1)
            & (ij_arr[:, 1] >= row_min - 1)
            & (ij_arr[:, 1] <= row_max - 1)
        )

    remove_ix: list[int] = []
    for idx in range(ij_arr.shape[0]):
        matches = np.flatnonzero((merged_arr[:, 0] == ij_arr[idx, 0]) & (merged_arr[:, 1] == ij_arr[idx, 1]))
        merged_index = int(merged_ix[matches[0]]) if matches.size else None
        if keep_patch[idx]:
            if merged_index is not None:
                remove_ix.append(merged_index)
        elif merged_index is None:
            keep_patch[idx] = True
    return {"keep_patch": keep_patch, "remove_ix": np.asarray(remove_ix, dtype=np.int64)}


def _stage5_patch_keep_mask_native(
    ij_cols: np.ndarray,
    merged_ij_cols: np.ndarray,
    merged_indices: np.ndarray,
    patch_bounds: np.ndarray | None = None,
    threads: int = 0,
) -> dict[str, np.ndarray]:
    native_fn = _native_export("stage5_patch_keep_mask")
    if native_fn is None:
        raise BackendUnavailableError(
            "Native backend requested for stage5_patch_keep_mask but the compiled extension does not export it"
        )
    bounds_arg = None
    if patch_bounds is not None:
        bounds_arg = np.ascontiguousarray(np.asarray(patch_bounds, dtype=np.int64).reshape(-1))
    payload = native_fn(
        np.ascontiguousarray(np.asarray(ij_cols, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(merged_ij_cols, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(merged_indices, dtype=np.int64).reshape(-1)),
        bounds_arg,
        int(threads),
    )
    return {
        "keep_patch": np.asarray(payload["keep_patch"], dtype=bool),
        "remove_ix": np.asarray(payload["remove_ix"], dtype=np.int64),
    }


DEFAULT_REGISTRY.register(
    "stage5_patch_keep_mask",
    python=_stage5_patch_keep_mask_python,
    native=_stage5_patch_keep_mask_native,
)
DEFAULT_REGISTRY.register("stage6_unwrap_grid", native=_stage6_unwrap_grid_native)
DEFAULT_REGISTRY.register(
    "stage6_extract_grid_values",
    python=_stage6_extract_grid_values_python,
    native=_stage6_extract_grid_values_native,
)
DEFAULT_REGISTRY.register(
    "stage6_prepare_cost_offsets",
    python=_stage6_prepare_cost_offsets_python,
    native=_stage6_prepare_cost_offsets_native,
)
DEFAULT_REGISTRY.register(
    "stage6_reconstruct_ps_phase",
    python=_stage6_reconstruct_ps_phase_python,
    native=_stage6_reconstruct_ps_phase_native,
)
DEFAULT_REGISTRY.register(
    "stage6_ps_grid_indices",
    python=_stage6_ps_grid_indices_python,
    native=_stage6_ps_grid_indices_native,
)
DEFAULT_REGISTRY.register(
    "stage6_select_ifgw",
    python=_stage6_select_ifgw_python,
    native=_stage6_select_ifgw_native,
)
DEFAULT_REGISTRY.register(
    "stage6_grid_accumulate",
    python=_stage6_grid_accumulate_python,
    native=_stage6_grid_accumulate_native,
)
DEFAULT_REGISTRY.register(
    "stage6_unwrap_ifg_sets",
    python=_stage6_unwrap_ifg_sets_python,
    native=_stage6_unwrap_ifg_sets_native,
)
DEFAULT_REGISTRY.register(
    "stage6_single_master_ifg_geometry",
    python=_stage6_single_master_ifg_geometry_python,
    native=_stage6_single_master_ifg_geometry_native,
)
DEFAULT_REGISTRY.register(
    "stage6_estimate_la_error",
    python=_stage6_estimate_la_error_python,
    native=_stage6_estimate_la_error_native,
)
DEFAULT_REGISTRY.register(
    "stage6_smooth_3d_full_single_master",
    python=_stage6_smooth_3d_full_single_master_python,
    native=_stage6_smooth_3d_full_single_master_native,
)
DEFAULT_REGISTRY.register(
    "stage8_edge_noise",
    python=_stage8_edge_noise_cpu,
    native=_stage8_edge_noise_native,
    cuda=_stage8_edge_noise_gpu,
)
DEFAULT_REGISTRY.register(
    "stage8_weighted_lstsq",
    python=_stage8_weighted_lstsq_python,
    native=_stage8_weighted_lstsq_native,
)
DEFAULT_REGISTRY.register(
    "weighted_affine_fit",
    python=_weighted_affine_fit_python,
    native=_weighted_affine_fit_native,
)
DEFAULT_REGISTRY.register(
    "weighted_slope_fit",
    python=_weighted_slope_fit_python,
    native=_weighted_slope_fit_native,
)


def run_stage7_scla_kernel(
    ph_proc: np.ndarray,
    ph_mean_v: np.ndarray,
    bperp_mat: np.ndarray,
    unwrap_ix: np.ndarray,
    solve_ix: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    backend: str = "auto",
    chunk_ps: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage7_scla")
    if not stage7_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage7_scla",
        backend,
        auto_order=("native", "python"),
        explicit_fallbacks={"cuda": ("python",)},
        implementations=implementations,
    )
    return resolved.fn(ph_proc, ph_mean_v, bperp_mat, unwrap_ix, solve_ix, day, master_ix, ifg_std, chunk_ps)


def run_stage4_edge_stats_kernel(
    ph_weed: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    bperp: np.ndarray,
    day: np.ndarray,
    time_win: float,
    small_baseline: bool,
    backend: str = "auto",
    threads: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage4_edge_stats")
    if not stage4_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage4_edge_stats",
        backend,
        auto_order=("native", "python"),
        explicit_fallbacks={"cuda": ("python",)},
        implementations=implementations,
    )
    return resolved.fn(ph_weed, node_a, node_b, bperp, day, time_win, small_baseline, threads)


def run_stage4_duplicate_keep_kernel(
    xy: np.ndarray,
    coh: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage4_duplicate_keep")
    if not stage4_duplicate_keep_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage4_duplicate_keep",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(xy, coh, threads)


def run_stage4_adjacent_component_keep_kernel(
    ij_cols23: np.ndarray,
    coh: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage4_adjacent_component_keep")
    if not stage4_adjacent_component_keep_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage4_adjacent_component_keep",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ij_cols23, coh, threads)


def run_stage4_weed_ifg_index_kernel(
    n_ifg: int,
    drop_ifg_index: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage4_weed_ifg_index")
    if not stage4_weed_ifg_index_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage4_weed_ifg_index",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(n_ifg, drop_ifg_index, threads)


def run_stage4_phase_correction_kernel(
    ph2: np.ndarray,
    ix_weed: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    bperp: np.ndarray,
    *,
    small_baseline: bool,
    master_ix: int,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage4_phase_correction")
    if not stage4_phase_correction_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage4_phase_correction",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph2, ix_weed, k_ps, c_ps, bperp, small_baseline, master_ix, threads)


def run_stage7_scla_smooth_kernel(
    k_ps_uw: np.ndarray,
    c_ps_uw: np.ndarray,
    edges: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage7_scla_smooth")
    if not stage7_scla_smooth_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage7_scla_smooth",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(k_ps_uw, c_ps_uw, edges, threads)


def run_stage7_mean_velocity_fit_kernel(
    ph_mean_v: np.ndarray,
    day: np.ndarray,
    master_ix: int,
    ifg_std: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage7_mean_velocity_fit")
    if not stage7_mean_velocity_fit_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage7_mean_velocity_fit",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph_mean_v, day, master_ix, ifg_std, threads)


def run_stage7_deramp_unwrapped_phase_kernel(
    xy: np.ndarray,
    ph_all: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage7_deramp_unwrapped_phase")
    if not stage7_deramp_unwrapped_phase_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage7_deramp_unwrapped_phase",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(xy, ph_all, threads)


def run_stage7_center_to_reference_kernel(
    ph: np.ndarray,
    ref_ix: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage7_center_to_reference")
    if not stage7_center_to_reference_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage7_center_to_reference",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph, ref_ix, threads)


def run_stage8_weighted_lstsq_kernel(
    design: np.ndarray,
    values: np.ndarray,
    covariance: np.ndarray | None = None,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage8_weighted_lstsq")
    if not stage8_weighted_lstsq_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage8_weighted_lstsq",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(design, values, covariance, threads)


def run_weighted_affine_fit_kernel(
    time_diff: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("weighted_affine_fit")
    if not weighted_affine_fit_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "weighted_affine_fit",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(time_diff, values, weights, threads)


def run_weighted_slope_fit_kernel(
    x: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("weighted_slope_fit")
    if not weighted_slope_fit_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "weighted_slope_fit",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(x, values, weights, threads)


def run_stage8_edge_noise_kernel(
    uw_ph: np.ndarray,
    node_a: np.ndarray,
    node_b: np.ndarray,
    backend: str = "auto",
    chunk_edges: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage8_edge_noise")
    if not stage8_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage8_edge_noise",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(uw_ph, node_a, node_b, chunk_edges)


def run_stage5_ifg_std_kernel(
    ph2: np.ndarray,
    ph_patch: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage5_ifg_std")
    if not stage5_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage5_ifg_std",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph2, ph_patch, bperp, k_ps, c_ps, threads)


def run_stage5_duplicate_keep_kernel(
    lonlat: np.ndarray,
    coh_ps: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage5_duplicate_keep")
    if not stage5_duplicate_keep_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage5_duplicate_keep",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(lonlat, coh_ps, threads)


def run_stage5_rc2_correction_kernel(
    ph2: np.ndarray,
    ph_patch: np.ndarray,
    bperp: np.ndarray,
    k_ps: np.ndarray,
    c_ps: np.ndarray,
    *,
    small_baseline: bool,
    master_ix: int,
    backend: str = "auto",
    threads: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage5_rc2_correction")
    if not stage5_rc2_correction_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage5_rc2_correction",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph2, ph_patch, bperp, k_ps, c_ps, small_baseline, master_ix, threads)


def run_stage5_format_merged_rc2_kernel(
    rc2_all: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage5_format_merged_rc2")
    if not stage5_format_merged_rc2_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage5_format_merged_rc2",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(rc2_all, threads)


def run_stage5_patch_keep_mask_kernel(
    ij_cols: np.ndarray,
    merged_ij_cols: np.ndarray,
    merged_indices: np.ndarray,
    patch_bounds: np.ndarray | None = None,
    backend: str = "auto",
    threads: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage5_patch_keep_mask")
    if not stage5_patch_keep_mask_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage5_patch_keep_mask",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ij_cols, merged_ij_cols, merged_indices, patch_bounds, threads)


def run_stage6_unwrap_grid_kernel(
    ifgw: np.ndarray,
    rowcost: np.ndarray,
    colcost: np.ndarray,
    backend: str = "auto",
    nshortcycle: float = 200.0,
    threads: int = 0,
) -> dict[str, np.ndarray | float]:
    implementations = DEFAULT_REGISTRY.implementations("stage6_unwrap_grid")
    if not stage6_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_unwrap_grid",
        backend,
        auto_order=("native",),
        implementations=implementations,
    )
    return resolved.fn(ifgw, rowcost, colcost, nshortcycle, threads)


def run_stage6_extract_grid_values_kernel(
    ifguw: np.ndarray,
    nzix: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage6_extract_grid_values")
    if not stage6_extract_grid_values_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_extract_grid_values",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ifguw, nzix, threads)


def run_stage6_prepare_cost_offsets_kernel(
    rowcost_base: np.ndarray,
    colcost_base: np.ndarray,
    rowix: np.ndarray,
    colix: np.ndarray,
    wrapped_space_uw: np.ndarray,
    dph_smooth: np.ndarray,
    nshortcycle: float = 200.0,
    backend: str = "auto",
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage6_prepare_cost_offsets")
    if not stage6_prepare_cost_offsets_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_prepare_cost_offsets",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(rowcost_base, colcost_base, rowix, colix, wrapped_space_uw, dph_smooth, nshortcycle, threads)


def run_stage6_reconstruct_ps_phase_kernel(
    ph_uw_grid: np.ndarray,
    ps_grid_idx: np.ndarray,
    ph_in: np.ndarray,
    phase_restore: np.ndarray | None = None,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage6_reconstruct_ps_phase")
    if not stage6_reconstruct_ps_phase_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_reconstruct_ps_phase",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph_uw_grid, ps_grid_idx, ph_in, phase_restore, threads)


def run_stage6_ps_grid_indices_kernel(
    nzix: np.ndarray,
    grid_ij: np.ndarray,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage6_ps_grid_indices")
    if not stage6_ps_grid_indices_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_ps_grid_indices",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(nzix, grid_ij, threads)


def run_stage6_select_ifgw_kernel(
    uw_ph: np.ndarray,
    z: np.ndarray,
    ifg_ix: int,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage6_select_ifgw")
    if not stage6_select_ifgw_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_select_ifgw",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(uw_ph, z, ifg_ix, threads)


def run_stage6_grid_accumulate_kernel(
    ph_in: np.ndarray,
    grid_lin: np.ndarray,
    n_cells: int,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage6_grid_accumulate")
    if not stage6_grid_accumulate_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_grid_accumulate",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(ph_in, grid_lin, n_cells, threads)


def run_stage6_unwrap_ifg_sets_kernel(
    n_ifg: int,
    master_ix: int,
    drop_ifg_index: np.ndarray,
    small_baseline: bool,
    backend: str = "auto",
    threads: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage6_unwrap_ifg_sets")
    if not stage6_unwrap_ifg_sets_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_unwrap_ifg_sets",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(n_ifg, master_ix, drop_ifg_index, small_baseline, threads)


def run_stage6_single_master_ifg_geometry_kernel(
    n_ifg: int,
    master_ix: int,
    backend: str = "auto",
    threads: int = 0,
) -> dict[str, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage6_single_master_ifg_geometry")
    if not stage6_single_master_ifg_geometry_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_single_master_ifg_geometry",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(n_ifg, master_ix, threads)


def run_stage6_estimate_la_error_kernel(
    dph_space: np.ndarray,
    day: np.ndarray,
    bperp: np.ndarray,
    n_trial_wraps: float,
    backend: str = "auto",
    threads: int = 0,
) -> np.ndarray:
    implementations = DEFAULT_REGISTRY.implementations("stage6_estimate_la_error")
    if not stage6_estimate_la_error_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_estimate_la_error",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(dph_space, day, bperp, n_trial_wraps, threads)


def run_stage6_smooth_3d_full_single_master_kernel(
    dph_space: np.ndarray,
    day: np.ndarray,
    time_win: float,
    backend: str = "auto",
    chunk_edges: int = 32768,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    implementations = DEFAULT_REGISTRY.implementations("stage6_smooth_3d_full_single_master")
    if not stage6_smooth_3d_full_single_master_native_available():
        implementations.pop("native", None)
    resolved = _resolve_generic_kernel(
        "stage6_smooth_3d_full_single_master",
        backend,
        auto_order=("native", "python"),
        implementations=implementations,
    )
    return resolved.fn(dph_space, day, time_win, chunk_edges, threads)


def describe_backend_matrix() -> dict[str, Any]:
    manifest = DEFAULT_REGISTRY.coverage_manifest()
    kernels = manifest.get("kernels", {})
    stage2_grid_indices_kernel = kernels.get("stage2_grid_indices")
    if stage2_grid_indices_kernel is not None and not stage2_grid_indices_native_available():
        stage2_grid_indices_kernel["available_backends"] = [
            backend for backend in stage2_grid_indices_kernel["available_backends"] if backend != "native"
        ]
    stage2_clap_kernel = kernels.get("stage2_clap_filter_kernel")
    if stage2_clap_kernel is not None and not stage2_clap_filter_kernel_native_available():
        stage2_clap_kernel["available_backends"] = [
            backend for backend in stage2_clap_kernel["available_backends"] if backend != "native"
        ]
    stage2_normalize_kernel = kernels.get("stage2_normalize_complex")
    if stage2_normalize_kernel is not None and not stage2_normalize_complex_native_available():
        stage2_normalize_kernel["available_backends"] = [
            backend for backend in stage2_normalize_kernel["available_backends"] if backend != "native"
        ]
    stage2_phase_norm_kernel = kernels.get("stage2_normalize_phase_matrix")
    if stage2_phase_norm_kernel is not None and not stage2_normalize_phase_matrix_native_available():
        stage2_phase_norm_kernel["available_backends"] = [
            backend for backend in stage2_phase_norm_kernel["available_backends"] if backend != "native"
        ]
    stage2_ph_weight_kernel = kernels.get("stage2_ph_weight_block")
    if stage2_ph_weight_kernel is not None and not stage2_ph_weight_block_native_available():
        stage2_ph_weight_kernel["available_backends"] = [
            backend for backend in stage2_ph_weight_kernel["available_backends"] if backend != "native"
        ]
    stage4_kernel = kernels.get("stage4_edge_stats")
    if stage4_kernel is not None and not stage4_native_available():
        stage4_kernel["available_backends"] = [backend for backend in stage4_kernel["available_backends"] if backend != "native"]
    stage4_duplicate_kernel = kernels.get("stage4_duplicate_keep")
    if stage4_duplicate_kernel is not None and not stage4_duplicate_keep_native_available():
        stage4_duplicate_kernel["available_backends"] = [
            backend for backend in stage4_duplicate_kernel["available_backends"] if backend != "native"
        ]
    stage4_adjacent_kernel = kernels.get("stage4_adjacent_component_keep")
    if stage4_adjacent_kernel is not None and not stage4_adjacent_component_keep_native_available():
        stage4_adjacent_kernel["available_backends"] = [
            backend for backend in stage4_adjacent_kernel["available_backends"] if backend != "native"
        ]
    stage4_ifg_kernel = kernels.get("stage4_weed_ifg_index")
    if stage4_ifg_kernel is not None and not stage4_weed_ifg_index_native_available():
        stage4_ifg_kernel["available_backends"] = [
            backend for backend in stage4_ifg_kernel["available_backends"] if backend != "native"
        ]
    stage4_phase_kernel = kernels.get("stage4_phase_correction")
    if stage4_phase_kernel is not None and not stage4_phase_correction_native_available():
        stage4_phase_kernel["available_backends"] = [
            backend for backend in stage4_phase_kernel["available_backends"] if backend != "native"
        ]
    stage3_select_kernel = kernels.get("stage3_select_ifg_index")
    if stage3_select_kernel is not None and not stage3_select_ifg_index_native_available():
        stage3_select_kernel["available_backends"] = [
            backend for backend in stage3_select_kernel["available_backends"] if backend != "native"
        ]
    stage3_clap_kernel = kernels.get("stage3_clap_filt_patch")
    if stage3_clap_kernel is not None and not stage3_clap_filt_patch_native_available():
        stage3_clap_kernel["available_backends"] = [
            backend for backend in stage3_clap_kernel["available_backends"] if backend != "native"
        ]
    stage3_clap_grid_kernel = kernels.get("stage3_clap_filt_grid")
    if stage3_clap_grid_kernel is not None and not stage3_clap_filt_grid_native_available():
        stage3_clap_grid_kernel["available_backends"] = [
            backend for backend in stage3_clap_grid_kernel["available_backends"] if backend != "native"
        ]
    stage3_clap_grid_stack_kernel = kernels.get("stage3_clap_filt_grid_stack")
    if stage3_clap_grid_stack_kernel is not None and not stage3_clap_filt_grid_stack_native_available():
        stage3_clap_grid_stack_kernel["available_backends"] = [
            backend for backend in stage3_clap_grid_stack_kernel["available_backends"] if backend != "native"
        ]
    stage3_wrap_kernel = kernels.get("stage3_wrap_filt")
    if stage3_wrap_kernel is not None and not stage3_wrap_filt_native_available():
        stage3_wrap_kernel["available_backends"] = [
            backend for backend in stage3_wrap_kernel["available_backends"] if backend != "native"
        ]
    stage3_wrap_global_kernel = kernels.get("stage3_wrap_filt_global")
    if stage3_wrap_global_kernel is not None and not stage3_wrap_filt_global_native_available():
        stage3_wrap_global_kernel["available_backends"] = [
            backend for backend in stage3_wrap_global_kernel["available_backends"] if backend != "native"
        ]
    stage3_thresh_kernel = kernels.get("stage3_coh_threshold")
    if stage3_thresh_kernel is not None and not stage3_coh_threshold_native_available():
        stage3_thresh_kernel["available_backends"] = [
            backend for backend in stage3_thresh_kernel["available_backends"] if backend != "native"
        ]
    stage5_kernel = kernels.get("stage5_ifg_std")
    if stage5_kernel is not None and not stage5_native_available():
        stage5_kernel["available_backends"] = [backend for backend in stage5_kernel["available_backends"] if backend != "native"]
    stage5_duplicate_kernel = kernels.get("stage5_duplicate_keep")
    if stage5_duplicate_kernel is not None and not stage5_duplicate_keep_native_available():
        stage5_duplicate_kernel["available_backends"] = [
            backend for backend in stage5_duplicate_kernel["available_backends"] if backend != "native"
        ]
    stage5_rc2_correction_kernel = kernels.get("stage5_rc2_correction")
    if stage5_rc2_correction_kernel is not None and not stage5_rc2_correction_native_available():
        stage5_rc2_correction_kernel["available_backends"] = [
            backend for backend in stage5_rc2_correction_kernel["available_backends"] if backend != "native"
        ]
    stage5_rc2_kernel = kernels.get("stage5_format_merged_rc2")
    if stage5_rc2_kernel is not None and not stage5_format_merged_rc2_native_available():
        stage5_rc2_kernel["available_backends"] = [
            backend for backend in stage5_rc2_kernel["available_backends"] if backend != "native"
        ]
    stage5_keep_kernel = kernels.get("stage5_patch_keep_mask")
    if stage5_keep_kernel is not None and not stage5_patch_keep_mask_native_available():
        stage5_keep_kernel["available_backends"] = [
            backend for backend in stage5_keep_kernel["available_backends"] if backend != "native"
        ]
    stage6_kernel = kernels.get("stage6_unwrap_grid")
    if stage6_kernel is not None and not stage6_native_available():
        stage6_kernel["available_backends"] = [backend for backend in stage6_kernel["available_backends"] if backend != "native"]
    stage6_extract_kernel = kernels.get("stage6_extract_grid_values")
    if stage6_extract_kernel is not None and not stage6_extract_grid_values_native_available():
        stage6_extract_kernel["available_backends"] = [
            backend for backend in stage6_extract_kernel["available_backends"] if backend != "native"
        ]
    stage6_cost_kernel = kernels.get("stage6_prepare_cost_offsets")
    if stage6_cost_kernel is not None and not stage6_prepare_cost_offsets_native_available():
        stage6_cost_kernel["available_backends"] = [
            backend for backend in stage6_cost_kernel["available_backends"] if backend != "native"
        ]
    stage6_reconstruct_kernel = kernels.get("stage6_reconstruct_ps_phase")
    if stage6_reconstruct_kernel is not None and not stage6_reconstruct_ps_phase_native_available():
        stage6_reconstruct_kernel["available_backends"] = [
            backend for backend in stage6_reconstruct_kernel["available_backends"] if backend != "native"
        ]
    stage6_grid_idx_kernel = kernels.get("stage6_ps_grid_indices")
    if stage6_grid_idx_kernel is not None and not stage6_ps_grid_indices_native_available():
        stage6_grid_idx_kernel["available_backends"] = [
            backend for backend in stage6_grid_idx_kernel["available_backends"] if backend != "native"
        ]
    stage6_select_kernel = kernels.get("stage6_select_ifgw")
    if stage6_select_kernel is not None and not stage6_select_ifgw_native_available():
        stage6_select_kernel["available_backends"] = [
            backend for backend in stage6_select_kernel["available_backends"] if backend != "native"
        ]
    stage6_grid_kernel = kernels.get("stage6_grid_accumulate")
    if stage6_grid_kernel is not None and not stage6_grid_accumulate_native_available():
        stage6_grid_kernel["available_backends"] = [
            backend for backend in stage6_grid_kernel["available_backends"] if backend != "native"
        ]
    stage6_ifg_sets_kernel = kernels.get("stage6_unwrap_ifg_sets")
    if stage6_ifg_sets_kernel is not None and not stage6_unwrap_ifg_sets_native_available():
        stage6_ifg_sets_kernel["available_backends"] = [
            backend for backend in stage6_ifg_sets_kernel["available_backends"] if backend != "native"
        ]
    stage6_single_master_kernel = kernels.get("stage6_single_master_ifg_geometry")
    if stage6_single_master_kernel is not None and not stage6_single_master_ifg_geometry_native_available():
        stage6_single_master_kernel["available_backends"] = [
            backend for backend in stage6_single_master_kernel["available_backends"] if backend != "native"
        ]
    stage6_la_kernel = kernels.get("stage6_estimate_la_error")
    if stage6_la_kernel is not None and not stage6_estimate_la_error_native_available():
        stage6_la_kernel["available_backends"] = [
            backend for backend in stage6_la_kernel["available_backends"] if backend != "native"
        ]
    stage6_smooth_kernel = kernels.get("stage6_smooth_3d_full_single_master")
    if stage6_smooth_kernel is not None and not stage6_smooth_3d_full_single_master_native_available():
        stage6_smooth_kernel["available_backends"] = [
            backend for backend in stage6_smooth_kernel["available_backends"] if backend != "native"
        ]
    stage7_kernel = kernels.get("stage7_scla")
    if stage7_kernel is not None and not stage7_native_available():
        stage7_kernel["available_backends"] = [backend for backend in stage7_kernel["available_backends"] if backend != "native"]
    stage7_mean_kernel = kernels.get("stage7_mean_velocity_fit")
    if stage7_mean_kernel is not None and not stage7_mean_velocity_fit_native_available():
        stage7_mean_kernel["available_backends"] = [
            backend for backend in stage7_mean_kernel["available_backends"] if backend != "native"
        ]
    stage7_deramp_kernel = kernels.get("stage7_deramp_unwrapped_phase")
    if stage7_deramp_kernel is not None and not stage7_deramp_unwrapped_phase_native_available():
        stage7_deramp_kernel["available_backends"] = [
            backend for backend in stage7_deramp_kernel["available_backends"] if backend != "native"
        ]
    stage7_center_kernel = kernels.get("stage7_center_to_reference")
    if stage7_center_kernel is not None and not stage7_center_to_reference_native_available():
        stage7_center_kernel["available_backends"] = [
            backend for backend in stage7_center_kernel["available_backends"] if backend != "native"
        ]
    stage7_smooth_kernel = kernels.get("stage7_scla_smooth")
    if stage7_smooth_kernel is not None and not stage7_scla_smooth_native_available():
        stage7_smooth_kernel["available_backends"] = [
            backend for backend in stage7_smooth_kernel["available_backends"] if backend != "native"
        ]
    stage8_kernel = kernels.get("stage8_edge_noise")
    if stage8_kernel is not None and not stage8_native_available():
        stage8_kernel["available_backends"] = [backend for backend in stage8_kernel["available_backends"] if backend != "native"]
    stage8_lstsq_kernel = kernels.get("stage8_weighted_lstsq")
    if stage8_lstsq_kernel is not None and not stage8_weighted_lstsq_native_available():
        stage8_lstsq_kernel["available_backends"] = [
            backend for backend in stage8_lstsq_kernel["available_backends"] if backend != "native"
        ]
    weighted_affine_kernel = kernels.get("weighted_affine_fit")
    if weighted_affine_kernel is not None and not weighted_affine_fit_native_available():
        weighted_affine_kernel["available_backends"] = [
            backend for backend in weighted_affine_kernel["available_backends"] if backend != "native"
        ]
    weighted_slope_kernel = kernels.get("weighted_slope_fit")
    if weighted_slope_kernel is not None and not weighted_slope_fit_native_available():
        weighted_slope_kernel["available_backends"] = [
            backend for backend in weighted_slope_kernel["available_backends"] if backend != "native"
        ]
    return manifest
