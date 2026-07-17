from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse, spatial


def _as_indices(values: Sequence[int] | np.ndarray | None, size: int, name: str) -> np.ndarray:
    if values is None:
        return np.arange(size, dtype=np.int64)
    indices = np.unique(np.asarray(values, dtype=np.int64).reshape(-1))
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError(f"{name} contains an index outside [0, {size})")
    return indices


def _temporal_weights(day: np.ndarray, master_index: int | None, time_window: float) -> np.ndarray:
    scaled = -0.5 * ((day[:, None] - day[None, :]) / time_window) ** 2
    if master_index is not None:
        scaled[:, master_index] = -np.inf
    row_max = np.max(scaled, axis=1, keepdims=True)
    if np.any(~np.isfinite(row_max)):
        raise ValueError("temporal filtering needs at least one non-master interferogram")
    weights = np.exp(scaled - row_max)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def _temporal_high_pass(
    phase: np.ndarray,
    day: np.ndarray,
    master_index: int | None,
    time_window: float,
) -> np.ndarray:
    weights = _temporal_weights(day, master_index, time_window)
    high_pass = phase @ weights.T
    high_pass *= -1.0
    high_pass += phase
    # The legacy incidence-matrix solve fixes the first PS to zero. For a
    # connected triangulation it is exactly this reference subtraction.
    high_pass -= high_pass[0].copy()
    return high_pass


def _spatial_low_pass(
    phase: np.ndarray,
    xy: np.ndarray,
    wavelength: float,
    chunk_points: int,
    workers: int,
) -> np.ndarray:
    tree = spatial.cKDTree(xy)
    radius = 4.0 * wavelength
    radius_sq = radius * radius
    denominator = 2.0 * wavelength * wavelength
    output = np.empty(phase.shape, dtype=np.float64)

    for start in range(0, xy.shape[0], chunk_points):
        stop = min(start + chunk_points, xy.shape[0])
        neighbors = tree.query_ball_point(xy[start:stop], radius, workers=workers)
        counts = np.fromiter((len(item) for item in neighbors), dtype=np.int64, count=stop - start)
        indices = np.concatenate(neighbors).astype(np.int64, copy=False)
        rows = np.repeat(np.arange(stop - start, dtype=np.int64), counts)
        delta = xy[indices] - xy[start + rows]
        dist_sq = np.einsum("ij,ij->i", delta, delta)

        # MATLAB uses a strict distance comparison after its bounding-box query.
        keep = dist_sq < radius_sq
        indices = indices[keep]
        rows = rows[keep]
        dist_sq = dist_sq[keep]
        kept_counts = np.bincount(rows, minlength=stop - start)
        if np.any(kept_counts == 0):
            raise ValueError("spatial filtering found a PS without itself as a neighbor")

        weights = np.exp(-dist_sq / denominator)
        sums = np.bincount(rows, weights=weights, minlength=stop - start)
        weights /= np.repeat(sums, kept_counts)
        indptr = np.concatenate((np.asarray([0]), np.cumsum(kept_counts)))
        matrix = sparse.csr_matrix(
            (weights, indices, indptr),
            shape=(stop - start, xy.shape[0]),
        )
        output[start:stop] = matrix @ phase

    return output


def build_scn_payload(
    ph_uw: np.ndarray,
    xy: np.ndarray,
    day: np.ndarray,
    *,
    master_index: int,
    unwrap_indices: Sequence[int] | np.ndarray | None = None,
    time_window: float = 365.0,
    wavelength: float = 100.0,
    ph_scla: np.ndarray | None = None,
    c_ps_uw: np.ndarray | None = None,
    scla_ramp: np.ndarray | None = None,
    deramp_indices: Sequence[int] | np.ndarray | None = None,
    chunk_points: int = 4096,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    """Build the three arrays written by legacy ``ps_scn_filt`` to ``scn2.mat``.

    Indices are zero-based. ``xy`` contains only the spatial x/y columns; all
    correction matrices use the full ``ph_uw`` shape.
    """
    phase_full = np.asarray(ph_uw)
    coords = np.asarray(xy, dtype=np.float64)
    days = np.asarray(day, dtype=np.float64).reshape(-1)
    if phase_full.ndim != 2 or phase_full.shape[0] == 0 or phase_full.shape[1] == 0:
        raise ValueError("ph_uw must be a non-empty PS-by-interferogram matrix")
    n_ps, n_ifg = phase_full.shape
    if coords.shape != (n_ps, 2) or np.any(~np.isfinite(coords)):
        raise ValueError("xy must be a finite n_ps-by-2 matrix")
    if days.size != n_ifg or np.any(~np.isfinite(days)):
        raise ValueError("day must contain one finite value per interferogram")
    if not 0 <= master_index < n_ifg:
        raise ValueError("master_index is outside ph_uw")
    if not np.isfinite(time_window) or time_window <= 0:
        raise ValueError("time_window must be finite and positive")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive")
    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")

    unwrap = _as_indices(unwrap_indices, n_ifg, "unwrap_indices")
    if unwrap.size == 0:
        raise ValueError("unwrap_indices must select at least one interferogram")
    full_unwrap = unwrap.size == n_ifg
    phase = np.array(phase_full, dtype=np.float64, copy=True) if full_unwrap else np.asarray(phase_full[:, unwrap], dtype=np.float64)
    for correction, name in ((ph_scla, "ph_scla"), (scla_ramp, "scla_ramp")):
        if correction is not None:
            correction_array = np.asarray(correction)
            if correction_array.shape != phase_full.shape:
                raise ValueError(f"{name} must match ph_uw")
            phase -= correction_array if full_unwrap else correction_array[:, unwrap]
    if c_ps_uw is not None:
        constant = np.asarray(c_ps_uw, dtype=np.float64).reshape(-1)
        if constant.size != n_ps:
            raise ValueError("c_ps_uw must contain one value per PS")
        phase -= constant[:, None]
    phase[np.isnan(phase)] = 0.0
    if np.any(~np.isfinite(phase)):
        raise ValueError("corrected unwrapped phase contains infinity")

    deramp_full = _as_indices(deramp_indices, n_ifg, "deramp_indices") if deramp_indices is not None else np.empty(0, dtype=np.int64)
    deramp_full = np.intersect1d(deramp_full, unwrap, assume_unique=True)
    deramp_local = np.searchsorted(unwrap, deramp_full)
    ramps = np.empty((n_ps, deramp_local.size), dtype=np.float64)
    if deramp_local.size:
        design = np.column_stack((np.ones(n_ps), coords))
        for ramp_column, phase_column in enumerate(deramp_local):
            coefficients = np.linalg.lstsq(design, phase[:, phase_column], rcond=None)[0]
            ramps[:, ramp_column] = design @ coefficients
            phase[:, phase_column] -= ramps[:, ramp_column]

    master_positions = np.flatnonzero(unwrap == master_index)
    local_master = int(master_positions[0]) if master_positions.size else None
    ph_hpt = _temporal_high_pass(phase, days[unwrap], local_master, time_window)
    if deramp_local.size:
        ph_hpt[:, deramp_local] += ramps
    ph_hpt = ph_hpt.astype(np.float32)

    ph_scn = _spatial_low_pass(ph_hpt, coords, wavelength, chunk_points, workers)
    ph_scn -= ph_scn[0]
    if full_unwrap:
        ph_scn_slave = ph_scn
    else:
        ph_scn_slave = np.zeros(phase_full.shape, dtype=np.float64)
        ph_scn_slave[:, unwrap] = ph_scn
    ph_scn_slave[:, master_index] = 0.0
    return {
        "ph_scn_slave": ph_scn_slave,
        "ph_hpt": ph_hpt,
        "ph_ramp": ramps,
    }
