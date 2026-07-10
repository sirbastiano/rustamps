from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class MtPrepError(RuntimeError):
    """Raised when a SNAP StaMPS export cannot be prepared for pySTAMPS."""


@dataclass(slots=True)
class MtPrepSummary:
    dataset_root: Path
    patch_count: int
    candidate_count: int
    patch_rows: list[dict[str, Any]]


_PAIR_RE = re.compile(r"(?P<master>\d{8})_(?P<slave>\d{8})")


def _parse_par(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        values[key.strip()] = value.strip().split()[0] if value.strip() else ""
    return values


def _par_int(path: Path, *keys: str) -> int | None:
    values = _parse_par(path)
    for key in keys:
        raw = values.get(key)
        if raw:
            return int(round(float(raw)))
    return None


def _dataset_shape(root: Path) -> tuple[int, int]:
    width_file = root / "width.txt"
    len_file = root / "len.txt"
    if width_file.exists() and len_file.exists():
        return int(width_file.read_text().strip()), int(len_file.read_text().strip())

    par_files = sorted((root / "rslc").glob("*.rslc.par"))
    if not par_files:
        raise MtPrepError(f"No rslc/*.rslc.par files found under {root}")
    width = _par_int(par_files[0], "range_samples", "width")
    length = _par_int(par_files[0], "azimuth_lines", "nlines")
    if width is None or length is None:
        raise MtPrepError(f"Unable to parse raster shape from {par_files[0]}")
    width_file.write_text(f"{width}\n", encoding="utf-8")
    len_file.write_text(f"{length}\n", encoding="utf-8")
    return width, length


def _resolve_master(root: Path, master_date: str | None) -> str:
    if master_date:
        return master_date
    name_match = re.search(r"INSAR_(\d{8})", root.name)
    if name_match:
        return name_match.group(1)
    masters = {
        match.group("master")
        for path in (root / "diff0").glob("*.diff")
        for match in [_PAIR_RE.search(path.name)]
        if match is not None
    }
    if len(masters) == 1:
        return masters.pop()
    raise MtPrepError("Pass master_date or use a dataset name like INSAR_YYYYMMDD")


def _rslc_files(root: Path) -> list[Path]:
    files = sorted((root / "rslc").glob("*.rslc"))
    if not files:
        raise MtPrepError(f"No rslc/*.rslc files found under {root}")
    return files


def _diff_files(root: Path, master_date: str) -> list[Path]:
    pairs: list[tuple[str, Path]] = []
    for path in sorted((root / "diff0").glob("*.diff")):
        match = _PAIR_RE.search(path.name)
        if match is None or match.group("master") != master_date:
            continue
        pairs.append((match.group("slave"), path))
    if not pairs:
        raise MtPrepError(f"No diff0/{master_date}_*.diff files found under {root}")
    return [path for _, path in sorted(pairs)]


def _memmap(path: Path, dtype: str, shape: tuple[int, int]) -> np.memmap:
    expected = int(np.dtype(dtype).itemsize * shape[0] * shape[1])
    if path.stat().st_size != expected:
        raise MtPrepError(f"Unexpected raster size for {path}: expected {expected} bytes")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _candidate_arrays(
    root: Path,
    master: str,
    rslc_files: list[Path],
    shape: tuple[int, int],
    amp_dispersion: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_amp = np.zeros(shape, dtype=np.float64)
    sum_sq = np.zeros(shape, dtype=np.float64)
    for path in rslc_files:
        amp = np.abs(_memmap(path, ">c8", shape)).astype(np.float64)
        sum_amp += amp
        sum_sq += amp * amp

    count = float(len(rslc_files))
    mean_amp = sum_amp / count
    var_amp = np.maximum(sum_sq / count - mean_amp * mean_amp, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        da = np.sqrt(var_amp) / mean_amp

    lon = _memmap(root / "geo" / f"{master}.lon", ">f4", shape)
    lat = _memmap(root / "geo" / f"{master}.lat", ">f4", shape)
    hgt = _memmap(root / "geo" / "elevation_dem.rdc", ">f4", shape)
    finite_geo = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(hgt)
    mask = finite_geo & np.isfinite(da) & (mean_amp > 0.0) & (da <= float(amp_dispersion))
    return mask, da.astype(np.float32), mean_amp.astype(np.float32)


def _ranges(size: int, count: int, overlap: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if count <= 0:
        raise MtPrepError("Patch count must be positive")
    edges = np.floor(np.linspace(1, size + 1, count + 1)).astype(int)
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i in range(count):
        no_start = int(edges[i])
        no_end = int(edges[i + 1] - 1)
        if i == count - 1:
            no_end = size
        patch_start = max(1, no_start - int(overlap))
        patch_end = min(size, no_end + int(overlap))
        out.append(((patch_start, patch_end), (no_start, no_end)))
    return out


def _remove_existing_patches(root: Path) -> None:
    for path in root.glob("PATCH_*"):
        if path.is_dir():
            shutil.rmtree(path)
    patch_list = root / "patch.list"
    if patch_list.exists():
        patch_list.unlink()


def _write_text_vector(path: Path, values: np.ndarray, fmt: str) -> None:
    if values.size == 0:
        path.write_text("", encoding="utf-8")
    else:
        np.savetxt(path, values, fmt=fmt)


def _write_phase(
    path: Path,
    diff_files: list[Path],
    shape: tuple[int, int],
    rows: np.ndarray,
    cols: np.ndarray,
) -> None:
    out = np.empty((len(diff_files), rows.size * 2), dtype=">f4")
    for idx, diff_file in enumerate(diff_files):
        raster = _memmap(diff_file, ">c8", shape)
        values = np.asarray(raster[rows, cols], dtype=np.complex64)
        out[idx, 0::2] = values.real.astype(">f4", copy=False)
        out[idx, 1::2] = values.imag.astype(">f4", copy=False)
    out.tofile(path)


def _write_patch(
    patch: Path,
    bounds: tuple[int, int, int, int],
    noover: tuple[int, int, int, int],
    selected_rows: np.ndarray,
    selected_cols: np.ndarray,
    rasters: dict[str, np.ndarray],
    diff_files: list[Path],
    shape: tuple[int, int],
) -> int:
    c0, c1, r0, r1 = bounds
    within = (
        (selected_cols + 1 >= c0)
        & (selected_cols + 1 <= c1)
        & (selected_rows + 1 >= r0)
        & (selected_rows + 1 <= r1)
    )
    rows = selected_rows[within]
    cols = selected_cols[within]
    if rows.size == 0:
        return 0

    patch.mkdir(parents=True, exist_ok=True)
    (patch / "patch.in").write_text("\n".join(str(v) for v in bounds) + "\n", encoding="utf-8")
    (patch / "patch_noover.in").write_text("\n".join(str(v) for v in noover) + "\n", encoding="utf-8")

    ids = np.arange(1, rows.size + 1, dtype=np.int64)
    ij = np.column_stack((ids, rows, cols))
    np.savetxt(patch / "pscands.1.ij", ij, fmt="%d")
    np.column_stack((cols, rows)).astype(">i4").tofile(patch / "pscands.1.ij.int")
    (patch / "pscands.1.ij0").write_text("", encoding="utf-8")

    lonlat = np.column_stack((rasters["lon"][rows, cols], rasters["lat"][rows, cols])).astype(">f4")
    lonlat.tofile(patch / "pscands.1.ll")
    np.asarray(rasters["hgt"][rows, cols], dtype=">f4").tofile(patch / "pscands.1.hgt")
    np.asarray(rasters["mean_amp"][rows, cols], dtype=">f4").tofile(patch / "mean_amp.flt")
    _write_text_vector(patch / "pscands.1.da", rasters["da"][rows, cols], "%.8g")
    _write_phase(patch / "pscands.1.ph", diff_files, shape, rows, cols)
    return int(rows.size)


def prepare_snap_mt_prep_inputs(
    dataset_root: str | Path,
    *,
    master_date: str | None = None,
    amp_dispersion: float = 0.4,
    range_patches: int = 1,
    azimuth_patches: int = 1,
    range_overlap: int = 50,
    azimuth_overlap: int = 50,
    force: bool = False,
) -> MtPrepSummary:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise MtPrepError(f"Dataset root does not exist: {root}")
    if force:
        _remove_existing_patches(root)

    width, length = _dataset_shape(root)
    shape = (length, width)
    master = _resolve_master(root, master_date)
    rslc = _rslc_files(root)
    diff = _diff_files(root, master)
    mask, da, mean_amp = _candidate_arrays(root, master, rslc, shape, float(amp_dispersion))

    lon = _memmap(root / "geo" / f"{master}.lon", ">f4", shape)
    lat = _memmap(root / "geo" / f"{master}.lat", ">f4", shape)
    hgt = _memmap(root / "geo" / "elevation_dem.rdc", ">f4", shape)
    selected_rows, selected_cols = np.nonzero(mask)
    col_ranges = _ranges(width, int(range_patches), int(range_overlap))
    row_ranges = _ranges(length, int(azimuth_patches), int(azimuth_overlap))

    patch_rows: list[dict[str, Any]] = []
    names: list[str] = []
    patch_index = 1
    rasters = {"lon": lon, "lat": lat, "hgt": hgt, "da": da, "mean_amp": mean_amp}
    for col_range, col_noover in col_ranges:
        for row_range, row_noover in row_ranges:
            patch_name = f"PATCH_{patch_index}"
            bounds = (col_range[0], col_range[1], row_range[0], row_range[1])
            noover = (col_noover[0], col_noover[1], row_noover[0], row_noover[1])
            count = _write_patch(root / patch_name, bounds, noover, selected_rows, selected_cols, rasters, diff, shape)
            if count:
                names.append(patch_name)
                patch_rows.append({"patch": patch_name, "candidates": count, "bounds": bounds, "noover": noover})
            patch_index += 1

    if not names:
        raise MtPrepError("No candidates passed the amplitude-dispersion threshold")
    (root / "patch.list").write_text("\n".join(names) + "\n", encoding="utf-8")
    return MtPrepSummary(root, len(names), int(sum(row["candidates"] for row in patch_rows)), patch_rows)
