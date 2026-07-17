from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat


class MatReadError(RuntimeError):
    """Raised for unsupported MAT formats."""


def _decode_h5_dataset(obj: Any, h5file: Any) -> Any:
    import h5py  # type: ignore

    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        arr = np.asarray(arr)

        # MATLAB complex arrays in v7.3 often appear as compound datasets.
        if arr.dtype.names and {"real", "imag"}.issubset(set(arr.dtype.names)):
            arr = arr["real"] + 1j * arr["imag"]

        # Dereference cell/object datasets recursively.
        if arr.dtype.kind == "O":
            out = np.empty(arr.shape, dtype=object)
            for idx, ref in np.ndenumerate(arr):
                out[idx] = _decode_h5_dataset(h5file[ref], h5file)
            arr = out

        # MATLAB stores arrays in column-major order; h5py exposes reversed axes.
        if arr.ndim >= 2:
            arr = np.transpose(arr, axes=tuple(reversed(range(arr.ndim))))
        return arr

    if isinstance(obj, h5py.Group):
        data: dict[str, Any] = {}
        for key in obj.keys():
            data[key] = _decode_h5_dataset(obj[key], h5file)
        return data

    return obj


def read_mat(path: str | Path) -> dict[str, Any]:
    mat_path = Path(path)
    try:
        payload = loadmat(mat_path, simplify_cells=True)
    except NotImplementedError:
        try:
            import mat73  # type: ignore

            payload = mat73.loadmat(str(mat_path))
            if isinstance(payload, dict) and all(value is not None for value in payload.values()):
                return payload
        except Exception:
            pass

        try:
            import h5py  # type: ignore
        except ImportError as import_exc:
            raise MatReadError(
                f"MAT v7.3 file requires h5py: {mat_path}. Install h5py or convert file format."
            ) from import_exc

        data: dict[str, Any] = {}
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                data[key] = _decode_h5_dataset(f[key], f)
        return data
    return {k: v for k, v in payload.items() if not k.startswith("__")}


def write_mat(path: str | Path, payload: dict[str, Any]) -> None:
    savemat(Path(path), payload)
