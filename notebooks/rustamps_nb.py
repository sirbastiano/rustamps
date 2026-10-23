"""Shared helpers for the Rustamps notebooks.

Every notebook proves its claims by running the real `rustamps` binary on real
golden inputs and checking the written files with an independent reader
(h5py), never through the Rust code that produced them. Missing inputs fail
loudly: a notebook that cannot run its evidence is not a passing notebook.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HF_REPO = "mdelgadoblasco/InSAR_dataset_test"
HF_REVISION = "cabed239a0e7f5fe4db8db45ce6d17466dc2b800"
PATCH = "PATCH_1"

PATCH_INPUTS = [
    "pscands.1.ij", "pscands.1.ll", "pscands.1.ph", "pscands.1.da", "pscands.1.hgt",
    "pscands.1.ij0", "parms.mat", "patch.in", "patch_noover.in",
]
PATCH_GOLDEN = [
    "ps1.mat", "ph1.mat", "bp1.mat", "la1.mat", "da1.mat", "hgt1.mat", "pm1.mat",
    "select1.mat", "weed1.mat", "ps2.mat", "ph2.mat", "bp2.mat", "la2.mat",
    "hgt2.mat", "pm2.mat", "rc2.mat", "STAMPS.log",
]
STAGE_SECONDS = {}


def repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "Cargo.toml").exists() and (candidate / "crates").is_dir():
            return candidate
    raise RuntimeError("Could not locate the repository root (Cargo.toml + crates/)")


ROOT = repo_root()
GOLDEN = ROOT / "inputs_and_outputs" / "InSAR_dataset_test"
WORK = ROOT / ".tmp" / "notebook_runs"


def _works(path: Path) -> bool:
    try:
        return subprocess.run([str(path), "--version"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def binary() -> Path:
    """The rustamps executable: $RUSTAMPS_BIN, target/release, target/debug, PATH."""
    candidates = [os.environ.get("RUSTAMPS_BIN")]
    candidates += [ROOT / "target" / "release" / "rustamps", ROOT / "target" / "debug" / "rustamps"]
    candidates += [shutil.which("rustamps")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and _works(Path(candidate)):
            return Path(candidate).resolve()
    raise AssertionError(
        "No working rustamps binary. Run `cargo build --release --locked` or set RUSTAMPS_BIN."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cli(*args, check=True, timeout=3600):
    """Run the CLI. Returns (returncode, stdout, stderr, seconds)."""
    started = time.perf_counter()
    done = subprocess.run([str(binary()), *map(str, args)], capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    if check and done.returncode != 0:
        raise AssertionError(f"rustamps {' '.join(map(str, args))} exited {done.returncode}\n{done.stderr[-2000:]}")
    return done.returncode, done.stdout, done.stderr, elapsed


def cli_json(*args, **kwargs):
    rc, out, err, seconds = cli(*args, **kwargs)
    start = out.find("[") if out.lstrip().startswith("[") else out.find("{")
    return json.loads(out[start:]), seconds


def hf_hint(missing) -> str:
    return (
        f"Missing golden inputs: {sorted(map(str, missing))[:6]}{' ...' if len(missing) > 6 else ''}\n"
        f"Fetch them with:\n  hf download {HF_REPO} --repo-type dataset --revision {HF_REVISION} "
        f'--include "{PATCH}/*" --include "parms.mat" --include "patch.list" --include "rslc/*.par" '
        f'--include "diff0/*.base" --local-dir {GOLDEN}'
    )


def require_golden():
    needed = [GOLDEN / PATCH / name for name in PATCH_INPUTS + PATCH_GOLDEN if name != "pscands.1.ij0"]
    needed += [GOLDEN / "parms.mat"]
    needed += list((GOLDEN / "diff0").glob("*.base")) or [GOLDEN / "diff0" / "*.base"]
    needed += list((GOLDEN / "rslc").glob("*.par")) or [GOLDEN / "rslc" / "*.par"]
    missing = [p for p in needed if not p.exists()]
    assert not missing, hf_hint(missing)
    return GOLDEN


def stage_inputs(name: str, fresh=True) -> Path:
    """Copy only the inputs of PATCH_1 into a scratch dataset. Golden stays read-only."""
    require_golden()
    run = WORK / name
    if fresh and run.exists():
        shutil.rmtree(run)
    (run / PATCH).mkdir(parents=True, exist_ok=True)
    for sub, pattern in (("diff0", "*.base"), ("rslc", "*.par")):
        (run / sub).mkdir(exist_ok=True)
        for path in (GOLDEN / sub).glob(pattern):
            shutil.copy2(path, run / sub / path.name)
    shutil.copy2(GOLDEN / "parms.mat", run / "parms.mat")
    (run / "patch.list").write_text(PATCH + "\n")
    for file in PATCH_INPUTS:
        if (GOLDEN / PATCH / file).exists():
            shutil.copy2(GOLDEN / PATCH / file, run / PATCH / file)
    return run


def run_stage(run: Path, stage: int, workers: int = 4):
    """Run one stage through the public CLI; return (summary dict, seconds)."""
    payload, seconds = cli_json("run", "--dataset", run, "--start-step", stage, "--end-step", stage,
                                "--cpu-workers", workers)
    assert all(item["status"] == "completed" for item in payload), payload
    return payload, seconds


def run_chain(name="patch1", last_stage=5, force=False):
    """Stages 1..last_stage on PATCH_1, cached by binary hash. Returns (run dir, manifest)."""
    run = WORK / name
    manifest_path = run / "run_manifest.json"
    binary_hash = sha256(binary())
    if not force and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["binary_sha256"] == binary_hash and manifest["last_stage"] >= last_stage:
            return run, manifest
    run = stage_inputs(name)
    stages = {}
    for stage in range(1, last_stage + 1):
        payload, seconds = run_stage(run, stage)
        stages[stage] = {"seconds": round(seconds, 2), "details": [item["details"] for item in payload]}
    manifest = {"binary_sha256": binary_hash, "last_stage": last_stage, "stages": stages,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return run, manifest


def read_mat(path: Path) -> dict:
    """Read MATLAB v7.3 (HDF5) with h5py and v5 with scipy; Rust writes v5, golden is v7.3."""
    out = {}
    with open(path, "rb") as handle:
        is_hdf5 = b"MATLAB 7.3" in handle.read(32)
    if not is_hdf5:
        import scipy.io

        for key, value in scipy.io.loadmat(path, squeeze_me=False, mat_dtype=False).items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.dtype.kind in "US":
                value = "".join(value.ravel().tolist())
            out[key] = value
        return out
    with h5py.File(path, "r") as handle:
        for key, ds in handle.items():
            if key.startswith("#") or not isinstance(ds, h5py.Dataset):
                continue
            if ds.attrs.get("MATLAB_empty", 0):
                out[key] = np.zeros((0,))
                continue
            arr = ds[()]
            if arr.dtype.names and "real" in arr.dtype.names:
                arr = arr["real"] + 1j * arr["imag"]
            arr = np.asarray(arr).T
            cls = ds.attrs.get("MATLAB_class", b"")
            if (cls.decode() if isinstance(cls, bytes) else cls) == "char":
                arr = "".join(chr(c) for c in arr.ravel())
            out[key] = arr
    return out


def compare_mat(run_file: Path, golden_file: Path, skip=()) -> pd.DataFrame:
    run, gold = read_mat(run_file), read_mat(golden_file)
    rows = []
    for key in sorted(set(gold) - set(skip)):
        if key not in run:
            rows.append(dict(file=golden_file.name, variable=key, status="MISSING", n_diff=-1, max_abs=np.nan))
            continue
        r, g = run[key], gold[key]
        if isinstance(r, str) or isinstance(g, str):
            rows.append(dict(file=golden_file.name, variable=key, status="EXACT" if r == g else "DIFF",
                             n_diff=int(r != g), max_abs=np.nan))
            continue
        if r.shape != g.shape:
            if r.size == g.size:
                r = r.reshape(g.shape)
            else:
                rows.append(dict(file=golden_file.name, variable=key, status=f"SHAPE {r.shape} vs {g.shape}",
                                 n_diff=-1, max_abs=np.nan))
                continue
        r, g = r.astype(np.complex128 if np.iscomplexobj(r) or np.iscomplexobj(g) else np.float64), g
        g = g.astype(r.dtype)
        diff = np.abs(r - g)
        diff[np.isnan(r) & np.isnan(g)] = 0
        n = int(np.count_nonzero(diff))
        rows.append(dict(file=golden_file.name, variable=key, status="EXACT" if n == 0 else "DIFF",
                         n_diff=n, max_abs=float(np.nanmax(diff)) if diff.size else 0.0))
    return pd.DataFrame(rows)


def golden_log_counts() -> dict:
    """PS counts the original MATLAB run printed; an expectation independent of this repo."""
    text = (GOLDEN / PATCH / "STAMPS.log").read_text()
    grab = lambda pattern: int(re.search(pattern, text).group(1))
    return {
        "stage1_candidates": grab(r"PS_EST_GAMMA_QUICK\s+(\d+) PS candidates to process"),
        "stage3_initial": grab(r"PS_SELECT\s+(\d+) PS selected initially"),
        "stage3_final": grab(r"PS_SELECT\s+(\d+) ps selected after re-estimation"),
        "stage4_kept": grab(r"PS_WEED\s+(\d+) PS kept after dropping noisy pixels"),
    }


def check(condition: bool, claim: str, evidence: str = "") -> dict:
    """Record a claim and fail the notebook if it does not hold."""
    assert condition, f"FAILED: {claim}. {evidence}"
    return {"claim": claim, "evidence": evidence, "ok": True}


FLOAT_BOUNDS = {
    ("ps1.mat", "bperp"): 1e-3, ("ps2.mat", "bperp"): 1e-3,
    ("la1.mat", "la"): 1e-12, ("la2.mat", "la"): 1e-12,
    ("pm1.mat", "K_ps"): 3e-4, ("pm1.mat", "C_ps"): 1e-2, ("pm1.mat", "coh_ps"): 3e-4,
    ("pm1.mat", "coh_ps_save"): 3e-4, ("pm1.mat", "gamma_change_save"): 1e-6,
    ("pm1.mat", "Nr"): 1e-2, ("pm1.mat", "coh_bins"): 1e-12, ("pm1.mat", "low_pass"): 1e-12,
    ("pm1.mat", "ph_grid"): 0.15, ("pm1.mat", "ph_patch"): 2e-2, ("pm1.mat", "ph_res"): 0.1,
    ("pm1.mat", "ph_weight"): 0.1,
    ("select1.mat", "K_ps2"): 3e-4, ("select1.mat", "C_ps2"): 1e-2, ("select1.mat", "coh_ps2"): 3e-4,
    ("select1.mat", "coh_thresh"): 1e-5, ("select1.mat", "coh_thresh_coeffs"): 1e-5,
    ("select1.mat", "max_percent_rand"): 1e-4, ("select1.mat", "ph_patch2"): 3e-2,
    ("select1.mat", "ph_res2"): 0.1,
    ("weed1.mat", "ps_std"): 1e-3, ("weed1.mat", "ps_max"): 3e-2,
    ("pm2.mat", "K_ps"): 3e-4, ("pm2.mat", "C_ps"): 1e-2, ("pm2.mat", "coh_ps"): 3e-4,
    ("pm2.mat", "ph_patch"): 2e-2, ("pm2.mat", "ph_res"): 0.1,
    ("rc2.mat", "ph_reref"): 2e-2,
}
RC_PHASE_BOUND = 0.1


def evaluate(df: pd.DataFrame, bounds=None) -> pd.DataFrame:
    """Add `bound` and `pass` columns.

    Variables not listed in `bounds` must be exact (indices, masks, counts, coordinates, phase). The listed float
    bounds cover differences caused by MATLAB evaluating the topographic fit in single precision while Rust uses f64;
    they are about 2-3x the maximum difference measured on PATCH_1 and are not StaMPS tolerances. `rc2.ph_rc` is
    compared by phase (RC_PHASE_BOUND, radians) because its amplitude is SLC-scaled.
    """
    bounds = FLOAT_BOUNDS if bounds is None else bounds
    df = df.copy()
    df["bound"] = [bounds.get((f, v), 0.0) for f, v in zip(df.file, df.variable)]
    df["pass"] = [
        s == "EXACT" or (s == "DIFF" and m <= b and b > 0)
        for s, m, b in zip(df.status, df.max_abs, df.bound)
    ]
    return df


def compare_files(run: Path, files, bounds=None) -> pd.DataFrame:
    frames = []
    for name in files:
        frames.append(compare_mat(run / PATCH / name, GOLDEN / PATCH / name, skip=("ph_rc",) if name == "rc2.mat" else ()))
    out = evaluate(pd.concat(frames, ignore_index=True), bounds)
    if "rc2.mat" in files:
        a, b = read_mat(run / PATCH / "rc2.mat")["ph_rc"], read_mat(GOLDEN / PATCH / "rc2.mat")["ph_rc"]
        phase = np.abs(np.angle(a.astype(np.complex128) * np.conj(b.astype(np.complex128))))
        out = pd.concat([out, pd.DataFrame([dict(file="rc2.mat", variable="ph_rc (phase)", status="DIFF" if phase.max() else "EXACT",
                         n_diff=int(np.count_nonzero(phase)), max_abs=float(phase.max()), bound=RC_PHASE_BOUND,
                         **{"pass": bool(phase.max() <= RC_PHASE_BOUND)})])], ignore_index=True)
    return out


def must_pass(df: pd.DataFrame, what: str):
    bad = df[~df["pass"]]
    assert bad.empty, f"{what}: {len(bad)} variable(s) outside their declared bound\n{bad.to_string()}"


def perturb_copy(source: Path, target: Path, variable: str, delta: float = 1e-3) -> Path:
    """Copy a MAT-v7.3 file and add `delta` to the first element of `variable` (negative control)."""
    shutil.copy2(source, target)
    with h5py.File(target, "r+") as handle:
        flat = handle[variable]
        flat[(0,) * flat.ndim] = flat[(0,) * flat.ndim] + delta
    return target


FORBIDDEN_CRATES = ("pyo3", "numpy", "hdf5-sys", "libhdf5", "snaphu", "triangle", "openblas", "lapack", "fftw")
ALLOWED_LIBS = ("linux-vdso", "libgcc_s", "libm.", "libc.", "ld-linux", "libdl", "libpthread", "librt")


def runtime_boundary() -> pd.DataFrame:
    rows = []
    payload, _ = cli_json("describe-backends")
    rows.append(check(payload["runtime_external_dependencies"] == [], "describe-backends lists no external runtime dependency",
                      json.dumps(payload["runtime_external_dependencies"])))
    rows.append(check(payload["providers"]["native"]["available"] is True, "native provider is available"))
    ldd = subprocess.run(["ldd", str(binary())], capture_output=True, text=True).stdout
    libs = [Path(line.split()[0]).name for line in ldd.splitlines() if line.strip()]
    extra = [lib for lib in libs if not lib.startswith(ALLOWED_LIBS)]
    rows.append(check(not extra, "binary links only the C runtime", ", ".join(libs)))
    lock = (ROOT / "Cargo.lock").read_text()
    names = set(re.findall(r'^name = "([^"]+)"', lock, re.M))
    hits = sorted(n for n in names if any(n == f or n.startswith(f + "-sys") for f in FORBIDDEN_CRATES))
    rows.append(check(not hits, "Cargo.lock has no Python, HDF5-C, SNAPHU, Triangle, BLAS or FFTW crates", f"{len(names)} crates; hits={hits}"))
    return pd.DataFrame(rows)
