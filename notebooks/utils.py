from __future__ import annotations
import ast
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import wkt
from pystamps.io.dataset import infer_patch_stage
from pystamps.io.mat import read_mat
from pystamps.notebooks import load_velocity_diagnostics
def install_notebook_globals(scope: dict[str, Any]) -> None:
    for key, value in scope.items():
        if key.startswith("__") or callable(value):
            continue
        globals()[key] = value
# --- notebook pipeline helpers ---
# Notebook pipeline helpers moved out of the ipynb.
def load_cdse_env_file() -> Path | None:
    candidates = [os.getenv("SARPYX_PYSTAMPS_CREDENTIAL_FILE"), "/tmp/sarpyx_pystamps_cdse_env.sh"]
    allowed = {"CDSE_USERNAME", "CDSE_PASSWORD", "CDSE_USR", "CDSE_PSW"}
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parts = shlex.split(line, posix=True)
            except ValueError:
                continue
            if parts and parts[0] == "export":
                parts = parts[1:]
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                if key in allowed and value and not os.getenv(key):
                    os.environ[key] = value
        return path
    return None

def first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None

def display_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    value = Path(path)
    try:
        return value.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()

def ensure_runtime_dirs() -> None:
    for path in (BURST_DIR, OUTPUT_DIR, STATS_DIR, PROCESS_DIR, EXPORT_DIR, TOOL_SHIM_DIR, SNAP_USERDIR, SNAP_TMPDIR):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = SNAP_TMPDIR.as_posix()
    java_tmp_opt = f"-Djava.io.tmpdir={SNAP_TMPDIR.as_posix()}"
    existing = os.getenv("JAVA_TOOL_OPTIONS", "").strip()
    if java_tmp_opt not in existing.split():
        os.environ["JAVA_TOOL_OPTIONS"] = f"{existing} {java_tmp_opt}".strip()

def prepend_path(path: Path) -> None:
    current = os.getenv("PATH", "")
    parts = current.split(os.pathsep) if current else []
    value = path.as_posix()
    if value not in parts:
        os.environ["PATH"] = os.pathsep.join([value, *parts])

def ensure_gawk_on_path() -> Path | None:
    existing = shutil.which("gawk")
    if existing:
        return Path(existing).resolve()
    awk = shutil.which("awk")
    if not awk:
        return None
    TOOL_SHIM_DIR.mkdir(parents=True, exist_ok=True)
    shim = TOOL_SHIM_DIR / "gawk"
    awk_quoted = shlex.quote(Path(awk).resolve().as_posix())
    shim.write_text(f"#!/bin/sh\nexec {awk_quoted} \"$@\"\n", encoding="utf-8")
    shim.chmod(0o755)
    prepend_path(TOOL_SHIM_DIR)
    return shim.resolve()

def configure_stamps_environment() -> None:
    if STAMPS_ROOT is not None:
        os.environ.setdefault("STAMPS_ROOT", STAMPS_ROOT.as_posix())
        os.environ.setdefault("STAMPS", STAMPS_ROOT.as_posix())

def content_start(row: pd.Series) -> pd.Timestamp:
    raw = row.get("ContentDate")
    if isinstance(raw, dict):
        raw = raw.get("Start")
    if raw is None:
        raw = row.get("OriginDate")
    return pd.to_datetime(raw, utc=True)

def yyyymmdd(row: pd.Series) -> str:
    return content_start(row).strftime("%Y%m%d")

def resolve_executable(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return path.resolve()
        found = shutil.which(str(candidate))
        if found:
            return Path(found).resolve()
    return None

def resolve_mt_prep_snap() -> Path | None:
    stamps_root = Path(os.getenv("STAMPS_ROOT")) if os.getenv("STAMPS_ROOT") else STAMPS_ROOT
    return resolve_executable(
        os.getenv("MT_PREP_SNAP"),
        stamps_root / "bin" / "mt_prep_snap" if stamps_root else None,
        STAMPS_ROOT_DEFAULT / "bin" / "mt_prep_snap",
        "mt_prep_snap",
    )

def gpt_looks_like_snap(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "not found"
    try:
        completed = subprocess.run(
            [path.as_posix(), "-h"],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception as exc:
        return False, str(exc)
    text = f"{completed.stdout}\n{completed.stderr}"
    ok = "gpt <op>|<graph-file>" in text or "SNAP" in text or "Graph Processing Tool" in text
    return ok, "SNAP GPT help detected" if ok else "executable did not look like ESA SNAP gpt"

def _platform_name(row: pd.Series) -> str:
    name = str(row.get("ParentProductName") or row.get("Name") or "")
    return name[:3]

def _pick_evenly_spaced(group: pd.DataFrame, target_size: int) -> pd.DataFrame:
    if target_size <= 0 or len(group) <= target_size:
        return group.copy()
    ix = np.linspace(0, len(group) - 1, num=target_size, dtype=int)
    return group.iloc[sorted(set(ix))].copy()

def select_same_burst_stack(df: pd.DataFrame, *, min_stack_size: int, target_stack_size: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    missing = [column for column in ["Id", *STACK_GROUP_COLUMNS] if column not in df.columns]
    if missing:
        raise ValueError("Search results missing required columns: " + ", ".join(missing))

    work = df.copy()
    work["_content_start"] = work.apply(content_start, axis=1)
    work["_platform"] = work.apply(_platform_name, axis=1)
    if REQUIRE_SINGLE_PLATFORM:
        work = work[work["_platform"].eq("S1A")].copy()
    if work.empty:
        raise ValueError("No search rows remain after the Sentinel-1A platform filter")

    ranked: list[dict[str, object]] = []
    for key, group in work.groupby(STACK_GROUP_COLUMNS, dropna=False):
        ordered = group.sort_values("_content_start").reset_index(drop=True)
        coverage = pd.to_numeric(ordered.get("coverage"), errors="coerce")
        ranked.append({
            "key": key,
            "count": int(len(ordered)),
            "coverage_mean": float(coverage.mean()) if coverage.notna().any() else float("nan"),
            "coverage_max": float(coverage.max()) if coverage.notna().any() else float("nan"),
            "date_min": ordered["_content_start"].min(),
            "date_max": ordered["_content_start"].max(),
            "group": ordered,
        })
    ranked.sort(key=lambda item: (item["count"], item["coverage_mean"] if np.isfinite(item["coverage_mean"]) else -1), reverse=True)
    best = ranked[0]
    if best["count"] < min_stack_size:
        raise ValueError(f"Best same-burst stack has {best['count']} acquisitions; need at least {min_stack_size}")

    selected = _pick_evenly_spaced(best["group"], target_stack_size).sort_values("_content_start").reset_index(drop=True)
    selected["date_yyyymmdd"] = selected["_content_start"].dt.strftime("%Y%m%d")
    selected["role"] = "secondary"
    master_ix = len(selected) // 2
    selected.loc[master_ix, "role"] = "master"
    selected["stack_index"] = np.arange(1, len(selected) + 1)
    return selected

def _candidate_safe_roots() -> list[Path]:
    roots = [
        BURST_DIR / "extracted",
        SARPYX_ROOT / "data" / "bursts" / "extracted",
        SARPYX_ROOT / "data" / "full_products" / "phidown",
        SARPYX_ROOT / "data" / "full_products" / "out" / "tiles",
    ]
    extra_roots = [value for value in os.getenv("SARPYX_PYSTAMPS_SAFE_CACHE", "").split(os.pathsep) if value]
    roots.extend(Path(value).expanduser() for value in extra_roots)
    return [root for root in roots if root.exists()]

def safe_cache_index() -> list[Path]:
    global SAFE_CACHE_INDEX
    if SAFE_CACHE_INDEX is None:
        safes: set[Path] = set()
        for root in _candidate_safe_roots():
            safes.update(path.resolve() for path in root.rglob("*.SAFE") if path.is_dir())
        SAFE_CACHE_INDEX = sorted(safes, key=lambda value: value.as_posix())
        print(f"Indexed {len(SAFE_CACHE_INDEX)} cached SAFE products from {len(_candidate_safe_roots())} cache roots.")
    return SAFE_CACHE_INDEX

def _safe_matches_row(safe_dir: Path, row: pd.Series) -> bool:
    text = safe_dir.as_posix()
    burst_uuid = str(row.get("Id", ""))
    product_name = str(row.get("Name", ""))
    normalized_product_name = product_name.replace("-", "_")
    if burst_uuid and burst_uuid in text:
        return True
    if product_name and (product_name in text or normalized_product_name in text):
        return True
    required = [
        str(row.get("date_yyyymmdd", "")),
        str(row.get("BurstId", "")),
        str(row.get("SwathIdentifier", "")),
        str(row.get("PolarisationChannels", "")),
    ]
    return all(value and value in text for value in required)

def find_cached_safe(row: pd.Series) -> Path | None:
    matches = [safe for safe in safe_cache_index() if _safe_matches_row(safe, row)]
    unique = sorted({path.resolve() for path in matches}, key=lambda value: len(value.as_posix()))
    if len(unique) > 1:
        print("Multiple cached SAFE products matched; using shortest path:")
        for value in unique[:5]:
            print(f"  - {display_path(value)}")
    return unique[0] if unique else None

def _candidate_archive_roots() -> list[Path]:
    roots = [BURST_DIR / "archives", SARPYX_ROOT / "data" / "bursts"]
    extra_roots = [value for value in os.getenv("SARPYX_PYSTAMPS_ARCHIVE_CACHE", "").split(os.pathsep) if value]
    roots.extend(Path(value).expanduser() for value in extra_roots)
    return [root for root in roots if root.exists()]

def archive_cache_index() -> list[Path]:
    global ARCHIVE_CACHE_INDEX
    if ARCHIVE_CACHE_INDEX is None:
        archives: set[Path] = set()
        for root in _candidate_archive_roots():
            archives.update(path.resolve() for path in root.rglob("*.zip") if path.is_file())
        ARCHIVE_CACHE_INDEX = sorted(archives, key=lambda value: value.as_posix())
        print(f"Indexed {len(ARCHIVE_CACHE_INDEX)} cached ZIP archives from {len(_candidate_archive_roots())} cache roots.")
    return ARCHIVE_CACHE_INDEX

def find_cached_archive(row: pd.Series) -> Path | None:
    burst_uuid = str(row["Id"])
    product_name = str(row.get("Name", ""))
    candidates = [path for path in archive_cache_index() if burst_uuid in path.as_posix() or (product_name and product_name in path.name)]
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) > 1:
        raise RuntimeError(f"Expected at most one cached archive for {burst_uuid}, found {len(unique)}")
    return unique[0] if unique else None

def _cached_archive_dir(burst_uuid: str) -> Path:
    return BURST_DIR / "archives" / burst_uuid

def require_cdse_credentials() -> tuple[str, str]:
    global CDSE_USERNAME, CDSE_PASSWORD
    if CDSE_USERNAME and CDSE_PASSWORD:
        return CDSE_USERNAME, CDSE_PASSWORD
    if os.getenv("SARPYX_PYSTAMPS_PROMPT_CREDENTIALS", "1") != "1":
        raise ValueError("Set CDSE_USERNAME/CDSE_PASSWORD or CDSE_USR/CDSE_PSW before downloading missing bursts")
    try:
        from getpass import getpass

        CDSE_USERNAME = input("CDSE email/username: ").strip()
        CDSE_PASSWORD = getpass("CDSE password: ")
    except (EOFError, KeyboardInterrupt) as exc:
        raise ValueError("Missing CDSE credentials; set CDSE_USERNAME/CDSE_PASSWORD or run interactively") from exc
    if not CDSE_USERNAME or not CDSE_PASSWORD:
        raise ValueError("Both CDSE username and password are required to download missing bursts")
    return CDSE_USERNAME, CDSE_PASSWORD

def ensure_burst_archive(row: pd.Series) -> Path:
    global ACCESS_TOKEN, ARCHIVE_CACHE_INDEX
    cached = find_cached_archive(row)
    if cached is not None:
        return cached

    burst_uuid = str(row["Id"])
    role_dir = _cached_archive_dir(burst_uuid)
    role_dir.mkdir(parents=True, exist_ok=True)
    username, password = require_cdse_credentials()

    from phidown.downloader import download_burst_on_demand, get_token

    if ACCESS_TOKEN is None:
        ACCESS_TOKEN = get_token(username=username, password=password)
    download_burst_on_demand(burst_id=burst_uuid, token=ACCESS_TOKEN, output_dir=role_dir, resume_mode="product")
    refreshed = sorted(role_dir.glob("*.zip"))
    if len(refreshed) != 1:
        raise RuntimeError(f"Expected exactly one archive after download in {role_dir}, found {len(refreshed)}")
    ARCHIVE_CACHE_INDEX = None
    return refreshed[0]

def extract_archive(archive_path: Path, burst_uuid: str) -> Path:
    from sarpyx.snapflow.burst_utils import extract_burst_archive

    return extract_burst_archive(archive_path, BURST_DIR / "extracted" / burst_uuid)

def resolve_burst_product(row: pd.Series) -> dict[str, str]:
    cached_safe = find_cached_safe(row)
    if cached_safe is not None:
        return {"archive": "<reused cached SAFE>", "safe_product": cached_safe.as_posix(), "source": "cached_safe"}
    archive = ensure_burst_archive(row)
    safe_product = extract_archive(archive, str(row["Id"]))
    return {"archive": archive.as_posix(), "safe_product": safe_product.as_posix(), "source": "downloaded_archive"}

def cleanup_split_intermediate(output_path: Path) -> None:
    split_path = output_path.with_name(f"{output_path.stem}_split.dim")
    for candidate in (split_path, split_path.with_suffix(".data")):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        elif candidate.exists():
            candidate.unlink()

def split_orbit_output_path(row: pd.Series) -> Path:
    label = f"{int(row['stack_index']):02d}_{row['date_yyyymmdd']}_split_orbit"
    return PROCESS_DIR / "split_orbit" / str(row["date_yyyymmdd"]) / f"{label}.dim"

def run_split_orbit(row: pd.Series) -> Path:
    from sarpyx.snapflow.snap2stamps import run_topsar_split_apply_orbit

    output_path = split_orbit_output_path(row)
    if output_path.exists():
        if CLEAN_INTERMEDIATES:
            cleanup_split_intermediate(output_path)
        return output_path
    source = Path(row["safe_product"])
    outdir = output_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    result = run_topsar_split_apply_orbit(
        [source],
        outdir=outdir,
        subswath=str(row["SwathIdentifier"]),
        polarisation=POLARISATION,
        polygon_wkt=SPLIT_AOI_WKT,
        output_name=output_path.stem,
        gpt_path=GPT_PATH.as_posix(),
        memory=GPT_MEMORY,
        parallelism=GPT_PARALLELISM,
        timeout=GPT_TIMEOUT,
        snap_userdir=SNAP_USERDIR,
    )
    if result is None:
        raise RuntimeError("TOPSAR split/orbit failed before producing an output product")
    result_path = Path(result)
    if CLEAN_INTERMEDIATES and result_path.exists():
        cleanup_split_intermediate(output_path)
    return result_path

def gpt_for(product: str | Path, outdir: str | Path):
    from sarpyx.snapflow.engine import GPT

    return GPT(
        product=product,
        outdir=outdir,
        format="BEAM-DIMAP",
        gpt_path=GPT_PATH.as_posix(),
        memory=GPT_MEMORY,
        parallelism=GPT_PARALLELISM,
        timeout=GPT_TIMEOUT,
        snap_userdir=SNAP_USERDIR,
    )

def require_gpt_product(path: str | Path | None, gpt, step: str) -> Path:
    if path is None:
        raise RuntimeError(f"{step} failed: {gpt.last_error_summary()}")
    product_path = Path(path)
    if not product_path.exists():
        raise RuntimeError(f"{step} reported {product_path}, but the product does not exist")
    return product_path

def require_gpt_success(result: str | Path | None, gpt, step: str) -> str:
    if result is None:
        raise RuntimeError(f"{step} failed: {gpt.last_error_summary()}")
    return str(result)

def remove_dimap_product(path: str | Path | None) -> None:
    if not path:
        return
    dim = Path(path)
    candidates = [dim]
    if dim.suffix == ".dim":
        candidates.append(dim.with_suffix(".data"))
    for candidate in candidates:
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        elif candidate.exists():
            candidate.unlink()

def clean_product_paths(paths: list[str | Path | None]) -> None:
    if not CLEAN_INTERMEDIATES:
        return
    for path in paths:
        remove_dimap_product(path)

def pair_marker_path(pair_id: str) -> Path:
    return PROCESS_DIR / "pair_markers" / f"{pair_id}.json"

def expected_stamps_export_files(pair_id: str, secondary_date: str) -> list[Path]:
    return [
        INSAR_DATASET / "diff0" / f"{pair_id}.diff",
        INSAR_DATASET / "diff0" / f"{pair_id}.diff.par",
        INSAR_DATASET / "diff0" / f"{pair_id}.base",
        INSAR_DATASET / "rslc" / f"{MASTER_DATE}.rslc",
        INSAR_DATASET / "rslc" / f"{MASTER_DATE}.rslc.par",
        INSAR_DATASET / "rslc" / f"{secondary_date}.rslc",
        INSAR_DATASET / "rslc" / f"{secondary_date}.rslc.par",
        INSAR_DATASET / "geo" / f"{MASTER_DATE}.lat",
        INSAR_DATASET / "geo" / f"{MASTER_DATE}.lon",
        INSAR_DATASET / "geo" / "elevation_dem.rdc",
        INSAR_DATASET / "dem" / "projected_dem.rslc",
    ]

def missing_stamps_export_files(pair_id: str, secondary_date: str) -> list[Path]:
    return [path for path in expected_stamps_export_files(pair_id, secondary_date) if not path.exists()]

def write_pair_marker(marker: Path, pair_id: str, secondary_date: str, status: str, exported: str | None = None) -> None:
    payload = {
        "pair_id": pair_id,
        "status": status,
        "secondary_date": secondary_date,
        "exported": exported,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def run_pair_export(master_product: Path, secondary_row: pd.Series) -> dict[str, str]:
    secondary_product = Path(secondary_row["prepared_product"])
    pair_id = f"{MASTER_DATE}_{secondary_row['date_yyyymmdd']}"
    pair_dir = PROCESS_DIR / "pairs" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    marker = pair_marker_path(pair_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    secondary_date = str(secondary_row["date_yyyymmdd"])
    if marker.exists():
        return {"pair_id": pair_id, "status": "skipped_existing", "export_marker": marker.as_posix()}
    if not missing_stamps_export_files(pair_id, secondary_date):
        write_pair_marker(marker, pair_id, secondary_date, "exported_existing")
        return {"pair_id": pair_id, "status": "exported_existing", "export_marker": marker.as_posix()}

    created: list[str | Path | None] = []
    try:
        coreg_gpt = gpt_for(master_product, pair_dir)
        coreg = require_gpt_product(coreg_gpt.topsar_coregistration(
            master_product=master_product,
            slave_product=secondary_product,
            use_esd=False,
            dem_name=DEM_NAME,
            output_name=f"{pair_id}_coreg",
        ), coreg_gpt, "TopsarCoregistration")
        created.append(coreg)

        coreg_deb_gpt = gpt_for(coreg, pair_dir)
        coreg_deb = require_gpt_product(
            coreg_deb_gpt.deburst(output_name=f"{pair_id}_coreg_deb"),
            coreg_deb_gpt,
            "TOPSAR-Deburst coregistered stack",
        )
        created.append(coreg_deb)

        ifg_gpt = gpt_for(coreg, pair_dir)
        ifg_raw = require_gpt_product(ifg_gpt.interferogram(
            subtract_flat_earth_phase=True,
            include_coherence=False,
            output_name=f"{pair_id}_ifg_raw",
        ), ifg_gpt, "Interferogram")
        created.append(ifg_raw)

        ifg_deb_gpt = gpt_for(ifg_raw, pair_dir)
        ifg_deb = require_gpt_product(
            ifg_deb_gpt.deburst(output_name=f"{pair_id}_ifg_deb"),
            ifg_deb_gpt,
            "TOPSAR-Deburst interferogram",
        )
        created.append(ifg_deb)

        topo_gpt = gpt_for(ifg_deb, pair_dir)
        topo = require_gpt_product(topo_gpt.topo_phase_removal(
            dem_name=DEM_NAME,
            output_topo_phase_band=True,
            output_elevation_band=True,
            output_lat_lon_bands=True,
            output_name=f"{pair_id}_ifg_topo",
        ), topo_gpt, "TopoPhaseRemoval")
        created.append(topo)

        subset_ifg_gpt = gpt_for(topo, pair_dir)
        subset_ifg = require_gpt_product(subset_ifg_gpt.subset(
            region=SUBSET_REGION,
            geo_region=AOI_WKT,
            copy_metadata=True,
            output_name=f"{pair_id}_ifg_subset",
        ), subset_ifg_gpt, "Subset interferogram")
        created.append(subset_ifg)

        subset_coreg_gpt = gpt_for(coreg_deb, pair_dir)
        subset_coreg = require_gpt_product(subset_coreg_gpt.subset(
            region=SUBSET_REGION,
            geo_region=AOI_WKT,
            copy_metadata=True,
            output_name=f"{pair_id}_coreg_subset",
        ), subset_coreg_gpt, "Subset coregistered stack")
        created.append(subset_coreg)

        export_gpt = gpt_for(subset_coreg, pair_dir)
        exported = require_gpt_success(export_gpt.stamps_export_pair(
            coreg_product=subset_coreg,
            ifg_product=subset_ifg,
            target_folder=INSAR_DATASET,
            psi_format=True,
            output_name=f"{pair_id}_stamps_export",
        ), export_gpt, "StampsExport")
        missing_after_export = missing_stamps_export_files(pair_id, secondary_date)
        if missing_after_export:
            missing_text = ", ".join(display_path(path) for path in missing_after_export[:8])
            raise RuntimeError(f"StampsExport completed but expected dataset files are missing: {missing_text}")
        write_pair_marker(marker, pair_id, secondary_date, "exported", exported)
        return {"pair_id": pair_id, "status": "exported", "coreg_product": str(subset_coreg), "ifg_product": str(subset_ifg), "export_marker": marker.as_posix()}
    finally:
        clean_product_paths(created)
        if CLEAN_INTERMEDIATES and pair_dir.exists() and not any(pair_dir.iterdir()):
            pair_dir.rmdir()

def existing_pair_export_result(secondary_row: pd.Series) -> dict[str, object]:
    secondary_date = str(secondary_row["date_yyyymmdd"])
    pair_id = f"{MASTER_DATE}_{secondary_date}"
    marker = pair_marker_path(pair_id)
    missing = missing_stamps_export_files(pair_id, secondary_date)
    if missing:
        return {
            "pair_id": pair_id,
            "status": "missing_export_files",
            "missing_count": len(missing),
            "missing_example": display_path(missing[0]),
        }
    marker.parent.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_pair_marker(marker, pair_id, secondary_date, "exported_existing")
    return {"pair_id": pair_id, "status": "skipped_existing", "export_marker": marker.as_posix()}

def existing_stack_export_results(stack: pd.DataFrame) -> pd.DataFrame:
    secondary_rows = stack.loc[stack["role"].ne("master")]
    return pd.DataFrame([existing_pair_export_result(row) for _, row in secondary_rows.iterrows()])

def stack_exports_ready(stack: pd.DataFrame) -> bool:
    if stack.empty or MASTER_DATE is None:
        return False
    if int(stack["role"].eq("master").sum()) != 1:
        return False
    results = existing_stack_export_results(stack)
    return not results.empty and bool(results["status"].eq("skipped_existing").all())

def resolve_and_prepare(row: pd.Series) -> pd.Series:
    if row.get("safe_product") in (None, "<resolved during streaming>") or not Path(str(row.get("safe_product", ""))).exists():
        resolved = resolve_burst_product(row)
        for key, value in resolved.items():
            row[key] = value
    prepared = run_split_orbit(row)
    row["prepared_product"] = prepared.as_posix()
    return row

def patch_input_summary(dataset_root: Path) -> pd.DataFrame:
    patches = sorted(path for path in dataset_root.glob("PATCH_*") if path.is_dir()) if dataset_root.exists() else []
    rows = []
    for patch in patches:
        rows.append({
            "patch": patch.name,
            "pscands.1.ij": (patch / "pscands.1.ij").exists(),
            "pscands.1.ph": (patch / "pscands.1.ph").exists(),
            "pscands.1.ll": (patch / "pscands.1.ll").exists(),
            "pscands.1.da": (patch / "pscands.1.da").exists(),
            "pscands.1.hgt": (patch / "pscands.1.hgt").exists(),
        })
    return pd.DataFrame(rows)

def patch_inputs_ready(dataset_root: Path) -> bool:
    summary = patch_input_summary(dataset_root)
    required_patch_list = (dataset_root / "patch.list").exists()
    required_files = {"pscands.1.ij", "pscands.1.ph", "pscands.1.ll"}
    return bool(required_patch_list and not summary.empty and summary[list(required_files)].all(axis=None))

def clean_stale_mt_prep_artifacts(dataset_root: Path) -> None:
    for name in ("f", "calamp.out", "selpsc.in", "pscphase.in", "pscdem.in", "psclonlat.in"):
        path = dataset_root / name
        if path.exists() and path.is_file():
            path.unlink()

def pystamps_config_args() -> list[str]:
    if not RUN_PYSTAMPS:
        return []
    config_path = OUTPUT_DIR / "pystamps_runtime.yaml"
    tools = {
        "triangle": TRIANGLE_PATH.as_posix() if TRIANGLE_PATH else "triangle",
        "snaphu": SNAPHU_PATH.as_posix() if SNAPHU_PATH else "snaphu",
    }
    payload = {
        "runtime": {
            "backend": os.getenv("PYSTAMPS_BACKEND", "auto"),
            "stage2_kernel_backend": os.getenv("PYSTAMPS_STAGE2_KERNEL_BACKEND", "auto"),
            "stage2_native_threads": int(os.getenv("PYSTAMPS_STAGE2_NATIVE_THREADS", "0")),
            "stage2_checkpoint_mode": os.getenv("PYSTAMPS_STAGE2_CHECKPOINT_MODE", "final"),
            "stage2_checkpoint_interval": int(os.getenv("PYSTAMPS_STAGE2_CHECKPOINT_INTERVAL", "1")),
            "stage2_debug": os.getenv("PYSTAMPS_STAGE2_DEBUG", "0") == "1",
            "io_workers": int(os.getenv("PYSTAMPS_IO_WORKERS", "8")),
            "cpu_workers": int(os.getenv("PYSTAMPS_CPU_WORKERS", "0")),
        },
        "tools": tools,
    }
    import yaml

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return ["--config", config_path.as_posix()]

def run_pystamps_command(args: list[str]) -> dict[str, object]:
    command = [sys.executable, "-m", "pystamps.cli", *pystamps_config_args(), *args]
    env = dict(os.environ)
    if args and args[0] != "run":
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "GOTO_NUM_THREADS": "1",
        })
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - started
    payload = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = completed.stdout.strip()
    result = {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "elapsed_sec": elapsed,
        "stdout": payload,
        "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(result, indent=2, default=str))
    return result
# --- end notebook pipeline helpers ---
@dataclass(slots=True)
class PlotContext:
    dataset_root: Path; output_dir: Path; plot_dir: Path; diag: Any
    ps2: dict[str, Any]; phuw2: dict[str, Any]; ifgstd2: dict[str, Any]; scla2: dict[str, Any]
    stack: pd.DataFrame; patch_flow: pd.DataFrame; dates: list[datetime]
def matlab_datenum_to_datetime(value: float) -> datetime:
    day = float(value); ordinal = int(day)
    return datetime.fromordinal(ordinal) + timedelta(days=day - ordinal) - timedelta(days=366)
def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)
def mat_scalar(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = np.asarray(payload.get(key, []), dtype=float).reshape(-1)
    return default if value.size == 0 else int(round(float(value[0])))
def _row_date(row: pd.Series) -> pd.Timestamp:
    if "_content_start" in row and pd.notna(row["_content_start"]):
        return pd.to_datetime(row["_content_start"], utc=True)
    raw = row.get("ContentDate")
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            raw = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            pass
    if isinstance(raw, dict):
        raw = raw.get("Start")
    return pd.to_datetime(raw if raw is not None else row.get("OriginDate"), utc=True)
def _geometry(value: Any):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text.startswith("geography'SRID=4326;") and text.endswith("'"):
        text = text.removeprefix("geography'SRID=4326;")[:-1]
    try:
        if text.startswith("{"):
            return shape(ast.literal_eval(text))
        return wkt.loads(text)
    except Exception:
        return None
def prepare_stack_frame(stack_products: pd.DataFrame | None, *, stats_dir: Path | None = None, master_date: str | None = None) -> pd.DataFrame:
    if stack_products is not None and not stack_products.empty:
        frame = stack_products.copy()
    elif stats_dir is not None and (stats_dir / "search_results.csv").exists():
        frame = pd.read_csv(stats_dir / "search_results.csv")
    else:
        return pd.DataFrame(columns=["plot_date", "date_yyyymmdd", "role"])
    frame["_content_start"] = frame.apply(_row_date, axis=1)
    frame["plot_date"] = pd.to_datetime(frame["_content_start"], utc=True).dt.tz_convert(None)
    if "date_yyyymmdd" not in frame:
        frame["date_yyyymmdd"] = frame["_content_start"].dt.strftime("%Y%m%d")
    if "role" not in frame:
        frame["role"] = np.where(frame["date_yyyymmdd"].astype(str).eq(str(master_date)), "master", "secondary")
    frame["geometry"] = frame.get("GeoFootprint", frame.get("Footprint", pd.Series(index=frame.index))).map(_geometry)
    return frame.sort_values("plot_date").reset_index(drop=True)
def summarize_patch_flow(dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    patches = sorted(dataset_root.glob("PATCH_*"), key=lambda value: int(value.name.split("_")[1]))
    for patch in patches:
        ps1 = read_mat(patch / "ps1.mat") if (patch / "ps1.mat").exists() else {}
        select1 = read_mat(patch / "select1.mat") if (patch / "select1.mat").exists() else {}
        weed1 = read_mat(patch / "weed1.mat") if (patch / "weed1.mat").exists() else {}
        ps2 = read_mat(patch / "ps2.mat") if (patch / "ps2.mat").exists() else {}
        ix = np.asarray(select1.get("ix", np.empty((0,))), dtype=float).reshape(-1)
        ix_weed = np.asarray(weed1.get("ix_weed", np.empty((0,))), dtype=bool).reshape(-1)
        rows.append({"patch": patch.name, "stage": infer_patch_stage(patch), "candidates": count_lines(patch / "pscands.1.ij"), "stage1_ps": mat_scalar(ps1, "n_ps"), "selected": int(ix.size), "weeded": int(np.count_nonzero(ix_weed)) if ix_weed.size else 0, "promoted": mat_scalar(ps2, "n_ps")})
    return pd.DataFrame(rows)
def build_plot_context(dataset_root: str | Path, output_dir: str | Path, *, stack_products: pd.DataFrame | None = None, stats_dir: str | Path | None = None, master_date: str | None = None) -> PlotContext:
    root = Path(dataset_root); out = Path(output_dir); plot_dir = out / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    diag = load_velocity_diagnostics(root, apply_step8=True, load_coherence=True)
    return PlotContext(root, out, plot_dir, diag, read_mat(root / "ps2.mat"), read_mat(root / "phuw2.mat"), read_mat(root / "ifgstd2.mat"), read_mat(root / "scla2.mat"), prepare_stack_frame(stack_products, stats_dir=Path(stats_dir) if stats_dir else None, master_date=master_date), summarize_patch_flow(root), [matlab_datenum_to_datetime(v) for v in diag.day])
def pipeline_summary(ctx: PlotContext) -> pd.DataFrame:
    return pd.DataFrame([
        {"boundary": "CDSE burst search", "artifact": "same-burst acquisitions", "count": len(ctx.stack)},
        {"boundary": "SARpyx/SNAP export", "artifact": "RSLC products", "count": len(list((ctx.dataset_root / "rslc").glob("*.rslc")))},
        {"boundary": "SARpyx/SNAP export", "artifact": "interferograms", "count": len(list((ctx.dataset_root / "diff0").glob("*.diff")))},
        {"boundary": "mt_prep_snap", "artifact": "PATCH directories", "count": len(ctx.patch_flow)},
        {"boundary": "pySTAMPS stage 1", "artifact": "candidate PS rows", "count": int(ctx.patch_flow["stage1_ps"].sum())},
        {"boundary": "pySTAMPS stage 3", "artifact": "selected PS rows", "count": int(ctx.patch_flow["selected"].sum())},
        {"boundary": "pySTAMPS stage 5", "artifact": "merged PS", "count": int(np.asarray(ctx.ps2["n_ps"]).reshape(-1)[0])},
        {"boundary": "pySTAMPS stage 8", "artifact": "velocity points", "count": int(ctx.diag.velocity.size)},
    ])
def snap2stamps_output_summary(ctx: PlotContext) -> pd.DataFrame:
    groups = [("RSLC", "rslc", ["*.rslc", "*.rslc.par"]), ("Interferograms", "diff0", ["*.diff", "*.diff.par", "*.base"]), ("Geometry", "geo", ["*.lat", "*.lon", "*.rdc"]), ("DEM", "dem", ["*"]), ("StaMPS patches", ".", ["PATCH_*/pscands.1.*"])]
    rows = []
    for label, folder, patterns in groups:
        files = [p for pattern in patterns for p in (ctx.dataset_root / folder).glob(pattern) if p.is_file()]
        rows.append({"output": label, "files": len(files), "size_mb": sum(p.stat().st_size for p in files) / 1024 / 1024})
    return pd.DataFrame(rows)
def _save(fig, ctx: PlotContext, name: str):
    path = ctx.plot_dir / name; fig.savefig(path, dpi=190, bbox_inches="tight"); fig._saved_path = path
    return fig
def _tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(max(min(lat, 85.0), -85.0)); scale = 2**zoom
    return int((lon + 180.0) / 360.0 * scale), int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale)
def _lonlat(x: int, y: int, zoom: int) -> tuple[float, float]:
    scale = 2**zoom; lon = x / scale * 360.0 - 180.0; n = math.pi - 2.0 * math.pi * y / scale
    return lon, math.degrees(math.atan(math.sinh(n)))
def _draw_osm(ax, bounds: tuple[float, float, float, float], ctx: PlotContext, zoom: int = 11) -> bool:
    min_lon, min_lat, max_lon, max_lat = bounds
    while zoom >= 8:
        x0, y1 = _tile(min_lon, min_lat, zoom); x1, y0 = _tile(max_lon, max_lat, zoom)
        if (x1 - x0 + 1) * (y1 - y0 + 1) <= 24:
            break
        zoom -= 1
    cache = ctx.plot_dir / "map_tiles" / str(zoom); cache.mkdir(parents=True, exist_ok=True)
    try:
        mosaic = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                path = cache / f"{x}_{y}.png"
                if not path.exists():
                    req = Request(f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png", headers={"User-Agent": "pystamps-notebook/1.0"})
                    path.write_bytes(urlopen(req, timeout=12).read())
                mosaic.paste(Image.open(path).convert("RGB"), ((x - x0) * 256, (y - y0) * 256))
        left, top = _lonlat(x0, y0, zoom); right, bottom = _lonlat(x1 + 1, y1 + 1, zoom)
        ax.imshow(mosaic, extent=(left, right, bottom, top), origin="upper", alpha=0.82, zorder=0)
        return True
    except Exception:
        return False
def _pad(bounds: tuple[float, float, float, float], frac: float = 0.08) -> tuple[float, float, float, float]:
    a, b, c, d = bounds; dx = max(c - a, 0.01) * frac; dy = max(d - b, 0.01) * frac
    return a - dx, b - dy, c + dx, d + dy
def _draw_geom(ax, geom, **kwargs) -> None:
    geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
    for item in geoms:
        x, y = item.exterior.xy; ax.fill(x, y, **kwargs)
def plot_burst_locations_map(ctx: PlotContext, aoi_wkt: str | None = None):
    geoms = [g for g in ctx.stack.get("geometry", []) if g is not None]
    aoi = _geometry(aoi_wkt) if aoi_wkt else None
    union = unary_union(geoms + ([aoi] if aoi is not None else [])); bounds = _pad(union.bounds, 0.12)
    fig, ax = plt.subplots(figsize=(9.8, 7.0)); used_map = _draw_osm(ax, bounds, ctx, zoom=11)
    for geom in geoms:
        _draw_geom(ax, geom, facecolor="#4c78a833", edgecolor="#1f5a96", linewidth=1.0, zorder=2)
    if aoi is not None:
        _draw_geom(ax, aoi, facecolor="#e1575930", edgecolor="#b21d2b", linewidth=2.1, zorder=4)
        ax.scatter([aoi.centroid.x], [aoi.centroid.y], s=55, c="#b21d2b", edgecolors="white", zorder=5, label="AOI")
    label = f"burst {ctx.stack['BurstId'].iloc[0]} / {ctx.stack['SwathIdentifier'].iloc[0]} / orbit {ctx.stack['RelativeOrbitNumber'].iloc[0]}" if not ctx.stack.empty and "BurstId" in ctx.stack else "burst stack"
    ax.set_title(f"Selected Sentinel-1 burst footprints over map ({len(ctx.stack)} acquisitions)\n{label}")
    ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(color="white" if used_map else "0.88", linewidth=0.5)
    ax.text(0.01, 0.01, "Basemap: OpenStreetMap tiles" if used_map else "Basemap unavailable; showing geographic axes", transform=ax.transAxes, fontsize=8, color="0.3")
    _save_burst_html(ctx, geoms, aoi)
    return _save(fig, ctx, "11_burst_locations_map.png")
def _save_burst_html(ctx: PlotContext, geoms: list[Any], aoi) -> None:
    try:
        import folium
        center = unary_union(geoms + ([aoi] if aoi is not None else [])).centroid
        m = folium.Map(location=[center.y, center.x], zoom_start=10, tiles="OpenStreetMap")
        for geom in geoms[:1]:
            folium.GeoJson(geom.__geo_interface__, name="burst footprint", style_function=lambda _: {"color": "#1f5a96", "weight": 2, "fillOpacity": 0.18}).add_to(m)
        if aoi is not None:
            folium.GeoJson(aoi.__geo_interface__, name="AOI", style_function=lambda _: {"color": "#b21d2b", "weight": 3, "fillOpacity": 0.12}).add_to(m)
        folium.LayerControl().add_to(m); m.save(ctx.plot_dir / "11_burst_locations_map.html")
    except Exception:
        return
def _spatial(ctx: PlotContext):
    lonlat = ctx.diag.lonlat; velocity = np.asarray(ctx.diag.velocity, dtype=float)
    mask = np.isfinite(velocity) & np.isfinite(lonlat[:, 0]) & np.isfinite(lonlat[:, 1])
    limit = max(float(np.nanpercentile(np.abs(velocity[mask]), 98)), 1e-6)
    return lonlat, velocity, mask, limit
def plot_pipeline_scorecard(ctx: PlotContext):
    table = pipeline_summary(ctx); fig, ax = plt.subplots(figsize=(13, 3.8)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    xs = np.linspace(0.06, 0.94, len(table)); colors = ["#4c78a8", "#4c78a8", "#4c78a8", "#59a14f", "#f28e2b", "#f28e2b", "#e15759", "#b07aa1"]
    for i, (_, row) in enumerate(table.iterrows()):
        ax.scatter(xs[i], 0.62, s=1100, color=colors[i], alpha=0.92); ax.text(xs[i], 0.65, f"{int(row['count']):,}", ha="center", va="center", color="white", weight="bold", fontsize=11)
        ax.text(xs[i], 0.38, str(row["artifact"]).replace(" ", "\n", 1), ha="center", va="top", fontsize=8); ax.text(xs[i], 0.18, str(row["boundary"]).replace(" ", "\n", 1), ha="center", va="top", fontsize=7, color="0.35")
        if i < len(table) - 1:
            ax.add_patch(FancyArrowPatch((xs[i] + 0.035, 0.62), (xs[i + 1] - 0.035, 0.62), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="0.45"))
    ax.set_title("End-to-end product flow: burst search to stage-8 velocity", fontsize=14, pad=14)
    return _save(fig, ctx, "00_pipeline_scorecard.png")
def plot_snap2stamps_sequence(ctx: PlotContext):
    steps = [("Split + orbit", f"{len(ctx.stack)} bursts"), ("Coregister", f"{max(len(ctx.stack)-1, 0)} pairs"), ("Deburst stack", "single burst"), ("Interferogram", "flat-earth removed"), ("Deburst IFG", "pair products"), ("Topo phase", "lat/lon/elev/topo"), ("Subset AOI", "Rome polygon"), ("StampsExport", "rslc/diff0/geo/dem"), ("mt_prep_snap", f"{len(ctx.patch_flow)} PATCH_*")]
    fig, ax = plt.subplots(figsize=(12.8, 4.2)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    xs = np.linspace(0.055, 0.945, len(steps))
    for i, (name, detail) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((xs[i]-0.047, 0.47), 0.094, 0.22, boxstyle="round,pad=0.015,rounding_size=0.018", facecolor="#eef3f7", edgecolor="#3f6f8f", linewidth=1.2))
        ax.text(xs[i], 0.61, name, ha="center", va="center", fontsize=8.5, weight="bold"); ax.text(xs[i], 0.52, detail, ha="center", va="center", fontsize=7.5, color="0.32")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((xs[i]+0.052, 0.58), (xs[i+1]-0.052, 0.58), arrowstyle="-|>", mutation_scale=12, lw=1.0, color="0.4"))
    ax.set_title("SNAP2StaMPS processing sequence performed by SARpyx/SNAP before pySTAMPS", fontsize=13, pad=12)
    ax.text(0.5, 0.23, "The handoff to pySTAMPS happens only after mt_prep_snap has converted the SNAP StampsExport tree into PATCH_* candidate files.", ha="center", fontsize=9, color="0.25")
    return _save(fig, ctx, "12_snap2stamps_sequence.png")
def plot_snap2stamps_outputs(ctx: PlotContext):
    frame = snap2stamps_output_summary(ctx); fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    colors = ["#4c78a8", "#f28e2b", "#59a14f", "#b07aa1", "#e15759"]
    axes[0].barh(frame["output"], frame["files"], color=colors); axes[0].invert_yaxis(); axes[0].set_title("SNAP2StaMPS output file counts"); axes[0].set_xlabel("Files")
    axes[1].barh(frame["output"], frame["size_mb"], color=colors); axes[1].invert_yaxis(); axes[1].set_title("SNAP2StaMPS output disk footprint"); axes[1].set_xlabel("MiB")
    for ax in axes:
        ax.grid(axis="x", color="0.9", linewidth=0.6)
    fig.suptitle("Files created for the pySTAMPS handoff: RSLC, interferogram, geometry, DEM, and PATCH candidates")
    return _save(fig, ctx, "13_snap2stamps_outputs.png")
def plot_stack_baseline(ctx: PlotContext):
    bperp = np.asarray(ctx.ps2.get("bperp"), dtype=float).reshape(-1); frame = ctx.stack.iloc[: bperp.size].copy(); frame["bperp"] = bperp[: len(frame)]
    fig, ax = plt.subplots(figsize=(10.5, 4.2)); colors = np.where(frame["role"].eq("master"), "#d1495b", "#1f77b4")
    ax.scatter(frame["plot_date"], frame["bperp"], c=colors, s=60, edgecolor="white", linewidth=0.8, zorder=3); ax.plot(frame["plot_date"], frame["bperp"], color="0.65", linewidth=1.0, zorder=1)
    ax.axhline(0, color="0.55", linewidth=0.8); ax.set_title("Temporal stack and perpendicular baseline"); ax.set_xlabel("Acquisition date"); ax.set_ylabel("Perpendicular baseline"); ax.grid(color="0.9", linewidth=0.7); fig.autofmt_xdate()
    return _save(fig, ctx, "01_stack_baseline.png")
def plot_patch_point_reduction(ctx: PlotContext):
    fig, ax = plt.subplots(figsize=(12.5, 4.7)); ctx.patch_flow.set_index("patch")[["candidates", "stage1_ps", "selected", "weeded", "promoted"]].plot(kind="bar", ax=ax, width=0.86)
    ax.set_yscale("log"); ax.set_ylabel("Point count (log scale)"); ax.set_title("Candidate-to-PS reduction by patch"); ax.grid(axis="y", which="both", color="0.9", linewidth=0.7); ax.legend(frameon=False, ncols=3)
    return _save(fig, ctx, "02_patch_point_reduction.png")
def plot_velocity_map(ctx: PlotContext):
    lonlat, velocity, mask, limit = _spatial(ctx); bounds = _pad((float(np.nanmin(lonlat[mask, 0])), float(np.nanmin(lonlat[mask, 1])), float(np.nanmax(lonlat[mask, 0])), float(np.nanmax(lonlat[mask, 1]))), 0.08)
    fig, ax = plt.subplots(figsize=(8.2, 6.6)); used_map = _draw_osm(ax, bounds, ctx, zoom=14)
    sc = ax.scatter(lonlat[mask, 0], lonlat[mask, 1], c=velocity[mask], s=1.2, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), linewidths=0, rasterized=True, alpha=0.92, zorder=3)
    ax.set_title("Stage-8 PS velocity map over local basemap"); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3]); ax.set_aspect("equal", adjustable="box"); ax.grid(color="white" if used_map else "0.9", linewidth=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.83, label=f"Velocity ({ctx.diag.velocity_source})")
    return _save(fig, ctx, "03_velocity_map.png")
def plot_velocity_hexbin(ctx: PlotContext):
    lonlat, velocity, mask, limit = _spatial(ctx); fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), constrained_layout=True)
    hb0 = axes[0].hexbin(lonlat[mask, 0], lonlat[mask, 1], gridsize=70, mincnt=1, cmap="viridis")
    hb1 = axes[1].hexbin(lonlat[mask, 0], lonlat[mask, 1], C=velocity[mask], gridsize=70, reduce_C_function=np.nanmedian, mincnt=1, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    for ax, title in zip(axes, ["PS density", "Median velocity by spatial bin"]):
        ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal", adjustable="box")
    fig.colorbar(hb0, ax=axes[0], label="PS count"); fig.colorbar(hb1, ax=axes[1], label="Median velocity")
    return _save(fig, ctx, "04_velocity_density_hexbin.png")
def plot_velocity_distribution_stability(ctx: PlotContext):
    velocity = np.asarray(ctx.diag.velocity, dtype=float); stability = np.asarray(ctx.diag.stability, dtype=float); mask = np.isfinite(velocity) & np.isfinite(stability)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3)); axes[0].hist(velocity[mask], bins=90, color="#4c78a8", edgecolor="white", linewidth=0.35)
    axes[0].axvline(np.nanmedian(velocity[mask]), color="#d1495b", linewidth=1.6, label="median"); axes[0].set_title("Velocity distribution"); axes[0].set_xlabel("Velocity"); axes[0].set_ylabel("PS count"); axes[0].legend(frameon=False)
    axes[1].hexbin(velocity[mask], stability[mask], gridsize=75, bins="log", mincnt=1, cmap="mako" if "mako" in plt.colormaps() else "viridis"); axes[1].set_title("Velocity versus stability proxy"); axes[1].set_xlabel("Velocity"); axes[1].set_ylabel(ctx.diag.stability_source); axes[1].grid(color="0.9", linewidth=0.6)
    return _save(fig, ctx, "05_velocity_distribution_stability.png")
def plot_scla_maps(ctx: PlotContext):
    lonlat = ctx.diag.lonlat; fields = [("K_ps_uw", "Topographic/linear atmosphere term"), ("C_ps_uw", "Constant phase term")]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    for ax, (key, title) in zip(axes, fields):
        values = np.asarray(ctx.scla2.get(key), dtype=float).reshape(-1)[: lonlat.shape[0]]; mask = np.isfinite(values); limit = max(float(np.nanpercentile(np.abs(values[mask]), 98)), 1e-6) if np.any(mask) else 1.0
        sc = ax.scatter(lonlat[mask, 0], lonlat[mask, 1], c=values[mask], s=1.2, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), linewidths=0, rasterized=True)
        ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal", adjustable="box"); fig.colorbar(sc, ax=ax, shrink=0.82, label=key)
    return _save(fig, ctx, "06_scla_component_maps.png")
def plot_phase_heatmap(ctx: PlotContext, sample_size: int = 700):
    ph_uw = np.asarray(ctx.phuw2.get("ph_uw"), dtype=float); velocity = np.asarray(ctx.diag.velocity, dtype=float); valid = np.flatnonzero(np.isfinite(velocity) & np.all(np.isfinite(ph_uw), axis=1))
    pick = valid[np.linspace(0, valid.size - 1, sample_size, dtype=int)] if valid.size > sample_size else valid
    if valid.size > sample_size:
        pick = valid[np.argsort(velocity[valid])][np.linspace(0, valid.size - 1, sample_size, dtype=int)]
    master_col = max(0, min(ph_uw.shape[1] - 1, int(ctx.diag.master_ix) - 1)); matrix = ph_uw[pick, :] - ph_uw[pick, master_col][:, None]; limit = max(float(np.nanpercentile(np.abs(matrix), 98)), 1e-6) if matrix.size else 1.0
    fig, ax = plt.subplots(figsize=(10.5, 5.5)); im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest")
    ax.set_title("Unwrapped phase matrix sampled across the velocity range"); ax.set_xlabel("Acquisition index"); ax.set_ylabel("Sampled PS sorted by velocity"); ax.set_xticks(range(len(ctx.dates))); ax.set_xticklabels([d.strftime("%y-%m-%d") for d in ctx.dates], rotation=70, ha="right", fontsize=7); fig.colorbar(im, ax=ax, label="Phase relative to master (rad)")
    return _save(fig, ctx, "07_phase_heatmap.png")
def plot_phase_samples(ctx: PlotContext):
    ph_uw = np.asarray(ctx.phuw2.get("ph_uw"), dtype=float); velocity = np.asarray(ctx.diag.velocity, dtype=float); valid = np.flatnonzero(np.isfinite(velocity) & np.all(np.isfinite(ph_uw), axis=1)); chosen = []
    for value in np.nanpercentile(velocity[valid], [5, 25, 50, 75, 95]):
        chosen.append(valid[np.nanargmin(np.abs(velocity[valid] - value))])
    master_col = max(0, min(ph_uw.shape[1] - 1, int(ctx.diag.master_ix) - 1)); fig, ax = plt.subplots(figsize=(10.2, 4.6))
    for idx in dict.fromkeys(int(value) for value in chosen):
        y = ph_uw[idx, :] - ph_uw[idx, master_col]; ax.plot(ctx.dates[: y.size], y, marker="o", linewidth=1.2, markersize=3, label=f"PS {idx} / v={velocity[idx]:.2f}")
    ax.axhline(0, color="0.45", linewidth=0.8); ax.set_title("Representative unwrapped phase histories"); ax.set_xlabel("Acquisition date"); ax.set_ylabel("Phase relative to master (rad)"); ax.grid(color="0.9", linewidth=0.6); ax.legend(frameon=False, fontsize=8, ncols=2); fig.autofmt_xdate()
    return _save(fig, ctx, "08_unwrapped_phase_samples.png")
def plot_ifg_standard_deviation(ctx: PlotContext):
    ifg_std = np.asarray(ctx.ifgstd2.get("ifg_std"), dtype=float).reshape(-1); fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.plot(ctx.dates[: ifg_std.size], ifg_std, marker="o", color="#6f4e7c", linewidth=1.5); ax.axvline(ctx.dates[int(ctx.diag.master_ix) - 1], color="#d1495b", linestyle="--", linewidth=1.2, label="master")
    ax.set_title("Merged interferogram standard deviation by acquisition"); ax.set_xlabel("Acquisition date"); ax.set_ylabel("ifgstd2.ifg_std"); ax.grid(color="0.9", linewidth=0.6); ax.legend(frameon=False); fig.autofmt_xdate()
    return _save(fig, ctx, "09_ifg_standard_deviation.png")
def plot_output_inventory(ctx: PlotContext):
    files = sorted(ctx.dataset_root.glob("*.mat"), key=lambda path: path.stat().st_size, reverse=True); frame = pd.DataFrame({"file": [p.name for p in files], "size_mb": [p.stat().st_size / 1024 / 1024 for p in files]})
    fig, ax = plt.subplots(figsize=(10, 4.8)); ax.barh(frame["file"], frame["size_mb"], color="#4c78a8"); ax.invert_yaxis(); ax.set_title("Stage-8 dataset inventory"); ax.set_xlabel("File size (MiB)"); ax.grid(axis="x", color="0.9", linewidth=0.6)
    return _save(fig, ctx, "10_output_inventory.png")
PLOT_SPECS = [
    ("Pipeline scorecard", "Counts at each boundary from burst search to velocity.", "00_pipeline_scorecard.png", plot_pipeline_scorecard), ("Stack baseline", "Acquisition dates, baseline, and master selection.", "01_stack_baseline.png", plot_stack_baseline), ("Patch point reduction", "Candidate PS reduction by patch.", "02_patch_point_reduction.png", plot_patch_point_reduction), ("Velocity map", "Stage-8 PS velocity over map basemap.", "03_velocity_map.png", plot_velocity_map),
    ("Velocity density", "PS density and median velocity by bin.", "04_velocity_density_hexbin.png", plot_velocity_hexbin), ("Velocity stability", "Velocity distribution and stability proxy.", "05_velocity_distribution_stability.png", plot_velocity_distribution_stability), ("SCLA components", "Look-angle/atmosphere correction fields.", "06_scla_component_maps.png", plot_scla_maps), ("Phase heatmap", "Sampled unwrapped phase matrix.", "07_phase_heatmap.png", plot_phase_heatmap),
    ("Phase samples", "Representative PS phase histories.", "08_unwrapped_phase_samples.png", plot_phase_samples), ("IFG standard deviation", "Merged IFG variability by date.", "09_ifg_standard_deviation.png", plot_ifg_standard_deviation), ("Stage-8 inventory", "Final MAT-file outputs and sizes.", "10_output_inventory.png", plot_output_inventory), ("Burst locations", "Burst footprints and AOI over OpenStreetMap.", "11_burst_locations_map.png", plot_burst_locations_map),
    ("SNAP2StaMPS sequence", "SNAP/SARpyx operations before pySTAMPS.", "12_snap2stamps_sequence.png", plot_snap2stamps_sequence), ("SNAP2StaMPS outputs", "RSLC, IFG, geometry, DEM, and PATCH outputs.", "13_snap2stamps_outputs.png", plot_snap2stamps_outputs),
]
def render_all_plots(ctx: PlotContext, *, aoi_wkt: str | None = None) -> pd.DataFrame:
    rows = []
    for title, description, name, func in PLOT_SPECS:
        fig = func(ctx, aoi_wkt) if func is plot_burst_locations_map else func(ctx)
        rows.append({"title": title, "description": description, "path": str(ctx.plot_dir / name)})
        plt.close(fig)
    return pd.DataFrame(rows)
def display_plot_gallery(manifest: pd.DataFrame) -> None:
    from IPython.display import Image as NotebookImage
    from IPython.display import Markdown, display
    for row in manifest.itertuples(index=False):
        display(Markdown(f"### {row.title}\n{row.description}\n\nSaved output: `{row.path}`"))
        display(NotebookImage(filename=row.path))
