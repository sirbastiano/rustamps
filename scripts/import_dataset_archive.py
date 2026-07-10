#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a downloaded dataset archive into a local path.")
    parser.add_argument("--archive", required=True, help="Path to a .zip, .tar, .tar.gz, or .tgz archive.")
    parser.add_argument(
        "--destination",
        default="inputs_and_outputs/InSAR_dataset_test",
        help="Local dataset directory to create.",
    )
    parser.add_argument(
        "--source-prefix",
        default="",
        help="Optional path inside the archive to import instead of auto-detecting the root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing destination directory.")
    return parser.parse_args()


def _safe_members(paths: list[str]) -> list[str]:
    safe: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {raw}")
        safe.append(raw)
    return safe


def _extract_archive(archive: Path, destination: Path) -> None:
    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            _safe_members(handle.namelist())
            handle.extractall(destination)
        return
    if suffixes.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            _safe_members([member.name for member in members])
            handle.extractall(destination, members=members)
        return
    raise ValueError(f"Unsupported archive type: {archive}")


def _single_child_directory(root: Path) -> Path | None:
    children = [path for path in root.iterdir() if path.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return None


def _candidate_roots(root: Path) -> list[Path]:
    candidates = [root]
    single = _single_child_directory(root)
    if single is not None:
        candidates.insert(0, single)
    candidates.extend(path for path in root.rglob("InSAR_dataset_test") if path.is_dir())
    return candidates


def _looks_like_dataset(path: Path) -> bool:
    return (path / "patch.list").exists() or any(path.glob("PATCH_*")) or any(path.glob("*.mat"))


def _source_root(extracted_root: Path, source_prefix: str) -> Path:
    if source_prefix:
        candidate = extracted_root / Path(*Path(source_prefix.strip("/")).parts)
        if not candidate.exists():
            raise FileNotFoundError(f"Archive source prefix not found: {source_prefix}")
        return candidate
    for candidate in _candidate_roots(extracted_root):
        if _looks_like_dataset(candidate):
            return candidate
    single = _single_child_directory(extracted_root)
    return single or extracted_root


def _copy_dataset(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {destination}")
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def import_archive(archive: Path, destination: Path, *, source_prefix: str = "", overwrite: bool = False) -> Path:
    if not archive.exists():
        raise FileNotFoundError(f"Archive does not exist: {archive}")
    with tempfile.TemporaryDirectory(prefix="pystamps-dataset-") as tmp_dir:
        extracted_root = Path(tmp_dir)
        _extract_archive(archive, extracted_root)
        source = _source_root(extracted_root, source_prefix)
        _copy_dataset(source, destination, overwrite=overwrite)
    return destination


def main() -> int:
    args = _parse_args()
    try:
        destination = import_archive(
            Path(args.archive).expanduser(),
            Path(args.destination).expanduser(),
            source_prefix=args.source_prefix,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    print(f"imported dataset -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
