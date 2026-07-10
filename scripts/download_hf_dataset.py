#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_REPO = "mdelgadoblasco/InSAR_dataset_test"
DEFAULT_REVISION = "main"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset repository tree.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Dataset repository id, for example owner/name.")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Repository revision or branch.")
    parser.add_argument(
        "--destination",
        default="inputs_and_outputs/InSAR_dataset_test",
        help="Local directory where repository files will be written.",
    )
    parser.add_argument(
        "--include-prefix",
        default="",
        help="Only download files below this repository path prefix.",
    )
    parser.add_argument(
        "--strip-prefix",
        default="",
        help="Remove this repository path prefix when writing local paths.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files.")
    parser.add_argument("--dry-run", action="store_true", help="List planned downloads without writing files.")
    parser.add_argument(
        "--backend",
        choices=("auto", "huggingface", "url"),
        default="auto",
        help="Download backend. 'huggingface' uses huggingface_hub.snapshot_download.",
    )
    return parser.parse_args()


def _tree_url(repo: str, revision: str) -> str:
    repo_quoted = quote(repo.strip("/"), safe="/")
    revision_quoted = quote(revision, safe="")
    return f"https://huggingface.co/api/datasets/{repo_quoted}/tree/{revision_quoted}?recursive=1"


def _file_url(repo: str, revision: str, path: str) -> str:
    repo_quoted = quote(repo.strip("/"), safe="/")
    revision_quoted = quote(revision, safe="")
    path_quoted = "/".join(quote(part, safe="") for part in path.split("/"))
    return f"https://huggingface.co/datasets/{repo_quoted}/resolve/{revision_quoted}/{path_quoted}"


def _read_json_url(url: str) -> object:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.tmp")
    with urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(destination)


def _entries(repo: str, revision: str) -> list[dict[str, object]]:
    payload = _read_json_url(_tree_url(repo, revision))
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Hugging Face tree payload")
    return [entry for entry in payload if isinstance(entry, dict)]


def _import_huggingface_hub() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --backend huggingface. "
            "Install it with `python -m pip install huggingface_hub` or update the conda env."
        ) from exc
    return HfApi, snapshot_download


def _hf_entries(repo: str, revision: str) -> list[dict[str, object]]:
    HfApi, _ = _import_huggingface_hub()
    api = HfApi()
    entries: list[dict[str, object]] = []
    for entry in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True):
        path = str(getattr(entry, "path", "")).strip("/")
        if not path:
            continue
        entry_type = str(getattr(entry, "type", "") or "")
        if not entry_type:
            entry_type = "file" if entry.__class__.__name__ == "RepoFile" else "directory"
        entries.append({"type": entry_type, "path": path})
    return entries


def _relative_output_path(path: str, strip_prefix: str) -> Path:
    normalized = path.strip("/")
    prefix = strip_prefix.strip("/")
    if prefix:
        prefix_with_sep = f"{prefix}/"
        if normalized == prefix:
            normalized = ""
        elif normalized.startswith(prefix_with_sep):
            normalized = normalized[len(prefix_with_sep) :]
    if not normalized:
        raise ValueError(f"Path {path!r} resolves to an empty output path")
    return Path(*normalized.split("/"))


def _selected_files(entries: list[dict[str, object]], include_prefix: str) -> list[str]:
    prefix = include_prefix.strip("/")
    selected: list[str] = []
    for entry in entries:
        if str(entry.get("type")) != "file":
            continue
        path = str(entry.get("path") or "").strip("/")
        if not path:
            continue
        if prefix and path != prefix and not path.startswith(f"{prefix}/"):
            continue
        selected.append(path)
    return sorted(selected)


def _hf_allow_patterns(include_prefix: str) -> list[str] | None:
    prefix = include_prefix.strip("/")
    if not prefix:
        return None
    return [prefix, f"{prefix}/**"]


def _download_with_huggingface_hub(
    *,
    repo: str,
    revision: str,
    destination: Path,
    include_prefix: str,
    strip_prefix: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if strip_prefix.strip("/"):
        raise RuntimeError("--strip-prefix is only supported with --backend url")
    _, snapshot_download = _import_huggingface_hub()
    if dry_run:
        files = _selected_files(_hf_entries(repo, revision), include_prefix)
        if not files:
            raise RuntimeError("No files matched the requested Hugging Face dataset tree")
        for path in files:
            print(f"download {path} -> {destination / Path(*path.split('/'))}")
        return
    local_dir = destination
    created_tmp = False
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        local_dir = destination.with_name(f".{destination.name}.tmp")
        if local_dir.exists():
            shutil.rmtree(local_dir)
        created_tmp = True
    try:
        local_path = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=_hf_allow_patterns(include_prefix),
            force_download=overwrite,
        )
    except Exception:
        if created_tmp:
            shutil.rmtree(local_dir, ignore_errors=True)
        raise
    if created_tmp:
        local_dir.replace(destination)
        local_path = str(destination)
    print(f"downloaded {repo}@{revision} -> {local_path}")


def _download_with_url_backend(args: argparse.Namespace, destination: Path) -> None:
    files = _selected_files(_entries(args.repo, args.revision), args.include_prefix)
    if not files:
        raise RuntimeError("No files matched the requested Hugging Face dataset tree")
    for path in files:
        output_path = destination / _relative_output_path(path, args.strip_prefix)
        if output_path.exists() and not args.overwrite:
            print(f"skip existing {output_path}")
            continue
        print(f"download {path} -> {output_path}")
        if not args.dry_run:
            _download_url(_file_url(args.repo, args.revision, path), output_path)


def main() -> int:
    args = _parse_args()
    destination = Path(args.destination).expanduser()
    try:
        backend = "url" if args.backend == "auto" and args.strip_prefix else args.backend
        if backend in {"auto", "huggingface"}:
            _download_with_huggingface_hub(
                repo=args.repo,
                revision=args.revision,
                destination=destination,
                include_prefix=args.include_prefix,
                strip_prefix=args.strip_prefix,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        else:
            _download_with_url_backend(args, destination)
    except (OSError, RuntimeError, URLError, ValueError) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
