from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from scripts import download_hf_dataset


def test_hf_tree_url_quotes_dataset_repo() -> None:
    assert (
        download_hf_dataset._tree_url("owner/name", "main")
        == "https://huggingface.co/api/datasets/owner/name/tree/main?recursive=1"
    )


def test_hf_file_url_quotes_path_parts() -> None:
    assert (
        download_hf_dataset._file_url("owner/name", "main", "PATCH 1/ps1.mat")
        == "https://huggingface.co/datasets/owner/name/resolve/main/PATCH%201/ps1.mat"
    )


def test_hf_selected_files_filters_directories_and_prefixes() -> None:
    entries = [
        {"type": "directory", "path": "InSAR_dataset_test"},
        {"type": "file", "path": "InSAR_dataset_test/ps2.mat"},
        {"type": "file", "path": "other/ps2.mat"},
    ]

    assert download_hf_dataset._selected_files(entries, "InSAR_dataset_test") == [
        "InSAR_dataset_test/ps2.mat",
    ]


def test_hf_relative_output_path_strips_prefix() -> None:
    assert download_hf_dataset._relative_output_path(
        "inputs_and_outputs/InSAR_dataset_test/ps2.mat",
        "inputs_and_outputs",
    ) == Path("InSAR_dataset_test") / "ps2.mat"


def test_hf_backend_calls_snapshot_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.update(kwargs)
        return str(tmp_path)

    fake_hub = types.SimpleNamespace(HfApi=object, snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    download_hf_dataset._download_with_huggingface_hub(
        repo="owner/name",
        revision="main",
        destination=tmp_path,
        include_prefix="PATCH_1",
        strip_prefix="",
        overwrite=True,
        dry_run=False,
    )

    assert calls == {
        "repo_id": "owner/name",
        "repo_type": "dataset",
        "revision": "main",
        "local_dir": str(tmp_path),
        "allow_patterns": ["PATCH_1", "PATCH_1/**"],
        "force_download": True,
    }


def test_hf_backend_uses_temp_dir_for_new_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "InSAR_dataset_test"
    calls: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.update(kwargs)
        Path(str(kwargs["local_dir"])).mkdir(parents=True)
        return str(kwargs["local_dir"])

    fake_hub = types.SimpleNamespace(HfApi=object, snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    download_hf_dataset._download_with_huggingface_hub(
        repo="owner/name",
        revision="main",
        destination=destination,
        include_prefix="",
        strip_prefix="",
        overwrite=False,
        dry_run=False,
    )

    assert calls["local_dir"] == str(tmp_path / ".InSAR_dataset_test.tmp")
    assert destination.is_dir()
    assert not (tmp_path / ".InSAR_dataset_test.tmp").exists()


def test_hf_backend_removes_temp_dir_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "InSAR_dataset_test"

    def fake_snapshot_download(**kwargs: object) -> str:
        Path(str(kwargs["local_dir"])).mkdir(parents=True)
        raise OSError("network down")

    fake_hub = types.SimpleNamespace(HfApi=object, snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(OSError, match="network down"):
        download_hf_dataset._download_with_huggingface_hub(
            repo="owner/name",
            revision="main",
            destination=destination,
            include_prefix="",
            strip_prefix="",
            overwrite=False,
            dry_run=False,
        )

    assert not destination.exists()
    assert not (tmp_path / ".InSAR_dataset_test.tmp").exists()


def test_hf_backend_rejects_strip_prefix(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="--strip-prefix"):
        download_hf_dataset._download_with_huggingface_hub(
            repo="owner/name",
            revision="main",
            destination=tmp_path,
            include_prefix="",
            strip_prefix="owner",
            overwrite=False,
            dry_run=False,
        )
