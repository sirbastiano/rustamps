from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.import_dataset_archive import import_archive


def test_import_dataset_archive_copies_single_root_directory(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("downloaded-root/patch.list", "PATCH_1\n")
        handle.writestr("downloaded-root/PATCH_1/ps1.mat", "placeholder")

    destination = tmp_path / "inputs_and_outputs" / "InSAR_dataset_test"
    import_archive(archive, destination)

    assert (destination / "patch.list").read_text(encoding="utf-8") == "PATCH_1\n"
    assert (destination / "PATCH_1" / "ps1.mat").read_text(encoding="utf-8") == "placeholder"


def test_import_dataset_archive_rejects_unsafe_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="Unsafe archive path"):
        import_archive(archive, tmp_path / "out")


def test_import_dataset_archive_requires_overwrite_for_existing_destination(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset/patch.list", "PATCH_1\n")
    destination = tmp_path / "dataset"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="Destination exists"):
        import_archive(archive, destination)
