from pathlib import Path

import pytest

from pystamps.io.dataset import DatasetError, discover_dataset, infer_merged_stage


def test_discover_dataset_rejects_missing_listed_patch(tmp_path: Path) -> None:
    (tmp_path / "PATCH_1").mkdir()
    (tmp_path / "patch.list").write_text("PATCH_1\nPATCH_2\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=r"missing patch directories: PATCH_2"):
        discover_dataset(tmp_path)


def test_stage8_progress_is_keyed_by_legacy_scn_product(tmp_path: Path) -> None:
    (tmp_path / "uw_space_time.mat").touch()
    assert infer_merged_stage(tmp_path) == 0

    (tmp_path / "scn2.mat").touch()
    assert infer_merged_stage(tmp_path) == 8
