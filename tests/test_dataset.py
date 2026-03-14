from pathlib import Path

from pystamps.io.dataset import discover_dataset, infer_merged_stage, infer_patch_stage


def test_discover_dataset_has_patches() -> None:
    dataset = Path("inputs_and_outputs/InSAR_dataset_test")
    layout = discover_dataset(dataset)

    assert layout.root.exists()
    assert len(layout.patches) >= 4


def test_stage_inference_on_reference_data() -> None:
    dataset = Path("inputs_and_outputs/InSAR_dataset_test")
    layout = discover_dataset(dataset)

    for patch in layout.patches:
        assert infer_patch_stage(patch) >= 4

    assert infer_merged_stage(dataset) >= 7
