from pathlib import Path

from pystamps.parity_contract import build_parity_contract


def test_build_parity_contract_without_datasets(tmp_path: Path) -> None:
    inputs_root = tmp_path / "inputs_and_outputs"
    inputs_root.mkdir()

    contract = build_parity_contract(inputs_root)

    assert contract["inputs_root"] == "inputs_and_outputs"
    assert contract["datasets"] == []
    assert contract["supported_audit"]["entrypoint"] == "scripts/validate_audit.py"
    assert contract["supported_audit"]["output_artifact"] == "inputs_and_outputs/validation_runs/latest_audit.json"
    assert contract["supported_audit"]["required_result_fields"] == [
        "generated_at_utc",
        "contract",
        "missing_datasets",
        "audits",
        "failed_workflows",
        "completed",
        "interrupted",
        "ok",
    ]
    assert contract["required_workflow_names"] == ["full_validation"]
    assert "full_validation" in contract["workflows"]
    assert contract["workflows"]["full_validation"]["driver"] == "scripts/validate_audit.py"
    assert contract["workflows"]["full_validation"]["output_artifact"] == "inputs_and_outputs/validation_runs/latest_audit.json"
