from __future__ import annotations

import pandas as pd

from notebooks import utils as nb_utils


def test_select_same_burst_stack_accepts_cached_content_date_strings(monkeypatch) -> None:
    monkeypatch.setattr(
        nb_utils,
        "STACK_GROUP_COLUMNS",
        [
            "BurstId",
            "SwathIdentifier",
            "RelativeOrbitNumber",
            "OrbitDirection",
            "PolarisationChannels",
        ],
        raising=False,
    )
    monkeypatch.setattr(nb_utils, "REQUIRE_SINGLE_PLATFORM", True, raising=False)

    frame = pd.DataFrame(
        [
            {
                "Id": "a",
                "Name": "S1A_a",
                "ParentProductName": "S1A_a",
                "ContentDate": "{'Start': '2024-01-01T00:00:00Z', 'End': '2024-01-01T00:01:00Z'}",
                "BurstId": 1,
                "SwathIdentifier": "IW2",
                "RelativeOrbitNumber": 117,
                "OrbitDirection": "ASCENDING",
                "PolarisationChannels": "VV",
                "coverage": 99.0,
            },
            {
                "Id": "b",
                "Name": "S1A_b",
                "ParentProductName": "S1A_b",
                "ContentDate": "{'Start': '2024-01-13T00:00:00Z', 'End': '2024-01-13T00:01:00Z'}",
                "BurstId": 1,
                "SwathIdentifier": "IW2",
                "RelativeOrbitNumber": 117,
                "OrbitDirection": "ASCENDING",
                "PolarisationChannels": "VV",
                "coverage": 99.0,
            },
        ]
    )

    selected = nb_utils.select_same_burst_stack(frame, min_stack_size=2, target_stack_size=2)

    assert selected["date_yyyymmdd"].tolist() == ["20240101", "20240113"]
