from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.stage6_hf_diagnostics import (
    _load_fixture,
    load_native_unwrap,
    oracle_threshold_shift_summary,
)


def coupled_threshold_summary(
    ifgw,
    rowcost,
    colcost,
    native,
    snaphu,
    *,
    nshortcycle: int = 200,
) -> dict[str, Any]:
    summary = oracle_threshold_shift_summary(
        ifgw,
        rowcost,
        colcost,
        native,
        snaphu,
        nshortcycle=nshortcycle,
    )
    steps = summary.get("thresholds", [])
    gains = [int(step["gain"]) for step in steps]
    negative_steps = [step for step in steps if int(step["gain"]) < 0]
    positive_gain = sum(gain for gain in gains if gain > 0)
    negative_gain = sum(gain for gain in gains if gain < 0)
    sequential_gain = int(summary["sequential_gain"])
    return {
        "correction_min": int(summary["correction_min"]),
        "correction_max": int(summary["correction_max"]),
        "threshold_count": len(steps),
        "sequential_gain": sequential_gain,
        "objective_delta_native_minus_snaphu": int(summary["objective_delta_native_minus_snaphu"]),
        "positive_gain": int(positive_gain),
        "negative_gain": int(negative_gain),
        "negative_step_count": len(negative_steps),
        "requires_coupled_acceptance": sequential_gain > 0 and bool(negative_steps),
        "thresholds": steps,
    }


def analyze_fixture(root: Path, native_file: Path, *, nshortcycle: int = 200) -> dict[str, Any]:
    _nzix, ifgw, rowcost, colcost, snaphu = _load_fixture(root)
    native = load_native_unwrap(native_file, ifgw.shape)
    return coupled_threshold_summary(
        ifgw,
        rowcost,
        colcost,
        native,
        snaphu,
        nshortcycle=nshortcycle,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize coupled Stage 6 oracle threshold shifts.")
    parser.add_argument("--root", required=True, type=Path, help="Stage 6 fixture root.")
    parser.add_argument("--native-file", required=True, type=Path, help="Cached native unwrap .npy.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--nshortcycle", type=int, default=200)
    args = parser.parse_args()

    payload = analyze_fixture(args.root, args.native_file, nshortcycle=args.nshortcycle)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
