#!/usr/bin/env python3
"""Developer-only standard-library harness for native Rust release gates."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("inputs_and_outputs/validation_runs/native_rust_step_validation_latest.json")


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)


NATIVE_BINARY = str(
    Path("target") / "release" / ("rustamps.exe" if os.name == "nt" else "rustamps")
)
CARGO = os.environ.get("CARGO") or shutil.which("cargo")
if CARGO is None:
    cargo_name = "cargo.exe" if os.name == "nt" else "cargo"
    rustup_cargo = Path.home() / ".cargo" / "bin" / cargo_name
    CARGO = str(rustup_cargo) if rustup_cargo.is_file() else "cargo"


STEPS: tuple[Step, ...] = (
    Step("rust-fmt", [CARGO, "fmt", "--all", "--", "--check"]),
    Step("rust-check", [CARGO, "check", "--workspace", "--locked"]),
    Step("rust-tests", [CARGO, "test", "--workspace", "--locked"]),
    Step("rust-release", [CARGO, "build", "--release", "--locked"]),
    Step("native-version", [NATIVE_BINARY, "--version"]),
    Step("native-backends", [NATIVE_BINARY, "describe-backends"]),
)


def _ru_maxrss_bytes(value: int) -> int:
    return int(value) if platform.system() == "Darwin" else int(value) * 1024


def _trim(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _measure_child(payload_path: Path) -> int:
    import resource

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    proc = subprocess.run(
        payload["command"],
        cwd=payload["cwd"],
        env=payload["env"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_bytes": _ru_maxrss_bytes(int(usage.ru_maxrss)),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    Path(payload["result_path"]).write_text(json.dumps(result), encoding="utf-8")
    return 0


def _run_step(step: Step, *, verbose: bool) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(step.env)

    with TemporaryDirectory(prefix="rustamps-rust-validate-") as tmp:
        payload_path = Path(tmp) / "payload.json"
        result_path = Path(tmp) / "result.json"
        payload_path.write_text(
            json.dumps(
                {
                    "command": step.command,
                    "cwd": str(REPO_ROOT),
                    "env": env,
                    "result_path": str(result_path),
                }
            ),
            encoding="utf-8",
        )
        helper = subprocess.run(
            [sys.executable, __file__, "--_measure-child", str(payload_path)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if helper.returncode != 0 or not result_path.exists():
            details = helper.stderr or helper.stdout
            raise RuntimeError(details or f"measurement helper failed for {step.name}")
        measured = json.loads(result_path.read_text(encoding="utf-8"))

    peak_bytes = measured["peak_rss_bytes"]
    result: dict[str, Any] = {
        "name": step.name,
        "command": step.command,
        "returncode": measured["returncode"],
        "elapsed_seconds": measured["elapsed_seconds"],
        "peak_rss_bytes": peak_bytes,
        "peak_rss_gb": round(peak_bytes / 1_000_000_000, 3) if peak_bytes is not None else None,
        "peak_rss_gib": round(peak_bytes / (1024**3), 3) if peak_bytes is not None else None,
    }
    if measured["returncode"] != 0 or verbose:
        result["stdout_tail"] = _trim(measured["stdout"])
        result["stderr_tail"] = _trim(measured["stderr"])
    return result


def _print_step(result: dict[str, Any]) -> None:
    status = "PASS" if result["returncode"] == 0 else "FAIL"
    peak_gb = result["peak_rss_gb"]
    peak_gib = result["peak_rss_gib"]
    peak = "unknown"
    if peak_gb is not None and peak_gib is not None:
        peak = f"{peak_gb:.3f} GB / {peak_gib:.3f} GiB"
    print(f"{status} {result['name']}: {result['elapsed_seconds']:.3f}s, peak_rss={peak}")
    if result["returncode"] != 0:
        if result.get("stdout_tail"):
            print(result["stdout_tail"])
        if result.get("stderr_tail"):
            print(result["stderr_tail"], file=sys.stderr)


def _selected_steps(names: list[str]) -> list[Step]:
    if not names:
        return list(STEPS)
    by_name = {step.name: step for step in STEPS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"unknown step(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone Rust validation steps with peak RSS output. "
            "This standard-library Python wrapper is developer tooling, not a runtime dependency."
        )
    )
    parser.add_argument(
        "--step",
        action="append",
        default=[],
        help="Run one named step; repeat for multiple steps.",
    )
    parser.add_argument("--list", action="store_true", help="List available steps and exit.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Keep command stdout/stderr tails for passing steps.",
    )
    parser.add_argument("--_measure-child", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args._measure_child is not None:
        return _measure_child(args._measure_child)
    if args.list:
        for step in STEPS:
            print(step.name)
        return 0

    report = {
        "profile": "native-rust-step-validation",
        "platform": platform.platform(),
        "steps": [],
    }
    failed = False
    for step in _selected_steps(args.step):
        result = _run_step(step, verbose=bool(args.verbose))
        report["steps"].append(result)
        _print_step(result)
        failed = failed or result["returncode"] != 0
        if failed:
            break

    report["ok"] = not failed
    report["elapsed_seconds"] = round(sum(item["elapsed_seconds"] for item in report["steps"]), 3)
    peak = max(report["steps"], key=lambda item: item["peak_rss_bytes"], default=None)
    report["max_peak_rss_bytes"] = None if peak is None else peak["peak_rss_bytes"]
    report["max_peak_rss_gb"] = None if peak is None else peak["peak_rss_gb"]
    report["max_peak_rss_gib"] = None if peak is None else peak["peak_rss_gib"]
    report["max_peak_rss_step"] = None if peak is None else peak["name"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
