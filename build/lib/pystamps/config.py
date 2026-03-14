from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RuntimeConfig:
    io_workers: int = 8
    cpu_workers: int = 0
    backend: str = "auto"
    stage7_chunk_ps: int = 100_000
    stage8_chunk_edges: int = 200_000
    enable_mat_stage_cache: bool = True
    stage2_debug: bool = False
    stage4_debug: bool = False


@dataclass(slots=True)
class ToleranceConfig:
    rtol: float = 1e-5
    atol: float = 1e-7
    wrap_equivalence: bool = True
    wrap_period: float = 2.0 * 3.141592653589793
    wrap_keys: tuple[str, ...] = ("ph_uw", "ph", "dph_noise", "dph_space_uw")


@dataclass(slots=True)
class ExternalToolsConfig:
    triangle: str = "triangle"
    snaphu: str = "snaphu"


@dataclass(slots=True)
class CompatibilityConfig:
    reference_root: str | None = None
    strict_reference: bool = False


@dataclass(slots=True)
class RunConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    tolerance: ToleranceConfig = field(default_factory=ToleranceConfig)
    tools: ExternalToolsConfig = field(default_factory=ExternalToolsConfig)
    compat: CompatibilityConfig = field(default_factory=CompatibilityConfig)


class ConfigError(ValueError):
    """Raised when configuration is malformed."""


def _load_raw(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    elif suffix == ".json":
        payload = json.loads(text)
    else:
        raise ConfigError("Config must be YAML or JSON")

    if not isinstance(payload, dict):
        raise ConfigError("Top-level config payload must be an object")
    return payload


def _as_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be an object")
    return value


def load_config(path: str | Path | None = None) -> RunConfig:
    if path is None:
        return RunConfig()

    raw = _load_raw(Path(path))
    runtime = RuntimeConfig(**_as_dict(raw, "runtime"))
    tol_payload = _as_dict(raw, "tolerance")
    wrap_keys = tol_payload.get("wrap_keys")
    if isinstance(wrap_keys, list):
        tol_payload = {**tol_payload, "wrap_keys": tuple(str(v) for v in wrap_keys)}
    tolerance = ToleranceConfig(**tol_payload)
    tools = ExternalToolsConfig(**_as_dict(raw, "tools"))
    compat = CompatibilityConfig(**_as_dict(raw, "compat"))
    return RunConfig(runtime=runtime, tolerance=tolerance, tools=tools, compat=compat)
