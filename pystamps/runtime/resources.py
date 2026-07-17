"""Host-aware CPU budgeting for native and process-backed work."""
from __future__ import annotations

import math
import os
from pathlib import Path


def _cpu_affinity_count() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return None
    try:
        return len(get_affinity(0))
    except OSError:
        return None


def _read_cpu_quota(path: Path) -> int | None:
    try:
        quota, period = path.read_text().strip().split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    try:
        quota_value, period_value = int(quota), int(period)
    except ValueError:
        return None
    if quota_value <= 0 or period_value <= 0:
        return None
    return max(1, math.ceil(quota_value / period_value))


def _cgroup_cpu_quota() -> int | None:
    quota = _read_cpu_quota(Path("/sys/fs/cgroup/cpu.max"))
    if quota is not None:
        return quota
    try:
        quota_value = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period_value = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
    except (OSError, ValueError):
        return None
    if quota_value <= 0 or period_value <= 0:
        return None
    return max(1, math.ceil(quota_value / period_value))


def cpu_budget() -> int:
    """Return the CPU count actually available to this process.

    This respects scheduler affinity and common Linux cgroup quotas, which can
    be smaller than the machine-wide count reported by ``os.cpu_count()``.
    """
    counts = [max(1, os.cpu_count() or 1)]
    affinity = _cpu_affinity_count()
    quota = _cgroup_cpu_quota()
    if affinity is not None:
        counts.append(max(1, affinity))
    if quota is not None:
        counts.append(max(1, quota))
    return min(counts)
