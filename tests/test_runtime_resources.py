from __future__ import annotations

from pystamps.runtime import resources


def test_cpu_budget_uses_smallest_available_limit(monkeypatch) -> None:
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(resources, "_cpu_affinity_count", lambda: 8)
    monkeypatch.setattr(resources, "_cgroup_cpu_quota", lambda: 3)

    assert resources.cpu_budget() == 3


def test_cpu_budget_handles_unrestricted_host(monkeypatch) -> None:
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(resources, "_cpu_affinity_count", lambda: None)
    monkeypatch.setattr(resources, "_cgroup_cpu_quota", lambda: None)

    assert resources.cpu_budget() == 6
