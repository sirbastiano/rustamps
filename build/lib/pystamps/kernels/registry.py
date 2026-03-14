from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


KernelFn = Callable[..., Any]


@dataclass(slots=True)
class KernelImplementation:
    cpu: KernelFn
    gpu: KernelFn | None = None
    native: KernelFn | None = None


class KernelRegistry:
    def __init__(self) -> None:
        self._impls: dict[str, KernelImplementation] = {}

    def register(self, name: str, cpu: KernelFn, gpu: KernelFn | None = None, native: KernelFn | None = None) -> None:
        self._impls[name] = KernelImplementation(cpu=cpu, gpu=gpu, native=native)

    def get(self, name: str, backend: str = "auto") -> KernelFn:
        if name not in self._impls:
            raise KeyError(f"Kernel '{name}' is not registered")

        impl = self._impls[name]
        if backend == "native" and impl.native is not None:
            return impl.native
        if backend == "gpu" and impl.gpu is not None:
            return impl.gpu
        return impl.cpu


DEFAULT_REGISTRY = KernelRegistry()
