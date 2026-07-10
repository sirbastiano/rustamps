import numpy as np

import pystamps.kernels.accelerated as accel
from pystamps.kernels import run_stage6_unwrap_grid_kernel


def test_stage6_unwrap_grid_preserves_native_flow_cycle_count(monkeypatch) -> None:
    class _FakeNative:
        def stage6_unwrap_grid(
            self,
            ifgw: np.ndarray,
            rowcost: np.ndarray,
            colcost: np.ndarray,
            nshortcycle: float,
            threads: int,
        ) -> dict[str, np.ndarray | float | int]:
            return {
                "ifguw": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "msd": 5.5,
                "flow_cycles": 7,
                "flow_objective": 12345,
                "post_label_flow_cycles": 3,
                "post_label_flow_objective": 6789,
            }

    monkeypatch.setattr(accel, "_load_stage2_native_module", lambda: _FakeNative())

    out = run_stage6_unwrap_grid_kernel(
        np.ones((2, 2), dtype=np.complex64),
        np.zeros((1, 8), dtype=np.int16),
        np.zeros((2, 4), dtype=np.int16),
        backend="native",
    )

    assert out["flow_cycles"] == 7
    assert out["flow_objective"] == 12345
    assert out["post_label_flow_cycles"] == 3
    assert out["post_label_flow_objective"] == 6789
