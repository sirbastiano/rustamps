import importlib.util

import numpy as np
import pytest

from pystamps.kernels import run_stage3_clap_filt_patch_stack_kernel


@pytest.mark.skipif(
    importlib.util.find_spec("pystamps.kernels._stage2_native") is None,
    reason="native stage-2 extension not available",
)
def test_stage3_clap_filt_patch_stack_native_matches_python() -> None:
    base = np.arange(1, 33, dtype=np.float64).reshape(4, 4, 2)
    ph_stack = (base + 1j * np.flip(base, axis=0)).astype(np.complex64)
    low_pass = np.ones((4, 4), dtype=np.float64)

    expected = run_stage3_clap_filt_patch_stack_kernel(
        ph_stack,
        alpha=1.1,
        beta=0.25,
        low_pass=low_pass,
        backend="python",
    )
    observed = run_stage3_clap_filt_patch_stack_kernel(
        ph_stack,
        alpha=1.1,
        beta=0.25,
        low_pass=low_pass,
        backend="native",
    )

    assert observed.dtype == np.complex128
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)
