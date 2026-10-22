"""In-process swap: FlashInfer CuTe fused-add RMSNorm -> norm_lab 1-row kernel for small M.

flashinfer.norm.{fused_add_rmsnorm, gemma_fused_add_rmsnorm} resolve the module global
`fused_add_rmsnorm_cute` at call time, so rebinding it routes SGLang's RMSNorm and
GemmaRMSNorm fused-add calls here. Installed packages are not modified. Must be installed
before CUDA-graph capture; replays then contain whichever kernel was captured.
"""

import os

import flashinfer.norm as fn

from local_cute import fused_add_rmsnorm_1row as one

# Dispatch range (inclusive); outside it the original kernel runs unchanged.
MIN_M = int(os.environ.get("NORM_LAB_MIN_M", "1"))
MAX_M = int(os.environ.get("NORM_LAB_MAX_M", "64"))
calls = {"1row": 0, "original": 0}
_original = fn.fused_add_rmsnorm_cute


def _dispatch(input, residual, weight, eps=1e-6, weight_bias=0.0, enable_pdl=False):
    if MIN_M <= input.shape[0] <= MAX_M:
        calls["1row"] += 1
        return one.fused_add_rmsnorm_cute(input, residual, weight, eps, weight_bias, enable_pdl)
    calls["original"] += 1
    return _original(input, residual, weight, eps, weight_bias, enable_pdl)


def install():
    assert not fn._use_cuda_norm(__import__("torch").device("cuda")), "CuTe backend expected"
    fn.fused_add_rmsnorm_cute = _dispatch
