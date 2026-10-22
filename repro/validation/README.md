# Validation-campaign scripts

These are the exact scripts that produced `data/kernel/validation/` and `data/model/guarded/`. The `local_cute/` kernels are identical to `repro/local_cute/`.

## Kernel source format

The measured baseline and candidate are CUDA kernels written with NVIDIA CuTe DSL, so their source files are Python rather than handwritten `.cu` files:

- `local_cute/fused_add_rmsnorm_copy.py`: unchanged FlashInfer 0.6.18 CuTe DSL source used as the local baseline.
- `local_cute/fused_add_rmsnorm_1row.py`: measured one-row-per-block candidate.

CuTe compiles these definitions for the target GPU. No separate handwritten `.cu` implementation was used in the reported comparison. Generated compiler-cache files and FlashInfer's different C++/CUDA JIT fallback are therefore not included as kernel source.

- `rowmap.py check|bench|diag --H <hidden> --dtype bfloat16|float16 --tag <name>`: correctness, paired benchmark, and the PDL/graph/cache diagnosis.
- `model_bench.py --set guard --kernel installed|1row`: Qwen3.5-2B decode A/B. Set the dispatch range with `NORM_LAB_MIN_M=4 NORM_LAB_MAX_M=64`.

Model runs used SGLang flags `--attention-backend triton --cuda-graph-backend-prefill disabled --disable-radix-cache --context-length 1024 --max-mamba-cache-size 36 --mem-fraction-static 0.8`. They also waited for 6.4 GB of free VRAM before each process, because on an 8 GB GPU shared with Windows a lower fraction or a busy GPU leaves no room for the KV cache.
