# Experiment data

This directory contains the evidence used in the article.

## `kernel/`

- `rowmap_check_*.txt` contains the 117-case equivalence run.
- `rowmap_*_r0.csv`, `r1.csv`, and `r2.csv` contain 60 paired small-M trials across three processes.
- `rowmap_kernels_*` contains CUPTI kernel durations recorded during those runs.
- The `r9` files are a separate exploratory round for M=144, 576, and 2304. They use fewer trials and should not be described as having the same protocol as the small-M results.

### `kernel/validation/`

Boundary, sensitivity, and runtime-diagnosis data produced by the scripts in `repro/validation/`:

- `rowmap_boundary_*_r0..r2.csv` (+ `rowmap_kernels_boundary_*`): 3-round boundary sweep at H=2048 BF16, M = 1 to 2304.
- `rowmap_sens_H*_*_r0..r1.csv` (+ kernels files): hidden-size and dtype sensitivity (H = 1024, 2560, 4096 BF16; H = 2048 FP16).
- `diag_*_r0..r1.csv`: PDL on/off × CUDA Graph/eager × warm/cold-L2 at M = 1, 2, 4, 8, with critical-path (exposed) time.
- `stages_*.log`: correctness results (all bitwise identical) and launch geometry for each configuration.

## `model/guarded/`

Guarded-dispatch (4 ≤ M ≤ 64) Qwen3.5-2B runs: `guard_*` for batch 1, 2, 4, 8 and 16, and `guard32_*` for batch 32. Batch 32 ran in its own process because request slots were not released between batch sizes. Every CSV contains 300 timed decode steps, with four alternating rounds per arm. Metadata contains a separate three-step CUPTI trace, the dispatch range, launch geometry, and critical-path norm time. The 16 token and final-logit files are omitted; their hashes are in `artifact_hashes.sha256`.

## `profiler/`

`ncu_rowmap_m1_4_5_8_metrics.csv` contains the selected Nsight Compute metrics used in the article. The full report remains in the private working directory because Nsight embedded the original WSL command and Linux home path in the binary. Its SHA-256 hash is recorded in `artifact_hashes.sha256` so the local capture can be identified later.

The profiler capture explains execution behavior. It is not a steady-state latency benchmark.
