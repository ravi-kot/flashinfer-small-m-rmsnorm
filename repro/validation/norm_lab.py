"""Production baseline for SGLang 0.5.21 gemma_fused_add_rmsnorm (Qwen3.6 decoder norms).

Modes:
  probe    runtime dispatch evidence: backend, enable_pdl, kernel name, launch geometry
  check    correctness vs explicit FP64 references + in-place contract
  bench    CUDA-graph latency with per-execution input reset, raw CSV
  profile  a few warmed eager calls inside cudaProfilerStart/Stop (for ncu/nsys)
"""

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

H_QWEN36 = 2048  # Qwen/Qwen3.6-35B-A3B-FP8 config.json: text_config.hidden_size
EPS = 1e-6  # same config: text_config.rms_norm_eps
BF16 = torch.bfloat16
OUT = Path(__file__).resolve().parent / "results"
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def environment():
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    props = torch.cuda.get_device_properties(0)

    def ver(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": props.multi_processor_count,
        "l2_bytes": props.L2_cache_size,
        "smem_per_block_optin": props.shared_memory_per_block_optin,
        "vram_bytes": props.total_memory,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": {
            n: ver(n)
            for n in ("sglang", "sglang-kernel", "flashinfer-python", "triton",
                      "nvidia-cutlass-dsl")
        },
    }


def telemetry():
    import pynvml

    h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def safe(f, *a):
        try:
            return f(h, *a)
        except pynvml.NVMLError:
            return None

    power = safe(pynvml.nvmlDeviceGetPowerUsage)
    return {
        "sm_clock_mhz": safe(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM),
        "mem_clock_mhz": safe(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_MEM),
        "temp_c": safe(pynvml.nvmlDeviceGetTemperature, pynvml.NVML_TEMPERATURE_GPU),
        "power_w": None if power is None else power / 1000,
    }


def inputs(M, H, seed, case="random", dtype=BF16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, device="cuda")
    x, r = rnd(M, H), rnd(M, H)
    w = 0.1 * rnd(H)  # Gemma weights are stored centred on 0 (scale = 1 + w)
    if case == "zeros":
        x.zero_(), r.zero_()
    elif case == "exact_cancel":
        r = -x.to(dtype).float()
    elif case == "near_cancel":  # h = x + r is tiny relative to x and r
        r = -x.to(dtype).float() + 1e-3 * rnd(M, H)
    elif case == "residual_dominant":  # deep-layer residual stream >> new output
        r = 64 * r
    elif case == "large":
        x, r = 256 * x, 256 * r
    elif case == "weight_minus_one":
        w = torch.full((H,), -1.0, device="cuda")
    return x.to(dtype), r.to(dtype), w.to(dtype)


def run_norm(x, r, w):
    from sgl_kernel import gemma_fused_add_rmsnorm

    return gemma_fused_add_rmsnorm(x, r, w, EPS)  # enable_pdl=None, as layernorm.py


# ---------------------------------------------------------------- probe
def probe():
    import flashinfer.norm as fn
    import sgl_kernel.elementwise as ew

    env = environment()
    dev = torch.device("cuda")
    x, r, w = inputs(1, H_QWEN36, 0)
    run_norm(x.clone(), r.clone(), w)  # compile outside the observed call
    torch.cuda.synchronize()

    # In-process spies on the module globals FlashInfer resolves at call time.
    calls, real_cute, real_jit = [], fn.fused_add_rmsnorm_cute, fn.get_norm_module

    def spy_cute(*a, **k):
        calls.append({"backend": "cute_dsl", "weight_bias": k.get("weight_bias"),
                      "enable_pdl": k.get("enable_pdl")})
        return real_cute(*a, **k)

    def spy_jit():
        calls.append({"backend": "cuda_jit"})
        return real_jit()

    fn.fused_add_rmsnorm_cute, fn.get_norm_module = spy_cute, spy_jit
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                run_norm(x.clone(), r.clone(), w)
            torch.cuda.synchronize()
    finally:
        fn.fused_add_rmsnorm_cute, fn.get_norm_module = real_cute, real_jit

    trace = OUT / f"probe_trace_{STAMP}.json"
    prof.export_chrome_trace(str(trace))
    kernels = {}
    for ev in json.loads(trace.read_text())["traceEvents"]:
        if ev.get("cat") != "kernel":
            continue
        a = ev.get("args", {})
        k = kernels.setdefault(ev["name"], {"count": 0, "dur_us": [], "grid": a.get("grid"),
                                            "block": a.get("block"),
                                            "regs": a.get("registers per thread"),
                                            "smem": a.get("shared memory")})
        k["count"] += 1
        k["dur_us"].append(ev.get("dur"))

    module_path = None
    try:  # the exact production entry point, called directly
        # layernorm.py transitively imports deep_gemm, which asserts CUDA_HOME at
        # import. No system toolkit here: point this process at the pip toolkit.
        import os
        import nvidia.cu13

        os.environ.setdefault("CUDA_HOME", list(nvidia.cu13.__path__)[0])
        from sglang.srt.layers.layernorm import GemmaRMSNorm

        m = GemmaRMSNorm(H_QWEN36, eps=EPS).to(device="cuda", dtype=BF16)
        m.weight.data.copy_(w)
        xa, ra, xb, rb = x.clone(), r.clone(), x.clone(), r.clone()
        m.forward_cuda(xa, ra)
        run_norm(xb, rb, w)
        module_path = {"bit_identical_x": torch.equal(xa, xb),
                       "bit_identical_residual": torch.equal(ra, rb)}
    except Exception as e:  # report, do not hide
        module_path = {"error": repr(e)}

    report = {
        **env,
        "sgl_kernel_has_flashinfer": ew._has_flashinfer,
        "FLASHINFER_USE_CUDA_NORM_forced": fn._USE_CUDA_NORM,
        "flashinfer_use_cuda_norm_on_device": fn._use_cuda_norm(dev),
        "observed_backend_calls": calls,
        "kernels_in_5_calls": kernels,
        "GemmaRMSNorm.forward_cuda_vs_sgl_kernel": module_path,
        "trace": str(trace),
    }
    (OUT / f"probe_{STAMP}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------- check
def bf16_key(t):
    """Monotonic integer key for 16-bit floats (bf16/fp16); |key(a)-key(b)| = ULP distance, -0 == +0."""
    i = t.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def compare(actual, expected):
    ulp = (bf16_key(actual) - bf16_key(expected)).abs()
    return {
        "mismatches": int((ulp != 0).sum()),
        "max_ulp": int(ulp.max()),
        "max_abs": float((actual.float() - expected.float()).abs().max()),
        "nonfinite": int((~torch.isfinite(actual)).sum()),
    }


def references(x, r, w):
    h = x.double() + r.double()  # exact sum of two bf16 values
    residual = h.to(x.dtype)
    scale = w.double() + 1.0

    def norm(v):
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS) * scale).to(x.dtype)

    # Production contract: normalise the unrounded sum. forward_native contract:
    # normalise the residual after it has been rounded to bf16.
    return residual, norm(h), norm(residual.double())


def check(Ms):
    print(json.dumps(environment(), indent=2))
    cases = [("random", s) for s in (0, 1, 2)] + [
        (c, 3) for c in ("zeros", "exact_cancel", "near_cancel", "residual_dominant",
                         "large", "weight_minus_one")]
    ok = True
    print(f"{'M':>2} {'case':<18} {'resid mism':>10} | {'out vs fp32-h: mism  ulp':>25} "
          f"| {'out vs bf16-h: mism  ulp':>25} | in-place")
    for M in Ms:
        for case, seed in cases:
            x, r, w = inputs(M, H_QWEN36, seed, case)
            res_ref, y_ref, y_native = references(x, r, w)
            x_in, r_in = x.clone(), r.clone()
            ptrs = (x.data_ptr(), r.data_ptr())
            ret = run_norm(x, r, w)
            torch.cuda.synchronize()
            # In-place contract: no return value, same storage, both tensors rewritten.
            changed = torch.equal(x, x_in) is False or torch.equal(x_in, y_ref)
            inplace = ret is None and ptrs == (x.data_ptr(), r.data_ptr()) and changed
            cr, cy, cn = compare(r, res_ref), compare(x, y_ref), compare(x, y_native)
            # Predeclared: residual bit-exact; output within 1 bf16 ULP of the FP64
            # production-contract reference (FP32 reduction + approximate rsqrt).
            passed = (inplace and cr["mismatches"] == 0 and cy["max_ulp"] <= 1
                      and cr["nonfinite"] == cy["nonfinite"] == 0)
            ok &= passed
            print(f"{M:>2} {case:<18} {cr['mismatches']:>10} | {cy['mismatches']:>18} "
                  f"{cy['max_ulp']:>5} | {cn['mismatches']:>18} {cn['max_ulp']:>5} | "
                  f"{'yes' if inplace else 'NO'}{'' if passed else '  FAIL'}")
    print("ALL PASSED" if ok else "FAILURES PRESENT")
    return ok


# ---------------------------------------------------------------- bench
def capture(body, ops):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(ops):
            body()
    return g


def bench(Ms, trials, replays, ops):
    env = environment()
    print(json.dumps(env, indent=2))
    path = OUT / f"bench_{STAMP}.csv"
    fields = ["stamp", "gpu", "M", "H", "arm", "trial", "replays", "ops_per_replay",
              "total_ms", "us_per_op", "sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w"]
    with path.open("w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=fields)
        out.writeheader()
        for M in Ms:
            x0, r0, w = inputs(M, H_QWEN36, 0)
            x, r = x0.clone(), r0.clone()

            def reset():
                x.copy_(x0)
                r.copy_(r0)

            def full():
                reset()
                run_norm(x, r, w)

            for _ in range(10):  # compile + warm outside capture
                full()
            torch.cuda.synchronize()
            graphs = {
                "reset_plus_norm": capture(full, ops),
                "reset_only": capture(reset, ops),
                # Cross-check only: inputs drift across ops. Valid for timing because
                # the kernel has no data-dependent control flow.
                "norm_no_reset": capture(lambda: run_norm(x, r, w), ops),
            }
            # Prove a timed execution starts from the original state: one graph op
            # sequence must leave exactly the eager single-call result.
            xe, re_ = x0.clone(), r0.clone()
            run_norm(xe, re_, w)
            graphs["reset_plus_norm"].replay()
            torch.cuda.synchronize()
            assert torch.equal(x, xe) and torch.equal(r, re_), "graph result != eager"

            for g in graphs.values():
                for _ in range(20):
                    g.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            names = list(graphs)
            for t in range(trials):
                for arm in names[t % 3:] + names[: t % 3]:  # rotate arm order
                    g = graphs[arm]
                    start.record()
                    for _ in range(replays):
                        g.replay()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    out.writerow({"stamp": STAMP, "gpu": env["gpu"], "M": M, "H": H_QWEN36,
                                  "arm": arm, "trial": t, "replays": replays,
                                  "ops_per_replay": ops, "total_ms": ms,
                                  "us_per_op": ms * 1e3 / (replays * ops), **telemetry()})
    summarize(path)


def summarize(path):
    rows = list(csv.DictReader(path.open()))
    print(f"\nraw: {path}")
    print(f"{'M':>2} {'arm':<16} {'n':>3} {'median us':>9} {'p10':>7} {'p90':>7}")
    for M in sorted({int(r["M"]) for r in rows}):
        by = {}
        for r in rows:
            if int(r["M"]) == M:
                by.setdefault(r["arm"], {})[int(r["trial"])] = float(r["us_per_op"])
        for arm, v in by.items():
            q = statistics.quantiles(v.values(), n=10)
            print(f"{M:>2} {arm:<16} {len(v):>3} {statistics.median(v.values()):>9.3f} "
                  f"{q[0]:>7.3f} {q[-1]:>7.3f}")
        diff = [by["reset_plus_norm"][t] - by["reset_only"][t] for t in by["reset_only"]]
        q = statistics.quantiles(diff, n=10)
        print(f"{M:>2} {'norm = full-reset':<16} {len(diff):>3} {statistics.median(diff):>9.3f} "
              f"{q[0]:>7.3f} {q[-1]:>7.3f}   (paired per trial)")


# ---------------------------------------------------------------- A/B backends
def observed_backends(call):
    """Run `call` with spies on FlashInfer's two norm backends; return what ran."""
    import flashinfer.norm as fn

    # With FLASHINFER_USE_CUDA_NORM=1 the CuTe functions are never imported.
    seen, real_cute, real_jit = [], getattr(fn, "fused_add_rmsnorm_cute", None), fn.get_norm_module

    def spy_cute(*a, **k):
        seen.append(("cute_dsl", k.get("enable_pdl")))
        return real_cute(*a, **k)

    def spy_jit():
        seen.append(("cuda_jit", None))
        return real_jit()

    if real_cute is not None:
        fn.fused_add_rmsnorm_cute = spy_cute
    fn.get_norm_module = spy_jit
    try:
        call()
        torch.cuda.synchronize()
    finally:
        if real_cute is not None:
            fn.fused_add_rmsnorm_cute = real_cute
        fn.get_norm_module = real_jit
    return seen


def kernel_geometry(call):
    """CUPTI-reported kernels launched by `call`: name, grid, block, regs, smem, dur."""
    import os
    import tempfile

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=OUT)
    os.close(fd)
    try:
        prof.export_chrome_trace(tmp)
        events = json.loads(Path(tmp).read_text())["traceEvents"]
    finally:
        os.unlink(tmp)
    return [{"name": e["name"][:90], "ts": e.get("ts"), "grid": e["args"].get("grid"),
             "block": e["args"].get("block"), "regs": e["args"].get("registers per thread"),
             "smem": e["args"].get("shared memory"), "dur_us": e.get("dur")}
            for e in events if e.get("cat") == "kernel"]


def sample_plan(M):
    """(ops per graph, replays per sample, trials): keep each sample ~>=2 ms."""
    if M <= 64:
        return 100, 20, 20
    if M <= 576:
        return 50, 10, 15
    if M <= 2304:
        return 20, 5, 10
    return 5, 4, 8


def ab(Ms, round_id, csv_path):
    import hashlib
    import os

    requested = "cuda_jit" if os.environ.get("FLASHINFER_USE_CUDA_NORM") == "1" else "cute_dsl"
    env = environment()
    meta = {"requested": requested, "round": round_id, **env, "per_M": {}}
    outputs = {}
    fields = ["stamp", "round", "backend", "M", "H", "arm", "trial", "replays",
              "ops_per_replay", "total_ms", "us_per_op", "input_sha256", "sm_clock_mhz",
              "mem_clock_mhz", "temp_c", "power_w"]
    # Laptop GPUs idle at low clocks; ramp them before the first shape is timed.
    a_ = torch.randn(4096, 4096, device="cuda", dtype=BF16)
    t_end = torch.cuda.Event(True)
    start_warm = torch.cuda.Event(True)
    start_warm.record()
    while True:
        for _ in range(20):
            a_ @ a_
        t_end.record()
        t_end.synchronize()
        if start_warm.elapsed_time(t_end) > 2000:
            break
    del a_
    meta["clock_after_warmup"] = telemetry()
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=fields)
        if new:
            out.writeheader()
        for M in Ms:
            x_init, r_init, w = inputs(M, H_QWEN36, 0)
            # x and r are views into one buffer so a single kernel resets both.
            xr0 = torch.stack([x_init, r_init])
            xr = xr0.clone()
            x, r = xr[0], xr[1]
            sha = hashlib.sha256(xr0.view(torch.int16).cpu().numpy().tobytes()).hexdigest()

            def reset():
                torch.mul(xr0, 1, out=xr)  # one elementwise kernel, bit-exact copy

            def norm():
                run_norm(x, r, w)

            def full():
                reset()
                norm()

            # Evidence first: which backend ran, what was launched.
            seen = observed_backends(full)  # also compiles on first use
            assert seen and all(b == requested for b, _ in seen), (requested, seen)
            reset()
            torch.cuda.synchronize()
            geometry = kernel_geometry(norm)
            # Correctness from pristine inputs vs the FP64 production contract.
            reset()
            norm()
            torch.cuda.synchronize()
            res_ref, y_ref, _ = references(x_init, r_init, w)
            meta["per_M"][M] = {"backend_calls": seen, "kernels": geometry,
                                "residual": compare(r, res_ref), "output": compare(x, y_ref),
                                "input_sha256": sha}
            if M <= 576 and round_id == 0:
                outputs[M] = (x.cpu().clone(), r.cpu().clone())

            ops, replays, trials = sample_plan(M)
            for _ in range(5):
                full()
            torch.cuda.synchronize()
            graphs = {
                "reset_plus_norm": capture(full, ops),
                "reset_only": capture(reset, ops),
                # Dependent chain on drifting inputs: back-to-back throughput only.
                "chain_drifting": capture(norm, ops),
            }
            reset()
            torch.cuda.synchronize()
            xe, re_ = x.clone(), r.clone()
            run_norm(xe, re_, w)
            graphs["reset_plus_norm"].replay()
            torch.cuda.synchronize()
            assert torch.equal(x, xe) and torch.equal(r, re_), "graph result != eager"
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            # Clocks fall during the CPU-only compile/check/capture above: re-ramp
            # with ~300 ms of the measured graphs immediately before timing.
            start.record()
            while True:
                for g in graphs.values():
                    g.replay()
                end.record()
                end.synchronize()
                if start.elapsed_time(end) > 300:
                    break
            names = list(graphs)
            for t in range(trials):
                for arm in names[t % 3:] + names[: t % 3]:
                    start.record()
                    for _ in range(replays):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    out.writerow({"stamp": STAMP, "round": round_id, "backend": requested,
                                  "M": M, "H": H_QWEN36, "arm": arm, "trial": t,
                                  "replays": replays, "ops_per_replay": ops, "total_ms": ms,
                                  "us_per_op": ms * 1e3 / (replays * ops),
                                  "input_sha256": sha, **telemetry()})
            del graphs
            torch.cuda.empty_cache()
            k = geometry[0] if geometry else {}
            print(f"[{requested} r{round_id}] M={M:<5} kernels={len(geometry)} grid={k.get('grid')} "
                  f"block={k.get('block')} resid_mism={meta['per_M'][M]['residual']['mismatches']} "
                  f"out_max_ulp={meta['per_M'][M]['output']['max_ulp']}", flush=True)
    (OUT / f"ab_meta_{requested}_r{round_id}_{STAMP}.json").write_text(
        json.dumps(meta, indent=2, default=str))
    if outputs:
        torch.save(outputs, OUT / f"ab_outputs_{requested}.pt")


def abcompare(csv_path):
    """Cross-backend bits + one comparison table from the A/B CSV."""
    a, b = (torch.load(OUT / f"ab_outputs_{n}.pt") for n in ("cute_dsl", "cuda_jit"))
    print("Cross-backend output differences (same pristine inputs):")
    for M in sorted(a):
        cx, cr = compare(a[M][0], b[M][0]), compare(a[M][1], b[M][1])
        print(f"  M={M:<4} x: {cx['mismatches']}/{a[M][0].numel()} max_ulp={cx['max_ulp']} | "
              f"residual: {cr['mismatches']} mismatches")

    rows = list(csv.DictReader(csv_path.open()))
    samples = {}
    for row in rows:
        key = (row["backend"], int(row["M"]), int(row["round"]), int(row["trial"]))
        samples.setdefault(key, {})[row["arm"]] = float(row["us_per_op"])
    norm, chain = {}, {}
    for (backend, M, _, _), arms in samples.items():
        if len(arms) == 3:
            norm.setdefault((backend, M), []).append(arms["reset_plus_norm"] - arms["reset_only"])
            chain.setdefault((backend, M), []).append(arms["chain_drifting"])
    clocks = [int(r["sm_clock_mhz"]) for r in rows if r["sm_clock_mhz"]]
    power = [float(r["power_w"]) for r in rows if r["power_w"]]
    temps = [int(r["temp_c"]) for r in rows if r["temp_c"]]
    print(f"\nTelemetry over {len(rows)} samples: SM clock {min(clocks)}-{max(clocks)} MHz "
          f"(median {statistics.median(clocks)}), power median {statistics.median(power):.1f} W, "
          f"temp {min(temps)}-{max(temps)} C")

    def stat(v):
        q = statistics.quantiles(v, n=4)
        return statistics.median(v), q[0], q[2]

    bytes_per_row = 4 * H_QWEN36 * 2  # read x, r + write x, r (weight is L2-resident)
    print(f"\n{'M':>5} | {'CuTe norm us [IQR]':>22} | {'JIT norm us [IQR]':>22} | JIT/CuTe | "
          f"{'CuTe GB/s':>9} {'JIT GB/s':>9} | chain CuTe / JIT us | n")
    for M in sorted({m for _, m in norm}):
        c, j = stat(norm[("cute_dsl", M)]), stat(norm[("cuda_jit", M)])
        cc, cj = statistics.median(chain[("cute_dsl", M)]), statistics.median(chain[("cuda_jit", M)])
        gb = lambda us: bytes_per_row * M / (us * 1e3)
        print(f"{M:>5} | {c[0]:>8.3f} [{c[1]:.3f},{c[2]:.3f}] | {j[0]:>8.3f} [{j[1]:.3f},{j[2]:.3f}] | "
              f"{j[0] / c[0]:>8.2f} | {gb(c[0]):>9.1f} {gb(j[0]):>9.1f} | "
              f"{cc:>8.3f} / {cj:<8.3f} | {len(norm[('cute_dsl', M)])}")
    reset_big = [float(r["us_per_op"]) for r in rows
                 if r["arm"] == "reset_only" and int(r["M"]) == max(m for _, m in norm)]
    Mb = max(m for _, m in norm)
    print(f"\nStreaming reference: reset kernel (copy of x,r) at M={Mb}: "
          f"{bytes_per_row * Mb / (statistics.median(reset_big) * 1e3):.1f} GB/s")


# ---------------------------------------------------------------- profile
def profile_region(M, calls):
    x0, r0, w = inputs(M, H_QWEN36, 0)
    x, r = x0.clone(), r0.clone()
    for _ in range(10):
        x.copy_(x0), r.copy_(r0), run_norm(x, r, w)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for i in range(calls):
        torch.cuda.nvtx.range_push(f"norm M={M} call={i}")
        x.copy_(x0), r.copy_(r0), run_norm(x, r, w)
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("probe", "check", "bench", "profile", "ab", "abcompare"))
    p.add_argument("--round", type=int, default=0, help="ab mode: process round id")
    p.add_argument("--csv", type=Path, default=None, help="ab/abcompare: shared CSV")
    p.add_argument("--M", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--replays", type=int, default=20)
    p.add_argument("--ops", type=int, default=100, help="norm ops captured per graph")
    p.add_argument("--calls", type=int, default=3, help="profile mode: calls per M")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    torch.cuda.init()
    if a.mode == "probe":
        probe()
    elif a.mode == "check":
        raise SystemExit(0 if check(a.M) else 1)
    elif a.mode == "bench":
        bench(a.M, a.trials, a.replays, a.ops)
    elif a.mode == "ab":
        ab(a.M, a.round, a.csv)
    elif a.mode == "abcompare":
        abcompare(a.csv)
    else:
        for M in a.M:
            profile_region(M, a.calls)
