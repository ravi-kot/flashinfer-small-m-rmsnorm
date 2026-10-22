"""One-row-per-CTA CuTe fused-add Gemma RMSNorm vs installed / local copy / CUDA JIT.

Arms (all in one process, interleaved per trial):
  installed  sgl_kernel.gemma_fused_add_rmsnorm -> FlashInfer CuTe (production path)
  copy       local_cute/fused_add_rmsnorm_copy.py (imports-only copy of the installed file)
  1row       local_cute/fused_add_rmsnorm_1row.py (num_threads = threads_per_row)
  jit        flashinfer.norm.get_norm_module() CUDA JIT kernel; built with nvcc 13.4
             against CUDA 13.0 headers via -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK

Modes: check | bench | profile | summary
"""

import argparse
import csv
import random
import statistics

import torch

from norm_lab import (BF16, EPS, H_QWEN36, OUT, STAMP, capture, compare, environment,
                      inputs, kernel_geometry, references, run_norm, sample_plan, telemetry)

ARMS = ("installed", "copy", "1row", "jit")
H, DTYPE, TAG = H_QWEN36, BF16, ""  # overridden by --H/--dtype/--tag


def make_arms():
    import flashinfer.norm as fn

    from local_cute import fused_add_rmsnorm_1row as one
    from local_cute import fused_add_rmsnorm_copy as cp

    assert not fn._use_cuda_norm(torch.device("cuda")), "installed arm must be CuTe"
    jit = fn.get_norm_module()
    # Production passes enable_pdl=None -> device_support_pdl() -> True on SM120.
    return {
        "installed": lambda x, r, w: run_norm(x, r, w),
        "copy": lambda x, r, w: cp.fused_add_rmsnorm_cute(x, r, w, EPS, 1.0, True),
        "1row": lambda x, r, w: one.fused_add_rmsnorm_cute(x, r, w, EPS, 1.0, True),
        "jit": lambda x, r, w: jit.gemma_fused_add_rmsnorm(x, r, w, EPS, True),
    }


def case_inputs(M, case):
    """Cases from norm_lab.inputs plus mapping-specific and extreme cases."""
    base = {"random0": ("random", 0), "random1": ("random", 1), "random2": ("random", 2)}
    if case in base:
        return inputs(M, H, base[case][1], base[case][0], dtype=DTYPE)
    if case in ("zeros", "exact_cancel", "near_cancel", "residual_dominant", "large",
                "weight_minus_one"):
        return inputs(M, H, 3, case, dtype=DTYPE)
    x, r, w = (t.float() for t in inputs(M, H, 4, dtype=DTYPE))
    if case == "wide_weights":  # nonuniform, large-magnitude Gemma weights
        w = 4 * w / 0.1
    elif case == "row_scaled":  # each row a different magnitude: exposes row mixing
        s = 10.0 ** (torch.arange(M, device="cuda") % 9 - 4).float()[:, None]
        x, r = x * s, r * s
    elif case == "huge":  # sum of squares overflows FP32 -> rstd = 0
        x, r = x * 1e19, r * 1e19
    elif case == "tiny":  # bf16/FP32 subnormal range
        x, r = x * 1e-39, r * 1e-39
    return x.to(DTYPE), r.to(DTYPE), w.to(DTYPE)


CASES = ("random0", "random1", "random2", "zeros", "exact_cancel", "near_cancel",
         "residual_dominant", "large", "weight_minus_one", "wide_weights", "row_scaled",
         "huge", "tiny")


def bits_equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def guarded(t):
    """Copy `t` into rows 1..M of an [M+2, H] buffer whose first/last rows are sentinels."""
    buf = torch.full((t.shape[0] + 2, t.shape[1]), 7.0, device="cuda", dtype=t.dtype)
    buf[1:-1] = t
    return buf


def check(Ms):
    arms = make_arms()
    ok = True
    print(f"{'M':>3} {'case':<18} 1row==copy==installed | sentinels | 1row vs FP64 max ulp "
          f"| jit vs copy mism (x/r)")
    for M in Ms:
        for case in CASES:
            x0, r0, w = case_inputs(M, case)
            res = {}
            for name, fn in arms.items():
                xb, rb = guarded(x0), guarded(r0)
                ret = fn(xb[1:-1], rb[1:-1], w)
                torch.cuda.synchronize()
                sent = all(bits_equal(b[i], torch.full_like(b[i], 7.0))
                           for b in (xb, rb) for i in (0, -1))
                res[name] = (xb[1:-1].clone(), rb[1:-1].clone(), ret is None, sent)
            same = all(bits_equal(res["1row"][k], res[a][k]) for a in ("copy", "installed")
                       for k in (0, 1))
            sentinels = all(res[a][3] and res[a][2] for a in ("installed", "copy", "1row"))
            _, y_ref, _ = references(x0, r0, w)
            ulp = compare(res["1row"][0], y_ref)["max_ulp"] if case not in ("huge",) else "-"
            jx = compare(res["jit"][0], res["copy"][0])["mismatches"]
            jr = compare(res["jit"][1], res["copy"][1])["mismatches"]
            ok &= same and sentinels
            print(f"{M:>3} {case:<18} {'yes' if same else 'NO  <-- DIAGNOSE':<21} | "
                  f"{'ok' if sentinels else 'VIOLATED'} | {ulp!s:>6} | {jx}/{jr}")
    print("ALL BIT-IDENTICAL" if ok else "DISCREPANCY FOUND")
    return ok


def geometry_report(Ms):
    arms = make_arms()
    for M in Ms:
        x0, r0, w = case_inputs(M, "random0")
        for name, fn in arms.items():
            x, r = x0.clone(), r0.clone()
            fn(x, r, w)  # compile outside the profiled call
            x, r = x0.clone(), r0.clone()
            for k in kernel_geometry(lambda: fn(x, r, w)):
                print(f"M={M:<3} {name:<9} grid={k['grid']} block={k['block']} regs={k['regs']} "
                      f"smem={k['smem']} {k['name'][:60]}")


def bench(Ms, round_id):
    arms = make_arms()
    env = environment()
    path = OUT / f"rowmap_{TAG}{STAMP}_r{round_id}.csv"
    fields = ["stamp", "round", "M", "graph", "trial", "replays", "ops_per_replay", "us_per_op",
              "sm_clock_mhz", "mem_clock_mhz", "temp_c", "power_w"]
    kpath = OUT / f"rowmap_kernels_{TAG}{STAMP}_r{round_id}.csv"
    with path.open("w", newline="") as fh, kpath.open("w", newline="") as kh:
        out = csv.DictWriter(fh, fieldnames=fields)
        out.writeheader()
        kout = csv.writer(kh)
        kout.writerow(["stamp", "round", "M", "arm", "kernel", "dur_us"])
        for M in Ms:
            x_init, r_init, w = inputs(M, H, 0, dtype=DTYPE)
            xr0 = torch.stack([x_init, r_init])
            xr = xr0.clone()
            x, r = xr[0], xr[1]
            wc = (torch.randn(H, 512, device="cuda") / 45).to(DTYPE)
            yc = torch.empty(M, 512, device="cuda", dtype=DTYPE)

            def reset():  # also the chain's producer: writes the norm's inputs
                torch.mul(xr0, 1, out=xr)

            def consumer():  # reads the norm's output x (RAW dependency)
                torch.mm(x, wc, out=yc)

            bodies = {"reset_only": reset, "chain_baseline": lambda: (reset(), consumer())}
            for name, fn in arms.items():
                bodies[f"iso_{name}"] = lambda fn=fn: (reset(), fn(x, r, w))
                bodies[f"chain_{name}"] = lambda fn=fn: (reset(), fn(x, r, w), consumer())
            for body in bodies.values():
                for _ in range(3):
                    body()
            torch.cuda.synchronize()
            ops, replays, trials = sample_plan(M)
            graphs = {k: capture(b, ops) for k, b in bodies.items()}

            # Each isolated graph must reproduce one eager call from pristine inputs.
            for name, fn in arms.items():
                xe, re_ = x_init.clone(), r_init.clone()
                fn(xe, re_, w)
                graphs[f"iso_{name}"].replay()
                torch.cuda.synchronize()
                assert torch.equal(x, xe) and torch.equal(r, re_), f"{name} graph != eager"

            # GPU-side kernel durations inside graph replay (PDL early start inflates these).
            for name in ARMS:
                for k in kernel_geometry(lambda: [graphs[f"iso_{name}"].replay()
                                                  for _ in range(2)]):
                    if "RMSNorm" in k["name"]:
                        kout.writerow([STAMP, round_id, M, name, k["name"][:60], k["dur_us"]])

            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            while True:  # re-ramp clocks right before timing
                for g in graphs.values():
                    g.replay()
                end.record()
                end.synchronize()
                if start.elapsed_time(end) > 300:
                    break
            names = list(graphs)
            for t in range(trials):
                k = t % len(names)
                for gname in names[k:] + names[:k]:
                    start.record()
                    for _ in range(replays):
                        graphs[gname].replay()
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    out.writerow({"stamp": STAMP, "round": round_id, "M": M, "graph": gname,
                                  "trial": t, "replays": replays, "ops_per_replay": ops,
                                  "us_per_op": ms * 1e3 / (replays * ops), **telemetry()})
            del graphs
            torch.cuda.empty_cache()
            print(f"r{round_id} M={M} done", flush=True)
    print(path, kpath)


def boot_ci(v, n=2000):
    rng = random.Random(0)
    meds = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def summary(paths, kpaths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    krows = [r for p in kpaths for r in csv.DictReader(open(p))]
    s = {}
    for r in rows:
        s.setdefault((int(r["M"]), r["round"], int(r["trial"])), {})[r["graph"]] = float(
            r["us_per_op"])
    clocks = [int(r["sm_clock_mhz"]) for r in rows]
    print(f"{len(rows)} samples; SM clock min/median/max {min(clocks)}/"
          f"{statistics.median(clocks)}/{max(clocks)} MHz; "
          f"{sum(c < 2000 for c in clocks)} samples < 2 GHz; temp "
          f"{min(int(r['temp_c']) for r in rows)}-{max(int(r['temp_c']) for r in rows)} C; "
          f"power median {statistics.median(float(r['power_w']) for r in rows):.1f} W")
    print(f"\n{'M':>4} | isolated = (reset+op)-reset, median us: "
          f"{'inst':>6} {'copy':>6} {'1row':>6} {'jit':>6} | 1row-copy [95% CI] | "
          f"chain 1row-copy [95% CI] | 1row-jit iso | kernel dur us inst/copy/1row/jit | n")
    for M in sorted({k[0] for k in s}):
        tr = [v for k, v in s.items() if k[0] == M and len(v) == 10]
        iso = {a: statistics.median(t[f"iso_{a}"] - t["reset_only"] for t in tr) for a in ARMS}
        d_iso = [t["iso_1row"] - t["iso_copy"] for t in tr]
        d_chain = [t["chain_1row"] - t["chain_copy"] for t in tr]
        d_jit = [t["iso_1row"] - t["iso_jit"] for t in tr]
        kd = {a: statistics.median(float(r["dur_us"]) for r in krows
                                   if int(r["M"]) == M and r["arm"] == a) for a in ARMS}
        ci, cc = boot_ci(d_iso), boot_ci(d_chain)
        print(f"{M:>4} | {'':>39}{iso['installed']:>6.3f} {iso['copy']:>6.3f} "
              f"{iso['1row']:>6.3f} {iso['jit']:>6.3f} | {statistics.median(d_iso):+.3f} "
              f"[{ci[0]:+.3f},{ci[1]:+.3f}] | {statistics.median(d_chain):+.3f} "
              f"[{cc[0]:+.3f},{cc[1]:+.3f}] | {statistics.median(d_jit):+.3f} | "
              f"{kd['installed']:.2f}/{kd['copy']:.2f}/{kd['1row']:.2f}/{kd['jit']:.2f} | "
              f"{len(tr)}")
    print("\nchain totals (producer+norm+consumer) median us/op:")
    for M in sorted({k[0] for k in s}):
        tr = [v for k, v in s.items() if k[0] == M and len(v) == 10]
        print(f"  M={M:<5} baseline(no norm) {statistics.median(t['chain_baseline'] for t in tr):.3f} "
              + " ".join(f"{a} {statistics.median(t[f'chain_{a}'] for t in tr):.3f}" for a in ARMS))
    # installed vs copy: A/A check
    print("\nA/A (copy - installed), isolated, median [95% CI]:")
    for M in sorted({k[0] for k in s}):
        tr = [v for k, v in s.items() if k[0] == M and len(v) == 10]
        d = [t["iso_copy"] - t["iso_installed"] for t in tr]
        lo, hi = boot_ci(d)
        print(f"  M={M:<5} {statistics.median(d):+.3f} [{lo:+.3f},{hi:+.3f}]")


def exposed_events(call):
    """Norm kernels from a CUPTI trace of `call`: duration and exposed time.

    exposed = end(norm) - end(previous kernel on the stream): the time the norm adds to
    the critical path. Unlike duration, it excludes PDL early-launch waiting.
    """
    ev = sorted(kernel_geometry(call), key=lambda k: k["ts"])
    out = []
    for prev, k in zip(ev, ev[1:]):
        if "RMSNorm" in k["name"]:
            out.append((k["dur_us"], k["ts"] + k["dur_us"] - (prev["ts"] + prev["dur_us"])))
    return out


def diag(Ms, round_id):
    """Small-M regression diagnosis: arm x PDL x cache x graph/eager, CUPTI timings."""
    from local_cute import fused_add_rmsnorm_1row as one
    from local_cute import fused_add_rmsnorm_copy as cp

    environment()  # initialises NVML for clock telemetry
    kern = {"copy": cp.fused_add_rmsnorm_cute, "1row": one.fused_add_rmsnorm_cute}
    flush = torch.empty(32 << 20, device="cuda", dtype=BF16)  # 64 MiB > 32 MiB L2
    path = OUT / f"diag_{TAG}{STAMP}_r{round_id}.csv"
    a_ = torch.randn(4096, 4096, device="cuda", dtype=BF16)
    for _ in range(300):
        a_ @ a_  # ramp clocks
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["stamp", "round", "M", "arm", "pdl", "cache", "mode", "n",
                      "dur_med_us", "exposed_med_us", "exposed_p10_us", "exposed_p90_us",
                      "sm_clock_mhz"])
        for M in Ms:
            x0, r0, w = inputs(M, H, 0, dtype=DTYPE)
            xr0 = torch.stack([x0, r0])
            xr = xr0.clone()
            x, r = xr[0], xr[1]
            for arm in ("copy", "1row"):
                for pdl in (True, False):
                    for cache in ("warm", "cold"):
                        def body(arm=arm, pdl=pdl, cache=cache):
                            torch.mul(xr0, 1, out=xr)
                            if cache == "cold":  # evict x, r and w from L2
                                flush.add_(1)
                            kern[arm](x, r, w, EPS, 1.0, pdl)
                        for _ in range(3):
                            body()
                        torch.cuda.synchronize()
                        g = capture(body, 40)
                        for _ in range(5):
                            g.replay()
                        runs = {"graph": lambda: [g.replay() for _ in range(5)],
                                "eager": lambda: [body() for _ in range(200)]}
                        for mode, call in runs.items():
                            e = exposed_events(call)
                            d = sorted(x_[0] for x_ in e)
                            xp = sorted(x_[1] for x_ in e)
                            q = statistics.quantiles(xp, n=10)
                            out.writerow([STAMP, round_id, M, arm, pdl, cache, mode, len(e),
                                          statistics.median(d), statistics.median(xp), q[0],
                                          q[-1], _sm_clock()])
                        del g
            print(f"diag r{round_id} M={M} done", flush=True)
    print(path)


def _sm_clock():
    return telemetry()["sm_clock_mhz"]


def diag_summary(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    key = lambda r: (int(r["M"]), r["mode"], r["cache"], r["pdl"])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), {}).setdefault(r["arm"], []).append(r)
    print(f"{'M':>2} {'mode':<5} {'cache':<5} {'PDL':<5} | exposed us copy -> 1row (delta) | "
          f"duration us copy -> 1row | rounds")
    for k in sorted(groups):
        g = groups[k]
        ex = {a: statistics.median(float(r["exposed_med_us"]) for r in g[a]) for a in g}
        du = {a: statistics.median(float(r["dur_med_us"]) for r in g[a]) for a in g}
        print(f"{k[0]:>2} {k[1]:<5} {k[2]:<5} {k[3]:<5} | {ex['copy']:6.3f} -> {ex['1row']:6.3f} "
              f"({ex['1row'] - ex['copy']:+.3f}) | {du['copy']:6.3f} -> {du['1row']:6.3f} | "
              f"{len(g['copy'])}")


def profile(Ms):
    arms = make_arms()
    data = {M: case_inputs(M, "random0") for M in Ms}
    for M, (x0, r0, w) in data.items():
        for fn in arms.values():
            fn(x0.clone(), r0.clone(), w)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for M, (x0, r0, w) in data.items():
        for fn in arms.values():  # order per M: installed, copy, 1row, jit
            x, r = x0.clone(), r0.clone()
            fn(x, r, w)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("check", "geometry", "bench", "profile", "summary", "diag", "diagsummary"))
    p.add_argument("--H", type=int, default=H_QWEN36)
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--tag", default="")
    p.add_argument("--M", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 16, 36])
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--csv", nargs="*", default=[])
    p.add_argument("--kcsv", nargs="*", default=[])
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    H, DTYPE = a.H, getattr(torch, a.dtype)
    TAG = a.tag + ("_" if a.tag else "") + ("" if (H, a.dtype) == (H_QWEN36, "bfloat16") else f"H{H}_{a.dtype}_")
    if a.mode == "check":
        raise SystemExit(0 if check(a.M) else 1)
    if a.mode == "geometry":
        geometry_report(a.M)
    elif a.mode == "bench":
        bench(a.M, a.round)
    elif a.mode == "profile":
        profile(a.M)
    elif a.mode == "diag":
        diag(a.M, a.round)
    elif a.mode == "diagsummary":
        diag_summary(a.csv)
    else:
        summary(a.csv, a.kcsv)
