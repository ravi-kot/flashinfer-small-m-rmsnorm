"""Qwen3.5-2B decode on SGLang 0.5.21 (CUDA graphs): installed CuTe fused-add RMSNorm vs 1-row.

One process = one kernel arm (the swap must precede CUDA-graph capture). Per batch size:
synthetic prompt (fixed seed) -> prefill -> 30 untimed decode steps -> timed decode steps
(CUDA events + synchronized wall clock, NVML clocks) -> CUPTI trace of 3 decode steps to
confirm which norm kernel is inside the replayed graph and how long it runs.

  python model_bench.py --kernel {installed,1row} --round N  [SGLang server args...]
  python model_bench.py --summary
"""

import argparse
import csv
import glob
import json
import random
import statistics
import sys
import time

import numpy as np
import torch

from norm_lab import OUT, STAMP, kernel_geometry, telemetry

FUSED_ADD = ("flashinfernormkernelsfused_add_rmsnorm", "local_cutefused_add_rmsnorm_1row")


def norm_kernel_kind(name):
    if FUSED_ADD[0] in name:
        return "fused_add_installed"
    if FUSED_ADD[1] in name:
        return "fused_add_1row"
    return None


def run(args, rest):
    if args.kernel == "1row":
        import onerow_patch

        onerow_patch.install()
    from sglang.benchmark import one_batch as ob
    from sglang.srt.server_args import ServerArgs

    def bench_body(server_args, port_args, bench_args, gpu_id, tp_rank):
        # Same initialisation as one_batch.latency_test, then our own measurement loop.
        ob.publish(server_args, role="scheduler", ranks=ob.SpawnRanks(
            world_rank=ob.spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
            gpu_id=gpu_id))
        ob.initialize_moe_config()
        ob.initialize_fp8_gemm_config()
        ob.initialize_fp4_gemm_config()
        ob.configure_logger(server_args, prefix=" TP0")
        runner, _ = ob.load_model(server_args, port_args, gpu_id, tp_rank)
        import pynvml

        pynvml.nvmlInit()  # telemetry() assumes an initialised NVML handle

        tag = f"{args.kernel}_r{args.round}_{STAMP}"
        prefix = args.set
        rows, meta, tokens = [], {"kernel": args.kernel, "round": args.round, "per_B": {}}, {}
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        for B in args.batch_sizes:
            np.random.seed(1000 + B)
            reqs = ob.prepare_synthetic_inputs_for_latency_test(B, args.input_len)
            ids, _, batch = runner.extend(reqs)
            seq = [ids.cpu()]
            for i in range(30 + args.steps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                e0.record()
                ids, logits = runner.decode(ids, batch)
                e1.record()
                e1.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1e3
                gpu_ms = e0.elapsed_time(e1)
                seq.append(ids.cpu())
                if i >= 30:
                    rows.append({"kernel": args.kernel, "round": args.round, "B": B,
                                 "step": i - 30, "wall_ms": wall_ms, "gpu_ms": gpu_ms,
                                 **telemetry()})
            tokens[B] = torch.stack(seq)
            tokens[f"logits_{B}"] = logits.float().cpu()

            state = {"ids": ids}

            def three_steps():
                for _ in range(3):
                    state["ids"], _ = runner.decode(state["ids"], batch)

            kernels = sorted(kernel_geometry(three_steps), key=lambda k: k["ts"])
            norm = [k for k in kernels if norm_kernel_kind(k["name"])]
            # Exposed time: end(norm) - end(previous kernel); excludes PDL early-launch wait.
            exposed = [k["ts"] + k["dur_us"] - (p_["ts"] + p_["dur_us"])
                       for p_, k in zip(kernels, kernels[1:]) if norm_kernel_kind(k["name"])]
            meta["per_B"][B] = {
                "kernels_per_step": len(kernels) / 3,
                "fused_add_norms_per_step": len(norm) / 3,
                "fused_add_kinds": sorted({norm_kernel_kind(k["name"]) for k in norm}),
                "fused_add_grid_block": sorted({(str(k["grid"]), str(k["block"])) for k in norm}),
                "fused_add_us_per_step": sum(k["dur_us"] for k in norm) / 3,
                "fused_add_exposed_us_per_step": sum(exposed) / 3,
                "fused_add_median_dur_us": statistics.median(k["dur_us"] for k in norm) if norm else None,
                "all_kernels_us_per_step": sum(k["dur_us"] for k in kernels) / 3,
            }
            runner.cleanup(batch)
            print(f"[{args.kernel} r{args.round}] B={B} median gpu_ms="
                  f"{statistics.median(r['gpu_ms'] for r in rows if r['B'] == B):.3f} "
                  f"{meta['per_B'][B]}", flush=True)
        if args.kernel == "1row":
            import onerow_patch

            meta["patch_calls_during_capture_and_run"] = dict(onerow_patch.calls)
            meta["dispatch_range"] = [onerow_patch.MIN_M, onerow_patch.MAX_M]
        with (OUT / f"{prefix}_{tag}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (OUT / f"{prefix}_meta_{tag}.json").write_text(json.dumps(meta, indent=2, default=str))
        torch.save(tokens, OUT / f"{prefix}_tokens_{tag}.pt")

    ob.latency_test = bench_body  # main() resolves this global at call time
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    ob.BenchArgs.add_cli_args(parser)
    cli = parser.parse_args(rest + ["--batch-size", *map(str, args.batch_sizes),
                                    "--input-len", str(args.input_len),
                                    "--output-len", str(args.steps + 40)])
    ob.main(ServerArgs.from_cli_args(cli), ob.BenchArgs.from_cli_args(cli))


def boot_ci(v, n=2000):
    rng = random.Random(0)
    m = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def summary(args):
    pre = OUT / args.set
    rows = [r for p in sorted(glob.glob(f"{pre}_*_r*_*.csv")) for r in csv.DictReader(open(p))]
    metas = [json.load(open(p)) for p in sorted(glob.glob(f"{pre}_meta_*.json"))]
    toks = {p: torch.load(p) for p in sorted(glob.glob(f"{pre}_tokens_*.pt"))}
    clocks = [int(r["sm_clock_mhz"]) for r in rows]
    print(f"{len(rows)} timed decode steps; SM clock min/median/max {min(clocks)}/"
          f"{statistics.median(clocks)}/{max(clocks)} MHz; <2 GHz: "
          f"{sum(c < 2000 for c in clocks)}; temp {min(int(r['temp_c']) for r in rows)}-"
          f"{max(int(r['temp_c']) for r in rows)} C")

    # Greedy outputs: every run must agree token-for-token and logit-bit-for-bit.
    ref_path, ref = next(iter(toks.items()))
    for p, t in toks.items():
        same_tok = all(torch.equal(t[k], ref[k]) for k in ref if not str(k).startswith("logits"))
        same_log = all(torch.equal(t[k], ref[k]) for k in ref if str(k).startswith("logits"))
        print(f"  {p.split('/')[-1]}: tokens identical={same_tok} last logits identical={same_log}")

    Bs = sorted({int(r["B"]) for r in rows})
    print(f"\n{'B':>3} | step GPU ms (median of round medians) inst / 1row | 1row-inst us "
          f"[95% CI, step-level] | per-round deltas us | fused-add norm GPU us/step inst / 1row "
          f"(CUPTI) | norms/step | 1row kernel ran")
    for B in Bs:
        med = {}
        for k in ("installed", "1row"):
            med[k] = {rd: statistics.median(float(r["gpu_ms"]) for r in rows if int(r["B"]) == B
                                            and r["kernel"] == k and r["round"] == rd)
                      for rd in sorted({r["round"] for r in rows if r["kernel"] == k})}
        per_round = [(med["1row"][rd] - med["installed"][rd]) * 1e3
                     for rd in med["installed"] if rd in med["1row"]]
        inst = [float(r["gpu_ms"]) for r in rows if int(r["B"]) == B and r["kernel"] == "installed"]
        one = [float(r["gpu_ms"]) for r in rows if int(r["B"]) == B and r["kernel"] == "1row"]
        d = (statistics.median(one) - statistics.median(inst)) * 1e3
        rng = random.Random(B)
        boots = sorted((statistics.median(rng.choices(one, k=len(one)))
                        - statistics.median(rng.choices(inst, k=len(inst)))) * 1e3 for _ in range(1000))
        nk = {k: [m["per_B"][str(B)] for m in metas if m["kernel"] == k] for k in ("installed", "1row")}
        nus = {k: statistics.median(x["fused_add_us_per_step"] for x in v) for k, v in nk.items()}
        nex = {k: statistics.median(x.get("fused_add_exposed_us_per_step", float("nan")) for x in v)
               for k, v in nk.items()}
        print(f"{B:>3} | {statistics.median(med['installed'].values()):8.3f} / "
              f"{statistics.median(med['1row'].values()):8.3f} | {d:+7.1f} [{boots[25]:+.1f},"
              f"{boots[974]:+.1f}] | {' '.join(f'{x:+.1f}' for x in per_round)} | "
              f"{nus['installed']:6.1f} / {nus['1row']:6.1f} (saves {nus['installed'] - nus['1row']:.1f}) | "
              f"exposed {nex['installed']:6.1f} / {nex['1row']:6.1f} (saves {nex['installed'] - nex['1row']:.1f}) | "
              f"{nk['1row'][0]['fused_add_norms_per_step']:.0f} | {nk['1row'][0]['fused_add_kinds']} "
              f"{nk['1row'][0]['fused_add_grid_block']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--kernel", choices=("installed", "1row"))
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--set", default="model", help="result-file prefix for a run set")
    args, rest = p.parse_known_args()
    OUT.mkdir(exist_ok=True)
    if args.summary:
        summary(args)
    else:
        run(args, rest)
