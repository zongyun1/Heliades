"""B4 throughput benchmark: Helios tomato (476 bodies, 497 leaflets, 27 levels) on the GPU
(heliades.sim.gpu.TreeSimGPU) versus the CPU numpy solver (heliades.sim.TreeSim).

    python scripts/gpu_benchmark.py [--quick] [--json out.json]

Reports env-steps/s and ms/step at B = 1, 64, 256, 1024, 4096 (and larger while memory
allows), float32 and float64, with and without CUDA graphs at B = 1, device memory per env,
and the batched inverse-statics + Sturm-count screening rate.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

from heliades.sim import TreeSim
from heliades.sim.gpu import TreeSimGPU, inverse_statics_gpu


def bench_cpu(asset, B, n):
    s = TreeSim(asset, [asset.physics.copy(stem_youngs_modulus=1e9)] * B)
    s.step(1 / 60)
    t = time.perf_counter()
    for _ in range(n):
        s.step(1 / 60)
    dt = (time.perf_counter() - t) / n
    return {"B": B, "ms_per_step": dt * 1e3, "env_steps_per_s": B / dt}


def bench_gpu(asset, B, precision, n, use_graph=True):
    s = TreeSimGPU(asset, [asset.physics.copy(stem_youngs_modulus=1e9)] * B, precision=precision,
                   use_graph=use_graph)
    for _ in range(3):
        s.step(1 / 60)
    wp.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        s.step(1 / 60)
    wp.synchronize()
    dt = (time.perf_counter() - t) / n
    mem = s.memory_bytes()
    x = s.state.x
    ok = bool(np.isfinite(x).all())
    del s
    return {"B": B, "precision": precision, "graph": use_graph, "ms_per_step": dt * 1e3,
            "env_steps_per_s": B / dt, "mem_per_env_kB": mem / B / 1e3, "mem_total_MB": mem / 1e6, "finite": ok}


def bench_inverse(asset, B, precision):
    Es = np.geomspace(3e8, 3e9, B)
    s = TreeSimGPU(asset, [asset.physics.copy(stem_youngs_modulus=E) for E in Es], precision=precision)
    inverse_statics_gpu(s, full=False)
    t = time.perf_counter()
    r = inverse_statics_gpu(s, full=False)
    dt = time.perf_counter() - t
    return {"B": B, "precision": precision, "ms": dt * 1e3, "us_per_env": dt / B * 1e6,
            "stable_fraction": float(r.stable.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default="", help="optional path for a JSON dump")
    ap.add_argument("--max-b", type=int, default=16384)
    args = ap.parse_args()
    wp.config.quiet = True
    wp.init()
    from planttwin.helios_import import build_helios_plant
    asset, _ = build_helios_plant("tomato", 75, seed=1)
    dev = wp.get_device("cuda:0")
    out = {"device": dev.name, "arch": dev.arch, "warp": wp.__version__, "cpu": [], "gpu": [], "inverse": []}

    print("CPU numpy (float64)")
    for B in ((1, 16) if args.quick else (1, 16, 64)):
        r = bench_cpu(asset, B, 20 if B < 64 else 5)
        out["cpu"].append(r)
        print(f"  B={B:5d}  {r['ms_per_step']:8.2f} ms/step  {r['env_steps_per_s']:10.0f} env-steps/s")
    cpu_ref = max(r["env_steps_per_s"] for r in out["cpu"])

    Bs = [1, 64, 256, 1024, 4096] + ([] if args.quick else [16384])
    Bs = [b for b in Bs if b <= args.max_b]
    for prec in ("float32", "float64"):
        print(f"GPU {prec}")
        for B in Bs:
            free = dev.free_memory if hasattr(dev, "free_memory") else None
            est = B * (400e3 if prec == "float32" else 800e3)
            if free is not None and est > 0.8 * free:
                print(f"  B={B:5d}  skipped (needs ~{est / 1e9:.1f} GB)")
                continue
            n = 200 if B <= 256 else (50 if B <= 4096 else 20)
            r = bench_gpu(asset, B, prec, n)
            out["gpu"].append(r)
            print(f"  B={B:5d}  {r['ms_per_step']:8.3f} ms/step  {r['env_steps_per_s']:10.0f} env-steps/s "
                  f"({r['env_steps_per_s'] / cpu_ref:6.0f}x CPU)  {r['mem_per_env_kB']:6.0f} kB/env  "
                  f"{r['mem_total_MB']:8.0f} MB")
        r = bench_gpu(asset, 1, prec, 100, use_graph=False)
        out["gpu"].append(r)
        print(f"  B=    1  no graph: {r['ms_per_step']:8.3f} ms/step")

    print("GPU inverse statics + Sturm count (summary readback)")
    for prec in ("float32", "float64"):
        for B in (1, 64, 1024, 4096):
            r = bench_inverse(asset, B, prec)
            out["inverse"].append(r)
            print(f"  {prec} B={B:5d}  {r['ms']:8.2f} ms  {r['us_per_env']:9.1f} us/env")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
