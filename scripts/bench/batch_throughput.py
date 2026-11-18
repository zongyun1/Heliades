# SPDX-License-Identifier: LGPL-2.1-or-later
"""Plants per hour: the Heliades batch builder (fast mode, N worker processes) against Helios in one process.

    python scripts/bench/batch_throughput.py tomato 120 --plants 32 --workers 16
    python scripts/bench/batch_throughput.py tomato 120 --plants 32 --workers 16 --helios 4

Both build seeds 1..plants. ``--helios K`` times K Helios builds (PyHelios, one Context per plant, sequential, in
this process) and extrapolates to plants per hour; 0 skips Helios.
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("age", type=float)
    ap.add_argument("--plants", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mode", default="fast")
    ap.add_argument("--helios", type=int, default=4)
    ap.add_argument("--out", help="save every snapshot (.npz) here")
    a = ap.parse_args()
    from heliades.plantarchitecture.batch import PlantJob, build_plants
    jobs = [PlantJob(a.model, a.age, seed=s, mode=a.mode) for s in range(1, a.plants + 1)]
    t0 = time.perf_counter()
    res = build_plants(jobs, workers=a.workers, out_dir=a.out)
    wall = time.perf_counter() - t0
    per = sum(r.seconds for r in res) / len(res)
    print(f"heliades[{a.mode}] x{a.workers}: {len(res)} plants in {wall:.1f}s wall ({per:.2f}s per build in a worker)"
          f" -> {3600 * len(res) / wall:.0f} plants/h")
    if a.helios > 0:
        from heliades.parity.backends import HeliosBackend
        from pyhelios.types import vec3
        ts = []
        for s in range(1, a.helios + 1):
            B = HeliosBackend()
            B.seed(s)
            B.plantarch_new()
            B.plantarch_load(a.model)
            t1 = time.perf_counter()
            B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(a.age))
            ts.append(time.perf_counter() - t1)
            del B
        h = sum(ts) / len(ts)
        print(f"helios x1: {h:.2f}s per build -> {3600 / h:.0f} plants/h")


if __name__ == "__main__":
    main()
