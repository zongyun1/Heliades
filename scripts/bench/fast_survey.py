# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fast mode against exact mode over the plant library (``heliades.parity.fastmode``), one JSON line per build.

    python scripts/bench/fast_survey.py --seeds 1-8 --workers 6 --out survey.jsonl [models...]
    python scripts/bench/fast_survey.py --sensitivity --out floor.jsonl [models...]
    python scripts/bench/fast_survey.py --distribution 1-32 --out dist.jsonl grapevine_VSP

``--sensitivity`` measures the floor fast mode cannot go below: exact mode against exact mode with every leaf area
read (``Phytomer::getLeafArea``, ``Context::getObjectArea``) scaled by 1 + 1e-7 (one float32 rounding).
``--distribution`` records the plant metrics of every seed in both modes (for ``compare_distributions``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

#: representative mature ages (days)
AGES = {
    "asparagus": 120, "bean": 60, "bindweed": 60, "bougainvillea": 180, "butterlettuce": 50, "capsicum": 160,
    "cheeseweed": 50, "cherrytomato": 110, "cowpea": 100, "cucumber": 165, "groundcherryweed": 90, "maize": 120,
    "puncturevine": 45, "rice": 90, "sorghum": 100, "soybean": 70, "strawberry": 100, "sugarbeet": 120, "tomato": 120,
    "wheat": 90, "grapevine_VSP": 120, "grapevine_Wye": 120,
    "almond": 548, "almond_aldrich": 548, "almond_independence": 548, "almond_wood_colony": 548, "apple": 548,
    "apple_fruitingwall": 548, "easternredbud": 548, "olive": 548, "pistachio": 548, "walnut": 548,
}
HEAVY = {"olive", "grapevine_Wye", "puncturevine", "almond", "sorghum"}


def _seeds(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _check(job):
    model, age, seed = job
    from heliades.parity.fastmode import check
    t0 = time.perf_counter()
    try:
        r = check(model, age, seed)
    except Exception as e:                      # report, keep going
        r = {"model": model, "age": age, "seed": seed, "error": f"{type(e).__name__}: {e}"}
    r["wall"] = time.perf_counter() - t0
    return r


def _sensitivity(job):
    model, age, seed = job
    import numpy as np
    from heliades.core.objects import ObjectsMixin
    from heliades.parity.fastmode import compare, record
    a = record(model, age, seed, "exact")
    orig = ObjectsMixin.object_area

    def bumped(self, oid):
        return np.float32(orig(self, oid) * (1 + 1e-7))
    ObjectsMixin.object_area = bumped
    try:
        b = record(model, age, seed, "exact")
    finally:
        ObjectsMixin.object_area = orig
    r = compare(a, b)
    r["kind"] = "sensitivity"
    return r


def _metrics(job):
    model, age, seed = job
    from heliades.parity.fastmode import record
    out = {"model": model, "age": age, "seed": seed}
    for mode in ("exact", "fast"):
        rec = record(model, age, seed, mode)
        out[mode] = rec.metrics
        out[mode + "_draws"] = rec.draws
        out[mode + "_seconds"] = rec.seconds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--age", type=float, help="override the age of every model")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--distribution", metavar="SEEDS")
    a = ap.parse_args()
    models = a.models or sorted(AGES, key=lambda m: (m in HEAVY, m))
    fn, seeds = _check, _seeds(a.seeds)
    if a.sensitivity:
        fn = _sensitivity
        seeds = seeds[:1] if a.seeds == "1-8" else seeds
    if a.distribution:
        fn, seeds = _metrics, _seeds(a.distribution)
    jobs = [(m, a.age or AGES[m], s) for m in models for s in seeds]
    import multiprocessing as mp
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                d = json.loads(line)
                done.add((d["model"], float(d["age"]), int(d["seed"])))
            except (ValueError, KeyError):
                pass
    jobs = [j for j in jobs if (j[0], float(j[1]), j[2]) not in done]
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(fn, jobs):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(json.dumps({k: r[k] for k in list(r)[:12]}, default=float), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
