# SPDX-License-Identifier: LGPL-2.1-or-later
"""Time one library plant build on Heliades or Helios (the oracle), report peak RSS, and optionally save the
whole-scene snapshot (hidden geometry included) so two runs can be compared bit for bit.

    python scripts/bench/plant_build.py heliades tomato 120 --seed 1 [--save out.npz] [--profile out.prof]
    python scripts/bench/plant_build.py helios grapevine_VSP 120
    python scripts/bench/plant_build.py compare a.npz b.npz
"""
from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np


def _save(snap, path):
    o = snap.objects
    np.savez_compressed(path, uuid=snap.uuid, type=snap.type, vertices=snap.vertices, nverts=snap.nverts,
                        area=snap.area, normal=snap.normal, color=snap.color, parent=snap.parent, uv=snap.uv,
                        nuv=snap.nuv, solid_fraction=snap.solid_fraction, hidden=snap.hidden,
                        texture=np.array([t.rsplit("/", 1)[-1] for t in snap.texture]), oid=o.oid, otype=o.type,
                        ocenter=o.center, otransform=o.transform, ohidden=o.hidden,
                        omembers=np.concatenate(o.uuids) if o.uuids else np.zeros(0, np.int64),
                        olens=np.array([len(u) for u in o.uuids], np.int64))


def compare(a, b) -> int:
    """Bitwise comparison (float arrays as their bit patterns, so -0 != 0 and NaN payloads count)."""
    A, B = np.load(a), np.load(b)
    bad = 0
    for k in A.files:
        x, y = A[k], B[k]
        if x.shape != y.shape:
            print(f"{k}: shape {x.shape} vs {y.shape}")
            bad += 1
            continue
        if x.dtype == np.float32:
            x, y = x.view(np.uint32), y.view(np.uint32)
        n = int((x != y).sum())
        if n:
            print(f"{k}: {n} differing entries")
            bad += 1
    print("IDENTICAL" if not bad else f"{bad} arrays differ")
    return bad


def main():
    if sys.argv[1] == "compare":
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=("heliades", "helios"))
    ap.add_argument("model")
    ap.add_argument("age", type=float)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save")
    ap.add_argument("--profile")
    a = ap.parse_args()
    from heliades.parity.backends import ForgeBackend, HeliosBackend
    B = ForgeBackend() if a.backend == "heliades" else HeliosBackend()
    B.seed(a.seed)
    B.plantarch_new()
    t0 = time.perf_counter()
    B.plantarch_load(a.model)
    if a.backend == "heliades":
        from heliades.plantarchitecture import leaf_prototype  # noqa: F401
        pos = (0.0, 0.0, 0.0)
    else:
        from pyhelios.types import vec3
        pos = vec3(0, 0, 0)
    t1 = time.perf_counter()
    pr = None
    if a.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    B.pa.buildPlantInstanceFromLibrary(pos, float(a.age))
    if pr is not None:
        pr.disable()
        pr.dump_stats(a.profile)
    t2 = time.perf_counter()
    snap = B.snapshot(include_hidden=True) if a.save else None
    t3 = time.perf_counter()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    n = B.scene.primitive_count() if a.backend == "heliades" else B.ctx.getPrimitiveCount()
    print(f"{a.backend} {a.model} age={a.age:g} seed={a.seed}: load {t1 - t0:.2f}s build {t2 - t1:.2f}s"
          f" snapshot {t3 - t2:.2f}s prims {n} peakRSS {rss:.0f}MB", flush=True)
    if a.save:
        _save(snap, a.save)


if __name__ == "__main__":
    main()
