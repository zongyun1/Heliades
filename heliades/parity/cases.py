# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases: each is written once against the backend API and run on Helios and Heliades.

A case returns either a Snapshot (compared primitive by primitive) or a 1-D sequence of numbers
(compared draw by draw). Register with ``@case(name, tier=...)``. Fuzz cases take a ``seed`` so the
runner can sweep many random inputs.
"""
from __future__ import annotations

import math

import numpy as np

CASES: dict[str, dict] = {}


def case(name: str, tier: str = "ulp", max_ulp: int = 4, fuzz: int = 1):
    def deco(fn):
        CASES[name] = dict(fn=fn, tier=tier, max_ulp=max_ulp, fuzz=fuzz)
        return fn
    return deco


# ---- core.random ---------------------------------------------------------------------------

@case("core.random.randu", tier="exact", fuzz=8)
def rng_randu(B, seed):
    B.seed(seed * 7919 + seed)
    return [B.randu() for _ in range(500)]


@case("core.random.randu_range", tier="exact", fuzz=8)
def rng_randu_range(B, seed):
    B.seed(seed)
    return [B.randu(-0.3 * seed, 2.5 + seed) for _ in range(500)]


@case("core.random.randu_int", tier="exact", fuzz=8)
def rng_randu_int(B, seed):
    B.seed(seed)
    return [B.randu(-3, 7 + 1000 * seed) for _ in range(500)]


@case("core.random.randn", tier="ulp", max_ulp=2, fuzz=8)
def rng_randn(B, seed):
    """log/sqrt go through libm, so allow an ulp or two."""
    B.seed(seed)
    return [B.randn() for _ in range(500)] + [B.randn(1.5, -0.2) for _ in range(500)]


@case("core.random.interleaved", tier="ulp", max_ulp=2, fuzz=4)
def rng_interleaved(B, seed):
    """Mixed calls check the normal's saved spare and the shared generator state."""
    B.seed(seed)
    out = []
    for k in range(300):
        out += [B.randu(), B.randn(), float(B.randu(0, 9)), B.randu(1.0, 2.0)]
        if k % 50 == 49:
            B.seed(seed + k)
    return out


# ---- core.primitives -----------------------------------------------------------------------

def _rand_inputs(seed, n=40):
    r = np.random.default_rng(seed)
    return r


@case("core.primitive.patch", fuzz=6)
def patch_fuzz(B, seed):
    r = _rand_inputs(seed)
    for _ in range(40):
        rot = (0.0, 0.0) if r.random() < 0.2 else (r.uniform(-math.pi, math.pi), r.uniform(0, 2 * math.pi))
        B.add_patch(r.uniform(-5, 5, 3), r.uniform(0.01, 3, 2), rot, r.random(3))
    return B.snapshot()


@case("core.primitive.triangle", fuzz=6)
def triangle_fuzz(B, seed):
    r = _rand_inputs(seed)
    for _ in range(40):
        c = r.uniform(-5, 5, 3)
        B.add_triangle(c + r.normal(0, 1, 3), c + r.normal(0, 1, 3), c + r.normal(0, 1, 3), r.random(3))
    return B.snapshot()


@case("core.primitive.transforms", fuzz=6)
def transform_fuzz(B, seed):
    r = _rand_inputs(seed)
    uu = [B.add_patch(r.uniform(-1, 1, 3), r.uniform(0.1, 1, 2)) for _ in range(10)]
    uu += [B.add_triangle(r.normal(0, 1, 3), r.normal(0, 1, 3), r.normal(0, 1, 3)) for _ in range(10)]
    B.add_tile((0, 0, 0), (1, 1), (0.2, 0.1), (2, 2))          # its members (next UUIDs) are locked
    uu += list(range(20, 24))
    for _ in range(60):
        u = uu[r.integers(len(uu))]
        k = r.integers(3)
        if k == 0:
            B.translate(u, r.uniform(-1, 1, 3))
        elif k == 1:
            B.rotate(u, r.uniform(-3, 3), "xyz"[r.integers(3)])
        else:
            B.scale(u, r.uniform(0.5, 2, 3))
    return B.snapshot()


# ---- core.objects --------------------------------------------------------------------------

@case("core.object.tile", fuzz=6)
def tile_fuzz(B, seed):
    r = _rand_inputs(seed)
    for _ in range(6):
        rot = (0.0, 0.0) if r.random() < 0.3 else (r.uniform(-math.pi, math.pi), r.uniform(0, 2 * math.pi))
        B.add_tile(r.uniform(-3, 3, 3), r.uniform(0.05, 2, 2), rot, (int(r.integers(1, 5)), int(r.integers(1, 5))),
                   r.random(3))
    return B.snapshot()


def _stem(r, n):
    """Random plant-like polyline: mostly upward with random bends, occasional kinks and repeats."""
    p = [r.uniform(-1, 1, 3)]
    d = np.array([0.0, 0.0, 1.0])
    for k in range(n - 1):
        u = r.random()
        if u < 0.1:
            d = r.normal(size=3)                       # sharp kink
        elif u < 0.2:
            pass                                       # straight (parallel-transport fallback path)
        else:
            d = d + 0.4 * r.normal(size=3)
        d = d / np.linalg.norm(d)
        step = 0.0 if r.random() < 0.03 else r.uniform(0.005, 0.1)   # coincident nodes
        p.append(p[-1] + step * d)
    return np.array(p)


@case("core.object.tube", fuzz=8)
def tube_fuzz(B, seed):
    r = _rand_inputs(seed)
    for _ in range(5):
        n = int(r.integers(2, 14))
        nodes = _stem(r, n)
        rad = r.uniform(0.0005, 0.02, n)
        rad[r.random(n) < 0.1] = 1e-7                  # below the 1e-5 clamp
        cols = r.random((n, 3)) if r.random() < 0.5 else None
        B.add_tube(int(r.integers(3, 40)), nodes, rad, cols)   # >= 13 hits non-correctly-rounded glibc sinf/cosf
    return B.snapshot()


@case("core.object.tube_axes", fuzz=1)
def tube_axes(B, seed):
    """Axis-aligned and near-degenerate tubes hit every initial-radial branch."""
    for d in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0.96, 0.28, 0], [0.3, 0.1, 0.95], [1e-8, 0, 0]):
        d = np.array(d, float)
        B.add_tube(6, [np.zeros(3), d, 2 * d, 2 * d + [0, 0, 0.1]], [0.01, 0.01, 0.008, 0.005])
    return B.snapshot()


@case("core.object.polymesh", fuzz=4)
def polymesh_fuzz(B, seed):
    r = _rand_inputs(seed)
    uu = [B.add_triangle(r.normal(size=3), r.normal(size=3), r.normal(size=3)) for _ in range(8)]
    uu += [B.add_patch(r.uniform(-1, 1, 3), r.uniform(0.1, 1, 2)) for _ in range(4)]
    a = B.add_polymesh(uu[:6])
    B.add_polymesh(uu[:6])                             # re-wrap returns the same object
    B.add_polymesh(uu[3:])                             # partially parented: only free ones join
    B.add_tile(r.uniform(-1, 1, 3), (1.0, 1.0), (0.2, 0.3), (2, 2))
    return B.snapshot()


@case("core.registry.selftest", tier="exact", fuzz=1)
def registry_selftest(B, seed):
    """An op from ops_example_selftest.py, reached through the dispatch fallback."""
    B.add_patch((0, 0, 0), (1, 1))
    B.add_tile((0, 0, 1), (1, 1), (0, 0), (2, 2))
    return [float(B.primitive_count())]


from .ops import load_plugins as _load  # noqa: E402  (cases_* modules register on import)

_load()
