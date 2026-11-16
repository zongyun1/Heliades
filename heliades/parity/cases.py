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
