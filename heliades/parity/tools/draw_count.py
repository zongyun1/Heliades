# SPDX-License-Identifier: LGPL-2.1-or-later
"""Engine-call probe: how many raw ``minstd_rand0`` calls did a stage consume? (SPEC §3.5)

Works the same on Helios (PyHelios) and Heliades, with no instrumentation of either:

1. seed the Context / Scene with ``s``;
2. run the stage;
3. read ``k`` values of ``randu()`` (one engine call each);
4. find the smallest ``n`` such that a fresh ``MinstdRand0(s)`` advanced ``n`` calls reproduces those
   ``k`` values.

``k = 3`` makes an accidental match essentially impossible (a single float32 value repeats about once in
2^24 positions, so ``k = 1`` is ambiguous beyond a few million calls).

Usage::

    python -m heliades.parity.tools.draw_count [--backend helios|heliades|both] [--seeds 1,2,5] [STAGE ...]

Stages: ``ctx`` (nothing), ``ctor`` (``PlantArchitecture(ctx)``), ``load`` (+ ``loadPlantModelFromLibrary``,
model from ``--model``, default tomato), ``build:AGE`` (+ ``buildPlantInstanceFromLibrary``; Helios only
until W3-E). From code: :func:`engine_calls` / :func:`probe_backend`.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from ...core.random import MinstdRand0

_A, _M = MinstdRand0.A, MinstdRand0.M
_CHUNK = 1 << 16
_POW = None


def _powers():
    global _POW
    if _POW is None:
        p = np.empty(_CHUNK, np.int64)
        p[0] = 1
        for i in range(1, _CHUNK):
            p[i] = (p[i - 1] * _A) % _M
        _POW = p
    return _POW


def _uniform_float(x: np.ndarray) -> np.ndarray:
    """``randu()`` value whose single engine call returned ``x`` (unitUniform<float>, 30-bit path)."""
    r = ((x - 1).astype(np.float64) / 2147483646.0).astype(np.float32)
    one = np.float32(1)
    return np.where(r >= one, np.nextafter(one, np.float32(0)), r)


def engine_calls(seed: int, probe, nmax: int = 50_000_000) -> int | None:
    """Engine calls consumed between ``seed(seed)`` and the first of the ``probe`` randu() values.

    ``probe`` is the sequence of float32 ``randu()`` values read right after the stage. Returns ``None``
    if no match within ``nmax`` calls.
    """
    probe = np.asarray(probe, np.float32)
    k = len(probe)
    if k == 0:
        raise ValueError("need at least one probe value")
    P = _powers()
    g = MinstdRand0(seed)
    x0 = g.x                                    # state after n calls, n = base
    base = 0
    tail = np.empty(0, np.float32)              # last k-1 values of the previous chunk
    while base < nmax:
        xs = (P * x0) % _M                      # states after base+0 .. base+CHUNK-1 calls
        nxt = (xs * _A) % _M                    # the engine output of call base+i+1
        u = np.concatenate([tail, _uniform_float(nxt)])
        off = base - len(tail)                  # u[j] is the randu() after off+j calls
        cand = np.nonzero(u[: len(u) - k + 1] == probe[0])[0] if len(u) >= k else []
        for j in cand:
            if np.array_equal(u[j:j + k], probe):
                return int(off + j)
        tail = u[len(u) - (k - 1):] if k > 1 else np.empty(0, np.float32)
        x0 = int((xs[-1] * _A) % _M)
        base += _CHUNK
    return None


def calls_between(seed: int, generator, nmax: int = 50_000_000) -> int | None:
    """Heliades-only shortcut: engine calls from ``seed`` to the current state of ``generator``."""
    target = generator.x
    g = MinstdRand0(seed)
    for n in range(nmax):
        if g.x == target:
            return n
        g()
    return None


def probe_backend(B, seed: int, stage, k: int = 3) -> int | None:
    """Seed backend ``B``, run ``stage(B)``, then probe. ``B`` is a parity backend (Helios or Heliades)."""
    B.seed(seed)
    stage(B)
    return engine_calls(seed, [B.randu() for _ in range(k)])


# ---- CLI ----------------------------------------------------------------------------------------

def _stage_fn(name: str, model: str):
    def run(B):
        if name == "ctx":
            return
        B.plantarch_new()
        if name == "ctor":
            return
        B.plantarch_load(model)
        if name == "load":
            return
        if name.startswith("build:"):
            B.plantarch_build((0.0, 0.0, 0.0), float(name.split(":", 1)[1]))
            return
        raise ValueError(f"unknown stage {name!r}")
    return run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="*", default=["ctx", "ctor", "load", "build:0", "build:2", "build:10"])
    ap.add_argument("--backend", default="both", choices=("helios", "heliades", "both"))
    ap.add_argument("--seeds", default="5")
    ap.add_argument("--model", default="tomato")
    a = ap.parse_args(argv)
    from ..backends import ForgeBackend, HeliosBackend
    from ..ops import load_plugins
    load_plugins()
    makers = {"helios": HeliosBackend, "heliades": ForgeBackend}
    names = ("helios", "heliades") if a.backend == "both" else (a.backend,)
    for seed in [int(s) for s in a.seeds.split(",")]:
        for st in a.stages:
            row = []
            for nm in names:
                try:
                    row.append(f"{nm}={probe_backend(makers[nm](), seed, _stage_fn(st, a.model))}")
                except NotImplementedError as e:
                    row.append(f"{nm}=n/a ({e})")
            print(f"seed {seed}  {st:<10} " + "  ".join(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
