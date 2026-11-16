# SPDX-License-Identifier: LGPL-2.1-or-later
"""Acceptance: the library-built tomato (W3-E). All at 0 float32 ulp.

The grid is ``AGES`` x Context seeds 1..8. Case seed ``s`` -> age ``AGES[s // 8]``, Context seed ``s % 8 + 1``,
so ``fuzz = 48`` sweeps it.

* ``plantarchitecture.tomato`` (ulp, 0): ``buildPlantInstanceFromLibrary((0,0,0), age)`` then the whole Context
  from ``snapshot(include_hidden=True)``: every primitive (UUID, type, vertices, normal, area, color, UVs, texture,
  solid fraction, parent, hidden flag) and every object (members, center, transform, hidden flag);
* ``plantarchitecture.tomato.state`` (ulp, 0): the same build, then
  - the materials and data of every primitive and object (``pa_accept_attrs``: material label, its color, texture,
    two-sided flag; each primitive datum; each object datum with its type and value);
  - the whole ``writePlantStructureXML`` (number by number);
  - ``plant_structure_dict`` against the oracle XML through the same parser, plus the organ object IDs;
  - the public query API (``pa_accept_api``): plant age/height/leaf area/leaf count/..., shoot topology, per-phytomer
    getters, organ object IDs, canopy-wide UUID lists, ``object_label`` selections;
  - last, the engine-call count of the whole build (probe, SPEC §3.5).
  The snapshot case caches this stream from its own build, so the grid builds each plant once per backend.

Multi-plant (Context seeds 1..8 unless noted):

* ``plantarchitecture.tomato.two_plants`` (snapshot) / ``.two_plants_state``: two 30-day tomatoes from two
  ``buildPlantInstanceFromLibrary`` calls in one PlantArchitecture. The second plant reuses the 35 cached blade
  prototypes: no new hidden objects, no private-stream draws, so its UUIDs and draws are the oracle's only if the
  cache is shared exactly as in Helios;
* ``plantarchitecture.tomato.canopy`` / ``.canopy_state`` (seeds 1..4): ``buildPlantCanopyFromLibrary`` 2 x 2 at 0.5 m,
  germination 0.75 (one ``randu`` per site; some sites stay empty), 20 days: the plants are built at age 0 and grown
  together, interleaved per sub-step;
* ``plantarchitecture.tomato.staggered`` / ``.staggered_state`` (seeds 1..4): plant A, ``advanceTime`` all, plant B
  built at age 4 in the same PlantArchitecture, then ``advanceTime([A, B])``, ``advanceTime(dt, plant_id=B)``, all.

**Flower gating.** From day 56 some seeds flower (W3-C), and the flower / fruit geometry is W3-D's. A pair whose
oracle plant ever built inflorescence geometry (``oracle_has_inflorescence``) is *gated* while
``inflorescence_geometry_implemented()`` is false: both backends then return the same sentinel (a one-primitive snapshot,
``[-999, age, seed]``), so the pair is skipped, not failed, and it switches on by itself when W3-D merges.
``python -m heliades.parity.cases_plantarch_acceptance`` prints the grid (ran / gated) and the timings.
"""
from __future__ import annotations

import numpy as np

from .cases import case
from .ops_plantarch_acceptance import (TIMINGS, flatten, inflorescence_geometry_implemented, oracle_has_inflorescence,
                                       pair_is_gated)
from .snapshot import Snapshot
from .tools.draw_count import engine_calls

f32 = np.float32

AGES = (10.0, 30.0, 50.0, 75.0, 100.0, 120.0)
SEEDS = tuple(range(1, 9))


def grid_pair(s: int):
    return AGES[s // len(SEEDS)], SEEDS[s % len(SEEDS)]


def _sentinel_snapshot(age, seed) -> Snapshot:
    """A gated pair: the same one-primitive snapshot on both backends (UUID -1, vertices = (age, seed, 0))."""
    V = np.zeros((1, 4, 3), f32)
    V[0, :, 0], V[0, :, 1] = age, seed
    return Snapshot(np.array([-1], np.int64), np.array([1], np.int32), V, np.array([3], np.int32),
                    np.zeros(1, f32), np.zeros((1, 3), f32), np.zeros((1, 4), f32))


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1.0 if n is None else float(n)


def _xml_stream(text):
    from .cases_plantarch_growth import xml_stream
    return xml_stream(text)


def _state(B, seed, pids) -> list:
    """Materials/data, XML, structure dict and API of the plants, then the engine-call count (consumes 3 draws)."""
    out = [-200.0] + B.pa_accept_attrs()
    for pid in pids:
        out += [-201.0, float(pid)] + _xml_stream(B.pa_accept_xml(pid))
        out += [-202.0] + flatten(B.pa_accept_structure(pid))
    out += [-203.0] + flatten(B.pa_accept_api(list(pids)))
    out += [-204.0, _probe(B, seed)]
    return out


# ---- the grid ---------------------------------------------------------------------------------------------

_STATE_CACHE: dict = {}
#: (age, seed) -> "ran" | "gated", filled as the grid runs
GRID_STATUS: dict = {}


def _grid_build(B, s):
    age, seed = grid_pair(s)
    if pair_is_gated(seed, age):
        GRID_STATUS[(age, seed)] = "gated"
        return None
    GRID_STATUS[(age, seed)] = "ran"
    B.pa_accept_build(seed, age)
    return age, seed


@case("plantarchitecture.tomato", tier="ulp", max_ulp=0, fuzz=len(AGES) * len(SEEDS))
def tomato_acceptance(B, s):
    p = _grid_build(B, s)
    if p is None:
        return _sentinel_snapshot(*grid_pair(s))
    snap = B.snapshot(include_hidden=True)
    _STATE_CACHE[(B.name, p)] = _state(B, p[1], [B.pid])
    return snap


@case("plantarchitecture.tomato.state", tier="ulp", max_ulp=0, fuzz=len(AGES) * len(SEEDS))
def tomato_acceptance_state(B, s):
    age, seed = grid_pair(s)
    cached = _STATE_CACHE.pop((B.name, (age, seed)), None)
    if cached is not None:
        return cached
    p = _grid_build(B, s)
    if p is None:
        return [-999.0, age, float(seed)]
    return _state(B, seed, [B.pid])


# ---- multiple plants --------------------------------------------------------------------------------------

TWO_PLANT_AGE = 30.0
TWO_PLANT_POSITIONS = ((0.0, 0.0, 0.0), (0.6, 0.25, 0.0))
CANOPY = dict(center=(0.1, -0.2, 0.0), spacing=(0.5, 0.5), count=(2, 2), age=20.0, germination=0.75)


def _multi(B, s, kind):
    seed = s + 1
    if kind == "two":
        pids = B.pa_accept_build_two(seed, TWO_PLANT_AGE, TWO_PLANT_POSITIONS)
    elif kind == "canopy":
        pids = B.pa_accept_canopy(seed, **CANOPY)
    else:
        pids = B.pa_accept_staggered(seed)
    return seed, [int(p) for p in pids]


def _multi_case(kind):
    def snap_case(B, s):
        seed, pids = _multi(B, s, kind)
        snap = B.snapshot(include_hidden=True)
        _STATE_CACHE[(B.name, kind, s)] = [-210.0, float(len(pids))] + _state(B, seed, pids)
        return snap

    def state_case(B, s):
        cached = _STATE_CACHE.pop((B.name, kind, s), None)
        if cached is not None:
            return cached
        seed, pids = _multi(B, s, kind)
        return [-210.0, float(len(pids))] + _state(B, seed, pids)
    return snap_case, state_case


for _kind, _name, _fuzz in (("two", "two_plants", 8), ("canopy", "canopy", 4), ("staggered", "staggered", 4)):
    _snap, _st = _multi_case(_kind)
    case(f"plantarchitecture.tomato.{_name}", tier="ulp", max_ulp=0, fuzz=_fuzz)(_snap)
    case(f"plantarchitecture.tomato.{_name}_state", tier="ulp", max_ulp=0, fuzz=_fuzz)(_st)


# ---- report ---------------------------------------------------------------------------------------------

def grid_report() -> str:
    lines = [f"inflorescence geometry implemented: {inflorescence_geometry_implemented()}",
             "age   " + "  ".join(f"s{s}" for s in SEEDS)]
    for age in AGES:
        row = []
        for seed in SEEDS:
            row.append("F " if oracle_has_inflorescence(seed, age) else ". ")
        lines.append(f"{age:5.0f} " + "  ".join(row))
    lines.append("(F = the oracle plant builds inflorescence geometry: gated until W3-D; . = flower-free, runs)")
    return "\n".join(lines)


def timing_report() -> str:
    lines = ["age  seed  helios_s  heliades_s  ratio"]
    keys = sorted({k[1] for k in TIMINGS if isinstance(k[1], tuple) and k[1][0] == "build"})
    for k in keys:
        h, f = TIMINGS.get(("helios", k)), TIMINGS.get(("heliades", k))
        if h is None or f is None:
            continue
        lines.append(f"{k[1]:4.0f} {k[2]:5d} {h:9.3f} {f:11.3f} {f / h:6.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(grid_report())
