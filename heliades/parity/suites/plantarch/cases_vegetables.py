# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the herbaceous vegetables (W5-B). All at 0 float32 ulp.

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One plant per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` on both sides (real W3-A blades on Heliades).

Per crop ``<c>``:

* ``plantarch.<c>.load`` (exact): engine calls after ``PlantArchitecture(ctx)``, after the load and after a second
  load, then every shoot type's ShootParameters as canonical float32 JSON;
* ``plantarch.<c>.age0_full`` (ulp 0): ``snapshot(include_hidden=True)`` of the age-0 plant (hidden prototypes too);
* ``plantarch.<c>.age0_attrs`` (exact): materials, primitive / object data, polymesh face sets of every polymesh;
* ``plantarch.<c>.steps`` (ulp 0): 2-day ``advanceTime`` calls to ``STEP_END[c]``; after each: engine calls, the
  whole plant XML, every shoot's nodes and radii, every internode / petiole tube (objID, UUIDs, vertices), every leaf
  (objID, 4x4, vertices); after the last the full snapshot;
* ``plantarch.<c>.ages`` (ulp 0): direct builds at the ``AGES[c]`` ladder (case seed ``s`` -> age ``AGES[c][s // 8]``,
  Context seed ``s % 8 + 1``): the floral bud states, the ``steps`` observation and the full snapshot.
"""
from __future__ import annotations

import numpy as np

from ...cases import case
from .cases_growth import xml_stream
from .cases_strawberry_maize import snapshot_stream, _probe

f32 = np.float32

STEP_DAYS = 2.0
SHOOT_TYPES = {
    "butterlettuce": ("mainstem",),
    "sugarbeet": ("mainstem",),
    "asparagus": ("main",),
    "cucumber": ("main",),
    "cowpea": ("unifoliate", "trifoliate"),
}
STEP_END = {"butterlettuce": 30, "sugarbeet": 30, "asparagus": 30, "cucumber": 30, "cowpea": 30}
#: direct-build ladders (days): every phenological stage each model reaches
AGES = {
    # rosette growing (10, 30), all 25 leaves out (50), long after (120, 365 = max_age)
    "butterlettuce": (10.0, 30.0, 50.0, 120.0),
    # 30 nodes by ~60 d; then still (120)
    "sugarbeet": (10.0, 30.0, 60.0, 120.0),
    # max_age 20; dormancy at 100 dd
    "asparagus": (10.0, 20.0, 40.0, 120.0),
    # first flowers / fruit (40), head at the wire, lowering, harvest (60, 90), leaves shed at 45 d (90, 120),
    # max_age (165)
    "cucumber": (20.0, 40.0, 60.0, 90.0, 120.0, 165.0),
    # vegetative (20), selfTest age (50), flowering (60, 80), pods (100)
    "cowpea": (20.0, 50.0, 60.0, 80.0, 100.0),
}
CROPS = tuple(SHOOT_TYPES)


def _seed(s):
    return s + 1


def observe(B, seed, *, leaf_vertices=True) -> list[float]:
    """As W4-C's ``observe``, but the tubes without the shoot label: the Helios side of ``pa_tube_objects`` numbers
    the internode objects 0, 1, 2, ... as if every shoot had one, which shifts the labels once a shoot's tube is not
    built yet (asparagus laterals); the objects themselves are compared by ID, UUIDs and vertices. Petioles come from
    W4-B's ``w4b_petiole_objects`` (plant order), which does not truncate at a node with two petioles (W4-B finding 1)."""
    out = [-101.0, float(_probe(B, seed))]
    out += [-102.0] + xml_stream(B.pa_xml_text())
    for verts, rad in B.pa_shoot_internodes(B.pa_shoot_ids()):
        out += [-105.0] + verts + rad
    out.append(-106.0)
    for sid, oid, uu, verts in B.pa_tube_objects("internode"):
        out += [float(oid), float(len(uu))] + [float(u) for u in uu] + verts
    out.append(-107.0)
    for oid, uu, verts in B.w4b_petiole_objects():
        out += [float(oid), float(len(uu))] + [float(u) for u in uu] + verts
    out.append(-108.0)
    for oid, T, verts in B.pa_leaf_objects(leaf_vertices):
        out += [float(oid)] + T + verts
    return out


def _start(B, crop, s, age=0.0):
    seed = _seed(s)
    B.seed(seed)
    B.w5b_build(crop, age)
    return seed


def _load(B, crop, s):
    from ....plantarchitecture.structure import canonical
    seed = _seed(s)
    out = []
    for stages in (1, 2, 3):
        B.seed(seed)
        B.plantarch_new()
        for _ in range(stages - 1):
            B.plantarch_load(crop)
        out.append(float(_probe(B, seed)))
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(crop)
    for label in SHOOT_TYPES[crop]:
        text = canonical(B.plantarch_shoot_params(label)).encode()
        out += [-1.0, float(len(text))] + [float(b) for b in text]
    return out


def _age0_full(B, crop, s):
    _start(B, crop, s)
    return B.snapshot(include_hidden=True)


def _age0_attrs(B, crop, s):
    from .cases_leafproto import encode_attrs
    _start(B, crop, s)
    O = B.snapshot(include_hidden=True).objects
    polys = [int(o) for o, t in zip(O.oid, O.type) if int(t) == 5]
    return encode_attrs(B.plantarch_leafproto_attrs(polys))


def _steps(B, crop, s):
    seed = _start(B, crop, s)
    out = []
    for age in range(2, STEP_END[crop] + 1, 2):
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + observe(B, seed)
    return out + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def _ages(B, crop, s):
    age = AGES[crop][s // 8]
    seed = _start(B, crop, s % 8, age)
    out = [-1.0, age, -2.0] + [float(x) for x in B.w5b_bud_states()]
    return out + observe(B, seed) + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def _register(crop):
    case(f"plantarch.{crop}.load", tier="exact", fuzz=8)(lambda B, s: _load(B, crop, s))
    case(f"plantarch.{crop}.age0_full", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _age0_full(B, crop, s))
    case(f"plantarch.{crop}.age0_attrs", tier="exact", fuzz=8)(lambda B, s: _age0_attrs(B, crop, s))
    case(f"plantarch.{crop}.steps", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _steps(B, crop, s))
    case(f"plantarch.{crop}.ages", tier="ulp", max_ulp=0, fuzz=8 * len(AGES[crop]))(lambda B, s: _ages(B, crop, s))


for _crop in CROPS:
    _register(_crop)


def _encode_normal_maps(rows):
    out = []
    for label, fname, kind, strength in rows:
        b = f"{label}|{fname}".encode()
        out += [float(len(b))] + [float(c) for c in b] + [float(kind), strength]
    return out


@case("plantarch.cucumber.bump_maps", tier="exact", fuzz=16)
def cucumber_bump_maps(B, s):
    """``CucumberMainPhytomerCreationFunction``'s ``setPrimitiveBumpMap`` (strength 12.58861) on every leaf: per leaf
    primitive, its material label and the material's normal map (file name, kind 2 = bump, strength). Case seeds 0..7
    at age 0, 8..15 at age 30 (Context seeds 1..8)."""
    _start(B, "cucumber", s % 8, 0.0 if s < 8 else 30.0)
    rows = B.w5b_leaf_normal_maps()
    assert rows and all(k == 2 for _, _, k, _ in rows)
    return [float(len(rows))] + _encode_normal_maps(rows)
