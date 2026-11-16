# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the strawberry and maize library plants (W4-C). All at 0 float32 ulp.

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One plant per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` on both sides (real blades: the W3-A
``GenericLeafPrototype`` is bound on Heliades).

Per crop (``<c>`` = ``strawberry`` / ``maize``):

* ``plantarch.<c>.load`` (exact): engine calls after ``PlantArchitecture(ctx)``, after the load, after a second
  load (draw-count probe), and the whole ``mainstem`` ShootParameters tree as canonical float32 JSON;
* ``plantarch.<c>.age0_full`` (ulp 0): ``snapshot(include_hidden=True)`` of the age-0 plant: every primitive and
  object, hidden prototypes included;
* ``plantarch.<c>.age0_attrs`` (exact): materials, primitive data, object data and polymesh face sets of every
  polymesh object;
* ``plantarch.<c>.steps`` (ulp 0): 2-day ``advanceTime`` calls up to ``STEP_END[c]``; after each: engine calls,
  the whole plant XML, every shoot's nodes and radii, every internode / petiole tube (objID, UUIDs, vertices),
  every leaf (objID, 4x4, and its vertices: maize blades are re-deflected every step), and at the last step the
  full ``snapshot(include_hidden=True)``;
* ``plantarch.<c>.gallery`` (ulp 0): a direct build at the gallery age (strawberry 60 d: 6 flowers; maize 50 d:
  the ear), the same observation as ``steps`` (XML with ``<flower>`` / ``<peduncle>`` records) plus the full snapshot;
* ``plantarch.<c>.last_vegetative`` (ulp 0): a direct build at the last flower-free age (strawberry 38 d, maize 44 d);
* ``plantarch.<c>.gallery_vegetative`` (ulp 0): the gallery build with W3-C's draws-only inflorescence on Heliades
  (vegetative geometry by value, XML without floral records): checks that path too;
* ``plantarch.strawberry.fruiting`` (ulp 0, 16): days 76 and 100 (flowers + berries; fruit gravity 0.65);
* ``plantarch.maize.flag_leaf`` (ulp 0): day 57, draws-only (the tassel prototype is not ported), engine calls not
  compared.

**Inflorescence gate.** Flower / fruit / ear geometry is W3-D's (``heliades.plantarchitecture.inflorescence``,
merged). ``inflorescence_ported()`` checks it is importable; while it was not, an (age, seed) pair with floral organs
on the oracle (decided by ``w4c_inflorescence_gate``, PyHelios on both backends) returned ``[-999, n]`` on both.
With W3-D present every pair runs in full. The maize tassel (``MaizeTasselPrototype``, terminal bud at the node cap,
day 57+) is not ported, so no case builds a real maize plant past day 56.
"""
from __future__ import annotations

import importlib.util

import numpy as np

from .cases import case
from .cases_plantarch_growth import xml_stream
from .tools.draw_count import engine_calls

f32 = np.float32

CROPS = ("strawberry", "maize")
STEP_END = {"strawberry": 30, "maize": 30}
GALLERY_AGE = {"strawberry": 60.0, "maize": 50.0}
#: the last 2-day age without a floral organ on the oracle, for Context seeds 1..8 (strawberry: first flower on day
#: 40, maize: the ear set as fruit on day 46)
LAST_VEGETATIVE_AGE = {"strawberry": 38.0, "maize": 44.0}
FLAG_LEAF_AGE = 57.0
STEP_DAYS = 2.0


def _seed(s):
    return s + 1


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1 if n is None else n


def inflorescence_ported() -> bool:
    """Capability check for W3-D's inflorescence geometry."""
    return importlib.util.find_spec("heliades.plantarchitecture.inflorescence") is not None


def _start(B, crop, s, age=0.0):
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(crop)
    B.w4c_build(age)
    return seed


def _tubes(B, kind):
    out = []
    for sid, oid, uu, verts in B.pa_tube_objects(kind):
        out += [float(sid), float(oid), float(len(uu))] + [float(u) for u in uu] + verts
    return out


def observe(B, seed, *, leaf_vertices=True) -> list[float]:
    out = [-101.0, float(_probe(B, seed))]
    out += [-102.0] + xml_stream(B.pa_xml_text())
    for verts, rad in B.pa_shoot_internodes(B.pa_shoot_ids()):
        out += [-105.0] + verts + rad
    out += [-106.0] + _tubes(B, "internode") + [-107.0] + _tubes(B, "petiole")
    out.append(-108.0)
    for oid, T, verts in B.pa_leaf_objects(leaf_vertices):
        out += [float(oid)] + T + verts
    return out


def _load(B, crop, s):
    from ..plantarchitecture.structure import canonical
    seed = _seed(s)
    out = []
    for stages in (1, 2, 3):
        B.seed(seed)
        B.plantarch_new()
        if stages >= 2:
            B.plantarch_load(crop)
        if stages >= 3:
            B.plantarch_load(crop)
        out.append(float(_probe(B, seed)))
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(crop)
    text = canonical(B.plantarch_shoot_params("mainstem"))
    return out + [-1.0] + [float(b) for b in text.encode()]


def _age0_full(B, crop, s):
    _start(B, crop, s)
    return B.snapshot(include_hidden=True)


def _age0_attrs(B, crop, s):
    from .cases_plantarch_leafproto import encode_attrs
    _start(B, crop, s)
    return encode_attrs(B.plantarch_leafproto_attrs(B.w4c_polymesh_objects()))


def _steps(B, crop, s):
    seed = _start(B, crop, s)
    out = []
    ages = list(range(2, STEP_END[crop] + 1, 2))
    for age in ages:
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + observe(B, seed)
    snap = B.snapshot(include_hidden=True)
    return out + [-200.0] + snapshot_stream(snap)


def snapshot_stream(snap) -> list[float]:
    """A Snapshot as a flat number stream (so it can follow a stream in one case)."""
    out = [float(len(snap.uuid))]
    out += [float(u) for u in snap.uuid] + [float(t) for t in snap.type] + [float(x) for x in np.asarray(snap.vertices, f32).reshape(-1)]
    out += [float(x) for x in np.asarray(snap.area, f32).reshape(-1)] + [float(x) for x in np.asarray(snap.normal, f32).reshape(-1)]
    out += [float(x) for x in np.asarray(snap.color, f32).reshape(-1)] + [float(p) for p in snap.parent]
    out += [float(x) for x in np.asarray(snap.solid_fraction, f32).reshape(-1)] + [float(h) for h in snap.hidden]
    O = snap.objects
    out += [float(len(O.oid))] + [float(o) for o in O.oid] + [float(h) for h in O.hidden]
    out += [float(x) for x in np.asarray(O.transform, f32).reshape(-1)]
    return out


def _gallery(B, crop, s, age=None):
    seed = _seed(s)
    age = GALLERY_AGE[crop] if age is None else age
    gate = B.w4c_inflorescence_gate(crop, seed, age)
    if gate and not inflorescence_ported():
        return [-999.0, float(gate)]
    _start(B, crop, s, age)
    return [-1.0, float(gate)] + observe(B, seed) + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def _last_vegetative(B, crop, s):
    """A direct build at the last flower-free age (oracle-checked for every seed by the gate)."""
    seed = _seed(s)
    age = LAST_VEGETATIVE_AGE[crop]
    gate = B.w4c_inflorescence_gate(crop, seed, age)
    if gate and not inflorescence_ported():
        return [-999.0, float(gate)]
    _start(B, crop, s, age)
    return [-1.0, float(gate)] + observe(B, seed) + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def _gallery_vegetative(B, crop, s, age=None, probe=True):
    """The gallery build (or one at ``age``) with the inflorescence in draws-only mode on Heliades (no W3-D geometry): draw count, the XML
    without ``<flower>`` / ``<peduncle>`` records, every shoot's nodes and radii, and the internode / petiole / leaf
    geometry by value (object IDs and UUIDs are shifted by the missing floral objects, so they are left out)."""
    from .cases_plantarch_growth import strip_flowers
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(crop)
    B.pa_draws_only_inflorescence()
    B.w4c_build(GALLERY_AGE[crop] if age is None else age)
    out = [-101.0, float(_probe(B, seed)) if probe else -1.0]
    out += [-102.0] + xml_stream(strip_flowers(B.pa_xml_text()))
    for verts, rad in B.pa_shoot_internodes(B.pa_shoot_ids()):
        out += [-105.0] + verts + rad
    for kind in ("internode", "petiole"):
        out.append(-106.0)
        for sid, oid, uu, verts in B.pa_tube_objects(kind):
            out += [float(sid)] + verts
    out.append(-108.0)
    for oid, T, verts in B.pa_leaf_objects(True):
        out += T + verts
    return out


def _register(crop):
    case(f"plantarch.{crop}.load", tier="exact", fuzz=8)(lambda B, s: _load(B, crop, s))
    case(f"plantarch.{crop}.age0_full", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _age0_full(B, crop, s))
    case(f"plantarch.{crop}.age0_attrs", tier="exact", fuzz=8)(lambda B, s: _age0_attrs(B, crop, s))
    case(f"plantarch.{crop}.steps", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _steps(B, crop, s))
    case(f"plantarch.{crop}.gallery", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _gallery(B, crop, s))
    case(f"plantarch.{crop}.last_vegetative", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _last_vegetative(B, crop, s))
    case(f"plantarch.{crop}.gallery_vegetative", tier="ulp", max_ulp=0, fuzz=8)(
        lambda B, s: _gallery_vegetative(B, crop, s))


for _crop in CROPS:
    _register(_crop)


@case("plantarch.maize.flag_leaf", tier="ulp", max_ulp=0, fuzz=8)
def maize_flag_leaf(B, s):
    """Day 57: node 20 (the flag leaf, turned to 23 degrees off the culm by ``setGrassFlagLeafAngle`` ->
    ``Phytomer::rotateLeaf``) and the terminal tassel bud appear in the last step, so the tassel has not yet loaded the
    culm's girth; draws-only inflorescence as in ``gallery_vegetative``. The engine-call count is not compared: the
    oracle's ``MaizeTasselPrototype`` makes ~10^5 Context draws of its own (W3-D / tassel port), after every vegetative
    draw of the build."""
    return _gallery_vegetative(B, "maize", s, FLAG_LEAF_AGE, probe=False)


@case("plantarch.strawberry.fruiting", tier="ulp", max_ulp=0, fuzz=16)
def strawberry_fruiting(B, s):
    """Strawberry with open flowers and berries (day 76: 2 flowers, 3-5 fruit; day 100: fruit only, grown to
    maturity). The berries hang under ``fruit_gravity_factor_fraction = 0.65``, the fruit-gravity rotation that the
    tomato (0) never reaches. Case seeds 0..7 -> day 76, 8..15 -> day 100 (Context seeds 1..8)."""
    return _gallery(B, "strawberry", s % 8, 76.0 if s < 8 else 100.0)
