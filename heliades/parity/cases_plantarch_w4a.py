# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the W4-A crops: ``cherrytomato`` and ``capsicum``. All at 0 float32 ulp (or exact).

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). Real leaf blades on both
sides (W3-A's ``GenericLeafPrototype``): every blade area, hence girth and lateral seating, and every UUID is
compared as is.

Per crop ``C``:

* ``plantarch.w4a.C.load`` (exact): engine calls after ``PlantArchitecture(ctx)``, after the load, after a second
  load (draw probe), then every shoot type's ``ShootParameters`` as PyHelios serialises them (float32 JSON);
* ``plantarch.w4a.C.age0_full`` (ulp 0): the whole age-0 plant, ``snapshot(include_hidden=True)``: every primitive
  and object, hidden blade prototypes included;
* ``plantarch.w4a.C.age0_full_attrs`` (exact): materials, primitive data and the polymesh face sets of every
  polymesh object;
* ``plantarch.w4a.C.steps`` (ulp 0): ages 2, 4, ..., 40 in 2-day ``advanceTime`` calls; after each: engine calls,
  the whole ``writePlantStructureXML``, every shoot's stored nodes and radii, every internode and petiole tube (object
  ID, UUIDs, vertices), every leaf object (ID, 4x4 transform), the inflorescence object count; the leaf vertices at
  the last step;
* ``plantarch.w4a.C.steps_late`` (ulp 0): the same sweep continued, observed at 42, 44, ..., 80 (the laterals:
  cherry tomato Context seeds 2 and 3 grow their first ``lateral`` shoot, capsicum its ``secondary`` shoots);
* ``plantarch.w4a.C.gallery`` (ulp 0): a direct build at the gallery age (cherrytomato 75 days, capsicum 80) with
  the same observations, leaf vertices included;
* ``plantarch.w4a.C.gallery_full`` (ulp 0): ``snapshot(include_hidden=True)`` of that build;
* ``plantarch.w4a.C.reproductive`` / ``reproductive_full`` (ulp 0, 16 = 8 Context seeds x 2 ages): direct builds with
  flowers and fruit (``REPRODUCTIVE_AGES``: cherry tomato 90 and 110 days, the latter pruned by the callback;
  capsicum 130 and 160 days): the observation above plus every peduncle / flower / fruit object (ID, primitives,
  transform, vertices), and the full snapshot.

**Inflorescence gate.** Flower and fruit geometry is W3-D. A (crop, age, seed) that builds an inflorescence runs
only when ``heliades.plantarchitecture.inflorescence`` exists (:func:`inflorescence_ported`); otherwise the case
returns the marker ``[-999]`` on both backends. W3-D is merged, so nothing is gated: the reproductive cases run.
The vegetative cases stay vegetative: neither crop builds a peduncle, flower or fruit by day 80 (2-day sweep) or by its gallery age (direct build) for Context seeds 1..8 (cherry tomato:
initiation needs 50 days of bud time on node 8 or above; the first closed flowers come on day 82 of a direct build,
Context seeds 3, 4, 6, 7; capsicum: fruit set needs 75 days of bud time, and the main stem's first six buds are dead). The
inflorescence object count is part of every observation, so a wrong gate shows up as a failure, not a silent skip.
"""
from __future__ import annotations

from .cases import case
from .cases_plantarch_growth import xml_stream
from .cases_plantarch_leafproto import encode_attrs
from .tools.draw_count import engine_calls

CROPS = ("cherrytomato", "capsicum")
GALLERY_AGE = {"cherrytomato": 75.0, "capsicum": 80.0}
#: direct-build ages with flowers / fruit (W3-D): cherry tomato flowers from day 82 and is pruned below 1 m from day 101
#: (the callback); capsicum sets its first fruit after day 123 (10 CapsicumFruitPrototype randn draws per cache fill)
REPRODUCTIVE_AGES = {"cherrytomato": (90.0, 110.0), "capsicum": (130.0, 160.0)}
STEP_AGES = tuple(range(2, 41, 2))       # module-level so tests can shorten the sweep
STEP_DAYS = 2.0
LATE_AGES = tuple(range(42, 81, 2))

#: (crop, age, Context seed) pairs that build inflorescence geometry (W3-D). Empty for the ages and seeds here.
GATED: frozenset = frozenset()


def _seed(s):
    return s + 1


def inflorescence_ported() -> bool:
    try:
        from ..plantarchitecture import inflorescence  # noqa: F401  (W3-D)
    except ImportError:
        return False
    return True


def _gated(crop, age, seed) -> bool:
    return (crop, float(age), int(seed)) in GATED and not inflorescence_ported()


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1 if n is None else n


def _observe(B, seed, leaf_vertices=False) -> list[float]:
    out = [-101.0, float(_probe(B, seed))]
    out += [-102.0] + xml_stream(B.pa_xml_text())
    for verts, rad in B.pa_shoot_internodes(B.pa_shoot_ids()):
        out += [-103.0] + verts + rad
    for kind, tag in (("internode", -104.0), ("petiole", -105.0)):
        for oid, uu, verts in B.w4a_tubes(kind):
            out += [tag, float(oid), float(len(uu))] + [float(u) for u in uu] + verts
    out.append(-106.0)
    for oid, T, verts in B.pa_leaf_objects(leaf_vertices):
        out += [float(oid)] + T + verts
    out += [-107.0, float(B.w4a_inflorescence_objects())]
    return out


def _load(B, s, crop):
    from ..plantarchitecture.structure import canonical
    seed = _seed(s)
    out = []
    for stages in (0, 1, 2, 3):
        B.seed(seed)
        if stages >= 1:
            B.plantarch_new()
        for _ in range(max(0, stages - 1)):
            B.plantarch_load(crop)
        out.append(float(_probe(B, seed)))
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(crop)
    labels = ("mainstem", "lateral") if crop == "cherrytomato" else ("mainstem", "secondary")
    for label in labels:
        blob = canonical(B.plantarch_shoot_params(label)).encode()
        out += [-1.0, float(len(blob))] + [float(b) for b in blob]
    return out


def _steps(B, s, crop):
    seed = _seed(s)
    B.seed(seed)
    B.w4a_full_build(crop, 0.0)
    out = []
    for k, age in enumerate(STEP_AGES):
        if _gated(crop, age, seed):
            return out + [-999.0]
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + _observe(B, seed, leaf_vertices=k == len(STEP_AGES) - 1)
    return out


def _steps_late(B, s, crop):
    """Continue the 2-day sweep from 40 to 80 days (no observation before 42): the first laterals (cherry tomato:
    bud break needs a phytomer older than 40 days, so Context seeds 2 and 3 grow one by day 80; capsicum: the
    ``secondary`` shoots with their own phyllochron)."""
    seed = _seed(s)
    B.seed(seed)
    B.w4a_full_build(crop, 0.0)
    for _ in STEP_AGES:
        B.pa_advance(STEP_DAYS)
    out = []
    for k, age in enumerate(LATE_AGES):
        if _gated(crop, age, seed):
            return out + [-999.0]
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + _observe(B, seed, leaf_vertices=k == len(LATE_AGES) - 1)
    return out


def _gallery(B, s, crop):
    seed, age = _seed(s), GALLERY_AGE[crop]
    if _gated(crop, age, seed):
        return [-999.0]
    B.seed(seed)
    B.w4a_full_build(crop, age)
    return _observe(B, seed, leaf_vertices=True)


def _gallery_full(B, s, crop):
    seed, age = _seed(s), GALLERY_AGE[crop]
    if _gated(crop, age, seed):
        return [-999.0]
    B.seed(seed)
    B.w4a_full_build(crop, age)
    return B.snapshot(include_hidden=True)


def _reproductive(B, s, crop, full):
    ages = REPRODUCTIVE_AGES[crop]
    seed, age = _seed(s % 8), ages[s // 8]
    if not inflorescence_ported():
        return [-999.0]
    B.seed(seed)
    B.w4a_full_build(crop, age)
    if full:
        return B.snapshot(include_hidden=True)
    return _observe(B, seed, leaf_vertices=True) + [-108.0] + B.w4a_inflorescence_geometry()


def _age0_full(B, s, crop):
    B.seed(_seed(s))
    B.w4a_full_build(crop, 0.0)
    return B.snapshot(include_hidden=True)


def _age0_full_attrs(B, s, crop):
    snap = _age0_full(B, s, crop)
    polys = [int(o) for o, t in zip(snap.objects.oid, snap.objects.type) if t == 5]
    return encode_attrs(B.plantarch_leafproto_attrs(polys))


def _register(crop):
    p = f"plantarch.w4a.{crop}."
    case(p + "load", tier="exact", fuzz=8)(lambda B, s: _load(B, s, crop))
    case(p + "age0_full", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _age0_full(B, s, crop))
    case(p + "age0_full_attrs", tier="exact", fuzz=8)(lambda B, s: _age0_full_attrs(B, s, crop))
    case(p + "steps", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _steps(B, s, crop))
    case(p + "steps_late", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _steps_late(B, s, crop))
    case(p + "gallery", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _gallery(B, s, crop))
    case(p + "gallery_full", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _gallery_full(B, s, crop))
    case(p + "reproductive", tier="ulp", max_ulp=0, fuzz=16)(lambda B, s: _reproductive(B, s, crop, False))
    case(p + "reproductive_full", tier="ulp", max_ulp=0, fuzz=16)(lambda B, s: _reproductive(B, s, crop, True))


for _crop in CROPS:
    _register(_crop)


#: (crop, plant age, nodes) for the literal-type probe: the plant ages where the float and the double forms of the
#: creation functions' scales round differently (cherry tomato: leaf 1, 7, 13, internode 3, 7; capsicum: internode 1).
#: A library build creates phytomers only at multiples of the phyllochron, so only a direct build reaches them.
LITERAL_PROBES = (("cherrytomato", 7.0, 1), ("cherrytomato", 13.0, 1), ("cherrytomato", 3.0, 1),
                  ("cherrytomato", 1.0, 1), ("capsicum", 1.0, 8))


@case("plantarch.w4a.literal_probe", tier="ulp", max_ulp=0, fuzz=len(LITERAL_PROBES))
def w4a_literal_probe(B, s):
    """``addPlantInstance(pos, age)`` + ``addBaseStemShoot`` + 2 days: draws, XML, nodes, tubes and leaves."""
    crop, age, nodes = LITERAL_PROBES[s]
    seed = 3
    B.seed(seed)
    B.w4a_direct_shoot(crop, age, nodes)
    return _observe(B, seed, leaf_vertices=True)
