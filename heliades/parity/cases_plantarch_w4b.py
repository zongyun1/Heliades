# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the bean and soybean (W4-B). All at 0 float32 ulp.

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One plant per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` and the real W3-A leaf blades.

Per crop (``bean``, ``soybean``):

* ``plantarch.<crop>.load``: engine calls after ``loadPlantModelFromLibrary`` (5 identifier draws), then the
  whole ``unifoliate`` / ``trifoliate`` (/ ``branch``) ShootParameters trees as PyHelios JSON;
* ``plantarch.<crop>.age0_full`` / ``age0_full_attrs``: the age-0 plant from ``snapshot(include_hidden=True)``
  (every primitive and object, hidden leaf prototypes included) and the attributes of every polymesh object;
* ``plantarch.<crop>.steps``: 2-day ``advanceTime`` calls up to ``STEP_AGES[-1]``; after each: engine calls,
  the whole structure XML, every shoot's nodes and radii, every internode / petiole tube object (UUIDs and
  vertices) and every leaflet object (ID and 4x4 transform), the leaflet vertices at the last step;
* ``plantarch.<crop>.steps_full``: a full ``snapshot(include_hidden=True)`` after the 2-day steps;
* ``plantarch.<crop>.flowering``: a direct build past the first flowers (bean 60, soybean 70 days), the number of
  flower / fruit buds and the ``_observe`` checks (gated while W3-D is missing);
* ``plantarch.<crop>.flowering_full``: the whole snapshot of that plant (flowers, pods, their hidden prototypes);
* ``plantarch.<crop>.gallery``: a direct build at the gallery age (bean 45, soybean 50): engine calls, XML,
  all tubes and leaflets with vertices, and ``plantarch.<crop>.gallery_full`` the whole snapshot.

**Inflorescence gating (W3-D).** Flower and pod geometry needs W3-D's ``updateInflorescence``. Until it is merged,
Heliades runs that seam in its draws-only mode, and a (seed, age) pair whose plant has a closed / open flower or a
fruiting bud on the oracle (``_has_inflorescence``) is compared without the flower geometry: draw count, XML minus
the ``<flower>`` / ``<peduncle>`` records, and tubes and leaflets by value instead of by UUID / object ID; the
full-snapshot cases of such a pair compare only a flag. Every vegetative pair runs all checks.
"""
from __future__ import annotations

import numpy as np

from .cases import case
from .cases_plantarch_growth import _probe, strip_flowers, xml_stream
from .cases_plantarch_leafproto import encode_attrs

f32 = np.float32

STEP_AGES = tuple(range(2, 31, 2))      # module-level so tests can shorten the sweep
STEP_DAYS = 2.0
GALLERY_AGE = {"bean": 45.0, "soybean": 50.0}
FLOWERING_AGE = {"bean": 60.0, "soybean": 70.0}   # past the first flowers of seeds 1..8
CROPS = ("bean", "soybean")
SHOOT_TYPES = {"bean": ("unifoliate", "trifoliate"), "soybean": ("unifoliate", "trifoliate", "branch")}

_FLORAL_GEOMETRY_STATES = (2, 3, 4)     # BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING


def _seed(s):
    return s + 1


def _start(B, s, crop, age=0.0):
    seed = _seed(s)
    B.seed(seed)
    B.w4b_build(crop, age)
    return seed


def _gated(B) -> bool:
    """True when the plant carries inflorescence geometry but this backend cannot build it yet. Both backends
    must give the same answer, so it is decided from the backend's own bud states plus the capability flag of
    the *Heliades* side (the Helios op reports True for its capability, the bud states are equal by parity)."""
    has = any(st in _FLORAL_GEOMETRY_STATES for st in B.w4b_floral_states())
    from .ops_plantarch_w4b import inflorescence_ported
    return has and not inflorescence_ported()


def _tubes(B, kind, uuids=True):
    """Tube objects in plant order: internodes with their shoot index (``pa_tube_objects``), petioles from
    ``w4b_petiole_objects`` (see there for why not ``pa_tube_objects``)."""
    out = []
    if kind == "internode":
        for sid, oid, uu, verts in B.pa_tube_objects(kind):
            out += [float(sid)] + ([float(oid), float(len(uu))] + [float(u) for u in uu] if uuids else []) + verts
    else:
        for oid, uu, verts in B.w4b_petiole_objects():
            out += ([float(oid), float(len(uu))] + [float(u) for u in uu] if uuids else [float(len(uu))]) + verts
    return out


def _observe(B, seed, *, leaf_vertices=False) -> list[float]:
    gated = _gated(B)
    ids = not gated
    out = [-100.0, float(gated), -101.0, float(_probe(B, seed))]
    text = B.pa_xml_text()
    if gated:
        text = strip_flowers(text)
    out += [-102.0] + xml_stream(text)
    for verts, rad in B.pa_shoot_internodes(B.pa_shoot_ids()):
        out += [-105.0] + verts + rad
    out += [-106.0] + _tubes(B, "internode", uuids=ids) + [-107.0] + _tubes(B, "petiole", uuids=ids)
    out.append(-108.0)
    for oid, T, verts in B.pa_leaf_objects(leaf_vertices):
        out += ([float(oid)] if ids else []) + T + verts
    if not gated:
        from .ops_plantarch_w4b import inflorescence_ported
        if inflorescence_ported():
            out += [-120.0] + [float(x) for x in B.inflo_plant_organs()]   # every peduncle / flower / fruit object
    return out


def _full(B):
    """The full snapshot, or (gated pair) a one-value stream that both sides produce identically."""
    if _gated(B):
        return [-1.0]
    return B.snapshot(include_hidden=True)


def _params_stream(B, crop) -> list[float]:
    from ..plantarchitecture.structure import canonical
    out = []
    for label in SHOOT_TYPES[crop]:
        b = canonical(B.plantarch_shoot_params(label)).encode()
        out += [float(len(b))] + [float(c) for c in b]
    return out


def _register(crop):
    @case(f"plantarch.{crop}.load", tier="exact", fuzz=8)
    def load(B, s):
        seed = _seed(s)
        B.seed(seed)
        B.plantarch_new()
        B.plantarch_load(crop)
        out = [float(_probe(B, seed))]
        B.seed(seed)
        B.plantarch_new()
        B.plantarch_load(crop)
        return out + _params_stream(B, crop)

    @case(f"plantarch.{crop}.age0_full", tier="ulp", max_ulp=0, fuzz=8)
    def age0_full(B, s):
        _start(B, s, crop)
        return B.snapshot(include_hidden=True)

    @case(f"plantarch.{crop}.age0_full_attrs", tier="exact", fuzz=8)
    def age0_full_attrs(B, s):
        _start(B, s, crop)
        snap = B.snapshot(include_hidden=True)
        polys = [int(o) for o, t in zip(snap.objects.oid, snap.objects.type) if t == 5]
        return encode_attrs(B.plantarch_leafproto_attrs(polys))

    @case(f"plantarch.{crop}.steps", tier="ulp", max_ulp=0, fuzz=8)
    def steps(B, s):
        seed = _start(B, s, crop)
        out = []
        for k, age in enumerate(STEP_AGES):
            B.pa_advance(STEP_DAYS)
            out += [-99.0, float(age)] + _observe(B, seed, leaf_vertices=k == len(STEP_AGES) - 1)
        return out

    @case(f"plantarch.{crop}.steps_full", tier="ulp", max_ulp=0, fuzz=8)
    def steps_full(B, s):
        _start(B, s, crop)
        for _ in STEP_AGES:
            B.pa_advance(STEP_DAYS)
        return _full(B)

    @case(f"plantarch.{crop}.gallery", tier="ulp", max_ulp=0, fuzz=8)
    def gallery(B, s):
        seed = _start(B, s, crop, GALLERY_AGE[crop])
        return _observe(B, seed, leaf_vertices=True)

    @case(f"plantarch.{crop}.gallery_full", tier="ulp", max_ulp=0, fuzz=8)
    def gallery_full(B, s):
        _start(B, s, crop, GALLERY_AGE[crop])
        return _full(B)


    @case(f"plantarch.{crop}.flowering", tier="ulp", max_ulp=0, fuzz=8)
    def flowering(B, s):
        """Past the first flowers (gated until W3-D): the floral state machine and its draws, everything else."""
        seed = _start(B, s, crop, FLOWERING_AGE[crop])
        n_floral = sum(st in _FLORAL_GEOMETRY_STATES for st in B.w4b_floral_states())
        return [-98.0, float(n_floral)] + _observe(B, seed)

    @case(f"plantarch.{crop}.flowering_full", tier="ulp", max_ulp=0, fuzz=8)
    def flowering_full(B, s):
        _start(B, s, crop, FLOWERING_AGE[crop])
        return _full(B)


for _crop in CROPS:
    _register(_crop)
