# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the W5-C weeds (``bindweed``, ``cheeseweed``, ``groundcherryweed``, ``puncturevine``) and the
ornamental ``bougainvillea``. All at 0 float32 ulp (or exact).

Seeds: case seed ``s`` -> Context seed ``s % 8 + 1`` (Context seeds 1..8). One plant per PlantArchitecture, built
with ``buildPlantInstanceFromLibrary`` (``gv_build``, W4-D; none of these models declares build parameters).

Per model ``M``:

* ``plantarch.M.load`` (exact): engine calls after ``PlantArchitecture(ctx)``, after the load, after a second load
  (draw probe), then every shoot type's ``ShootParameters`` as PyHelios serialises them (float32 JSON);
* ``plantarch.M.age0`` (ulp 0): ``snapshot(include_hidden=True)`` of the age-0 plant;
* ``plantarch.M.steps`` (ulp 0): 2-day ``advanceTime`` calls from age 0 to ``STEP_END[M]`` (past ``max_age`` for the
  weeds, so the last calls are the no-growth branch). After each call: the engine-call probe, the whole
  ``writePlantStructureXML``, every shoot's node count / stored nodes / radii and the number of peduncle, flower and
  fruit objects; the full snapshot stream after the last call;
* ``plantarch.M.ages`` (ulp 0, 8 Context seeds per age of ``AGES[M]``): direct builds at a ladder of ages through
  every stage the model reaches (vegetative, flowering, fruit set for ``groundcherryweed``, ``max_age``, and past
  it), including the Helios selfTest reference age; the same observation plus the full snapshot stream;
* ``plantarch.puncturevine.ages_late`` (ulp 0, Context seeds 1..2): a direct build past ``max_age`` (55 days).
"""
from __future__ import annotations

from ...cases import case
from ...tools.draw_count import engine_calls
from .cases_grapevine import snapshot_stream
from .cases_growth import xml_stream

MODELS = ("bindweed", "cheeseweed", "groundcherryweed", "puncturevine", "bougainvillea")
SHOOT_TYPES = {
    "bindweed": ("base_bindweed", "primary_bindweed", "secondary_bindweed"),
    "cheeseweed": ("base",),
    "groundcherryweed": ("mainstem",),
    "puncturevine": ("base_puncturevine", "primary_puncturevine", "secondary_puncturevine"),
    "bougainvillea": ("mainstem", "secondary", "tertiary", "quaternary"),
}
#: selfTest reference ages (plugins/plantarchitecture/tests/selfTest.cpp:233-266): bindweed 45, cheeseweed 40,
#: groundcherryweed 60, puncturevine 30, bougainvillea 180; max_age: 50, 40, 80, 45, 365
AGES = {
    "bindweed": (12.0, 24.0, 45.0, 50.0, 60.0),
    "cheeseweed": (10.0, 24.0, 40.0, 50.0),
    "groundcherryweed": (15.0, 30.0, 45.0, 60.0, 80.0, 90.0),
    "puncturevine": (12.0, 20.0, 30.0, 45.0),
    "bougainvillea": (40.0, 80.0, 120.0, 180.0),
}
#: direct builds past max_age on 2 Context seeds only (the puncturevine is some 670k primitives by then, ~150 s per
#: build on Heliades); the other weeds have a past-max_age age in AGES
LATE_AGES = {"puncturevine": (55.0,)}
LATE_SEEDS = 2
#: last age of the 2-day sweep: past max_age for bindweed and cheeseweed; the puncturevine sweep stops at its
#: reference age (it doubles its primitive count every ~5 days), the groundcherry and bougainvillea sweeps once
#: flowering is under way
STEP_END = {"bindweed": 56, "cheeseweed": 46, "groundcherryweed": 60, "puncturevine": 30, "bougainvillea": 90}
STEP_DAYS = 2.0
SEEDS = 8


def _seed(s):
    return s % SEEDS + 1


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1.0 if n is None else float(n)


def _start(B, seed, model, age=0.0):
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(model)
    B.gv_build(age, None)


def _observe(B, seed) -> list[float]:
    out = [-201.0, _probe(B, seed), -202.0] + xml_stream(B.pa_xml_text()) + [-203.0]
    for n, verts, radii in B.gv_shoot_summary():
        out += [float(n), float(len(verts))] + verts + radii
    return out + [-204.0, float(B.w4a_inflorescence_objects())]


def _load(B, s, model):
    from ....plantarchitecture.structure import canonical
    seed = _seed(s)
    out = []
    for stages in (0, 1, 2, 3):
        B.seed(seed)
        if stages >= 1:
            B.plantarch_new()
        for _ in range(max(0, stages - 1)):
            B.plantarch_load(model)
        out.append(_probe(B, seed))
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(model)
    for label in SHOOT_TYPES[model]:
        blob = canonical(B.plantarch_shoot_params(label)).encode()
        out += [-1.0, float(len(blob))] + [float(b) for b in blob]
    return out


def _age0(B, s, model):
    _start(B, _seed(s), model)
    return B.snapshot(True)


def _steps(B, s, model):
    seed = _seed(s)
    _start(B, seed, model)
    out = []
    for age in range(2, STEP_END[model] + 1, 2):
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + _observe(B, seed)
    return out + [-205.0] + snapshot_stream(B.snapshot(True))


def _ages(B, s, model):
    seed, age = _seed(s), AGES[model][s // SEEDS]
    _start(B, seed, model, age)
    return [float(age)] + _observe(B, seed) + [-205.0] + snapshot_stream(B.snapshot(True))


def _late(B, s, model):
    seed, age = _seed(s % LATE_SEEDS), LATE_AGES[model][s // LATE_SEEDS]
    _start(B, seed, model, age)
    return [float(age)] + _observe(B, seed) + [-205.0] + snapshot_stream(B.snapshot(True))


def _register(model):
    p = f"plantarch.{model}."
    case(p + "load", tier="exact", fuzz=SEEDS)(lambda B, s: _load(B, s, model))
    case(p + "age0", tier="ulp", max_ulp=0, fuzz=SEEDS)(lambda B, s: _age0(B, s, model))
    case(p + "steps", tier="ulp", max_ulp=0, fuzz=SEEDS)(lambda B, s: _steps(B, s, model))
    case(p + "ages", tier="ulp", max_ulp=0, fuzz=SEEDS * len(AGES[model]))(lambda B, s: _ages(B, s, model))
    if model in LATE_AGES:
        case(p + "ages_late", tier="ulp", max_ulp=0, fuzz=LATE_SEEDS * len(LATE_AGES[model]))(
            lambda B, s: _late(B, s, model))


for _model in MODELS:
    _register(_model)
