# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for species files (S0): a species file against the Helios library model it re-expresses. 0 ulp.

``heliades/species/defs/cucumber_from_tomato.yaml`` builds Helios's cucumber from the *tomato* template (overrides,
named rules, a generic builder). On Heliades it is registered as plant model ``cucumber`` (so the XML's
``<plant_model>`` matches); on Helios the library ``cucumber`` is built. The observations are those of the W5-B
vegetable cases (``cases_vegetables``), Context seeds 1..8:

* ``plantarch.species.cucumber_from_tomato.load`` (exact): engine calls after the PlantArchitecture, one and two
  loads, and the ``main`` ShootParameters as canonical float32 JSON;
* ``...age0_full`` (ulp 0) / ``...age0_attrs`` (exact): the age-0 snapshot (hidden prototypes too) and the materials,
  primitive / object data and polymesh face sets;
* ``...steps`` (ulp 0): 2-day steps to day 30, engine calls + XML + shoots + tubes + leaves after each, the full
  snapshot at the end;
* ``...ages`` (ulp 0): direct builds at 20, 40, 60, 90, 120 and 165 days (flowering, fruit, the head at the wire and
  the lowering, harvest, leaf shedding, max_age): bud states, the ``steps`` observation and the full snapshot;
* ``...bump_maps`` (exact): every leaf primitive's material and bump map at day 0 and day 30.
"""
from __future__ import annotations

from ...cases import case
from .cases_strawberry_maize import _probe, snapshot_stream
from .cases_vegetables import AGES, STEP_DAYS, STEP_END, _encode_normal_maps, observe

SPECIES = {
    # species file -> (Helios library model it reproduces, shoot types)
    "cucumber_from_tomato": ("cucumber", ("main",)),
}


def _seed(s):
    return s + 1


def _start(B, spec, s, age=0.0):
    seed = _seed(s)
    B.seed(seed)
    B.species_build(spec, SPECIES[spec][0], age)
    return seed


def _load(B, spec, s):
    from ....plantarchitecture.structure import canonical
    label, shoot_types = SPECIES[spec]
    seed = _seed(s)
    out = []
    for stages in (1, 2, 3):
        B.seed(seed)
        B.species_new(spec, label)
        for _ in range(stages - 1):
            B.plantarch_load(label)
        out.append(float(_probe(B, seed)))
    B.seed(seed)
    B.species_new(spec, label)
    B.plantarch_load(label)
    for st in shoot_types:
        text = canonical(B.plantarch_shoot_params(st)).encode()
        out += [-1.0, float(len(text))] + [float(b) for b in text]
    return out


def _age0_full(B, spec, s):
    _start(B, spec, s)
    return B.snapshot(include_hidden=True)


def _age0_attrs(B, spec, s):
    from .cases_leafproto import encode_attrs
    _start(B, spec, s)
    O = B.snapshot(include_hidden=True).objects
    polys = [int(o) for o, t in zip(O.oid, O.type) if int(t) == 5]
    return encode_attrs(B.plantarch_leafproto_attrs(polys))


def _steps(B, spec, s):
    seed = _start(B, spec, s)
    out = []
    for age in range(2, STEP_END[SPECIES[spec][0]] + 1, 2):
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + observe(B, seed)
    return out + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def _ages(B, spec, s):
    age = AGES[SPECIES[spec][0]][s // 8]
    seed = _start(B, spec, s % 8, age)
    out = [-1.0, age, -2.0] + [float(x) for x in B.w5b_bud_states()]
    return out + observe(B, seed) + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def _bump_maps(B, spec, s):
    _start(B, spec, s % 8, 0.0 if s < 8 else 30.0)
    rows = B.w5b_leaf_normal_maps()
    assert rows and all(k == 2 for _, _, k, _ in rows)
    return [float(len(rows))] + _encode_normal_maps(rows)


def _register(spec):
    p = f"plantarch.species.{spec}"
    n_ages = len(AGES[SPECIES[spec][0]])
    case(f"{p}.load", tier="exact", fuzz=8)(lambda B, s: _load(B, spec, s))
    case(f"{p}.age0_full", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _age0_full(B, spec, s))
    case(f"{p}.age0_attrs", tier="exact", fuzz=8)(lambda B, s: _age0_attrs(B, spec, s))
    case(f"{p}.steps", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: _steps(B, spec, s))
    case(f"{p}.ages", tier="ulp", max_ulp=0, fuzz=8 * n_ages)(lambda B, s: _ages(B, spec, s))
    case(f"{p}.bump_maps", tier="exact", fuzz=16)(lambda B, s: _bump_maps(B, spec, s))


for _spec in SPECIES:
    _register(_spec)
