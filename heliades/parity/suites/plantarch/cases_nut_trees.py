# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the nut trees (W5-D): ``almond``, ``almond_aldrich``, ``almond_independence``,
``almond_wood_colony``, ``pistachio``, ``walnut``. Everything at 0 float32 ulp.

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One tree per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` (``tree_build``).

Per model ``<m>``:

* ``plantarch.<m>.load`` (exact): engine calls after ``PlantArchitecture(ctx)``, after the load, after a second
  load and after an age-0 build;
* ``plantarch.<m>.params`` (exact): every ShootParameters tree of the model as PyHelios serialises it;
* ``plantarch.<m>.age<A>`` (ulp 0): a direct build at age ``A``: the engine-call probe, per shoot its node count and
  stored nodes and radii, ``writePlantStructureXML`` as a token stream, and the whole hidden-inclusive snapshot as a
  stream (UUIDs, vertices, colours, UVs, solid fractions, hidden flags, objects, members and transforms).

The ages (``AGES``) walk the multi-year cycle. The almonds break dormancy on day 90 (the others on day 165), flower
3 days later, set fruit 7 days after that, and reach the season boundary on day 365, where (after shade shedding and
growth priority) every shoot goes dormant, leaves fall and fruit is harvested; the second season breaks on day 455
(almonds) / 530 and the reference age of Helios's own plausibility test is 548 (the summer of the second season).
Shade shedding first draws at the second boundary (day 730: a branch must have survived one dormancy), covered by
``almond.age732`` and ``walnut.age732``; ``almond.age913`` is the trunk-thickening reference age.

* ``plantarch.trees.build_parameters`` (exact): accepted names of every tree model, and the maps each rejects
  (unknown key, out-of-range values, a non-integer scaffold count, a trunk too short for its scaffolds);
* ``plantarch.<m>.build_parameters_age<A>`` (ulp 0): non-default trunk height, scaffold count and angle;
* ``plantarch.almond.steps`` / ``plantarch.pistachio.steps`` (ulp 0): 5-day ``advanceTime`` calls across flowering
  and fruit set (almond) and the first-season scaffold tipping with its deferred cuts (pistachio); after each call
  the structure and the full snapshot.
"""
from __future__ import annotations

import numpy as np

from ...cases import case
from ...tools.draw_count import engine_calls
from .cases_grapevine import snapshot_stream
from .cases_growth import xml_stream

f32 = np.float32

MODELS = ("almond", "almond_aldrich", "almond_independence", "almond_wood_colony", "pistachio", "walnut")
SHOOT_TYPES = {
    "almond": ("trunk", "scaffold", "proleptic", "sylleptic", "spur"),
    "almond_aldrich": ("trunk", "scaffold", "proleptic", "sylleptic"),
    "almond_independence": ("trunk", "scaffold", "proleptic", "sylleptic"),
    "almond_wood_colony": ("trunk", "scaffold", "proleptic", "sylleptic"),
    "pistachio": ("trunk", "scaffold", "proleptic", "spur"),
    "walnut": ("trunk", "scaffold", "proleptic"),
}
#: (age, fuzz) per model; module-level so tests can shorten them
AGES = {
    "almond": ((0, 8), (95, 8), (130, 8), (250, 8), (366, 8), (548, 8), (732, 4), (913, 2)),
    "almond_aldrich": ((0, 8), (95, 8), (200, 8), (366, 8), (548, 8)),
    "almond_independence": ((0, 8), (95, 8), (200, 8), (366, 8), (548, 8)),
    "almond_wood_colony": ((0, 8), (95, 8), (200, 8), (366, 8), (548, 8)),
    "pistachio": ((0, 8), (170, 8), (220, 8), (300, 8), (366, 8), (548, 8)),
    "walnut": ((0, 8), (170, 8), (220, 8), (366, 8), (548, 8), (732, 2)),
}
NON_DEFAULT = {
    "almond": {"trunk_height": 1.2, "num_scaffolds": 3, "scaffold_angle": 40.0},
    "almond_independence": {"trunk_height": 0.6, "num_scaffolds": 6, "scaffold_angle": 65.0},
    "pistachio": {"trunk_height": 0.8, "num_scaffolds": 5, "scaffold_angle": 45.0},
    "walnut": {"trunk_height": 1.1, "num_scaffolds": 3, "scaffold_angle": 30.0},
}
REJECTED = {
    "almond": ({"tree_height": 3.0}, {"trunk_height": 3.5}, {"num_scaffolds": 2.5}, {"num_scaffolds": 9},
               {"scaffold_angle": 75.0}, {"trunk_height": float("nan")},
               {"trunk_height": 0.1, "num_scaffolds": 8}),           # 3 trunk nodes + 1 cannot carry 8 scaffolds
    "almond_aldrich": ({"trunk_height": 0.8},),                        # declares no build parameters
    "almond_independence": ({"scaffold_angle": 15.0}, {"trunk_height": 0.1, "num_scaffolds": 5}),
    "almond_wood_colony": ({"num_scaffolds": 4},),
    "pistachio": ({"trunk_height": 0.05}, {"trunk_height": 0.1, "num_scaffolds": 8}),   # 4 pairs, 3 trunk nodes
    "walnut": ({"scaffold_angle": 65.0}, {"scaffold_angle": 20.0}, {"trunk_height": 0.1, "num_scaffolds": 5}),
}
STEPS = {"almond": (85.0, 5.0, 10), "pistachio": (160.0, 5.0, 14)}     # (start age, step, count)


def _seed(s):
    return s + 1


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1.0 if n is None else float(n)


def _start(B, model, s, age=0.0, build_parameters=None):
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(model)
    B.tree_build(age, build_parameters)
    return seed


def _structure(B, seed) -> list[float]:
    out = [-201.0, _probe(B, seed), -202.0]
    for sid, n, verts, radii in B.tree_shoot_summary():
        out += [float(sid), float(n), float(len(verts))] + verts + radii
    return out


def observe(B, seed) -> list[float]:
    """Probe + shoot summary, the plant XML, the whole snapshot."""
    return (_structure(B, seed) + [-203.0] + xml_stream(B.pa_xml_text()) + [-204.0]
            + snapshot_stream(B.snapshot(True)))


def _make_cases(model):
    @case(f"plantarch.{model}.load", tier="exact", fuzz=8)
    def load(B, s, _m=model):
        seed = _seed(s)
        out = []
        for stage in range(5):
            B.seed(seed)
            if stage >= 1:
                B.plantarch_new()
            if stage >= 2:
                B.plantarch_load(_m)
            if stage == 3:
                B.plantarch_load(_m)
            if stage == 4:
                B.tree_build(0.0, None)
            out.append(_probe(B, seed))
        return out

    @case(f"plantarch.{model}.params", tier="exact", fuzz=1)
    def params(B, s, _m=model):
        from ....plantarchitecture.structure import canonical
        B.seed(_seed(s))
        B.plantarch_new()
        B.plantarch_load(_m)
        txt = "|".join(canonical(B.plantarch_shoot_params(label)) for label in SHOOT_TYPES[_m])
        return list(txt.encode())

    for age, fuzz in AGES[model]:
        def grown(B, s, _m=model, _a=float(age)):
            seed = _start(B, _m, s, _a)
            return observe(B, seed)
        case(f"plantarch.{model}.age{int(age)}", tier="ulp", max_ulp=0, fuzz=fuzz)(grown)

    if model in NON_DEFAULT:
        for age, fuzz in ((0, 4), (200, 2)):
            def nondefault(B, s, _m=model, _a=float(age)):
                seed = _start(B, _m, s, _a, NON_DEFAULT[_m])
                return observe(B, seed)
            case(f"plantarch.{model}.build_parameters_age{age}", tier="ulp", max_ulp=0, fuzz=fuzz)(nondefault)

    if model in STEPS:
        @case(f"plantarch.{model}.steps", tier="ulp", max_ulp=0, fuzz=4)
        def steps(B, s, _m=model):
            start, step, count = STEPS[_m]
            seed = _start(B, _m, s, start)
            out = []
            age = start
            for _ in range(count):
                B.pa_advance(step)
                age += step
                out += [-100.0, float(age)] + _structure(B, seed) + [-203.0] + snapshot_stream(B.snapshot(True))
            return out + [-205.0] + xml_stream(B.pa_xml_text())


for _model in MODELS:
    _make_cases(_model)


@case("plantarch.trees.build_parameters", tier="exact", fuzz=1)
def trees_build_parameters(B, s):
    out = []
    for model in MODELS:
        B.seed(_seed(s))
        B.plantarch_new()
        B.plantarch_load(model)
        out += list(f"{model}:{','.join(B.tree_build_parameter_names(model))};".encode())
        for bp in REJECTED[model]:
            out.append(B.tree_build_error(0.0, bp))
        if model in NON_DEFAULT:
            out.append(B.tree_build_error(0.0, NON_DEFAULT[model]))
    return out
