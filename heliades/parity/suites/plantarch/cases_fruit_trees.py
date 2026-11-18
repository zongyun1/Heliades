# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the fruit / ornamental trees (W5-E): ``apple``, ``apple_fruitingwall``, ``olive``,
``easternredbud``. Everything at 0 float32 ulp.

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8); the process-wide generator
behind ``helios::randu()`` (drawn by the redbud callback) is seeded with the same value. One tree per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` (``tree_build``, W5-D's op).

Per model ``<m>`` (as for the nut trees, ``cases_nut_trees``):

* ``plantarch.<m>.load`` (exact), ``plantarch.<m>.params`` (exact);
* ``plantarch.<m>.age<A>`` (ulp 0): engine-call probe, per shoot its node count and stored nodes and radii, the plant
  XML as a token stream, the whole hidden-inclusive snapshot;
* ``plantarch.<m>.data<A>`` (ulp 0): every primitive's material label and primitive data (the spectral class maps
  of the apple's leaves, bark and fruit, the fruit's reflectivity spectrum and waxy layer, object labels);
* ``plantarch.<m>.build_parameters_age<A>`` (ulp 0) for the models that declare build parameters;
* ``plantarch.fruit_trees.build_parameters`` (exact): accepted names and the maps each model rejects;
* ``plantarch.<m>.steps`` (ulp 0): ``advanceTime`` steps across a phase change, the structure and the snapshot after
  each.

Ages walk each model's cycle. All four break dormancy on day 165 (the redbud is built with dormancy already
broken, so it grows from day 0), flower 3 days later and set fruit after 7 more; the season boundary is day 365
(leaf fall except for the evergreen olive, harvest, growth priority, shade shedding from the second boundary on).
Apple: flower thinning by the past summer's light runs on dormant phytomers from the first boundary. Fruiting wall:
winter pruning (hedging, competitor pinching, renewal cuts, heading of the leader at the wall height) runs from
the leader's tip each dormant step, summer training (stub regrowth, bud release below the heading cut) each
growing step. Olive: leaves older than 600 days are shed (first at ~day 770). 548 is the reference age of Helios's
plausibility test (the summer of the second season).
"""
from __future__ import annotations

import numpy as np

from ...cases import case
from .cases_nut_trees import _probe, _structure
from .cases_grapevine import snapshot_stream
from .cases_growth import xml_stream

f32 = np.float32

MODELS = ("apple", "apple_fruitingwall", "olive", "easternredbud")
SHOOT_TYPES = {
    "apple": ("trunk", "proleptic", "spur"),
    "apple_fruitingwall": ("leader", "branch", "proleptic", "renewal", "spur"),
    "olive": ("trunk", "scaffold", "proleptic"),
    "easternredbud": ("eastern_redbud_trunk", "eastern_redbud_shoot"),
}
#: (age, fuzz) per model; module-level so tests can shorten them
AGES = {
    "apple": ((0, 8), (170, 8), (200, 8), (300, 8), (366, 8), (548, 8), (732, 2), (913, 1), (1460, 1)),
    "apple_fruitingwall": ((0, 8), (170, 8), (250, 8), (366, 8), (548, 4), (732, 2), (913, 1), (1460, 1), (3650, 1)),
    "olive": ((0, 8), (170, 8), (250, 8), (366, 8), (548, 8), (800, 2), (1100, 1),
              (1825, 1)),       # 1825 d: 1.30M primitives, ~5 min and 1.6 GB per Heliades build
    "easternredbud": ((0, 8), (30, 8), (120, 8), (250, 8), (366, 8), (548, 8), (732, 2), (1100, 1), (1460, 1)),
}
DATA_AGES = {"apple": ((250, 2),), "apple_fruitingwall": ((250, 2),), "olive": ((250, 2),),
             "easternredbud": ((250, 2),)}
NON_DEFAULT = {
    "apple": {"trunk_height": 1.0, "num_scaffolds": 3, "scaffold_angle": 55.0},
    "apple_fruitingwall": {"tree_height": 2.5, "trunk_height": 0.5, "num_feathers": 13, "tree_spacing": 1.2,
                           "row_spacing": 4.0},
    "olive": {"trunk_height": 0.9},
}
REJECTED = {
    "apple": ({"tree_height": 3.0}, {"trunk_height": 3.5}, {"num_scaffolds": 9}, {"scaffold_angle": 75.0},
              {"trunk_height": 0.1, "num_scaffolds": 8}),
    "apple_fruitingwall": ({"tree_height": 1.9}, {"tree_height": 4.6}, {"trunk_height": 1.01}, {"trunk_height": 0.05},
                           {"trunk_height": float(np.nextafter(f32(1), f32(2)))},              # the float max: accepted
                           {"trunk_height": float(np.nextafter(np.nextafter(f32(1), f32(2)), f32(2)))},
                           {"num_feathers": 5}, {"num_feathers": 7.5}, {"tree_spacing": 0.4}, {"row_spacing": 6.0},
                           {"num_scaffolds": 4}),
    "olive": ({"trunk_height": 0.1}, {"trunk_height": 1.6}, {"num_scaffolds": 4}),
    "easternredbud": ({"trunk_height": 0.8},),
}
#: cheaper variants that reach the later training events early: a 2 m wall heads its leader in the first winter
VARIANTS = {"apple_fruitingwall": (("short_wall", {"tree_height": 2.0}, ((400, 4), (800, 2), (1100, 1))),)}
STEPS = {"apple": (360.0, 2.0, 6), "apple_fruitingwall": (360.0, 2.0, 6), "easternredbud": (360.0, 2.0, 6)}


def _seed(s):
    return s + 1


def _start(B, model, s, age=0.0, build_parameters=None):
    seed = _seed(s)
    B.seed(seed)
    B.global_seed(seed)
    B.plantarch_new()
    B.plantarch_load(model)
    B.tree_build(age, build_parameters)
    return seed


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
            B.global_seed(seed)
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

    for age, fuzz in DATA_AGES[model]:
        def data(B, s, _m=model, _a=float(age)):
            _start(B, _m, s, _a)
            return B.ft_prim_data()
        case(f"plantarch.{model}.data{int(age)}", tier="ulp", max_ulp=0, fuzz=fuzz)(data)

    if model in NON_DEFAULT:
        for age, fuzz in ((0, 4), (200, 2)):
            def nondefault(B, s, _m=model, _a=float(age)):
                seed = _start(B, _m, s, _a, NON_DEFAULT[_m])
                return observe(B, seed)
            case(f"plantarch.{model}.build_parameters_age{age}", tier="ulp", max_ulp=0, fuzz=fuzz)(nondefault)

    for label, bp, ladder in VARIANTS.get(model, ()):
        for age, fuzz in ladder:
            def variant(B, s, _m=model, _a=float(age), _bp=bp):
                seed = _start(B, _m, s, _a, _bp)
                return observe(B, seed)
            case(f"plantarch.{model}.{label}_age{age}", tier="ulp", max_ulp=0, fuzz=fuzz)(variant)

    if model in STEPS:
        @case(f"plantarch.{model}.steps", tier="ulp", max_ulp=0, fuzz=2)
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


@case("plantarch.fruit_trees.build_parameters", tier="exact", fuzz=1)
def fruit_trees_build_parameters(B, s):
    out = []
    for model in MODELS:
        B.seed(_seed(s))
        B.global_seed(_seed(s))
        B.plantarch_new()
        B.plantarch_load(model)
        out += list(f"{model}:{','.join(B.tree_build_parameter_names(model))};".encode())
        for bp in REJECTED[model]:
            out.append(B.tree_build_error(0.0, bp))
        if model in NON_DEFAULT:
            out.append(B.tree_build_error(0.0, NON_DEFAULT[model]))
    return out
