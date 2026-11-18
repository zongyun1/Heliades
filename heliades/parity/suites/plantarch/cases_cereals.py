# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the library grasses (W5-A): the maize tassel (maize through maturity), wheat, rice and sorghum.
All at 0 float32 ulp (``exact`` for draw counts / names / error texts).

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One plant per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` on both sides (real blades and real inflorescences).

The observation of a build (``observe`` of W4-C) is: engine calls after the build (draw-count probe), the whole plant
XML, every shoot's nodes and radii, every internode / petiole tube, every leaf (4x4 and vertices); then the full
``snapshot(include_hidden=True)`` as a stream (every primitive and object, hidden prototypes included). Builds with
build parameters leave out the XML (the Heliades writer does not write build parameters; W4-D).

Maize (``maize``): the ear is set on day 46 and the tassel (terminal bud at node 20, ``MaizeTasselPrototype``,
~10^5 Context draws per prototype, 3 prototypes) on day 57; the tassel expands over its 6-day
``inflorescence_maturity_period``, the ear over the 58-day fruit-maturity threshold. Ages: 58 (tassel just formed),
64 (tassel at full size, anthesis), 120 (ear mature), 365 (the age the Helios gallery renders, utilities/
render_plant_library: last full-leaf age with fruit, in 5-day steps up to max_age 365; the plant stops changing
after ~125 d). ``tassel_steps``: days 54..66 in 2-day calls (the tassel forming and expanding under advanceTime).
``tassel_build_parameters``: non-default organ build parameters (tassel_type presets and blends, single keys,
spikelet_subdivisions) at 60 d. ``build_parameters``: accepted names and the error texts of rejected maps.
"""
from __future__ import annotations

import numpy as np

from ...cases import case
from .cases_growth import xml_stream
from .cases_strawberry_maize import _probe, snapshot_stream

f32 = np.float32


def _seed(s):
    return s + 1


def start(B, crop, s, age=0.0, build_parameters=None):
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(crop)
    B.gv_build(age, build_parameters)
    return seed


def _tubes(B, kind):
    """Tube objects (objID, UUIDs, vertices) in shoot order. The shoot of a petiole is left out: the oracle op infers
    it from per-shoot node counts, which does not hold once a tiller carries a phytomer whose petiole is still to come
    (the objects themselves are compared)."""
    out = []
    for sid, oid, uu, verts in B.pa_tube_objects(kind):
        out += ([float(sid)] if kind == "internode" else []) + [float(oid), float(len(uu))] + [float(u) for u in uu] + verts
    return out


def observe(B, seed, *, xml=True) -> list[float]:
    out = [-101.0, float(_probe(B, seed))]
    out += [-103.0] + [float(x) for x in B.cereal_counts()]
    if xml:
        out += [-102.0] + xml_stream(B.pa_xml_text())
    for verts, rad in B.pa_shoot_internodes(B.pa_shoot_ids()):
        out += [-105.0] + verts + rad
    out += [-106.0] + _tubes(B, "internode") + [-107.0] + _tubes(B, "petiole")
    out.append(-108.0)
    for oid, T, verts in B.pa_leaf_objects(True):
        out += [float(oid)] + T + verts
    return out


def build_case(B, crop, s, age, build_parameters=None):
    seed = start(B, crop, s, age, build_parameters)
    out = observe(B, seed, xml=not build_parameters)
    return out + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def steps_case(B, crop, s, start_age, end_age, step, build_parameters=None, full_every=False):
    seed = start(B, crop, s, start_age, build_parameters)
    out = []
    age = start_age
    while age < end_age:
        B.pa_advance(step)
        age += step
        out += [-100.0, float(age)] + observe(B, seed, xml=not build_parameters)
        if full_every:
            out += [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))
    return out + [-200.0] + snapshot_stream(B.snapshot(include_hidden=True))


def names_and_errors(B, crop, rejected, accepted=()):
    """Accepted build-parameter names, then per map the error text (empty for an accepted map) as bytes."""
    B.seed(1)
    B.plantarch_new()
    B.plantarch_load(crop)
    out = [float(b) for b in ",".join(B.cereal_build_parameter_names(crop)).encode()]
    for bp in list(rejected) + list(accepted):
        text = B.cereal_build_error_text(bp)
        out += [-1.0, float(len(text))] + [float(b) for b in text.encode()]
    return out


# ---- maize ------------------------------------------------------------------------------------------------------

MAIZE_AGES = {"tassel": 58.0, "anthesis": 64.0, "mature": 120.0, "gallery365": 365.0}

for _name, _age in MAIZE_AGES.items():
    case(f"plantarch.maize.{_name}", tier="ulp", max_ulp=0, fuzz=8)(
        lambda B, s, _a=_age: build_case(B, "maize", s, _a))


@case("plantarch.maize.tassel_steps", tier="ulp", max_ulp=0, fuzz=4)
def maize_tassel_steps(B, s):
    """Days 54..66 in 2-day calls from a direct build at 54 d: the tassel bud appears (day 57), its prototypes are
    built and placed, and it expands to full size under advanceTime."""
    return steps_case(B, "maize", s, 54.0, 66.0, 2.0)


#: per case seed: a non-default organ build-parameter map for the tassel
MAIZE_TASSEL_BUILD_PARAMETERS = [
    {"tassel_type": 0.0},
    {"tassel_type": 1.0},
    {"tassel_type": 0.3},
    {"tassel_type": 0.8, "tassel_branch_count": 14, "spikelet_subdivisions": 5},
    {"tassel_length": 0.3, "tassel_axis_curvature": 0.0, "tassel_secondary_branch_count": 0},
    {"tassel_branch_count": 0, "anther_fraction": 1.0},
    {"anther_fraction": 0.0, "glume_color_r": 0.6, "tassel_branch_droop_base": 80.0, "filament_length": 0.0},
    {"spikelet_subdivisions": 3, "tassel_spike_spikelet_density": 30.0, "tassel_branch_bare_fraction": 0.5},
]


@case("plantarch.maize.tassel_build_parameters", tier="ulp", max_ulp=0, fuzz=8)
def maize_tassel_build_parameters(B, s):
    return build_case(B, "maize", s, 60.0, MAIZE_TASSEL_BUILD_PARAMETERS[s])


MAIZE_REJECTED = [
    {"tiller_count": 2},                                # not a maize parameter
    {"tassel_length": 0.9},                             # out of range
    {"tassel_type": -0.1},
    {"tassel_branch_count": 2.5},                       # not a whole number
    {"spikelet_subdivisions": 2},
    {"anther_color_g": 1.5},
    {"tassel_branch_bare_fraction": float("nan")},
]
MAIZE_ACCEPTED = [{"tassel_type": 0.25, "tassel_branch_count": 12, "anther_color_r": 0.9}]


@case("plantarch.maize.build_parameters", tier="exact", fuzz=1)
def maize_build_parameters(B, s):
    return names_and_errors(B, "maize", MAIZE_REJECTED, MAIZE_ACCEPTED)


# ---- shared per-crop cases ----------------------------------------------------------------------------------------

def load_case(B, crop, s, shoot_types):
    """Engine calls after the ctor, the load and a second load; every shoot type's ShootParameters as canonical
    float32 JSON (as PyHelios serialises them)."""
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
    for label in shoot_types:
        out += [-1.0] + [float(b) for b in canonical(B.plantarch_shoot_params(label)).encode()]
    return out


def age0_case(B, crop, s, build_parameters=None):
    start(B, crop, s, 0.0, build_parameters)
    return B.snapshot(include_hidden=True)


def register_crop(crop, *, shoot_types, ages, steps, tiller_builds=()):
    """``plantarch.<crop>.{load, age0, steps, <age name>..., <tiller build name>...}``."""
    case(f"plantarch.{crop}.load", tier="exact", fuzz=8)(lambda B, s: load_case(B, crop, s, shoot_types))
    case(f"plantarch.{crop}.age0", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s: age0_case(B, crop, s))
    start_age, end_age, step, fuzz = steps
    case(f"plantarch.{crop}.steps", tier="ulp", max_ulp=0, fuzz=fuzz)(
        lambda B, s: steps_case(B, crop, s, start_age, end_age, step))
    for name, age in ages.items():
        case(f"plantarch.{crop}.{name}", tier="ulp", max_ulp=0, fuzz=8)(
            lambda B, s, _a=age: build_case(B, crop, s, _a))
    for name, age, bps in tiller_builds:
        case(f"plantarch.{crop}.{name}", tier="ulp", max_ulp=0, fuzz=len(bps))(
            lambda B, s, _a=age, _b=bps: build_case(B, crop, s, _a, _b[s]))


# ---- wheat ------------------------------------------------------------------------------------------------------
# Oracle schedule (Context seeds 1..8, no tillers): a leaf every 6 days, 8 leaves by ~day 44; the ear (terminal bud,
# WheatSpike.obj) set on day ~42-44, at full size 12 days later; grain fill 39 days; no change after ~day 90. With
# tiller_count 4 the tillers emerge from day ~20 and carry their own ears from day ~50.

#: tiller_count per case seed (non-default structure build parameter; 0 is the default)
WHEAT_TILLERS = [{"tiller_count": n} for n in (1, 2, 3, 4, 4, 3, 2, 1)]

register_crop("wheat", shoot_types=("mainstem", "tiller"),
              ages={"vegetative": 30.0, "heading": 46.0, "ear": 58.0, "ripe": 90.0, "gallery365": 365.0},
              steps=(0.0, 60.0, 2.0, 8),
              tiller_builds=(("tillers_vegetative", 36.0, WHEAT_TILLERS), ("tillers_heading", 56.0, WHEAT_TILLERS),
                             ("tillers_ripe", 100.0, WHEAT_TILLERS)))

WHEAT_REJECTED = [{"tiller_count": 5}, {"tiller_count": 1.5}, {"tiller_count": -1}, {"tassel_length": 0.3},
                  {"tiller_count": float("nan")}]


@case("plantarch.wheat.build_parameters", tier="exact", fuzz=1)
def wheat_build_parameters(B, s):
    return names_and_errors(B, "wheat", WHEAT_REJECTED, [{"tiller_count": 0}, {"tiller_count": 4}])


# ---- sorghum ----------------------------------------------------------------------------------------------------
# Oracle schedule (Context seed 1, no tillers): a leaf every 3.5 days, 16 leaves by ~day 53; the panicle (terminal bud,
# SorghumPaniclePrototype: ~44k triangles and ~6k Context draws per prototype, 3 prototypes) set on day ~53, at full size
# after the 35-day fruit-maturity threshold; no change after ~day 90. With tiller_count 6 the tillers carry their own
# panicles from day ~55 (7 panicles, 535k primitives at day 65).

SORGHUM_TILLERS = [{"tiller_count": n} for n in (1, 2, 3, 4, 5, 6, 3, 6)]

register_crop("sorghum", shoot_types=("mainstem", "tiller"),
              ages={"vegetative": 30.0, "flag_leaf": 50.0, "panicle": 54.0, "grain_fill": 70.0, "mature": 100.0,
                    "gallery365": 365.0},
              steps=(0.0, 50.0, 2.0, 8),
              tiller_builds=(("tillers_vegetative", 40.0, SORGHUM_TILLERS), ("tillers_panicle", 66.0, SORGHUM_TILLERS)))


@case("plantarch.sorghum.panicle_steps", tier="ulp", max_ulp=0, fuzz=4)
def sorghum_panicle_steps(B, s):
    """Days 50..60 in 2-day calls from a direct build at 50 d: the panicle bud appears, its prototypes are built and
    placed, and it starts expanding under advanceTime."""
    return steps_case(B, "sorghum", s, 50.0, 60.0, 2.0)


#: per case seed: a non-default organ build-parameter map for the panicle
SORGHUM_PANICLE_BUILD_PARAMETERS = [
    {"panicle_type": 0.0},
    {"panicle_type": 1.0},
    {"panicle_type": 0.3},
    {"panicle_type": 0.7, "panicle_branch_count": 40, "grain_subdivisions": 5},
    {"panicle_rachis_curvature": 60.0, "grain_count": 600, "panicle_compactness": 0.3},
    {"glume_length_fraction": 0.0, "grain_color_r": 0.95, "panicle_splayed_branch_fraction": 0.5},
    {"grain_subdivisions": 3, "panicle_grains_per_raceme": 9, "panicle_branch_bare_fraction": 0.6, "tiller_count": 2},
    {"panicle_length": 0.4, "panicle_raceme_angle": 70.0, "panicle_branch_upcurve": 60.0, "grain_count": 300},
]


@case("plantarch.sorghum.panicle_build_parameters", tier="ulp", max_ulp=0, fuzz=8)
def sorghum_panicle_build_parameters(B, s):
    return build_case(B, "sorghum", s, 56.0, SORGHUM_PANICLE_BUILD_PARAMETERS[s])


SORGHUM_REJECTED = [{"tiller_count": 7}, {"tiller_count": 2.5}, {"panicle_type": 1.5}, {"grain_count": 50},
                    {"panicle_branch_count": 3.5}, {"grain_subdivisions": 13}, {"glume_color_b": -0.2},
                    {"tassel_length": 0.3}, {"panicle_compactness": float("nan")}]
SORGHUM_ACCEPTED = [{"tiller_count": 6, "panicle_type": 0.2, "grain_count": 1500}]


@case("plantarch.sorghum.build_parameters", tier="exact", fuzz=1)
def sorghum_build_parameters(B, s):
    return names_and_errors(B, "sorghum", SORGHUM_REJECTED, SORGHUM_ACCEPTED)
