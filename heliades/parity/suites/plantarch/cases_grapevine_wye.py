# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the grapevine on a Wye trellis (``grapevine_Wye``, W5-C).

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One vine per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` (``gv_build``, W4-D). Observations as for
``grapevine_VSP`` (``cases_grapevine``): the engine-call probe, every shoot's node count, stored internode nodes
and radii, and ``snapshot(include_hidden=True)`` (as a Snapshot or as a stream).

* ``plantarch.grapewye.load`` (exact): engine calls after ``PlantArchitecture(ctx)``, the load, a second load,
  an age-0 build, an age-0 build with fixed build parameters;
* ``plantarch.grapewye.params`` (exact): the five ShootParameters trees;
* ``plantarch.grapewye.build_parameters`` (exact): the accepted names, and which maps are rejected (unknown key,
  each structure parameter out of range, an organ value out of range, NaN, and the builder's own
  ``catch_wire_height <= trunk_height`` check);
* ``plantarch.grapewye.age0`` (ulp 0): the trained wood (trunk, two arms, four cordons, hidden head / arm-top
  nodes, arm domes) and the hidden leaf prototypes;
* ``plantarch.grapewye.age0_build_parameters`` (ulp 0, 4 Context seeds x 2 parameter sets): the same with
  non-default trellis dimensions (a wide, low trellis with a long cordon; the smallest accepted trellis);
* ``plantarch.grapewye.steps`` (ulp 0): ages 2..40 in 2-day ``advanceTime`` calls, each observed with the full
  snapshot stream. The cordon buds break on day 30 and the gable wires steer the shoots from then on;
* ``plantarch.grapewye.age76`` (ulp 0): a direct build at 76 days: fruit-zone leaves pulled, no cluster set;
* ``plantarch.grapewye.age84`` (ulp 0, Context seeds 1..2): a few days after the clusters set;
* ``plantarch.grapewye.age100`` (ulp 0): clusters set and growing;
* ``plantarch.grapewye.age120`` (ulp 0): the Helios selfTest reference age (866k primitives). It took ~50 min per
  build on Heliades before P1 (W5-C note) and ~20 s since;
* ``plantarch.grapewye.small_age150`` (ulp 0, Context seeds 1..2): past fruit maturity (fruit set + 45 days), on
  a vine of the smallest accepted vine spacing (0.5 m).
"""
from __future__ import annotations

from ...cases import case
from .cases_grapevine import cluster_geometry_ported, snapshot_stream
from ...tools.draw_count import engine_calls

MODEL = "grapevine_Wye"
SHOOT_TYPES = ("grapevine_trunk", "grapevine_cordon", "grapevine_shoot", "grapevine_shoot_mirrored", "grapevine_lateral")
STEP_AGES = tuple(range(2, 41, 2))       # module-level so tests can shorten the sweep
STEP_DAYS = 2.0
FIXED_BUILD_PARAMETERS = {"trunk_height": 0.8, "cordon_spacing": 1.6, "vine_spacing": 3.3, "catch_wire_height": 2.1,
                          "berry_diameter": 0.015, "cluster_berry_count": 120.0}
NON_DEFAULT_BUILD_PARAMETERS = (
    FIXED_BUILD_PARAMETERS,
    {"trunk_height": 0.5, "cordon_spacing": 0.2, "vine_spacing": 0.5, "catch_wire_height": 0.55},
)
REJECTED_BUILD_PARAMETERS = (
    {"vine_spacing": 2.0, "row_spacing": 3.0},          # not accepted by this model
    {"trunk_height": 0.3},                              # outside [0.5, 2]
    {"cordon_spacing": 2.5},                            # outside [0.2, 2]
    {"vine_spacing": 6.0},                              # outside [0.5, 5]
    {"catch_wire_height": 4.5},                         # outside [0.5, 4]
    {"trunk_height": 1.2, "catch_wire_height": 1.0},    # builder: catch wire not above the cordon wire
    {"trunk_height": 1.2, "catch_wire_height": 1.2},
    {"trunk_height": 1.9},                              # default catch wire height 2.5: accepted
    {"berry_diameter": 0.5},                            # organ parameter
    {"cordon_spacing": float("nan")},
)


def _seed(s):
    return s + 1


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1.0 if n is None else float(n)


def _start(B, s, age=0.0, build_parameters=None):
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load(MODEL)
    B.gv_build(age, build_parameters)
    return seed


def _structure(B, seed) -> list[float]:
    out = [-201.0, _probe(B, seed), -202.0]
    for n, verts, radii in B.gv_shoot_summary():
        out += [float(n), float(len(verts))] + verts + radii
    return out


def _full(B, seed) -> list[float]:
    return _structure(B, seed) + [-203.0] + snapshot_stream(B.snapshot(True))


@case("plantarch.grapewye.load", tier="exact", fuzz=8)
def grapewye_load(B, s):
    seed = _seed(s)
    out = []
    for stage in range(6):
        B.seed(seed)
        if stage >= 1:
            B.plantarch_new()
        if stage >= 2:
            B.plantarch_load(MODEL)
        if stage == 3:
            B.plantarch_load(MODEL)
        if stage == 4:
            B.gv_build(0.0, None)
        if stage == 5:
            B.gv_build(0.0, FIXED_BUILD_PARAMETERS)
        out.append(_probe(B, seed))
    return out


@case("plantarch.grapewye.params", tier="exact", fuzz=2)
def grapewye_params(B, s):
    from ....plantarchitecture.structure import canonical
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load(MODEL)
    return list("|".join(canonical(B.plantarch_shoot_params(label)) for label in SHOOT_TYPES).encode())


@case("plantarch.grapewye.build_parameters", tier="exact", fuzz=1)
def grapewye_build_parameters(B, s):
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load(MODEL)
    out = list(",".join(B.gvw_build_parameter_names(MODEL)).encode())
    for bp in REJECTED_BUILD_PARAMETERS:
        out.append(B.gv_build_error(bp))
    out.append(B.gv_build_error(FIXED_BUILD_PARAMETERS))
    return out


@case("plantarch.grapewye.age0", tier="ulp", max_ulp=0, fuzz=8)
def grapewye_age0(B, s):
    _start(B, s)
    return B.snapshot(True)


@case("plantarch.grapewye.age0_build_parameters", tier="ulp", max_ulp=0, fuzz=8)
def grapewye_age0_build_parameters(B, s):
    seed = _start(B, s % 4, 0.0, NON_DEFAULT_BUILD_PARAMETERS[s // 4])
    return _full(B, seed)


@case("plantarch.grapewye.steps", tier="ulp", max_ulp=0, fuzz=8)
def grapewye_steps(B, s):
    seed = _start(B, s)
    out = []
    for age in STEP_AGES:
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + _full(B, seed)
    return out


@case("plantarch.grapewye.age76", tier="ulp", max_ulp=0, fuzz=8)
def grapewye_age76(B, s):
    seed = _start(B, s, 76.0)
    return _full(B, seed)


def _clusters(B, s, age):
    if not cluster_geometry_ported():
        return [-1.0]
    seed = _start(B, s, age)
    return _full(B, seed)


@case("plantarch.grapewye.age84", tier="ulp", max_ulp=0, fuzz=2)
def grapewye_age84(B, s):
    return _clusters(B, s, 84.0)


@case("plantarch.grapewye.age100", tier="ulp", max_ulp=0, fuzz=8)
def grapewye_age100(B, s):
    """The clusters set and growing (866k primitives for Context seed 1, ~230 s on Heliades)."""
    return _clusters(B, s, 100.0)


@case("plantarch.grapewye.age120", tier="ulp", max_ulp=0, fuzz=8)
def grapewye_age120(B, s):
    """The Helios selfTest reference age, Context seeds 1..8 (866k primitives; ~20 s per build on Heliades since
    the P1 structure-of-arrays core, ~50 min before)."""
    return _clusters(B, s, 120.0)


#: a one-row-metre vine (cordons of 3 nodes, 12 shoots): the stages after the reference age at an affordable size
SMALL_VINE = {"vine_spacing": 0.5}


@case("plantarch.grapewye.small_age150", tier="ulp", max_ulp=0, fuzz=2)
def grapewye_small_age150(B, s):
    """Past fruit maturity (fruit set + 45 days) on the small vine."""
    if not cluster_geometry_ported():
        return [-1.0]
    seed = _start(B, s, 150.0, SMALL_VINE)
    return _full(B, seed)
