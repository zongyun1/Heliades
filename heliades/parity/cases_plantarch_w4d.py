# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the grapevine VSP model (W4-D).

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One vine per
PlantArchitecture, built with ``buildPlantInstanceFromLibrary`` (gv_build).

* ``plantarch.grape.load`` (exact): engine calls after ``PlantArchitecture(ctx)``, after
  ``loadPlantModelFromLibrary("grapevine_VSP")``, after a second load, after an age-0 build with no build
  parameters and after one with fixed build parameters (a constant is taken without a draw: the count
  depends on the parameters only through the geometry they shape);
* ``plantarch.grape.params`` (exact): the five ShootParameters trees (``getCurrentShootParameters``);
* ``plantarch.grape.build_parameters`` (exact): the accepted names and which maps are rejected (unknown key,
  out-of-range structure value, out-of-range organ value, NaN);
* ``plantarch.grape.age0`` / ``age0_build_parameters`` (ulp, 0): ``snapshot(include_hidden=True)`` of the
  age-0 vine (trunk, canes and the hidden leaf prototypes), without / with fixed build parameters;
* ``plantarch.grape.steps`` (ulp, 0): ages 2..40 in 2-day ``advanceTime`` calls; after each: the engine-call
  probe, every shoot's node count, stored internode nodes and radii, and the whole hidden-inclusive snapshot as
  a stream (UUIDs, vertices, colours, UVs, hidden flags, object IDs, members and transforms). The shoots break
  on day 30 and are steered by the catch wires (attraction points) from then on;
* ``plantarch.grape.age76`` (ulp, 0): a direct build at 76 days (fruit-zone leaf pulling has happened,
  no cluster has set yet on any of the 8 seeds): full snapshot plus the probe and shoot summary;
* ``plantarch.grape.age120`` (gated): the gallery age. Clusters set from day ~79, and cluster geometry needs
  the inflorescence machinery (W3-D) and ``GrapevineFruitPrototype``. Until both are ported the case returns
  the gate marker ``[-1]`` on both backends; when they are, it compares the full snapshot.
"""
from __future__ import annotations

import numpy as np

from .cases import case
from .tools.draw_count import engine_calls

f32 = np.float32

MODEL = "grapevine_VSP"
STEP_AGES = tuple(range(2, 41, 2))       # module-level so tests can shorten the sweep
STEP_DAYS = 2.0
FIXED_BUILD_PARAMETERS = {"vine_spacing": 1.8, "trunk_height": 0.6, "berry_diameter": 0.015,
                          "cluster_berry_count": 120.0}
REJECTED_BUILD_PARAMETERS = (
    {"vine_spacing": 2.0, "row_spacing": 3.0},          # not accepted by this model
    {"vine_spacing": 6.0},                              # outside [0.5, 5]
    {"trunk_height": 0.01},                             # outside [0.05, 1]
    {"berry_diameter": 0.5},                            # organ parameter, validated at build time
    {"cluster_berry_count": 5.0},
    {"trunk_height": float("nan")},
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


def snapshot_stream(snap) -> list[float]:
    """Every field of a Snapshot as one stream (UUID and object IDs as numbers, then the floats)."""
    out = [float(len(snap))] + snap.uuid.astype(float).tolist() + snap.type.astype(float).tolist()
    out += snap.nverts.astype(float).tolist() + snap.parent.astype(float).tolist()
    out += snap.hidden.astype(float).tolist()
    pad = np.arange(4)[None, :] >= snap.nverts[:, None]
    v = snap.vertices.copy()
    v[pad] = 0
    out += v.reshape(-1).tolist() + snap.color.reshape(-1).tolist()
    uv = snap.uv.copy()
    uv[np.arange(4)[None, :] >= snap.nuv[:, None]] = 0
    out += snap.nuv.astype(float).tolist() + uv.reshape(-1).tolist() + snap.solid_fraction.tolist()
    o = snap.objects
    out += [float(len(o))] + o.oid.astype(float).tolist() + o.type.astype(float).tolist()
    out += o.hidden.astype(float).tolist()
    for uu in o.uuids:
        out += [float(len(uu))] + np.asarray(uu, float).tolist()
    out += o.transform.reshape(-1).tolist()
    return out


def _structure(B, seed) -> list[float]:
    out = [-201.0, _probe(B, seed), -202.0]
    for n, verts, radii in B.gv_shoot_summary():
        out += [float(n), float(len(verts))] + verts + radii
    return out


@case("plantarch.grape.load", tier="exact", fuzz=8)
def grape_load(B, s):
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


@case("plantarch.grape.params", tier="exact", fuzz=2)
def grape_params(B, s):
    from ..plantarchitecture.structure import canonical
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load(MODEL)
    txt = "|".join(canonical(B.plantarch_shoot_params(label)) for label in (
        "grapevine_trunk", "grapevine_cane", "grapevine_shoot", "grapevine_shoot_mirrored", "grapevine_lateral"))
    return list(txt.encode())


@case("plantarch.grape.build_parameters", tier="exact", fuzz=1)
def grape_build_parameters(B, s):
    B.seed(_seed(s))
    B.plantarch_new()
    B.plantarch_load(MODEL)
    out = list(",".join(B.gv_build_parameter_names()).encode())
    for bp in REJECTED_BUILD_PARAMETERS:
        out.append(B.gv_build_error(bp))
    out.append(B.gv_build_error(FIXED_BUILD_PARAMETERS))
    return out


@case("plantarch.grape.age0", tier="ulp", max_ulp=0, fuzz=8)
def grape_age0(B, s):
    _start(B, s)
    return B.snapshot(True)


@case("plantarch.grape.age0_build_parameters", tier="ulp", max_ulp=0, fuzz=4)
def grape_age0_build_parameters(B, s):
    _start(B, s, 0.0, FIXED_BUILD_PARAMETERS)
    return B.snapshot(True)


@case("plantarch.grape.steps", tier="ulp", max_ulp=0, fuzz=8)
def grape_steps(B, s):
    seed = _start(B, s)
    out = []
    for age in STEP_AGES:
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + _structure(B, seed) + [-203.0] + snapshot_stream(B.snapshot(True))
    return out


@case("plantarch.grape.age76", tier="ulp", max_ulp=0, fuzz=8)
def grape_age76(B, s):
    seed = _start(B, s, 76.0)
    return _structure(B, seed) + [-203.0] + snapshot_stream(B.snapshot(True))


def cluster_geometry_ported() -> bool:
    """Capability gate for ages at which clusters set: W3-D's inflorescence module exists and the grape cluster
    prototype is implemented (the same answer on both backends)."""
    import importlib.util
    if importlib.util.find_spec("heliades.plantarchitecture.inflorescence") is None:
        return False
    from ..plantarchitecture.library import grapevine
    return not getattr(grapevine, "FRUIT_PROTOTYPE_IS_STUB", True)


@case("plantarch.grape.age120", tier="ulp", max_ulp=0, fuzz=8)
def grape_age120(B, s):
    if not cluster_geometry_ported():
        return [-1.0]
    seed = _start(B, s, 120.0)
    return _structure(B, seed) + [-203.0] + snapshot_stream(B.snapshot(True))
