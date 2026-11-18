# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for the library rice (W5-A). All at 0 float32 ulp (``exact`` for names / error texts).

Oracle schedule (Context seeds 1..8, default ``tiller_count`` 12): a main-stem leaf every 6 days, tillers from day
~24 (13 culms, 93 leaves by day 78); panicles (``RiceFloweringPaniclePrototype``, heading stage, closed flower) on day
~78, open (anthesis stage, with anthers) ~4 days later, set as ripening panicles (``RicePaniclePrototype``) ~3 days after
that; no change after ~day 90. Leaf texture stages: every leaf older than 55 / 72 days shows the second / third image
of its series (``Phytomer::updateLeafTextureStage`` -> ``Context::setPrimitiveColorTexture``); each observation also
compares the color texture of every leaf primitive.

Cases: ``load``, ``age0``, ``steps`` (2..60 d in 6-day calls, 4 seeds: tillering), direct builds at 40 (tillers),
66 (first stage-1 leaves), 79 (heading), 83 (anthesis), 90 (ripening), 120 (stage-2 leaves, mature) and 365 (the age
the Helios gallery renders: render_plant_library's last full-leaf age with fruit, the plant no longer changing);
``tillers_*``: non-default ``tiller_count`` (0..18) at 50 and 90 d; ``panicle_build_parameters``: non-default organ
parameters at 90 d (ripening) and 83 d (anthesis); ``build_parameters``: accepted names and error texts.
"""
from __future__ import annotations

from ...cases import case
from .cases_cereals import build_case, names_and_errors, observe, register_crop, snapshot_stream, start, steps_case

CROP = "rice"


def _leaf_textures(B):
    return B.rice_color_textures([oid for oid, _T, _v in B.pa_leaf_objects(False)])


def rice_build_case(B, s, age, build_parameters=None):
    out = build_case(B, CROP, s, age, build_parameters)
    return out + [-300.0] + _leaf_textures(B)


register_crop(CROP, shoot_types=("mainstem", "tiller"), ages={}, steps=(0.0, 60.0, 6.0, 4))

RICE_AGES = {"tillering": 40.0, "stage1": 66.0, "heading": 79.0, "anthesis": 83.0, "ripening": 90.0,
             "mature": 120.0, "gallery365": 365.0}
for _name, _age in RICE_AGES.items():
    case(f"plantarch.rice.{_name}", tier="ulp", max_ulp=0, fuzz=8)(lambda B, s, _a=_age: rice_build_case(B, s, _a))


#: tiller_count per case seed (12 is the default)
RICE_TILLERS = [{"tiller_count": n} for n in (0, 1, 4, 7, 9, 13, 16, 18)]

case("plantarch.rice.tillers_vegetative", tier="ulp", max_ulp=0, fuzz=8)(
    lambda B, s: rice_build_case(B, s, 50.0, RICE_TILLERS[s]))
case("plantarch.rice.tillers_ripening", tier="ulp", max_ulp=0, fuzz=8)(
    lambda B, s: rice_build_case(B, s, 90.0, RICE_TILLERS[s]))

#: per case seed: non-default panicle (organ) build parameters; seeds 0..3 built at 90 d (ripening panicles),
#: 4..7 at 83 d (anthesis: open flowering panicles with anthers)
RICE_PANICLE_BUILD_PARAMETERS = [
    {"panicle_type": 0.0, "tiller_count": 2},
    {"panicle_type": 1.0, "tiller_count": 2},
    {"panicle_type": 0.3, "ripeness": 0.4, "tiller_count": 1},
    {"branch_count": 6, "spikelets_per_secondary": 5, "spikelet_subdivisions": 4, "pedicel_length": 0.0, "tiller_count": 0},
    {"panicle_type": 0.7, "anther_fraction": 1.0, "tiller_count": 1},
    {"anther_fraction": 0.0, "hull_green_color_g": 0.8, "tiller_count": 0},
    {"filament_length": 0.0, "secondary_branches_per_primary": 0.0, "spikelet_subdivisions": 3, "tiller_count": 1},
    {"rachis_droop": 170.0, "branch_hang": 1.0, "anther_color_r": 1.0, "tiller_count": 0},
]


@case("plantarch.rice.panicle_build_parameters", tier="ulp", max_ulp=0, fuzz=8)
def rice_panicle_build_parameters(B, s):
    return rice_build_case(B, s, 90.0 if s < 4 else 83.0, RICE_PANICLE_BUILD_PARAMETERS[s])


RICE_REJECTED = [
    {"tiller_count": 19}, {"tiller_count": 2.5}, {"tassel_length": 0.3}, {"panicle_type": 1.5},
    {"branch_count": 0}, {"spikelet_subdivisions": 13}, {"hull_ripe_color_b": -0.1}, {"ripeness": float("nan")},
]
RICE_ACCEPTED = [{"tiller_count": 0}, {"tiller_count": 18, "panicle_type": 0.2, "branch_count": 15}]


@case("plantarch.rice.build_parameters", tier="exact", fuzz=1)
def rice_build_parameters(B, s):
    return names_and_errors(B, CROP, RICE_REJECTED, RICE_ACCEPTED)


__all__ = ["observe", "snapshot_stream", "start", "steps_case"]
