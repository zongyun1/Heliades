# SPDX-License-Identifier: LGPL-2.1-or-later
"""W5-A: the library rice (unit tests + oracle mutation checks)."""
from __future__ import annotations

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture, hooks
from heliades.plantarchitecture import shoot as SH
from heliades.plantarchitecture.library import rice as RC

from test_strawberry_maize import _patch_source  # noqa: E402  (tests/plantarch is on sys.path under pytest)

f32 = np.float32


def _plant(seed=1, age=0.0, build_parameters=None):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("rice")
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age, build_parameters)
    return s, pa, pid


def test_rice_registered_and_hooks():
    s = Scene()
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("rice")
    assert set(pa.shoot_types) == {"mainstem", "tiller"}
    assert hooks.RicePaniclePrototype.impl is RC.RicePaniclePrototype
    assert hooks.RiceFloweringPaniclePrototype.impl is RC.RiceFloweringPaniclePrototype
    from heliades.plantarchitecture.library.build_parameters import getPlantModelBuildParameterNames
    names = getPlantModelBuildParameterNames(pa, "rice")
    assert names[0] == "tiller_count" and "panicle_type" in names and len(names) == 1 + 2 + 28 + 2 + 9


def test_rice_tables():
    assert RC.riceBladeLengthFraction(13, 14) == f32(0.78) and RC.riceBladeLengthFraction(12, 14) == f32(1)
    assert RC.riceBladeLengthFraction(0, 14) == max(f32(0.2), f32(f32(0.52) - f32(f32(0.1) * f32(6))))
    assert RC.riceCulmInternodeLength(0, 14) == f32(0.003)
    assert RC.riceTillerCount({}) == 12


def test_panicle_presets_and_validation():
    p = RC.RicePanicleParameters.fromBuildParameters({"panicle_type": 0.25})
    assert p.primary_branch_count == int(RC.GI.roundf(f32(f32(f32(0.5) * f32(13)) + f32(f32(0.5) * f32(10)))))
    with pytest.raises(Exception):
        RC.RicePanicleParameters.fromBuildParameters({"spikelet_subdivisions": 2})
    s = Scene()
    s.seed_random_generator(3)
    oid = RC.buildRicePanicle(s, RC.RicePanicleParameters(), RC.ANTHESIS)
    assert s.object_primitive_count(oid) > 1000
    bad = RC.RicePanicleParameters()
    bad.secondary_zone_fraction = f32(0)
    with pytest.raises(Exception, match="secondary_zone_fraction"):
        RC.buildRicePanicle(s, bad, RC.RIPENING)


def test_leaf_texture_stages():
    s, pa, pid = _plant(age=66)
    staged = [u for u, p in s.primitives.items() if getattr(p, "color_texture_file", "")]
    assert staged and all(s.primitive_color_texture_file(u).endswith("_1.png") for u in staged)
    assert all(s.primitive_texture_file(u).endswith("_0.png") for u in staged)


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


MUTATIONS = {
    # tiller positions: the primary tiller of leaf n counted from leaf 1 instead of leaf 2
    "rice_tiller_positions": (RC, None, "RicePhytomerCreationFunction",
                              "        bud_makes_tiller = any(t == 0 and m == shoot_node_index + 1 for m, t in positions)\n",
                              "        bud_makes_tiller = any(t == 0 and m == shoot_node_index + 2 for m, t in positions)\n",
                              "plantarch.rice.tillering"),
    # the crown leaves leave the culm at randu(3, 12) degrees
    "rice_crown_leaf_angle": (RC, None, "RicePhytomerCreationFunction",
                              "        setGrassFlagLeafAngle(phytomer, ctx.randu(f32(3), f32(12)))\n",
                              "        setGrassFlagLeafAngle(phytomer, ctx.randu(f32(3), f32(13)))\n",
                              "plantarch.rice.tillering"),
    # the anther scatter make_vec3(0, randu, randu) is drawn right to left (z first)
    "rice_anther_scatter_left_to_right": (RC, None, "buildRicePanicle",
                                          "                sz = ctx.randu(f32(-0.35), f32(0.35))\n"
                                          "                sy = ctx.randu(f32(-0.35), f32(0.35))\n",
                                          "                sy = ctx.randu(f32(-0.35), f32(0.35))\n"
                                          "                sz = ctx.randu(f32(-0.35), f32(0.35))\n",
                                          "plantarch.rice.anthesis"),
    # the heading-stage panicle holds its branches at 0.35 of the ripe spread
    "rice_heading_spread": (RC, None, "buildRicePanicle",
                            "droop_fraction, hang_fraction, spread_fraction = f32(0.1), f32(0.1), f32(0.35)",
                            "droop_fraction, hang_fraction, spread_fraction = f32(0.1), f32(0.1), f32(0.4)",
                            "plantarch.rice.heading"),
}

# Not distinguishable (left out): drawing the base yaw before the roll (``make_AxisRotation(0, yaw, roll)`` with
# pitch 0 turns the upright culm about z twice, so swapping the two draws gives the same plant).
SLOW_MUTATIONS = {"rice_anther_scatter_left_to_right", "rice_heading_spread"}   # ~45 s each


@pytest.mark.parametrize("mutation", [pytest.param(m, marks=pytest.mark.slow) if m in SLOW_MUTATIONS else m
                                      for m in sorted(MUTATIONS)])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    module, owner, attr, old, new, case_name = MUTATIONS[mutation]
    _patch_source(monkeypatch, module, owner, attr, old, new)
    assert _fails(case_name)


@pytest.mark.slow
def test_mutation_no_leaf_texture_stage(monkeypatch):
    """Phytomer::updateLeafTextureStage: without the stage images on the blades, plantarch.rice.stage1 fails."""
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    monkeypatch.setattr(SH.Shoot, "_update_leaf_texture_stage", staticmethod(lambda phytomer: None))
    assert _fails("plantarch.rice.stage1")
