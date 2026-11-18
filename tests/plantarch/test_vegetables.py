# SPDX-License-Identifier: LGPL-2.1-or-later
"""W5-B: herbaceous vegetables (cowpea, cucumber, butterlettuce, sugarbeet, asparagus): unit tests + oracle mutation checks."""
from __future__ import annotations

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import leaf_angle_tracking as LAT
from heliades.plantarchitecture.library import asparagus as AS
from heliades.plantarchitecture.library import butterlettuce as BL
from heliades.plantarchitecture.library import cowpea as CP
from heliades.plantarchitecture.library import cucumber as CU
from heliades.plantarchitecture.library import sugarbeet as SB

f32 = np.float32


def _plant(model, seed=1, age=0.0):
    from heliades.plantarchitecture import leaf_prototype  # noqa: F401
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(model)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age)
    return s, pa, pid


@pytest.mark.parametrize("model,types", [("butterlettuce", {"mainstem"}), ("sugarbeet", {"mainstem"}), ("asparagus", {"main"}),
                                         ("cucumber", {"main"}), ("cowpea", {"unifoliate", "trifoliate"})])
def test_models_registered(model, types):
    pa = PlantArchitecture(Scene())
    pa.loadPlantModelFromLibrary(model)
    assert set(pa.shoot_types) == types


def test_cucumber_string_and_path():
    pts = CU.CucumberStringAttractionPoints((0, 0, 0))
    assert len(pts) == 88 and pts[0][2] == f32(0.05)
    assert CU.cucumberLoweredStemPath(1.0, 0.05) == (f32(0), f32(0.05))
    y, z = CU.cucumberLoweredStemPath(2.0, 1.5)          # on the run along the row, at the laid height
    assert abs(float(z) - 0.3) < 1e-6 and y > 0


def test_cucumber_bump_map_and_tracking_flags():
    s, pa, pid = _plant("cucumber")
    leaf = pa.plant_instances[pid].shoot_tree[0].phytomers[0].leaf_objIDs[0][0]
    u = s.object_primitive_uuids(leaf)[0]
    assert s.get_primitive_normal_map_file(u).endswith("CucumberLeaf_0_bump.png")
    assert pa.plant_instances[pid].attraction_points_enabled
    with pytest.raises(ValueError):
        s.set_primitive_bump_map([u], s.get_primitive_normal_map_file(u), 0.0)


def test_cowpea_tracking_enabled():
    s, pa, pid = _plant("cowpea")
    inst = pa.plant_instances[pid]
    assert inst.leaf_angle_tracking_enabled
    t = inst.leaf_angle_tracker
    assert abs(float(t.bin_target_fraction.sum()) - 1) < 1e-5 and t.lambda_ == f32(np.pi)
    with pytest.raises(Exception):
        LAT.enablePlantLeafElevationAngleDistributionTracking(pa, pid, 0.0, 1.0, 10)


def test_beta_cdf_monotone():
    v = [float(LAT.evaluate_Beta_distribution_CDF(f32(x), 1.398, 1.574)) for x in np.linspace(0, np.pi / 2, 10)]
    assert v[0] == 0 and abs(v[-1] - 1) < 1e-6 and v == sorted(v)


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


MUTATIONS = {
    # the lettuce leaves are pitched by 60 deg * (max - node) / max
    "lettuce_no_leaf_pitch": (BL, None, "_butterlettuce_phytomer_creation",
                              "    phytomer.rotateLeaf(0, 0, make_AxisRotation(f32(-deg2rad(60) * fact), 0, 0))\n",
                              "    pass\n", "plantarch.butterlettuce.age0_full", range(2)),
    # the sugar beet's discarded yaw draw comes first (right to left)
    "sugarbeet_draw_order": (SB, None, "buildSugarbeetPlant",
                             "    yaw = f32(f32(0) * pa.context_ptr.randu(f32(0), f32(2.0 * np.pi)))\n"
                             "    pitch = pa.context_ptr.randu(f32(0), f32(float(f32(0.01)) * np.pi))\n",
                             "    pitch = pa.context_ptr.randu(f32(0), f32(float(f32(0.01)) * np.pi))\n"
                             "    yaw = f32(f32(0) * pa.context_ptr.randu(f32(0), f32(2.0 * np.pi)))\n",
                             "plantarch.sugarbeet.age0_full", range(2)),
    # the cladode is rolled about x by a draw from the prototype's own stream
    "asparagus_no_cladode_roll": (AS, None, "_asparagus_leaf_prototype",
                                  '    context.rotate_object(objID, randu(0, f32(2) * PI_F), "x")\n', "",
                                  "plantarch.asparagus.age0_full", range(2)),
    # blind nodes: the first three cladode whorls at 0.6 scale
    "asparagus_blind_nodes": (AS, None, "_asparagus_phytomer_creation", "    if shoot_node_index <= 2:\n",
                              "    if shoot_node_index <= 1:\n", "plantarch.asparagus.steps", range(2)),
    # the cucumber leaf bump-map strength
    "cucumber_bump_strength": (CU, None, "_cucumber_main_phytomer_creation", "f32(12.58861))", "f32(12.5886))",
                               "plantarch.cucumber.bump_maps", range(2)),
    # fruit picked once full size
    "cucumber_no_harvest": (CU, None, "_cucumber_main_phytomer_callback",
                            "                phytomer.setFloralBudState(BUD_DEAD, fbud)\n", "                pass\n",
                            "plantarch.cucumber.ages", range(16, 18)),
    # lowering and leaning once the head reaches the wire
    "cucumber_no_lowering": (CU, None, "_cucumber_main_phytomer_callback", "    if upright_base == 0:\n",
                             "    if True:\n", "plantarch.cucumber.ages", range(16, 18)),
    # cowpea floral buds from node max(5, max_nodes / 2)
    "cowpea_first_floral_node": (CP, None, "_cowpea_phytomer_creation",
                                 "    first_floral_node = max(5, int(shoot_max_nodes) // 2)\n",
                                 "    first_floral_node = max(5, int(shoot_max_nodes) // 3)\n",
                                 "plantarch.cowpea.steps", range(2)),
    # the target is drawn inside the chosen bin, not at its lower edge
    "tracking_no_within_bin_draw": (LAT, None, "chooseLeafAngleTarget",
                                    "    target_inclination = f32(lower + f32(within_bin(tracker.generator) * f32(upper - lower)))\n",
                                    "    target_inclination = lower\n", "plantarch.cowpea.steps", range(2)),
    # the steering step covers the completed fraction of the remaining growth
    "tracking_full_step": (LAT, None, "updatePlantLeafAngleDistributionTracking",
                           "                    if separation > f32(1e-4) and step_fraction < 1:\n",
                           "                    if False:\n", "plantarch.cowpea.steps", range(2)),
    # the recorded roll is atan2(+y.z, z.z) (written to the XML)
    "setleafnormal_roll_sign": (LAT, None, "setLeafNormal", "    recorded_roll = atan2f(axis_y[2], axis_z[2])\n",
                                "    recorded_roll = atan2f(-axis_y[2], axis_z[2])\n", "plantarch.cowpea.steps", range(2)),
}

SLOW_MUTATIONS = {"cucumber_no_harvest", "cucumber_no_lowering", "cowpea_first_floral_node", "tracking_no_within_bin_draw",
                  "tracking_full_step", "setleafnormal_roll_sign", "asparagus_blind_nodes"}


@pytest.mark.parametrize("mutation", [pytest.param(m, marks=pytest.mark.slow) if m in SLOW_MUTATIONS else m
                                      for m in sorted(MUTATIONS)])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from tests.plantarch.test_strawberry_maize import _patch_source
    from heliades.parity.ops import load_plugins
    from heliades.parity.suites.plantarch import cases_vegetables as CV
    load_plugins()
    module, owner, attr, old, new, case_name, seeds = MUTATIONS[mutation]
    for crop in CV.STEP_END:
        monkeypatch.setitem(CV.STEP_END, crop, 16)
    _patch_source(monkeypatch, module, owner, attr, old, new)
    if attr.startswith("build"):                       # the registry holds the builder itself
        from heliades.plantarchitecture import library
        for name, fn in list(library.PLANT_BUILDERS.items()):
            if getattr(fn, "__name__", "") == attr:
                monkeypatch.setitem(library.PLANT_BUILDERS, name, getattr(module, attr))
    assert _fails(case_name, seeds)
