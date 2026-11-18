# SPDX-License-Identifier: LGPL-2.1-or-later
"""W5-C: grapevine_Wye (trunk, arms and four cordons trained to a Wye trellis, gable wires, build parameters)."""
from __future__ import annotations

import types

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import library as L
from heliades.plantarchitecture.library import build_parameters as BP
from heliades.plantarchitecture.library import grapevine_wye as GW
from heliades.plantarchitecture.params import BUD_ACTIVE, BUD_DEAD
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

f32 = np.float32


def _vine(seed=1, age=0.0, build_parameters=None):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("grapevine_Wye")
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age, build_parameters)
    return s, pa, pid


# ---- unit checks -----------------------------------------------------------------------------------

def test_registered():
    s = Scene()
    pa = PlantArchitecture(s)
    assert "grapevine_Wye" in pa.getAvailablePlantModels()
    assert BP.getPlantModelBuildParameterNames(pa, "grapevine_Wye") == [
        "trunk_height", "cordon_spacing", "vine_spacing", "catch_wire_height", "berry_diameter", "cluster_berry_count"]
    # VSP is unchanged
    assert BP.getPlantModelBuildParameterNames(pa, "grapevine_VSP") == ["vine_spacing", "trunk_height",
                                                                        "berry_diameter", "cluster_berry_count"]


def test_catch_wire_must_be_above_cordon_wire():
    s = Scene()
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("grapevine_Wye")
    with pytest.raises(PlantArchitectureError, match="catch_wire_height"):
        pa.buildPlantInstanceFromLibrary((0, 0, 0), 0.0, {"trunk_height": 1.2, "catch_wire_height": 1.2})
    for bad in ({"trunk_height": 0.3}, {"cordon_spacing": 2.5}, {"catch_wire_height": 4.5}):
        with pytest.raises(PlantArchitectureError):
            pa.buildPlantInstanceFromLibrary((0, 0, 0), 0.0, bad)


def test_wood_dome():
    pos = [np.array([0, 0, 0], f32), np.array([0, 0, 1], f32)]
    radii = [f32(0.1), f32(0.1)]
    GW.appendWoodDome(pos, radii)
    assert len(pos) == 5 and float(radii[-1]) < 0.01 and all(float(p[2]) > 1 for p in pos[2:])
    assert float(pos[-1][2]) < 1.1


def test_age0_structure():
    s, pa, pid = _vine(1)
    tree = pa.plant_instances[pid].shoot_tree
    T, C = "grapevine_trunk", "grapevine_cordon"
    assert [sh.shoot_type_label for sh in tree] == [T, T, C, C, T, C, C]     # trunk, then each arm and its two cordons
    assert [sh.parent_shoot_ID for sh in tree] == [-1, 0, 1, 1, 0, 4, 4]
    cordons = [sh for sh in tree if sh.shoot_type_label == C]
    assert all(len(sh.phytomers) == 11 for sh in cordons)   # int(0.9 / 0.08) cordon nodes at the default spacing
    assert all(not sh.meristem_is_alive and not sh.isdormant for sh in tree)
    assert all(not ph.hasLeaf() for sh in tree for ph in sh.phytomers)
    assert all(b.state == BUD_DEAD for sh in tree if sh.shoot_type_label == T for ph in sh.phytomers
               for p in ph.axillary_vegetative_buds for b in p)
    assert all(b.state == BUD_ACTIVE for sh in cordons for ph in sh.phytomers for p in ph.axillary_vegetative_buds
               for b in p)
    inst = pa.plant_instances[pid]
    # 4 gable levels (round(0.6 / 0.2) + 1) x 2 sides x int(1.8 / 0.06) samples
    assert inst.attraction_points_enabled and inst.attraction_points.shape == (4 * 2 * 30, 3)
    # the cordons run along the row (y), on the cordon wires at x = +-0.3, z = 1.4
    for sh in cordons:
        tip = np.asarray(sh.shoot_internode_vertices[-1][-1])
        assert abs(abs(float(tip[0])) - 0.3) < 0.05 and abs(float(tip[2]) - 1.4) < 0.05 and abs(float(tip[1])) > 0.6


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


def _patch_source(monkeypatch, module, attr, old, new):
    src = open(module.__file__).read()
    assert src.count(old) == 1, old
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
    exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
    obj = getattr(mod, attr)
    monkeypatch.setattr(module, attr, obj)
    return obj


SLOW_MUTATIONS = {"gable_spread", "cordon_buds_dead"}


@pytest.mark.parametrize("mutation", [pytest.param(m, marks=pytest.mark.slow) if m in SLOW_MUTATIONS else m
                                      for m in ["arm_bow_sign", "dome_angles", "arm_draws_swapped", "gable_spread",
                                                "cordon_buds_dead"]])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    from heliades.parity.suites.plantarch import cases_grapevine_wye as CW
    load_plugins()
    monkeypatch.setattr(CW, "STEP_AGES", tuple(range(2, 37, 2)))

    def builder(old, new):
        fn = _patch_source(monkeypatch, GW, "buildGrapevineWye", old, new)
        monkeypatch.setitem(L.PLANT_BUILDERS, "grapevine_Wye", fn)

    if mutation == "arm_bow_sign":
        # the arm bows downward (node_position.z -= arm_bow * sin(pi * along))
        builder("node_position[2] = f32(node_position[2] - f32(arm_bow * s))",
                "node_position[2] = f32(node_position[2] + f32(arm_bow * s))")
        assert _fails("plantarch.grapewye.age0", range(1))
    elif mutation == "dome_angles":
        # appendWoodDome closes the arm at 30, 60, 85 degrees
        monkeypatch.setattr(GW, "_DOME_ANGLES_DEGREES", (f32(30), f32(60), f32(80)))
        assert _fails("plantarch.grapewye.age0", range(1))
    elif mutation == "arm_draws_swapped":
        # arm_bow is drawn before arm_radius
        builder("        arm_bow = f32(ctx.randu(f32(0.05), f32(0.15)) * arm_length)\n"
                "        arm_radius = f32(f32(f32(0.55) * head_radius) * ctx.randu(f32(0.9), f32(1.1)))\n",
                "        arm_radius = f32(f32(f32(0.55) * head_radius) * ctx.randu(f32(0.9), f32(1.1)))\n"
                "        arm_bow = f32(ctx.randu(f32(0.05), f32(0.15)) * arm_length)\n")
        assert _fails("plantarch.grapewye.age0", range(1))
    elif mutation == "gable_spread":
        # the gable wires lean out 0.6 m per metre of rise (only the growth they steer shows it)
        builder("gable_spread_per_rise = f32(0.6)", "gable_spread_per_rise = f32(0.4)")
        assert _fails("plantarch.grapewye.steps", range(1))
    elif mutation == "cordon_buds_dead":
        # only the cordons bear shoots: their buds are woken
        builder('bears_shoots = trained.shoot_type_label == "grapevine_cordon"', "bears_shoots = False")
        assert _fails("plantarch.grapewye.steps", range(1))
