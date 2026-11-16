# SPDX-License-Identifier: LGPL-2.1-or-later
"""W4-D: grapevine_VSP (trained trunk and canes, VSP catch wires, build parameters)."""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import phytomer as PH
from heliades.plantarchitecture import growth as G
from heliades.plantarchitecture import library as L
from heliades.plantarchitecture.library import build_parameters as BP
from heliades.plantarchitecture.library import grapevine as GV
from heliades.plantarchitecture.params import BUD_ACTIVE, BUD_DEAD
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

f32 = np.float32


def _vine(seed=1, age=0.0, build_parameters=None):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("grapevine_VSP")
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age, build_parameters)
    return s, pa, pid


# ---- unit checks -----------------------------------------------------------------------------------

def test_registered():
    s = Scene()
    pa = PlantArchitecture(s)
    assert "grapevine_VSP" in pa.getAvailablePlantModels()
    assert BP.getPlantModelBuildParameterNames(pa, "grapevine_VSP") == ["vine_spacing", "trunk_height",
                                                                        "berry_diameter", "cluster_berry_count"]
    assert BP.getPlantModelBuildParameterNames(pa, "tomato") == []


def test_bounded_wander_is_bounded():
    w = GV.BoundedWander()
    w.addComponent(0.01, 0.5, 1.0)
    w.addComponent(0.002, 0.2, 0.3)
    vals = [abs(float(w.evaluate(d))) for d in np.linspace(0, 10, 500)]
    assert max(vals) <= 0.012 + 1e-6 and max(vals) > 0.005


def test_linspace_matches_helios_rule():
    pts = GV._linspace3((0, -1.2, 1.0), (0, 1.2, 1.0), 40)
    assert len(pts) == 40 and float(pts[-1][1]) == float(f32(1.2)) and float(pts[0][1]) == float(f32(-1.2))


def test_build_parameter_validation():
    s = Scene()
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("grapevine_VSP")
    for bad in ({"row_spacing": 3.0}, {"vine_spacing": 6.0}, {"berry_diameter": 0.5}, {"trunk_height": float("nan")}):
        with pytest.raises(PlantArchitectureError):
            pa.buildPlantInstanceFromLibrary((0, 0, 0), 0.0, bad)
    # a distribution over the whole range is checked up front; a constant only when it is read
    with pytest.raises(PlantArchitectureError):
        BP.validateBuildParameters(pa, "grapevine_VSP", {"vine_spacing": BP.BuildParameterDistribution.uniform(1, 6)},
                                   "test")
    assert BP.validateBuildParameters(pa, "grapevine_VSP",
                                      {"vine_spacing": BP.BuildParameterDistribution.uniform(1, 3)}, "test") == {}


def test_constant_build_parameters_draw_nothing():
    s = Scene()
    s.seed_random_generator(3)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("grapevine_VSP")
    x = pa.generator.x
    plant, organ = BP.prepare_build_parameters(pa, "grapevine_VSP", {"vine_spacing": 1.8, "berry_diameter": 0.01})
    assert pa.generator.x == x
    assert plant == {"berry_diameter": f32(0.01), "vine_spacing": f32(1.8)} and list(organ) == ["berry_diameter"]


def test_uniform_build_parameter_draws_once():
    s = Scene()
    s.seed_random_generator(3)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("grapevine_VSP")
    x = pa.generator.x
    plant, _ = BP.prepare_build_parameters(pa, "grapevine_VSP",
                                           {"vine_spacing": BP.BuildParameterDistribution.uniform(1.0, 3.0)})
    assert pa.generator.x != x and 1.0 <= float(plant["vine_spacing"]) <= 3.0


def test_age0_structure():
    s, pa, pid = _vine(1)
    tree = pa.plant_instances[pid].shoot_tree
    assert [sh.shoot_type_label for sh in tree] == ["grapevine_trunk", "grapevine_cane", "grapevine_cane"]
    assert [len(sh.phytomers) for sh in tree] == [21, 12, 12]
    assert all(sh.growth_state.geometry_is_prescribed and not sh.meristem_is_alive and not sh.isdormant for sh in tree)
    assert all(not ph.hasLeaf() for sh in tree for ph in sh.phytomers)
    assert all(b.state == BUD_DEAD for ph in tree[0].phytomers for p in ph.axillary_vegetative_buds for b in p)
    assert all(b.state == BUD_ACTIVE for sh in tree[1:] for ph in sh.phytomers for p in ph.axillary_vegetative_buds
               for b in p)
    inst = pa.plant_instances[pid]
    assert inst.attraction_points_enabled and inst.attraction_points.shape == (8 * 40, 3)
    # the trunk is textured bark, the canes are coloured tubes
    assert s.primitive_texture_file(s.object_primitive_uuids(tree[0].internode_tube_objID)[0]).endswith("GrapeBark.jpg")


def test_cluster_prototype_builds():
    from heliades.parity.cases_plantarch_w4d import cluster_geometry_ported
    assert cluster_geometry_ported()
    s = Scene()
    s.seed_random_generator(5)
    oid = GV.GrapevineFruitPrototype(s, 1, {})
    uu = s.object_primitive_uuids(oid)
    assert len(uu) % (2 * 8 * 7) == 0 and 75 * 112 <= len(uu) <= 105 * 112     # 8-division spheres, 75..105 berries
    assert all(s.get_primitive_data(u, "object_label") == "fruit" for u in uu[:5])


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


def _patch_source(monkeypatch, module, owner, attr, old, new):
    """Recompile ``module`` with one source edit; install ``attr`` (of class ``owner``, or the module itself
    when ``owner`` is None) from the edited copy."""
    src = open(module.__file__).read()
    assert src.count(old) == 1, old
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
    exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
    obj = getattr(mod, attr) if owner is None else getattr(getattr(mod, owner), attr)
    monkeypatch.setattr(module if owner is None else getattr(module, owner), attr, obj)
    return obj


@pytest.mark.parametrize("mutation", ["wander_left_to_right", "prescribed_radius_constant",
                                      "no_attraction", "cordon_buds_left_dormant",
                                      "no_leaf_pulling", "berry_scatter_left_to_right"])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity import cases_plantarch_w4d as CW
    from heliades.parity.ops import load_plugins
    load_plugins()
    monkeypatch.setattr(CW, "STEP_AGES", tuple(range(2, 37, 2)))
    if mutation == "wander_left_to_right":
        # addComponent(randu(amplitude), randu(wavelength), randu(phase)) is evaluated right to left (GCC)
        def left_to_right(ctx, wander, amp_range, wl_range, phase_range):
            a = ctx.randu(f32(amp_range[0]), f32(amp_range[1]))
            w = ctx.randu(f32(wl_range[0]), f32(wl_range[1]))
            p = ctx.randu(f32(phase_range[0]), f32(phase_range[1]))
            wander.addComponent(a, w, p)
        monkeypatch.setattr(GV, "_add_component", left_to_right)
        assert _fails("plantarch.grape.age0", range(2))
    elif mutation == "bark_uv_no_wrap":
        # the bark texture coordinate wraps every 0.25 m of shoot length
        _patch_source(monkeypatch, PH, "Phytomer", "__init__", "uv_y.append(f32(t - f32(np.floor(t))))",
                      "uv_y.append(t)")
        assert _fails("plantarch.grape.age0", range(2))
    elif mutation == "prescribed_radius_constant":
        # a prescribed internode interpolates its radius from base to tip
        _patch_source(monkeypatch, PH, "Phytomer", "__init__",
                      "phytomer_internode_radii[inode_segment] = f32(br + f32(segment_fraction * f32(tr - br)))",
                      "phytomer_internode_radii[inode_segment] = br")
        assert _fails("plantarch.grape.age0", range(2))
    elif mutation == "no_attraction":
        # the catch wires steer the shoots (setPlantAttractionPoints, weight 0.15)
        monkeypatch.setattr(PH, "calculateAttractionPointDirection",
                            lambda *a: (np.zeros(3, f32), False))
        assert _fails("plantarch.grape.steps", range(2))
    elif mutation == "no_forced_context_update":
        # an attraction-steered sub-step pushes the shoot nodes to the Context (force_update_on_collision)
        _patch_source(monkeypatch, G, None, "_advance_plant_one_step",
                      "if pa.collision_avoidance_applied and pa.force_update_on_collision:", "if False:")
        assert _fails("plantarch.grape.steps", range(2))
    elif mutation == "cordon_buds_left_dormant":
        # the builder wakes every cordon bud (ACTIVE, not the 0.9 sampled break)
        src_old = "phytomer.setVegetativeBudState(BUD_ACTIVE if bears_shoots else BUD_DEAD, vbud)"
        fn = _patch_source(monkeypatch, GV, None, "buildGrapevineVSP", src_old, "pass")
        monkeypatch.setitem(L.PLANT_BUILDERS, "grapevine_VSP", fn)
        assert _fails("plantarch.grape.steps", range(2))
    elif mutation == "no_leaf_pulling":
        # fruit-zone leaves are pulled once the shoot is 45 days old
        monkeypatch.setattr(GV, "grapevine_leaf_pulling_shoot_age", f32(1e6))
        assert _fails("plantarch.grape.age76", range(1))
    elif mutation == "berry_scatter_left_to_right":
        # make_vec3(randn(), randn(), randn()) for a berry's scatter is evaluated right to left (GCC)
        _patch_source(monkeypatch, GV, None, "_grapevine_fruit_prototype",
                      "            sz = ctx.randn()                             # make_vec3(randn(), randn(), randn()): right to left\n"
                      "            sy = ctx.randn()\n            sx = ctx.randn()\n",
                      "            sx = ctx.randn()\n            sy = ctx.randn()\n            sz = ctx.randn()\n")
        monkeypatch.setattr(GV.GrapevineFruitPrototype, "impl", GV._grapevine_fruit_prototype)
        assert _fails("plantarch.grape.age84", range(1))
