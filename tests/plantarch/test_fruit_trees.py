# SPDX-License-Identifier: LGPL-2.1-or-later
"""W5-E: fruit / ornamental trees (apple, apple_fruitingwall, olive, easternredbud)."""
from __future__ import annotations

import numpy as np
import pytest

from heliades.core import random as R
from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture.library import apple as AP
from heliades.plantarchitecture.library import build_parameters as BP
from heliades.plantarchitecture.library import easternredbud as RB
from heliades.plantarchitecture.library import olive as OL
from heliades.plantarchitecture.library import trees as TR
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

f32 = np.float32
MODELS = ("apple", "apple_fruitingwall", "olive", "easternredbud")


def _tree(model, seed=1, age=0.0, build_parameters=None):
    s = Scene()
    s.seed_random_generator(seed)
    R.seed_global_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(model)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age, build_parameters)
    return s, pa, pid


# ---- unit checks -----------------------------------------------------------------------------------

def test_registered():
    pa = PlantArchitecture(Scene())
    for m in MODELS:
        assert m in pa.getAvailablePlantModels() and pa.plant_type_map[m] == "tree"
    assert BP.getPlantModelBuildParameterNames(pa, "apple_fruitingwall") == [
        "tree_height", "trunk_height", "num_feathers", "tree_spacing", "row_spacing"]
    assert BP.getPlantModelBuildParameterNames(pa, "olive") == ["trunk_height"]
    assert BP.getPlantModelBuildParameterNames(pa, "easternredbud") == []
    # constexpr float 1.5f - 0.4f - 0.05f - 0.05f
    assert AP.apple_fruitingwall_trunk_height.max_value == np.nextafter(f32(1), f32(2))


def test_mt19937_matches_std():
    g = R.MT19937()                                   # default seed 5489
    for _ in range(9999):
        g()
    assert g() == 4123659995                          # the C++ standard's check value for std::mt19937
    R.seed_global_random_generator(42)
    a = [R.global_randu(2, 4) for _ in range(50)]
    assert set(a) == {2, 3, 4}
    assert R.global_randu(3, 3) == 3 and R.global_randu(5, 2) == 5      # imin >= imax: imin, no draw
    R.seed_global_random_generator(42)
    assert [R.global_randu(2, 4) for _ in range(50)] == a


def test_textured_tile_and_olive_leaf():
    s = Scene()
    from heliades.plantarchitecture.library.bean import plugin_asset
    uu = s.add_tile_textured((0.5, 0, 0), (1, 0.2), (0, 0), (2, 2), plugin_asset("assets/textures/OliveLeaf_upper.png"))
    assert 0 < len(uu) <= 4 and not s.objects
    lp = type("P", (), {"subdivisions": 1})()
    oid = OL.OliveLeafPrototype(s, lp, 0)
    prims = s.object_primitive_uuids(oid)
    assert len(prims) == 2                             # upper + lower tile, back to back


def test_redbud_trunk_rotation_is_whole_radians():
    # randu(0, 2 * M_PI) selects Context::randu(int, int): roll and yaw in {0..6}, pitch randu(0, 0) = 0
    s = Scene()
    s.seed_random_generator(3)
    roll, yaw, pitch = s.randu(0, 6), s.randu(0, 6), s.randu(0, 0)
    assert isinstance(roll, int) and isinstance(yaw, int) and pitch == 0


def test_age0_trees():
    for m in MODELS:
        _, pa, pid = _tree(m)
        tree = pa.plant_instances[pid].shoot_tree
        if m == "easternredbud":
            assert not any(s.isdormant for s in tree) and len(tree) == 1
        else:
            assert all(s.isdormant for s in tree)
            assert not any(ph.hasLeaf() for s in tree for ph in s.phytomers)
    _, pa, pid = _tree("apple_fruitingwall", build_parameters={"num_feathers": 12})
    inst = pa.plant_instances[pid]
    assert inst.build_parameters["num_feathers"] == 12 and inst.build_parameters["row_spacing"] == f32(3.3)
    assert inst.shoot_tree[0].shoot_type_label == "leader"
    assert all(s.shoot_type_label == "branch" for s in inst.shoot_tree[1:])
    with pytest.raises(PlantArchitectureError):
        TR.setShootInternodeLengthMax(pa, pid, 0, 0)
    _, pa, pid = _tree("olive", build_parameters={"trunk_height": 0.9})
    assert pa.plant_instances[pid].is_evergreen and len(pa.plant_instances[pid].shoot_tree) == 5


def test_apple_light_thinning_keeps_a_share():
    class Bud:
        def __init__(self, terminal):
            self.isterminal, self.state = terminal, 0

    class Ph:
        def __init__(self):
            self.floral_buds = [[Bud(False), Bud(False), Bud(False)], [Bud(True)], [Bud(True)]]

        def setFloralBudState(self, state, fbud):
            fbud.state = state

    ph = Ph()
    AP.thinAppleFlowerClusterForLight(ph, False, f32(0.6))     # keeps ceil(0.43 * 3) = 2 of the axillary cluster
    assert [b.state for b in ph.floral_buds[0]] == [0, 0, 5]
    AP.thinAppleFlowerClusterForLight(ph, True, f32(0.3))      # below the minimum: the terminal cluster dies
    assert ph.floral_buds[1][0].state == 5 and ph.floral_buds[2][0].state == 5


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(1)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


SLOW_MUTATIONS = {"apple_light_threshold", "wall_heading_window", "olive_leaf_flip", "redbud_child_nodes"}


@pytest.mark.parametrize("mutation", [pytest.param(m, marks=pytest.mark.slow) if m in SLOW_MUTATIONS else m
                                      for m in ("redbud_trunk_float_randu", "wall_feather_round", "apple_light_threshold",
                                                "wall_heading_window", "olive_leaf_flip", "redbud_child_nodes")])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    if mutation == "redbud_trunk_float_randu":
        # randu(0, 2 * M_PI) is the int overload: a float draw gives a different trunk
        orig = RB.buildEasternRedbudPlant

        def build(pa, pos):
            ctx = pa.context_ptr
            real = ctx.randu
            monkeypatch.setattr(ctx, "randu", lambda a=None, b=None: real(f32(a), f32(b + 0.28)) if b == 6 else real(a, b),
                                raising=False)
            try:
                return orig(pa, pos)
            finally:
                monkeypatch.setattr(ctx, "randu", real, raising=False)
        from heliades.plantarchitecture import library as L
        monkeypatch.setitem(L.PLANT_BUILDERS, "easternredbud", build)
        assert _fails("plantarch.easternredbud.age0")
    elif mutation == "wall_feather_round":
        # feather node = round(z / 0.03): truncation moves feathers
        monkeypatch.setattr(AP, "_f32_round", lambda x: int(float(f32(x))))
        assert _fails("plantarch.apple_fruitingwall.age0", range(2))
    elif mutation == "apple_light_threshold":
        # a flower cluster keeps all its flowers above 0.8 of full light (the young tree is that well lit)
        monkeypatch.setattr(AP, "apple_flowering_light_exposure_full", f32(2.0))
        assert _fails("plantarch.apple.age548")   # bud states show in the next spring
    elif mutation == "wall_heading_window":
        # the leader is headed back to the highest branch within 0.5 m below the wall height
        monkeypatch.setattr(AP, "apple_fruitingwall_heading_window", f32(0.2))
        assert _fails("plantarch.apple_fruitingwall.short_wall_age400")
    elif mutation == "olive_leaf_flip":
        # the lower leaf tile is flipped about x
        from heliades.core.scene import Scene as S
        real = S.rotate_primitive
        monkeypatch.setattr(S, "rotate_primitive", lambda self, u, r, a: real(self, u, r, "y" if a == "x" else a))
        assert _fails("plantarch.olive.age170")
    elif mutation == "redbud_child_nodes":
        # the last 2-4 nodes of a dormant redbud shoot stay vegetative (helios::randu)
        monkeypatch.setattr(RB, "global_randu", lambda a, b: 2)
        assert _fails("plantarch.easternredbud.age548")
