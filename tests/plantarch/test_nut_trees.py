# SPDX-License-Identifier: LGPL-2.1-or-later
"""W5-D: nut trees (almond x4, pistachio, walnut) and the shared tree infrastructure (seasons, pruning, shade
shedding, growth priority, leaf-angle tracking)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import leaf_angle_tracking as LAT
from heliades.plantarchitecture import seasons as SE
from heliades.plantarchitecture.library import almond as AL
from heliades.plantarchitecture.library import build_parameters as BP
from heliades.plantarchitecture.library import pistachio as PI
from heliades.plantarchitecture.library import trees as TR
from heliades.plantarchitecture.library import walnut as WA
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

f32 = np.float32
MODELS = ("almond", "almond_aldrich", "almond_independence", "almond_wood_colony", "pistachio", "walnut")


def _tree(model, seed=1, age=0.0, build_parameters=None):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(model)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age, build_parameters)
    return s, pa, pid


# ---- unit checks -----------------------------------------------------------------------------------

def test_registered():
    pa = PlantArchitecture(Scene())
    for m in MODELS:
        assert m in pa.getAvailablePlantModels() and pa.plant_type_map[m] == "tree"
    names = ["trunk_height", "num_scaffolds", "scaffold_angle"]
    for m in ("almond", "almond_independence", "apple", "pistachio", "walnut"):
        assert BP.getPlantModelBuildParameterNames(pa, m) == names
    assert BP.getPlantModelBuildParameterNames(pa, "almond_aldrich") == []
    assert tuple(float(x) for x in pa.getPlantModelLeafInclinationDistribution("pistachio")) == (float(f32(1.32)),
                                                                                                float(f32(1.85)))


def test_trunk_helpers():
    assert TR.trunkMaxNodes(TR.tree_trunk_height, TR.walnut_trunk_internode_length) == 76
    assert TR.trunkMaxNodes(TR.tree_trunk_height, TR.pistachio_trunk_internode_length) == 61
    assert TR.trunk_node_count(f32(0.78), f32(0.03)) == 26 and TR.trunk_node_count(f32(0.01), f32(0.03)) == 1
    with pytest.raises(PlantArchitectureError):
        TR.checkScaffoldsFitTrunk("buildAlmondTree", 8, 3, 0.1)


def test_std_sort_is_a_sort():
    rng = random.Random(3)
    for n in (0, 1, 5, 16, 17, 40, 300):
        keys = [rng.randint(0, 9) for _ in range(n)]
        out = SE.std_sort(list(enumerate(keys)), lambda a, b: a[1] > b[1])
        assert [k for _, k in out] == sorted(keys, reverse=True)
        assert sorted(i for i, _ in out) == list(range(n))


def test_std_sort_insertion_order_for_short_runs():
    # libstdc++ sorts 16 or fewer elements by insertion sort, which keeps ties in input order
    out = SE.std_sort([(1, "a"), (0, "b"), (1, "c"), (0, "d")], lambda a, b: a[0] > b[0])
    assert out == [(1, "a"), (1, "c"), (0, "b"), (0, "d")]


def test_beta_cdf():
    assert LAT.evaluate_Beta_distribution_CDF(0, 1.32, 1.85) == 0
    assert LAT.evaluate_Beta_distribution_CDF(f32(0.5) * f32(math.pi), 1.32, 1.85) == 1
    xs = [LAT.evaluate_Beta_distribution_CDF(f32(t), 1.32, 1.85) for t in np.linspace(0, 1.5, 20)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))


def test_age0_trees_are_dormant_and_trained():
    for m in MODELS:
        _, pa, pid = _tree(m)
        tree = pa.plant_instances[pid].shoot_tree
        assert all(s.isdormant and s.dormancy_cycles == 1 for s in tree)
        assert tree[0].shoot_type_label == "trunk" and not tree[0].meristem_is_alive
        assert all(s.shoot_type_label == "scaffold" for s in tree[1:])
        assert not any(ph.hasLeaf() for s in tree for ph in s.phytomers)
    _, pa, pid = _tree("almond", build_parameters={"num_scaffolds": 3})
    assert len(pa.plant_instances[pid].shoot_tree) == 4


def test_pruning_removes_subtree():
    s, pa, pid = _tree("walnut", age=200)
    tree = pa.plant_instances[pid].shoot_tree
    parent = next(sh for sh in tree if sh.parent_shoot_ID >= 0 and sh.childIDs)
    child = next(iter(parent.childIDs.values()))[0]
    n_before = len(s.primitives)
    SE.pruneBranch(pa, pid, parent.ID, 0)
    assert parent.isPruned() and tree[child].isPruned() and len(s.primitives) < n_before
    assert all(parent.ID not in ids for ids in tree[parent.parent_shoot_ID].childIDs.values())
    assert SE.pruneBranch(pa, pid, parent.ID, 0) == []           # pruning a pruned shoot is a no-op


@pytest.mark.slow
def test_season_boundary_drops_leaves():
    _, pa, pid = _tree("walnut", age=366)
    inst = pa.plant_instances[pid]
    assert all(s.isdormant for s in inst.shoot_tree if not s.isPruned())
    assert not any(ph.hasLeaf() for s in inst.shoot_tree for ph in s.phytomers)
    assert inst.season_peak_leaf_area == 0 and float(inst.time_since_dormancy) < 2.0       # 2-day sub-steps (phyllochron_min)


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(1)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


SLOW_MUTATIONS = {"pistachio_tip_length", "growth_priority_gain", "shedding_threshold", "no_leaf_tracking"}


@pytest.mark.parametrize("mutation", [pytest.param(m, marks=pytest.mark.slow) if m in SLOW_MUTATIONS else m
                                      for m in ("trunk_wander_left_to_right", "no_spurs", "pistachio_roll_after_pitch",
                                                "walnut_not_blind", "no_leaf_tracking", "pistachio_tip_length",
                                                "growth_priority_gain", "shedding_threshold")])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    if mutation == "trunk_wander_left_to_right":
        # addComponent(randu(amp), L * randu(wl), randu(phase)) is evaluated right to left (GCC)
        def left_to_right(ctx, wander, amp_range, scale, wl_range, phase_range):
            a = ctx.randu(f32(amp_range[0]), f32(amp_range[1]))
            w = ctx.randu(f32(wl_range[0]), f32(wl_range[1]))
            if scale is not None:
                w = f32(f32(scale) * w)
            p = ctx.randu(f32(phase_range[0]), f32(phase_range[1]))
            wander.addComponent(a, w, p)
        monkeypatch.setattr(AL, "_add_component", left_to_right)
        assert _fails("plantarch.almond.age0")
    elif mutation == "no_spurs":
        # unbroken lateral buds of year-old almond wood grow spurs (fraction 0.2)
        monkeypatch.setattr(AL, "almond_spur_fraction", f32(0))
        assert _fails("plantarch.almond.age95")
    elif mutation == "pistachio_roll_after_pitch":
        # make_AxisRotation(pitch + randu(), azimuth, randu()): the roll is drawn first
        orig = PI.buildPistachioTree.__code__
        src = open(PI.__file__).read()
        old = ("        roll = ctx.randu(f32(0), f32(f32(2) * PI_F))\n"
               "        p = f32(pitch + ctx.randu(f32(-0.15), f32(0.15)))\n")
        new = ("        p = f32(pitch + ctx.randu(f32(-0.15), f32(0.15)))\n"
               "        roll = ctx.randu(f32(0), f32(f32(2) * PI_F))\n")
        assert src.count(old) == 1 and orig is not None
        ns = dict(PI.__dict__)
        exec(compile(src.replace(old, new), PI.__file__, "exec"), ns)
        from heliades.plantarchitecture import library as L
        monkeypatch.setitem(L.PLANT_BUILDERS, "pistachio", ns["buildPistachioTree"])
        assert _fails("plantarch.pistachio.age0")
    elif mutation == "walnut_not_blind":
        # the first four nodes of a walnut shoot are blind
        monkeypatch.setattr(WA.WalnutPhytomerCreationFunction, "impl", lambda *a: None)
        assert _fails("plantarch.walnut.age170")
    elif mutation == "no_leaf_tracking":
        # pistachio leaves are steered toward Beta(1.32, 1.85)
        monkeypatch.setattr(LAT, "_choose_target", lambda *a: (False, None))
        assert _fails("plantarch.pistachio.age170")
    elif mutation == "pistachio_tip_length":
        # scaffolds are tipped at 0.45 m in their first season
        monkeypatch.setattr(PI, "pistachio_scaffold_tip_length", f32(0.40))
        assert _fails("plantarch.pistachio.age220")
    elif mutation == "growth_priority_gain":
        # the best-lit siblings are promoted by 1.6 at the season boundary
        monkeypatch.setattr(SE, "growth_priority_gain", f32(1.5))
        assert _fails("plantarch.almond.age548")
    elif mutation == "shedding_threshold":
        # branches below 0.85 of the plant's mean light are shed stochastically
        monkeypatch.setattr(SE, "branch_shedding_relative_light_threshold", f32(0.80))
        assert _fails("plantarch.almond.age732")
