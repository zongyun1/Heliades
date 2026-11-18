# SPDX-License-Identifier: LGPL-2.1-or-later
"""``setPlantBasePosition`` (moving a grown plant) and the shadow-grid origin."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import seasons as SE
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

f32 = np.float32


def _tomato(age=30.0):
    ctx = Scene()
    ctx.seed_random_generator(2)
    pa = PlantArchitecture(ctx)
    pa.loadPlantModelFromLibrary("tomato")
    pid = pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), age)
    return ctx, pa, pid


def test_move_translates_geometry_and_state():
    ctx, pa, pid = _tomato()
    bases = [b.copy() for b in pa.getPlantLeafBases(pid)]
    pa.setPlantBasePosition(pid, (1.0, -2.0, 0.5))
    assert np.array_equal(pa.getPlantBasePosition(pid), np.array([1.0, -2.0, 0.5], f32))
    shift = np.array([1.0, -2.0, 0.5], f32)
    after = pa.getPlantLeafBases(pid)
    assert all(np.array_equal(a, b + shift) for a, b in zip(after, bases))
    assert np.array_equal(pa.plant_instances[pid].shadow_grid_origin, shift)


def test_move_unknown_plant_or_inside_callback_is_rejected():
    ctx, pa, pid = _tomato(0.0)
    with pytest.raises(PlantArchitectureError, match="does not exist"):
        pa.setPlantBasePosition(pid + 3, (0.0, 0.0, 0.0))
    pa.phytomer_callback_depth = 1
    with pytest.raises(PlantArchitectureError, match="inside a phytomer callback"):
        pa.setPlantBasePosition(pid, (0.0, 0.0, 0.0))


def test_shadow_grid_is_keyed_relative_to_the_origin():
    bases = np.array([[0.11, 0.07, 1.3], [0.3, -0.2, 1.1], [0.05, 0.4, 1.6]], f32)
    shift = np.array([1.53, -0.71, 0.0], f32)
    g0 = SE.ShadowGrid(bases)
    g1 = SE.ShadowGrid(bases + shift, shift)
    for p in ([0.1, 0.1, 0.9], [0.2, -0.1, 1.0], [0.0, 0.3, 1.2]):
        p = np.array(p, f32)
        assert g1.light_exposure(p + shift, shift) == g0.light_exposure(p)


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(1)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


@pytest.mark.parametrize("mutation", ["internode_nodes_not_moved",
                                      pytest.param("shadow_grid_ignores_origin", marks=pytest.mark.slow)])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    if mutation == "internode_nodes_not_moved":
        orig = PlantArchitecture.setPlantBasePosition

        def forgetful(self, plantID, base_position):
            kept = [copy.deepcopy(s.shoot_internode_vertices) for s in self.plant_instances[plantID].shoot_tree]
            orig(self, plantID, base_position)
            for s, v in zip(self.plant_instances[plantID].shoot_tree, kept):
                s.shoot_internode_vertices = v
        monkeypatch.setattr(PlantArchitecture, "setPlantBasePosition", forgetful)
        assert _fails("plantarch.move.tomato")
    elif mutation == "shadow_grid_ignores_origin":
        monkeypatch.setattr(SE, "_origin", lambda inst: None)
        assert _fails("plantarch.move.almond")
