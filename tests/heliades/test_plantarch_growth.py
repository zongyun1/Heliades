# SPDX-License-Identifier: LGPL-2.1-or-later
"""W3-C: tomato growth (advanceTime and the structural operations it drives), and mutation checks against the
oracle."""
from __future__ import annotations

import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.parity.tools.draw_count import calls_between
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import growth as G
from heliades.plantarchitecture import shoot_min as SM
from heliades.plantarchitecture.params import BUD_ACTIVE, BUD_DEAD
from heliades.plantarchitecture.plant import cout, plant_structure_xml

f32 = np.float32


def _tomato(seed=5):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("tomato")
    return s, pa


# ---- sub-steps ------------------------------------------------------------------------------------

@pytest.mark.parametrize("days, expected", [
    (2.0, [1, 1]),                    # T <= phyllochron_min: int(T) one-day steps
    (1.5, [1, 0.5]),                  # ... plus the float remainder
    (5.0, [2, 2, 1]),                 # floor(T / pmin) steps of pmin, then the remainder
    (20.0, [2] * 10),
    (0.5, [0.5]),
])
def test_substep_schedule(monkeypatch, days, expected):
    s, pa = _tomato()
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    seen = []
    monkeypatch.setattr(G, "_advance_plant_one_step", lambda pa_, plantID, dt: seen.append(float(dt)))
    pa.advanceTime(pid, days)
    assert seen == expected


def test_split_identities():
    """advanceTime(20) == advanceTime(10) twice, but != advanceTime(5) four times (SPEC §6)."""
    texts = []
    for split in ((20.0,), (10.0, 10.0), (5.0,) * 4):
        s, pa = _tomato(3)
        pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
        for d in split:
            pa.advanceTime(pid, d)
        texts.append(plant_structure_xml(pa, pid))
    assert texts[0] == texts[1]
    assert texts[0] != texts[2]


# ---- draw counts (oracle values from SPEC §3.3: seed 5, build at age 0/2/4/10 = 36/66/101/208 incl. load) ---

@pytest.mark.parametrize("age, calls", [(0, 36), (2, 66), (4, 101), (10, 208)])
def test_growth_engine_calls_seed5(age, calls):
    s, pa = _tomato(5)
    pa.buildPlantInstanceFromLibrary((0, 0, 0), age)
    assert calls_between(5, s.rng.generator) == calls


def test_lateral_shoots_and_node_cap():
    s, pa = _tomato(3)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 50)
    tree = pa.plant_instances[pid].shoot_tree
    assert len(tree) > 1
    main = tree[0]
    assert main.current_node_number == 16 and not main.meristem_is_alive        # max_nodes 16
    for shoot in tree[1:]:
        assert shoot.rank == tree[shoot.parent_shoot_ID].rank + 1
        assert shoot.ID in tree[shoot.parent_shoot_ID].childIDs[shoot.parent_node_index]
        vb = tree[shoot.parent_shoot_ID].phytomers[shoot.parent_node_index].axillary_vegetative_buds
        assert any(b.shoot_ID == shoot.ID and b.state == BUD_DEAD for petiole in vb for b in petiole)
        # the lateral is seated 0.9 parent radii off the parent's node
        node = tree[shoot.parent_shoot_ID].shoot_internode_vertices[shoot.parent_node_index][-1]
        assert np.allclose(shoot.shoot_internode_vertices[0][0], node + shoot.cached_base_offset)
    for ph in main.phytomers:
        assert ph.shoot_index[1] == main.current_node_number                     # refreshed on each append


def test_tube_objects_follow_the_stored_nodes():
    s, pa = _tomato(2)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 30)
    for shoot in pa.plant_instances[pid].shoot_tree:
        nodes = s.tube_object_nodes(shoot.internode_tube_objID)
        stored = np.array([v for row in shoot.shoot_internode_vertices for v in row], f32)
        assert np.array_equal(nodes, stored)
        for ph in shoot.phytomers:
            assert np.array_equal(s.tube_object_nodes(ph.petiole_objIDs[0]), np.array(ph.petiole_vertices[0], f32))


def test_flowering_needs_w3d_or_draws_only():
    """Seed 6 initiates its first flower bud on day 56 (node 8, created day 16, +40 days)."""
    s, pa = _tomato(6)
    with pytest.raises(NotImplementedError, match="W3-D"):
        pa.buildPlantInstanceFromLibrary((0, 0, 0), 56)
    s, pa = _tomato(6)
    pa.inflorescence_geometry_draws_only = True
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 56)
    assert pa.inflorescence_geometry_is_stub
    main = pa.plant_instances[pid].shoot_tree[0]
    assert not main.meristem_is_alive                                            # determinate growth
    assert all(b.state == BUD_DEAD for ph in main.phytomers for p in ph.axillary_vegetative_buds for b in p)


def test_no_flowers_before_day_55():
    for seed in range(1, 9):
        s, pa = _tomato(seed)
        pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 54)       # raises if updateInflorescence were reached
        assert not pa.inflorescence_geometry_is_stub
        assert pa.plant_instances[pid].current_age == f32(54)


def test_xml_writer_shape():
    s, pa = _tomato(4)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 12)
    root = ET.fromstring(plant_structure_xml(pa, pid))
    shoots = root.findall("plant_instance/shoot")
    assert len(shoots) == 1 and len(shoots[0].findall("phytomer")) == 7
    assert root.find("plant_instance/plant_age").text.strip() == "12"
    assert cout(f32(0.1)) == "0.1" and cout(1.0) == "1" and cout(f32(1e-5)) == "1e-05"


def test_child_outward_axis_is_perpendicular():
    s, pa = _tomato(3)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 40)
    tree = pa.plant_instances[pid].shoot_tree
    for shoot in tree[1:]:
        parent = tree[shoot.parent_shoot_ID]
        axis = parent.phytomers[shoot.parent_node_index].getInternodeAxisVector(f32(1))
        out = SM.childShootOutwardAxis(parent, shoot.parent_node_index, shoot.parent_petiole_index, axis,
                                       shoot.base_rotation)
        assert abs(float(np.dot(out, axis))) < 1e-5 and abs(float(np.linalg.norm(out)) - 1) < 1e-5


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
    if owner is None:
        monkeypatch.setattr(module, attr, getattr(mod, attr))
    else:
        monkeypatch.setattr(getattr(module, owner), attr, getattr(getattr(mod, owner), attr))


@pytest.mark.parametrize("mutation", ["drop_leaf_scale_resample", "bud_break_ge", "girth_full_step",
                                      "no_child_offset_update", "single_substep", "elongate_own_internode_only"])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity import cases_plantarch_growth as CG
    from heliades.parity.ops import load_plugins
    load_plugins()
    from heliades.plantarchitecture import phytomer as PH
    monkeypatch.setattr(CG, "STEP_AGES", tuple(range(2, 37, 2)))
    if mutation == "drop_leaf_scale_resample":
        # the per-expanding-blade prototype_scale draw of PA:12133
        _patch_source(monkeypatch, G, None, "_expand_petioles_and_leaves",
                      "        phytomer.phytomer_parameters.leaf.prototype_scale.resample()\n", "")
        assert _fails("plantarch.tomato.steps")
    elif mutation == "bud_break_ge":
        # vegetative bud break at phytomer.age + dt > vegetative_bud_break_time (strict)
        _patch_source(monkeypatch, G, None, "_advance_plant_one_step",
                      "f32(phytomer.age + dt) > sp.vegetative_bud_break_time.val()",
                      "f32(phytomer.age + dt) >= sp.vegetative_bud_break_time.val()")
        assert _fails("plantarch.tomato.steps")
    elif mutation == "girth_full_step":
        # radius + 0.5 * (target - radius): a full step instead of half; only blade-area-exact checks see it
        _patch_source(monkeypatch, G, None, "incrementPhytomerInternodeGirth",
                      "float(segment[i]) + 0.5 * float(", "float(segment[i]) + 1.0 * float(")
        assert _fails("plantarch.tomato.steps_replay")
    elif mutation == "no_child_offset_update":
        # updateShootNodes must re-apply the lateral's outward offset as the parent thickens
        _patch_source(monkeypatch, SM, "Shoot", "updateShootNodes",
                      "self.cached_base_offset = (f32(0.9) * outward_axis) * parent_radius",
                      "self.cached_base_offset = self.cached_base_offset")
        assert _fails("plantarch.tomato.steps_replay")
    elif mutation == "single_substep":
        # advanceTime(T <= phyllochron_min) runs int(T) one-day steps, not one step of T
        _patch_source(monkeypatch, G, None, "advanceTime",
                      "        Nsteps = int(time_step_days)                              # float -> int truncation\n"
                      "        dt_max_days = f32(1)\n",
                      "        Nsteps = 1\n        dt_max_days = time_step_days\n")
        assert _fails("plantarch.tomato.steps")
    elif mutation == "elongate_own_internode_only":
        # setInternodeLengthScaleFraction carries every downstream node of the shoot along
        _patch_source(monkeypatch, PH, "Phytomer", "setInternodeLengthScaleFraction",
                      "for p_downstream in range(p, len(verts)):", "for p_downstream in range(p, p + 1):")
        assert _fails("plantarch.tomato.steps")
