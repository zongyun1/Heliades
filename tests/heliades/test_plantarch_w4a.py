# SPDX-License-Identifier: LGPL-2.1-or-later
"""W4-A: cherry tomato and capsicum (library entries, phytomer creation / callback functions, removeLeaf, the
shed-petiole XML), and mutation checks against the oracle."""
from __future__ import annotations

import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture, hooks
from heliades.plantarchitecture.xml_io import plant_structure_xml

f32 = np.float32


def _pa(crop, seed=3):
    from heliades.plantarchitecture import leaf_prototype  # noqa: F401
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(crop)
    return s, pa


def test_registered():
    s, pa = _pa("tomato")
    for crop in ("cherrytomato", "capsicum"):
        assert crop in pa.getAvailablePlantModels()
        pa.loadPlantModelFromLibrary(crop)
        assert set(pa.shoot_types) == ({"mainstem", "lateral"} if crop == "cherrytomato" else {"mainstem", "secondary"})


def test_capsicum_secondary_has_no_creation_function():
    s, pa = _pa("capsicum")
    assert pa.shoot_types["mainstem"].phytomer_parameters.phytomer_creation_function is hooks.CapsicumPhytomerCreationFunction
    assert pa.shoot_types["secondary"].phytomer_parameters.phytomer_creation_function is None
    assert int(pa.shoot_types["secondary"].max_nodes.val()) == 7


def test_capsicum_first_leaves_removed():
    s, pa = _pa("capsicum", 2)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 30)
    main = pa.plant_instances[pid].shoot_tree[0]
    for i, ph in enumerate(main.phytomers):
        if i < 6:
            assert not ph.hasLeaf() and not ph.leaf_objIDs and not ph.petiole_objIDs and not ph.petiole_length
            assert len(ph.petiole_vertices) == 1 and len(ph.petiole_vertices[0]) == 2      # centerline kept
        else:
            assert ph.hasLeaf()
    root = ET.fromstring(plant_structure_xml(pa, pid))
    phs = root.findall("plant_instance/shoot")[0].findall("phytomer")
    assert all(p.find("internode/shed_petiole/centerline_offsets") is not None for p in phs[:6])
    assert all(p.find("internode/petiole") is not None for p in phs[6:])


def test_cherrytomato_callback_runs_per_phytomer_and_substep(monkeypatch):
    s, pa = _pa("cherrytomato", 4)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    calls = []
    impl = hooks.CherryTomatoPhytomerCallbackFunction.impl
    monkeypatch.setattr(hooks.CherryTomatoPhytomerCallbackFunction, "impl",
                        lambda ph: (calls.append((ph.shoot_index[0], pa.phytomer_callback_depth)), impl(ph)))
    pa.advanceTime(pid, 2.0)                                       # two one-day sub-steps, one phytomer
    assert calls == [(0, 1), (0, 1)] and pa.phytomer_callback_depth == 0


def test_cherrytomato_callback_prunes_low_leaves_from_day_101():
    s, pa = _pa("cherrytomato", 4)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 20)
    ph = pa.plant_instances[pid].shoot_tree[0].phytomers[0]
    leaves = [o for p in ph.leaf_objIDs for o in p]
    hooks.CherryTomatoPhytomerCallbackFunction(ph)
    assert ph.hasLeaf()
    pa.plant_instances[pid].current_age = f32(101)
    hooks.CherryTomatoPhytomerCallbackFunction(ph)
    assert not ph.hasLeaf() and not any(s.does_object_exist(o) for o in leaves)


@pytest.mark.parametrize("crop, age", [("cherrytomato", 75), ("capsicum", 80)])
def test_vegetative_to_gallery_age(crop, age):
    """No (crop, gallery age, Context seed 1..8) builds a peduncle, flower or fruit."""
    for seed in range(1, 9):
        s, pa = _pa(crop, seed)
        pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age)
        assert pa.plant_instances[pid].current_age == f32(age)
        assert not any(b.peduncle_objIDs or b.inflorescence_objIDs for sh in pa.plant_instances[pid].shoot_tree
                       for ph in sh.phytomers for p in ph.floral_buds for b in p)


def _probe_matches():
    """``plantarch.w4a.literal_probe`` (direct builds at the plant ages where float and double scales differ)."""
    return not _fails("plantarch.w4a.literal_probe", range(5))


# ---- mutation checks (oracle) -------------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


def _patch_source(monkeypatch, module, attr, old, new, owner=None):
    """Recompile ``module`` with one source edit; install ``attr`` (of class ``owner``, or of the module) from the
    edited copy. Hook bindings made by the re-executed module are restored by ``monkeypatch``."""
    for name in ("CherryTomatoPhytomerCreationFunction", "CherryTomatoPhytomerCallbackFunction",
                 "CapsicumPhytomerCreationFunction", "CapsicumFruitPrototype", "AlmondFlowerPrototype"):
        h = getattr(hooks, name)
        monkeypatch.setattr(h, "impl", h.impl)
    src = open(module.__file__).read()
    assert src.count(old) == 1, old
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
    exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
    if owner is None:
        monkeypatch.setattr(module, attr, getattr(mod, attr))
        return getattr(mod, attr)
    monkeypatch.setattr(getattr(module, owner), attr, getattr(getattr(mod, owner), attr))


MUTATIONS = ["cherry_leaf_scale_float", "cherry_inode_scale_float", "capsicum_inode_scale_double",
             "capsicum_keeps_first_leaves", "remove_leaf_drops_centerline", "capsicum_secondary_creation_function",
             "shed_petiole_offsets_from_tip", "cherry_lateral_type_is_mainstem", "capsicum_fruit_no_randn",
             "cherry_pruning_height"]


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity import cases_plantarch_w4a as CW
    from heliades.parity.ops import load_plugins
    from heliades.plantarchitecture import library as L
    from heliades.plantarchitecture import phytomer as PH
    from heliades.plantarchitecture import xml_io as PL
    from heliades.plantarchitecture.library import capsicum as CAP
    from heliades.plantarchitecture.library import cherrytomato as CT
    load_plugins()
    monkeypatch.setattr(CW, "STEP_AGES", tuple(range(2, 21, 2)))
    assert _probe_matches()
    if mutation == "cherry_leaf_scale_float":
        _patch_source(monkeypatch, CT, "_cherry_tomato_phytomer_creation",
                      "f32(min(1.0, 0.7 + 0.3 * age / 15.0))", "min(f32(1), f32(f32(0.7) + f32(f32(f32(0.3) * f32(age)) / f32(15))))")
        assert not _probe_matches()
    elif mutation == "cherry_inode_scale_float":
        _patch_source(monkeypatch, CT, "_cherry_tomato_phytomer_creation",
                      "f32(min(1.0, 0.7 + 0.3 * age / 10.0))", "min(f32(1), f32(f32(0.7) + f32(f32(f32(0.3) * f32(age)) / f32(10))))")
        assert not _probe_matches()
    elif mutation == "capsicum_inode_scale_double":
        _patch_source(monkeypatch, CAP, "_capsicum_phytomer_creation",
                      "min(f32(1), f32(f32(0.05) + f32(f32(f32(0.95) * f32(plant_age)) / f32(15))))",
                      "f32(min(1.0, 0.05 + 0.95 * float(f32(plant_age)) / 15.0))")
        assert not _probe_matches()
    elif mutation == "capsicum_keeps_first_leaves":
        _patch_source(monkeypatch, CAP, "_capsicum_phytomer_creation", "        phytomer.removeLeaf()\n", "")
        assert _fails("plantarch.w4a.capsicum.age0_full", range(2))
    elif mutation == "remove_leaf_drops_centerline":
        # removeLeaf() keeps petiole_vertices (PA:6770 is commented out): the next phytomer's parent petiole axis
        _patch_source(monkeypatch, PH, "removeLeaf", "        self.petiole_radii = []\n",
                      "        self.petiole_radii = []\n        self.petiole_vertices = []\n", owner="Phytomer")
        assert _fails("plantarch.w4a.capsicum.steps", range(2))
    elif mutation == "capsicum_secondary_creation_function":
        # phytomer_parameters_secondary is copied before the creation function is set
        fn = _patch_source(monkeypatch, CAP, "initializeCapsicumShoots",
                           "    sp_secondary.phytomer_parameters = pp_secondary                              # no creation function\n", "")
        monkeypatch.setitem(L.SHOOT_INITIALIZERS, "capsicum", fn)
        assert _fails("plantarch.w4a.capsicum.gallery", range(2))
    elif mutation == "shed_petiole_offsets_from_tip":
        _patch_source(monkeypatch, PL, "_write_phytomer", "np.asarray(centerline[0], f32) for n in centerline",
                      "np.asarray(centerline[-1], f32) for n in centerline")
        assert _fails("plantarch.w4a.capsicum.steps", range(1))
    elif mutation == "cherry_lateral_type_is_mainstem":
        # laterals never break buds (probability 0): as a second main stem they would
        fn = _patch_source(monkeypatch, CT, "initializeCherryTomatoShoots", 'sp.defineChildShootTypes(["lateral"], [1.0])',
                           'sp.defineChildShootTypes(["mainstem"], [1.0])')
        monkeypatch.setitem(L.SHOOT_INITIALIZERS, "cherrytomato", fn)
        assert _fails("plantarch.w4a.cherrytomato.load", range(2))
    elif mutation == "capsicum_fruit_no_randn":
        # CapsicumFruitPrototype's randn() picks the colour and is a Context draw per prototype
        _patch_source(monkeypatch, CAP, "_capsicum_fruit_prototype", "if float(context.randn()) < 0.4:", "if True:")
        assert _fails("plantarch.w4a.capsicum.reproductive", range(8, 10))
    elif mutation == "cherry_pruning_height":
        _patch_source(monkeypatch, CT, "_cherry_tomato_phytomer_callback", "pruning_height = f32(1)",
                      "pruning_height = f32(0.8)")
        assert _fails("plantarch.w4a.cherrytomato.reproductive", range(8, 10))
