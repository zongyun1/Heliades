# SPDX-License-Identifier: LGPL-2.1-or-later
"""W5-C: the weeds (bindweed, cheeseweed, groundcherryweed, puncturevine) and bougainvillea."""
from __future__ import annotations

import types

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import library as L
from heliades.plantarchitecture.library import bougainvillea as BG
from heliades.plantarchitecture.library import puncturevine as PV
from heliades.plantarchitecture.params import BUD_DEAD

f32 = np.float32
MODELS = ("bindweed", "cheeseweed", "groundcherryweed", "puncturevine", "bougainvillea")


def _plant(model, seed=1, age=0.0):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary(model)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age)
    return s, pa, pid


# ---- unit checks -----------------------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_registered_and_builds(model):
    s, pa, pid = _plant(model)
    assert model in L.SHOOT_INITIALIZERS and model in L.PLANT_BUILDERS
    assert len(pa.plant_instances[pid].shoot_tree) == 1
    assert len(s.all_uuids()) > 0


def test_max_ages():
    expected = {"bindweed": 50, "cheeseweed": 40, "groundcherryweed": 80, "puncturevine": 45, "bougainvillea": 365}
    for model, max_age in expected.items():
        _, pa, pid = _plant(model)
        assert pa.plant_instances[pid].max_age == f32(max_age)


def test_bougainvillea_stem_is_stripped():
    _, pa, pid = _plant("bougainvillea")
    stem = pa.plant_instances[pid].shoot_tree[0]
    assert len(stem.phytomers) == 3
    assert all(not ph.hasLeaf() for ph in stem.phytomers)
    assert all(b.state == BUD_DEAD for ph in stem.phytomers for p in ph.axillary_vegetative_buds for b in p)
    assert all(b.state == BUD_DEAD for ph in stem.phytomers for p in ph.floral_buds for b in p)


def test_puncturevine_shoot_types():
    s = Scene()
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("puncturevine")
    st = pa.shoot_types
    assert sorted(st) == ["base_puncturevine", "primary_puncturevine", "secondary_puncturevine"]
    assert int(st["primary_puncturevine"].phytomer_parameters.petiole.petioles_per_internode) == 2
    assert int(st["base_puncturevine"].phytomer_parameters.petiole.petioles_per_internode) == 0
    assert int(st["primary_puncturevine"].phytomer_parameters.leaf.leaves_per_petiole.val()) == 10


def test_bindweed_grows_runners():
    _, pa, pid = _plant("bindweed", 1, 20.0)
    labels = {sh.shoot_type_label for sh in pa.plant_instances[pid].shoot_tree}
    assert {"base_bindweed", "primary_bindweed"} <= labels


# ---- mutation checks (oracle) -------------------------------------------------------------------

def _fails(name, seeds=range(8)):
    from heliades.parity import run_case
    return any(not run_case(name, s).ok for s in seeds)


def _patch_source(monkeypatch, module, attr, old, new):
    """Recompile ``module`` (or the module of class ``module``) with one source edit; install ``attr`` from it."""
    if isinstance(module, type):
        import sys
        cls, module = module, sys.modules[module.__module__]
        src = open(module.__file__).read()
        assert src.count(old) == 1, old
        mod = types.ModuleType(module.__name__)
        mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
        exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
        obj = getattr(getattr(mod, cls.__name__), attr)
        monkeypatch.setattr(cls, attr, obj)
        return obj
    src = open(module.__file__).read()
    assert src.count(old) == 1, old
    mod = types.ModuleType(module.__name__)
    mod.__dict__.update(__name__=module.__name__, __package__=module.__package__, __file__=module.__file__)
    exec(compile(src.replace(old, new), module.__file__, "exec"), mod.__dict__)
    obj = getattr(mod, attr)
    monkeypatch.setattr(module, attr, obj)
    return obj


@pytest.mark.parametrize("mutation", ["puncturevine_leaflet_offset", "bougainvillea_buds_kept", "bud_break_signed"])
def test_mutations_are_caught(monkeypatch, mutation):
    pytest.importorskip("pyhelios")
    from heliades.parity.ops import load_plugins
    load_plugins()
    if mutation == "puncturevine_leaflet_offset":
        # pp.leaf.leaflet_offset = 0.18 (the 10-leaflet pinnate leaf)
        fn = _patch_source(monkeypatch, PV, "initializePuncturevineShoots", "pp.leaf.leaflet_offset = 0.18",
                           "pp.leaf.leaflet_offset = 0.17")
        monkeypatch.setitem(L.SHOOT_INITIALIZERS, "puncturevine", fn)
        assert _fails("plantarch.puncturevine.ages", range(1))
    elif mutation == "bougainvillea_buds_kept":
        # the builder kills the stem's vegetative buds (removeShootVegetativeBuds)
        fn = _patch_source(monkeypatch, BG, "buildBougainvilleaPlant", "    removeShootVegetativeBuds(pa, plantID, uID_stem)\n",
                           "")
        monkeypatch.setitem(L.PLANT_BUILDERS, "bougainvillea", fn)
        assert _fails("plantarch.bougainvillea.ages", range(1))
    elif mutation == "bud_break_signed":
        # Shoot::sampleVegetativeBudBreak, decay > 0: float(current_node_number - node_index - 1) is uint arithmetic,
        # so a bud sampled as its phytomer is appended wraps to 2^32 - 1 (probability_min), not -1
        from heliades.plantarchitecture import shoot as SH
        _patch_source(monkeypatch, SH.Shoot, "sampleVegetativeBudBreak",
                      "nodes_above = (self.current_node_number - int(node_index) - 1) % (1 << 32)",
                      "nodes_above = self.current_node_number - int(node_index) - 1")
        assert _fails("plantarch.puncturevine.ages", range(8, 9))      # 20 days, Context seed 1
