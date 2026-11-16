# SPDX-License-Identifier: LGPL-2.1-or-later
"""W3-E: the public PlantArchitecture API, plant_structure_dict, multi-plant builds and the acceptance gating."""
from __future__ import annotations

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture import leaf_prototype  # noqa: F401  (real blades)
from heliades.plantarchitecture.api import PUBLIC_API, SNAKE_ALIASES
from heliades.plantarchitecture.structure import xml_to_structure_dict

f32 = np.float32


def _tomato(seed=1):
    s = Scene()
    s.seed_random_generator(seed)
    pa = PlantArchitecture(s)
    pa.loadPlantModelFromLibrary("tomato")
    return s, pa


@pytest.fixture(scope="module")
def plant12():
    s, pa = _tomato(2)
    pid = pa.buildPlantInstanceFromLibrary((0.1, -0.2, 0.0), 12)
    return s, pa, pid


def test_snake_aliases_are_the_same_functions():
    for camel, snake in SNAKE_ALIASES.items():
        assert snake.islower() and " " not in snake
        assert getattr(PlantArchitecture, snake) is getattr(PlantArchitecture, camel), camel
    assert SNAKE_ALIASES["getAllLeafUUIDs"] == "get_all_leaf_uuids"
    assert SNAKE_ALIASES["getPlantLeafObjectIDs"] == "get_plant_leaf_object_ids"
    assert SNAKE_ALIASES["writePlantStructureXML"] == "write_plant_structure_xml"
    assert len(set(SNAKE_ALIASES.values())) == len(PUBLIC_API)


def test_metrics_are_consistent(plant12):
    s, pa, pid = plant12
    leaves = pa.getPlantLeafObjectIDs(pid)
    assert pa.getPlantLeafCount(pid) == len(leaves) == 7 * len(pa.getPlantPetioleObjectIDs(pid))
    area = f32(0)
    for o in leaves:
        area = f32(area + pa.getLeafBladeArea(o))
    assert pa.getPlantLeafArea(pid) == area == pa.sumPlantLeafArea(pid)
    assert pa.getPlantAge(pid) == f32(12)
    assert 0 < pa.getPlantStemHeight(pid) <= pa.getPlantHeight(pid)
    assert np.array_equal(pa.getPlantBasePosition(pid), np.array([0.1, -0.2, 0], f32))
    assert pa.getPlantName(pid) == "tomato" and pa.getAllPlantIDs() == [pid]
    # organ UUID lists partition the plant's visible UUIDs
    organs = (pa.getAllInternodeUUIDs(), pa.getAllPetioleUUIDs(), pa.getAllLeafUUIDs())
    assert sorted(u for o in organs for u in o) == sorted(pa.getAllPlantUUIDs(pid)) == s.all_uuids()
    hidden = set(pa.getAllPlantUUIDs(pid, include_hidden=True)) - set(pa.getAllPlantUUIDs(pid))
    assert hidden == set(s.all_uuids(include_hidden=True)) - set(s.all_uuids())
    assert len(pa.getAllPrototypeObjectIDs()) == 35
    # object_label selections
    leaf_u = set(pa.getPlantUUIDsByObjectLabel(pid, "leaf"))
    petiolule_u = set(pa.getPlantUUIDsByObjectLabel(pid, "petiolule"))
    assert leaf_u | petiolule_u == set(pa.getAllLeafUUIDs()) and not leaf_u & petiolule_u
    assert set(pa.getPlantUUIDsByObjectLabel(pid, "shoot")) == set(pa.getAllInternodeUUIDs())
    assert pa.getPlantFlowerObjectIDs(pid) == pa.getPlantFruitObjectIDs(pid) == pa.getPlantPeduncleObjectIDs(pid) == []


def test_shoot_topology(plant12):
    s, pa, pid = plant12
    shoots = pa.getAllShootIDs(pid)
    assert shoots[0] == 0 and pa.getParentShootID(pid, 0) == -1 and pa.getShootDepth(pid, 0) == 0
    for sid in shoots[1:]:
        path = pa.getPathToRoot(pid, sid)
        assert path[0] == sid and path[-1] == 0 and len(path) == pa.getShootDepth(pid, sid) + 1
        assert sid in pa.getChildShootIDs(pid, pa.getParentShootID(pid, sid))
    assert sorted(pa.getAllDescendantShootIDs(pid, 0)) == shoots[1:]
    by_rank = pa.getShootIDsByRank(pid)
    assert sorted(x for v in by_rank.values() for x in v) == shoots
    assert pa.getShootNodeCount(pid, 0) == len(pa.plant_instances[pid].shoot_tree[0].phytomers)
    with pytest.raises(Exception, match="out of range"):
        pa.getShootNodeCount(pid, 999)
    with pytest.raises(Exception, match="does not exist"):
        pa.getPlantAge(42)


def test_structure_dict(plant12):
    s, pa, pid = plant12
    d = pa.plant_structure_dict(pid)
    tree = pa.plant_instances[pid].shoot_tree
    assert [sd["ID"] for sd in d["shoots"]] == [sh.ID for sh in tree]
    for sd, sh in zip(d["shoots"], tree):
        assert len(sd["phytomers"]) == len(sh.phytomers)
        assert sd["organs"]["internode"] == sh.internode_tube_objID
        for pd, ph in zip(sd["phytomers"], sh.phytomers):
            pets = pd["internode"].get("petioles", [])
            assert len(pets) == len(ph.petiole_length)
            assert pd["organs"]["leaves"] == [list(x) for x in ph.leaf_objIDs]
            for pet in pets:
                assert len(pet["leaves"]) == 7 and all("leaf_pitch" in lf for lf in pet["leaves"])
    assert d["organs"]["leaf"] == pa.getPlantLeafObjectIDs(pid)
    assert d["plant_age"] == 12.0 and d["shoots"][0]["shoot_type_label"] == "mainstem"
    # the parser is the same for any writer's text
    assert xml_to_structure_dict(pa.writePlantStructureXML(pid)) == pa.plant_structure_dict(pid, organs=False)


def test_second_plant_shares_the_prototype_cache():
    """The second library plant creates no hidden objects and draws exactly what the first one drew (the blade
    cache build is on a private stream, so the Context stream sees no difference either way; the objects do)."""
    s, pa = _tomato(4)
    a = pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    n_hidden = len(s.all_object_ids(include_hidden=True)) - len(s.all_object_ids())
    b = pa.buildPlantInstanceFromLibrary((1, 0, 0), 0)
    assert len(s.all_object_ids(include_hidden=True)) - len(s.all_object_ids()) == n_hidden == 35
    assert pa.getAllPlantIDs() == [a, b] and pa.getAllPrototypeObjectIDs() == list(range(3, 38))
    assert len(pa.getAllUUIDs()) == len(pa.getAllPlantUUIDs(a)) + len(pa.getAllPlantUUIDs(b))


def test_canopy_layout():
    s, pa = _tomato(3)
    pids = pa.buildPlantCanopyFromLibrary((1.0, 2.0, 0.0), (0.5, 0.25), (3, 2), 0, 1.0)
    assert pids == list(range(6))
    xy = [tuple(pa.getPlantBasePosition(p)[:2]) for p in pids]
    assert xy == [(0.5, 1.875), (1.0, 1.875), (1.5, 1.875), (0.5, 2.125), (1.0, 2.125), (1.5, 2.125)]
    with pytest.raises(Exception, match="greater than zero"):
        pa.buildPlantCanopyFromLibrary((0, 0, 0), (0.5, 0.5), (0, 2), 0)


def test_advance_time_keyword_forms():
    s, pa = _tomato(5)
    a = pa.buildPlantInstanceFromLibrary((0, 0, 0), 0)
    b = pa.buildPlantInstanceFromLibrary((1, 0, 0), 0)
    pa.advanceTime(2.0, plant_id=b)
    assert (pa.getPlantAge(a), pa.getPlantAge(b)) == (0, 2)
    pa.advanceTime(1.0, plant_ids=[a, b])
    pa.advanceTime(a, 1.0)
    assert (pa.getPlantAge(a), pa.getPlantAge(b)) == (2, 3)
    with pytest.raises(ValueError):
        pa.advanceTime(1.0, plant_id=a, plant_ids=[b])


def test_gating(monkeypatch):
    from heliades.parity import cases_plantarch_acceptance as CA
    from heliades.parity import ops_plantarch_acceptance as OA
    assert [CA.grid_pair(s) for s in (0, 7, 8, 47)] == [(10.0, 1), (10.0, 8), (30.0, 1), (120.0, 8)]
    monkeypatch.setenv("HELIADES_PARITY_FLOWERING", "0")
    assert not OA.inflorescence_geometry_implemented()
    monkeypatch.setenv("HELIADES_PARITY_FLOWERING", "1")
    assert OA.inflorescence_geometry_implemented()
    monkeypatch.delenv("HELIADES_PARITY_FLOWERING")
    assert OA.inflorescence_geometry_implemented(), "W3-D is merged: no pair of the grid may be gated"


def test_flowering_pair_builds_flowers():
    """Seed 2 flowers by day 75 in the oracle; Heliades builds the flowers / fruit and the queries see them."""
    pytest.importorskip("pyhelios")
    from heliades.parity import ops_plantarch_acceptance as OA
    assert not OA.oracle_has_inflorescence(1, 75.0) and OA.oracle_has_inflorescence(2, 75.0)
    s, pa = _tomato(2)
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 75)
    organs = pa.getPlantFlowerObjectIDs(pid) + pa.getPlantFruitObjectIDs(pid)
    assert organs and pa.getPlantPeduncleObjectIDs(pid)
    assert set(organs) <= set(pa.getAllPlantObjectIDs(pid))
    assert len(pa.getAllPrototypeObjectIDs()) > 35            # + the flower / fruit prototypes
    d = pa.plant_structure_dict(pid)
    buds = [fb for sd in d["shoots"] for pd in sd["phytomers"] for pet in pd["internode"].get("petioles", [])
            for fb in pet.get("floral_buds", [])]
    assert any(fb["inflorescence"].get("flowers") for fb in buds)


def test_flatten_marks_structure():
    from heliades.parity.ops_plantarch_acceptance import flatten
    assert flatten([[1, 2], [3]]) != flatten([[1], [2, 3]])
    assert flatten(np.array([1, 2, 3], f32)) == flatten(type("V", (), {"x": 1, "y": 2, "z": 3})())
