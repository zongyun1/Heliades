# SPDX-License-Identifier: LGPL-2.1-or-later
"""Species files: validation errors, registration, and the cucumber-from-tomato file against the library cucumber."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from heliades.species import SpeciesError, available_species, load_spec, register, validate
from heliades.species.loader import parse_spec

MINIMAL = {
    "format": "heliades-species/1",
    "name": "testplant",
    "template": "tomato",
    "metadata": {"latin_name": "Testus plantus", "common_name": "test plant"},
    "sources": {"smith2020": {"citation": "Smith J (2020) A test reference. J. Tests 1:1-2."}},
    "shoot_types": {"mainstem": {"shoot": {"max_nodes": {"value": 12, "source": "smith2020"}}}},
}


def _with(path, value):
    d = copy.deepcopy(MINIMAL)
    o = d
    for k in path[:-1]:
        o = o.setdefault(k, {})
    o[path[-1]] = value
    return d


def _scene_pa():
    from heliades.core.scene import Scene
    from heliades.plantarchitecture import PlantArchitecture
    sc = Scene()
    sc.seed_random_generator(3)
    return sc, PlantArchitecture(sc)


def test_minimal_parses():
    s = parse_spec(copy.deepcopy(MINIMAL))
    assert s.name == "testplant" and s.template == "tomato"
    assert s.summary() == {"name": "testplant", "template": "tomato", "overrides": 1, "cited": 1, "judged": 0,
                           "traits": 0}


@pytest.mark.parametrize("path,value,msg", [
    (("colour",), 1, "unknown key 'colour'"),
    (("format",), "v2", "format"),
    (("template",), "banana", "not a library model"),
    (("name",), "tomato", "is a Helios library model"),
    (("metadata", "latin"), "x", "unknown key 'latin'"),
    (("shoot_types", "mainstem", "shoot", "max_nodes"), {"value": 12}, "needs a 'source'"),
    (("shoot_types", "mainstem", "shoot", "max_nodes"), {"value": 12, "source": "jones"}, "'jones' is not in 'sources'"),
    (("shoot_types", "mainstem", "shoot", "max_nodes"), {"value": 0, "source": "judged"}, "outside the valid range"),
    (("shoot_types", "mainstem", "shoot", "max_nodes"), {"value": 1.5, "source": "judged"}, "whole number"),
    (("shoot_types", "mainstem", "shoot", "max_nodes"), {"normal": [10, 2], "source": "judged"}, "unknown key 'normal'"),
    (("shoot_types", "mainstem", "shoot", "fruit_set_probability"), {"value": 1.2, "source": "judged"}, "outside"),
    (("shoot_types", "mainstem", "shoot", "phyllochron_min"), {"uniform": [3, 2], "source": "judged"}, "above"),
    (("shoot_types", "mainstem", "shoot", "phyllochron_min"), {"normal": [2, -1], "source": "judged"}, "deviation"),
    (("shoot_types", "mainstem", "shoot", "phylochron_min"), {"value": 2, "source": "judged"}, "unknown key"),
    (("shoot_types", "mainstem", "internode", "color"), {"value": [0.2, 0.3], "source": "judged"}, "three numbers"),
    (("shoot_types", "mainstem", "leaf_prototype", "leaf_texture_file"), {"texture": "nope", "source": "judged"},
     "not one of this file's leaf_textures"),
    (("shoot_types", "mainstem", "phytomer_creation"), [{"rule": "grow_faster", "source": "judged"}], "unknown rule"),
    (("shoot_types", "mainstem", "phytomer_creation"), [{"rule": "harvest_ripe_fruit", "source": "judged"}],
     "is a callback rule"),
    (("shoot_types", "mainstem", "phytomer_creation"), [{"rule": "kill_buds", "which": "all", "source": "judged"}],
     "must be one of"),
    (("shoot_types", "mainstem", "phytomer_creation"), [{"rule": "kill_buds", "source": "judged"}],
     "needs parameter 'which'"),
    (("shoot_types", "mainstem", "phytomer_creation"),
     [{"rule": "leaf_scale_ramp", "intercept": 0.5, "slope": 0.5, "days": 10, "speed": 1, "source": "judged"}],
     "no parameter 'speed'"),
    (("shoot_types", "mainstem", "fruit"), {"ellipsoid": {"diameter": 0.3}, "source": "judged"}, "'color' is required"),
    (("shoot_types", "mainstem", "fruit"), {"corolla": {}, "source": "judged"}, "corolla is a flower shape"),
    (("shoot_types", "mainstem", "flower"), {"obj_variants": "f_{index}.obj", "count": 2, "source": "judged"},
     "{state}"),
    (("shoot_types", "mainstem", "child_shoot_types"), {"value": {"a": 0.5, "b": 0.4}, "source": "judged"}, "sum to 1"),
    (("build",), {"builder": "generic", "source": "judged"}, "required by the generic builder"),
    (("build",), {"builder": "template", "base_stem": {}}, "only for builder: generic"),
    (("build",), {"phenology": {"fruit_set": {"value": 3}}}, "needs a 'source'"),
    (("traits",), [{"trait": "height", "age": 10, "range": [0, 1], "source": "judged"}], "unknown trait"),
    (("traits",), [{"trait": "plant_height", "age": 10, "range": [1, 0], "source": "judged"}], "expected [min, max]"),
    (("leaf_textures",), {"lf": {"aspect": 0.5, "lobes": {"count": 3}, "source": "judged"}}, "'depth' is required"),
])
def test_validation_errors(path, value, msg):
    with pytest.raises(SpeciesError) as e:
        parse_spec(_with(path, value))
    assert msg in str(e.value), str(e.value)


def test_deep_validation_errors():
    sc, pa = _scene_pa()
    spec = parse_spec(_with(("shoot_types", "main"), {"from": "trunk"}))
    register(pa, spec)
    with pytest.raises(SpeciesError, match="no shoot type|has the label 'trunk'"):
        pa.loadPlantModelFromLibrary("testplant")
    spec = parse_spec(_with(("shoot_types", "mainstem", "child_shoot_types"),
                            {"value": {"branch": 1.0}, "source": "judged"}))
    register(pa, spec)
    with pytest.raises(SpeciesError, match="'branch', which the species does not define"):
        pa.loadPlantModelFromLibrary("testplant")
    spec = parse_spec(_with(("build",), {"build_parameter_defaults": {"tiller_count": {"value": 2,
                                                                                         "source": "judged"}}}))
    with pytest.raises(SpeciesError, match="declares no build parameter 'tiller_count'"):
        register(pa, spec)


def test_registry_unchanged_until_loaded():
    sc, pa = _scene_pa()
    before = pa.getAvailablePlantModels()
    assert "eggplant" not in before and len(before) == 32
    pa.loadPlantModelFromLibrary("eggplant")
    assert "eggplant" in pa.getAvailablePlantModels()
    with pytest.raises(Exception, match="does not exist in the library"):
        pa.loadPlantModelFromLibrary("no_such_plant")


def test_species_files_validate():
    names = available_species()
    assert {"cucumber_from_tomato", "eggplant", "barley"} <= set(names)
    for n in names:
        validate(n)


def test_from_an_earlier_shoot_type():
    """``from`` may name a shoot type defined earlier in the file: the copy carries the species' overrides."""
    d = _with(("shoot_types", "mainstem", "petiole"), {"length": {"value": 0.11, "source": "judged"}})
    d["shoot_types"]["mainstem"]["child_shoot_types"] = {"value": {"side": 1.0}, "source": "judged"}
    d["shoot_types"]["side"] = {"from": "mainstem", "shoot": {"max_nodes": {"value": 5, "source": "judged"}},
                                "child_shoot_types": {"value": {"side": 1.0}, "source": "judged"}}
    sc, pa = _scene_pa()
    register(pa, parse_spec(d))
    pa.loadPlantModelFromLibrary("testplant")
    assert sorted(pa.shoot_types) == ["mainstem", "side"]
    side = pa.shoot_types["side"]
    assert float(side.phytomer_parameters.petiole.length.val()) == pytest.approx(0.11)
    assert int(side.max_nodes.val()) == 5 and int(pa.shoot_types["mainstem"].max_nodes.val()) == 12
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 50.0)
    assert {s.shoot_type_label for s in pa.plant_instances[pid].shoot_tree} >= {"mainstem", "side"}


def test_distribution_on_a_parameter_without_generator():
    """leaflet_scale is set by operator= in PhytomerParameters() (no generator): a distribution override gives it the
    Context generator, so it samples instead of raising."""
    d = _with(("shoot_types", "mainstem", "leaf"), {"leaflet_scale": {"uniform": [0.5, 0.9], "source": "judged"}})
    sc, pa = _scene_pa()
    register(pa, parse_spec(d))
    pa.loadPlantModelFromLibrary("testplant")
    p = pa.shoot_types["mainstem"].phytomer_parameters.leaf.leaflet_scale
    assert p.generator is pa.generator and 0.5 <= float(p.copy().val()) <= 0.9


def _cucumber_pair(seed, age, mode="exact"):
    from heliades.core.scene import Scene
    from heliades.plantarchitecture import PlantArchitecture
    out = []
    for use_spec in (False, True):
        sc = Scene(mode=mode)
        sc.seed_random_generator(seed)
        pa = PlantArchitecture(sc)
        if use_spec:
            register(pa, "cucumber_from_tomato", "cucumber")
        pa.loadPlantModelFromLibrary("cucumber")
        pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), age)
        out.append((sc, pa, pid))
    return out


@pytest.mark.parametrize("seed,age", [(1, 0.0), (4, 0.0), (2, 16.0)])
def test_cucumber_from_tomato_matches_library(seed, age):
    """The fast in-process check (the oracle comparison is the parity suite plantarch.species.cucumber_from_tomato)."""
    from heliades.plantarchitecture.structure import canonical, shoot_parameters_to_dict
    (sa, pa_a, a), (sb, pa_b, b) = _cucumber_pair(seed, age)
    assert canonical(shoot_parameters_to_dict(pa_a.shoot_types["main"])) == \
        canonical(shoot_parameters_to_dict(pa_b.shoot_types["main"]))
    A, B = sa.snapshot(True), sb.snapshot(True)
    assert np.array_equal(A.uuid, B.uuid) and np.array_equal(A.vertices, B.vertices)
    assert np.array_equal(A.color, B.color) and A.texture == B.texture
    assert pa_a.writePlantStructureXML(a) == pa_b.writePlantStructureXML(b)
    assert sa.rng.generator.__dict__ == sb.rng.generator.__dict__


def test_cucumber_from_tomato_shoot_parameters_source_marks():
    """Every value the file sets is marked; the cited ones are those Helios cites."""
    s = load_spec("cucumber_from_tomato")
    cited = {(o.section, o.field): o.source for st in s.shoot_types for o in st.overrides if o.source != "judged"}
    assert cited[("shoot", "phyllochron_min")] == "kahlen_chen_2015"
    assert cited[("leaf", "prototype_scale")] == "saba_2026"
    assert cited[("inflorescence", "fruit_prototype_scale")] == "english_fruit_grades_2025"


def test_batch_and_fast_mode_build_species():
    from heliades.plantarchitecture.batch import PlantJob, build_plants
    res = build_plants([PlantJob("barley", 30.0, seed=2), PlantJob("eggplant", 30.0, seed=2, mode="exact")], workers=1)
    assert all(r.primitives > 0 for r in res)
    from heliades.parity.fastmode import check
    r = check("eggplant", 45.0, 3)
    assert r["draws"] and r["topology"], r


def test_build_parameter_defaults_and_inheritance():
    sc, pa = _scene_pa()
    pa.loadPlantModelFromLibrary("barley")
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 30.0)
    assert '<build_parameter name="tiller_count"> 3 </build_parameter>' in pa.writePlantStructureXML(pid)
    assert pa.getPlantBuildParameters(pid)["tiller_count"] == 3
    pid2 = pa.buildPlantInstanceFromLibrary((1, 0, 0), 30.0, {"tiller_count": 1})
    assert pa.getPlantBuildParameters(pid2)["tiller_count"] == 1
    with pytest.raises(Exception, match="outside the valid range|reaches outside"):
        pa.buildPlantInstanceFromLibrary((2, 0, 0), 30.0, {"tiller_count": 9})
