# SPDX-License-Identifier: LGPL-2.1-or-later
"""The named growth rules of species files (heliades.species.rules)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from heliades.species import rules as R

f32 = np.float32


class _Phytomer:
    def __init__(self, rank):
        self.rank = rank
        self.calls = []

    def setFloralBudState(self, state, *where):
        self.calls.append(("floral", int(state)))

    def setVegetativeBudState(self, state, *where):
        self.calls.append(("vegetative", int(state)))

    def scaleLeafPrototypeScale(self, s):
        self.calls.append(("leaf", s))

    def scaleInternodeMaxLength(self, s):
        self.calls.append(("internode", s))


def _run(name, params, rank, node, age=0.0):
    r = R.RULES[name]
    fn = R.make(name, r.kind, R.check_params(r, params, name, _raise))
    ph = _Phytomer(rank)
    fn(ph, node, 0, 20, age)
    return ph.calls


def _raise(msg):
    raise ValueError(msg)


def test_every_rule_is_documented():
    for name, r in R.RULES.items():
        assert r.doc and r.kind in ("creation", "callback", "build"), name


@pytest.mark.parametrize("params,rank,node,expected", [
    ({"which": "floral", "node_below": 8, "rank": 0}, 0, 7, [("floral", 5)]),
    ({"which": "floral", "node_below": 8, "rank": 0}, 0, 8, []),
    ({"which": "floral", "node_below": 8, "rank": 0}, 1, 2, []),
    ({"which": "both", "rank_above": 1}, 2, 0, [("floral", 5), ("vegetative", 5)]),
    ({"which": "both", "rank_above": 1}, 1, 0, []),
    ({"which": "vegetative", "node_from": 3}, 0, 3, [("vegetative", 5)]),
    ({"which": "vegetative", "node_from": 3}, 0, 2, []),
])
def test_kill_buds(params, rank, node, expected):
    assert _run("kill_buds", params, rank, node) == expected


@pytest.mark.parametrize("age", [0.0, 0.5, 3.25, 7.0, 9.999, 10.0, 31.5])
def test_ramps_match_the_tomato_formula(age):
    """``fmin(1.f, 0.5 + 0.5 * plant_age / 10.f)`` in double, narrowed once (AS:4104-4107)."""
    a = float(f32(age))
    leaf = _run("leaf_scale_ramp", {"intercept": 0.5, "slope": 0.5, "days": 10.0}, 0, 0, age)
    inode = _run("internode_scale_ramp", {"intercept": 0.7, "slope": 0.3, "days": 10.0}, 0, 0, age)
    assert leaf == [("leaf", f32(min(1.0, 0.5 + 0.5 * a / 10.0)))]
    assert inode == [("internode", f32(min(1.0, 0.7 + 0.3 * a / 10.0)))]


def test_check_params_defaults_and_errors():
    r = R.RULES["high_wire_string"]
    assert R.check_params(r, {}, "x", _raise) == {"view_half_angle_deg": 80.0, "look_ahead_distance": 0.15,
                                                  "attraction_weight": 0.8}
    with pytest.raises(ValueError, match="outside"):
        R.check_params(r, {"attraction_weight": 2.0}, "x", _raise)
    with pytest.raises(ValueError, match="expected float"):
        R.check_params(r, {"attraction_weight": "high"}, "x", _raise)


def test_library_function():
    from heliades.plantarchitecture import hooks
    fn = R.make("library_function", "creation", {"name": "TomatoPhytomerCreationFunction"})
    assert callable(fn) and hooks.TomatoPhytomerCreationFunction.impl is not None
    with pytest.raises(ValueError, match="no library function"):
        R.make("library_function", "creation", {"name": "NoSuchFunction"})


def _tomato_spec(creation):
    from heliades.species.loader import parse_spec
    return parse_spec({"format": "heliades-species/1", "name": "tomato_rules", "template": "tomato",
                       "metadata": {"latin_name": "Solanum lycopersicum", "common_name": "tomato"}, "sources": {},
                       "shoot_types": {"mainstem": {"phytomer_creation": copy.deepcopy(creation)}},
                       "build": {"builder": "template"}})


TOMATO_AS_RULES = [
    {"rule": "kill_buds", "which": "floral", "node_below": 8, "rank": 0, "source": "judged"},
    {"rule": "kill_buds", "which": "both", "rank_above": 1, "source": "judged"},
    {"rule": "leaf_scale_ramp", "intercept": 0.5, "slope": 0.5, "days": 10.0, "source": "judged"},
    {"rule": "internode_scale_ramp", "intercept": 0.7, "slope": 0.3, "days": 10.0, "source": "judged"},
]


@pytest.mark.parametrize("creation", [TOMATO_AS_RULES,
                                      [{"rule": "library_function", "name": "TomatoPhytomerCreationFunction",
                                        "source": "judged"}]])
def test_tomato_creation_function_as_rules(creation):
    """The tomato's creation function written as four generic rules (or by name) grows the library tomato exactly."""
    from heliades.core.scene import Scene
    from heliades.plantarchitecture import PlantArchitecture
    from heliades.species import register
    snaps = []
    for use_spec in (False, True):
        sc = Scene()
        sc.seed_random_generator(5)
        pa = PlantArchitecture(sc)
        if use_spec:
            register(pa, _tomato_spec(creation), "tomato")
        pa.loadPlantModelFromLibrary("tomato")
        pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 24.0)
        snaps.append((sc.snapshot(True), pa.writePlantStructureXML(pid)))
    (a, xa), (b, xb) = snaps
    assert np.array_equal(a.vertices, b.vertices) and np.array_equal(a.uuid, b.uuid) and xa == xb
