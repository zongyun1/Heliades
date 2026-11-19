# SPDX-License-Identifier: LGPL-2.1-or-later
"""Plausibility harness: every trait a species file expects is inside its range (fast mode, the listed seeds)."""
from __future__ import annotations

import pytest

from heliades.species import available_species, load_spec
from heliades.species.traits import TRAITS, measure

CASES = [pytest.param(n, i, marks=[pytest.mark.slow] if t.get("slow") else [])
         for n in available_species() for i, t in enumerate(load_spec(n).traits)]


@pytest.mark.parametrize("name,index", CASES)
def test_trait_in_expected_range(name, index):
    t = load_spec(name).traits[index]
    (r,) = measure(name, [t])
    assert r.ok, r.row()


def test_pilot_species_list_traits():
    for name in ("eggplant", "barley"):
        traits = load_spec(name).traits
        assert len(traits) >= 6
        assert {t["trait"] for t in traits} >= {"plant_height", "leaf_length_max"}


def test_trait_functions_on_a_library_plant():
    from heliades.plantarchitecture.batch import build_plant
    sc, pa, pid = build_plant("tomato", 30.0, seed=1)
    for name, (units, fn) in TRAITS.items():
        if fn is not None:
            assert fn(sc, pa, pid) >= 0, name
    assert TRAITS["plant_height"][1](sc, pa, pid) == pytest.approx(float(pa.getPlantHeight(pid)))
