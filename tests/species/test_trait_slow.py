# SPDX-License-Identifier: LGPL-2.1-or-later
"""The optional ``slow`` key of a trait (S1): parsed as a bool, defaults to false, marks the trait test slow."""
from __future__ import annotations

import copy

import pytest
import yaml

from heliades.species import loader as L


def _raw():
    with open(L.find_species_file("barley")) as f:
        return yaml.safe_load(f)


def test_slow_trait_key():
    raw = _raw()
    assert all(t["slow"] is False for t in L.parse_spec(copy.deepcopy(raw)).traits)
    raw["traits"][0]["slow"] = True
    assert L.parse_spec(copy.deepcopy(raw)).traits[0]["slow"] is True
    raw["traits"][0]["slow"] = "yes"
    with pytest.raises(L.SpeciesError, match="slow"):
        L.parse_spec(raw)


def test_slow_traits_are_marked_in_the_trait_test():
    from tests.species import test_traits as T
    marked = {(p.values[0], p.values[1]) for p in T.CASES if any(m.name == "slow" for m in p.marks)}
    expected = {(n, i) for n in L.available_species() for i, t in enumerate(L.load_spec(n).traits) if t["slow"]}
    assert marked == expected
