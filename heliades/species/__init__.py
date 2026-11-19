# SPDX-License-Identifier: LGPL-2.1-or-later
"""Declarative species: plant models defined by a data file on top of a Helios library model.

    from heliades.plantarchitecture import PlantArchitecture
    pa = PlantArchitecture(scene)
    pa.loadPlantModelFromLibrary("eggplant")          # heliades/species/defs/eggplant.yaml, registered on first use
    pid = pa.buildPlantInstanceFromLibrary((0, 0, 0), 60.0)

* :mod:`.loader`: the file format (``docs/species/FORMAT.md``), validation, registration;
* :mod:`.rules`: the named growth rules a file may use;
* :mod:`.organs`: flower / fruit prototypes (library, OBJ, procedural);
* :mod:`.leaf_texture`: procedural leaf textures (outline -> RGBA PNG);
* :mod:`.traits`: the plausibility harness (expected trait ranges).

``docs/species/ADDING.md`` is the how-to.
"""
from __future__ import annotations

from .loader import (SpeciesError, SpeciesSpec, available_species, find_species_file, load_spec, register,  # noqa: F401
                     register_if_species, validate)

__all__ = ["SpeciesError", "SpeciesSpec", "available_species", "find_species_file", "load_spec", "register",
           "register_if_species", "validate"]
