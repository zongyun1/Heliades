# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:444-505 (Copyright (C) 2016-2026 Brian Bailey).
"""The plant library registry (``initializePlantModelRegistrations``).

Every Helios model is registered, in the C++ order and with its plant type and leaf-inclination Beta,
so ``getAvailablePlantModels()`` matches Helios. Only ``tomato`` has a ported shoot initializer; the
others raise ``NotImplementedError`` when loaded. No builder is ported yet (W3-E).
"""
from __future__ import annotations

from . import tomato

# (name, plant_type, leaf_inclination_Beta) in PlantLibrary.cpp:446-505 order
MODELS = (
    ("almond", "tree", None), ("almond_aldrich", "tree", None), ("almond_independence", "tree", None),
    ("almond_wood_colony", "tree", None), ("apple", "tree", None), ("apple_fruitingwall", "tree", None),
    ("asparagus", "herbaceous", None), ("bindweed", "weed", None), ("bean", "herbaceous", None),
    ("bougainvillea", "herbaceous", None), ("capsicum", "herbaceous", None), ("cheeseweed", "weed", None),
    ("cowpea", "herbaceous", (1.398, 1.574)), ("grapevine_VSP", "herbaceous", None),
    ("grapevine_Wye", "herbaceous", None), ("groundcherryweed", "weed", None), ("maize", "herbaceous", None),
    ("olive", "tree", None), ("pistachio", "tree", (1.32, 1.85)), ("puncturevine", "weed", None),
    ("easternredbud", "tree", (1.00, 2.20)), ("rice", "herbaceous", None), ("butterlettuce", "herbaceous", None),
    ("sorghum", "herbaceous", None), ("soybean", "herbaceous", None), ("strawberry", "herbaceous", None),
    ("sugarbeet", "herbaceous", None), ("tomato", "herbaceous", None), ("cherrytomato", "herbaceous", None),
    ("walnut", "tree", None), ("wheat", "herbaceous", None), ("cucumber", "herbaceous", None),
)

# Ported shoot initializers: name -> fn(plant_architecture)
SHOOT_INITIALIZERS = {"tomato": tomato.initializeTomatoShoots}
# Ported plant builders: name -> fn(plant_architecture, base_position) -> plantID (W3-E)
PLANT_BUILDERS: dict = {}


def _not_ported(what, name):
    def fn(*a, **k):
        raise NotImplementedError(f"{what} for plant model '{name}' is not ported to Heliades yet")
    return fn


def register_all(pa):
    """PlantArchitecture::initializePlantModelRegistrations (structure part; build-parameter
    declarations, PL:507-520, are not ported: tomato declares none)."""
    for name, plant_type, beta in MODELS:
        init = SHOOT_INITIALIZERS.get(name)
        build = PLANT_BUILDERS.get(name)
        pa.registerPlantModel(name,
                              (lambda f=init: f(pa)) if init else _not_ported("shoot initializer", name),
                              (lambda pos, f=build: f(pa, pos)) if build else _not_ported("plant builder", name),
                              plant_type, beta if beta is not None else (0.0, 0.0))
