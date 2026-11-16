# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp:382-403, 1225-1232 and
# src/PlantLibrary.cpp:444-520, 631-660, 755-880 (Copyright (C) 2016-2026 Brian Bailey).
"""``PlantArchitecture``: the plugin entry object (data layer only, W2-C).

It keeps a *pointer* to the Scene's engine (``generator = context_ptr->getRandomGenerator()``, PA:383):
the very same :class:`~heliades.core.random.MinstdRand0` object, so ``scene.seed_random_generator(s)``
before or after construction is equivalent (SPEC §3.4) and every ``RandomParameter`` initialised with
``self.generator`` draws from the Scene stream.

Growth (``buildPlantInstanceFromLibrary``, ``advanceTime``, ...) belongs to W3 and is not here.
"""
from __future__ import annotations

from ..migration_registry import ports
from .params import ShootParameters, vec3
from .random_parameter import PlantArchitectureError

import numpy as np

f32 = np.float32


def context_generator(scene):
    """``Context::getRandomGenerator()``: the Scene's engine object itself (never a copy)."""
    getter = getattr(scene, "get_random_generator", None)
    if getter is not None:
        return getter()
    return scene.rng.generator


class PlantArchitecture:
    """helios PlantArchitecture (parameter / library part)."""

    @ports("PlantArchitecture::PlantArchitecture")
    def __init__(self, context):
        self.context_ptr = context
        self.generator = context_generator(context)
        self.current_plant_model = ""
        # std::map members: Python dicts, iterated through sorted() wherever the C++ iterates the map
        self.shoot_initializers: dict = {}
        self.plant_builders: dict = {}
        self.plant_type_map: dict = {}
        self.plant_leaf_inclination_distribution_map: dict = {}
        self.shoot_types: dict[str, ShootParameters] = {}
        self.initializePlantModelRegistrations()
        self.output_object_data = {k: False for k in (
            "age", "rank", "plantID", "plant_name", "plant_height", "plant_type", "phenology_stage", "leafID",
            "peduncleID", "closedflowerID", "openflowerID", "fruitID", "carbohydrate_concentration",
            "leaf_texture_stage")}

    # ---- registry -------------------------------------------------------------------------------

    @ports("PlantArchitecture::initializePlantModelRegistrations")
    def initializePlantModelRegistrations(self):
        from .library import register_all
        register_all(self)

    @ports("PlantArchitecture::registerPlantModel")
    def registerPlantModel(self, name, shoot_init, plant_build, plant_type="herbaceous", leaf_inclination_Beta=(0.0, 0.0)):
        self.shoot_initializers[name] = shoot_init
        self.plant_builders[name] = plant_build
        self.plant_type_map[name] = plant_type
        bx, by = f32(leaf_inclination_Beta[0]), f32(leaf_inclination_Beta[1])
        declares_distribution = bx > 0 and by > 0
        declares_nothing = bx == 0 and by == 0
        if not declares_distribution and not declares_nothing:
            raise PlantArchitectureError(
                f"ERROR (PlantArchitecture::registerPlantModel): Plant model '{name}' declares an incomplete leaf "
                f"inclination distribution of ({bx:f}, {by:f}). Both Beta parameters must be positive to declare a "
                "distribution, or both zero to declare none.")
        if declares_distribution:
            self.plant_leaf_inclination_distribution_map[name] = (bx, by)

    @ports("PlantArchitecture::getAvailablePlantModels")
    def getAvailablePlantModels(self) -> list[str]:
        return sorted(self.shoot_initializers)

    @ports("PlantArchitecture::getPlantModelLeafInclinationDistribution")
    def getPlantModelLeafInclinationDistribution(self, name):
        if name not in self.plant_builders:
            raise PlantArchitectureError("ERROR (PlantArchitecture::getPlantModelLeafInclinationDistribution): plant "
                                         f"label of {name} does not exist in the library.")
        return self.plant_leaf_inclination_distribution_map.get(name, (f32(0), f32(0)))

    # ---- loading --------------------------------------------------------------------------------

    @ports("PlantArchitecture::loadPlantModelFromLibrary")
    def loadPlantModelFromLibrary(self, plant_label: str):
        # skipStructuralOperationInCallback: phytomer callbacks are not ported yet, so never inside one.
        self.current_plant_model = plant_label
        self.initializeDefaultShoots(plant_label)

    @ports("PlantArchitecture::initializeDefaultShoots")
    def initializeDefaultShoots(self, plant_label: str):
        self.shoot_types.clear()
        init = self.shoot_initializers.get(plant_label)
        if init is None:
            raise PlantArchitectureError("ERROR (PlantArchitecture::loadPlantModelFromLibrary): plant label of "
                                         f"{plant_label} does not exist in the library.")
        init()

    @ports("PlantArchitecture::defineShootType")
    def defineShootType(self, shoot_type_label: str, shoot_params: ShootParameters):
        """Existing label: ShootParameters::operator=; new label: map emplace = copy constructor. No draws."""
        if shoot_type_label in self.shoot_types:
            self.shoot_types[shoot_type_label]._assign_from(shoot_params)
        else:
            self.shoot_types[shoot_type_label] = shoot_params.copy()

    @ports("PlantArchitecture::getCurrentShootParameters")
    def getCurrentShootParameters(self, shoot_type_label: str | None = None):
        """By value, as in C++ (a copy; the cache state travels with it)."""
        if shoot_type_label is None:
            if not self.shoot_types:
                import sys
                print("WARNING (PlantArchitecture::getCurrentShootParameters): No plant models have been loaded. You "
                      "need to first load a plant model from the library (see loadPlantModelFromLibrary()) or manually "
                      "add shoot parameters (see updateCurrentShootParameters()).", file=sys.stderr)
            return {k: v.copy() for k, v in self.shoot_types.items()}
        if shoot_type_label not in self.shoot_types:
            raise PlantArchitectureError("ERROR (PlantArchitecture::getCurrentShootParameters): shoot type label of "
                                         f"{shoot_type_label} does not exist in the current shoot parameters.")
        return self.shoot_types[shoot_type_label].copy()

    @ports("PlantArchitecture::updateCurrentShootParameters")
    def updateCurrentShootParameters(self, label_or_map, params: ShootParameters | None = None):
        """``(label, params)``: ``shoot_types[label] = params`` (default-construct if absent, then operator=);
        ``(map)``: map copy assignment. Neither draws."""
        if params is not None:
            self.shoot_types.setdefault(label_or_map, ShootParameters())._assign_from(params)
        else:
            self.shoot_types = {k: v.copy() for k, v in label_or_map.items()}

    def getRandomGenerator(self):
        return self.generator

    # ---- not yet ported (W3) --------------------------------------------------------------------

    @ports("PlantArchitecture::buildPlantInstanceFromLibrary")
    def buildPlantInstanceFromLibrary(self, base_position, age, build_parameters=None):
        if not self.current_plant_model:
            raise PlantArchitectureError("ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): current plant "
                                         "model has not been initialized from library. You must call "
                                         "loadPlantModelFromLibrary() first.")
        if self.current_plant_model not in self.plant_builders:
            raise PlantArchitectureError("ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): plant label of "
                                         f"{self.current_plant_model} does not exist in the library.")
        raise NotImplementedError("buildPlantInstanceFromLibrary is W3-E (needs the Phytomer/Shoot ports)")


__all__ = ["PlantArchitecture", "context_generator", "vec3"]
