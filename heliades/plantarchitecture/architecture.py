# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp:382-403, 1225-1232 and
# src/PlantLibrary.cpp:444-520, 631-660, 755-880 (Copyright (C) 2016-2026 Brian Bailey).
"""``PlantArchitecture``: the plugin entry object (data layer only, W2-C).

It keeps a *pointer* to the Scene's engine (``generator = context_ptr->getRandomGenerator()``, PA:383):
the very same :class:`~heliades.core.random.MinstdRand0` object, so ``scene.seed_random_generator(s)``
before or after construction is equivalent (SPEC §3.4) and every ``RandomParameter`` initialised with
``self.generator`` draws from the Scene stream.

W3-B adds the minimal plant-building entry points (``addPlantInstance``, ``addBaseStemShoot``,
``breakPlantDormancy``, ``setPlantPhenologicalThresholds``, ``buildPlantInstanceFromLibrary`` at age 0); they
delegate to ``shoot_min``. Growth (``advanceTime``) is W3-C.
"""
from __future__ import annotations

from ..migration_registry import ports
from .api import PlantArchitectureAPI, install_snake_aliases
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


def _as_vec3(v):
    """A PyHelios-style ``vec3`` (``.x .y .z``) or any 3-sequence -> float32 (3,)."""
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        v = (v.x, v.y, v.z)
    return np.asarray(v, f32).reshape(3)


@install_snake_aliases
class PlantArchitecture(PlantArchitectureAPI):
    """helios PlantArchitecture. The query API (organ IDs, plant metrics, shoot topology, canopy build,
    ``plant_structure_dict``) and the snake_case aliases are in :mod:`.api` (W3-E)."""

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
        from . import shoot_min
        shoot_min.init_plant_state(self)
        # the inflorescence prototype caches (H:6700-6704), keyed by (prototype function, organ build parameters);
        # inflorescence._caches would create them lazily, here they exist from the start like the C++ members
        self.unique_closed_flower_prototype_objIDs = {}
        self.unique_open_flower_prototype_objIDs = {}
        self.unique_fruit_prototype_objIDs = {}

    # ---- registry -------------------------------------------------------------------------------

    @ports("PlantArchitecture::initializePlantModelRegistrations")
    def initializePlantModelRegistrations(self):
        from .library import register_all
        from .library.tomato import buildTomatoPlant
        register_all(self)
        # W3-B: the tomato builder (age-0 part). library/__init__.py still lists no builder (W3-E owns it).
        self.plant_builders["tomato"] = lambda pos: buildTomatoPlant(self, pos)

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

    # ---- plant building (W3-B minimal path; growth is W3-C) -------------------------------------

    def invalidateOrganIndex(self):
        self.organ_index_current = False

    @ports("PlantArchitecture::addPlantInstance")
    def addPlantInstance(self, base_position, current_age):
        from .shoot_min import addPlantInstance
        return addPlantInstance(self, _as_vec3(base_position), current_age)

    @ports("PlantArchitecture::addBaseStemShoot")
    def addBaseStemShoot(self, plantID, current_node_number, base_rotation, internode_radius, internode_length_max,
                         internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper,
                         shoot_type_label):
        from .shoot_min import addBaseStemShoot
        return addBaseStemShoot(self, plantID, current_node_number, base_rotation, internode_radius,
                                internode_length_max, internode_length_scale_factor_fraction,
                                leaf_scale_factor_fraction, radius_taper, shoot_type_label)

    @ports("PlantArchitecture::appendShoot")
    def appendShoot(self, plantID, parent_shoot_ID, current_node_number, base_rotation, internode_radius,
                    internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction,
                    radius_taper, shoot_type_label):
        from .shoot_min import appendShoot
        return appendShoot(self, plantID, parent_shoot_ID, current_node_number, base_rotation, internode_radius,
                           internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction,
                           radius_taper, shoot_type_label)

    @ports("PlantArchitecture::breakPlantDormancy")
    def breakPlantDormancy(self, plantID):
        from .shoot_min import breakPlantDormancy
        breakPlantDormancy(self, plantID)

    @ports("PlantArchitecture::setPlantPhenologicalThresholds")
    def setPlantPhenologicalThresholds(self, plantID, *args, **kw):
        from .shoot_min import setPlantPhenologicalThresholds
        setPlantPhenologicalThresholds(self, plantID, *args, **kw)

    @ports("PlantArchitecture::buildPlantInstanceFromLibrary")
    def buildPlantInstanceFromLibrary(self, base_position, age, build_parameters=None):
        """PL:653-760 for a model without build parameters, collision or leaf-inclination tracking."""
        if not self.current_plant_model:
            raise PlantArchitectureError("ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): current plant "
                                         "model has not been initialized from library. You must call "
                                         "loadPlantModelFromLibrary() first.")
        if self.current_plant_model not in self.plant_builders:
            raise PlantArchitectureError("ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): plant label of "
                                         f"{self.current_plant_model} does not exist in the library.")
        if build_parameters:
            raise NotImplementedError("build parameters are not ported (tomato declares none)")
        self.current_build_parameters = {}
        self.current_organ_build_parameters = {}
        self.current_build_plant_name = self.current_plant_model
        try:
            plantID = self.plant_builders[self.current_plant_model](_as_vec3(base_position))
        finally:
            self.current_build_parameters = {}
            self.current_organ_build_parameters = {}
            self.current_build_plant_name = ""
        self.plant_instances[plantID].plant_name = self.current_plant_model
        if self.output_object_data.get("plant_name") or self.output_object_data.get("plant_type"):
            raise NotImplementedError("optional plant_name / plant_type object data are not ported")
        if self.current_plant_model in self.plant_leaf_inclination_distribution_map:
            raise NotImplementedError("leaf inclination distribution tracking is not ported")
        if f32(age) > 0:
            self.advanceTime(plantID, age)
        return plantID

    # ---- growth (W3-C, growth.py) ------------------------------------------------------------

    @ports("PlantArchitecture::advanceTime")
    def advanceTime(self, *args, **kw):
        """advanceTime(days) (all plants), advanceTime(plantID or [plantIDs], days). The PyHelios keyword form
        ``advanceTime(dt, plant_id=..)`` / ``plant_ids=..`` / ``years=..`` is accepted too."""
        from .growth import advanceTime
        if kw:
            if len(kw) > 1 or len(args) != 1:
                raise ValueError("Pass at most one of plant_id, plant_ids, or years")
            (k, v), = kw.items()
            if k == "plant_id":
                return advanceTime(self, [int(v)], args[0])
            if k == "plant_ids":
                return advanceTime(self, [int(p) for p in v], args[0]) if v else None
            if k == "years":
                return self.advanceTimeYears(v, args[0])
            raise TypeError(f"advanceTime() got an unexpected keyword argument '{k}'")
        if len(args) == 1:
            return advanceTime(self, sorted(self.plant_instances), args[0])
        a, b = args
        if isinstance(a, (list, tuple)):
            return advanceTime(self, list(a), b)
        return advanceTime(self, [int(a)], b)

    def advanceTimeYears(self, time_step_years: int, time_step_days):
        """advanceTime(int years, float days): float(years) * 365.f + days."""
        from .growth import advanceTime
        return advanceTime(self, sorted(self.plant_instances), f32(f32(f32(time_step_years) * f32(365)) + f32(time_step_days)))

    @ports("PlantArchitecture::addChildShoot")
    def addChildShoot(self, plantID, parent_shoot_ID, parent_node_index, current_node_number, shoot_base_rotation,
                      internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                      leaf_scale_factor_fraction, radius_taper, shoot_type_label, petiole_index=0):
        from .growth import addChildShoot
        return addChildShoot(self, plantID, parent_shoot_ID, parent_node_index, current_node_number, shoot_base_rotation,
                             internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                             leaf_scale_factor_fraction, radius_taper, shoot_type_label, petiole_index)

    @ports("PlantArchitecture::appendPhytomerToShoot")
    def appendPhytomerToShoot(self, plantID, shootID, phytomer_parameters, internode_radius, internode_length_max,
                              internode_length_scale_factor_fraction, leaf_scale_factor_fraction):
        from .growth import appendPhytomerToShoot
        return appendPhytomerToShoot(self, plantID, shootID, phytomer_parameters, internode_radius, internode_length_max,
                                     internode_length_scale_factor_fraction, leaf_scale_factor_fraction)

    @ports("PlantArchitecture::incrementPhytomerInternodeGirth")
    def incrementPhytomerInternodeGirth(self, plantID, shootID, node_number, dt, update_context_geometry):
        from .growth import incrementPhytomerInternodeGirth
        incrementPhytomerInternodeGirth(self, plantID, shootID, node_number, dt, update_context_geometry)


__all__ = ["PlantArchitecture", "context_generator", "vec3"]
