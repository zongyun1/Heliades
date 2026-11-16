# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp (Copyright (C) 2016-2026 Brian Bailey):
#   Shoot::Shoot                         PA:6951-6973
#   Shoot::buildShootPhytomers           PA:6975-6991
#   Shoot::sampleChildShootType          PA:7013-7063
#   Shoot::sampleVegetativeBudBreak      PA:7065-7102
#   Shoot::appendPhytomer(Internal)      PA:2007-2219 (non-prescribed path)
#   Shoot::breakDormancy                 PA:2221-2262
#   Shoot::propagateDownstreamLeafArea   PA:2550-2561
#   PlantArchitecture::addPlantInstance  PA:11410-11431
#   PlantArchitecture::addBaseStemShoot  PA:7146-7173
#   PlantArchitecture::validateShootTypes PA:7512-7521
#   PlantArchitecture::breakPlantDormancy PA:10069-10081
#   PlantArchitecture::setPlantPhenologicalThresholds PA:11669-11688
"""The minimal Shoot / PlantInstance path a plant needs at age 0 (W3-B).

Only what ``buildTomatoPlant`` reaches before ``advanceTime``: one base-stem shoot built with its initial
phytomer(s), bud sampling, the phytomer creation callback and ``breakPlantDormancy``. The growth half of
``Shoot`` (elongation, girth, ``updateShootNodes``, child shoots, the phyllochron) is W3-C, which should
extend these classes rather than replace them.
"""
from __future__ import annotations

import numpy as np

from ..migration_registry import ports
from .params import BUD_ACTIVE, BUD_DEAD, BUD_DORMANT, ShootParameters
from .random_parameter import PlantArchitectureError
from .tube_utils import NO_INTERNODE_TUBE_OBJID

f32 = np.float32


class ShootGrowthState:
    """Shoot::growth_state (H, per-shoot overrides of the shoot type)."""

    def __init__(self):
        self.growth_type_label = ""
        self.max_nodes_growth_cap = -1
        self.max_nodes_per_season_growth_cap = -1
        self.internode_radius_cap = f32(0)
        self.geometry_is_prescribed = False


class PlantInstance:
    """PlantInstance (H:3366-3615), members used up to age 0."""

    def __init__(self, base_position, current_age, plant_name: str, context_ptr):
        self.base_position = np.asarray(base_position, f32).reshape(3).copy()
        self.current_age = f32(current_age)
        self.plant_name = plant_name
        self.context_ptr = context_ptr
        self.shoot_tree: list = []
        self.time_since_dormancy = f32(0)
        self.shoot_types_snapshot: dict[str, ShootParameters] = {}
        self.build_parameters: dict = {}
        self.organ_build_parameters: dict = {}
        self.dd_to_dormancy_break = f32(0)
        self.dd_to_flower_initiation = f32(0)
        self.dd_to_flower_opening = f32(0)
        self.dd_to_fruit_set = f32(0)
        self.dd_to_fruit_maturity = f32(0)
        self.dd_to_dormancy = f32(0)
        self.max_leaf_lifespan = f32(1e6)
        self.is_evergreen = False
        self.max_age = f32(999)
        self.minimum_leaf_texture_stage = 0


class Shoot:
    """helios Shoot (construction and the age-0 bud logic)."""

    @ports("Shoot::Shoot")
    def __init__(self, plant_ID, shoot_ID, parent_shoot_ID, parent_node, parent_petiole_index, rank, shoot_base_position,
                 shoot_base_rotation, current_node_number, internode_length_shoot_initial, shoot_params: ShootParameters,
                 shoot_type_label: str, plant_architecture_ptr):
        self.current_node_number = int(current_node_number)
        self.base_position = np.asarray(shoot_base_position, f32).reshape(3).copy()
        self.base_rotation = shoot_base_rotation.copy()
        self.ID = int(shoot_ID)
        self.parent_shoot_ID = int(parent_shoot_ID)
        self.plantID = int(plant_ID)
        self.parent_node_index = int(parent_node)
        self.rank = int(rank)
        self.parent_petiole_index = int(parent_petiole_index)
        self.internode_length_max_shoot_initial = f32(internode_length_shoot_initial)
        self.shoot_parameters = shoot_params.copy()          # member copy-constructed from the argument
        self.shoot_type_label = shoot_type_label
        self.plantarchitecture_ptr = plant_architecture_ptr
        # in-class defaults (H:2968-3364)
        self.nodes_this_season = 0
        self.dormancy_cycles = 0
        self.meristem_is_alive = True
        self.curvature_perturbation = f32(0)
        self.yaw_perturbation = f32(0)
        self.internode_tube_objID = NO_INTERNODE_TUBE_OBJID
        self.shoot_internode_vertices: list = []
        self.shoot_internode_radii: list = []
        self.phytomers: list = []
        self.childIDs: dict[int, list[int]] = {}
        self.radial_outward_axis = np.zeros(3, f32)
        self.initial_heading_horizontal = np.zeros(3, f32)
        self.growth_state = ShootGrowthState()
        self.is_axis_continuation = False
        # constructor body
        self.sugar_pool_molC = f32(0)
        self.phyllochron_counter = f32(0)
        self.isdormant = True
        # shoot_params.gravitropic_curvature.val(): the *argument*, so a first draw caches on the caller's copy
        self.gravitropic_curvature = shoot_params.gravitropic_curvature.val()
        self.context_ptr = plant_architecture_ptr.context_ptr
        self.phyllochron_instantaneous = self.shoot_parameters.phyllochron_min.val()
        self.elongation_rate_instantaneous = self.shoot_parameters.elongation_rate_max.val()
        leaf_rate = self.shoot_parameters.leaf_expansion_rate_max.val()
        self.leaf_expansion_rate_instantaneous = self.elongation_rate_instantaneous if leaf_rate < 0 else leaf_rate
        if self.parent_shoot_ID >= 0:
            parent = plant_architecture_ptr.plant_instances[self.plantID].shoot_tree[self.parent_shoot_ID]
            parent.childIDs.setdefault(self.parent_node_index, []).append(self.ID)

    @ports("Shoot::buildShootPhytomers")
    def buildShootPhytomers(self, internode_radius, internode_length, internode_length_scale_factor_fraction,
                            leaf_scale_factor_fraction, radius_taper):
        for i in range(self.current_node_number):
            taper = f32(1)
            if self.current_node_number > 1:
                taper = f32(f32(1) - f32(f32(f32(radius_taper) * f32(i)) / f32(self.current_node_number - 1)))
            self.appendPhytomer(f32(f32(internode_radius) * taper), internode_length,
                                internode_length_scale_factor_fraction, leaf_scale_factor_fraction,
                                self.shoot_parameters.phytomer_parameters)

    def getGrowthTypeLabel(self) -> str:
        return self.growth_state.growth_type_label or self.shoot_type_label

    def getGrowthNodeCap(self) -> int:
        if self.growth_state.max_nodes_growth_cap >= 0:
            return self.growth_state.max_nodes_growth_cap
        return int(self.shoot_parameters.max_nodes.val())

    @ports("Shoot::sampleChildShootType")
    def sampleChildShootType(self) -> str:
        label = self.getGrowthTypeLabel()
        gp = self.shoot_parameters
        if label != self.shoot_type_label:
            snap = self.plantarchitecture_ptr.plant_instances[self.plantID].shoot_types_snapshot
            if label not in snap:
                raise PlantArchitectureError(f"ERROR (Shoot::sampleChildShootType): Shoot {self.ID} grows as shoot type "
                                             f"{label}, which no longer exists for this plant.")
            gp = snap[label]
        labels, probs = gp.child_shoot_type_labels, gp.child_shoot_type_probabilities
        if not labels:
            return label
        if len(labels) == 1:
            return labels[0]
        randf = self.context_ptr.randu()
        index = -1
        cumulative = f32(0)
        for s in range(len(labels)):
            cumulative = f32(cumulative + probs[s])
            if randf < cumulative:
                index = s
                break
        if index < 0:
            index = len(labels) - 1
        return labels[index]

    @ports("Shoot::sampleVegetativeBudBreak")
    def sampleVegetativeBudBreak(self, node_index: int) -> bool:
        if node_index >= len(self.phytomers):
            raise PlantArchitectureError("ERROR (PlantArchitecture::sampleVegetativeBudBreak): Invalid node index.")
        sp = self.shoot_parameters
        pmin = sp.vegetative_bud_break_probability_min.val()
        pmax = sp.vegetative_bud_break_probability_max.val()
        decay = sp.vegetative_bud_break_probability_decay_rate.val()
        if not sp.growth_requires_dormancy and decay < 0:
            p = pmin
        elif decay > 0:
            p = max(pmin, f32(pmax - f32(decay * f32(self.current_node_number - node_index - 1))))   # std::fmax(float, float)
        elif decay < 0:
            # fabs() (C, double) promotes the right-hand side to double; fmax(float, double) -> double -> float
            p = f32(max(float(pmin), float(pmax) - abs(float(decay)) * float(f32(node_index))))
        else:
            p = pmin if decay == 0 else pmax
        return not (self.context_ptr.randu() > p)

    def appendPhytomer(self, internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                       leaf_scale_factor_fraction, phytomer_parameters) -> int:
        return self.appendPhytomerInternal(internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                                           leaf_scale_factor_fraction, phytomer_parameters, None)

    @ports("Shoot::appendPhytomer", "Shoot::appendPhytomerInternal")
    def appendPhytomerInternal(self, internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                               leaf_scale_factor_fraction, phytomer_parameters, prescribed_internode=None) -> int:
        from .phytomer import Phytomer
        if prescribed_internode is not None:
            raise NotImplementedError("appendPhytomerFromNodePositions is not ported")
        pa = self.plantarchitecture_ptr
        shoot_tree = pa.plant_instances[self.plantID].shoot_tree
        if not self.phytomers:
            if self.parent_shoot_ID == -1:
                parent_internode_axis = np.array([0, 0, 1], f32)
                parent_petiole_axis = np.array([0, -1, 0], f32)
            else:
                parent = shoot_tree[self.parent_shoot_ID].phytomers[self.parent_node_index]
                parent_internode_axis = parent.getInternodeAxisVector(f32(1))
                parent_petiole_axis = self._parent_petiole_axis(parent, parent_internode_axis, self.parent_petiole_index)
            internode_base_position = self.base_position.copy()
        else:
            last = self.phytomers[-1]
            parent_internode_axis = last.getInternodeAxisVector(f32(1))
            parent_petiole_axis = self._parent_petiole_axis(last, parent_internode_axis, 0)
            internode_base_position = self.shoot_internode_vertices[-1][-1].copy()

        phytomer = Phytomer(phytomer_parameters, self, len(self.phytomers), parent_internode_axis, parent_petiole_axis,
                            internode_base_position, self.base_rotation, internode_radius, internode_length_max,
                            internode_length_scale_factor_fraction, leaf_scale_factor_fraction, self.rank, pa,
                            self.context_ptr)
        shoot_tree[self.ID].phytomers.append(phytomer)

        for petiole in phytomer.axillary_vegetative_buds:
            child_label = self.sampleChildShootType()
            for vbud in petiole:
                phytomer.setVegetativeBudState(BUD_DORMANT, vbud)
                vbud.shoot_type_label = child_label
                if not self.shoot_parameters.growth_requires_dormancy:
                    if pa.carbon_model_enabled:
                        raise NotImplementedError("carbohydrate model is not ported")
                    if self.sampleVegetativeBudBreak(phytomer.shoot_index[0]):
                        phytomer.setVegetativeBudState(BUD_ACTIVE, vbud)
                    else:
                        phytomer.setVegetativeBudState(BUD_DEAD, vbud)

        for petiole_index, petiole in enumerate(phytomer.floral_buds):
            for bud_index, fbud in enumerate(petiole):
                phytomer.setFloralBudState(BUD_DORMANT, fbud)
                if not self.shoot_parameters.flowers_require_dormancy and fbud.state != BUD_DEAD:
                    phytomer.setFloralBudState(BUD_ACTIVE, fbud)
                fbud.parent_index = petiole_index
                fbud.bud_index = bud_index

        self.propagateDownstreamLeafArea(self, phytomer.shoot_index[0], phytomer.getLeafArea())

        phytomer.age = f32(0)
        if any(pa.output_object_data.get(k) for k in ("age", "rank", "plantID", "plant_name", "leafID",
                                                       "leaf_texture_stage")):
            raise NotImplementedError("optional output object data is not ported")
        self._update_leaf_texture_stage(phytomer)

        fn = phytomer_parameters.phytomer_creation_function
        if fn is not None:
            pa.phytomer_callback_depth += 1
            try:
                fn(phytomer, int(phytomer.shoot_index[0]), self.parent_node_index, int(self.shoot_parameters.max_nodes.val()),
                   pa.plant_instances[self.plantID].current_age)
            finally:
                pa.phytomer_callback_depth -= 1
        return len(self.phytomers) - 1

    @staticmethod
    def _parent_petiole_axis(parent, parent_internode_axis, petiole_index):
        from ..core import vec as V
        if not parent.petiole_vertices:
            axis = V.cross(parent_internode_axis, (0, 0, 1))
            if V.magnitude(axis) < f32(0.01):
                axis = np.array([0, 1, 0], f32)
            axis = V.normalize(axis)
            return V.rotate_point_about_line(axis, np.zeros(3, f32), parent_internode_axis,
                                             parent.internode_phyllotactic_azimuth)
        return parent.getPetioleAxisVector(f32(0), petiole_index)

    @staticmethod
    def _update_leaf_texture_stage(phytomer):
        """Phytomer::updateLeafTextureStage (PA:4855): a no-op without texture stages and stage output."""
        if len(phytomer.phytomer_parameters.leaf.prototype.leaf_texture_stage_ages) >= 2:
            raise NotImplementedError("leaf texture stages are not ported")

    @ports("Shoot::propagateDownstreamLeafArea")
    def propagateDownstreamLeafArea(self, shoot, node_index: int, leaf_area):
        for i in range(node_index, -1, -1):
            ph = shoot.phytomers[i]
            ph.downstream_leaf_area = f32(ph.downstream_leaf_area + f32(leaf_area))
            ph.downstream_leaf_area = max(f32(0), ph.downstream_leaf_area)
        if shoot.parent_shoot_ID >= 0:
            parent = self.plantarchitecture_ptr.plant_instances[shoot.plantID].shoot_tree[shoot.parent_shoot_ID]
            self.propagateDownstreamLeafArea(parent, shoot.parent_node_index, leaf_area)

    @ports("Shoot::breakDormancy")
    def breakDormancy(self):
        pa = self.plantarchitecture_ptr
        self.isdormant = False
        for phytomer_ind, phytomer in enumerate(self.phytomers):
            for petiole in phytomer.floral_buds:
                for fbud in petiole:
                    if fbud.state != BUD_DEAD:
                        phytomer.setFloralBudState(BUD_ACTIVE, fbud)
                    fbud.time_counter = 0
            for petiole in phytomer.axillary_vegetative_buds:
                for vbud in petiole:
                    if vbud.state != BUD_DEAD:
                        if pa.carbon_model_enabled:
                            raise NotImplementedError("carbohydrate model is not ported")
                        if self.sampleVegetativeBudBreak(phytomer_ind):
                            phytomer.setVegetativeBudState(BUD_ACTIVE, vbud)
                        else:
                            phytomer.setVegetativeBudState(BUD_DEAD, vbud)
            phytomer.isdormant = False


# ---- PlantArchitecture entry points (bound in architecture.py) -----------------------------------

def init_plant_state(pa):
    """Members of PlantArchitecture the plant-building path reads (H:6600-7110 defaults)."""
    pa.plant_instances = {}
    pa.plant_count = 0
    pa.current_build_plant_name = ""
    pa.current_build_parameters = {}
    pa.current_organ_build_parameters = {}
    pa.unique_leaf_prototype_objIDs = {}           # (identifier, leaves_per_petiole) -> [prototype][leaf] objID
    pa.unique_leaf_prototype_rest_geometry = {}
    pa.organ_material_labels = {}                  # (automatic label, base name) -> OrganMaterial
    pa.build_context_geometry_internode = True
    pa.build_context_geometry_petiole = True
    pa.build_context_geometry_peduncle = True
    pa.carbon_model_enabled = False
    pa.nitrogen_model_enabled = False
    pa.collision_detection_enabled = False
    pa.petiole_collision_detection_enabled = False
    pa.attraction_points_enabled = False
    pa.solid_obstacle_avoidance_enabled = False
    pa.phytomer_callback_depth = 0
    pa.organ_index_current = False
    pa.leaf_geometry_is_stub = False
    pa.leaf_prototype_function_override = None


@ports("PlantArchitecture::addPlantInstance")
def addPlantInstance(pa, base_position, current_age) -> int:
    if f32(current_age) < 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::addPlantInstance): Current age must be greater than or "
                                     "equal to zero.")
    name = pa.current_build_plant_name or "custom"
    inst = PlantInstance(base_position, current_age, name, pa.context_ptr)
    pid = pa.plant_count
    pa.plant_instances[pid] = inst
    inst.shoot_types_snapshot = {k: v.copy() for k, v in pa.shoot_types.items()}     # map copy, no draws
    inst.build_parameters = dict(pa.current_build_parameters)
    inst.organ_build_parameters = dict(pa.current_organ_build_parameters)
    pa.plant_count += 1
    return pid


@ports("PlantArchitecture::validateShootTypes")
def validateShootTypes(shoot_parameters, shoot_types_ref):
    labels, probs = shoot_parameters.child_shoot_type_labels, shoot_parameters.child_shoot_type_probabilities
    for ind in range(len(labels) - 1, -1, -1):
        if labels[ind] not in shoot_types_ref:
            del labels[ind]
            del probs[ind]


@ports("PlantArchitecture::addBaseStemShoot")
def addBaseStemShoot(pa, plantID, current_node_number, base_rotation, internode_radius, internode_length_max,
                     internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper,
                     shoot_type_label) -> int:
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::addBaseStemShoot): Plant with ID of {plantID} does not "
                                     "exist.")
    inst = pa.plant_instances[plantID]
    if shoot_type_label not in inst.shoot_types_snapshot:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::addBaseStemShoot): Shoot type with label of "
                                     f"{shoot_type_label} does not exist.")
    shoot_parameters = inst.shoot_types_snapshot[shoot_type_label].copy()
    validateShootTypes(shoot_parameters, inst.shoot_types_snapshot)
    if current_node_number > shoot_parameters.max_nodes.val():
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::addBaseStemShoot): Cannot add shoot with "
                                     f"{current_node_number} nodes since the specified max node number is "
                                     f"{shoot_parameters.max_nodes.val()}.")
    shootID = len(inst.shoot_tree)
    shoot = Shoot(plantID, shootID, -1, 0, 0, 0, inst.base_position, base_rotation, current_node_number,
                  internode_length_max, shoot_parameters, shoot_type_label, pa)
    inst.shoot_tree.append(shoot)
    shoot.buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                              leaf_scale_factor_fraction, radius_taper)
    return shootID


@ports("PlantArchitecture::breakPlantDormancy")
def breakPlantDormancy(pa, plantID):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::breakPlantDormancy): Plant with ID of {plantID} does "
                                     "not exist.")
    for shoot in pa.plant_instances[plantID].shoot_tree:
        shoot.breakDormancy()
        if pa.carbon_model_enabled:
            raise NotImplementedError("carbohydrate model is not ported")


@ports("PlantArchitecture::setPlantPhenologicalThresholds")
def setPlantPhenologicalThresholds(pa, plantID, time_to_dormancy_break, time_to_flower_initiation,
                                   time_to_flower_opening, time_to_fruit_set, time_to_fruit_maturity, time_to_dormancy,
                                   max_leaf_lifespan=0, is_evergreen=False):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::setPlantPhenologicalThresholds): Plant with ID of "
                                     f"{plantID} does not exist.")
    p = pa.plant_instances[plantID]
    p.dd_to_dormancy_break = f32(time_to_dormancy_break)
    p.dd_to_flower_initiation = f32(time_to_flower_initiation)
    p.dd_to_flower_opening = f32(time_to_flower_opening)
    p.dd_to_fruit_set = f32(time_to_fruit_set)
    p.dd_to_fruit_maturity = f32(time_to_fruit_maturity)
    p.dd_to_dormancy = f32(time_to_dormancy)
    p.max_leaf_lifespan = f32(1e6) if f32(max_leaf_lifespan) == 0 else f32(max_leaf_lifespan)
    p.is_evergreen = bool(is_evergreen)


__all__ = ["Shoot", "PlantInstance", "ShootGrowthState", "init_plant_state", "addPlantInstance", "addBaseStemShoot",
           "breakPlantDormancy", "setPlantPhenologicalThresholds", "validateShootTypes"]
