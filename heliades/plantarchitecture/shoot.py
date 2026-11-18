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
#   Shoot::updateShootNodes / childShootOutwardAxis / terminateApicalBud / ... PA:2264-2547, 2598-2640 (W3-C)
"""The minimal Shoot / PlantInstance path a plant needs at age 0 (W3-B).

Only what ``buildTomatoPlant`` reaches before ``advanceTime``: one base-stem shoot built with its initial
phytomer(s), bud sampling, the phytomer creation callback and ``breakPlantDormancy``. W3-C extended
``Shoot`` with its growth half (``updateShootNodes``, ``childShootOutwardAxis``, apical/axillary termination,
terminal floral buds, volumes, downstream inflorescence area); ``advanceTime`` itself is in ``growth.py``.
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
        # W4-D: per-plant attraction points (H:3600-3613 defaults)
        self.attraction_points_enabled = False
        self.attraction_points = np.zeros((0, 3), f32)
        self.attraction_cone_half_angle_rad = f32(80.0 * np.pi / 180.0)     # 80.f * M_PI / 180.f: double
        self.attraction_cone_height = f32(0.1)
        self.attraction_weight = f32(0.6)
        self.attraction_obstacle_reduction_factor = f32(0.5)


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
        # growth members (H:3200-3270, W3-C)
        self.cached_base_offset = np.zeros(3, f32)
        self.base_offset_pinned = False
        self.last_parent_internode_axis = np.array([1e30, 1e30, 1e30], f32)
        self.last_parent_node_radius = f32(-1)
        self.old_shoot_volume = f32(0)
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
            # std::fmax(float, float). current_node_number and node_index are uint: a phytomer sampled as it is
            # appended has node_index == current_node_number (W5-C), so the difference wraps to 2^32 - 1 and p = pmin.
            nodes_above = (self.current_node_number - int(node_index) - 1) % (1 << 32)
            p = max(pmin, f32(pmax - f32(decay * f32(nodes_above))))
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
                            self.context_ptr, prescribed_internode)
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

        # W4-D: a prescribed internode keeps its measured length whatever the creation function does (PA:2193-2216)
        index = len(shoot_tree[self.ID].phytomers) - 1
        if prescribed_internode is not None:
            measured_vertices = [v.copy() for v in shoot_tree[self.ID].shoot_internode_vertices[index]]
            measured_radii = list(shoot_tree[self.ID].shoot_internode_radii[index])
            measured_length_max = phytomer.internode_length_max

        fn = phytomer_parameters.phytomer_creation_function
        if fn is not None:
            pa.phytomer_callback_depth += 1
            try:
                fn(phytomer, int(phytomer.shoot_index[0]), self.parent_node_index, int(self.shoot_parameters.max_nodes.val()),
                   pa.plant_instances[self.plantID].current_age)
            finally:
                pa.phytomer_callback_depth -= 1

        if prescribed_internode is not None and (phytomer.internode_length_max != measured_length_max
                                                 or phytomer.current_internode_scale_factor != 1):
            prescribed_shoot = shoot_tree[self.ID]
            prescribed_shoot.shoot_internode_vertices[index] = measured_vertices
            prescribed_shoot.shoot_internode_radii[index] = measured_radii
            phytomer.internode_length_max = measured_length_max
            phytomer.current_internode_scale_factor = f32(1)
            prescribed_shoot.updateShootNodes(True)
        return len(self.phytomers) - 1

    @ports("Shoot::appendPhytomerFromNodePositions")
    def appendPhytomerFromNodePositions(self, prescribed_internode, phytomer_parameters) -> int:
        """PA:2011-2015 (W4-D)."""
        from ..core import vec as V
        length = V.magnitude(np.asarray(prescribed_internode.tip_node, f32) - np.asarray(prescribed_internode.base_node, f32))
        return self.appendPhytomerInternal(prescribed_internode.base_radius, length, f32(1), f32(1), phytomer_parameters,
                                           prescribed_internode)

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
        """Phytomer::updateLeafTextureStage (PA:4855-4903): a no-op without texture stages and stage output. With stages
        (W5-A, rice) each existing leaf's blade primitives get the stage image as their color texture ("" at stage 0).
        ``leaf_texture_stage`` starts at 0 here, not at UINT_MAX: the first C++ call at stage 0 only clears a color
        texture the fresh blades do not have, so skipping it is equivalent."""
        prototype = phytomer.phytomer_parameters.leaf.prototype
        ages = prototype.leaf_texture_stage_ages
        if len(ages) < 2:
            return
        pa = phytomer.plantarchitecture_ptr
        last_stage = len(ages) - 1
        stage = max(prototype.getTextureStageAtAge(phytomer.age),
                    min(int(pa.plant_instances[phytomer.plantID].minimum_leaf_texture_stage), last_stage))
        if stage == phytomer.leaf_texture_stage:
            return
        phytomer.leaf_texture_stage = stage
        from .leaf_prototype import resolveTextureFile
        ctx = pa.context_ptr
        for petiole, leaves in enumerate(phytomer.leaf_objIDs):
            leaf_count = len(leaves)
            for leaf, objID_leaf in enumerate(leaves):
                if not ctx.does_object_exist(objID_leaf):
                    continue
                ind_from_tip = f32(f32(leaf) - f32(f32(leaf_count - 1) / f32(2)))
                prototype_index = phytomer.leaf_prototype_index[petiole][leaf]
                variant = prototype.getTextureSeriesVariant(int(ind_from_tip), prototype_index if prototype_index >= 0 else 0)
                blade_texture = resolveTextureFile(variant[0])
                blade_UUIDs = [u for u in ctx.object_primitive_uuids(objID_leaf)
                               if ctx.primitive_texture_file(u) == blade_texture]
                ctx.set_primitive_color_texture(blade_UUIDs, "" if stage == 0 else resolveTextureFile(variant[stage]))

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

    # ---- growth (W3-C) ---------------------------------------------------------------------------

    def isPruned(self) -> bool:
        return not self.phytomers

    @ports("Shoot::getGrowthNodeCapPerSeason")
    def getGrowthNodeCapPerSeason(self) -> int:
        if self.growth_state.max_nodes_per_season_growth_cap >= 0:
            return self.growth_state.max_nodes_per_season_growth_cap
        return int(self.shoot_parameters.max_nodes_per_season.val())

    @ports("Shoot::terminateApicalBud")
    def terminateApicalBud(self):
        self.meristem_is_alive = False
        self.phyllochron_counter = f32(0)

    @ports("Shoot::terminateAxillaryVegetativeBuds")
    def terminateAxillaryVegetativeBuds(self):
        for phytomer in self.phytomers:
            for petiole in phytomer.axillary_vegetative_buds:
                for vbud in petiole:
                    phytomer.setVegetativeBudState(BUD_DEAD, vbud)

    @ports("Shoot::makeDormant")
    def makeDormant(self):
        """PA:2264-2294 (W5-D, in seasons.py)."""
        from .seasons import makeDormant
        makeDormant(self)

    @ports("Shoot::addTerminalFloralBud")
    def addTerminalFloralBud(self):
        """PA:2311-2333 (the buds only; their geometry is updateInflorescence, W3-D)."""
        from .params import FloralBud, deg2rad, make_AxisRotation
        from .tube_utils import PI_F
        nbuds = int(self.shoot_parameters.max_terminal_floral_buds.val())
        for bud in range(nbuds):
            b = FloralBud()
            b.isterminal = True
            b.parent_index = 0
            b.bud_index = bud
            b.base_position = tuple(self.shoot_internode_vertices[-1][-1])
            pitch_adjustment = f32(0)
            if nbuds > 1:
                pitch_adjustment = deg2rad(30)
            yaw_adjustment = f32(f32(f32(f32(b.bud_index) * f32(2)) * PI_F) / f32(nbuds))
            b.base_rotation = make_AxisRotation(pitch_adjustment, yaw_adjustment, 0)
            b.bending_axis = (1, 0, 0)
            b.dd_to_inflorescence_maturity = \
                self.shoot_parameters.phytomer_parameters.inflorescence.inflorescence_maturity_period.val()
            self.phytomers[-1].floral_buds.append([b])
        self.shoot_parameters.max_terminal_floral_buds.resample()

    @ports("Shoot::calculateShootInternodeVolume")
    def calculateShootInternodeVolume(self) -> np.float32:
        volume = f32(0)
        for p, ph in enumerate(self.phytomers):
            volume = f32(volume + ph.calculatePhytomerVolume(p))
        return volume

    @ports("Shoot::calculateShootLength")
    def calculateShootLength(self) -> np.float32:
        """PA:2345-2351 (W5-D): sum of the internode lengths (float)."""
        length = f32(0)
        for ph in self.phytomers:
            length = f32(length + ph.getInternodeLength())
        return length

    @ports("Shoot::refreshDownstreamInflorescenceArea")
    def refreshDownstreamInflorescenceArea(self) -> np.float32:
        if self.isPruned():
            return f32(0)
        tree = self.plantarchitecture_ptr.plant_instances[self.plantID].shoot_tree
        running_total = f32(0)
        for p in range(len(self.phytomers) - 1, -1, -1):
            ph = self.phytomers[p]
            if ph.inflorescenceLoadsStem():
                running_total = f32(running_total + ph.getInflorescenceArea())
            for child in self.childIDs.get(p, []):
                running_total = f32(running_total + tree[child].refreshDownstreamInflorescenceArea())
            ph.downstream_inflorescence_area = running_total
        return running_total

    @ports("Shoot::sumDownstreamInflorescenceArea")
    def sumDownstreamInflorescenceArea(self, start_node_index: int) -> np.float32:
        if self.isPruned() or start_node_index >= len(self.phytomers):
            return f32(0)
        return self.phytomers[start_node_index].downstream_inflorescence_area

    @ports("Shoot::updateShootNodes")
    def updateShootNodes(self, update_context_geometry: bool):
        """PA:2429-2547: re-seat a lateral on its parent's node (with the cached outward offset), optionally
        push the internode tube to the Context, re-seat every phytomer's petiole/leaves, then the children."""
        from ..core import vec as V
        if not self.phytomers:
            return
        pa, ctx = self.plantarchitecture_ptr, self.context_ptr
        tree = pa.plant_instances[self.plantID].shoot_tree
        if self.parent_shoot_ID >= 0:
            parent_shoot = tree[self.parent_shoot_ID]
            current_origin = self.shoot_internode_vertices[0][0].copy()
            parent_node_position = parent_shoot.shoot_internode_vertices[self.parent_node_index][-1]
            parent_phytomer = parent_shoot.phytomers[self.parent_node_index]
            parent_radius = parent_phytomer.getInternodeRadius(f32(1))
            parent_internode_axis = parent_phytomer.getInternodeAxisVector(f32(1))
            if self.is_axis_continuation:
                self.cached_base_offset = np.zeros(3, f32)
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    axis_moved = V.magnitude(parent_internode_axis - self.last_parent_internode_axis) > f32(1e-7)
                if axis_moved or abs(f32(parent_radius - self.last_parent_node_radius)) > f32(1e-9):
                    if self.base_offset_pinned:
                        raise NotImplementedError("pinned branch offsets (posed plants) are not ported")
                    outward_axis = childShootOutwardAxis(parent_shoot, self.parent_node_index, self.parent_petiole_index,
                                                         parent_internode_axis, self.base_rotation)
                    self.cached_base_offset = (f32(0.9) * outward_axis) * parent_radius
                    self.last_parent_internode_axis = parent_internode_axis.copy()
                    self.last_parent_node_radius = f32(parent_radius)
            updated_origin = parent_node_position + self.cached_base_offset
            shift = updated_origin - current_origin
            if V.magnitude(shift) > f32(1e-6):
                for row in self.shoot_internode_vertices:
                    for i in range(len(row)):
                        row[i] = row[i] + shift

        if update_context_geometry and pa.build_context_geometry_internode and ctx.does_object_exist(self.internode_tube_objID):
            ctx.set_tube_radii(self.internode_tube_objID, [r for row in self.shoot_internode_radii for r in row])
            ctx.set_tube_nodes(self.internode_tube_objID, [v for row in self.shoot_internode_vertices for v in row])

        for p, ph in enumerate(self.phytomers):
            ph.setPetioleBase(self.shoot_internode_vertices[p][-1])

        for node in sorted(self.childIDs):                    # std::map<int, std::vector<int>> order
            for child_shoot_ID in self.childIDs[node]:
                tree[child_shoot_ID].updateShootNodes(update_context_geometry)


@ports("childShootOutwardAxis")
def childShootOutwardAxis(parent_shoot, parent_node_index: int, petiole_index: int, parent_internode_axis,
                          shoot_base_rotation):
    """PA:2366-2427 (file-static): unit vector from the parent's axis toward the side the child emerges on."""
    from ..core import vec as V
    nullorigin = np.zeros(3, f32)
    parent_internode_axis = np.asarray(parent_internode_axis, f32)
    parent_phytomer = parent_shoot.phytomers[parent_node_index]
    parent_petiole_axis = V.cross(parent_internode_axis, (0, 0, 1))
    if V.magnitude(parent_petiole_axis) < f32(0.01):
        parent_petiole_axis = np.array([0, 1, 0], f32)
    parent_petiole_axis = V.normalize(parent_petiole_axis)
    parent_petiole_axis = V.rotate_point_about_line(parent_petiole_axis, nullorigin, parent_internode_axis,
                                                    parent_phytomer.internode_phyllotactic_azimuth)
    if parent_phytomer.petiole_vertices:
        petiole_axis = parent_phytomer.getPetioleAxisVector(f32(0), petiole_index)
        outward = petiole_axis - V.dot(petiole_axis, parent_internode_axis) * parent_internode_axis
        if V.magnitude(outward) < f32(1e-6):
            return parent_petiole_axis
        return V.normalize(outward)
    child_axis = parent_internode_axis.copy()
    if shoot_base_rotation.roll != 0:
        child_axis = V.rotate_point_about_line(child_axis, nullorigin, parent_internode_axis, shoot_base_rotation.roll)
    if shoot_base_rotation.pitch != 0:
        base_pitch_axis = f32(-1) * V.cross(parent_internode_axis, parent_petiole_axis)
        child_axis = V.rotate_point_about_line(child_axis, nullorigin, base_pitch_axis, f32(-shoot_base_rotation.pitch))
    if shoot_base_rotation.yaw != 0:
        child_axis = V.rotate_point_about_line(child_axis, nullorigin, parent_internode_axis, shoot_base_rotation.yaw)
    outward = child_axis - V.dot(child_axis, parent_internode_axis) * parent_internode_axis
    if V.magnitude(outward) < f32(1e-6):
        return parent_petiole_axis
    return V.normalize(outward)


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
    # W3-C growth state
    pa.ground_clipping_height = f32(-99999)
    pa.fruit_branch_collision_enabled = False
    pa.solid_obstacle_pruning_enabled = False
    pa.inflorescence_geometry_draws_only = False   # opt-in: replay updateInflorescence's draws only (no W3-D geometry)
    pa.inflorescence_geometry_is_stub = False      # set once a draws-only inflorescence was 'built'
    # W4-D: set by an attraction-steered phytomer; forces the per-step Context update (force_update_on_collision)
    pa.collision_avoidance_applied = False
    pa.force_update_on_collision = True


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


@ports("PlantArchitecture::appendShoot")
def appendShoot(pa, plantID, parent_shoot_ID, current_node_number, base_rotation, internode_radius, internode_length_max,
                internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper,
                shoot_type_label) -> int:
    """PA:7178-7237 (W4-B): continue the parent's axis from its tip node with a new shoot."""
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::appendShoot): Plant with ID of {plantID} does not exist.")
    inst = pa.plant_instances[plantID]
    if shoot_type_label not in inst.shoot_types_snapshot:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::appendShoot): Shoot type with label of "
                                     f"{shoot_type_label} does not exist.")
    tree = inst.shoot_tree
    shoot_parameters = inst.shoot_types_snapshot[shoot_type_label].copy()
    validateShootTypes(shoot_parameters, inst.shoot_types_snapshot)
    if not tree:
        raise PlantArchitectureError("ERROR (PlantArchitecture::appendShoot): Cannot append shoot to empty shoot. You "
                                     "must call addBaseStemShoot() first for each plant.")
    if parent_shoot_ID >= len(tree):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::appendShoot): Parent with ID of {parent_shoot_ID} does "
                                     "not exist.")
    if current_node_number > shoot_parameters.max_nodes.val():
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::appendShoot): Cannot add shoot with "
                                     f"{current_node_number} nodes since the specified max node number is "
                                     f"{shoot_parameters.max_nodes.val()}.")
    parent = tree[parent_shoot_ID]
    if not parent.phytomers:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::appendShoot): Parent shoot with ID of {parent_shoot_ID} "
                                     "has no phytomers, so there is no node to append to. A shoot pruned at node 0 is "
                                     "left in this state.")
    # the parent stops growing at its apex: growth cap, the parameter itself (RandomParameter_int = uint), apex
    parent.growth_state.max_nodes_growth_cap = int(parent.current_node_number)
    parent.shoot_parameters.max_nodes = int(parent.current_node_number)
    parent.terminateApicalBud()
    appended_shootID = len(tree)
    parent_node = parent.current_node_number - 1
    rank = parent.rank
    base_position = parent.shoot_internode_vertices[-1][-1].copy()
    shoot = Shoot(plantID, appended_shootID, parent_shoot_ID, parent_node, 0, rank, base_position, base_rotation,
                  current_node_number, internode_length_max, shoot_parameters, shoot_type_label, pa)
    shoot.is_axis_continuation = True
    shoot.growth_state.internode_radius_cap = parent.growth_state.internode_radius_cap
    tree.append(shoot)
    shoot.buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                              leaf_scale_factor_fraction, radius_taper)
    return appended_shootID


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


# ---- W4-D: prescribed (trained) shoots, attraction points, leaf removal ----------------------------

class PrescribedInternode:
    """PrescribedInternode (H): the measured end nodes and radii of one internode."""
    __slots__ = ("base_node", "tip_node", "base_radius", "tip_radius")

    def __init__(self, base_node, tip_node, base_radius, tip_radius):
        self.base_node = np.asarray(base_node, f32).reshape(3).copy()
        self.tip_node = np.asarray(tip_node, f32).reshape(3).copy()
        self.base_radius = f32(base_radius)
        self.tip_radius = f32(tip_radius)


@ports("PlantArchitecture::addShootFromNodePositions", "PlantArchitecture::prescribedShoot")
def addShootFromNodePositions(pa, plantID, parent_shoot_ID, parent_node_index, internode_node_positions, internode_radii,
                              shoot_type_label, growth_shoot_type_label=None, petiole_index=0,
                              continues_parent_axis=False, caller="addShootFromNodePositions") -> int:
    """PA:7286-7465 (``prescribedShoot``)."""
    from ..core import vec as V
    from .params import make_AxisRotation
    if growth_shoot_type_label is None:
        growth_shoot_type_label = shoot_type_label
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Plant with ID of {plantID} does not exist.")
    inst = pa.plant_instances[plantID]
    if shoot_type_label not in inst.shoot_types_snapshot:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Shoot type with label of {shoot_type_label} "
                                     "does not exist.")
    if growth_shoot_type_label not in inst.shoot_types_snapshot:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Growth shoot type with label of "
                                     f"{growth_shoot_type_label} does not exist.")
    nodes = [np.asarray(p, f32).reshape(3).copy() for p in internode_node_positions]
    radii = [f32(r) for r in internode_radii]
    if len(nodes) < 2:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): At least 2 node positions are required to "
                                     f"define a shoot, but {len(nodes)} were given.")
    if len(radii) != len(nodes):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Number of radii ({len(radii)}) must equal the "
                                     f"number of node positions ({len(nodes)}).")
    for node, r in enumerate(radii):
        if r <= 0:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Internode radius at node {node} is "
                                         f"{float(r):f}, but all radii must be greater than zero.")
    for node in range(len(nodes) - 1):
        if V.magnitude(nodes[node + 1] - nodes[node]) < f32(1e-9):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Node positions {node} and {node + 1} are "
                                         "coincident, which does not define an internode axis.")
    tree = inst.shoot_tree
    shoot_parameters = inst.shoot_types_snapshot[shoot_type_label].copy()
    validateShootTypes(shoot_parameters, inst.shoot_types_snapshot)
    node_count = len(nodes) - 1
    if node_count > shoot_parameters.max_nodes.val():
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Cannot add shoot with {node_count} nodes since "
                                     f"the specified max node number is {shoot_parameters.max_nodes.val()}.")
    rank = 0
    shoot_base_position = nodes[0].copy()
    if parent_shoot_ID >= 0:
        if parent_shoot_ID >= len(tree):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Parent with ID of {parent_shoot_ID} does "
                                         "not exist.")
        parent = tree[parent_shoot_ID]
        if len(parent.phytomers) <= parent_node_index:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): Parent shoot does not have a node "
                                         f"{parent_node_index}.")
        if continues_parent_axis:
            raise NotImplementedError("appendShootFromNodePositions (axis continuation) is not ported")
        attachment_node = parent.shoot_internode_vertices[parent_node_index][-1]
        parent_radius = parent.phytomers[parent_node_index].getInternodeRadius(f32(1))
        attachment_distance = V.magnitude(nodes[0] - attachment_node)
        attachment_tolerance = max(f32(1e-3), f32(f32(2) * parent_radius))
        if attachment_distance > attachment_tolerance:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{caller}): The first node position is "
                                         f"{float(attachment_distance):f} m from node {parent_node_index} of shoot "
                                         f"{parent_shoot_ID}, which exceeds the attachment tolerance of "
                                         f"{float(attachment_tolerance):f} m. The shoot would not be connected to its parent.")
        rank = parent.rank + 1
        parent_internode_axis = parent.phytomers[parent_node_index].getInternodeAxisVector(f32(1))
        axis_vector = childShootOutwardAxis(parent, parent_node_index, petiole_index, parent_internode_axis,
                                            make_AxisRotation(0, 0, 0))
        shoot_base_position = attachment_node + (f32(0.9) * axis_vector) * parent_radius
    base_shift = shoot_base_position - nodes[0]
    total_length = f32(0)
    for node in range(node_count):
        total_length = f32(total_length + V.magnitude(nodes[node + 1] - nodes[node]))
    mean_internode_length = f32(total_length / f32(node_count))
    shootID = len(tree)
    shoot = Shoot(plantID, shootID, parent_shoot_ID, parent_node_index, petiole_index, rank, shoot_base_position,
                  make_AxisRotation(0, 0, 0), node_count, mean_internode_length, shoot_parameters, shoot_type_label, pa)
    shoot.is_axis_continuation = False
    tree.append(shoot)
    shoot.growth_state.geometry_is_prescribed = True
    if growth_shoot_type_label != shoot_type_label:
        growth_parameters = inst.shoot_types_snapshot[growth_shoot_type_label].copy()
        validateShootTypes(growth_parameters, inst.shoot_types_snapshot)
        shoot.growth_state.growth_type_label = growth_shoot_type_label
        shoot.growth_state.max_nodes_growth_cap = max(node_count, int(growth_parameters.max_nodes.val()))
        shoot.growth_state.max_nodes_per_season_growth_cap = int(growth_parameters.max_nodes_per_season.val())
        shoot.gravitropic_curvature = growth_parameters.gravitropic_curvature.val()
    for node in range(node_count):
        pi = PrescribedInternode(nodes[node] + base_shift, nodes[node + 1] + base_shift, radii[node], radii[node + 1])
        shoot.appendPhytomerFromNodePositions(pi, shoot_parameters.phytomer_parameters)
    return shootID


@ports("PlantArchitecture::removeShootLeaves")
def removeShootLeaves(pa, plantID, shootID):
    """PA:9998-10012."""
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::removePlantLeaves): Plant with ID of {plantID} does not "
                                     "exist.")
    tree = pa.plant_instances[plantID].shoot_tree
    if shootID >= len(tree):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::removeShootLeaves): Shoot with ID of {shootID} does not "
                                     "exist.")
    for phytomer in tree[shootID].phytomers:
        phytomer.removeLeaf()


def _shoot_or_raise(pa, plantID, shootID, fn):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{fn}): Plant with ID of {plantID} does not exist.")
    tree = pa.plant_instances[plantID].shoot_tree
    if shootID >= len(tree):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::{fn}): Shoot with ID of {shootID} does not exist.")
    return tree[shootID]


@ports("PlantArchitecture::removeShootVegetativeBuds")
def removeShootVegetativeBuds(pa, plantID, shootID):
    """PA:10014-10028 (W5-C)."""
    from .params import BUD_DEAD
    for phytomer in _shoot_or_raise(pa, plantID, shootID, "removeShootVegetativeBuds").phytomers:
        phytomer.setVegetativeBudState(BUD_DEAD)


@ports("PlantArchitecture::removeShootFloralBuds")
def removeShootFloralBuds(pa, plantID, shootID):
    """PA:10030-10044 (W5-C)."""
    from .params import BUD_DEAD
    for phytomer in _shoot_or_raise(pa, plantID, shootID, "removeShootFloralBuds").phytomers:
        phytomer.setFloralBudState(BUD_DEAD)


@ports("PlantArchitecture::setPlantAttractionPoints")
def setPlantAttractionPoints(pa, plantID, attraction_points, view_half_angle_deg=80.0, look_ahead_distance=0.1,
                             attraction_weight=0.6, obstacle_reduction_factor=0.75):
    """PA:13556-13582."""
    from ..core import vec as V
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::setPlantAttractionPoints): Plant with ID {plantID} does "
                                     "not exist.")
    view_half_angle_deg, look_ahead_distance = f32(view_half_angle_deg), f32(look_ahead_distance)
    obstacle_reduction_factor = f32(obstacle_reduction_factor)
    if view_half_angle_deg <= 0 or view_half_angle_deg > 180:
        raise PlantArchitectureError("ERROR (PlantArchitecture::setPlantAttractionPoints): view_half_angle_deg must be "
                                     "between 0 and 180 degrees.")
    if look_ahead_distance <= 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::setPlantAttractionPoints): look_ahead_distance must be "
                                     "greater than 0.")
    if len(attraction_points) == 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::setPlantAttractionPoints): attraction_points cannot be "
                                     "empty.")
    if obstacle_reduction_factor < 0 or obstacle_reduction_factor > 1:
        raise PlantArchitectureError("ERROR (PlantArchitecture::setPlantAttractionPoints): obstacle_reduction_factor must "
                                     "be between 0 and 1.")
    plant = pa.plant_instances[plantID]
    plant.attraction_points_enabled = True
    plant.attraction_points = np.asarray(attraction_points, f32).reshape(-1, 3).copy()
    plant.attraction_cone_half_angle_rad = V.deg2rad(view_half_angle_deg)
    plant.attraction_cone_height = look_ahead_distance
    plant.attraction_weight = f32(attraction_weight)
    plant.attraction_obstacle_reduction_factor = obstacle_reduction_factor


__all__ = ["Shoot", "PlantInstance", "ShootGrowthState", "init_plant_state", "addPlantInstance", "addBaseStemShoot",
           "breakPlantDormancy", "setPlantPhenologicalThresholds", "validateShootTypes", "appendShoot"]
