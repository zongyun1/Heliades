# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp (Copyright (C) 2016-2026 Brian Bailey):
#   PlantArchitecture::addChildShoot                     PA:7239-7283
#   PlantArchitecture::appendPhytomerToShoot             PA:7523-7593
#   PlantArchitecture::incrementPhytomerInternodeGirth   PA:7625-7680
#   PlantArchitecture::advanceTime                       PA:11715-12465
"""Growth: ``advanceTime`` and the structural operations it drives (W3-C).

The sub-step loop is a line-by-line port of PA:11728-12465 for a plant without the optional subsystems
(SPEC §7): carbon / nitrogen models, collision avoidance, attraction points, solid obstacles, fruit-branch
collisions, ground clipping, leaf-angle tracking, epicormic shoots and posed plants all raise
``NotImplementedError`` when switched on. Pieces that run every sub-step but have no effect on the tomato
are noted where they are skipped:

* ``bakePlantPose`` returns at once (no pose state); ``clearBVHCache`` has nothing to clear;
* ``updateShadowGrid`` / ``sumPlantLeafArea`` / ``captureSeasonPeakLeaves`` only feed ``shedShadedBranches``,
  which runs at the season boundary (``time_since_dormancy >= dd_to_dormancy_break + dd_to_dormancy``, 1000
  days for the tomato, never reached before ``max_age`` 365). They make no draws and write no geometry;
* ``flushDeferredPrunes`` has no cuts queued (no phytomer callbacks), ``updatePlantLeafAngleDistributionTracking``
  is off, ``adjustFruitForObstacleCollision`` only invalidates the organ index.

Flower and fruit geometry is W3-D: the floral state machine here is complete (transitions, draws, determinate
growth), and reaches ``Phytomer.updateInflorescence``, which raises unless ``pa.inflorescence_geometry_draws_only``
is set (then it replays only the draws). The first initiation can happen at day 56 (bud on node 8, created on
day 16, 40 days of ``time_counter``), so ages <= 54 never reach it.
"""
from __future__ import annotations

import math

import numpy as np

from ..migration_registry import ports
from .params import (BUD_ACTIVE, BUD_DEAD, BUD_DORMANT, BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING,
                     deg2rad, make_AxisRotation)
from .random_parameter import PlantArchitectureError
from .shoot import Shoot, childShootOutwardAxis, validateShootTypes
from .tube_utils import PI_F

f32 = np.float32


def _fmin(a, b):
    """``fmin`` of two floats (exact in float or double)."""
    a, b = f32(a), f32(b)
    return a if a < b else b


# ---- structural operations ------------------------------------------------------------------------

@ports("PlantArchitecture::addChildShoot")
def addChildShoot(pa, plantID, parent_shoot_ID, parent_node_index, current_node_number, shoot_base_rotation,
                  internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                  leaf_scale_factor_fraction, radius_taper, shoot_type_label, petiole_index=0) -> int:
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::addChildShoot): Plant with ID of {plantID} does not exist.")
    inst = pa.plant_instances[plantID]
    if shoot_type_label not in inst.shoot_types_snapshot:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::addChildShoot): Shoot type with label of "
                                     f"{shoot_type_label} does not exist.")
    tree = inst.shoot_tree
    if parent_shoot_ID <= -1 or parent_shoot_ID >= len(tree):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::addChildShoot): Parent with ID of {parent_shoot_ID} "
                                     "does not exist.")
    if len(tree[parent_shoot_ID].phytomers) <= parent_node_index:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::addChildShoot): Parent shoot does not have a node "
                                     f"{parent_node_index}.")
    shoot_parameters = inst.shoot_types_snapshot[shoot_type_label].copy()
    validateShootTypes(shoot_parameters, inst.shoot_types_snapshot)
    parent_rank = tree[parent_shoot_ID].rank
    childID = len(tree)
    parent = tree[parent_shoot_ID]
    shoot_base_position = parent.shoot_internode_vertices[parent_node_index][-1].copy()
    parent_phytomer = parent.phytomers[parent_node_index]
    parent_internode_axis = parent_phytomer.getInternodeAxisVector(f32(1))
    axis_vector = childShootOutwardAxis(parent, parent_node_index, petiole_index, parent_internode_axis,
                                        shoot_base_rotation)
    shoot_base_position = shoot_base_position + (f32(0.9) * axis_vector) * parent_phytomer.getInternodeRadius(f32(1))
    shoot = Shoot(plantID, childID, parent_shoot_ID, parent_node_index, petiole_index, parent_rank + 1,
                  shoot_base_position, shoot_base_rotation, current_node_number, internode_length_max,
                  shoot_parameters, shoot_type_label, pa)
    tree.append(shoot)
    shoot.buildShootPhytomers(internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                              leaf_scale_factor_fraction, radius_taper)
    return childID


@ports("PlantArchitecture::appendPhytomerToShoot")
def appendPhytomerToShoot(pa, plantID, shootID, phytomer_parameters, internode_radius, internode_length_max,
                          internode_length_scale_factor_fraction, leaf_scale_factor_fraction) -> int:
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::appendPhytomerToShoot): Plant with ID of {plantID} does "
                                     "not exist.")
    tree = pa.plant_instances[plantID].shoot_tree
    if shootID >= len(tree):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::appendPhytomerToShoot): Parent with ID of {shootID} does "
                                     "not exist.")
    shoot = tree[shootID]
    pID = shoot.appendPhytomer(internode_radius, internode_length_max, internode_length_scale_factor_fraction,
                               leaf_scale_factor_fraction, phytomer_parameters)
    shoot.current_node_number += 1
    shoot.nodes_this_season += 1
    for ph in shoot.phytomers:
        ph.shoot_index[1] = shoot.current_node_number
    sp = shoot.shoot_parameters
    if shoot.current_node_number == shoot.getGrowthNodeCap():
        if not sp.flowers_require_dormancy and sp.max_terminal_floral_buds.val() > 0:
            shoot.addTerminalFloralBud()
            inf = sp.phytomer_parameters.inflorescence
            if inf.flower_prototype_function is not None:
                state = BUD_FLOWER_CLOSED
            elif inf.fruit_prototype_function is not None:
                state = BUD_FRUITING
            else:
                return pID
            last = shoot.phytomers[-1]
            for fbuds in last.floral_buds:
                for fbud in fbuds:
                    if fbud.isterminal:
                        fbud.state = state
                        last.updateInflorescence(fbud)
                        if state == BUD_FRUITING and fbud.dd_to_inflorescence_maturity != 0:
                            last.setInflorescenceScaleFraction(fbud, f32(0.25))
    elif shoot.nodes_this_season >= shoot.getGrowthNodeCapPerSeason():
        if not sp.flowers_require_dormancy and sp.max_terminal_floral_buds.val() > 0:
            shoot.addTerminalFloralBud()
            last = shoot.phytomers[-1]
            for fbuds in last.floral_buds:
                for fbud in fbuds:
                    if fbud.isterminal:
                        fbud.state = BUD_DORMANT
                        last.updateInflorescence(fbud)
        shoot.phytomers[pID].isdormant = True
    return pID


@ports("PlantArchitecture::incrementPhytomerInternodeGirth")
def incrementPhytomerInternodeGirth(pa, plantID, shootID, node_number, dt, update_context_geometry):
    """PA:7625-7680, the pipe model: radius -> radius + 0.5*(sqrt(factor*area*1e-4/pi) - radius) if larger."""
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Plant with ID of "
                                     f"{plantID} does not exist.")
    ctx = pa.context_ptr
    shoot = pa.plant_instances[plantID].shoot_tree[shootID]
    if node_number >= shoot.current_node_number:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::incrementPhytomerInternodeGirth): Cannot scale internode "
                                     f"{node_number} because there are only {shoot.current_node_number} nodes in this "
                                     "shoot.")
    phytomer = shoot.phytomers[node_number]
    leaf_area = f32(phytomer.downstream_leaf_area)
    if ctx.does_object_exist(shoot.internode_tube_objID):
        ctx.set_object_data(shoot.internode_tube_objID, "leaf_area", leaf_area)
    girth_area_factor = shoot.shoot_parameters.girth_area_factor.val()
    supported_area = f32(leaf_area + shoot.sumDownstreamInflorescenceArea(node_number))
    internode_area = f32(float(f32(girth_area_factor * supported_area)) * 1e-4)        # float * double -> float
    phytomer_radius = f32(np.sqrt(f32(internode_area / PI_F)))                         # sqrtf
    segment = shoot.shoot_internode_radii[node_number]
    for i in range(len(segment)):
        if phytomer_radius > segment[i]:
            # radius + 0.5 * (phytomer_radius - radius): the 0.5 literal is double
            segment[i] = f32(float(segment[i]) + 0.5 * float(f32(phytomer_radius - segment[i])))
    if update_context_geometry and ctx.does_object_exist(shoot.internode_tube_objID):
        ctx.set_tube_radii(shoot.internode_tube_objID, [r for row in shoot.shoot_internode_radii for r in row])
    phytomer.updatePeduncleRadii()


# ---- advanceTime --------------------------------------------------------------------------------

def _check_supported(pa):
    for flag in ("carbon_model_enabled", "nitrogen_model_enabled", "collision_detection_enabled",
                 "attraction_points_enabled", "solid_obstacle_avoidance_enabled"):
        if getattr(pa, flag, False):
            raise NotImplementedError(f"advanceTime with {flag} is not ported")
    if getattr(pa, "fruit_branch_collision_enabled", False) or getattr(pa, "solid_obstacle_pruning_enabled", False):
        raise NotImplementedError("fruit-branch collisions / solid-obstacle pruning are not ported")
    if f32(getattr(pa, "ground_clipping_height", f32(-99999))) != f32(-99999):
        raise NotImplementedError("ground clipping is not ported")


@ports("PlantArchitecture::advanceTime")
def advanceTime(pa, plantIDs, time_step_days):
    """``advanceTime(const std::vector<uint>&, float)`` (PA:11728-12465)."""
    if isinstance(plantIDs, (int, np.integer)):
        plantIDs = [int(plantIDs)]
    plantIDs = [int(p) for p in plantIDs]
    time_step_days = f32(time_step_days)
    for pid in plantIDs:
        if pid not in pa.plant_instances:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::advanceTime): Plant with ID of {pid} does not exist.")
    _check_supported(pa)
    # bakePlantPose: no pose state -> return; clearBVHCache / BVH rebuild: collision detection is off

    phyllochron_min = f32(9999)
    for pid in plantIDs:
        inst = pa.plant_instances[pid]
        if not inst.shoot_tree:
            continue
        for shoot in inst.shoot_tree:
            phyllochron_min = min(phyllochron_min, shoot.shoot_parameters.phyllochron_min.val())
        for label in sorted(inst.shoot_types_snapshot):          # std::map order
            phyllochron_min = min(phyllochron_min, inst.shoot_types_snapshot[label].phyllochron_min.val())
    if phyllochron_min == f32(9999):
        return

    if time_step_days <= phyllochron_min:
        Nsteps = int(time_step_days)                              # float -> int truncation
        dt_max_days = f32(1)
    else:
        Nsteps = int(math.floor(float(f32(time_step_days / phyllochron_min))))
        dt_max_days = phyllochron_min
    remainder_time = f32(time_step_days - f32(dt_max_days * f32(Nsteps)))
    if remainder_time > 0:
        Nsteps += 1

    for timestep in range(Nsteps):
        if timestep == Nsteps - 1 and remainder_time != 0:
            dt_max_days = remainder_time
        for pid in plantIDs:
            _advance_plant_one_step(pa, pid, dt_max_days)

    # adjustFruitForObstacleCollision: invalidates the organ index, then returns (no obstacles)
    pa.invalidateOrganIndex()
    for pid in plantIDs:
        if pid in pa.plant_instances:
            pa.plant_instances[pid].shoot_tree[0].updateShootNodes(True)
    if pa.output_object_data.get("age"):
        raise NotImplementedError("optional 'age' object data is not ported")


def _advance_plant_one_step(pa, plantID, dt):
    ctx = pa.context_ptr
    inst = pa.plant_instances[plantID]
    tree = inst.shoot_tree
    if not tree:
        return
    if inst.current_age <= inst.max_age and f32(inst.current_age + dt) > inst.max_age:
        pass                                                       # the last step up to max_age still grows
    elif inst.current_age >= inst.max_age:
        if not getattr(inst, "mature_geometry_synced", False):
            tree[0].updateShootNodes(True)
            inst.mature_geometry_synced = True
        return

    inst.current_age = f32(inst.current_age + dt)
    inst.time_since_dormancy = f32(inst.time_since_dormancy + dt)
    # W5-D: updateShadowGrid / sumPlantLeafArea / captureSeasonPeakLeaves (PA:11866-11880, seasons.py). They make
    # no draws and write no geometry; they only feed the season boundary, so a plant that never reaches it
    # (dd_to_dormancy_break + dd_to_dormancy beyond max_age, every herbaceous crop) skips them.
    dormancy_period = f32(inst.dd_to_dormancy_break + inst.dd_to_dormancy)
    if dormancy_period > 0 and dormancy_period <= inst.max_age:
        from .seasons import season_step
        season_step(pa, plantID)

    if dormancy_period > 0 and inst.time_since_dormancy >= dormancy_period:
        from .seasons import season_boundary                       # W5-D (PA:11888-11914)
        season_boundary(pa, plantID)
        return

    for shoot in tree:
        if shoot.parent_shoot_ID < 0:
            shoot.refreshDownstreamInflorescenceArea()

    shoot_count = len(tree)
    for i in range(shoot_count):
        shoot = tree[i]
        sp = shoot.shoot_parameters
        inf = sp.phytomer_parameters.inflorescence
        for phytomer in shoot.phytomers:
            phytomer.age = f32(phytomer.age + dt)
            shoot._update_leaf_texture_stage(phytomer)
            if phytomer.phytomer_parameters.phytomer_callback_function is not None:
                # PhytomerCallbackScope (PA:11936, W4-A); a structural operation inside it is skipped/rejected in C++
                pa.phytomer_callback_depth += 1
                try:
                    phytomer.phytomer_parameters.phytomer_callback_function(phytomer)
                finally:
                    pa.phytomer_callback_depth -= 1

        # ---- phenological transitions
        if shoot.isdormant and inst.time_since_dormancy >= inst.dd_to_dormancy_break:
            shoot.phyllochron_counter = f32(0)
            shoot.breakDormancy()
        if shoot.isdormant:
            continue

        for phytomer in shoot.phytomers:
            if phytomer.age > inst.max_leaf_lifespan:
                if pa.nitrogen_model_enabled:
                    raise NotImplementedError("resorbLeafNitrogen is not ported")
                phytomer.removeLeaf()                                      # W4-A
            if not phytomer.floral_buds:
                continue
            for petiole in phytomer.floral_buds:
                for fbud in petiole:
                    _floral_bud_step(pa, inst, shoot, phytomer, fbud, dt)

        # ---- growth / scaling of the current phytomers
        for node_index, phytomer in enumerate(shoot.phytomers):
            if phytomer.current_internode_scale_factor < 1:
                dL_internode = f32(f32(dt * shoot.elongation_rate_instantaneous) * phytomer.internode_length_max)
                length_scale = _fmin(f32(1), f32(f32(phytomer.getInternodeLength() + dL_internode)
                                                 / phytomer.internode_length_max))
                phytomer.setInternodeLengthScaleFraction(length_scale, False)
            if sp.girth_area_factor.val() > 0:
                incrementPhytomerInternodeGirth(pa, plantID, shoot.ID, node_index, dt, False)

        for node_index, phytomer in enumerate(shoot.phytomers):
            if phytomer.hasLeaf():
                _expand_petioles_and_leaves(shoot, phytomer, dt)

            # fruit growth
            for petiole in phytomer.floral_buds:
                for fbud in petiole:
                    dd_to_maturity = (fbud.dd_to_inflorescence_maturity if fbud.dd_to_inflorescence_maturity > 0
                                      else inst.dd_to_fruit_maturity)
                    if (fbud.state == BUD_FRUITING and fbud.time_counter > 0 and dd_to_maturity > 0
                            and fbud.dd_to_inflorescence_maturity != 0):
                        fbud.previous_fruit_scale_factor = fbud.current_fruit_scale_factor
                        # fmin(1, 0.25f + 0.75f * tc / dd): the int 1 makes this the double fmin
                        scale = f32(min(1.0, float(f32(f32(0.25) + f32(f32(f32(0.75) * fbud.time_counter)
                                                                         / dd_to_maturity)))))
                        phytomer.setInflorescenceScaleFraction(fbud, scale)

            # new child shoots from vegetative buds
            for parent_petiole_index, petiole in enumerate(phytomer.axillary_vegetative_buds):
                for vbud in petiole:
                    if vbud.state == BUD_ACTIVE and f32(phytomer.age + dt) > sp.vegetative_bud_break_time.val():
                        _break_vegetative_bud(pa, plantID, inst, shoot, phytomer, node_index, parent_petiole_index, vbud)

        if shoot.current_node_number >= shoot.getGrowthNodeCap():
            shoot.terminateApicalBud()
        if not shoot.meristem_is_alive:
            continue

        # ---- phyllochron: new phytomers
        shoot.phyllochron_counter = f32(shoot.phyllochron_counter + dt)
        while (shoot.phyllochron_instantaneous > 0 and shoot.phyllochron_counter >= shoot.phyllochron_instantaneous
               and not shoot.phytomers[-1].isdormant and shoot.current_node_number < shoot.getGrowthNodeCap()):
            rp = sp.phytomer_parameters.internode.radius_initial
            internode_radius = rp.val()
            rp.resample()
            if shoot.growth_state.internode_radius_cap > 0:
                internode_radius = _fmin(internode_radius, shoot.growth_state.internode_radius_cap)
            internode_length_max = shoot.internode_length_max_shoot_initial
            appendPhytomerToShoot(pa, plantID, shoot.ID, sp.phytomer_parameters, internode_radius, internode_length_max,
                                  f32(0.01), f32(0.01))
            if getattr(inst, "leaf_angle_tracking_enabled", False) and shoot.phytomers:
                from .leaf_angle_tracking import add_to_cohort             # W5-D (PA:12249)
                add_to_cohort(inst, shoot.ID, len(shoot.phytomers) - 1)
            shoot.phyllochron_counter = f32(shoot.phyllochron_counter - shoot.phyllochron_instantaneous)

        if getattr(inst, "epicormic_shoot_label", ""):
            raise NotImplementedError("epicormic shoots are not ported")

    # W5-D: cuts requested by this sub-step's phytomer callbacks (PA:12267)
    from .seasons import flushDeferredPrunes
    flushDeferredPrunes(pa)
    # W5-D: leaf angle distribution tracking (PA:12274; a no-op unless enabled for the plant)
    if getattr(inst, "leaf_angle_tracking_enabled", False):
        from .leaf_angle_tracking import updatePlantLeafAngleDistributionTracking
        updatePlantLeafAngleDistributionTracking(pa, plantID)
    # W4-D: an attraction-steered phytomer forces the Context update (PA:12312-12326; collision detection is off)
    if pa.collision_avoidance_applied and pa.force_update_on_collision:
        tree[0].updateShootNodes(True)
    else:
        tree[0].updateShootNodes(False)
    pa.collision_avoidance_applied = False

    for shoot in tree:
        if not ctx.does_object_exist(shoot.internode_tube_objID):
            continue
        shoot_volume = shoot.calculateShootInternodeVolume()
        volume_ratio = f32(shoot.old_shoot_volume / shoot_volume)
        ctx.set_object_data(shoot.internode_tube_objID, "volume_ratio", volume_ratio)
        ctx.set_object_data(shoot.internode_tube_objID, "old_shoot_volume", shoot_volume)
        shoot.old_shoot_volume = shoot_volume
    if pa.output_object_data.get("plant_height") or pa.output_object_data.get("phenology_stage"):
        raise NotImplementedError("optional plant_height / phenology_stage object data are not ported")


def _floral_bud_step(pa, inst, shoot, phytomer, fbud, dt):
    """The per-bud body of PA:11970-12060: clocks, then initiation / opening / fruit set."""
    ctx = pa.context_ptr
    sp = shoot.shoot_parameters
    inf = sp.phytomer_parameters.inflorescence
    if fbud.state != BUD_DORMANT and fbud.state != BUD_DEAD:
        fbud.time_counter = f32(fbud.time_counter + dt)
        if fbud.state != BUD_ACTIVE:
            fbud.age = f32(fbud.age + dt)

    def determinate():
        if sp.determinate_shoot_growth:
            shoot.terminateApicalBud()
            shoot.terminateAxillaryVegetativeBuds()

    if inf.flower_prototype_function is not None:
        if fbud.state == BUD_ACTIVE and inst.dd_to_flower_initiation >= 0:
            if fbud.time_counter >= inst.dd_to_flower_initiation:           # both dormancy branches are the same test
                fbud.time_counter = f32(0)
                if ctx.randu() < sp.flower_bud_break_probability.val():
                    phytomer.setFloralBudState(BUD_FLOWER_CLOSED, fbud)
                else:
                    phytomer.setFloralBudState(BUD_DEAD, fbud)
                determinate()
        elif ((fbud.state == BUD_FLOWER_CLOSED and inst.dd_to_flower_opening >= 0)
              or (fbud.state == BUD_ACTIVE and inst.dd_to_flower_initiation < 0 and inst.dd_to_flower_opening >= 0)):
            if fbud.time_counter >= inst.dd_to_flower_opening:
                fbud.time_counter = f32(0)
                if fbud.state == BUD_FLOWER_CLOSED:
                    phytomer.setFloralBudState(BUD_FLOWER_OPEN, fbud)
                else:
                    if ctx.randu() < sp.flower_bud_break_probability.val():
                        phytomer.setFloralBudState(BUD_FLOWER_OPEN, fbud)
                    else:
                        phytomer.setFloralBudState(BUD_DEAD, fbud)
                determinate()

    if inf.fruit_prototype_function is not None:
        if ((fbud.state == BUD_FLOWER_OPEN and inst.dd_to_fruit_set >= 0)
                or (fbud.state == BUD_ACTIVE and inst.dd_to_flower_initiation < 0
                    and (inst.dd_to_flower_opening < 0 or inf.flower_prototype_function is None)
                    and inst.dd_to_fruit_set >= 0)
                or (fbud.state == BUD_FLOWER_CLOSED and inst.dd_to_flower_opening < 0 and inst.dd_to_fruit_set >= 0)):
            if fbud.time_counter >= inst.dd_to_fruit_set:
                fbud.time_counter = f32(0)
                fruit_set_prob = sp.fruit_set_probability.val()
                if fbud.state == BUD_ACTIVE:
                    fruit_set_prob = f32(fruit_set_prob * sp.flower_bud_break_probability.val())
                if ctx.randu() < fruit_set_prob:
                    phytomer.setFloralBudState(BUD_FRUITING, fbud)
                else:
                    phytomer.setFloralBudState(BUD_DEAD, fbud)
                determinate()


def _expand_petioles_and_leaves(shoot, phytomer, dt):
    """PA:12080-12150: petiole elongation on the shoot's elongation rate, blade expansion on the leaf rate (one
    ``prototype_scale`` draw per still-expanding blade)."""
    for petiole_index in range(len(phytomer.current_leaf_scale_factor)):
        leaf_still_expanding = phytomer.current_leaf_scale_factor[petiole_index] < 1
        petiole_still_elongating = (petiole_index < len(phytomer.current_petiole_scale_factor)
                                    and phytomer.current_petiole_scale_factor[petiole_index] < 1)
        if not leaf_still_expanding and not petiole_still_elongating:
            phytomer.bendPetioleUnderLeafWeight(petiole_index)
            for leaf in range(len(phytomer.leaf_objIDs[petiole_index])):
                phytomer.deformLeafUnderSelfWeight(petiole_index, leaf)
            continue
        petiole_scale = (phytomer.current_petiole_scale_factor[petiole_index]
                         if petiole_index < len(phytomer.current_petiole_scale_factor) else f32(1))
        if petiole_still_elongating:
            petiole_scale = _fmin(f32(1), f32(phytomer.current_petiole_scale_factor[petiole_index]
                                              + f32(dt * shoot.elongation_rate_instantaneous)))
        if not leaf_still_expanding:
            phytomer.setPetioleScaleFraction(petiole_index, petiole_scale)
            continue
        sizes = phytomer.leaf_size_max[petiole_index]
        tip_ind = int(math.ceil(float(f32(f32(len(sizes) - 1) / f32(2)))))
        leaf_length = f32(phytomer.current_leaf_scale_factor[petiole_index] * sizes[tip_ind])
        dL_leaf = f32(f32(dt * shoot.leaf_expansion_rate_instantaneous) * sizes[tip_ind])
        phytomer.phytomer_parameters.leaf.prototype_scale.val()            # leaf_scale (unused) reads it
        leaf_size_full = sizes[tip_ind]
        scale = _fmin(f32(1), f32(f32(leaf_length + dL_leaf) / leaf_size_full)) if leaf_size_full > 0 else f32(1)
        phytomer.phytomer_parameters.leaf.prototype_scale.resample()
        phytomer.setPetioleAndLeafScaleFraction(petiole_index, petiole_scale, scale)


def _break_vegetative_bud(pa, plantID, inst, shoot, phytomer, node_index, parent_petiole_index, vbud):
    """PA:12160-12225: a lateral from an active axillary bud."""
    nsp = inst.shoot_types_snapshot[vbud.shoot_type_label]
    parent_node_count = shoot.current_node_number
    insertion_angle_adjustment = _fmin(
        f32(nsp.insertion_angle_tip.val() + f32(nsp.insertion_angle_decay_rate.val()
                                                * f32(parent_node_count - phytomer.shoot_index[0] - 1))),
        f32(90))
    # make_AxisRotation(deg2rad(insertion), deg2rad(base_yaw.val()), deg2rad(base_roll.val())): GCC evaluates the
    # arguments right to left, so base_roll is read before base_yaw (only base_yaw draws for the tomato)
    roll = deg2rad(nsp.base_roll.val())
    yaw = deg2rad(nsp.base_yaw.val())
    base_rotation = make_AxisRotation(deg2rad(insertion_angle_adjustment), yaw, roll)
    nsp.base_yaw.resample()
    nsp.base_roll.resample()
    nsp.insertion_angle_tip.resample()
    nsp.insertion_angle_decay_rate.resample()
    if nsp.growth_requires_dormancy:
        internode_length_max = max(f32(nsp.internode_length_max.val() - f32(nsp.internode_length_decay_rate.val()
                                                                           * f32(parent_node_count
                                                                                 - phytomer.shoot_index[0] - 1))),
                                   nsp.internode_length_min.val())
    else:
        internode_length_max = nsp.internode_length_max.val()
    nsp.internode_length_max.resample()
    nsp.internode_length_decay_rate.resample()
    nsp.internode_length_min.resample()
    internode_radius = _fmin(nsp.phytomer_parameters.internode.radius_initial.val(), phytomer.internode_radius_initial)
    nsp.phytomer_parameters.internode.radius_initial.resample()
    childID = addChildShoot(pa, plantID, shoot.ID, node_index, 1, base_rotation, internode_radius, internode_length_max,
                            f32(0.01), f32(0.01), f32(0), vbud.shoot_type_label, parent_petiole_index)
    tree = inst.shoot_tree
    if phytomer.internode_radius_initial > 0:
        tree[childID].growth_state.internode_radius_cap = phytomer.internode_radius_initial
    if getattr(inst, "leaf_angle_tracking_enabled", False):
        from .leaf_angle_tracking import add_to_cohort                 # W5-D (PA:12221)
        add_to_cohort(inst, childID, 0)
    phytomer.setVegetativeBudState(BUD_DEAD, vbud)
    vbud.shoot_ID = childID
    tree[childID].isdormant = False


__all__ = ["advanceTime", "addChildShoot", "appendPhytomerToShoot", "incrementPhytomerInternodeGirth"]
