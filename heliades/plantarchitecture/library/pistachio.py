# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:4853-5105 and src/Assets.cpp:2151-2268
# (Copyright (C) 2016-2026 Brian Bailey).
"""Pistachio tree (W5-D): ``initializePistachioTreeShoots``, ``buildPistachioTree`` and its assets.

The builder lays out a straight trunk of ``trunk_height / 0.05`` nodes with a random lean and roll, kills its buds,
and attaches the scaffolds in opposite pairs (pair ``m`` on the ``m``-th node below the top, rotated by
``m * pi / num_pairs``), each a single blind node. Phenology 165 / -1 / -1 / 7 / 20 / 200, ``max_age`` 1825.

The phytomer callback trains the young tree (UC ANR pistachio establishment, AS:2179-2268): every scaffold is
tipped at 0.45 m in its first season and regrows from five buds below the cut; in the first winter the
secondaries of each scaffold are thinned to the three longest and headed to 0.30 m, regrowing from two buds below
each cut. The cuts are deferred to the end of the time step (``seasons.pruneBranch`` from a callback).

The model declares a leaf-inclination Beta of (1.32, 1.85), so ``buildPlantInstanceFromLibrary`` enables leaf
elevation-angle tracking for it (``leaf_angle_tracking.py``).
"""
from __future__ import annotations

import math

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_DEAD, BUD_DORMANT, LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from ..random_parameter import PlantArchitectureError
from .build_parameters import getBuildParameterIntegerValue, getBuildParameterValue
from .cherrytomato import _hook
from .trees import (cutShootToLength, growSpursFromUnbrokenBuds, kill_trunk_buds, pistachio_trunk_internode_length,
                    releaseBudsBelowCut, tree_num_scaffolds, tree_scaffold_angle, tree_trunk_height, trunkMaxNodes,
                    trunk_node_count)

f32 = np.float32
PI_F = f32(math.pi)

PistachioFruitPrototype = _hook("PistachioFruitPrototype")                      # AS:2151
PistachioFlowerPrototype = _hook("PistachioFlowerPrototype")                    # AS:2157 (not referenced by the model)
PistachioPhytomerCreationFunction = _hook("PistachioPhytomerCreationFunction")  # AS:2163
PistachioPhytomerCallbackFunction = _hook("PistachioPhytomerCallbackFunction")  # AS:2198

# AS:2179-2196
pistachio_scaffold_tip_length = f32(0.45)
pistachio_secondary_head_length = f32(0.30)
pistachio_secondaries_per_scaffold = 3
pistachio_buds_released_per_cut = 2
pistachio_buds_released_at_tipping = 5
pistachio_spur_fraction = f32(0)


@ports("PlantArchitecture::initializePistachioTreeShoots")
def initializePistachioTreeShoots(pa):
    gen = context_generator(pa.context_ptr)
    lp = LeafPrototype(gen)
    lp.leaf_texture_file[0] = "PistachioLeaf.png"
    lp.leaf_aspect_ratio = f32(0.65)
    lp.midrib_fold_fraction = 0.
    lp.longitudinal_curvature.uniformDistribution(-0.4, 0.4)
    lp.lateral_curvature = 0.
    lp.wave_period = f32(0.3)
    lp.wave_amplitude = f32(0.1)
    lp.subdivisions = 3
    lp.unique_prototypes = 5

    pp = PhytomerParameters(gen)
    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(132.5, 142.5)
    pp.internode.radius_initial = 0.002
    pp.internode.length_segments = 1
    pp.internode.image_texture = "OliveBark.jpg"
    pp.internode.max_floral_buds_per_petiole = 3

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(-60, -45)
    pp.petiole.taper = 0.1
    pp.petiole.curvature.uniformDistribution(-800, 800)
    pp.petiole.length = 0.075
    pp.petiole.radius = 0.001
    pp.petiole.length_segments = 1
    pp.petiole.radial_subdivisions = 3
    pp.petiole.color = make_RGBcolor(0.6, 0.6, 0.4)

    pp.leaf.leaves_per_petiole = 3
    pp.leaf.prototype_scale = 0.105
    pp.leaf.leaflet_offset = 0.3
    pp.leaf.leaflet_scale = 0.7
    pp.leaf.pitch.uniformDistribution(-20, 20)
    pp.leaf.roll.uniformDistribution(-20, 20)
    pp.leaf.prototype = lp

    pp.peduncle.length = 0.1
    pp.peduncle.radius = 0.001
    pp.peduncle.pitch = 60
    pp.peduncle.roll = 0
    pp.peduncle.length_segments = 1
    pp.peduncle.curvature.uniformDistribution(500, 900)
    pp.peduncle.color = make_RGBcolor(0.7, 0.72, 0.7)

    pp.inflorescence.flowers_per_peduncle = 16
    pp.inflorescence.flower_offset = 0.08
    pp.inflorescence.pitch.uniformDistribution(50, 70)
    pp.inflorescence.roll.uniformDistribution(0, 360)
    pp.inflorescence.flower_prototype_scale = 0.025
    pp.inflorescence.fruit_prototype_scale = 0.025
    pp.inflorescence.fruit_prototype_function = PistachioFruitPrototype

    trunk = ShootParameters(gen)
    trunk.phytomer_parameters = pp
    trunk.phytomer_parameters.internode.phyllotactic_angle = 180
    trunk.phytomer_parameters.internode.radius_initial = 0.015
    trunk.phytomer_parameters.internode.radial_subdivisions = 20
    trunk.max_nodes = trunkMaxNodes(tree_trunk_height, pistachio_trunk_internode_length)
    trunk.girth_area_factor = f32(1.4)
    trunk.vegetative_bud_break_probability_min = 0
    trunk.vegetative_bud_break_time = 0
    trunk.tortuosity = 15
    trunk.internode_length_max = 0.05
    trunk.internode_length_decay_rate = 0
    trunk.defineChildShootTypes(["proleptic"], [1])

    pro = ShootParameters(gen)
    pro.phytomer_parameters = pp
    pro.phytomer_parameters.phytomer_creation_function = PistachioPhytomerCreationFunction
    pro.phytomer_parameters.phytomer_callback_function = PistachioPhytomerCallbackFunction
    pro.phytomer_parameters.internode.pitch.uniformDistribution(-4, 4)
    pro.phytomer_parameters.internode.radius_initial = 0.004
    pro.max_nodes = 20
    pro.max_nodes_per_season.uniformDistribution(8, 10)
    pro.phyllochron_min = 2.0
    pro.elongation_rate_max = 0.25
    pro.girth_area_factor = f32(2.2)
    pro.vegetative_bud_break_probability_max = 0.38
    pro.vegetative_bud_break_probability_min = 0.38
    pro.vegetative_bud_break_probability_decay_rate = 0.5
    pro.vegetative_bud_break_time = 0
    pro.gravitropic_curvature = 300
    pro.tortuosity = 30
    pro.insertion_angle_tip.uniformDistribution(40, 50)
    pro.insertion_angle_decay_rate = 18
    pro.base_roll.uniformDistribution(0, 360)
    pro.internode_length_max = 0.06
    pro.internode_length_min = 0.004
    pro.internode_length_decay_rate = 0.06
    pro.fruit_set_probability = 0.2
    pro.flower_bud_break_probability = 0.35
    pro.max_terminal_floral_buds = 0
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False

    sca = pro.copy()
    sca.vegetative_bud_break_probability_max = 1.0
    sca.vegetative_bud_break_probability_min = 0.2
    sca.vegetative_bud_break_probability_decay_rate = 0.7
    sca.max_nodes = 30
    sca.gravitropic_curvature = 180
    sca.defineChildShootTypes(["proleptic"], [1.0])

    spur = pro.copy()
    spur.phytomer_parameters.internode.radius_initial = 0.002
    spur.max_nodes = 40
    spur.max_nodes_per_season = 4
    spur.internode_length_max = 0.004
    spur.internode_length_min = 0.004
    spur.internode_length_decay_rate = 0
    spur.vegetative_bud_break_probability_max = 0
    spur.vegetative_bud_break_probability_min = 0
    spur.vegetative_bud_break_probability_decay_rate = 0
    spur.gravitropic_curvature = 0
    spur.defineChildShootTypes(["spur"], [1.0])

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("scaffold", sca)
    pa.defineShootType("proleptic", pro)
    pa.defineShootType("spur", spur)


@ports("PlantArchitecture::buildPistachioTree")
def buildPistachioTree(pa, base_position):
    """PL:5031-5105."""
    if not pa.shoot_types:
        initializePistachioTreeShoots(pa)
    ctx = pa.context_ptr
    bp = pa.current_build_parameters
    trunk_height = getBuildParameterValue(bp, tree_trunk_height, f32(1.0))
    num_scaffolds = int(getBuildParameterIntegerValue(bp, tree_num_scaffolds, 4))
    scaffold_angle = getBuildParameterValue(bp, tree_scaffold_angle, f32(57))
    trunk_internode_length = pistachio_trunk_internode_length
    trunk_nodes = trunk_node_count(trunk_height, trunk_internode_length)
    scaffold_radius, scaffold_length, scaffold_nodes = f32(0.007), f32(0.06), 1

    plantID = pa.addPlantInstance(base_position, 0)
    radius = pa.shoot_types["trunk"].phytomer_parameters.internode.radius_initial.val()
    # make_AxisRotation(randu(0, 0.05f * M_PI), randu(0, 2.f * M_PI), randu(0, 2.f * M_PI)): right to left
    roll = ctx.randu(f32(0), f32(2.0 * math.pi))
    yaw = ctx.randu(f32(0), f32(2.0 * math.pi))
    pitch = ctx.randu(f32(0), f32(float(f32(0.05)) * math.pi))
    uID_trunk = pa.addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(pitch, yaw, roll), radius,
                                    trunk_internode_length, f32(1), f32(1), f32(0), "trunk")
    pa.appendPhytomerToShoot(plantID, uID_trunk, pa.shoot_types["trunk"].phytomer_parameters, f32(0), f32(0.01),
                             f32(1), f32(1))
    kill_trunk_buds(pa, plantID, uID_trunk)

    from ...core import vec as V
    pitch = V.deg2rad(scaffold_angle)
    num_pairs = (num_scaffolds + 1) // 2
    tree = pa.plant_instances[plantID].shoot_tree
    trunk_node_count_ = tree[uID_trunk].current_node_number
    if num_pairs > trunk_node_count_:
        raise PlantArchitectureError(
            f"ERROR (PlantArchitecture::buildPistachioTree): {num_scaffolds} scaffolds need {num_pairs} trunk nodes to "
            f"attach to, but a trunk_height of {float(trunk_height):f} m gives only {trunk_node_count_}. Increase "
            "trunk_height or reduce num_scaffolds.")
    for scaffold in range(num_scaffolds):
        pair = scaffold // 2
        azimuth = f32(f32(f32(f32(pair) * PI_F) / f32(num_pairs)) + f32(f32(scaffold % 2) * PI_F))
        # make_AxisRotation(pitch + randu(-0.15, 0.15), azimuth, randu(0, 2 * float(M_PI))): roll drawn first
        roll = ctx.randu(f32(0), f32(f32(2) * PI_F))
        p = f32(pitch + ctx.randu(f32(-0.15), f32(0.15)))
        scaffoldID = pa.addChildShoot(plantID, uID_trunk, trunk_node_count_ - 1 - pair, scaffold_nodes,
                                      make_AxisRotation(p, azimuth, roll), scaffold_radius, scaffold_length, f32(1),
                                      f32(1), f32(0.5), "scaffold", 0)
        for phytomer in tree[scaffoldID].phytomers:
            phytomer.setVegetativeBudState(BUD_DEAD)
            phytomer.setFloralBudState(BUD_DEAD)
    from ..seasons import makePlantDormant
    makePlantDormant(pa, plantID)
    pa.setPlantPhenologicalThresholds(plantID, 165, -1, -1, 7, 20, 200)
    pa.plant_instances[plantID].max_age = f32(1825)
    return plantID


@PistachioFruitPrototype.implement
@ports("PistachioFruitPrototype")
def _pistachio_fruit_prototype(context, subdivisions, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/PistachioNut.obj")


@PistachioFlowerPrototype.implement
@ports("PistachioFlowerPrototype")
def _pistachio_flower_prototype(context, subdivisions, flower_is_open, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/OliveFlower_open.obj")


@PistachioPhytomerCreationFunction.implement
@ports("PistachioPhytomerCreationFunction")
def _pistachio_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:2163-2177: node 0 is blind; a lateral with short internodes (< 0.01, compared in double) is a spur."""
    if shoot_node_index == 0:
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.setFloralBudState(BUD_DEAD)
    if float(phytomer.internode_length_max) < 0.01:
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.parent_shoot_ptr.growth_state.max_nodes_per_season_growth_cap = 6


@PistachioPhytomerCallbackFunction.implement
@ports("PistachioPhytomerCallbackFunction")
def _pistachio_phytomer_callback(phytomer):
    """AS:2198-2268: spurs (fraction 0: a no-op), dormant axillary flower buds below the top four nodes, and the
    training rules, run once per shoot per time step from its tip phytomer."""
    from ..seasons import pruneBranch, std_sort
    growSpursFromUnbrokenBuds(phytomer, pistachio_spur_fraction)
    if phytomer.isdormant:
        if int(phytomer.shoot_index[0]) <= int(phytomer.shoot_index[1]) - 4:
            phytomer.setFloralBudState(BUD_DORMANT)
    shoot = phytomer.parent_shoot_ptr
    if not shoot.phytomers or phytomer is not shoot.phytomers[-1]:
        return
    pa = shoot.plantarchitecture_ptr
    tree = pa.plant_instances[shoot.plantID].shoot_tree
    is_scaffold = shoot.shoot_type_label == "scaffold"
    is_secondary = (shoot.shoot_type_label != "spur" and shoot.parent_shoot_ID >= 0
                    and tree[shoot.parent_shoot_ID].shoot_type_label == "scaffold")
    if (is_scaffold or is_secondary) and not shoot.meristem_is_alive and not shoot.isdormant and shoot.dormancy_cycles == 1:
        buds_released = pistachio_buds_released_at_tipping if is_scaffold else pistachio_buds_released_per_cut
        releaseBudsBelowCut(shoot, len(shoot.phytomers), buds_released, "proleptic")
        for childID in pa.getChildShootIDs(shoot.plantID, shoot.ID):
            child = tree[childID]
            if child.dormancy_cycles == 0 and child.parent_node_index + buds_released >= len(shoot.phytomers):
                child.internode_length_max_shoot_initial = child.shoot_parameters.internode_length_max.val()
    if is_scaffold:
        if shoot.meristem_is_alive and not shoot.isdormant and shoot.calculateShootLength() >= pistachio_scaffold_tip_length:
            cutShootToLength(shoot, pistachio_scaffold_tip_length)
            shoot.meristem_is_alive = False
        if shoot.isdormant and shoot.dormancy_cycles == 2:
            secondaries = [(tree[c].calculateShootLength(), c) for c in pa.getChildShootIDs(shoot.plantID, shoot.ID)]
            secondaries = std_sort(secondaries, lambda a, b: a[0] > b[0])
            for i, (_, sid) in enumerate(secondaries):
                if i >= pistachio_secondaries_per_scaffold:
                    pruneBranch(pa, shoot.plantID, sid, 0)
                else:
                    cutShootToLength(tree[sid], pistachio_secondary_head_length)


__all__ = ["initializePistachioTreeShoots", "buildPistachioTree"]
