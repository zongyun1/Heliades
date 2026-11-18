# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:6362-6540 and src/Assets.cpp:4142-4164
# (Copyright (C) 2016-2026 Brian Bailey).
"""Walnut tree (W5-D): ``initializeWalnutTreeShoots``, ``buildWalnutTree`` and its assets.

The builder lays out a straight trunk of ``trunk_height / 0.04`` nodes (``addBaseStemShoot`` with a small random
lean), appends a zero-radius phytomer on top, kills the trunk's buds, attaches ``num_scaffolds`` scaffolds of
5-7 nodes one per trunk node below the top, and makes the tree dormant (phenology 165 / -1 / 3 / 7 / 20 / 200,
``max_age`` 1460). Walnut leaves are 5-leaflet compound leaves on 5-segment curved petioles.
"""
from __future__ import annotations

import math

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from .build_parameters import getBuildParameterIntegerValue, getBuildParameterValue
from .cherrytomato import _hook
from .trees import (checkScaffoldsFitTrunk, kill_trunk_buds, scaffold_yaw, tree_num_scaffolds, tree_trunk_height,
                    trunkMaxNodes, trunk_node_count, walnut_scaffold_angle, walnut_trunk_internode_length)

f32 = np.float32

WalnutFruitPrototype = _hook("WalnutFruitPrototype")                      # AS:4142
WalnutFlowerPrototype = _hook("WalnutFlowerPrototype")                    # AS:4148
WalnutPhytomerCreationFunction = _hook("WalnutPhytomerCreationFunction")  # AS:4154
WalnutPhytomerCallbackFunction = _hook("WalnutPhytomerCallbackFunction")  # AS:4163 (not referenced by the model)


@ports("PlantArchitecture::initializeWalnutTreeShoots")
def initializeWalnutTreeShoots(pa):
    gen = context_generator(pa.context_ptr)
    lp = LeafPrototype(gen)
    lp.leaf_texture_file[0] = "WalnutLeaf.png"
    lp.leaf_aspect_ratio = f32(0.5)
    lp.midrib_fold_fraction = f32(0.15)
    lp.longitudinal_curvature = f32(-0.2)
    lp.lateral_curvature = f32(0.1)
    lp.wave_period.uniformDistribution(0.08, 0.15)
    lp.wave_amplitude.uniformDistribution(0.02, 0.04)
    lp.subdivisions = 3
    lp.unique_prototypes = 5

    pp = PhytomerParameters(gen)
    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(137.5 - 20, 137.5 + 20)
    pp.internode.radius_initial = 0.004
    pp.internode.length_segments = 1
    pp.internode.image_texture = "AppleBark.jpg"
    pp.internode.max_floral_buds_per_petiole = 3

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(-80, -70)
    pp.petiole.taper = 0.2
    pp.petiole.curvature.uniformDistribution(-1000, 0)
    pp.petiole.length = 0.15
    pp.petiole.radius = 0.0015
    pp.petiole.length_segments = 5
    pp.petiole.radial_subdivisions = 3
    pp.petiole.color = make_RGBcolor(0.61, 0.5, 0.24)

    pp.leaf.leaves_per_petiole = 5
    pp.leaf.pitch.uniformDistribution(-40, 0)
    pp.leaf.prototype_scale = 0.14
    pp.leaf.leaflet_scale = 0.7
    pp.leaf.leaflet_offset = 0.35
    pp.leaf.prototype = lp

    pp.peduncle.length = 0.02
    pp.peduncle.radius = 0.0005
    pp.peduncle.pitch = 90
    pp.peduncle.roll = 90
    pp.peduncle.length_segments = 1

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.flower_prototype_scale = 0.03
    pp.inflorescence.flower_prototype_function = WalnutFlowerPrototype
    pp.inflorescence.fruit_prototype_scale = 0.075
    pp.inflorescence.fruit_prototype_function = WalnutFruitPrototype

    trunk = ShootParameters(gen)
    trunk.phytomer_parameters = pp
    trunk.phytomer_parameters.internode.phyllotactic_angle = 0
    trunk.phytomer_parameters.internode.radius_initial = 0.01
    trunk.phytomer_parameters.internode.radial_subdivisions = 24
    trunk.max_nodes = trunkMaxNodes(tree_trunk_height, walnut_trunk_internode_length)
    trunk.girth_area_factor = f32(1.4)
    trunk.vegetative_bud_break_probability_min = 0
    trunk.vegetative_bud_break_time = 0
    trunk.tortuosity = 20
    trunk.internode_length_max = 0.05
    trunk.internode_length_decay_rate = 0
    trunk.defineChildShootTypes(["scaffold"], [1])

    pro = ShootParameters(gen)
    pro.phytomer_parameters = pp
    pro.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2)
    pro.phytomer_parameters.phytomer_creation_function = WalnutPhytomerCreationFunction
    pro.max_nodes = 24
    pro.max_nodes_per_season = 12
    pro.phyllochron_min = 2.
    pro.elongation_rate_max = 0.15
    pro.girth_area_factor = f32(6.7)
    pro.vegetative_bud_break_probability_min = 0.05
    pro.vegetative_bud_break_probability_decay_rate = 0.7
    pro.vegetative_bud_break_time = 3
    pro.gravitropic_curvature = 300
    pro.tortuosity = 50
    pro.insertion_angle_tip.uniformDistribution(20, 25)
    pro.insertion_angle_decay_rate = 15
    pro.internode_length_max = 0.08
    pro.internode_length_min = 0.01
    pro.internode_length_decay_rate = 0.006
    pro.fruit_set_probability = 0.3
    pro.flower_bud_break_probability = 0.3
    pro.max_terminal_floral_buds = 4
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False
    pro.defineChildShootTypes(["proleptic"], [1.0])

    sca = pro.copy()
    sca.girth_area_factor = f32(2.6)
    sca.phytomer_parameters.internode.radial_subdivisions = 10
    sca.max_nodes = 30
    sca.gravitropic_curvature = 300
    sca.internode_length_max = 0.06
    sca.tortuosity = 66.7
    sca.defineChildShootTypes(["proleptic"], [1.0])

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("scaffold", sca)
    pa.defineShootType("proleptic", pro)


@ports("PlantArchitecture::buildWalnutTree")
def buildWalnutTree(pa, base_position):
    """PL:6484-6540."""
    if not pa.shoot_types:
        initializeWalnutTreeShoots(pa)
    ctx = pa.context_ptr
    bp = pa.current_build_parameters
    trunk_height = getBuildParameterValue(bp, tree_trunk_height, f32(0.8))
    num_scaffolds = int(getBuildParameterIntegerValue(bp, tree_num_scaffolds, 4))
    scaffold_angle = getBuildParameterValue(bp, walnut_scaffold_angle, f32(40))
    trunk_internode_length = walnut_trunk_internode_length
    trunk_nodes = trunk_node_count(trunk_height, trunk_internode_length)
    checkScaffoldsFitTrunk("buildWalnutTree", num_scaffolds, trunk_nodes, trunk_height)
    scaffold_radius, scaffold_length = f32(0.007), f32(0.06)

    plantID = pa.addPlantInstance(base_position, 0)
    radius = pa.shoot_types["trunk"].phytomer_parameters.internode.radius_initial.val()
    # make_AxisRotation(randu(0, 0.05f * M_PI), randu(0, 2.f * M_PI), 0.f * M_PI): right to left, yaw first
    yaw = ctx.randu(f32(0), f32(2.0 * math.pi))
    pitch = ctx.randu(f32(0), f32(float(f32(0.05)) * math.pi))
    uID_trunk = pa.addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(pitch, yaw, 0), radius,
                                    trunk_internode_length, f32(1), f32(1), f32(0), "trunk")
    pa.appendPhytomerToShoot(plantID, uID_trunk, pa.shoot_types["trunk"].phytomer_parameters, f32(0), f32(0.01),
                             f32(1), f32(1))
    kill_trunk_buds(pa, plantID, uID_trunk)
    tree = pa.plant_instances[plantID].shoot_tree
    for i in range(num_scaffolds):
        pitch = f32(V.deg2rad(scaffold_angle) + ctx.randu(f32(-0.1), f32(0.1)))
        scaffold_nodes = int(ctx.randu(5, 7))
        yaw = scaffold_yaw(ctx, i, num_scaffolds)
        pa.addChildShoot(plantID, uID_trunk, tree[uID_trunk].current_node_number - i - 1, scaffold_nodes,
                         make_AxisRotation(pitch, yaw, 0), scaffold_radius, scaffold_length, f32(1), f32(1), f32(0.5),
                         "scaffold", 0)
    from ..seasons import makePlantDormant
    makePlantDormant(pa, plantID)
    pa.setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200)
    pa.plant_instances[plantID].max_age = f32(1460)
    return plantID


@WalnutFruitPrototype.implement
@ports("WalnutFruitPrototype")
def _walnut_fruit_prototype(context, subdivisions, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/WalnutHull.obj")


@WalnutFlowerPrototype.implement
@ports("WalnutFlowerPrototype")
def _walnut_flower_prototype(context, subdivisions, flower_is_open, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/AlmondFlower.obj")


@WalnutPhytomerCreationFunction.implement
@ports("WalnutPhytomerCreationFunction")
def _walnut_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:4154-4161: the first four nodes of a shoot are blind."""
    if shoot_node_index < 4:
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.setFloralBudState(BUD_DEAD)


@WalnutPhytomerCallbackFunction.implement
@ports("WalnutPhytomerCallbackFunction")
def _walnut_phytomer_callback(phytomer):
    return None


__all__ = ["initializeWalnutTreeShoots", "buildWalnutTree"]
