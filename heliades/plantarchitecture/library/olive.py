# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:404-408, 4695-4851 and
# src/Assets.cpp:2112-2149 (Copyright (C) 2016-2026 Brian Bailey).
"""Olive tree (W5-E): ``initializeOliveTreeShoots``, ``buildOliveTree`` and its assets.

The builder lays out a straight trunk of ``trunk_height / 0.03`` nodes with a small random lean and roll, appends a
zero-radius phytomer on top, kills the trunk's buds, attaches four scaffolds of 5-7 nodes one per trunk node below
the top, and makes the tree dormant. Phenology 165 / -1 / 3 / 7 / 20 / 200 with a 600-day leaf lifespan, evergreen;
``max_age`` 1825.

Olive leaves are a custom prototype (``OliveLeafPrototype``): two textured tiles back to back, the lower one flipped
about x. The model carries no flowers (the flower prototype is commented out in the C++), only fruit
(``OliveFruitPrototype``, the OBJ's ``fruit`` part coloured green). ``OlivePhytomerCallbackFunction`` kills the
floral buds of all but the last eight nodes of a dormant shoot. ``OlivePhytomerCreationFunction`` and
``OliveFlowerPrototype`` are defined but not referenced by the model (ported anyway).
"""
from __future__ import annotations

import math

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from .build_parameters import BuildParameterDeclaration, getBuildParameterValue
from .cherrytomato import _hook
from .trees import kill_trunk_buds, olive_trunk_internode_length, scaffold_yaw, trunkMaxNodes

f32 = np.float32
PI_F = f32(math.pi)

OliveLeafPrototype = _hook("OliveLeafPrototype")                        # AS:2112
OliveFruitPrototype = _hook("OliveFruitPrototype")                      # AS:2125
OliveFlowerPrototype = _hook("OliveFlowerPrototype")                    # AS:2133 (not referenced by the model)
OlivePhytomerCreationFunction = _hook("OlivePhytomerCreationFunction")  # AS:2139 (not referenced by the model)
OlivePhytomerCallbackFunction = _hook("OlivePhytomerCallbackFunction")  # AS:2142

# PL:408
olive_trunk_height = BuildParameterDeclaration("trunk_height", 0.2, 1.5, False, "trunk height in meters")


@ports("PlantArchitecture::initializeOliveTreeShoots")
def initializeOliveTreeShoots(pa):
    gen = context_generator(pa.context_ptr)
    lp = LeafPrototype(gen)
    lp.prototype_function = OliveLeafPrototype
    lp.unique_prototypes = 1

    pp = PhytomerParameters(gen)
    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(80, 100)
    pp.internode.radius_initial = 0.002
    pp.internode.length_segments = 1
    pp.internode.image_texture = "OliveBark.jpg"
    pp.internode.max_floral_buds_per_petiole = 3

    pp.petiole.petioles_per_internode = 2
    pp.petiole.pitch.uniformDistribution(-40, -20)
    pp.petiole.taper = 0.1
    pp.petiole.curvature = 0
    pp.petiole.length = 0.01
    pp.petiole.radius = 0.0005
    pp.petiole.length_segments = 1
    pp.petiole.radial_subdivisions = 3
    pp.petiole.color = make_RGBcolor(0.61, 0.5, 0.24)

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.prototype_scale = 0.06
    pp.leaf.prototype = lp

    pp.peduncle.length = 0.065
    pp.peduncle.radius = 0.001
    pp.peduncle.pitch = 60
    pp.peduncle.roll = 0
    pp.peduncle.length_segments = 1
    pp.peduncle.color = make_RGBcolor(0.7, 0.72, 0.7)

    pp.inflorescence.flowers_per_peduncle = 10
    pp.inflorescence.flower_offset = 0.13
    pp.inflorescence.pitch.uniformDistribution(80, 100)
    pp.inflorescence.roll.uniformDistribution(0, 360)
    pp.inflorescence.flower_prototype_scale = 0.01
    pp.inflorescence.fruit_prototype_scale = 0.025
    pp.inflorescence.fruit_prototype_function = OliveFruitPrototype

    trunk = ShootParameters(gen)
    trunk.phytomer_parameters = pp
    trunk.phytomer_parameters.internode.phyllotactic_angle = 0
    trunk.phytomer_parameters.internode.radius_initial = 0.015
    trunk.phytomer_parameters.internode.radial_subdivisions = 20
    trunk.max_nodes = trunkMaxNodes(olive_trunk_height, olive_trunk_internode_length)
    trunk.girth_area_factor = f32(1.6)
    trunk.vegetative_bud_break_probability_min = 0
    trunk.vegetative_bud_break_time = 0
    trunk.tortuosity = 20
    trunk.internode_length_max = 0.05
    trunk.internode_length_decay_rate = 0
    trunk.defineChildShootTypes(["scaffold"], [1])

    pro = ShootParameters(gen)
    pro.phytomer_parameters = pp
    pro.phytomer_parameters.phytomer_callback_function = OlivePhytomerCallbackFunction
    pro.max_nodes.uniformDistribution(16, 24)
    pro.max_nodes_per_season.uniformDistribution(8, 12)
    pro.phyllochron_min = 2.0
    pro.elongation_rate_max = 0.25
    pro.girth_area_factor = f32(3.4)
    pro.vegetative_bud_break_probability_min = 0.025
    pro.vegetative_bud_break_probability_decay_rate = 1.0
    pro.vegetative_bud_break_time = 30
    pro.gravitropic_curvature.uniformDistribution(550, 650)
    pro.tortuosity = 100
    pro.insertion_angle_tip.uniformDistribution(35, 40)
    pro.insertion_angle_decay_rate = 2
    pro.internode_length_max = 0.05
    pro.internode_length_min = 0.03
    pro.internode_length_decay_rate = 0.004
    pro.fruit_set_probability = 0.25
    pro.flower_bud_break_probability = 0.25
    pro.max_terminal_floral_buds = 0
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False
    pro.defineChildShootTypes(["proleptic"], [1.0])

    sca = pro.copy()
    sca.girth_area_factor = f32(1.1)
    sca.phytomer_parameters.internode.radial_subdivisions = 10
    sca.max_nodes = 30
    sca.max_nodes_per_season = 10
    sca.gravitropic_curvature = 700
    sca.internode_length_max = 0.04
    sca.tortuosity = 75
    sca.defineChildShootTypes(["proleptic"], [1.0])

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("scaffold", sca)
    pa.defineShootType("proleptic", pro)


@ports("PlantArchitecture::buildOliveTree")
def buildOliveTree(pa, base_position):
    """PL:4810-4851."""
    if not pa.shoot_types:
        initializeOliveTreeShoots(pa)
    ctx = pa.context_ptr
    trunk_height = getBuildParameterValue(pa.current_build_parameters, olive_trunk_height, f32(0.6))
    trunk_nodes = max(1, int(f32(f32(trunk_height) / olive_trunk_internode_length)))

    plantID = pa.addPlantInstance(base_position, 0)
    radius = pa.shoot_types["trunk"].phytomer_parameters.internode.radius_initial.val()
    # make_AxisRotation(randu(0, 0.025f * M_PI), randu(0, 2.f * M_PI), randu(0, 2.f * M_PI)): right to left
    roll = ctx.randu(f32(0), f32(2.0 * math.pi))
    yaw = ctx.randu(f32(0), f32(2.0 * math.pi))
    pitch = ctx.randu(f32(0), f32(float(f32(0.025)) * math.pi))
    uID_trunk = pa.addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(pitch, yaw, roll), radius,
                                    olive_trunk_internode_length, f32(1), f32(1), f32(0), "trunk")
    pa.appendPhytomerToShoot(plantID, uID_trunk, pa.shoot_types["trunk"].phytomer_parameters, f32(0), f32(0.01),
                             f32(1), f32(1))
    kill_trunk_buds(pa, plantID, uID_trunk)

    tree = pa.plant_instances[plantID].shoot_tree
    Nscaffolds = 4
    for i in range(Nscaffolds):
        # randu(deg2rad(30), deg2rad(35)): deg2rad(int) -> float, right to left
        hi = V.deg2rad(f32(35))
        lo = V.deg2rad(f32(30))
        pitch = ctx.randu(lo, hi)
        # addChildShoot(..., randu(5, 7), make_AxisRotation(pitch, yaw(randu), 0), ...): yaw drawn first
        yaw = scaffold_yaw(ctx, i, Nscaffolds)
        nodes = int(ctx.randu(5, 7))
        pa.addChildShoot(plantID, uID_trunk, tree[uID_trunk].current_node_number - i - 1, nodes,
                         make_AxisRotation(pitch, yaw, 0), f32(0.007),
                         pa.shoot_types["scaffold"].internode_length_max.val(), f32(1), f32(1), f32(0.5), "scaffold", 0)
    from ..seasons import makePlantDormant
    makePlantDormant(pa, plantID)
    pa.setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 20, 200, 600, True)
    pa.plant_instances[plantID].max_age = f32(1825)
    return plantID


@OliveLeafPrototype.implement
@ports("OliveLeafPrototype")
def _olive_leaf_prototype(context, prototype_parameters, compound_leaf_index) -> int:
    """AS:2112-2123."""
    from .bean import plugin_asset
    n = int(prototype_parameters.subdivisions)
    upper = context.add_tile_textured((0.5, 0, 0), (1, 0.2), (0.0, 0.0), (n, n),
                                      plugin_asset("assets/textures/OliveLeaf_upper.png"))
    lower = context.add_tile_textured((0.5, 0, -1e-4), (1, 0.2), (0.0, 0.0), (n, n),
                                      plugin_asset("assets/textures/OliveLeaf_lower.png"))
    for u in lower:
        context.rotate_primitive(u, f32(math.pi), "x")
    return context.add_polymesh_object(list(upper) + list(lower))


@OliveFruitPrototype.implement
@ports("OliveFruitPrototype")
def _olive_fruit_prototype(context, subdivisions, build_parameters=None):
    """AS:2125-2131."""
    from .bean import load_obj_black
    objID = load_obj_black(context, "assets/obj/OliveFruit.obj")
    fruit = context.filter_primitives_by_data(context.object_primitive_uuids(objID), "object_label", "fruit")
    context.set_primitive_color(fruit, make_RGBcolor(0.65, 0.7, 0.4))
    return objID


@OliveFlowerPrototype.implement
@ports("OliveFlowerPrototype")
def _olive_flower_prototype(context, subdivisions, flower_is_open, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/OliveFlower_open.obj")


@OlivePhytomerCreationFunction.implement
@ports("OlivePhytomerCreationFunction")
def _olive_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    return None


@OlivePhytomerCallbackFunction.implement
@ports("OlivePhytomerCallbackFunction")
def _olive_phytomer_callback(phytomer):
    """AS:2142-2149: a dormant phytomer more than eight nodes below the tip loses its floral buds."""
    if phytomer.isdormant:
        if int(phytomer.shoot_index[0]) < int(phytomer.shoot_index[1]) - 8:
            phytomer.setFloralBudState(BUD_DEAD)


__all__ = ["initializeOliveTreeShoots", "buildOliveTree", "olive_trunk_height"]
