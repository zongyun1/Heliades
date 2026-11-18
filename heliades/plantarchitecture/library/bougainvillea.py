# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:3072-3205 and src/Assets.cpp:1644-1648
# (Copyright (C) 2016-2026 Brian Bailey).
"""Bougainvillea (``initializeBougainvilleaShoots``, ``buildBougainvilleaPlant``, ``BougainvilleaFlowerPrototype``), W5-C.

Line for line with the C++. Four shoot types (``mainstem`` -> ``secondary`` -> ``tertiary`` -> ``quaternary``,
whose buds never break). The capsicum blade texture, a vestigial petiole (0.1 mm), a distributed
``flowers_per_peduncle`` (4-6) on 3-segment curved peduncles, ``flower_offset`` 0.12.

The builder plants a 3-node stem and strips it (``removeShootVegetativeBuds``, ``removeShootFloralBuds``,
``removeShootLeaves``): the first growth comes from the apex.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, make_RGBcolor
from .bean import load_obj_silent
from .strawberry import named_hook
from .tomato import base_rotation_draws

f32 = np.float32

BougainvilleaFlowerPrototype = named_hook("BougainvilleaFlowerPrototype")     # AS:1644


@ports("PlantArchitecture::initializeBougainvilleaShoots")
def initializeBougainvilleaShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.leaf_texture_file[0] = "CapsicumLeaf.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.8)
    leaf_prototype.midrib_fold_fraction = f32(0.2)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.15, f32(-0.05))
    leaf_prototype.lateral_curvature = f32(-0.25)
    leaf_prototype.wave_period = f32(0.35)
    leaf_prototype.wave_amplitude = f32(0.1)
    leaf_prototype.subdivisions = 7
    leaf_prototype.unique_prototypes = 5

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)

    pp.internode.pitch = 10
    pp.internode.phyllotactic_angle.uniformDistribution(160, 190)
    pp.internode.radius_initial = 0.001
    pp.internode.color = make_RGBcolor(0.213, 0.270, 0.056)
    pp.internode.length_segments = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(-60, -40)
    pp.petiole.radius = 0.0001
    pp.petiole.length = 0.0001
    pp.petiole.taper = 1
    pp.petiole.curvature = 0
    pp.petiole.color = pp.internode.color
    pp.petiole.length_segments = 1

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch = 0
    pp.leaf.yaw = 10
    pp.leaf.roll = 0
    pp.leaf.prototype_scale.uniformDistribution(0.08, 0.12)
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length = 0.15
    pp.peduncle.radius = 0.0015
    pp.peduncle.pitch.uniformDistribution(70, 90)
    pp.peduncle.roll.uniformDistribution(0, 180)
    pp.peduncle.curvature.uniformDistribution(-200, 200)
    pp.peduncle.color = pp.internode.color
    pp.peduncle.length_segments = 3
    pp.peduncle.radial_subdivisions = 6

    pp.inflorescence.flowers_per_peduncle.uniformDistribution(4, 6)
    pp.inflorescence.pitch = 80
    pp.inflorescence.roll.uniformDistribution(-180, 180)
    pp.inflorescence.flower_prototype_scale = 0.05
    pp.inflorescence.flower_prototype_function = BougainvilleaFlowerPrototype
    pp.inflorescence.unique_prototypes = 1
    pp.inflorescence.flower_offset = 0.12

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp

    sp.max_nodes = 60
    sp.insertion_angle_tip = 25
    sp.insertion_angle_decay_rate = 0
    sp.internode_length_max = 0.04
    sp.internode_length_min = 0.0
    sp.internode_length_decay_rate = 0
    sp.base_roll = 90
    sp.base_yaw.uniformDistribution(-20, 20)
    sp.gravitropic_curvature = 500
    sp.tortuosity = 75

    sp.phyllochron_min = 2.75
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(1)
    sp.vegetative_bud_break_time = 30
    sp.vegetative_bud_break_probability_min = 0.075
    sp.vegetative_bud_break_probability_decay_rate = 0.8
    sp.flower_bud_break_probability = 0.5
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False

    sp.defineChildShootTypes(["secondary"], [1.0])

    pa.defineShootType("mainstem", sp)

    secondary = sp.copy()
    secondary.max_nodes = 15
    secondary.defineChildShootTypes(["tertiary"], [1.0])
    pa.defineShootType("secondary", secondary)

    tertiary = secondary.copy()
    tertiary.defineChildShootTypes(["quaternary"], [1.0])
    pa.defineShootType("tertiary", tertiary)

    quaternary = tertiary.copy()
    quaternary.vegetative_bud_break_probability_min = 0
    quaternary.vegetative_bud_break_probability_max = 0
    pa.defineShootType("quaternary", quaternary)


@ports("PlantArchitecture::buildBougainvilleaPlant")
def buildBougainvilleaPlant(pa, base_position):
    """PL:3182-3205."""
    from ..shoot import removeShootFloralBuds, removeShootLeaves, removeShootVegetativeBuds
    if not pa.shoot_types:
        initializeBougainvilleaShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    base_rotation = base_rotation_draws(pa)          # 2 randu, right to left (roll first)
    internode_length_max = pa.shoot_types["mainstem"].internode_length_max.val()
    uID_stem = pa.addBaseStemShoot(plantID, 3, base_rotation, 0.002, internode_length_max, 0.01, 0.01, 0, "mainstem")
    removeShootVegetativeBuds(pa, plantID, uID_stem)
    removeShootFloralBuds(pa, plantID, uID_stem)
    removeShootLeaves(pa, plantID, uID_stem)
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, 75, 1000, 1000, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


@BougainvilleaFlowerPrototype.implement
@ports("BougainvilleaFlowerPrototype")
def _bougainvillea_flower(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:1644-1648: the 2-argument ``loadOBJ(file, true)``."""
    return load_obj_silent(context, "assets/obj/BougainvilleaFlower.obj")


__all__ = ["initializeBougainvilleaShoots", "buildBougainvilleaPlant", "BougainvilleaFlowerPrototype"]
