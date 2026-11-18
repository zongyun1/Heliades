# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:4459-4573 (Copyright (C) 2016-2026 Brian Bailey).
"""Ground cherry weed (``initializeGroundCherryWeedShoots``, ``buildGroundCherryWeedPlant``), W5-C.

Line for line with the C++. One ``mainstem`` shoot type that branches into more ``mainstem`` shoots, with
``GenericLeafPrototype`` blades (``GroundCherryLeaf.png``, distributed aspect ratio). The flower is the bindweed
flower (``BindweedFlowerPrototype``). The fruit prototype is commented out in Helios
(``// fruit_prototype_function = GroundCherryFruitPrototype``), so a fruit that sets builds no geometry: the
null-function branches of the inflorescence code (``fruit_prototype_function == nullptr``) are what this model
reaches after fruit set.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, make_RGBcolor
from .bindweed import BindweedFlowerPrototype
from .tomato import base_rotation_draws

f32 = np.float32


@ports("PlantArchitecture::initializeGroundCherryWeedShoots")
def initializeGroundCherryWeedShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.leaf_texture_file[0] = "GroundCherryLeaf.png"
    leaf_prototype.leaf_aspect_ratio.uniformDistribution(0.3, 0.5)
    leaf_prototype.midrib_fold_fraction = f32(0.2)
    leaf_prototype.longitudinal_curvature = f32(0.1)
    leaf_prototype.lateral_curvature = f32(-0.3)
    leaf_prototype.wave_period = f32(0.35)
    leaf_prototype.wave_amplitude = f32(0.08)
    leaf_prototype.subdivisions = 6
    leaf_prototype.unique_prototypes = 5

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)

    pp.internode.pitch = 5
    pp.internode.phyllotactic_angle = 137.5
    pp.internode.radius_initial = 0.0005
    pp.internode.color = make_RGBcolor(0.266, 0.3375, 0.0700)
    pp.internode.length_segments = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(45, 60)
    pp.petiole.radius = 0.0005
    pp.petiole.length = 0.025
    pp.petiole.taper = 0.15
    pp.petiole.curvature.uniformDistribution(-150, -50)
    pp.petiole.color = pp.internode.color
    pp.petiole.length_segments = 2

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-30, 5)
    pp.leaf.yaw = 10
    pp.leaf.roll = 0
    pp.leaf.prototype_scale.uniformDistribution(0.06, 0.08)
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length = 0.01
    pp.peduncle.radius = 0.001
    pp.peduncle.pitch = 20
    pp.peduncle.roll = 0
    pp.peduncle.curvature = -700
    pp.peduncle.color = pp.internode.color
    pp.peduncle.length_segments = 2
    pp.peduncle.radial_subdivisions = 6

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll.uniformDistribution(-30, 30)
    pp.inflorescence.flower_prototype_scale = 0.01
    pp.inflorescence.flower_prototype_function = BindweedFlowerPrototype
    pp.inflorescence.fruit_prototype_scale = 0.06
    # pp.inflorescence.fruit_prototype_function = GroundCherryFruitPrototype   (commented out in Helios)
    pp.inflorescence.fruit_gravity_factor_fraction = 0.75

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp

    sp.max_nodes = 26
    sp.insertion_angle_tip = 50
    sp.insertion_angle_decay_rate = 0
    sp.internode_length_max = 0.015
    sp.internode_length_min = 0.0
    sp.internode_length_decay_rate = 0
    sp.base_roll = 90
    sp.base_yaw.uniformDistribution(-20, 20)
    sp.gravitropic_curvature = 700
    sp.tortuosity = 200

    sp.phyllochron_min = 1
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(2)
    sp.vegetative_bud_break_time = 10
    sp.vegetative_bud_break_probability_min = 0.1
    sp.vegetative_bud_break_probability_decay_rate = -0.5
    sp.flower_bud_break_probability = 0.25
    sp.fruit_set_probability = 0.5
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False

    sp.defineChildShootTypes(["mainstem"], [1.0])

    pa.defineShootType("mainstem", sp)


@ports("PlantArchitecture::buildGroundCherryWeedPlant")
def buildGroundCherryWeedPlant(pa, base_position):
    """PL:4553-4573."""
    if not pa.shoot_types:
        initializeGroundCherryWeedShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    base_rotation = base_rotation_draws(pa)          # 2 randu, right to left (roll first)
    pa.addBaseStemShoot(plantID, 1, base_rotation, 0.0025, 0.018, 0.01, 0.01, 0, "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, 20, -1, 20, 30, 1000)
    pa.plant_instances[plantID].max_age = f32(80)
    return plantID


__all__ = ["initializeGroundCherryWeedShoots", "buildGroundCherryWeedPlant"]
