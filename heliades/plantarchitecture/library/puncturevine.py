# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:5107-5227 and src/Assets.cpp:2269-2273
# (Copyright (C) 2016-2026 Brian Bailey).
"""Puncturevine (``initializePuncturevineShoots``, ``buildPuncturevinePlant``, ``PuncturevineFlowerPrototype``), W5-C.

Line for line with the C++. The structure of the bindweed (a ``base_puncturevine`` crown whose buds all break
into prostrate ``primary_puncturevine`` runners bearing ``secondary_puncturevine`` runners), with opposite leaves
(``petioles_per_internode = 2``) and even-pinnate leaves of 10 leaflets (``leaflet_offset`` 0.18): the
first library model with more than three leaflets, so the ``(fabs(ind) - 0.5f) * offset * length`` leaflet
offset (W3-B finding 2, double ``fabs``) is exercised with ``|ind| - 0.5`` other than 0.5.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from .bean import load_obj_black
from .strawberry import named_hook

f32 = np.float32

PuncturevineFlowerPrototype = named_hook("PuncturevineFlowerPrototype")     # AS:2269


@ports("PlantArchitecture::initializePuncturevineShoots")
def initializePuncturevineShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.leaf_texture_file[0] = "PuncturevineLeaf.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.4)
    leaf_prototype.midrib_fold_fraction = f32(0.2)
    leaf_prototype.longitudinal_curvature = f32(-0.1)
    leaf_prototype.lateral_curvature = f32(0.4)
    leaf_prototype.subdivisions = 1
    leaf_prototype.unique_prototypes = 1

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)

    pp.internode.pitch.uniformDistribution(0, 15)
    pp.internode.phyllotactic_angle = f32(180)
    pp.internode.radius_initial = 0.001
    pp.internode.color = make_RGBcolor(0.28, 0.18, 0.13)
    pp.internode.length_segments = 1

    # Puncturevine leaves are opposite.
    pp.petiole.petioles_per_internode = 2
    pp.petiole.pitch.uniformDistribution(60, 80)
    pp.petiole.radius = 0.0005
    pp.petiole.length = 0.03
    pp.petiole.taper = 0
    pp.petiole.curvature = 0
    pp.petiole.color = pp.internode.color
    pp.petiole.length_segments = 1

    pp.leaf.leaves_per_petiole = 10
    pp.leaf.pitch.uniformDistribution(0, 40)
    pp.leaf.yaw = 30
    pp.leaf.roll.uniformDistribution(-5, 5)
    pp.leaf.prototype_scale = 0.012
    pp.leaf.leaflet_offset = 0.18
    pp.leaf.leaflet_scale = 1
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length = 0.001
    pp.peduncle.color = pp.internode.color

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = f32(-90)
    pp.inflorescence.flower_prototype_function = PuncturevineFlowerPrototype
    pp.inflorescence.flower_prototype_scale = 0.01

    # ---- Shoot Parameters ---- //
    primary = ShootParameters(gen)
    primary.phytomer_parameters = pp
    primary.vegetative_bud_break_probability_min = 0.1
    primary.vegetative_bud_break_probability_decay_rate = f32(1)
    primary.vegetative_bud_break_time = 10
    primary.base_roll = 90
    primary.phyllochron_min = 1
    primary.elongation_rate_max = 0.2
    primary.girth_area_factor = f32(0)
    primary.internode_length_max = 0.02
    primary.internode_length_decay_rate = 0
    primary.insertion_angle_tip.uniformDistribution(75, 85)
    primary.insertion_angle_decay_rate = 0
    primary.flowers_require_dormancy = False
    primary.growth_requires_dormancy = False
    primary.flower_bud_break_probability = 0.2
    primary.determinate_shoot_growth = False
    primary.max_nodes = 15
    primary.gravitropic_curvature = 25
    primary.tortuosity = 0
    primary.defineChildShootTypes(["secondary_puncturevine"], [1.0])

    base = primary.copy()
    base.phytomer_parameters = pp
    base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5 - 10, 137.5 + 10)
    base.phytomer_parameters.petiole.petioles_per_internode = 0
    base.phytomer_parameters.internode.pitch = 0
    base.phytomer_parameters.petiole.pitch = 0
    base.vegetative_bud_break_probability_min = 1
    base.vegetative_bud_break_time = 2
    base.phyllochron_min = 2
    base.elongation_rate_max = 0.15
    base.gravitropic_curvature = 0
    base.internode_length_max = 0.01
    base.internode_length_decay_rate = 0
    base.insertion_angle_tip = 90
    base.insertion_angle_decay_rate = 0
    base.flowers_require_dormancy = False
    base.growth_requires_dormancy = False
    base.flower_bud_break_probability = 0.0
    base.max_nodes.uniformDistribution(3, 5)
    base.defineChildShootTypes(["primary_puncturevine"], [1.0])

    children = primary.copy()
    children.base_roll = 0

    pa.defineShootType("base_puncturevine", base)
    pa.defineShootType("primary_puncturevine", primary)
    pa.defineShootType("secondary_puncturevine", children)


@ports("PlantArchitecture::buildPuncturevinePlant")
def buildPuncturevinePlant(pa, base_position):
    """PL:5211-5227."""
    if not pa.shoot_types:
        initializePuncturevineShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    pa.addBaseStemShoot(plantID, 3, make_AxisRotation(0, f32(0), f32(0)), 0.001, 0.001, 1, 1, 0, "base_puncturevine")
    pa.breakPlantDormancy(plantID)
    pa.plant_instances[plantID].max_age = f32(45)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, 14, -1, -1, 1000)
    return plantID


@PuncturevineFlowerPrototype.implement
@ports("PuncturevineFlowerPrototype")
def _puncturevine_flower(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:2269-2273: ``loadOBJ(file, (0,0,0), 0, nullrotation, RGB::black, "ZUP", true)``."""
    return load_obj_black(context, "assets/obj/PuncturevineFlower.obj")


__all__ = ["initializePuncturevineShoots", "buildPuncturevinePlant", "PuncturevineFlowerPrototype"]
