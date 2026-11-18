# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:3336-3405 and src/Assets.cpp:1686-1690
# (Copyright (C) 2016-2026 Brian Bailey).
"""Cheeseweed (``initializeCheeseweedShoots``, ``buildCheeseweedPlant``), W5-C.

Line for line with the C++. One ``base`` shoot type: a rosette of 8 nodes on a 1.5 mm internode, every
petiole a 5-segment curved tube carrying one OBJ leaf (``CheeseweedLeaf.obj`` through ``OBJ_model_file`` of the
``GenericLeafPrototype``). No flowers (``flower_bud_break_probability = 0``).

``CheeseweedLeafPrototype`` (AS:1686) is declared in Assets.cpp but the library entry does not use it (it sets
``OBJ_model_file`` instead); it is ported for completeness and not oracle-tested.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from .bean import load_obj_silent
from .strawberry import named_hook

f32 = np.float32

CheeseweedLeafPrototype = named_hook("CheeseweedLeafPrototype")         # AS:1686 (unused by the library)


@ports("PlantArchitecture::initializeCheeseweedShoots")
def initializeCheeseweedShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.OBJ_model_file = "CheeseweedLeaf.obj"

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)

    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(f32(127.5), 147.5)
    pp.internode.radius_initial = 0.0005
    pp.internode.color = make_RGBcolor(0.60, 0.65, 0.40)
    pp.internode.length_segments = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(45, 75)
    pp.petiole.radius = 0.0005
    pp.petiole.length.uniformDistribution(0.02, 0.06)
    pp.petiole.taper = 0
    pp.petiole.curvature = -300
    pp.petiole.length_segments = 5
    pp.petiole.color = pp.internode.color

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-30, 0)
    pp.leaf.yaw = 0
    pp.leaf.roll = 0
    pp.leaf.prototype_scale = 0.035
    pp.leaf.prototype = leaf_prototype

    # ---- Shoot Parameters ---- //
    base = ShootParameters(gen)
    base.phytomer_parameters = pp
    base.vegetative_bud_break_probability_min = 0.2
    base.vegetative_bud_break_time = 6
    base.phyllochron_min = 2
    base.elongation_rate_max = 0.1
    base.girth_area_factor = f32(10)
    base.gravitropic_curvature = 0
    base.internode_length_max = 0.0015
    base.internode_length_decay_rate = 0
    base.flowers_require_dormancy = False
    base.growth_requires_dormancy = False
    base.flower_bud_break_probability = 0.
    base.max_nodes = 8

    pa.defineShootType("base", base)


@ports("PlantArchitecture::buildCheeseweedPlant")
def buildCheeseweedPlant(pa, base_position):
    """PL:3389-3405."""
    if not pa.shoot_types:
        initializeCheeseweedShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    pa.addBaseStemShoot(plantID, 1, make_AxisRotation(0, f32(0), f32(0)), 0.0001, 0.0025, 0.1, 0.1, 0, "base")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000)
    pa.plant_instances[plantID].max_age = f32(40)
    return plantID


@CheeseweedLeafPrototype.implement
@ports("CheeseweedLeafPrototype")
def _cheeseweed_leaf(context, prototype_parameters, compound_leaf_index) -> int:
    """AS:1686-1690 (not used by the library entry)."""
    return load_obj_silent(context, "assets/obj/CheeseweedLeaf.obj")


__all__ = ["initializeCheeseweedShoots", "buildCheeseweedPlant", "CheeseweedLeafPrototype"]
