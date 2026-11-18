# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:2800-2909 and src/Assets.cpp:1638-1642
# (Copyright (C) 2016-2026 Brian Bailey).
"""Field bindweed (``initializeBindweedShoots``, ``buildBindweedPlant``, ``BindweedFlowerPrototype``), W5-C.

Line for line with the C++, literals with their C++ types (see ``tomato.py``). A prostrate weed of three shoot
types: a short ``base_bindweed`` crown (no petioles), whose every bud breaks into a ``primary_bindweed`` runner,
which bears ``secondary_bindweed`` runners. The leaf is an OBJ model (``BindweedLeaf.obj``) set on the default
leaf prototype of ``PhytomerParameters`` (``leaf.prototype.OBJ_model_file``), so there is no ``LeafPrototype`` of
its own: three identifier draws at load (``PhytomerParameters(gen)``).

``BindweedFlowerPrototype`` is also the flower of ``groundcherryweed``.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from .bean import load_obj_silent
from .strawberry import named_hook

f32 = np.float32

BindweedFlowerPrototype = named_hook("BindweedFlowerPrototype")         # AS:1638


@ports("PlantArchitecture::initializeBindweedShoots")
def initializeBindweedShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)

    pp.internode.pitch.uniformDistribution(0, 15)
    pp.internode.phyllotactic_angle = f32(180)
    pp.internode.radius_initial = 0.0012
    pp.internode.color = make_RGBcolor(0.3, 0.38, 0.21)
    pp.internode.length_segments = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(80, 100)
    pp.petiole.radius = 0.001
    pp.petiole.length = 0.006
    pp.petiole.taper = 0
    pp.petiole.curvature = 0
    pp.petiole.color = pp.internode.color
    pp.petiole.length_segments = 1

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(5, 30)
    pp.leaf.yaw = 0
    pp.leaf.roll = 0
    pp.leaf.prototype_scale = 0.05
    pp.leaf.prototype.OBJ_model_file = "BindweedLeaf.obj"

    pp.peduncle.length = 0.01
    pp.peduncle.radius = 0.0005
    pp.peduncle.color = pp.internode.color

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = f32(-90)
    pp.inflorescence.flower_prototype_function = BindweedFlowerPrototype
    pp.inflorescence.flower_prototype_scale = 0.045

    # ---- Shoot Parameters ---- //
    primary = ShootParameters(gen)
    primary.phytomer_parameters = pp
    primary.vegetative_bud_break_probability_min = 0.1
    primary.vegetative_bud_break_probability_decay_rate = -1.
    primary.vegetative_bud_break_time = 10
    primary.base_roll = 90
    primary.phyllochron_min = 1
    primary.elongation_rate_max = 0.25
    primary.girth_area_factor = 0
    primary.internode_length_max = 0.03
    primary.internode_length_decay_rate = 0
    primary.insertion_angle_tip.uniformDistribution(70, 80)
    primary.flowers_require_dormancy = False
    primary.growth_requires_dormancy = False
    primary.flower_bud_break_probability = 0.2
    primary.determinate_shoot_growth = False
    primary.max_nodes = 15
    primary.gravitropic_curvature = 30
    primary.tortuosity = 0
    primary.defineChildShootTypes(["secondary_bindweed"], [1.0])

    base = primary.copy()
    base.phytomer_parameters = pp
    base.phytomer_parameters.internode.phyllotactic_angle.uniformDistribution(137.5 - 10, 137.5 + 10)
    base.phytomer_parameters.petiole.petioles_per_internode = 0
    base.phytomer_parameters.internode.pitch = 0
    base.phytomer_parameters.petiole.pitch = 0
    base.vegetative_bud_break_probability_min = 1.0
    base.vegetative_bud_break_time = 2
    base.phyllochron_min = 2
    base.elongation_rate_max = 0.15
    base.girth_area_factor = f32(0)
    base.gravitropic_curvature = 0
    base.internode_length_max = 0.01
    base.internode_length_decay_rate = 0
    base.insertion_angle_tip = 95
    base.insertion_angle_decay_rate = 0
    base.flowers_require_dormancy = False
    base.growth_requires_dormancy = False
    base.flower_bud_break_probability = 0.0
    base.max_nodes.uniformDistribution(3, 5)
    base.defineChildShootTypes(["primary_bindweed"], [1.0])

    children = primary.copy()
    children.base_roll = 0
    children.insertion_angle_tip.uniformDistribution(50, 80)

    pa.defineShootType("base_bindweed", base)
    pa.defineShootType("primary_bindweed", primary)
    pa.defineShootType("secondary_bindweed", children)


@ports("PlantArchitecture::buildBindweedPlant")
def buildBindweedPlant(pa, base_position):
    """PL:2891-2909."""
    if not pa.shoot_types:
        initializeBindweedShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    pa.addBaseStemShoot(plantID, 3, make_AxisRotation(0, f32(0), f32(0)), 0.001, 0.001, 1, 1, 0, "base_bindweed")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, 14, -1, -1, 1000)
    pa.plant_instances[plantID].max_age = f32(50)
    return plantID


@BindweedFlowerPrototype.implement
@ports("BindweedFlowerPrototype")
def _bindweed_flower(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:1638-1642: the 2-argument ``loadOBJ(file, true)`` (open and closed flowers share the mesh)."""
    return load_obj_silent(context, "assets/obj/BindweedFlower.obj")


__all__ = ["initializeBindweedShoots", "buildBindweedPlant", "BindweedFlowerPrototype"]
