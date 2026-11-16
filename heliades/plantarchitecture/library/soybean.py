# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:5745-5898 and src/Assets.cpp:4031-4066
# (Copyright (C) 2016-2026 Brian Bailey).
"""Soybean (``PlantArchitecture::initializeSoybeanShoots`` / ``buildSoybeanPlant``) and its Assets.cpp functions.

Line for line with the C++, literals with their C++ types (see ``tomato.py``).

RNG at load: ``LeafPrototype(gen)`` (1), ``PhytomerParameters(gen)`` (2, 3), ``trifoliate.leaf.prototype = lp``
(4), the copied unifoliate phytomer parameters (no draw), ``unifoliate.leaf.prototype = lp`` (5). Both shoot types
use the same ``SoybeanLeaf.png`` blade, but with different identifiers, so they have separate leaf caches.

The unifoliate type copies the trifoliate *phytomer* parameters before the creation function is set on the
trifoliate *shoot* parameters, and ``ShootParameters::phytomer_parameters = phytomer_parameters_unifoliate`` is
the implicit ``PhytomerParameters::operator=``, which copies that null pointer: the unifoliate has no creation
function, as for the bean (where it is also set to ``nullptr`` explicitly).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_RGBcolor
from .bean import build_legume, load_obj_black

f32 = np.float32

SoybeanFlowerPrototype = hooks.Hook("SoybeanFlowerPrototype")                    # AS:4037
SoybeanFruitPrototype = hooks.Hook("SoybeanFruitPrototype")                      # AS:4031
SoybeanPhytomerCreationFunction = hooks.Hook("SoybeanPhytomerCreationFunction")  # AS:4048


@ports("PlantArchitecture::initializeSoybeanShoots")
def initializeSoybeanShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    lp = LeafPrototype(gen)                                                      # identifier draw 1
    lp.leaf_texture_file[0] = "SoybeanLeaf.png"
    lp.leaf_aspect_ratio = f32(1)
    lp.midrib_fold_fraction = f32(0.1)
    lp.longitudinal_curvature.uniformDistribution(0.1, 0.2)
    lp.lateral_curvature = 0.45
    lp.subdivisions = 8
    lp.unique_prototypes = 5
    lp.build_petiolule = True
    lp.petiolule_length = {0: f32(0.025)}

    pt = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pt.internode.pitch = 20
    pt.internode.phyllotactic_angle.uniformDistribution(145, 215)
    pt.internode.radius_initial = 0.002
    pt.internode.max_floral_buds_per_petiole = 1
    pt.internode.max_vegetative_buds_per_petiole = 1
    pt.internode.color = make_RGBcolor(0.2, 0.25, 0.05)
    pt.internode.length_segments = 5

    pt.petiole.petioles_per_internode = 1
    pt.petiole.pitch.uniformDistribution(15, 40)
    pt.petiole.radius = 0.002
    pt.petiole.length.uniformDistribution(0.12, 0.16)
    pt.petiole.taper = 0.25
    pt.petiole.curvature.uniformDistribution(-250, 50)
    pt.petiole.color = pt.internode.color
    pt.petiole.length_segments = 5
    pt.petiole.radial_subdivisions = 6

    pt.leaf.leaves_per_petiole = 3
    pt.leaf.pitch.uniformDistribution(-30, 10)
    pt.leaf.yaw = 10
    pt.leaf.roll.uniformDistribution(-25, 5)
    pt.leaf.leaflet_offset = 0.5
    pt.leaf.leaflet_scale = 0.9
    pt.leaf.prototype_scale.uniformDistribution(0.1, 0.14)
    pt.leaf.prototype = lp                                                       # identifier draw 4

    pt.peduncle.length = 0.01
    pt.peduncle.radius = 0.0005
    pt.peduncle.pitch.uniformDistribution(0, 40)
    pt.peduncle.roll = 90
    pt.peduncle.curvature.uniformDistribution(-500, 500)
    pt.peduncle.color = pt.internode.color
    pt.peduncle.length_segments = 1
    pt.peduncle.radial_subdivisions = 6

    pt.inflorescence.flowers_per_peduncle.uniformDistribution(1, 4)
    pt.inflorescence.flower_offset = 0.2
    pt.inflorescence.pitch.uniformDistribution(50, 70)
    pt.inflorescence.roll.uniformDistribution(-20, 20)
    pt.inflorescence.flower_prototype_scale = 0.015
    pt.inflorescence.flower_prototype_function = SoybeanFlowerPrototype
    pt.inflorescence.fruit_prototype_scale.uniformDistribution(0.1, 0.12)
    pt.inflorescence.fruit_prototype_function = SoybeanFruitPrototype
    pt.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8, 1.0)

    pu = pt.copy()                                                               # implicit copy ctor: no draw
    pu.internode.pitch = 0
    pu.internode.max_vegetative_buds_per_petiole = 0
    pu.internode.max_floral_buds_per_petiole = 0
    pu.petiole.petioles_per_internode = 2
    pu.petiole.length = 0.01
    pu.petiole.radius = 0.001
    pu.petiole.pitch.uniformDistribution(60, 80)
    pu.leaf.leaves_per_petiole = 1
    pu.leaf.prototype_scale = 0.02
    pu.leaf.pitch.uniformDistribution(-10, 10)
    pu.leaf.prototype = lp                                                       # identifier draw 5

    # ---- Shoot Parameters ---- //
    st = ShootParameters(gen)
    st.phytomer_parameters = pt
    st.phytomer_parameters.phytomer_creation_function = SoybeanPhytomerCreationFunction

    st.max_nodes.uniformDistribution(16, 19)
    st.insertion_angle_tip.uniformDistribution(20, 30)
    st.internode_length_max = 0.045
    st.base_roll = 90
    st.base_yaw.uniformDistribution(-20, 20)
    st.gravitropic_curvature = 400

    st.phyllochron_min = 3.5
    st.elongation_rate_max = 0.1
    st.girth_area_factor = f32(2)
    st.vegetative_bud_break_time = 15
    st.vegetative_bud_break_probability_min = 0.5
    st.vegetative_bud_break_probability_decay_rate = 0
    st.flower_bud_break_probability.uniformDistribution(0.8, 1.0)
    st.fruit_set_probability = 0.4
    st.determinate_shoot_growth = False

    st.defineChildShootTypes(["branch"], [1.0])

    sb = st.copy()
    sb.max_nodes.uniformDistribution(3, 7)
    sb.internode_length_max = 0.035
    sb.vegetative_bud_break_probability_min = 0.05
    sb.defineChildShootTypes(["branch"], [1.0])

    su = st.copy()
    su.phytomer_parameters = pu                                                  # copies the null creation function
    su.max_nodes = 1
    su.flower_bud_break_probability = 0
    su.insertion_angle_tip = 0
    su.insertion_angle_decay_rate = 0
    su.vegetative_bud_break_time = 8
    su.defineChildShootTypes(["trifoliate"], [1.0])

    pa.defineShootType("unifoliate", su)
    pa.defineShootType("trifoliate", st)
    pa.defineShootType("branch", sb)


@ports("PlantArchitecture::buildSoybeanPlant")
def buildSoybeanPlant(pa, base_position):
    return build_legume(pa, base_position, initializeSoybeanShoots, 0.04)


@SoybeanPhytomerCreationFunction.implement
@ports("SoybeanPhytomerCreationFunction")
def _soybean_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:4048-4066 (double ``fmin``, as for the bean)."""
    if phytomer.rank > 0 or shoot_node_index >= 5:
        phytomer.setVegetativeBudState(BUD_DEAD)
    if phytomer.rank == 0 and shoot_node_index < 5:
        phytomer.setFloralBudState(BUD_DEAD)
    age = float(f32(plant_age))
    leaf_scale = f32(min(1.0, 0.2 + 0.8 * age / 15.0))
    phytomer.scaleLeafPrototypeScale(leaf_scale)
    inode_scale = f32(min(1.0, 0.1 + 0.9 * age / 15.0))
    phytomer.scaleInternodeMaxLength(inode_scale)


@SoybeanFruitPrototype.implement
@ports("SoybeanFruitPrototype")
def _soybean_fruit(context, subdivisions, build_parameters=None) -> int:
    """AS:4031-4035."""
    return load_obj_black(context, "assets/obj/SoybeanPod.obj")


@SoybeanFlowerPrototype.implement
@ports("SoybeanFlowerPrototype")
def _soybean_flower(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:4037-4046 (the closed flower is the bean's mesh)."""
    if flower_is_open:
        return load_obj_black(context, "assets/obj/SoybeanFlower_open_white.obj")
    return load_obj_black(context, "assets/obj/BeanFlower_closed_white.obj")


__all__ = ["initializeSoybeanShoots", "buildSoybeanPlant", "SoybeanFlowerPrototype", "SoybeanFruitPrototype",
           "SoybeanPhytomerCreationFunction"]
