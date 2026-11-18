# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:3409-3566 and src/Assets.cpp:1692-1757
# (Copyright (C) 2016-2026 Brian Bailey).
"""Cowpea (``initializeCowpeaShoots`` / ``buildCowpeaPlant``) and its Assets.cpp functions.

Line for line with the C++, literals with their C++ types (see ``tomato.py``). The bean's structure (``bean.py``): a
one-node ``unifoliate`` base stem with two petioles, the ``trifoliate`` main stem appended to it, ``GenericLeafPrototype``
blades with one texture per leaflet, ``CowpeaPhytomerCreationFunction`` (floral buds on the upper half of every axis,
branches from the basal nodes of ranks 0-1), OBJ flowers and pods.

The library registers the cowpea with a leaf-inclination Beta (1.398, 1.574), so ``buildPlantInstanceFromLibrary``
switches on leaf inclination tracking (``leaf_angle_tracking.py``, lambda 180 degrees) before the plant grows.

RNG at load: 5 identifier draws, as for the bean (the unifoliate prototype is a copy, then assigned). Build: the bean's
3 base-rotation ``randu`` (right to left).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from .strawberry import named_hook
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, make_RGBcolor, ShootParameters
from .bean import base_rotation_draws, load_obj_black

f32 = np.float32

CowpeaFlowerPrototype = named_hook("CowpeaFlowerPrototype")                     # AS:1718
CowpeaFruitPrototype = named_hook("CowpeaFruitPrototype")                       # AS:1712
CowpeaPhytomerCreationFunction = named_hook("CowpeaPhytomerCreationFunction")   # AS:1729
CowpeaLeafPrototype_unifoliate_OBJ = named_hook("CowpeaLeafPrototype_unifoliate_OBJ")   # AS:1692 (unused by the library)
CowpeaLeafPrototype_trifoliate_OBJ = named_hook("CowpeaLeafPrototype_trifoliate_OBJ")   # AS:1699 (unused by the library)


@ports("PlantArchitecture::initializeCowpeaShoots")
def initializeCowpeaShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    lp_tri = LeafPrototype(gen)                                                  # identifier draw 1
    lp_tri.leaf_texture_file[0] = "CowpeaLeaf_tip_centered.png"
    lp_tri.leaf_texture_file[-1] = "CowpeaLeaf_right_centered.png"
    lp_tri.leaf_texture_file[1] = "CowpeaLeaf_left_centered.png"
    lp_tri.leaf_aspect_ratio = f32(0.7)
    lp_tri.midrib_fold_fraction = 0.2
    lp_tri.longitudinal_curvature.uniformDistribution(f32(-0.3), f32(-0.1))
    lp_tri.lateral_curvature = f32(-0.4)
    lp_tri.subdivisions = 6
    lp_tri.unique_prototypes = 5
    lp_tri.build_petiolule = True
    lp_tri.petiolule_length = {0: f32(0.038)}

    lp_uni = lp_tri.copy()                                                       # copy ctor: duplicate(), no draw
    lp_uni.leaf_texture_file.clear()
    lp_uni.leaf_texture_file[0] = "CowpeaLeaf_unifoliate_centered.png"

    # ---- Phytomer Parameters ---- //
    pt = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pt.internode.pitch = 20
    pt.internode.phyllotactic_angle.uniformDistribution(145, 215)
    pt.internode.radius_initial = 0.0015
    pt.internode.max_floral_buds_per_petiole = 1
    pt.internode.max_vegetative_buds_per_petiole = 1
    pt.internode.color = make_RGBcolor(0.15, 0.2, 0.1)
    pt.internode.length_segments = 2

    pt.petiole.petioles_per_internode = 1
    pt.petiole.pitch.uniformDistribution(45, 60)
    pt.petiole.radius = 0.0018
    pt.petiole.length.uniformDistribution(0.06, 0.08)
    pt.petiole.taper = 0.25
    pt.petiole.curvature.uniformDistribution(-200, -50)
    pt.petiole.color = make_RGBcolor(0.2, 0.25, 0.06)
    pt.petiole.length_segments = 5
    pt.petiole.radial_subdivisions = 6

    pt.leaf.leaves_per_petiole = 3
    pt.leaf.pitch.normalDistribution(45, 20)
    pt.leaf.yaw = 10
    pt.leaf.roll = -15
    pt.leaf.leaflet_offset = 0.4
    pt.leaf.leaflet_scale = 0.9
    pt.leaf.prototype_scale.uniformDistribution(0.08, 0.105)
    pt.leaf.prototype = lp_tri                                                   # identifier draw 4

    pt.peduncle.length.uniformDistribution(0.3, 0.4)
    pt.peduncle.radius = 0.00225
    pt.peduncle.pitch.uniformDistribution(0, 30)
    pt.peduncle.roll = 90
    pt.peduncle.curvature.uniformDistribution(125, 200)
    pt.peduncle.color = make_RGBcolor(0.17, 0.213, 0.051)
    pt.peduncle.length_segments = 6
    pt.peduncle.radial_subdivisions = 6

    pt.inflorescence.flowers_per_peduncle.uniformDistribution(1, 3)
    pt.inflorescence.flower_offset = 0.05
    pt.inflorescence.pitch.uniformDistribution(40, 60)
    pt.inflorescence.roll.uniformDistribution(-20, 20)
    pt.inflorescence.flower_prototype_scale = 0.03
    pt.inflorescence.flower_prototype_function = CowpeaFlowerPrototype
    pt.inflorescence.fruit_prototype_scale.uniformDistribution(0.09, 0.1)
    pt.inflorescence.fruit_prototype_function = CowpeaFruitPrototype
    pt.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.5, 0.7)

    pu = pt.copy()                                                               # implicit copy ctor: no draw
    pu.internode.pitch = 0
    pu.internode.max_vegetative_buds_per_petiole = 1
    pu.internode.max_floral_buds_per_petiole = 0
    pu.petiole.petioles_per_internode = 2
    pu.petiole.length = 0.0001
    pu.petiole.radius = 0.0004
    pu.petiole.pitch.uniformDistribution(60, 80)
    pu.leaf.leaves_per_petiole = 1
    pu.leaf.prototype_scale = 0.02
    pu.leaf.pitch.uniformDistribution(-10, 10)
    pu.leaf.prototype = lp_uni                                                   # identifier draw 5

    # ---- Shoot Parameters ---- //
    st = ShootParameters(gen)
    st.phytomer_parameters = pt
    st.phytomer_parameters.phytomer_creation_function = CowpeaPhytomerCreationFunction

    st.max_nodes = 13
    st.insertion_angle_tip.uniformDistribution(40, 60)
    st.internode_length_max = 0.04
    st.base_roll = 90
    st.base_yaw.uniformDistribution(-20, 20)
    st.gravitropic_curvature = 200

    st.phyllochron_min = 2
    st.elongation_rate_max = 0.1
    st.girth_area_factor = f32(1.5)
    st.vegetative_bud_break_time = 10
    st.vegetative_bud_break_probability_min = 0.2
    st.vegetative_bud_break_probability_decay_rate = -0.4
    st.flower_bud_break_probability.uniformDistribution(0.3, 0.4)
    st.fruit_set_probability = 0.4

    st.defineChildShootTypes(["trifoliate"], [1.0])

    su = st.copy()                                                               # copy ctor: no draw
    su.phytomer_parameters = pu
    su.max_nodes = 1
    su.girth_area_factor = f32(1)
    su.vegetative_bud_break_probability_min = 1
    su.flower_bud_break_probability = 0
    su.insertion_angle_tip = 40
    su.insertion_angle_decay_rate = 0
    su.vegetative_bud_break_time = 8
    su.defineChildShootTypes(["trifoliate"], [1.0])

    pa.defineShootType("unifoliate", su)
    pa.defineShootType("trifoliate", st)


@ports("PlantArchitecture::buildCowpeaPlant")
def buildCowpeaPlant(pa, base_position):
    """PL:3545-3566 (``bean.build_legume`` with the cowpea's 3 cm unifoliate internode)."""
    from ..params import make_AxisRotation
    if not pa.shoot_types:
        initializeCowpeaShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    base_rotation = base_rotation_draws(pa)
    uID_unifoliate = pa.addBaseStemShoot(plantID, 1, base_rotation, 0.0005, 0.03, 0.01, 0.01, 0, "unifoliate")
    tri = pa.shoot_types["trifoliate"]
    length_max = tri.internode_length_max.val()                     # arguments right to left (both constant)
    radius = tri.phytomer_parameters.internode.radius_initial.val()
    pa.appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, f32(float(f32(0.5)) * np.pi)), radius,
                   length_max, 0.1, 0.1, 0, "trifoliate")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


@CowpeaPhytomerCreationFunction.implement
@ports("CowpeaPhytomerCreationFunction")
def _cowpea_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:1729-1757. ``std::max(5u, shoot_max_nodes / 2)`` is unsigned; the scale ``fmin`` chains are double."""
    first_floral_node = max(5, int(shoot_max_nodes) // 2)
    can_branch = phytomer.rank <= 1
    if shoot_node_index >= first_floral_node:
        phytomer.setVegetativeBudState(BUD_DEAD)
    elif can_branch:
        phytomer.setFloralBudState(BUD_DEAD)
    else:
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.setFloralBudState(BUD_DEAD)
    age = float(f32(plant_age))
    phytomer.scaleLeafPrototypeScale(f32(min(1.0, 0.6 + 0.4 * age / 8.0)))
    if phytomer.rank == 0:
        phytomer.scaleInternodeMaxLength(f32(min(1.0, 0.2 + 0.8 * age / 10.0)))


# ---- OBJ prototypes (AS:1692-1727) ---------------------------------------------------------------------

@CowpeaLeafPrototype_unifoliate_OBJ.implement
@ports("CowpeaLeafPrototype_unifoliate_OBJ")
def _cowpea_leaf_unifoliate_obj(context, prototype_parameters, compound_leaf_index) -> int:
    return load_obj_black(context, "assets/obj/CowpeaLeaf_unifoliate.obj")


@CowpeaLeafPrototype_trifoliate_OBJ.implement
@ports("CowpeaLeafPrototype_trifoliate_OBJ")
def _cowpea_leaf_trifoliate_obj(context, prototype_parameters, compound_leaf_index) -> int:
    if compound_leaf_index < 0:
        return load_obj_black(context, "assets/obj/CowpeaLeaf_right_highres.obj")
    if compound_leaf_index == 0:
        return load_obj_black(context, "assets/obj/CowpeaLeaf_tip_highres.obj")
    return load_obj_black(context, "assets/obj/CowpeaLeaf_left_highres.obj")


@CowpeaFruitPrototype.implement
@ports("CowpeaFruitPrototype")
def _cowpea_fruit(context, subdivisions, build_parameters=None) -> int:
    """AS:1712-1716: ``loadOBJ(CowpeaPod.obj, (0,0,0), 0.75, nullrotation, RGB::black, "ZUP", true)``."""
    return load_obj_black(context, "assets/obj/CowpeaPod.obj", 0.75)


@CowpeaFlowerPrototype.implement
@ports("CowpeaFlowerPrototype")
def _cowpea_flower(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:1718-1727."""
    if flower_is_open:
        return load_obj_black(context, "assets/obj/CowpeaFlower_open_yellow.obj")
    return load_obj_black(context, "assets/obj/CowpeaFlower_closed_yellow.obj")


__all__ = ["initializeCowpeaShoots", "buildCowpeaPlant", "CowpeaFlowerPrototype", "CowpeaFruitPrototype",
           "CowpeaPhytomerCreationFunction"]
