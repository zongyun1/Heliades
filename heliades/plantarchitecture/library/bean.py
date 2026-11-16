# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:2911-3070 and src/Assets.cpp:1578-1641
# (Copyright (C) 2016-2026 Brian Bailey).
"""Common bean (``PlantArchitecture::initializeBeanShoots`` / ``buildBeanPlant``) and its Assets.cpp functions.

Line for line with the C++, literals with their C++ types (see ``tomato.py``).

RNG at load (all from the Context stream; 2, 4, ... engine calls each):

1. ``LeafPrototype leaf_prototype_trifoliate(gen)``             -> ctor ``sampleIdentifier``
2. ``LeafPrototype leaf_prototype_unifoliate = trifoliate``      -> copy ctor = ``duplicate()``: no draw
3. ``PhytomerParameters phytomer_parameters_trifoliate(gen)``    -> 2 (as for the tomato)
4. ``phytomer_parameters_trifoliate.leaf.prototype = ...``       -> operator= (kept by the trifoliate type)
5. ``PhytomerParameters phytomer_parameters_unifoliate = ...``   -> implicit copy: no draw
6. ``phytomer_parameters_unifoliate.leaf.prototype = ...``       -> operator= (kept by the unifoliate type)

So 5 identifier draws. The bean leaves are ``GenericLeafPrototype`` blades with one texture per leaflet
(``leaf_texture_file[-1/0/1]``); the OBJ-leaf functions ``BeanLeafPrototype_*_OBJ`` are defined in Assets.cpp but
not used by the library entry, and are ported here only for completeness.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor

f32 = np.float32

# Function-pointer slots (identity-compared, named like the C++ symbols; see hooks.py).
BeanFlowerPrototype = hooks.Hook("BeanFlowerPrototype")                  # AS:1605
BeanFruitPrototype = hooks.Hook("BeanFruitPrototype")                    # AS:1599
BeanPhytomerCreationFunction = hooks.Hook("BeanPhytomerCreationFunction")  # AS:1616
BeanLeafPrototype_unifoliate_OBJ = hooks.Hook("BeanLeafPrototype_unifoliate_OBJ")   # AS:1578 (unused by the library)
BeanLeafPrototype_trifoliate_OBJ = hooks.Hook("BeanLeafPrototype_trifoliate_OBJ")   # AS:1586 (unused by the library)


@ports("PlantArchitecture::initializeBeanShoots")
def initializeBeanShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    lp_tri = LeafPrototype(gen)                                                  # identifier draw 1
    lp_tri.leaf_texture_file[0] = "BeanLeaf_tip.png"
    lp_tri.leaf_texture_file[-1] = "BeanLeaf_right_centered.png"
    lp_tri.leaf_texture_file[1] = "BeanLeaf_left_centered.png"
    lp_tri.leaf_aspect_ratio = f32(1)
    lp_tri.midrib_fold_fraction = 0.2
    lp_tri.longitudinal_curvature.uniformDistribution(f32(-0.3), f32(-0.2))
    lp_tri.lateral_curvature = f32(-1)
    lp_tri.subdivisions = 6
    lp_tri.unique_prototypes = 5
    lp_tri.build_petiolule = True
    lp_tri.petiolule_length = {0: f32(0.02)}

    lp_uni = lp_tri.copy()                                                       # copy ctor: duplicate(), no draw
    lp_uni.leaf_texture_file.clear()
    lp_uni.leaf_texture_file[0] = "BeanLeaf_unifoliate_centered.png"
    lp_uni.unique_prototypes = 2

    # ---- Phytomer Parameters ---- //
    pt = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pt.internode.pitch = 20
    pt.internode.phyllotactic_angle.uniformDistribution(145, 215)
    pt.internode.radius_initial = 0.001
    pt.internode.max_floral_buds_per_petiole = 1
    pt.internode.max_vegetative_buds_per_petiole = 1
    pt.internode.color = make_RGBcolor(0.2, 0.25, 0.05)
    pt.internode.length_segments = 2

    pt.petiole.petioles_per_internode = 1
    pt.petiole.pitch.uniformDistribution(20, 50)
    pt.petiole.radius = 0.0015
    pt.petiole.length.uniformDistribution(0.1, 0.14)
    pt.petiole.taper = 0.
    pt.petiole.curvature.uniformDistribution(-100, 200)
    pt.petiole.color = make_RGBcolor(0.28, 0.35, 0.07)
    pt.petiole.length_segments = 5
    pt.petiole.radial_subdivisions = 6

    pt.leaf.leaves_per_petiole = 3
    pt.leaf.pitch.normalDistribution(0, 20)
    pt.leaf.yaw = 10
    pt.leaf.roll = -15
    pt.leaf.leaflet_offset = 0.3
    pt.leaf.leaflet_scale = 0.9
    pt.leaf.prototype_scale.uniformDistribution(0.09, 0.11)
    pt.leaf.prototype = lp_tri                                                   # identifier draw 4

    pt.peduncle.length = 0.04
    pt.peduncle.radius = 0.00075
    pt.peduncle.pitch.uniformDistribution(0, 40)
    pt.peduncle.roll = 90
    pt.peduncle.curvature.uniformDistribution(-500, 500)
    pt.peduncle.color = pt.petiole.color
    pt.peduncle.length_segments = 1
    pt.peduncle.radial_subdivisions = 6

    pt.inflorescence.flowers_per_peduncle.uniformDistribution(1, 4)
    pt.inflorescence.flower_offset = 0.2
    pt.inflorescence.pitch.uniformDistribution(50, 70)
    pt.inflorescence.roll.uniformDistribution(-20, 20)
    pt.inflorescence.flower_prototype_scale = 0.03
    pt.inflorescence.flower_prototype_function = BeanFlowerPrototype
    pt.inflorescence.fruit_prototype_scale.uniformDistribution(0.15, 0.2)
    pt.inflorescence.fruit_prototype_function = BeanFruitPrototype
    pt.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0.8, 1.0)

    pu = pt.copy()                                                               # implicit copy ctor: no draw
    pu.internode.pitch = 0
    pu.internode.max_vegetative_buds_per_petiole = 1
    pu.internode.max_floral_buds_per_petiole = 0
    pu.petiole.petioles_per_internode = 2
    pu.petiole.length = 0.0001
    pu.petiole.radius = 0.0004
    pu.petiole.pitch.uniformDistribution(50, 70)
    pu.leaf.leaves_per_petiole = 1
    pu.leaf.prototype_scale = 0.04
    pu.leaf.pitch.uniformDistribution(-10, 10)
    pu.leaf.prototype = lp_uni                                                   # identifier draw 5

    # ---- Shoot Parameters ---- //
    st = ShootParameters(gen)
    st.phytomer_parameters = pt
    st.phytomer_parameters.phytomer_creation_function = BeanPhytomerCreationFunction

    st.max_nodes.uniformDistribution(4, 7)
    st.insertion_angle_tip.uniformDistribution(40, 60)
    st.internode_length_max = 0.025
    st.base_roll = 90
    st.base_yaw.uniformDistribution(-20, 20)
    st.gravitropic_curvature = 200

    st.phyllochron_min = 3
    st.elongation_rate_max = 0.1
    st.girth_area_factor = f32(1.5)
    st.vegetative_bud_break_time = 15
    st.vegetative_bud_break_probability_min = 0.3
    st.vegetative_bud_break_probability_decay_rate = -0.4
    st.flower_bud_break_probability = 0.6
    st.fruit_set_probability = 0.4

    st.defineChildShootTypes(["trifoliate"], [1.0])

    su = st.copy()                                                               # copy ctor: no draw
    su.phytomer_parameters = pu
    su.phytomer_parameters.phytomer_creation_function = None
    su.max_nodes = 1
    su.girth_area_factor = f32(1)
    su.vegetative_bud_break_probability_min = 1.0
    su.flower_bud_break_probability = 0
    su.insertion_angle_tip = 50
    su.insertion_angle_decay_rate = 0
    su.vegetative_bud_break_time = 8
    su.defineChildShootTypes(["trifoliate"], [1.0])

    pa.defineShootType("unifoliate", su)
    pa.defineShootType("trifoliate", st)


def base_rotation_draws(pa):
    """``make_AxisRotation(randu(0.f, 0.05f*M_PI), randu(0.f, 2.f*M_PI), randu(0.f, 2.f*M_PI))`` (PL:3058, 5886).

    GCC evaluates the arguments right to left: roll, then yaw, then pitch. ``0.05f * M_PI`` and ``2.f * M_PI``
    are double, narrowed to float by ``randu(float, float)``."""
    hi = f32(2.0 * np.pi)
    roll = pa.context_ptr.randu(f32(0), hi)
    yaw = pa.context_ptr.randu(f32(0), hi)
    pitch = pa.context_ptr.randu(f32(0), f32(float(f32(0.05)) * np.pi))
    return make_AxisRotation(pitch, yaw, roll)


def build_legume(pa, base_position, initialize, unifoliate_length):
    """The shared body of ``buildBeanPlant`` (PL:3049-3069) and ``buildSoybeanPlant`` (PL:5877-5897): a
    one-node unifoliate base stem, the trifoliate main stem appended to it."""
    if not pa.shoot_types:
        initialize(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    base_rotation = base_rotation_draws(pa)
    uID_unifoliate = pa.addBaseStemShoot(plantID, 1, base_rotation, 0.0005, unifoliate_length, 0.01, 0.01, 0,
                                         "unifoliate")
    tri = pa.shoot_types["trifoliate"]
    # arguments right to left: internode_length_max.val() before radius_initial.val() (both constant here)
    length_max = tri.internode_length_max.val()
    radius = tri.phytomer_parameters.internode.radius_initial.val()
    pa.appendShoot(plantID, uID_unifoliate, 1, make_AxisRotation(0, 0, f32(float(f32(0.5)) * np.pi)), radius,
                   length_max, 0.1, 0.1, 0, "trifoliate")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


@ports("PlantArchitecture::buildBeanPlant")
def buildBeanPlant(pa, base_position):
    return build_legume(pa, base_position, initializeBeanShoots, 0.03)


@BeanPhytomerCreationFunction.implement
@ports("BeanPhytomerCreationFunction")
def _bean_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:1616-1637. ``fmin(1.f, 0.6 + 0.4 * plant_age / 8.f)`` is double (the literals 0.6 / 0.4 are double)."""
    if phytomer.rank > 0 or shoot_node_index >= 2:
        phytomer.setVegetativeBudState(BUD_DEAD)
    if phytomer.rank == 0 and shoot_node_index < 2:
        phytomer.setFloralBudState(BUD_DEAD)
    age = float(f32(plant_age))
    leaf_scale = f32(min(1.0, 0.6 + 0.4 * age / 8.0))
    phytomer.scaleLeafPrototypeScale(leaf_scale)
    if phytomer.rank == 0:
        inode_scale = f32(min(1.0, 0.2 + 0.8 * age / 10.0))
        phytomer.scaleInternodeMaxLength(inode_scale)


# ---- OBJ prototypes (Assets.cpp) ----------------------------------------------------------------------

def plugin_asset(asset_path: str) -> str:
    """``helios::resolvePluginAsset("plantarchitecture", asset_path)``, falling back to the pinned Helios tree."""
    from ...core.scene import resolve_plugin_asset, try_resolve_plugin_asset
    p = try_resolve_plugin_asset("plantarchitecture", asset_path)
    if p:
        return p
    from ..leaf_prototype import _try_plugin_asset
    p = _try_plugin_asset(asset_path)
    return p or resolve_plugin_asset("plantarchitecture", asset_path)


def load_obj_silent(context, asset_path: str) -> int:
    """``addPolymeshObject(loadOBJ(file, true))``: the 2-argument overload = origin 0, unscaled, no rotation,
    default colour RGB::blue, "ZUP"."""
    return context.add_polymesh_object(context.load_obj(plugin_asset(asset_path), silent=True))


def load_obj_black(context, asset_path: str, height=0.0) -> int:
    """``addPolymeshObject(loadOBJ(file, (0,0,0), height, nullrotation, RGB::black, "ZUP", true))``."""
    uu = context.load_obj(plugin_asset(asset_path), (0.0, 0.0, 0.0), height=float(height), rotation=(0.0, 0.0),
                          color=(0.0, 0.0, 0.0), upaxis="ZUP", silent=True)
    return context.add_polymesh_object(uu)


@BeanLeafPrototype_unifoliate_OBJ.implement
@ports("BeanLeafPrototype_unifoliate_OBJ")
def _bean_leaf_unifoliate_obj(context, prototype_parameters, compound_leaf_index) -> int:
    return load_obj_silent(context, "assets/obj/BeanLeaf_unifoliate.obj")


@BeanLeafPrototype_trifoliate_OBJ.implement
@ports("BeanLeafPrototype_trifoliate_OBJ")
def _bean_leaf_trifoliate_obj(context, prototype_parameters, compound_leaf_index) -> int:
    if compound_leaf_index == 0:
        return load_obj_silent(context, "assets/obj/BeanLeaf_tip.obj")
    if compound_leaf_index < 0:
        return load_obj_silent(context, "assets/obj/BeanLeaf_right.obj")
    return load_obj_silent(context, "assets/obj/BeanLeaf_left.obj")


@BeanFruitPrototype.implement
@ports("BeanFruitPrototype")
def _bean_fruit(context, subdivisions, build_parameters=None) -> int:
    """AS:1599-1603."""
    return load_obj_silent(context, "assets/obj/BeanPod.obj")


@BeanFlowerPrototype.implement
@ports("BeanFlowerPrototype")
def _bean_flower(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:1605-1614."""
    if flower_is_open:
        return load_obj_silent(context, "assets/obj/BeanFlower_open_white.obj")
    return load_obj_silent(context, "assets/obj/BeanFlower_closed_white.obj")


__all__ = ["initializeBeanShoots", "buildBeanPlant", "BeanFlowerPrototype", "BeanFruitPrototype",
           "BeanPhytomerCreationFunction", "build_legume", "base_rotation_draws"]
