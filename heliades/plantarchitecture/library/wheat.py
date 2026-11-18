# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:6542-6687 and src/Assets.cpp:4166-4260
# (Copyright (C) 2016-2026 Brian Bailey).
"""Wheat shoot parameters, builder, phytomer creation function and spike prototype (W5-A).

A grass like the maize (``library/maize.py``): sessile flexible strap leaves (``SorghumLeaf.png`` with its midrib class
map), 8 main-stem phytomers, the ear (``WheatSpike.obj``, height 0) as the terminal fruit. Tillers (shoot type
``tiller``) grow from the lowest main-stem axils when the build parameter ``tiller_count`` (0-4, default 0) asks for
them; ``WheatPhytomerCreationFunction`` kills every other bud, sets a tiller's leaf number from its position, builds
each blade at its fraction of the longest blade, runs the internodes from ligule to ligule (culm internode plus own
sheath less the sheath below), turns the flag leaf 33 degrees off the culm and shortens the peduncle by the part hidden
in the flag-leaf sheath.

RNG: load as the maize (identifier draws); build: ``make_AxisRotation(randu(0, 0.05 pi), randu(0, 2 pi),
randu(0, 2 pi))`` right to left (roll, yaw, pitch).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from .build_parameters import BuildParameterDeclaration, getBuildParameterIntegerValue, registerPlantModelBuildParameters
from .maize import setGrassFlagLeafAngle, setInternodeMaxLength
from .strawberry import named_hook, obj_prototype

f32 = np.float32

#: PlantArchitectureInternal.h: ``wheat_tiller_count``
wheat_tiller_count = BuildParameterDeclaration("tiller_count", 0.0, 4.0, True, "number of tillers of a wheat plant")


def register_build_parameters(pa):
    """PL:535."""
    registerPlantModelBuildParameters(pa, "wheat", [wheat_tiller_count])


@ports("PlantArchitecture::initializeWheatShoots")
def initializeWheatShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png"
    leaf_prototype.leaf_class_map_file[0] = "SorghumLeaf_classes.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.065)
    leaf_prototype.midrib_fold_fraction = f32(0.3)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.07, -0.04)
    leaf_prototype.longitudinal_curvature_exponent = f32(3)
    leaf_prototype.flexibility_taper = f32(150)
    leaf_prototype.flexibility_aging = f32(11)
    leaf_prototype.flexibility_aging_max = f32(5)
    leaf_prototype.lateral_curvature = -0.3
    leaf_prototype.petiole_roll = f32(0)
    leaf_prototype.wave_period = f32(0.1)
    leaf_prototype.wave_amplitude = f32(0.04)
    leaf_prototype.flexibility.uniformDistribution(0.8, 1.2)
    leaf_prototype.subdivisions = 20
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)
    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(175, 185)
    pp.internode.radius_initial = 0.001
    pp.internode.color = make_RGBcolor(0.27, 0.31, 0.16)
    pp.internode.length_segments = 1
    pp.internode.radial_subdivisions = 6
    pp.internode.max_floral_buds_per_petiole = 0
    pp.internode.max_vegetative_buds_per_petiole = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.length = 0

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-40, -20)
    pp.leaf.yaw = 0
    pp.leaf.roll = 0
    pp.leaf.prototype_scale = 0.24
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length = 0.32
    pp.peduncle.radius = 0.002
    pp.peduncle.color = pp.internode.color
    pp.peduncle.curvature = 0
    pp.peduncle.radial_subdivisions = 6

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.fruit_prototype_scale = 0.1
    pp.inflorescence.fruit_prototype_function = named_hook("WheatSpikePrototype")
    pp.inflorescence.inflorescence_maturity_period = 12

    pp.phytomer_creation_function = named_hook("WheatPhytomerCreationFunction")

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.vegetative_bud_break_probability_min = 1
    sp.vegetative_bud_break_time = 15
    sp.flower_bud_break_probability = 1
    sp.phyllochron_min = 6
    sp.elongation_rate_max = 0.04
    sp.leaf_expansion_rate_max = 0.11
    sp.girth_area_factor = f32(6)
    sp.internode_length_max = 0.15
    sp.gravitropic_curvature.uniformDistribution(-500, -200)
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False
    sp.fruit_set_probability = 1.0
    sp.defineChildShootTypes(["tiller"], [1.0])
    sp.max_nodes = 8
    sp.max_terminal_floral_buds = 1

    st = sp.copy()                                                               # copy ctor: no draw
    st.phytomer_parameters.internode.max_vegetative_buds_per_petiole = 0
    st.vegetative_bud_break_probability_min = 0
    st.defineChildShootTypes(["tiller"], [1.0])
    st.insertion_angle_tip.uniformDistribution(15, 30)
    st.base_yaw.uniformDistribution(-20, 20)
    st.base_roll.uniformDistribution(-20, 20)
    st.gravitropic_curvature.uniformDistribution(200, 400)

    pa.defineShootType("mainstem", sp)
    pa.defineShootType("tiller", st)


@ports("wheatTillerCount")
def wheatTillerCount(build_parameters) -> int:
    return getBuildParameterIntegerValue(build_parameters, wheat_tiller_count, 0)


@ports("PlantArchitecture::buildWheatPlant")
def buildWheatPlant(pa, base_position):
    """PL:6659-6687."""
    if not pa.shoot_types:
        initializeWheatShoots(pa)
    wheatTillerCount(pa.current_build_parameters)
    plantID = pa.addPlantInstance(np.asarray(base_position, f32) - np.array([0, 0, 0.025], f32), 0)
    ctx = pa.context_ptr
    internode_length_max = pa.shoot_types["mainstem"].internode_length_max.val()
    roll = ctx.randu(f32(0), f32(2.0 * np.pi))
    yaw = ctx.randu(f32(0), f32(2.0 * np.pi))
    pitch = ctx.randu(f32(0), f32(float(f32(0.05)) * np.pi))
    pa.addBaseStemShoot(plantID, 1, make_AxisRotation(pitch, yaw, roll), 0.001, internode_length_max, 0.01, 0.01, 0,
                        "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 39, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


# ---- phytomer creation function (AS:4172-4260) ---------------------------------------------------------------

def wheatBladeLengthFraction(node_index, shoot_max_nodes) -> np.float32:
    """AS:4179-4186 (file-static). ``std::fmin`` of floats: float."""
    node_index, shoot_max_nodes = int(node_index), int(shoot_max_nodes)
    fraction = min(f32(1), f32(f32(0.4) + f32(f32(f32(0.6) * f32(node_index)) / f32(5))))
    if shoot_max_nodes > 1:
        ranks_from_top = f32(shoot_max_nodes - 1 - node_index)
        fraction = min(fraction, f32(f32(0.88) + f32(f32(0.12) * min(f32(1), ranks_from_top))))
    return fraction


_SHEATH_TO_BLADE = [f32(x) for x in (0.78, 0.60, 0.56, 0.62, 0.55, 0.46, 0.36, 0.33)]


def wheatSheathLength(node_index, shoot_max_nodes, blade_length_max) -> np.float32:
    """AS:4197-4202."""
    ranks_from_top = max(0, int(shoot_max_nodes) - 1 - int(node_index))
    ratio = _SHEATH_TO_BLADE[min(ranks_from_top, len(_SHEATH_TO_BLADE) - 1)]
    return f32(f32(ratio * f32(blade_length_max)) * wheatBladeLengthFraction(node_index, shoot_max_nodes))


@named_hook("WheatPhytomerCreationFunction").implement
@ports("WheatPhytomerCreationFunction")
def WheatPhytomerCreationFunction(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:4208-4260."""
    shoot_node_index, shoot_max_nodes = int(shoot_node_index), int(shoot_max_nodes)
    shoot = phytomer.parent_shoot_ptr
    pa = shoot.plantarchitecture_ptr
    if phytomer.rank == 0:
        tiller_count = wheatTillerCount(pa.getPlantBuildParameters(phytomer.plantID))
        if shoot_node_index >= tiller_count:
            phytomer.setVegetativeBudState(BUD_DEAD)
    elif shoot_node_index == 0:
        main_stem = pa.plant_instances[phytomer.plantID].shoot_tree[shoot.parent_shoot_ID]
        tiller_max_nodes = max(1, int(main_stem.shoot_parameters.max_nodes.val()) - int(parent_shoot_node_index + 1))
        shoot.shoot_parameters.max_nodes = tiller_max_nodes
        phytomer.shoot_index[2] = tiller_max_nodes
        shoot_max_nodes = tiller_max_nodes

    phytomer.scaleLeafPrototypeScale(wheatBladeLengthFraction(shoot_node_index, shoot_max_nodes))

    elongated_internodes = 4
    internode_ranks_from_top = shoot_max_nodes - 1 - shoot_node_index
    if internode_ranks_from_top >= elongated_internodes:
        culm_internode_length = f32(0.008)
    else:
        culm_internode_length = f32(phytomer.internode_length_max
                                    * f32(f32(1) - f32(f32(0.25) * f32(internode_ranks_from_top))))
    blade_length_max = phytomer.phytomer_parameters.leaf.prototype_scale.val()
    sheath_length = wheatSheathLength(shoot_node_index, shoot_max_nodes, blade_length_max)
    sheath_length_below = (wheatSheathLength(shoot_node_index - 1, shoot_max_nodes, blade_length_max)
                           if shoot_node_index > 0 else f32(0))
    setInternodeMaxLength(phytomer, f32(f32(culm_internode_length + sheath_length) - sheath_length_below))

    if shoot_max_nodes > 0 and shoot_node_index + 1 == shoot_max_nodes:
        setGrassFlagLeafAngle(phytomer, f32(33))
        peduncle_length = shoot.shoot_parameters.phytomer_parameters.peduncle.length.val()
        phytomer.phytomer_parameters.peduncle.length = max(f32(0.01), f32(peduncle_length - sheath_length))


@named_hook("WheatSpikePrototype").implement
@ports("WheatSpikePrototype")
def WheatSpikePrototype(scene, subdivisions, build_parameters=None):
    """AS:4166-4170."""
    return obj_prototype(scene, "WheatSpike.obj")
