# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:4574-4692 and src/Assets.cpp:2007-2110, 3086-3600
# (Copyright (C) 2016-2026 Brian Bailey).
"""Maize shoot parameters, builder and phytomer creation function.

``initializeMaizeShoots`` / ``buildMaizePlant`` / ``MaizePhytomerCreationFunction`` line for line, literals with
their C++ types. A maize phytomer has a sessile strap leaf (``petiole.length = 0``: the blade sits on the culm
at its drawn leaf pitch), a flexible blade that droops under its own weight (``LeafPrototype::flexibility``
U(1.5, 2.3), with taper and ageing), a midrib class map (``SorghumLeaf_classes.png``), and at most one floral
bud: the ear at node ``max_nodes - 6`` (made ACTIVE by the creation function, set as fruit after 4 days) and the
tassel as the shoot's terminal bud (``max_terminal_floral_buds = 1``). The ear and tassel geometry are the
inflorescence machinery's (W3-D); their prototype functions are bound here.

RNG on load: four identifier draws, like the tomato. Build: ``make_AxisRotation(randu(0, 0.035 pi), randu(0, 2 pi),
randu(0, 2 pi))`` evaluated right to left (roll, yaw, then pitch); ``internode_length_max.val()`` is constant.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_ACTIVE, BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, deg2rad, make_AxisRotation, make_RGBcolor
from .strawberry import named_hook, obj_prototype

f32 = np.float32

#: MaizeTasselParameters::branch_color (PlantArchitectureInternal.h:465)
MAIZE_TASSEL_BRANCH_COLOR = (0.34, 0.42, 0.20)


@ports("PlantArchitecture::initializeMaizeShoots")
def initializeMaizeShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png"
    leaf_prototype.leaf_class_map_file[0] = "SorghumLeaf_classes.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.15)
    leaf_prototype.midrib_fold_fraction = f32(0.3)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.07, -0.04)
    leaf_prototype.longitudinal_curvature_exponent = f32(3)
    leaf_prototype.flexibility_taper = f32(150)
    leaf_prototype.flexibility_aging = f32(11)
    leaf_prototype.flexibility_aging_max = f32(6)
    leaf_prototype.lateral_curvature = f32(-0.3)
    leaf_prototype.petiole_roll = f32(0)
    leaf_prototype.wave_period = f32(0.1)
    leaf_prototype.wave_amplitude = f32(0.06)
    leaf_prototype.flexibility.uniformDistribution(1.5, 2.3)
    leaf_prototype.subdivisions = 50
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(170, 190)
    pp.internode.radius_initial = 0.0075
    pp.internode.color = make_RGBcolor(0.126, 0.182, 0.084)
    pp.internode.length_segments = 2
    pp.internode.radial_subdivisions = 10
    pp.internode.max_floral_buds_per_petiole = 1
    pp.internode.max_vegetative_buds_per_petiole = 0

    pp.petiole.petioles_per_internode = 1
    pp.petiole.length = 0

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-30, -16)
    pp.leaf.yaw = 0
    pp.leaf.roll = 0
    pp.leaf.prototype_scale = 0.9
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    pp.peduncle.length = f32(0.14)
    pp.peduncle.radius = 0.004
    pp.peduncle.curvature = 0
    pp.peduncle.color = make_RGBcolor(*(f32(c) for c in MAIZE_TASSEL_BRANCH_COLOR))   # MaizeTasselParameters().branch_color
    pp.peduncle.radial_subdivisions = 10
    pp.peduncle.length_segments = 8

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.fruit_prototype_scale = 1
    pp.inflorescence.fruit_prototype_function = named_hook("MaizeTasselPrototype")
    pp.inflorescence.unique_prototypes = 3
    pp.inflorescence.inflorescence_maturity_period = 6

    pp.phytomer_creation_function = named_hook("MaizePhytomerCreationFunction")

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.vegetative_bud_break_probability_min = 0.5
    sp.flower_bud_break_probability = 1
    sp.phyllochron_min = 3
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(6)
    sp.gravitropic_curvature.uniformDistribution(-250, 0)
    sp.internode_length_max = 0.1
    sp.tortuosity = 10
    sp.internode_length_decay_rate = 0
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False
    sp.flower_bud_break_probability = 1.0
    sp.fruit_set_probability = 1.0
    sp.defineChildShootTypes(["mainstem"], [1.0])
    sp.max_nodes = 20
    sp.max_terminal_floral_buds = 1

    pa.defineShootType("mainstem", sp)


@ports("PlantArchitecture::buildMaizePlant")
def buildMaizePlant(pa, base_position):
    """PL:4672-4692. The base rotation's three ``randu`` are drawn right to left (GCC): roll, yaw, pitch.
    ``0.035f * M_PI`` and ``2.f * M_PI`` are double products narrowed by ``randu(float, float)``."""
    if not pa.shoot_types:
        initializeMaizeShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    ctx = pa.context_ptr
    # addBaseStemShoot's arguments, right to left: the constant internode_length_max.val() makes no draw
    internode_length_max = pa.shoot_types["mainstem"].internode_length_max.val()
    roll = ctx.randu(f32(0), f32(2.0 * np.pi))
    yaw = ctx.randu(f32(0), f32(2.0 * np.pi))
    pitch = ctx.randu(f32(0), f32(float(f32(0.035)) * np.pi))
    pa.addBaseStemShoot(plantID, 1, make_AxisRotation(pitch, yaw, roll), 0.003, internode_length_max, 0.01, 0.01, 0.2,
                        "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 58, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


# ---- phytomer creation function (AS:2014-2110) -----------------------------------------------------

def setGrassFlagLeafAngle(phytomer, angle_off_culm_degrees):
    """AS:2014-2027 (file-static): turn the flag leaf to ``angle`` off the culm with ``Phytomer::rotateLeaf``."""
    if not phytomer.leaf_objIDs or not phytomer.leaf_objIDs[0]:
        return
    drawn_angle_off_culm = abs(f32(phytomer.petiole_pitch[0]))                       # std::fabs(float)
    phytomer.rotateLeaf(0, 0, make_AxisRotation(f32(drawn_angle_off_culm - deg2rad(f32(angle_off_culm_degrees))), 0, 0))


@named_hook("MaizePhytomerCreationFunction").implement
@ports("MaizePhytomerCreationFunction")
def MaizePhytomerCreationFunction(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    shoot_node_index, shoot_max_nodes = int(shoot_node_index), int(shoot_max_nodes)
    scale = f32(1)
    if shoot_node_index <= 5:
        # fmin(1.f, 0.7 + 0.3 * float(i) / 5.f): double arithmetic, double fmin, narrowed
        scale = f32(min(1.0, 0.7 + 0.3 * float(f32(shoot_node_index)) / 5.0))
        phytomer.scaleInternodeMaxLength(scale)

    ranks_from_top = f32(int(phytomer.shoot_index[2]) - 1 - shoot_node_index)
    inner = f32(f32(0.475) + f32(f32(0.092) * ranks_from_top))
    inner = f32(min(1.0, float(inner)))                                              # fmin(1.f, float): exact
    scale = f32(min(float(scale), float(inner)))
    phytomer.scaleLeafPrototypeScale(scale)

    if phytomer.shoot_index[2] > 1:
        rank_fraction = f32(f32(shoot_node_index) / f32(int(phytomer.shoot_index[2]) - 1))
        # leaf_flexibility *= 1.f - 0.75f * fmax(0.f, rank_fraction - 0.45f) / 0.55f: C fmax -> double chain
        fm = max(0.0, float(f32(rank_fraction - f32(0.45))))
        factor = 1.0 - float(f32(0.75)) * fm / float(f32(0.55))
        phytomer.leaf_flexibility = f32(float(phytomer.leaf_flexibility) * factor)

    nodes_below_tassel = 6
    apical_ear_node = shoot_max_nodes - nodes_below_tassel
    ears_per_plant = 1
    is_ear = shoot_max_nodes > 0 and (shoot_node_index == apical_ear_node
                                      or (ears_per_plant >= 2 and shoot_node_index == apical_ear_node - 1))

    pp = phytomer.phytomer_parameters
    if is_ear:
        pp.inflorescence.flowers_per_peduncle = 1
        pp.inflorescence.fruit_prototype_function = named_hook("MaizeEarPrototype")
        pp.inflorescence.fruit_prototype_scale = 0.2
        pp.inflorescence.pitch.uniformDistribution(0, 30)
        pp.inflorescence.unique_prototypes = 1
        pp.peduncle.color = pp.internode.color
        pp.peduncle.length = f32(0.05)
        pp.peduncle.radius = 0.01
        pp.peduncle.pitch = 5
        phytomer.setFloralBudState(BUD_ACTIVE)
    else:
        pp.inflorescence.fruit_prototype_function = named_hook("MaizeTasselPrototype")
        phytomer.setFloralBudState(BUD_DEAD)

    if shoot_max_nodes > 0 and shoot_node_index + 1 == shoot_max_nodes:
        setGrassFlagLeafAngle(phytomer, 23.0)


# ---- organ prototypes ------------------------------------------------------------------------------

@named_hook("MaizeEarPrototype").implement
@ports("MaizeEarPrototype")
def MaizeEarPrototype(scene, subdivisions, build_parameters=None):
    """AS:2007-2011."""
    return obj_prototype(scene, "MaizeEar.obj")


@named_hook("MaizeTasselPrototype").implement
@ports("MaizeTasselPrototype")
def MaizeTasselPrototype(scene, subdivisions, build_parameters=None):
    """AS:3549 (procedural tassel: axis, primary/secondary branches, spikelets, anthers). Not ported: it is only
    reached from the inflorescence machinery (W3-D), for the terminal bud of a maize shoot at its node cap."""
    raise NotImplementedError("MaizeTasselPrototype (procedural maize tassel) is not ported yet")
