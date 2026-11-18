# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:387-398, 2131-2694 and
# src/Assets.cpp:1049-1529 (Copyright (C) 2016-2026 Brian Bailey).
"""Apple trees (W5-E): ``apple`` (open-centre tree on four scaffolds) and ``apple_fruitingwall`` (tall spindle).

``apple``
    ``initializeAppleTreeShoots`` (trunk / proleptic / spur) and ``buildAppleTree``: a straight trunk of
    ``trunk_height / 0.04`` nodes with a small lean (``addBaseStemShoot``), a short phytomer on top, its buds and
    leaves removed, ``num_scaffolds`` proleptic scaffolds of 7-9 nodes one per trunk node below the top, the tree
    made dormant. Phenology 165 / -1 / 3 / 7 / 30 / 200, ``max_age`` 1460.

``apple_fruitingwall``
    ``initializeAppleFruitingWallShoots`` (leader / branch / proleptic / renewal / spur) and
    ``buildAppleFruitingWall``: a feathered nursery whip built from prescribed node positions
    (``addShootFromNodePositions``), 6-15 prescribed feathers tied below horizontal, the wall dimensions recorded
    as plant build parameters. ``max_age`` 3650.

Assets (AS:1049-1529): ``AppleFruitPrototype`` (OBJ + reflectivity spectrum / waxy layer primitive data),
``AppleFlowerPrototype`` (the almond flower OBJ), ``ApplePhytomerCreationFunction`` (spurs, short shoots, axillary
flower zones), ``ApplePhytomerCallbackFunction`` (spur release, flower thinning by the past summer's light, the
distal zone without axillary flowers) and the fruiting-wall pair, whose callback runs the winter pruning and the
summer training of the tall spindle once per time step from the leader's tip phytomer.

GCC evaluates function arguments right to left, which fixes the draw order in ``make_AxisRotation(randu(),
randu(), 0)`` (yaw first) and in ``addChildShoot(..., make_AxisRotation(pitch, yaw(randu), 0), ...)``.
"""
from __future__ import annotations

import math

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import (BUD_ACTIVE, BUD_DEAD, BUD_DORMANT, LeafPrototype, PhytomerParameters, ShootParameters,
                      make_AxisRotation, make_RGBcolor)
from ..random_parameter import PlantArchitectureError
from .build_parameters import BuildParameterDeclaration, getBuildParameterIntegerValue, getBuildParameterValue
from .cherrytomato import _hook
from .trees import (_no_shoot, apple_trunk_internode_length, checkScaffoldsFitTrunk, cutShootToLength,
                    growSpursFromUnbrokenBuds, kill_trunk_buds, releaseBudsBelowCut, scaffold_yaw,
                    setShootInternodeLengthMax, tree_num_scaffolds, tree_scaffold_angle, tree_trunk_height,
                    trunkMaxNodes, trunk_node_count)

f32 = np.float32
PI_F = f32(math.pi)

AppleFruitPrototype = _hook("AppleFruitPrototype")                                              # AS:1049
AppleFlowerPrototype = _hook("AppleFlowerPrototype")                                            # AS:1063
ApplePhytomerCreationFunction = _hook("ApplePhytomerCreationFunction")                          # AS:1123
ApplePhytomerCallbackFunction = _hook("ApplePhytomerCallbackFunction")                          # AS:1195
AppleFruitingWallPhytomerCreationFunction = _hook("AppleFruitingWallPhytomerCreationFunction")  # AS:1497
AppleFruitingWallPhytomerCallbackFunction = _hook("AppleFruitingWallPhytomerCallbackFunction")  # AS:1513

# ---- fruiting-wall build parameters (PL:387-398) ------------------------------------------------------------
apple_fruitingwall_leader_height_min = f32(1.5)
apple_fruitingwall_leader_height_max = f32(1.8)
apple_fruitingwall_dormant_bud_zone = f32(0.4)
apple_fruitingwall_feather_gap = f32(0.05)
apple_fruitingwall_tree_height = BuildParameterDeclaration("tree_height", 2.0, 4.5, False,
                                                           "height at which the leader is headed, in meters")
# constexpr float arithmetic: 1.5f - 0.4f - 0.05f - 0.05f
apple_fruitingwall_trunk_height = BuildParameterDeclaration(
    "trunk_height", 0.1,
    float(f32(f32(f32(apple_fruitingwall_leader_height_min - apple_fruitingwall_dormant_bud_zone)
                      - apple_fruitingwall_feather_gap) - f32(0.05))),
    False, "height of the lowest feather on the leader in meters")
apple_fruitingwall_num_feathers = BuildParameterDeclaration("num_feathers", 6.0, 15.0, True,
                                                            "number of feathers on the nursery tree")
apple_fruitingwall_tree_spacing = BuildParameterDeclaration("tree_spacing", 0.5, 2.0, False,
                                                            "in-row tree spacing in meters")
apple_fruitingwall_row_spacing = BuildParameterDeclaration("row_spacing", 2.0, 5.0, False,
                                                           "between-row spacing in meters")
APPLE_FRUITINGWALL_PARAMETERS = [apple_fruitingwall_tree_height, apple_fruitingwall_trunk_height,
                                 apple_fruitingwall_num_feathers, apple_fruitingwall_tree_spacing,
                                 apple_fruitingwall_row_spacing]

# ---- apple constants (AS:1070-1086) ---------------------------------------------------------------------------
apple_spur_fraction = f32(0.2)
apple_axillary_flowering_basal_nodes = 4
apple_axillary_flowering_distal_nodes = 3
apple_axillary_flowering_probability = f32(0.3)
apple_flowering_light_exposure_min = f32(0.45)
apple_flowering_light_exposure_full = f32(0.8)

# ---- fruiting-wall training constants (AS:1209-1257) ----------------------------------------------------------
apple_fruitingwall_spur_fraction = f32(0.3)
apple_fruitingwall_leader_spur_fraction = f32(0.1)
apple_fruitingwall_renewals_per_winter = 3
apple_fruitingwall_default_tree_height = f32(3.1)
apple_fruitingwall_default_trunk_height = f32(0.7)
apple_fruitingwall_default_tree_spacing = f32(0.9)
apple_fruitingwall_default_row_spacing = f32(3.3)
apple_fruitingwall_wall_width_base = f32(1.5)
apple_fruitingwall_wall_width_top = f32(0.75)
apple_fruitingwall_reference_row_spacing = f32(3.3)
apple_fruitingwall_row_overlap = f32(0.05)
apple_fruitingwall_height_tolerance = f32(0.03)
apple_fruitingwall_heading_window = f32(0.5)
apple_fruitingwall_branch_diameter_ratio_max = f32(0.5)
apple_fruitingwall_branch_diameter_max = f32(0.025)
apple_fruitingwall_branch_age_max = 6
apple_fruitingwall_branch_ratio_rule_age_min = 3
apple_fruitingwall_branch_length_max = f32(0.75)
apple_fruitingwall_branch_length_headed = f32(0.6)
apple_fruitingwall_upright_side_shoot_length_max = f32(0.25)
apple_fruitingwall_side_shoot_length_max = f32(0.35)
apple_fruitingwall_side_shoot_length_headed = f32(0.2)
apple_fruitingwall_competitor_zone = f32(0.2)
apple_fruitingwall_competitor_nodes_kept = 3
apple_fruitingwall_buds_released_at_heading = 2
apple_fruitingwall_low_feather_tolerance = f32(0.05)

NO_ID = 0xFFFFFFFF          # uint(-1)


def _apple_leaf_prototype(gen):
    lp = LeafPrototype(gen)
    lp.leaf_texture_file[0] = "AppleLeaf.png"
    lp.leaf_class_map_file[0] = "AppleLeaf_classes.png"
    lp.leaf_aspect_ratio = f32(0.6)
    lp.midrib_fold_fraction = f32(0.4)
    lp.longitudinal_curvature = f32(-0.3)
    lp.lateral_curvature = f32(0.1)
    lp.subdivisions = 6
    lp.unique_prototypes = 1
    return lp


def _apple_phytomer_parameters(gen, creation=None, callback=None):
    """PL:2147-2191 (``apple``) / PL:2351-2397 (``apple_fruitingwall``, which also sets the two functions)."""
    pp = PhytomerParameters(gen)
    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(130, 145)
    pp.internode.radius_initial = 0.004
    pp.internode.length_segments = 1
    pp.internode.image_texture = "AppleBark.jpg"
    pp.internode.image_class_map_file = "AppleBark_classes.png"
    pp.internode.max_floral_buds_per_petiole = 3
    if creation is not None:
        pp.phytomer_creation_function = creation
        pp.phytomer_callback_function = callback

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(-40, -25)
    pp.petiole.taper = 0.1
    pp.petiole.curvature = 0
    pp.petiole.length = 0.04
    pp.petiole.radius = 0.00075
    pp.petiole.length_segments = 1
    pp.petiole.radial_subdivisions = 3
    pp.petiole.color = make_RGBcolor(0.61, 0.5, 0.24)

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.prototype_scale = 0.09
    pp.leaf.prototype = _apple_leaf_prototype(gen)

    pp.peduncle.length = 0.025
    pp.peduncle.radius = 0.001
    pp.peduncle.pitch.uniformDistribution(40, 70)
    pp.peduncle.roll = 90
    pp.peduncle.length_segments = 1

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.flower_prototype_scale = 0.03
    pp.inflorescence.flower_prototype_function = AppleFlowerPrototype
    pp.inflorescence.fruit_prototype_scale = 0.08
    pp.inflorescence.fruit_prototype_function = AppleFruitPrototype
    pp.inflorescence.fruit_class_map_file = "AppleFruit_classes.png"
    pp.inflorescence.fruit_gravity_factor_fraction = 0.7
    return pp


@ports("PlantArchitecture::initializeAppleTreeShoots")
def initializeAppleTreeShoots(pa):
    gen = context_generator(pa.context_ptr)
    pp = _apple_phytomer_parameters(gen)

    trunk = ShootParameters(gen)
    trunk.phytomer_parameters = pp
    trunk.phytomer_parameters.internode.phyllotactic_angle = 0
    trunk.phytomer_parameters.internode.radius_initial = 0.01
    trunk.phytomer_parameters.internode.radial_subdivisions = 24
    trunk.max_nodes = trunkMaxNodes(tree_trunk_height, apple_trunk_internode_length)
    trunk.girth_area_factor = f32(1.35)
    trunk.vegetative_bud_break_probability_min = 0
    trunk.vegetative_bud_break_time = 0
    trunk.tortuosity = 20
    trunk.internode_length_max = 0.05
    trunk.internode_length_decay_rate = 0
    trunk.defineChildShootTypes(["proleptic"], [1])

    pro = ShootParameters(gen)
    pro.phytomer_parameters = pp
    pro.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2)
    pro.phytomer_parameters.phytomer_creation_function = ApplePhytomerCreationFunction
    pro.phytomer_parameters.phytomer_callback_function = ApplePhytomerCallbackFunction
    pro.max_nodes = 40
    pro.max_nodes_per_season = 20
    pro.phyllochron_min = 2.0
    pro.elongation_rate_max = 0.15
    pro.girth_area_factor = f32(2.0)
    pro.vegetative_bud_break_probability_min = 0.1
    pro.vegetative_bud_break_probability_decay_rate = 0.4
    pro.vegetative_bud_break_time = 0
    pro.gravitropic_curvature.uniformDistribution(450, 500)
    pro.tortuosity = 75
    pro.insertion_angle_tip.uniformDistribution(30, 40)
    pro.insertion_angle_decay_rate = 20
    pro.internode_length_max = 0.04
    pro.internode_length_min = 0.01
    pro.internode_length_decay_rate = 0.004
    pro.fruit_set_probability = 0.1
    pro.flower_bud_break_probability = 0.5
    pro.max_terminal_floral_buds = 3
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False
    pro.defineChildShootTypes(["proleptic"], [1.0])

    spur = pro.copy()
    spur.phytomer_parameters.internode.radius_initial = 0.002
    spur.phytomer_parameters.peduncle.pitch.uniformDistribution(0, 30)
    spur.max_nodes = 40
    spur.max_nodes_per_season = 4
    spur.internode_length_max = 0.0025
    spur.internode_length_min = 0.0025
    spur.internode_length_decay_rate = 0
    spur.vegetative_bud_break_probability_max = 0
    spur.vegetative_bud_break_probability_min = 0
    spur.vegetative_bud_break_probability_decay_rate = 0
    spur.gravitropic_curvature = 0
    spur.max_terminal_floral_buds.uniformDistribution(5, 6)
    spur.flower_bud_break_probability = 1.0
    spur.defineChildShootTypes(["spur"], [1.0])

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("proleptic", pro)
    pa.defineShootType("spur", spur)


@ports("PlantArchitecture::buildAppleTree")
def buildAppleTree(pa, base_position):
    """PL:2278-2333."""
    if not pa.shoot_types:
        initializeAppleTreeShoots(pa)
    ctx = pa.context_ptr
    bp = pa.current_build_parameters
    trunk_height = getBuildParameterValue(bp, tree_trunk_height, f32(0.8))
    num_scaffolds = int(getBuildParameterIntegerValue(bp, tree_num_scaffolds, 4))
    scaffold_angle = getBuildParameterValue(bp, tree_scaffold_angle, f32(40))
    trunk_internode_length = apple_trunk_internode_length
    trunk_nodes = trunk_node_count(trunk_height, trunk_internode_length)
    checkScaffoldsFitTrunk("buildAppleTree", num_scaffolds, trunk_nodes, trunk_height)
    trunk_radius, scaffold_radius, scaffold_length = f32(0.008), f32(0.005), f32(0.04)

    plantID = pa.addPlantInstance(base_position, 0)
    yaw = ctx.randu(f32(0), f32(2.0 * math.pi))
    pitch = ctx.randu(f32(0), f32(float(f32(0.05)) * math.pi))
    uID_trunk = pa.addBaseStemShoot(plantID, trunk_nodes, make_AxisRotation(pitch, yaw, 0), trunk_radius,
                                    trunk_internode_length, f32(1), f32(1), f32(0), "trunk")
    pa.appendPhytomerToShoot(plantID, uID_trunk, pa.shoot_types["trunk"].phytomer_parameters, f32(0), f32(0.01),
                             f32(1), f32(1))
    kill_trunk_buds(pa, plantID, uID_trunk)
    tree = pa.plant_instances[plantID].shoot_tree
    for i in range(num_scaffolds):
        pitch = f32(V.deg2rad(scaffold_angle) + ctx.randu(f32(-0.1), f32(0.1)))
        scaffold_nodes = int(ctx.randu(7, 9))
        yaw = scaffold_yaw(ctx, i, num_scaffolds)
        pa.addChildShoot(plantID, uID_trunk, tree[uID_trunk].current_node_number - i - 1, scaffold_nodes,
                         make_AxisRotation(pitch, yaw, 0), scaffold_radius, scaffold_length, f32(1), f32(1), f32(0.5),
                         "proleptic", 0)
    from ..seasons import makePlantDormant
    makePlantDormant(pa, plantID)
    pa.setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200)
    pa.plant_instances[plantID].max_age = f32(1460)
    return plantID


@ports("PlantArchitecture::initializeAppleFruitingWallShoots")
def initializeAppleFruitingWallShoots(pa):
    gen = context_generator(pa.context_ptr)
    pp = _apple_phytomer_parameters(gen, AppleFruitingWallPhytomerCreationFunction,
                                    AppleFruitingWallPhytomerCallbackFunction)

    leader = ShootParameters(gen)
    leader.phytomer_parameters = pp
    leader.phytomer_parameters.internode.radial_subdivisions = 24
    leader.phytomer_parameters.internode.radius_initial = 0.004
    leader.max_nodes = 400
    leader.max_nodes_per_season = 28
    leader.phyllochron_min = 4.0
    leader.elongation_rate_max = 0.15
    leader.girth_area_factor = f32(0.7)
    leader.gravitropic_curvature = 400
    leader.tortuosity = 15
    leader.internode_length_max = 0.025
    leader.internode_length_min = 0.025
    leader.internode_length_decay_rate = 0
    leader.vegetative_bud_break_probability_max = 0.8
    leader.vegetative_bud_break_probability_min = 0.15
    leader.vegetative_bud_break_probability_decay_rate = 0.06
    leader.vegetative_bud_break_time = 0
    leader.max_terminal_floral_buds = 0
    leader.flower_bud_break_probability = 0
    leader.fruit_set_probability = 0
    leader.flowers_require_dormancy = True
    leader.growth_requires_dormancy = True
    leader.determinate_shoot_growth = False
    leader.defineChildShootTypes(["branch"], [1])

    branch = ShootParameters(gen)
    branch.phytomer_parameters = pp
    branch.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2)
    branch.max_nodes = 80
    branch.max_nodes_per_season = 10
    branch.phyllochron_min = 2.0
    branch.elongation_rate_max = 0.15
    branch.girth_area_factor = f32(0.55)
    branch.vegetative_bud_break_probability_max = 0.15
    branch.vegetative_bud_break_probability_min = 0.03
    branch.vegetative_bud_break_probability_decay_rate = 0.03
    branch.vegetative_bud_break_time = 0
    branch.gravitropic_curvature.uniformDistribution(-100, -20)
    branch.tortuosity = 60
    branch.insertion_angle_tip.uniformDistribution(75, 90)
    branch.insertion_angle_decay_rate = 5
    branch.internode_length_max = 0.02
    branch.internode_length_min = 0.012
    branch.internode_length_decay_rate = 0.001
    branch.fruit_set_probability = 0.1
    branch.flower_bud_break_probability = 0.5
    branch.max_terminal_floral_buds = 3
    branch.flowers_require_dormancy = True
    branch.growth_requires_dormancy = True
    branch.determinate_shoot_growth = False
    branch.defineChildShootTypes(["proleptic"], [1.0])

    pro = branch.copy()
    pro.phytomer_parameters.internode.radius_initial = 0.0025
    pro.max_nodes = 40
    pro.max_nodes_per_season = 22
    pro.girth_area_factor = f32(1.5)
    pro.vegetative_bud_break_probability_max = 1.0
    pro.vegetative_bud_break_probability_min = 0.1
    pro.vegetative_bud_break_probability_decay_rate = 0.4
    pro.gravitropic_curvature.uniformDistribution(150, 300)
    pro.tortuosity = 75
    pro.insertion_angle_tip.uniformDistribution(50, 80)
    pro.insertion_angle_decay_rate = 10
    pro.internode_length_max = 0.018
    pro.internode_length_min = 0.01
    pro.internode_length_decay_rate = 0.002
    pro.defineChildShootTypes(["proleptic"], [1.0])

    renewal = branch.copy()
    renewal.phytomer_parameters.internode.radius_initial = 0.003
    renewal.max_nodes_per_season = 14
    renewal.gravitropic_curvature.uniformDistribution(-80, -30)
    renewal.insertion_angle_tip.uniformDistribution(5, 25)
    renewal.insertion_angle_decay_rate = 0
    renewal.internode_length_max = 0.025

    spur = branch.copy()
    spur.phytomer_parameters.internode.radius_initial = 0.002
    spur.phytomer_parameters.peduncle.pitch.uniformDistribution(0, 30)
    spur.max_nodes = 40
    spur.max_nodes_per_season = 6
    spur.internode_length_max = 0.0025
    spur.internode_length_min = 0.0025
    spur.internode_length_decay_rate = 0
    spur.vegetative_bud_break_probability_max = 0
    spur.vegetative_bud_break_probability_min = 0
    spur.vegetative_bud_break_probability_decay_rate = 0
    spur.gravitropic_curvature = 0
    spur.tortuosity = 75
    spur.insertion_angle_tip.uniformDistribution(50, 80)
    spur.max_terminal_floral_buds.uniformDistribution(5, 6)
    spur.flower_bud_break_probability = 1.0
    spur.defineChildShootTypes(["spur"], [1.0])

    pa.defineShootType("leader", leader)
    pa.defineShootType("branch", branch)
    pa.defineShootType("proleptic", pro)
    pa.defineShootType("renewal", renewal)
    pa.defineShootType("spur", spur)


def _mark_spur_considered(phytomer):
    for petiole in phytomer.axillary_vegetative_buds:
        for vbud in petiole:
            vbud.shoot_type_label = "spur"


def _f32_round(x) -> int:
    """``uint(std::round(float))`` (std::round on a float: half away from zero)."""
    x = float(f32(x))
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


@ports("PlantArchitecture::buildAppleFruitingWall")
def buildAppleFruitingWall(pa, base_position):
    """PL:2529-2694."""
    if not pa.shoot_types:
        initializeAppleFruitingWallShoots(pa)
    ctx = pa.context_ptr
    bp = pa.current_build_parameters
    tree_height = f32(getBuildParameterValue(bp, apple_fruitingwall_tree_height, f32(3.1)))
    trunk_height = f32(getBuildParameterValue(bp, apple_fruitingwall_trunk_height, f32(0.7)))
    num_feathers = int(getBuildParameterIntegerValue(bp, apple_fruitingwall_num_feathers, 10))
    tree_spacing = f32(getBuildParameterValue(bp, apple_fruitingwall_tree_spacing, f32(0.9)))
    row_spacing = f32(getBuildParameterValue(bp, apple_fruitingwall_row_spacing, f32(3.3)))

    plantID = pa.addPlantInstance(base_position, 0)
    inst = pa.plant_instances[plantID]
    inst.build_parameters[apple_fruitingwall_tree_height.name] = tree_height
    inst.build_parameters[apple_fruitingwall_trunk_height.name] = trunk_height
    inst.build_parameters[apple_fruitingwall_num_feathers.name] = f32(num_feathers)
    inst.build_parameters[apple_fruitingwall_tree_spacing.name] = tree_spacing
    inst.build_parameters[apple_fruitingwall_row_spacing.name] = row_spacing
    base_position = np.asarray(base_position, f32)

    # ---- nursery whip
    # std::fmin(float, float) -> double, narrowed to the float leader_height
    leader_height = f32(min(float(ctx.randu(apple_fruitingwall_leader_height_min, apple_fruitingwall_leader_height_max)),
                            float(f32(tree_height - f32(0.3)))))
    leader_node_spacing = f32(0.03)
    leader_base_radius = f32(0.008)
    leader_tip_radius = f32(0.003)
    leader_nodes = max(2, int(math.ceil(float(f32(leader_height / leader_node_spacing)))))

    lean_azimuth = ctx.randu(f32(0), f32(f32(2) * PI_F))
    lean_angle = f32(V.deg2rad(ctx.randu(f32(0), f32(1.5))))
    tan_lean = V.tanf(lean_angle)
    cos_az, sin_az = V.cosf(lean_azimuth), V.sinf(lean_azimuth)
    leader_positions, leader_radii = [], []
    for node in range(leader_nodes + 1):
        fraction = f32(f32(node) / f32(leader_nodes))
        height = f32(fraction * leader_height)
        lean = f32(f32(height * tan_lean) * fraction)
        leader_positions.append(base_position + np.array([f32(lean * cos_az), f32(lean * sin_az), height], f32))
        leader_radii.append(f32(leader_base_radius + f32(f32(leader_tip_radius - leader_base_radius) * fraction)))

    from ..shoot import addShootFromNodePositions
    uID_leader = addShootFromNodePositions(pa, plantID, -1, 0, leader_positions, leader_radii, "leader")
    leader = inst.shoot_tree[uID_leader]
    setShootInternodeLengthMax(pa, plantID, uID_leader, pa.shoot_types["leader"].internode_length_max.val())

    leader_dormant_bud_zone = apple_fruitingwall_dormant_bud_zone
    for node in range(len(leader.phytomers)):
        phytomer = leader.phytomers[node]
        phytomer.removeLeaf()
        phytomer.setFloralBudState(BUD_DEAD)
        node_height = f32(leader.shoot_internode_vertices[node][-1][2] - base_position[2])
        if node_height > f32(leader_height - leader_dormant_bud_zone):
            phytomer.setVegetativeBudState(BUD_DORMANT)
        else:
            phytomer.setVegetativeBudState(BUD_DEAD)
            if node_height < trunk_height:
                _mark_spur_considered(phytomer)

    # ---- feathers
    feather_zone_bottom = trunk_height
    feather_zone_top = f32(f32(leader_height - leader_dormant_bud_zone) - apple_fruitingwall_feather_gap)
    feather_node_spacing = f32(0.025)
    feather_nodes = []
    zone = f32(feather_zone_top - feather_zone_bottom)
    for feather in range(num_feathers):
        z_low = f32(feather_zone_bottom + f32(f32(zone * f32(feather)) / f32(num_feathers)))
        z_high = f32(feather_zone_bottom + f32(f32(zone * f32(feather + 1)) / f32(num_feathers)))
        z = ctx.randu(z_low, z_high)
        node = min(max(_f32_round(f32(z / leader_node_spacing)), 1), len(leader.phytomers) - 1)
        if feather_nodes and node <= feather_nodes[-1]:
            node = feather_nodes[-1] + 1
        if node >= len(leader.phytomers):
            break
        feather_nodes.append(node)

    for node in feather_nodes:
        attachment = np.asarray(leader.shoot_internode_vertices[node][-1], f32).copy()
        outward = np.asarray(leader.phytomers[node].getPetioleAxisVector(f32(0), 0), f32).copy()
        outward[2] = f32(0)
        if V.magnitude(outward) < f32(1e-6):
            outward = np.array([1, 0, 0], f32)
        outward = V.normalize(outward)
        azimuth = f32(V.atan2f(outward[1], outward[0]) + V.deg2rad(ctx.randu(f32(-15), f32(15))))

        feather_length = ctx.randu(f32(0.15), f32(0.30))
        feather_node_count = max(3, _f32_round(f32(feather_length / feather_node_spacing)))
        segment_length = f32(feather_length / f32(feather_node_count))
        angle_base = f32(V.deg2rad(ctx.randu(f32(5), f32(15))))
        angle_tip = f32(V.deg2rad(ctx.randu(f32(20), f32(40))))
        # std::fmax(0.0015f, randu * radius): double, narrowed
        feather_base_radius = f32(max(float(f32(0.0015)), float(f32(ctx.randu(f32(0.3), f32(0.45))
                                                                       * leader.shoot_internode_radii[node][-1]))))
        feather_tip_radius = f32(min(float(f32(0.0015)), float(feather_base_radius)))

        positions = [attachment]
        radii = [feather_base_radius]
        for segment in range(feather_node_count):
            fraction = f32(f32(f32(segment) + f32(0.5)) / f32(feather_node_count))
            angle = f32(angle_base + f32(f32(angle_tip - angle_base) * fraction))
            azimuth = f32(azimuth + V.deg2rad(ctx.randu(f32(-3), f32(3))))
            ca = V.cosf(angle)
            direction = np.array([f32(ca * V.cosf(azimuth)), f32(ca * V.sinf(azimuth)), -V.sinf(angle)], f32)
            positions.append(positions[segment] + segment_length * direction)
            radii.append(f32(feather_base_radius + f32(f32(f32(feather_tip_radius - feather_base_radius)
                                                           * f32(segment + 1)) / f32(feather_node_count))))
        uID_feather = addShootFromNodePositions(pa, plantID, int(uID_leader), node, positions, radii, "branch")
        feather = inst.shoot_tree[uID_feather]
        setShootInternodeLengthMax(pa, plantID, uID_feather, pa.shoot_types["branch"].internode_length_max.val())
        for feather_node, phytomer in enumerate(feather.phytomers):
            phytomer.removeLeaf()
            phytomer.setFloralBudState(BUD_DEAD)
            if feather_node == 0:
                phytomer.setVegetativeBudState(BUD_DEAD)
                _mark_spur_considered(phytomer)
            else:
                phytomer.setVegetativeBudState(BUD_DORMANT)

    from ..seasons import makePlantDormant
    makePlantDormant(pa, plantID)
    pa.setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200)
    inst.max_age = f32(3650)
    return plantID


# ---- assets ---------------------------------------------------------------------------------------------------

@AppleFruitPrototype.implement
@ports("AppleFruitPrototype")
def _apple_fruit_prototype(context, subdivisions, build_parameters=None):
    """AS:1049-1061."""
    from .bean import plugin_asset
    uu = context.load_obj(plugin_asset("assets/obj/AppleFruit.obj"), (0.0, 0.0, 0.0), height=0.0, rotation=(0.0, 0.0),
                          color=(0.0, 0.0, 0.0), upaxis="ZUP", silent=True)
    objID = context.add_polymesh_object(uu)
    context.set_primitive_data(uu, "reflectivity_spectrum", "apple_fruit_reflectivity_0000", "string")
    context.set_primitive_data(uu, "specular_roughness", f32(0.15), "float")
    context.set_primitive_data(uu, "specular_n", f32(1.45), "float")
    return objID


@AppleFlowerPrototype.implement
@ports("AppleFlowerPrototype")
def _apple_flower_prototype(context, subdivisions, flower_is_open, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/AlmondFlower.obj")


@ports("thinAppleFlowerClusterForLight")
def thinAppleFlowerClusterForLight(phytomer, terminal: bool, light_exposure):
    """AS:1095-1121."""
    span = f32(apple_flowering_light_exposure_full - apple_flowering_light_exposure_min)
    kept_fraction = min(max(f32(f32(f32(light_exposure) - apple_flowering_light_exposure_min) / span), f32(0)), f32(1))
    clusters = [[]] if terminal else []
    for petiole in phytomer.floral_buds:
        if not terminal:
            clusters.append([])
        for fbud in petiole:
            if bool(fbud.isterminal) == terminal:
                clusters[-1].append(fbud)
    for cluster in clusters:
        kept = int(math.ceil(float(f32(kept_fraction * f32(len(cluster))))))
        for i in range(kept, len(cluster)):
            if cluster[i].state == BUD_DORMANT:
                phytomer.setFloralBudState(BUD_DEAD, cluster[i])


@ApplePhytomerCreationFunction.implement
@ports("ApplePhytomerCreationFunction")
def _apple_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:1123-1157."""
    shoot = phytomer.parent_shoot_ptr
    if shoot.shoot_type_label == "spur":
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.scaleLeafPrototypeScale(f32(0.8))
        phytomer.setFloralBudState(BUD_DEAD)
        return
    if f32(phytomer.internode_length_max) < f32(0.011):
        phytomer.setVegetativeBudState(BUD_DEAD)
        _mark_spur_considered(phytomer)
        phytomer.scaleLeafPrototypeScale(f32(0.8))
        shoot.growth_state.max_nodes_per_season_growth_cap = 6
    growth_unit_node_index = int(shoot_node_index) if shoot.dormancy_cycles == 0 else int(shoot.nodes_this_season)
    if (growth_unit_node_index < apple_axillary_flowering_basal_nodes
            or shoot.context_ptr.randu() >= apple_axillary_flowering_probability):
        phytomer.setFloralBudState(BUD_DEAD)


def applePhytomerCallback(phytomer, spur_fraction):
    """AS:1160-1193."""
    growSpursFromUnbrokenBuds(phytomer, spur_fraction)
    shoot = phytomer.parent_shoot_ptr
    if not shoot.isdormant:
        return
    from ..seasons import getSeasonPeakLightExposureAtPoint
    light_exposure = getSeasonPeakLightExposureAtPoint(shoot.plantarchitecture_ptr, shoot.plantID,
                                                       phytomer.getInternodeNodePositions()[-1])
    thinAppleFlowerClusterForLight(phytomer, False, light_exposure)
    x, y = int(phytomer.shoot_index[0]), int(phytomer.shoot_index[1])
    if x == y - 1:
        thinAppleFlowerClusterForLight(phytomer, True, light_exposure)
    if shoot.shoot_type_label == "spur":
        return
    nodes_below_tip = y - x - 1
    if nodes_below_tip >= apple_axillary_flowering_distal_nodes:
        return
    for petiole in phytomer.floral_buds:
        for fbud in petiole:
            if not fbud.isterminal and fbud.state == BUD_DORMANT:
                phytomer.setFloralBudState(BUD_DEAD, fbud)


@ApplePhytomerCallbackFunction.implement
@ports("ApplePhytomerCallbackFunction")
def _apple_phytomer_callback(phytomer):
    applePhytomerCallback(phytomer, apple_spur_fraction)


# ---- tall-spindle training (AS:1259-1529) ----------------------------------------------------------------------

class AppleFruitingWallDimensions:
    __slots__ = ("tree_height", "trunk_height", "tree_spacing", "row_spacing")


@ports("appleFruitingWallDimensions")
def appleFruitingWallDimensions(pa, plantID) -> AppleFruitingWallDimensions:
    bp = pa.plant_instances[plantID].build_parameters
    wall = AppleFruitingWallDimensions()
    wall.tree_height = f32(bp.get("tree_height", apple_fruitingwall_default_tree_height))
    wall.trunk_height = f32(bp.get("trunk_height", apple_fruitingwall_default_trunk_height))
    wall.tree_spacing = f32(bp.get("tree_spacing", apple_fruitingwall_default_tree_spacing))
    wall.row_spacing = f32(bp.get("row_spacing", apple_fruitingwall_default_row_spacing))
    return wall


@ports("appleFruitingWallHalfWidth")
def appleFruitingWallHalfWidth(wall, height) -> np.float32:
    # std::fmax(float, float) is the float overload, so the whole expression stays in float
    denom = f32(max(f32(wall.tree_height - wall.trunk_height), f32(0.01)))
    fraction = min(max(f32(f32(f32(height) - wall.trunk_height) / denom), f32(0)), f32(1))
    width = f32(apple_fruitingwall_wall_width_base
                + f32(f32(apple_fruitingwall_wall_width_top - apple_fruitingwall_wall_width_base) * fraction))
    return f32(f32(f32(f32(0.5) * width) * wall.row_spacing) / apple_fruitingwall_reference_row_spacing)


def appleFruitingWallIsStub(shoot) -> bool:
    return len(shoot.phytomers) == 1 and not shoot.meristem_is_alive


class AppleFruitingWallBranch:
    __slots__ = ("leader_node", "stubID", "branchID")

    def __init__(self, leader_node, stubID, branchID):
        self.leader_node, self.stubID, self.branchID = int(leader_node), int(stubID), int(branchID)


@ports("appleFruitingWallBranches")
def appleFruitingWallBranches(pa, leader) -> list:
    plantID = leader.plantID
    tree = pa.plant_instances[plantID].shoot_tree
    branches = []
    for childID in pa.getChildShootIDs(plantID, int(leader.ID)):
        if pa.isShootPruned(plantID, childID) or tree[childID].shoot_type_label == "spur":
            continue
        branch = AppleFruitingWallBranch(tree[childID].parent_node_index, NO_ID, childID)
        depth = 0
        while depth < 16 and branch.branchID != NO_ID:
            shoot = tree[branch.branchID]
            if not appleFruitingWallIsStub(shoot):
                break
            branch.stubID = branch.branchID
            branch.branchID = NO_ID
            for regrownID in pa.getChildShootIDs(plantID, branch.stubID):
                if not pa.isShootPruned(plantID, regrownID) and tree[regrownID].shoot_type_label != "spur":
                    branch.branchID = regrownID
                    break
            depth += 1
        branches.append(branch)
    return branches


def _z(v):
    return f32(np.asarray(v, f32)[2])


@ports("appleFruitingWallWinterPruning")
def appleFruitingWallWinterPruning(leader, wall):
    """AS:1333-1469."""
    from ..seasons import pruneBranch
    pa = leader.plantarchitecture_ptr
    plantID = leader.plantID
    tree = pa.plant_instances[plantID].shoot_tree
    base = np.asarray(pa.getPlantBasePosition(plantID), f32)
    z_cap = f32(f32(base[2] + wall.tree_height) + apple_fruitingwall_height_tolerance)
    row_half_extent = f32(f32(f32(0.5) * wall.tree_spacing) + apple_fruitingwall_row_overlap)

    branches = appleFruitingWallBranches(pa, leader)

    if leader.meristem_is_alive and _z(leader.shoot_internode_vertices[-1][-1]) > z_cap:
        cut_node = len(leader.shoot_internode_vertices)
        for node in range(len(leader.shoot_internode_vertices)):
            if _z(leader.shoot_internode_vertices[node][-1]) > z_cap:
                cut_node = node
                break
        top_branch_node = 0
        top_branch_found = False
        for entry in branches:
            base_height = f32(_z(leader.shoot_internode_vertices[entry.leader_node][-1]) - base[2])
            if (entry.branchID != NO_ID and entry.leader_node + 1 < cut_node
                    and base_height >= f32(wall.tree_height - apple_fruitingwall_heading_window)
                    and (not top_branch_found or entry.leader_node > top_branch_node)):
                top_branch_node = entry.leader_node
                top_branch_found = True
        if top_branch_found:
            cut_node = top_branch_node + 1
        if 1 <= cut_node < len(leader.phytomers):
            pruneBranch(pa, plantID, int(leader.ID), cut_node)
        leader.meristem_is_alive = False

    for shootID in pa.getAllShootIDs(plantID):
        if pa.isShootPruned(plantID, shootID) or shootID == int(leader.ID):
            continue
        shoot = tree[shootID]
        hedged = shoot.parent_shoot_ID >= 0 and shoot.shoot_type_label != "spur"
        for node in range(len(shoot.shoot_internode_vertices)):
            tip = np.asarray(shoot.shoot_internode_vertices[node][-1], f32)
            outside = tip[2] > z_cap
            if hedged:
                # std::fabs(float) is the float overload here (<cmath>), compared with a float
                outside = (outside or abs(f32(tip[0] - base[0])) > appleFruitingWallHalfWidth(wall, f32(tip[2] - base[2]))
                           or abs(f32(tip[1] - base[1])) > row_half_extent)
            if outside:
                pruneBranch(pa, plantID, shootID, node)
                break

    leader_tip_z = _z(leader.shoot_internode_vertices[-1][-1])
    renewals_this_winter = sum(1 for b in branches if b.branchID == NO_ID)
    renewal_candidates = []

    for entry in branches:
        if entry.branchID == NO_ID:
            continue
        branch = tree[entry.branchID]
        branch_base_height = f32(_z(leader.shoot_internode_vertices[entry.leader_node][-1]) - base[2])

        if branch_base_height < f32(wall.trunk_height - apple_fruitingwall_low_feather_tolerance):
            pruneBranch(pa, plantID, entry.branchID if entry.stubID == NO_ID else entry.stubID, 0)
            continue

        if (leader.meristem_is_alive and branch.dormancy_cycles == 1
                and f32(leader_tip_z - f32(branch_base_height + base[2])) < apple_fruitingwall_competitor_zone
                and len(branch.phytomers) > apple_fruitingwall_competitor_nodes_kept):
            pruneBranch(pa, plantID, entry.branchID, apple_fruitingwall_competitor_nodes_kept)
            branch.meristem_is_alive = False
            continue

        branch_radius = f32(branch.shoot_internode_radii[0][0])
        leader_radius = f32(leader.shoot_internode_radii[entry.leader_node][-1])
        too_thick_for_leader = (branch.dormancy_cycles >= apple_fruitingwall_branch_ratio_rule_age_min
                                and branch_radius > f32(apple_fruitingwall_branch_diameter_ratio_max * leader_radius))
        if len(branch.phytomers) > 1 and (too_thick_for_leader
                                          or f32(f32(2) * branch_radius) > apple_fruitingwall_branch_diameter_max
                                          or branch.dormancy_cycles >= apple_fruitingwall_branch_age_max):
            renewal_candidates.append((branch_radius, entry.branchID))

        if branch.calculateShootLength() > apple_fruitingwall_branch_length_max:
            cutShootToLength(branch, apple_fruitingwall_branch_length_headed)

        for sideID in pa.getChildShootIDs(plantID, entry.branchID):
            if pa.isShootPruned(plantID, sideID):
                continue
            side = tree[sideID]
            if side.shoot_type_label == "spur":
                continue
            side_radius = f32(side.shoot_internode_radii[0][0])
            branch_radius_at_side = f32(branch.shoot_internode_radii[side.parent_node_index][-1])
            side_axis = side.phytomers[0].getInternodeAxisVector(f32(0.5))
            side_length = side.calculateShootLength()
            if ((side.dormancy_cycles >= 2 and side_radius > f32(f32(0.5) * branch_radius_at_side))
                    or (f32(side_axis[2]) > f32(0.5) and side_length > apple_fruitingwall_upright_side_shoot_length_max)):
                pruneBranch(pa, plantID, sideID, 0)
            elif side_length > apple_fruitingwall_side_shoot_length_max:
                cutShootToLength(side, apple_fruitingwall_side_shoot_length_headed)

    from ..seasons import std_sort
    renewal_candidates = std_sort(renewal_candidates, lambda a, b: a[0] > b[0])
    for radius, branchID in renewal_candidates:
        if renewals_this_winter >= apple_fruitingwall_renewals_per_winter:
            break
        pruneBranch(pa, plantID, branchID, 1)
        tree[branchID].meristem_is_alive = False
        renewals_this_winter += 1


@ports("appleFruitingWallSummerTraining")
def appleFruitingWallSummerTraining(leader):
    """AS:1472-1495."""
    pa = leader.plantarchitecture_ptr
    plantID = leader.plantID
    tree = pa.plant_instances[plantID].shoot_tree
    for entry in appleFruitingWallBranches(pa, leader):
        if entry.branchID != NO_ID or entry.stubID == NO_ID:
            continue
        stub = tree[entry.stubID]
        for petiole in stub.phytomers[0].axillary_vegetative_buds:
            for vbud in petiole:
                if _no_shoot(vbud):
                    vbud.shoot_type_label = "renewal"
                    vbud.state = BUD_ACTIVE
    if not leader.meristem_is_alive:
        releaseBudsBelowCut(leader, len(leader.phytomers), apple_fruitingwall_buds_released_at_heading, "branch")


@AppleFruitingWallPhytomerCreationFunction.implement
@ports("AppleFruitingWallPhytomerCreationFunction")
def _apple_fruitingwall_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes,
                                          plant_age):
    """AS:1497-1511."""
    ApplePhytomerCreationFunction(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age)
    if shoot_node_index == 0 and phytomer.parent_shoot_ptr.parent_shoot_ID >= 0:
        phytomer.setVegetativeBudState(BUD_DEAD)
        _mark_spur_considered(phytomer)


@AppleFruitingWallPhytomerCallbackFunction.implement
@ports("AppleFruitingWallPhytomerCallbackFunction")
def _apple_fruitingwall_phytomer_callback(phytomer):
    """AS:1513-1529."""
    applePhytomerCallback(phytomer, apple_fruitingwall_leader_spur_fraction
                          if phytomer.parent_shoot_ptr.parent_shoot_ID < 0 else apple_fruitingwall_spur_fraction)
    leader = phytomer.parent_shoot_ptr
    if leader.parent_shoot_ID >= 0 or not leader.phytomers or phytomer is not leader.phytomers[-1]:
        return
    if leader.isdormant:
        appleFruitingWallWinterPruning(leader, appleFruitingWallDimensions(leader.plantarchitecture_ptr, leader.plantID))
    else:
        appleFruitingWallSummerTraining(leader)


__all__ = ["initializeAppleTreeShoots", "buildAppleTree", "initializeAppleFruitingWallShoots",
           "buildAppleFruitingWall", "APPLE_FRUITINGWALL_PARAMETERS"]
