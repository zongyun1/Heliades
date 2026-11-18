# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:63-111, 924-2129 and
# src/Assets.cpp:928-938, 1019-1047 (Copyright (C) 2016-2026 Brian Bailey).
"""Almond trees (W5-D): ``almond`` (Nonpareil), ``almond_aldrich``, ``almond_independence``, ``almond_wood_colony``.

* ``initializeAlmondTree*Shoots``: trunk / scaffold / proleptic / sylleptic (+ ``spur`` for Nonpareil);
* ``buildAlmondTree*``: a prescribed, leaning trunk (``almondTrunkPath``, PL:79-111) with a short phytomer on
  top, its buds and leaves removed, scaffolds attached one per trunk node below the top, the plant made dormant
  (each scaffold gets its terminal floral buds), phenology 90 / -1 / 3 / 7 / 20 / 275 (one 365-day season);
* the assets ``AlmondFruitPrototype``, ``AlmondFlowerPrototype`` (OBJ meshes), ``AlmondPhytomerCreationFunction``
  (short laterals become spur-like shoots), ``AlmondPhytomerCallbackFunction`` (empty) and
  ``AlmondSpurPhytomerCallbackFunction`` (spurs from unbroken buds, see ``trees.growSpursFromUnbrokenBuds``).

GCC evaluates function arguments right to left: in ``addChildShoot(..., randu(7, 9), make_AxisRotation(pitch,
yaw(randu), 0), ...)`` the yaw is drawn before the node count, and in ``addComponent(randu(a), L * randu(b),
randu(c))`` the phase is drawn first.
"""
from __future__ import annotations

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor
from .build_parameters import getBuildParameterIntegerValue, getBuildParameterValue
from .cherrytomato import _hook
from .grapevine import BoundedWander
from .trees import (almond_trunk_internode_length, checkScaffoldsFitTrunk, kill_trunk_buds, scaffold_yaw,
                    tree_num_scaffolds, tree_scaffold_angle, tree_trunk_height, trunk_node_count,
                    growSpursFromUnbrokenBuds)

f32 = np.float32
PI_F = f32(np.pi)

AlmondFruitPrototype = _hook("AlmondFruitPrototype")                              # AS:928
AlmondFlowerPrototype = _hook("AlmondFlowerPrototype")                            # AS:934 (shared with capsicum)
AlmondPhytomerCreationFunction = _hook("AlmondPhytomerCreationFunction")          # AS:1023
AlmondPhytomerCallbackFunction = _hook("AlmondPhytomerCallbackFunction")          # AS:1042
AlmondSpurPhytomerCallbackFunction = _hook("AlmondSpurPhytomerCallbackFunction")  # AS:1045

almond_spur_fraction = f32(0.2)                                                   # AS:1020


# ---- trunk path (PL:63-111) -------------------------------------------------------------------------------

def _add_component(ctx, wander, amp_range, wavelength_scale, wl_range, phase_range):
    """``addComponent(randu(amp), [scale *] randu(wl), randu(phase))``, arguments right to left."""
    phase = ctx.randu(f32(phase_range[0]), f32(phase_range[1]))
    wl = ctx.randu(f32(wl_range[0]), f32(wl_range[1]))
    if wavelength_scale is not None:
        wl = f32(f32(wavelength_scale) * wl)
    amplitude = ctx.randu(f32(amp_range[0]), f32(amp_range[1]))
    wander.addComponent(amplitude, wl, phase)


@ports("almondTrunkPath")
def almondTrunkPath(ctx, base_position, trunk_nodes: int, internode_length, radius):
    """PL:79-111. Returns (node_positions, node_radii)."""
    base_position = np.asarray(base_position, f32).reshape(3)
    internode_length, radius = f32(internode_length), f32(radius)
    trunk_height = f32(f32(trunk_nodes) * internode_length)
    lean = V.deg2rad(ctx.randu(f32(1.5), f32(11)))
    lean_azimuth = ctx.randu(f32(0), f32(f32(2) * PI_F))
    two_pi = f32(f32(2) * PI_F)
    sway_x, sway_y = BoundedWander(), BoundedWander()
    for sway in (sway_x, sway_y):
        _add_component(ctx, sway, (0.01, 0.075), trunk_height, (1.5, 3.0), (0.0, two_pi))
        _add_component(ctx, sway, (0.002, 0.008), None, (0.3, 0.6), (0.0, two_pi))
    tan_lean = V.tanf(lean)
    cos_az, sin_az = V.cosf(lean_azimuth), V.sinf(lean_azimuth)
    positions, radii = [], []
    for node in range(trunk_nodes + 1):
        height = f32(f32(node) * internode_length)
        lean_offset = f32(tan_lean * height)
        along = f32(height / trunk_height)
        bows = []
        for sway in (sway_x, sway_y):
            e0 = sway.evaluate(f32(0))
            bows.append(f32(f32(sway.evaluate(height) - e0) - f32(along * f32(sway.evaluate(trunk_height) - e0))))
        offset_x = f32(f32(lean_offset * cos_az) + bows[0])
        offset_y = f32(f32(lean_offset * sin_az) + bows[1])
        positions.append(base_position + np.array([offset_x, offset_y, height], f32))
        radii.append(radius)
    return positions, radii


# ---- shoot types ----------------------------------------------------------------------------------------------

def _almond_leaf_prototype(gen):
    lp = LeafPrototype(gen)
    lp.leaf_texture_file[0] = "AlmondLeaf.png"
    lp.leaf_aspect_ratio = f32(0.33)
    lp.midrib_fold_fraction = f32(0.1)
    lp.longitudinal_curvature = 0.05
    lp.lateral_curvature = f32(0.1)
    lp.subdivisions = 1
    lp.unique_prototypes = 1
    return lp


def _almond_phytomer_parameters(gen, internode_pitch_range):
    """The ``phytomer_parameters_almond`` block shared by the four almond models. ``internode_pitch_range``:
    ``(-7, 7)`` for a uniform distribution, or a constant (3)."""
    pp = PhytomerParameters(gen)
    if isinstance(internode_pitch_range, tuple):
        pp.internode.pitch.uniformDistribution(*internode_pitch_range)
    else:
        pp.internode.pitch = internode_pitch_range
    pp.internode.phyllotactic_angle.uniformDistribution(120, 160)
    pp.internode.radius_initial = 0.002
    pp.internode.length_segments = 1
    pp.internode.image_texture = "AlmondBark.jpg"
    pp.internode.max_floral_buds_per_petiole = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(-145, -90)
    pp.petiole.taper = 0.1
    pp.petiole.curvature = 0
    pp.petiole.length = 0.04
    pp.petiole.radius = 0.0005
    pp.petiole.length_segments = 1
    pp.petiole.radial_subdivisions = 3
    pp.petiole.color = make_RGBcolor(0.61, 0.5, 0.24)

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.roll.uniformDistribution(-10, 10)
    pp.leaf.prototype_scale = 0.12
    pp.leaf.prototype = _almond_leaf_prototype(gen)

    pp.peduncle.length = 0.002
    pp.peduncle.radius = 0.0005
    pp.peduncle.pitch = 80
    pp.peduncle.roll = 90
    pp.peduncle.length_segments = 1
    pp.petiole.radial_subdivisions = 3

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.flower_prototype_scale = 0.04
    pp.inflorescence.flower_prototype_function = AlmondFlowerPrototype
    pp.inflorescence.fruit_prototype_scale = 0.04
    pp.inflorescence.fruit_prototype_function = AlmondFruitPrototype
    return pp


def _almond_trunk(gen, pp, max_nodes, girth_area_factor, tortuosity=10, internode_length_max=0.05):
    t = ShootParameters(gen)
    t.phytomer_parameters = pp
    t.phytomer_parameters.internode.pitch = 0
    t.phytomer_parameters.internode.phyllotactic_angle = 0
    t.phytomer_parameters.internode.radius_initial = 0.005
    t.phytomer_parameters.internode.radial_subdivisions = 24
    t.max_nodes = max_nodes
    t.girth_area_factor = f32(girth_area_factor)
    t.vegetative_bud_break_probability_min = 0
    t.vegetative_bud_break_time = 0
    t.tortuosity = tortuosity
    t.internode_length_max = internode_length_max
    t.internode_length_decay_rate = 0
    t.defineChildShootTypes(["scaffold"], [1])
    return t


def _almond_proleptic_common(gen, pp, callback):
    p = ShootParameters(gen)
    p.phytomer_parameters = pp
    p.phytomer_parameters.internode.color = make_RGBcolor(0.3, 0.2, 0.2)
    p.phytomer_parameters.internode.radial_subdivisions = 5
    p.phytomer_parameters.phytomer_creation_function = AlmondPhytomerCreationFunction
    p.phytomer_parameters.phytomer_callback_function = callback
    return p


@ports("PlantArchitecture::initializeAlmondTreeShoots")
def initializeAlmondTreeShoots(pa):
    """PL:924-1156 (Nonpareil)."""
    gen = context_generator(pa.context_ptr)
    pp = _almond_phytomer_parameters(gen, (-7, 7))
    trunk = _almond_trunk(gen, pp, 101, 0.85)

    pro = _almond_proleptic_common(gen, pp, AlmondSpurPhytomerCallbackFunction)
    pro.max_nodes.uniformDistribution(20, 55)
    pro.max_nodes_per_season = 15
    pro.phyllochron_min = 1
    pro.elongation_rate_max = 0.3
    pro.girth_area_factor = f32(1.8)
    pro.vegetative_bud_break_probability_min = 0.05
    pro.vegetative_bud_break_probability_max = 0.31
    pro.vegetative_bud_break_probability_decay_rate = 0.15
    pro.vegetative_bud_break_time = 0
    pro.gravitropic_curvature = 180
    pro.tortuosity = 37.5
    pro.tortuosity_persistence_length = 0.3
    pro.insertion_angle_tip = 45
    pro.insertion_angle_decay_rate = 15
    pro.internode_length_max = 0.04
    pro.internode_length_min = 0.002
    pro.internode_length_decay_rate = 0.004
    pro.fruit_set_probability = 0.4
    pro.flower_bud_break_probability = 0.3
    pro.max_terminal_floral_buds = 3
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False
    pro.defineChildShootTypes(["proleptic", "sylleptic"], [1.0, 0.])

    syl = pro.copy()
    syl.phytomer_parameters.internode.image_texture = ""
    syl.phytomer_parameters.leaf.prototype_scale = 0.14
    syl.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20)
    syl.insertion_angle_tip = 0
    syl.insertion_angle_decay_rate = 0
    syl.phyllochron_min = 1
    syl.vegetative_bud_break_probability_min = 0.0
    syl.vegetative_bud_break_probability_max = 0.7
    syl.gravitropic_curvature = 720
    syl.internode_length_max = 0.02
    syl.tortuosity = 75
    syl.flowers_require_dormancy = True
    syl.growth_requires_dormancy = True
    syl.defineChildShootTypes(["proleptic"], [1.0])
    syl.max_nodes = 60
    syl.vegetative_bud_break_probability_decay_rate = 0.15

    sca = pro.copy()
    sca.phytomer_parameters.internode.radial_subdivisions = 10
    sca.max_nodes = 105
    sca.gravitropic_curvature = 276
    sca.tortuosity = 10
    sca.tortuosity_persistence_length = 2.5
    sca.defineChildShootTypes(["proleptic"], [1.0])
    sca.vegetative_bud_break_probability_decay_rate = 0.02
    sca.vegetative_bud_break_probability_max = 0.72
    sca.vegetative_bud_break_probability_min = 0.10
    sca.girth_area_factor = f32(0.85)

    spur = pro.copy()
    spur.max_nodes = 40
    spur.max_nodes_per_season = 4
    spur.internode_length_max = 0.003
    spur.internode_length_min = 0.003
    spur.internode_length_decay_rate = 0
    spur.vegetative_bud_break_probability_max = 0
    spur.vegetative_bud_break_probability_min = 0
    spur.vegetative_bud_break_probability_decay_rate = 0
    spur.gravitropic_curvature = 0
    spur.defineChildShootTypes(["spur"], [1.0])

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("scaffold", sca)
    pa.defineShootType("proleptic", pro)
    pa.defineShootType("sylleptic", syl)
    pa.defineShootType("spur", spur)


@ports("PlantArchitecture::initializeAlmondTreeAldrichShoots")
def initializeAlmondTreeAldrichShoots(pa):
    """PL:1232-1558."""
    gen = context_generator(pa.context_ptr)
    pp = _almond_phytomer_parameters(gen, 3)
    trunk = _almond_trunk(gen, pp, 101, 5.8)

    pro = _almond_proleptic_common(gen, pp, AlmondPhytomerCallbackFunction)
    pro.max_nodes.uniformDistribution(20, 55)
    pro.max_nodes_per_season = 15
    pro.phyllochron_min = 1
    pro.elongation_rate_max = 0.3
    pro.girth_area_factor = f32(12)
    pro.vegetative_bud_break_probability_min = 0.05
    pro.vegetative_bud_break_probability_max = 0.21
    pro.vegetative_bud_break_probability_decay_rate = 0.15
    pro.vegetative_bud_break_time = 0
    pro.gravitropic_curvature = 180
    pro.tortuosity = 37.5
    pro.tortuosity_persistence_length = 0.3
    pro.insertion_angle_tip = 45
    pro.insertion_angle_decay_rate = 5
    pro.internode_length_max = 0.04
    pro.internode_length_min = 0.002
    pro.internode_length_decay_rate = 0.004
    pro.fruit_set_probability = 0.4
    pro.flower_bud_break_probability = 0.3
    pro.max_terminal_floral_buds = 3
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False
    pro.defineChildShootTypes(["proleptic", "sylleptic"], [1.0, 0.])

    syl = pro.copy()
    syl.phytomer_parameters.internode.image_texture = ""
    syl.phytomer_parameters.leaf.prototype_scale = 0.12
    syl.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20)
    syl.insertion_angle_tip = 0
    syl.insertion_angle_decay_rate = 0
    syl.phyllochron_min = 1
    syl.vegetative_bud_break_probability_min = 0.0
    syl.vegetative_bud_break_probability_max = 0.6
    syl.gravitropic_curvature = 720
    syl.internode_length_max = 0.02
    syl.tortuosity = 75
    syl.flowers_require_dormancy = True
    syl.growth_requires_dormancy = True
    syl.defineChildShootTypes(["proleptic"], [1.0])

    sca = pro.copy()
    sca.phytomer_parameters.internode.radial_subdivisions = 10
    sca.max_nodes = 105
    sca.gravitropic_curvature = 276
    sca.internode_length_max = 0.04
    sca.tortuosity = 12.5
    sca.tortuosity_persistence_length = 2.5
    sca.defineChildShootTypes(["proleptic"], [1.0])
    sca.vegetative_bud_break_probability_max = 0.72
    sca.vegetative_bud_break_probability_decay_rate = 0.02
    sca.vegetative_bud_break_probability_min = 0.10
    sca.girth_area_factor = f32(5.8)

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("scaffold", sca)
    pa.defineShootType("proleptic", pro)
    pa.defineShootType("sylleptic", syl)


@ports("PlantArchitecture::initializeAlmondTreeIndependenceShoots")
def initializeAlmondTreeIndependenceShoots(pa):
    """PL:1651-1871."""
    gen = context_generator(pa.context_ptr)
    pp = _almond_phytomer_parameters(gen, (-7, 7))
    trunk = _almond_trunk(gen, pp, 101, 5.7)

    pro = _almond_proleptic_common(gen, pp, AlmondPhytomerCallbackFunction)
    pro.max_nodes.uniformDistribution(20, 55)
    pro.max_nodes_per_season = 15
    pro.phyllochron_min = 1
    pro.elongation_rate_max = 0.3
    pro.girth_area_factor = f32(4.5)
    pro.vegetative_bud_break_probability_min = 0.05
    pro.vegetative_bud_break_probability_max = 0.31
    pro.vegetative_bud_break_probability_decay_rate = 0.15
    pro.vegetative_bud_break_time = 0
    pro.gravitropic_curvature = 180
    pro.tortuosity = 37.5
    pro.tortuosity_persistence_length = 0.3
    pro.insertion_angle_tip = 45
    pro.insertion_angle_decay_rate = 15
    pro.internode_length_max = 0.04
    pro.internode_length_min = 0.002
    pro.internode_length_decay_rate = 0.004
    pro.fruit_set_probability = 0.4
    pro.flower_bud_break_probability = 0.3
    pro.max_terminal_floral_buds = 3
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False
    pro.defineChildShootTypes(["proleptic", "sylleptic"], [1.0, 0.])

    syl = pro.copy()
    syl.phytomer_parameters.internode.image_texture = ""
    syl.phytomer_parameters.leaf.prototype_scale = 0.14
    syl.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20)
    syl.insertion_angle_tip = 0
    syl.insertion_angle_decay_rate = 0
    syl.phyllochron_min = 1
    syl.vegetative_bud_break_probability_min = 0.0
    syl.vegetative_bud_break_probability_max = 0.7
    syl.gravitropic_curvature = 720
    syl.internode_length_max = 0.02
    syl.tortuosity = 75
    syl.flowers_require_dormancy = True
    syl.growth_requires_dormancy = True
    syl.defineChildShootTypes(["proleptic"], [1.0])
    syl.max_nodes = 60
    syl.vegetative_bud_break_probability_decay_rate = 0.15

    sca = pro.copy()
    sca.phytomer_parameters.internode.radial_subdivisions = 10
    sca.max_nodes = 90
    sca.gravitropic_curvature = 276
    sca.tortuosity = 10
    sca.tortuosity_persistence_length = 2.5
    sca.defineChildShootTypes(["proleptic"], [1.0])
    sca.vegetative_bud_break_probability_decay_rate = 0.02
    sca.vegetative_bud_break_probability_max = 0.55
    sca.vegetative_bud_break_probability_min = 0.10
    sca.girth_area_factor = f32(5.7)

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("scaffold", sca)
    pa.defineShootType("proleptic", pro)
    pa.defineShootType("sylleptic", syl)


@ports("PlantArchitecture::initializeAlmondTreeWoodColonyShoots")
def initializeAlmondTreeWoodColonyShoots(pa):
    """PL:1948-2086. Helios sets ``vegetative_bud_break_probability_max = 0.6`` on the *proleptic* type after the
    sylleptic type was copied from it (the scaffolds, copied later, inherit 0.6); reproduced as written."""
    gen = context_generator(pa.context_ptr)
    pp = _almond_phytomer_parameters(gen, 3)
    trunk = _almond_trunk(gen, pp, 15, 2.3, tortuosity=30.8, internode_length_max=0.0325)

    pro = _almond_proleptic_common(gen, pp, AlmondPhytomerCallbackFunction)
    pro.max_nodes = 20
    pro.max_nodes_per_season = 15
    pro.phyllochron_min = 1
    pro.elongation_rate_max = 0.3
    pro.girth_area_factor = f32(5.9)
    pro.vegetative_bud_break_probability_min = 0.1
    pro.vegetative_bud_break_probability_max = 0.5
    pro.vegetative_bud_break_probability_decay_rate = 0.15
    pro.vegetative_bud_break_time = 0
    pro.gravitropic_curvature = 150
    pro.tortuosity = 150
    pro.insertion_angle_tip.uniformDistribution(30, 75)
    pro.insertion_angle_decay_rate = 17.5
    pro.internode_length_max = 0.02
    pro.internode_length_min = 0.002
    pro.internode_length_decay_rate = 0.002
    pro.fruit_set_probability = 0.4
    pro.flower_bud_break_probability = 0.3
    pro.max_terminal_floral_buds = 3
    pro.flowers_require_dormancy = True
    pro.growth_requires_dormancy = True
    pro.determinate_shoot_growth = False
    pro.defineChildShootTypes(["proleptic", "sylleptic"], [1.0, 0.])

    syl = pro.copy()
    syl.phytomer_parameters.internode.image_texture = ""
    syl.phytomer_parameters.leaf.prototype_scale = 0.12
    syl.phytomer_parameters.leaf.pitch.uniformDistribution(-45, -20)
    syl.insertion_angle_tip = 0
    syl.insertion_angle_decay_rate = 0
    syl.phyllochron_min = 1
    syl.vegetative_bud_break_probability_min = 0.0
    pro.vegetative_bud_break_probability_max = 0.6            # sic: proleptic, not sylleptic
    syl.gravitropic_curvature = 600
    syl.internode_length_max = 0.02
    syl.flowers_require_dormancy = True
    syl.growth_requires_dormancy = True
    syl.defineChildShootTypes(["proleptic"], [1.0])

    sca = pro.copy()
    sca.girth_area_factor = f32(1.95)
    sca.phytomer_parameters.internode.radial_subdivisions = 10
    sca.max_nodes = 15
    sca.gravitropic_curvature = 100
    sca.internode_length_max = 0.02
    sca.tortuosity = 50
    sca.defineChildShootTypes(["proleptic"], [1.0])

    pa.defineShootType("trunk", trunk)
    pa.defineShootType("scaffold", sca)
    pa.defineShootType("proleptic", pro)
    pa.defineShootType("sylleptic", syl)


# ---- builders -------------------------------------------------------------------------------------------------

def _build_trunk(pa, plantID, base_position, trunk_nodes, internode_length, radius):
    """``almondTrunkPath`` + ``addShootFromNodePositions(..., "trunk")`` + a short phytomer on its top
    (``appendPhytomerToShoot(..., 0.01, 0.01, 1, 1)``) + :func:`trees.kill_trunk_buds`."""
    from ..shoot import addShootFromNodePositions
    positions, radii = almondTrunkPath(pa.context_ptr, base_position, trunk_nodes, internode_length, radius)
    uID_trunk = addShootFromNodePositions(pa, plantID, -1, 0, positions, radii, "trunk")
    pa.appendPhytomerToShoot(plantID, uID_trunk, pa.shoot_types["trunk"].phytomer_parameters, f32(0.01), f32(0.01),
                             f32(1), f32(1))
    kill_trunk_buds(pa, plantID, uID_trunk)
    return uID_trunk


def _finish(pa, plantID, max_age):
    from ..seasons import makePlantDormant
    makePlantDormant(pa, plantID)
    pa.setPlantPhenologicalThresholds(plantID, 90, -1, 3, 7, 20, 275)
    pa.plant_instances[plantID].max_age = f32(max_age)
    return plantID


def _build_trained_almond(pa, base_position, builder_name):
    """buildAlmondTree (PL:1158-1230) and buildAlmondTreeIndependence (PL:1873-1945): identical bodies."""
    ctx = pa.context_ptr
    bp = pa.current_build_parameters
    trunk_height = getBuildParameterValue(bp, tree_trunk_height, f32(0.78))
    num_scaffolds = int(getBuildParameterIntegerValue(bp, tree_num_scaffolds, 4))
    scaffold_angle = getBuildParameterValue(bp, tree_scaffold_angle, f32(52))
    trunk_internode_length = almond_trunk_internode_length
    trunk_nodes = trunk_node_count(trunk_height, trunk_internode_length)
    checkScaffoldsFitTrunk(builder_name, num_scaffolds, trunk_nodes, trunk_height)
    trunk_radius, scaffold_radius, scaffold_length = f32(0.015), f32(0.007), f32(0.06)

    plantID = pa.addPlantInstance(base_position, 0)
    uID_trunk = _build_trunk(pa, plantID, base_position, trunk_nodes, trunk_internode_length, trunk_radius)
    tree = pa.plant_instances[plantID].shoot_tree
    for i in range(num_scaffolds):
        pitch = f32(V.deg2rad(scaffold_angle) + ctx.randu(f32(-0.1), f32(0.1)))
        scaffold_nodes = int(ctx.randu(5, 5))
        yaw = scaffold_yaw(ctx, i, num_scaffolds)
        uID_shoot = pa.addChildShoot(plantID, uID_trunk, tree[uID_trunk].current_node_number - i - 1, scaffold_nodes,
                                     make_AxisRotation(pitch, yaw, 0), scaffold_radius, scaffold_length, f32(1), f32(1),
                                     f32(0.5), "scaffold", 0)
        tree[uID_shoot].phytomers[0].setVegetativeBudState(BUD_DEAD)
    return _finish(pa, plantID, 3000)


@ports("PlantArchitecture::buildAlmondTree")
def buildAlmondTree(pa, base_position):
    if not pa.shoot_types:
        initializeAlmondTreeShoots(pa)
    return _build_trained_almond(pa, base_position, "buildAlmondTree")


@ports("PlantArchitecture::buildAlmondTreeIndependence")
def buildAlmondTreeIndependence(pa, base_position):
    if not pa.shoot_types:
        initializeAlmondTreeIndependenceShoots(pa)
    return _build_trained_almond(pa, base_position, "buildAlmondTreeIndependence")


def _build_fixed_almond(pa, base_position, trunk_nodes, num_scaffolds, pitch_range_deg, kill_crotch_bud, max_age):
    """buildAlmondTreeAldrich (PL:1560-1649) and buildAlmondTreeWoodColony (PL:2088-2129)."""
    ctx = pa.context_ptr
    plantID = pa.addPlantInstance(base_position, 0)
    uID_trunk = _build_trunk(pa, plantID, base_position, trunk_nodes, f32(0.03), f32(0.015))
    tree = pa.plant_instances[plantID].shoot_tree
    for i in range(num_scaffolds):
        pitch = ctx.randu(V.deg2rad(pitch_range_deg[0]), V.deg2rad(pitch_range_deg[1]))
        yaw = scaffold_yaw(ctx, i, num_scaffolds)                    # make_AxisRotation(...) first (right to left)
        nodes = int(ctx.randu(7, 9))
        uID_shoot = pa.addChildShoot(plantID, uID_trunk, tree[uID_trunk].current_node_number - i - 1, nodes,
                                     make_AxisRotation(pitch, yaw, 0), f32(0.007), f32(0.06), f32(1), f32(1), f32(0.5),
                                     "scaffold", 0)
        if kill_crotch_bud:
            tree[uID_shoot].phytomers[0].setVegetativeBudState(BUD_DEAD)
    return _finish(pa, plantID, max_age)


@ports("PlantArchitecture::buildAlmondTreeAldrich")
def buildAlmondTreeAldrich(pa, base_position):
    if not pa.shoot_types:
        initializeAlmondTreeShoots(pa)               # sic (PL:1564): the Nonpareil shoots
    return _build_fixed_almond(pa, base_position, 26, 4, (46, 58), True, 3000)


@ports("PlantArchitecture::buildAlmondTreeWoodColony")
def buildAlmondTreeWoodColony(pa, base_position):
    if not pa.shoot_types:
        initializeAlmondTreeShoots(pa)               # sic (PL:2092)
    return _build_fixed_almond(pa, base_position, 14, 5, (45, 60), False, 1825)


# ---- assets (AS:928-938, 1019-1047) ---------------------------------------------------------------------------

def _load_obj_polymesh(context, asset_path):
    from .bean import load_obj_black
    return load_obj_black(context, asset_path)


@AlmondFruitPrototype.implement
@ports("AlmondFruitPrototype")
def _almond_fruit_prototype(context, subdivisions, build_parameters=None):
    return _load_obj_polymesh(context, "assets/obj/AlmondHull.obj")


@ports("AlmondFlowerPrototype")
def _almond_flower_prototype(context, subdivisions, flower_is_open, build_parameters=None):
    return _load_obj_polymesh(context, "assets/obj/AlmondFlower.obj")


if AlmondFlowerPrototype.impl is None:
    AlmondFlowerPrototype.implement(_almond_flower_prototype)


@AlmondPhytomerCreationFunction.implement
@ports("AlmondPhytomerCreationFunction")
def _almond_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:1023-1040: a lateral with short internodes (``internode_length_max < 0.01``, compared in double) is a
    spur-like shoot: buds dead, leaves 0.8x, at most 7 nodes a season (a ``spur`` shoot keeps its own cap)."""
    if float(phytomer.internode_length_max) < 0.01:
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.scaleLeafPrototypeScale(f32(0.8))
        phytomer.setFloralBudState(BUD_DEAD)
        if phytomer.parent_shoot_ptr.shoot_type_label != "spur":
            phytomer.parent_shoot_ptr.growth_state.max_nodes_per_season_growth_cap = 7


@AlmondPhytomerCallbackFunction.implement
@ports("AlmondPhytomerCallbackFunction")
def _almond_phytomer_callback(phytomer):
    return None


@AlmondSpurPhytomerCallbackFunction.implement
@ports("AlmondSpurPhytomerCallbackFunction")
def _almond_spur_phytomer_callback(phytomer):
    growSpursFromUnbrokenBuds(phytomer, almond_spur_fraction)


__all__ = ["initializeAlmondTreeShoots", "initializeAlmondTreeAldrichShoots", "initializeAlmondTreeIndependenceShoots",
           "initializeAlmondTreeWoodColonyShoots", "buildAlmondTree", "buildAlmondTreeAldrich",
           "buildAlmondTreeIndependence", "buildAlmondTreeWoodColony", "almondTrunkPath"]
