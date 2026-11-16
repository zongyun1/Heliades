# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:6226-6359 and src/Assets.cpp:4111-4140
# (Copyright (C) 2016-2026 Brian Bailey).
"""Cherry tomato (``initializeCherryTomatoShoots``, ``buildCherryTomatoPlant`` and its phytomer functions).

Line for line with the C++, literals with their C++ types (see ``library/tomato.py``). Differences from the
tomato path that the shared code had to grow (W4-A):

* a ``phytomer_callback_function`` (``CherryTomatoPhytomerCallbackFunction``) run for every phytomer in every
  ``advanceTime`` sub-step: it prunes the leaves below 1 m from day 101 on (``Phytomer::removeLeaf``);
* ``leaf.roll`` is distributed (one Context draw per leaflet, already on the ported path);
* ``determinate_shoot_growth = false``;
* two shoot types, ``mainstem`` and ``lateral`` (``lateral`` buds never break: probability 0, still a draw).

The flower and fruit prototypes are the tomato's (``TomatoFlowerPrototype`` / ``TomatoFruitPrototype`` hooks).

RNG at load: four identifier draws, as for the tomato (``ShootParameters`` copies draw nothing).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, make_RGBcolor
from .tomato import base_rotation_draws

f32 = np.float32


def _hook(name):
    """The shared hook of that name (created once, so every crop that uses it sees the same pointer)."""
    h = getattr(hooks, name, None)
    if h is None:
        h = hooks.Hook(name)
        setattr(hooks, name, h)
    return h


CherryTomatoPhytomerCreationFunction = _hook("CherryTomatoPhytomerCreationFunction")    # AS:4111
CherryTomatoPhytomerCallbackFunction = _hook("CherryTomatoPhytomerCallbackFunction")    # AS:4127


@ports("PlantArchitecture::initializeCherryTomatoShoots")
def initializeCherryTomatoShoots(pa):
    from ..params import ShootParameters
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "TomatoLeaf_blade.png"
    leaf_prototype.leaf_aspect_ratio = f32(f32(0.6) / f32(0.9031))
    leaf_prototype.midrib_fold_fraction = f32(0.1)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.3, f32(-0.15))
    leaf_prototype.lateral_curvature = f32(-0.8)
    leaf_prototype.wave_period = f32(0.35)
    leaf_prototype.wave_amplitude = f32(0.08)
    leaf_prototype.subdivisions = 7
    leaf_prototype.unique_prototypes = 5
    leaf_prototype.build_petiolule = True
    leaf_prototype.petiolule_length = {0: f32(0.07)}

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 5
    pp.internode.phyllotactic_angle.uniformDistribution(137.5 - 40, 137.5 + 40)
    pp.internode.radius_initial = 0.001
    pp.internode.color = make_RGBcolor(0.2495, 0.3162, 0.0657)
    pp.internode.length_segments = 1
    pp.internode.radial_subdivisions = 14

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(45, 60)
    pp.petiole.radius = 0.0025
    pp.petiole.length = 0.25
    pp.petiole.taper = 0.7
    pp.petiole.curvature.uniformDistribution(-250, 0)
    pp.petiole.color = make_RGBcolor(0.32, 0.37, 0.12)
    pp.petiole.length_segments = 5

    pp.leaf.leaves_per_petiole = 9
    pp.leaf.pitch.uniformDistribution(-30, 5)
    pp.leaf.yaw = 10
    pp.leaf.roll.uniformDistribution(-20, 20)
    pp.leaf.leaflet_offset = 0.22
    pp.leaf.leaflet_scale = 0.9
    pp.leaf.prototype_scale.uniformDistribution(f32(f32(0.12) * f32(0.9031)), f32(f32(0.17) * f32(0.9031)))
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    pp.peduncle.length = 0.2
    pp.peduncle.radius = 0.0015
    pp.peduncle.pitch = 20
    pp.peduncle.roll = 0
    pp.peduncle.curvature = -1000
    pp.peduncle.color = pp.internode.color
    pp.peduncle.length_segments = 5
    pp.peduncle.radial_subdivisions = 8

    pp.inflorescence.flowers_per_peduncle = 6
    pp.inflorescence.flower_offset = 0.15
    pp.inflorescence.pitch = 80
    pp.inflorescence.roll = 180
    pp.inflorescence.flower_prototype_scale = 0.05
    pp.inflorescence.flower_prototype_function = hooks.TomatoFlowerPrototype
    pp.inflorescence.fruit_prototype_scale = 0.1
    pp.inflorescence.fruit_prototype_function = hooks.TomatoFruitPrototype
    pp.inflorescence.fruit_gravity_factor_fraction = 0.2

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.phytomer_parameters.phytomer_creation_function = CherryTomatoPhytomerCreationFunction
    sp.phytomer_parameters.phytomer_callback_function = CherryTomatoPhytomerCallbackFunction

    sp.max_nodes = 100
    sp.insertion_angle_tip = 30
    sp.insertion_angle_decay_rate = 0
    sp.internode_length_max = 0.04
    sp.internode_length_min = 0.0
    sp.internode_length_decay_rate = 0
    sp.base_roll = 90
    sp.base_yaw.uniformDistribution(-20, 20)
    sp.gravitropic_curvature = 800
    sp.tortuosity = 37.5

    sp.phyllochron_min = 4
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(2.0)
    sp.vegetative_bud_break_time = 40
    sp.vegetative_bud_break_probability_min = 0.05
    sp.vegetative_bud_break_probability_decay_rate = 0.
    sp.flower_bud_break_probability = 0.5
    sp.fruit_set_probability = 0.9
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False

    sp.defineChildShootTypes(["lateral"], [1.0])

    pa.defineShootType("mainstem", sp)

    sp_lateral = sp.copy()                                                       # copy constructor, no draw
    sp_lateral.vegetative_bud_break_probability_min = 0
    sp_lateral.vegetative_bud_break_probability_max = 0

    pa.defineShootType("lateral", sp_lateral)


@ports("PlantArchitecture::buildCherryTomatoPlant")
def buildCherryTomatoPlant(pa, base_position):
    """PL:6341-6359."""
    if not pa.shoot_types:
        initializeCherryTomatoShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    base_rotation = base_rotation_draws(pa)          # 2 randu, right to left (roll first)
    pa.addBaseStemShoot(plantID, 1, base_rotation, 0.002, 0.06, 0.01, 0.01, 0, "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, 50, 10, 5, 30, 1000)
    pa.plant_instances[plantID].max_age = f32(175)
    return plantID


@CherryTomatoPhytomerCreationFunction.implement
@ports("CherryTomatoPhytomerCreationFunction")
def _cherry_tomato_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:4111-4125. ``fmin(1.f, 0.7 + 0.3 * plant_age / 15.f)``: left to right, ``0.3 * plant_age`` is double,
    so the division and the sum are double too; the double ``fmin``, then narrowed by the assignment."""
    from ..params import BUD_DEAD
    if shoot_node_index < 8 or phytomer.rank > 1:
        phytomer.setFloralBudState(BUD_DEAD)
        phytomer.setVegetativeBudState(BUD_DEAD)
    age = float(f32(plant_age))
    leaf_scale = f32(min(1.0, 0.7 + 0.3 * age / 15.0))
    phytomer.scaleLeafPrototypeScale(leaf_scale)
    inode_scale = f32(min(1.0, 0.7 + 0.3 * age / 10.0))
    phytomer.scaleInternodeMaxLength(inode_scale)


@CherryTomatoPhytomerCallbackFunction.implement
@ports("CherryTomatoPhytomerCallbackFunction")
def _cherry_tomato_phytomer_callback(phytomer):
    """AS:4127-4140: from day 101 remove every leaf whose node sits below 1 m."""
    pruning_height = f32(1)
    pruning_day = f32(101)
    plant_age = phytomer.parent_shoot_ptr.plantarchitecture_ptr.plant_instances[phytomer.plantID].current_age
    if phytomer.hasLeaf() and plant_age >= pruning_day:
        height = phytomer.getInternodeNodePositions()[0][2]
        if height < pruning_height:
            phytomer.removeLeaf()
