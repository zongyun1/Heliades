# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:3207-3331 and src/Assets.cpp:1650-1683
# (Copyright (C) 2016-2026 Brian Bailey).
"""Capsicum (``initializeCapsicumShoots``, ``buildCapsicumPlant`` and its phytomer / fruit functions).

Line for line with the C++, literals with their C++ types (see ``library/tomato.py``). Differences from the
tomato path that the shared code had to grow (W4-A):

* one simple leaf per petiole (``leaves_per_petiole = 1``, the single-leaf branch of ``orientLeaf``), no petiolule,
  a vestigial petiole (0.1 mm long and thick) with constant curvature 0;
* ``CapsicumPhytomerCreationFunction`` removes the leaves of the first six main-stem nodes (``Phytomer::removeLeaf``);
  the XML then writes ``<shed_petiole>`` records;
* two shoot types, ``mainstem`` and ``secondary``. ``secondary`` takes ``phytomer_parameters_secondary``, a copy
  made *before* the creation function is set, so secondary phytomers have no creation function;
* distributed peduncle pitch, inflorescence roll and fruit scale, 10 unique inflorescence prototypes, and
  ``CapsicumFruitPrototype``, which draws a ``randn`` from the Context per prototype (W3-D builds them).

RNG at load: four identifier draws, as for the tomato (the ``PhytomerParameters`` copy and the
``ShootParameters`` copies draw nothing: ``LeafParameters::operator=`` copies the prototype with ``duplicate``).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, make_RGBcolor
from .cherrytomato import _hook
from .tomato import base_rotation_draws

f32 = np.float32

CapsicumPhytomerCreationFunction = _hook("CapsicumPhytomerCreationFunction")    # AS:1662
CapsicumFruitPrototype = _hook("CapsicumFruitPrototype")                        # AS:1650
AlmondFlowerPrototype = _hook("AlmondFlowerPrototype")                          # AS (almond); W3-D / almond port


@ports("PlantArchitecture::initializeCapsicumShoots")
def initializeCapsicumShoots(pa):
    from ..params import ShootParameters
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "CapsicumLeaf.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.45)
    leaf_prototype.midrib_fold_fraction = f32(0.1)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.15, f32(-0.05))
    leaf_prototype.lateral_curvature = f32(-0.15)
    leaf_prototype.wave_period = f32(0.35)
    leaf_prototype.wave_amplitude = f32(0.0)
    leaf_prototype.subdivisions = 5
    leaf_prototype.unique_prototypes = 5

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 10
    pp.internode.phyllotactic_angle.uniformDistribution(137.5 - 10, 137.5 + 10)
    pp.internode.radius_initial = 0.001
    pp.internode.color = make_RGBcolor(0.213, 0.270, 0.056)
    pp.internode.length_segments = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(-60, -40)
    pp.petiole.radius = 0.0001
    pp.petiole.length = 0.0001
    pp.petiole.taper = 1
    pp.petiole.curvature = 0
    pp.petiole.color = pp.internode.color
    pp.petiole.length_segments = 1

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch = 0
    pp.leaf.yaw = 10
    pp.leaf.roll = 0
    pp.leaf.prototype_scale.uniformDistribution(0.12, 0.15)
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    pp.peduncle.length = 0.01
    pp.peduncle.radius = 0.001
    pp.peduncle.pitch.uniformDistribution(10, 30)
    pp.peduncle.roll = 0
    pp.peduncle.curvature = -700
    pp.peduncle.color = pp.internode.color
    pp.peduncle.length_segments = 3
    pp.peduncle.radial_subdivisions = 6

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 20
    pp.inflorescence.roll.uniformDistribution(-30, 30)
    pp.inflorescence.flower_prototype_scale = 0.025
    pp.inflorescence.flower_prototype_function = AlmondFlowerPrototype
    pp.inflorescence.fruit_prototype_scale.uniformDistribution(0.12, 0.16)
    pp.inflorescence.fruit_prototype_function = CapsicumFruitPrototype
    pp.inflorescence.fruit_gravity_factor_fraction = 0.9
    pp.inflorescence.unique_prototypes = 10

    pp_secondary = pp.copy()                                                     # copy constructor, no draw

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.phytomer_parameters.phytomer_creation_function = CapsicumPhytomerCreationFunction

    sp.max_nodes = 30
    sp.insertion_angle_tip = 35
    sp.insertion_angle_decay_rate = 0
    sp.internode_length_max = 0.04
    sp.internode_length_min = 0.0
    sp.internode_length_decay_rate = 0
    sp.base_roll = 90
    sp.base_yaw.uniformDistribution(-20, 20)
    sp.gravitropic_curvature = 300
    sp.tortuosity = 75

    sp.phyllochron_min = 3
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(2.0)
    sp.vegetative_bud_break_time = 30
    sp.vegetative_bud_break_probability_min = 0.8
    sp.vegetative_bud_break_probability_decay_rate = 0
    sp.flower_bud_break_probability = 0.5
    sp.fruit_set_probability = 0.05
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = True

    sp.defineChildShootTypes(["secondary"], [1.0])

    pa.defineShootType("mainstem", sp)

    sp_secondary = sp.copy()                                                     # copy constructor, no draw
    sp_secondary.phytomer_parameters = pp_secondary                              # no creation function
    sp_secondary.max_nodes = 7
    sp_secondary.phyllochron_min = 6
    sp_secondary.vegetative_bud_break_probability_min = 0.06

    pa.defineShootType("secondary", sp_secondary)


@ports("PlantArchitecture::buildCapsicumPlant")
def buildCapsicumPlant(pa, base_position):
    """PL:3315-3331."""
    if not pa.shoot_types:
        initializeCapsicumShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    base_rotation = base_rotation_draws(pa)          # 2 randu, right to left (roll first)
    internode_length_max = pa.shoot_types["mainstem"].internode_length_max.val()
    pa.addBaseStemShoot(plantID, 1, base_rotation, 0.002, internode_length_max, 0.01, 0.01, 0, "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, 75, 14, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


@CapsicumPhytomerCreationFunction.implement
@ports("CapsicumPhytomerCreationFunction")
def _capsicum_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:1662-1683. All float: ``0.4f * shoot_node_index`` converts the uint to float, ``std::min(float, float)``."""
    from ..params import BUD_DEAD
    if shoot_node_index < 6 and phytomer.rank == 0:
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.setFloralBudState(BUD_DEAD)
        phytomer.removeLeaf()
    if phytomer.rank >= 2:
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.setFloralBudState(BUD_DEAD)
    leaf_scale = min(f32(1), f32(f32(0.6) + f32(f32(f32(0.4) * f32(shoot_node_index)) / f32(5))))
    phytomer.scaleLeafPrototypeScale(leaf_scale)
    if phytomer.rank == 0:
        inode_scale = min(f32(1), f32(f32(0.05) + f32(f32(f32(0.95) * f32(plant_age)) / f32(15))))
        phytomer.scaleInternodeMaxLength(inode_scale)


@CapsicumFruitPrototype.implement
@ports("CapsicumFruitPrototype")
def _capsicum_fruit_prototype(context, subdivisions, build_parameters=None):
    """AS:1650-1660: ``randn() < 0.4`` picks the green or the red fruit OBJ (one Context ``randn`` per prototype),
    loaded Z-up at the origin (height 0: unscaled, ``nullrotation``, ``RGB::black``), then ``addPolymeshObject``.
    Not yet oracle-checked: no capsicum fruit sets before day 80 (see migration/notes/W4-A.md)."""
    from ..leaf_prototype import _try_plugin_asset
    if float(context.randn()) < 0.4:
        obj_file = _try_plugin_asset("assets/obj/CapsicumFruit_green.obj")
    else:
        obj_file = _try_plugin_asset("assets/obj/CapsicumFruit_red.obj")
    uuids = context.load_obj(obj_file, origin=(0.0, 0.0, 0.0), height=0.0, rotation=(0.0, 0.0),
                             color=(0.0, 0.0, 0.0), upaxis="ZUP", silent=True)
    return context.add_polymesh_object(uuids)
