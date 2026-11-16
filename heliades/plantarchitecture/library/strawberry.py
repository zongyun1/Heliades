# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:5900-6012 and src/Assets.cpp:4068-4078
# (Copyright (C) 2016-2026 Brian Bailey).
"""Strawberry shoot parameters and builder (``initializeStrawberryShoots``, ``buildStrawberryPlant``).

Line for line with the C++, literals with their C++ types (see ``tomato.py``). In 1.3.98 the strawberry is
a single ``mainstem`` shoot type (a crown of 2/5-spiral phytomers with trifoliate leaves): no stolon shoot
type, no phytomer creation function, no build parameters. Its flower and fruit are OBJ prototypes
(``StrawberryFlowerPrototype`` / ``StrawberryFruitPrototype``), built by the inflorescence machinery (W3-D).

RNG on load: the same four identifier draws as the tomato (``LeafPrototype(gen)``, ``PhytomerParameters(gen)``
x2, ``leaf.prototype = ...``). Build: two ``randu`` for the base rotation, right to left (roll first).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, make_RGBcolor

f32 = np.float32


def named_hook(name: str) -> hooks.Hook:
    """The ``hooks.<name>`` slot, created on first use (so the shared ``hooks.py`` needs no edit). Identity
    matters (prototype caches are keyed by the function pointer), so every caller gets the same object."""
    h = getattr(hooks, name, None)
    if h is None:
        h = hooks.Hook(name)
        setattr(hooks, name, h)
    return h


@ports("PlantArchitecture::initializeStrawberryShoots")
def initializeStrawberryShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "StrawberryLeaf.png"
    leaf_prototype.leaf_aspect_ratio = f32(1)
    leaf_prototype.midrib_fold_fraction = f32(0.2)
    leaf_prototype.longitudinal_curvature = f32(0.15)
    leaf_prototype.lateral_curvature = f32(-0.35)
    leaf_prototype.wave_period = f32(0.4)
    leaf_prototype.wave_amplitude = f32(0.03)
    leaf_prototype.subdivisions = 6
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 10
    pp.internode.phyllotactic_angle.uniformDistribution(144 - 10, 144 + 10)
    pp.internode.radius_initial = 0.001
    pp.internode.color = make_RGBcolor(0.15, 0.2, 0.1)
    pp.internode.length_segments = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(10, 45)
    pp.petiole.radius = 0.0025
    pp.petiole.length.uniformDistribution(0.15, 0.35)
    pp.petiole.taper = 0.5
    pp.petiole.curvature.uniformDistribution(-150, -50)
    pp.petiole.color = make_RGBcolor(0.18, 0.23, 0.1)
    pp.petiole.length_segments = 5

    pp.leaf.leaves_per_petiole = 3
    pp.leaf.pitch.uniformDistribution(-35, 0)
    pp.leaf.yaw = -30
    pp.leaf.roll = -30
    pp.leaf.leaflet_offset = 0.01
    pp.leaf.leaflet_scale = 1.0
    pp.leaf.prototype_scale = 0.12
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    pp.peduncle.length.uniformDistribution(0.16, 0.2)
    pp.peduncle.radius = 0.0018
    pp.peduncle.pitch.uniformDistribution(35, 55)
    pp.peduncle.roll = 90
    pp.peduncle.curvature = -150
    pp.peduncle.length_segments = 5
    pp.peduncle.radial_subdivisions = 6
    pp.peduncle.color = pp.petiole.color

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.flower_offset = 0.2
    pp.inflorescence.pitch = 70
    pp.inflorescence.roll = 90
    pp.inflorescence.flower_prototype_scale = 0.04
    pp.inflorescence.flower_prototype_function = named_hook("StrawberryFlowerPrototype")
    pp.inflorescence.fruit_prototype_scale = 0.085
    pp.inflorescence.fruit_prototype_function = named_hook("StrawberryFruitPrototype")
    pp.inflorescence.fruit_gravity_factor_fraction = 0.65

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp

    sp.max_nodes = 15
    sp.insertion_angle_tip = 40
    sp.insertion_angle_decay_rate = 0
    sp.internode_length_max = 0.015
    sp.internode_length_decay_rate = 0
    sp.internode_length_min = 0.0
    sp.base_roll = 90
    sp.base_yaw.uniformDistribution(-20, 20)
    sp.gravitropic_curvature.uniformDistribution(-10, 0)
    sp.tortuosity = 0

    sp.phyllochron_min = 2
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(2)
    sp.vegetative_bud_break_time = 15
    sp.vegetative_bud_break_probability_min = 0.
    sp.vegetative_bud_break_probability_decay_rate = -0.5
    sp.flower_bud_break_probability = 1
    sp.fruit_set_probability = 0.3
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = True

    sp.defineChildShootTypes(["mainstem"], [1.0])

    pa.defineShootType("mainstem", sp)


@ports("PlantArchitecture::buildStrawberryPlant")
def buildStrawberryPlant(pa, base_position):
    """PL:5996-6012."""
    from .tomato import base_rotation_draws
    if not pa.shoot_types:
        initializeStrawberryShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    base_rotation = base_rotation_draws(pa)          # make_AxisRotation(0, randu(0, 2pi), randu(0, 2pi)), right to left
    pa.addBaseStemShoot(plantID, 1, base_rotation, 0.001, 0.004, 0.01, 0.01, 0, "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, 40, 5, 5, 30, 1000)
    pa.plant_instances[plantID].max_age = f32(120)
    return plantID


# ---- organ prototypes (AS:4068-4078), built by the inflorescence machinery (W3-D) ----------------

def obj_prototype(scene, filename, height=0.0):
    """``loadOBJ(resolvePluginAsset(.../assets/obj/<filename>), (0,0,0), height, nullrotation, black, "ZUP", true)``
    then ``addPolymeshObject``. Height 0 keeps the OBJ's own size."""
    from ..leaf_prototype import _try_plugin_asset
    from ..random_parameter import PlantArchitectureError
    path = _try_plugin_asset("assets/obj/" + filename)
    if not path:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture): Could not resolve plugin asset assets/obj/{filename}.")
    uuids = scene.load_obj(path, (0.0, 0.0, 0.0), f32(height), (0.0, 0.0), (0.0, 0.0, 0.0), "ZUP", True)
    return scene.add_polymesh_object(list(uuids))


@named_hook("StrawberryFlowerPrototype").implement
@ports("StrawberryFlowerPrototype")
def StrawberryFlowerPrototype(scene, subdivisions, flower_is_open, build_parameters=None):
    return obj_prototype(scene, "StrawberryFlower.obj")


@named_hook("StrawberryFruitPrototype").implement
@ports("StrawberryFruitPrototype")
def StrawberryFruitPrototype(scene, subdivisions, build_parameters=None):
    return _obj_prototype(scene, "StrawberryFruit.obj")
