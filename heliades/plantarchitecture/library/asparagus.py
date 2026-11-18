# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:2696-2798 and src/Assets.cpp:1531-1575
# (Copyright (C) 2016-2026 Brian Bailey).
"""Asparagus (``initializeAsparagusShoots`` / ``buildAsparagusPlant``, ``AsparagusLeafPrototype``,
``AsparagusPhytomerCreationFunction``).

Line for line with the C++, literals with their C++ types (see ``tomato.py``). One ``main`` shoot type whose child
shoots are again ``main``: thin fern-like stems carrying whorls of 5 cladodes per node. A cladode is not a blade but a
tube object (``AsparagusLeafPrototype``: 4 nodes, 8 radial subdivisions, rolled about x), shaped from the *prototype's*
generator, which ``Phytomer``'s prototype cache redirects at the private ``minstd_rand0(identifier)`` stream.

RNG on load: ``PhytomerParameters(gen)`` (2 identifier draws: its own ``LeafPrototype(gen)`` and the assignment);
no separate ``LeafPrototype`` (``leaf.prototype.prototype_function`` is set on the member). The build draws nothing
(``make_AxisRotation(0, 0, 0)``).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from .strawberry import named_hook
from ..architecture import context_generator
from ..random_parameter import PlantArchitectureError
from ..params import BUD_DEAD, RGB, PhytomerParameters, ShootParameters, make_AxisRotation
from ...core.random import UniformReal

f32 = np.float32
PI_F = f32(np.pi)

AsparagusLeafPrototype = named_hook("AsparagusLeafPrototype")                        # AS:1531
AsparagusPhytomerCreationFunction = named_hook("AsparagusPhytomerCreationFunction")  # AS:1569


@ports("PlantArchitecture::initializeAsparagusShoots")
def initializeAsparagusShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 1, 2

    pp.internode.pitch = 1
    pp.internode.phyllotactic_angle.uniformDistribution(127.5, 147.5)
    pp.internode.radius_initial = 0.00025
    pp.internode.max_floral_buds_per_petiole = 0
    pp.internode.max_vegetative_buds_per_petiole = 1
    pp.internode.color = RGB.forestgreen
    pp.internode.length_segments = 2

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch = 90
    pp.petiole.radius = 0.0001
    pp.petiole.length = 0.0005
    pp.petiole.taper = 0.5
    pp.petiole.curvature = 0
    pp.petiole.color = pp.internode.color
    pp.petiole.length_segments = 1
    pp.petiole.radial_subdivisions = 5

    pp.leaf.leaves_per_petiole = 5
    pp.leaf.pitch.normalDistribution(-5, 30)
    pp.leaf.yaw = 30
    pp.leaf.roll = 0
    pp.leaf.leaflet_offset = 0
    pp.leaf.leaflet_scale = 0.9
    pp.leaf.prototype.prototype_function = AsparagusLeafPrototype
    pp.leaf.prototype_scale.uniformDistribution(0.018, 0.02)

    pp.peduncle.length = 0.17
    pp.peduncle.radius = 0.0015
    pp.peduncle.pitch.uniformDistribution(0, 30)
    pp.peduncle.roll = 90
    pp.peduncle.curvature.uniformDistribution(50, 250)
    pp.peduncle.length_segments = 6
    pp.peduncle.radial_subdivisions = 6

    pp.inflorescence.flowers_per_peduncle.uniformDistribution(1, 3)
    pp.inflorescence.flower_offset = 0.
    pp.inflorescence.pitch.uniformDistribution(50, 70)
    pp.inflorescence.roll.uniformDistribution(-20, 20)
    pp.inflorescence.flower_prototype_scale = 0.015
    pp.inflorescence.fruit_prototype_scale.uniformDistribution(0.02, 0.025)
    pp.inflorescence.fruit_gravity_factor_fraction.uniformDistribution(0., 0.5)

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.phytomer_parameters.phytomer_creation_function = AsparagusPhytomerCreationFunction

    sp.max_nodes = 20
    sp.insertion_angle_tip.uniformDistribution(40, 70)
    sp.internode_length_max = 0.015
    sp.base_roll = 90
    sp.base_yaw.uniformDistribution(-20, 20)
    sp.gravitropic_curvature = -200

    sp.phyllochron_min = 1
    sp.elongation_rate_max = 0.15
    sp.girth_area_factor = 30
    sp.vegetative_bud_break_time = 5
    sp.vegetative_bud_break_probability_min = 0.25
    sp.fruit_set_probability = 0.

    sp.defineChildShootTypes(["main"], [1.0])

    pa.defineShootType("main", sp)


@ports("PlantArchitecture::buildAsparagusPlant")
def buildAsparagusPlant(pa, base_position):
    """PL:2780-2798."""
    if not pa.shoot_types:
        initializeAsparagusShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    pa.addBaseStemShoot(plantID, 1, make_AxisRotation(0, f32(0), f32(0)), 0.0003, 0.015, 1, 0.1, 0.1, "main")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, 5, 100)
    pa.plant_instances[plantID].max_age = f32(20)
    return plantID


@AsparagusLeafPrototype.implement
@ports("AsparagusLeafPrototype")
def _asparagus_leaf_prototype(context, prototype_parameters, compound_leaf_index) -> int:
    """AS:1531-1567: a cladode tube shaped from the prototype's own generator (``helios::uniform_real_distribution<float>``,
    ``minval + u * (maxval - minval)`` in float). The node arguments are evaluated in order (one draw each)."""
    generator = prototype_parameters.getRandomGenerator()
    if generator is None:
        raise PlantArchitectureError(
            "ERROR (AsparagusLeafPrototype): Leaf prototype has no random number generator. Construct LeafPrototype with "
            "the Context's generator, which can be retrieved with Context::getRandomGenerator().")
    unif = UniformReal(0.0, 1.0, real="float")

    def randu(minval, maxval):
        minval, maxval = f32(minval), f32(maxval)
        return f32(minval + f32(unif(generator) * f32(maxval - minval)))

    curve_magnitude = randu(0, f32(0.2))
    nodes = [(f32(0), f32(0), f32(0)),
             (randu(f32(0.4), f32(0.7)), f32(0), f32(f32(-0.25) * curve_magnitude)),
             (f32(0.95), f32(0), f32(f32(-0.9) * curve_magnitude)),
             (f32(1), f32(0), f32(-curve_magnitude))]
    radius = [f32(0.015), f32(0.015), f32(0.015), f32(0.0)]
    colors = [RGB.forestgreen] * 4
    objID = context.add_tube_object(8, nodes, radius, colors)
    context.rotate_object(objID, randu(0, f32(2) * PI_F), "x")
    return objID


@AsparagusPhytomerCreationFunction.implement
@ports("AsparagusPhytomerCreationFunction")
def _asparagus_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:1569-1576: blind nodes (the first three) get smaller cladodes and no lateral bud. ``scaleLeafPrototypeScale(0.6)``
    takes a float."""
    if shoot_node_index <= 2:
        phytomer.scaleLeafPrototypeScale(f32(0.6))
        phytomer.setVegetativeBudState(BUD_DEAD)


__all__ = ["initializeAsparagusShoots", "buildAsparagusPlant", "AsparagusLeafPrototype",
           "AsparagusPhytomerCreationFunction"]
