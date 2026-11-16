# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:6101-6203 (Copyright (C) 2016-2026 Brian Bailey).
"""Tomato shoot parameters (``PlantArchitecture::initializeTomatoShoots``).

Line for line with the C++, literals with their C++ types: an ``f`` suffix or a value stored into a
``float`` is narrowed with ``f32``; ``137.5 - 40`` and ``0.12f * 0.9031f`` are evaluated as the C++ does
(double, then narrowed by the float parameter; float*float).

RNG (SPEC §3.3 "Load"): four identifier draws from the Context stream, each 2, 4, 6... engine calls:

1. ``LeafPrototype leaf_prototype(gen)``                         -> ctor ``sampleIdentifier``
2. ``PhytomerParameters phytomer_parameters(gen)``                 -> ``LeafPrototype(gen)`` ctor
3.                                                                 -> ``leaf.prototype = ...`` operator=
4. ``phytomer_parameters.leaf.prototype = leaf_prototype``         -> operator= (this identifier is kept)

``ShootParameters(gen)``, ``shoot_parameters.phytomer_parameters = ...`` and ``defineShootType`` draw nothing.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, make_RGBcolor

f32 = np.float32


@ports("PlantArchitecture::initializeTomatoShoots")
def initializeTomatoShoots(pa):
    gen = context_generator(pa.context_ptr)   # context_ptr->getRandomGenerator(): the Scene engine itself

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "TomatoLeaf_blade.png"
    leaf_prototype.leaf_aspect_ratio = f32(f32(0.5) / f32(0.9031))
    leaf_prototype.midrib_fold_fraction = f32(0.1)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.45, f32(-0.2))
    leaf_prototype.lateral_curvature = f32(-0.3)
    leaf_prototype.wave_period = f32(0.35)
    leaf_prototype.wave_amplitude = f32(0.08)
    leaf_prototype.subdivisions = 6
    leaf_prototype.unique_prototypes = 5
    leaf_prototype.build_petiolule = True
    leaf_prototype.petiolule_length = {0: f32(0.085)}

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 10
    pp.internode.phyllotactic_angle.uniformDistribution(137.5 - 40, 137.5 + 40)
    pp.internode.radius_initial = 0.001
    pp.internode.color = make_RGBcolor(0.2495, 0.3162, 0.0657)
    pp.internode.length_segments = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(45, 60)
    pp.petiole.radius = 0.002
    pp.petiole.length = 0.2
    pp.petiole.taper = 0.7
    pp.petiole.curvature.uniformDistribution(-150, -50)
    pp.petiole.color = make_RGBcolor(0.3, 0.37, 0.0657)
    pp.petiole.length_segments = 5

    pp.leaf.leaves_per_petiole = 7
    pp.leaf.pitch.uniformDistribution(-30, 5)
    pp.leaf.yaw = 10
    pp.leaf.roll = 0
    pp.leaf.leaflet_offset = 0.15
    pp.leaf.leaflet_scale = 0.7
    pp.leaf.prototype_scale.uniformDistribution(f32(f32(0.12) * f32(0.9031)), f32(f32(0.18) * f32(0.9031)))
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    pp.peduncle.length = 0.18
    pp.peduncle.radius = 0.0015
    pp.peduncle.pitch = 20
    pp.peduncle.roll = 0
    pp.peduncle.curvature = -900
    pp.peduncle.color = pp.internode.color
    pp.peduncle.length_segments = 5
    pp.peduncle.radial_subdivisions = 8

    pp.inflorescence.flowers_per_peduncle = 6
    pp.inflorescence.flower_offset = 0.15
    pp.inflorescence.pitch = 80
    pp.inflorescence.roll = 180
    pp.inflorescence.flower_prototype_scale = 0.05
    pp.inflorescence.flower_prototype_function = hooks.TomatoFlowerPrototype
    pp.inflorescence.fruit_prototype_scale = 0.15
    pp.inflorescence.fruit_prototype_function = hooks.TomatoFruitPrototype
    pp.inflorescence.fruit_gravity_factor_fraction = 0.

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp                                                  # no draw (duplicate())
    sp.phytomer_parameters.phytomer_creation_function = hooks.TomatoPhytomerCreationFunction

    sp.max_nodes = 16
    sp.insertion_angle_tip = 30
    sp.insertion_angle_decay_rate = 0
    sp.internode_length_max = 0.04
    sp.internode_length_min = 0.0
    sp.internode_length_decay_rate = 0
    sp.base_roll = 90
    sp.base_yaw.uniformDistribution(-20, 20)
    sp.gravitropic_curvature = 150
    sp.tortuosity = 75

    sp.phyllochron_min = 2
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(2.0)
    sp.vegetative_bud_break_time = 30
    sp.vegetative_bud_break_probability_min = 0.25
    sp.vegetative_bud_break_probability_decay_rate = -0.25
    sp.flower_bud_break_probability = 0.2
    sp.fruit_set_probability = 0.8
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = True

    sp.defineChildShootTypes(["mainstem"], [1.0])

    pa.defineShootType("mainstem", sp)


@ports("PlantArchitecture::buildTomatoPlant")
def buildTomatoPlant(pa, base_position):
    """PL:6205-6224. Only the parts W2-C owns are here; ``addPlantInstance`` onward is W3."""
    if not pa.shoot_types:
        initializeTomatoShoots(pa)
    raise NotImplementedError("buildTomatoPlant needs addPlantInstance/addBaseStemShoot (W3-C)")


def base_rotation_draws(pa):
    """``make_AxisRotation(0, randu(0.f, 2.f*M_PI), randu(0.f, 2.f*M_PI))`` (PL:6214).

    GCC evaluates the arguments right to left: the *roll* is drawn first (oracle-verified, SPEC §3.3).
    ``2.f * M_PI`` is float*double -> double, narrowed to float by ``randu(float, float)``.
    """
    from ..params import make_AxisRotation
    hi = f32(2.0 * np.pi)
    roll = pa.context_ptr.randu(f32(0), hi)
    yaw = pa.context_ptr.randu(f32(0), hi)
    return make_AxisRotation(0, yaw, roll)
