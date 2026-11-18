# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:5510-5588 and src/Assets.cpp:2407-2417
# (Copyright (C) 2016-2026 Brian Bailey).
"""Butter lettuce (``initializeButterLettuceShoots`` / ``buildButterLettucePlant``, ``ButterLettucePhytomerCreationFunction``).

Line for line with the C++, literals with their C++ types (see ``tomato.py``). One ``mainstem`` shoot type: a rosette of
25 phytomers with 1 mm internodes and 1 mm petioles, ``GenericLeafPrototype`` blades (``RomaineLettuceLeaf.png``, 30
subdivisions, 10 unique prototypes), every leaf pitched down by ``60 deg * (max_nodes - node) / max_nodes`` by the
creation function (``Phytomer::rotateLeaf``).

RNG on load: the tomato's four identifier draws. Build: ``make_AxisRotation(randu(0, 0.03f*M_PI), 0, randu(0, 2*M_PI))``,
arguments right to left (roll first).
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from .. import hooks
from .strawberry import named_hook
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, deg2rad, make_AxisRotation, make_RGBcolor

f32 = np.float32

ButterLettucePhytomerCreationFunction = named_hook("ButterLettucePhytomerCreationFunction")   # AS:2407


@ports("PlantArchitecture::initializeButterLettuceShoots")
def initializeButterLettuceShoots(pa):
    gen = context_generator(pa.context_ptr)

    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "RomaineLettuceLeaf.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.85)
    leaf_prototype.midrib_fold_fraction = f32(0.2)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.2, 0.05)
    leaf_prototype.lateral_curvature = f32(-0.4)
    leaf_prototype.wave_period.uniformDistribution(0.15, 0.25)
    leaf_prototype.wave_amplitude.uniformDistribution(0.05, 0.1)
    leaf_prototype.subdivisions = 30
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle = 137.5
    pp.internode.radius_initial = 0.02
    pp.internode.color = make_RGBcolor(0.402, 0.423, 0.413)
    pp.internode.length_segments = 1
    pp.internode.radial_subdivisions = 10

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(0, 30)
    pp.petiole.radius = 0.001
    pp.petiole.length = 0.001
    pp.petiole.length_segments = 1
    pp.petiole.radial_subdivisions = 3
    pp.petiole.color = make_RGBcolor(1, 0, 0)                                    # RGB::red

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch = 10
    pp.leaf.yaw = 0
    pp.leaf.roll = 0
    pp.leaf.prototype_scale.uniformDistribution(0.15, 0.25)
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    pp.phytomer_creation_function = ButterLettucePhytomerCreationFunction

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.vegetative_bud_break_probability_min = 0
    sp.phyllochron_min = 2
    sp.elongation_rate_max = 0.15
    sp.girth_area_factor = f32(0)
    sp.gravitropic_curvature = 10
    sp.internode_length_max = 0.001
    sp.internode_length_decay_rate = 0
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.flower_bud_break_probability = 0.0
    sp.max_nodes = 25

    pa.defineShootType("mainstem", sp)


@ports("PlantArchitecture::buildButterLettucePlant")
def buildButterLettucePlant(pa, base_position):
    """PL:5570-5588."""
    if not pa.shoot_types:
        initializeButterLettuceShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    # make_AxisRotation(randu(0.f, 0.03f * M_PI), 0.f, randu(0.f, 2.f * M_PI)): right to left, roll first;
    # 0.03f * M_PI and 2.f * M_PI are double, narrowed by randu(float, float)
    roll = pa.context_ptr.randu(f32(0), f32(2.0 * np.pi))
    pitch = pa.context_ptr.randu(f32(0), f32(float(f32(0.03)) * np.pi))
    pa.addBaseStemShoot(plantID, 3, make_AxisRotation(pitch, f32(0), roll), 0.005, 0.001, 1, 1, 0, "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


@ButterLettucePhytomerCreationFunction.implement
@ports("ButterLettucePhytomerCreationFunction")
def _butterlettuce_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:2407-2417. ``shoot_max_nodes - shoot_node_index`` is uint arithmetic (modulo 2^32), then float."""
    fact = f32(f32((int(shoot_max_nodes) - int(shoot_node_index)) % 2**32) / f32(int(shoot_max_nodes)))
    phytomer.rotateLeaf(0, 0, make_AxisRotation(f32(-deg2rad(60) * fact), 0, 0))


__all__ = ["initializeButterLettuceShoots", "buildButterLettucePlant", "ButterLettucePhytomerCreationFunction"]
