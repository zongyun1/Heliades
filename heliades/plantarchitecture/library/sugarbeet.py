# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:6017-6099 (Copyright (C) 2016-2026 Brian Bailey).
"""Sugar beet (``initializeSugarbeetShoots`` / ``buildSugarbeetPlant``).

Line for line with the C++, literals with their C++ types (see ``tomato.py``). One ``mainstem`` shoot type: a rosette
of 30 phytomers with 1 mm internodes, curved 15-20 cm petioles (8 segments, taper 0.6) and ``GenericLeafPrototype``
blades textured with ``SugarbeetLeaf.png`` (20 subdivisions, 10 unique prototypes). No creation function, no
inflorescence.

The shipped OBJ leaf ``assets/obj/SugarBeetLeaf.obj`` (whose ``mtllib SugarbeetLeaf.mtl`` does not resolve on a
case-sensitive filesystem) is referenced by no function in 1.3.98: the library entry builds procedural blades, so the
broken OBJ is never loaded and the sugar beet builds without error on Helios.

RNG on load: the tomato's four identifier draws. Build: ``make_AxisRotation(randu(0, 0.01f*M_PI),
0.f * randu(0, 2*M_PI), 0.25f*M_PI)``, arguments right to left: the (discarded) yaw draw, then the pitch.
"""
from __future__ import annotations

import numpy as np

from ...migration_registry import ports
from ..architecture import context_generator
from ..params import LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, make_RGBcolor

f32 = np.float32


@ports("PlantArchitecture::initializeSugarbeetShoots")
def initializeSugarbeetShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "SugarbeetLeaf.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.4)
    leaf_prototype.midrib_fold_fraction = f32(0.1)
    leaf_prototype.longitudinal_curvature = f32(-0.2)
    leaf_prototype.lateral_curvature = f32(-0.4)
    leaf_prototype.petiole_roll = f32(0.75)
    leaf_prototype.wave_period.uniformDistribution(f32(0.08), f32(0.15))
    leaf_prototype.wave_amplitude.uniformDistribution(0.02, 0.04)
    leaf_prototype.subdivisions = 20
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle = 137.5
    pp.internode.radius_initial = 0.005
    pp.internode.color = make_RGBcolor(0.44, 0.58, 0.19)
    pp.internode.length_segments = 1
    pp.internode.max_vegetative_buds_per_petiole = 0
    pp.internode.max_floral_buds_per_petiole = 0

    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.uniformDistribution(0, 40)
    pp.petiole.radius = 0.005
    pp.petiole.length.uniformDistribution(0.15, 0.2)
    pp.petiole.taper = 0.6
    pp.petiole.curvature.uniformDistribution(-300, 100)
    pp.petiole.color = pp.internode.color
    pp.petiole.length_segments = 8

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-10, 0)
    pp.leaf.yaw.uniformDistribution(-5, 5)
    pp.leaf.roll.uniformDistribution(-15, 15)
    pp.leaf.prototype_scale.uniformDistribution(0.15, 0.25)
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.vegetative_bud_break_probability_min = 0
    sp.phyllochron_min = 2
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(20)
    sp.gravitropic_curvature = 10
    sp.internode_length_max = 0.001
    sp.internode_length_decay_rate = 0
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.flower_bud_break_probability = 0.0
    sp.max_nodes = 30

    pa.defineShootType("mainstem", sp)


@ports("PlantArchitecture::buildSugarbeetPlant")
def buildSugarbeetPlant(pa, base_position):
    """PL:6081-6099."""
    if not pa.shoot_types:
        initializeSugarbeetShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    # right to left: 0.25f * M_PI (no draw), 0.f * randu(0.f, 2.f * M_PI) (drawn, then multiplied by 0 in float),
    # then randu(0.f, 0.01f * M_PI); the M_PI products are double, narrowed to float
    yaw = f32(f32(0) * pa.context_ptr.randu(f32(0), f32(2.0 * np.pi)))
    pitch = pa.context_ptr.randu(f32(0), f32(float(f32(0.01)) * np.pi))
    pa.addBaseStemShoot(plantID, 3, make_AxisRotation(pitch, yaw, f32(float(f32(0.25)) * np.pi)), 0.005, 0.001, 1, 1, 0,
                        "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, -1, -1, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


__all__ = ["initializeSugarbeetShoots", "buildSugarbeetPlant"]
