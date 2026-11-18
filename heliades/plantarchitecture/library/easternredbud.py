# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:5231-5364 and src/Assets.cpp:2275-2297
# (Copyright (C) 2016-2026 Brian Bailey).
"""Eastern redbud (W5-E): ``initializeEasternRedbudShoots``, ``buildEasternRedbudPlant`` and its assets.

The builder grows a 16-node trunk with ``addBaseStemShoot``, makes the plant dormant and breaks dormancy at once,
and kills the vegetative buds of the trunk's first 12 nodes. Phenology 165 / -1 / 3 / 7 / 30 / 200, ``max_age``
1460. The model declares a leaf-inclination Beta of (1.00, 2.20), so ``buildPlantInstanceFromLibrary`` enables
leaf elevation-angle tracking for it.

Helios quirks reproduced:

* the trunk rotation is ``make_AxisRotation(randu(0, 0.1 * M_PI), randu(0, 2 * M_PI), randu(0, 2 * M_PI))``: an
  ``int`` and a ``double`` argument select ``Context::randu(int, int)`` (the ``int`` argument is an exact match),
  so the pitch is ``randu(0, 0)`` = 0 without a draw and the yaw and roll are whole radians drawn from {0..6}
  (roll first: GCC evaluates arguments right to left);
* ``RedbudPhytomerCallbackFunction`` draws ``helios::randu(2, 4)``, the free function on the process-wide
  ``std::mt19937`` (``heliades.core.random.global_randu``), on every call for a dormant phytomer. Helios seeds that
  generator from ``std::random_device``, so a redbud past its first dormancy differs from run to run unless
  ``helios::seedRandomGenerator`` (PyHelios ``Global.seedRandomGenerator``) is called;
* the leaf is ``RedbudLeaf.png`` on the generic prototype; the shipped ``RedbudLeaf.obj`` (which does not load,
  HELIOS_ISSUES 17) is not used by the model.
"""
from __future__ import annotations

import numpy as np

from ...core.random import global_randu
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, RGBcolor, ShootParameters, make_AxisRotation, make_RGBcolor
from .cherrytomato import _hook

f32 = np.float32

RedbudFlowerPrototype = _hook("RedbudFlowerPrototype")                    # AS:2275
RedbudFruitPrototype = _hook("RedbudFruitPrototype")                      # AS:2280
RedbudPhytomerCreationFunction = _hook("RedbudPhytomerCreationFunction")  # AS:2285
RedbudPhytomerCallbackFunction = _hook("RedbudPhytomerCallbackFunction")  # AS:2288


def _scaled(color, s) -> RGBcolor:
    """``RGBcolor::scale`` (no clamp)."""
    s = f32(s)
    return RGBcolor(f32(color[0] * s), f32(color[1] * s), f32(color[2] * s))


@ports("PlantArchitecture::initializeEasternRedbudShoots")
def initializeEasternRedbudShoots(pa):
    gen = context_generator(pa.context_ptr)
    lp = LeafPrototype(gen)
    lp.leaf_texture_file[0] = "RedbudLeaf.png"
    lp.leaf_aspect_ratio = f32(1.0)
    lp.midrib_fold_fraction = f32(0.2)
    lp.longitudinal_curvature = f32(-0.15)
    lp.lateral_curvature = f32(-0.1)
    lp.wave_period = f32(0.3)
    lp.wave_amplitude = f32(0.025)
    lp.subdivisions = 5
    lp.unique_prototypes = 5
    lp.leaf_offset = (f32(-0.3), f32(0), f32(0))

    pp = PhytomerParameters(gen)
    pp.internode.pitch = 15
    pp.internode.phyllotactic_angle.uniformDistribution(170, 190)
    pp.internode.radius_initial = 0.0015
    pp.internode.image_texture = "WesternRedbudBark.jpg"
    pp.internode.color = _scaled(pp.internode.color, 0.3)
    pp.internode.length_segments = 1
    pp.internode.max_floral_buds_per_petiole = 5

    pp.petiole.petioles_per_internode = 1
    pp.petiole.color = make_RGBcolor(0.65, 0.52, 0.39)
    pp.petiole.pitch.uniformDistribution(20, 40)
    pp.petiole.radius = 0.002
    pp.petiole.length = 0.075
    pp.petiole.taper = 0
    pp.petiole.curvature = 0
    pp.petiole.length_segments = 1

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-110, -80)
    pp.leaf.yaw = 0
    pp.leaf.roll.uniformDistribution(-5, 5)
    pp.leaf.prototype_scale = 0.1
    pp.leaf.prototype = lp

    pp.peduncle.length = 0.02
    pp.peduncle.pitch.uniformDistribution(50, 90)
    pp.peduncle.color = make_RGBcolor(0.32, 0.05, 0.13)

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.flower_prototype_function = RedbudFlowerPrototype
    pp.inflorescence.flower_prototype_scale = 0.04
    pp.inflorescence.fruit_prototype_function = RedbudFruitPrototype
    pp.inflorescence.fruit_prototype_scale = 0.1
    pp.inflorescence.fruit_gravity_factor_fraction = 0.7

    pp.phytomer_creation_function = RedbudPhytomerCreationFunction
    pp.phytomer_callback_function = RedbudPhytomerCallbackFunction

    main = ShootParameters(gen)
    main.phytomer_parameters = pp
    main.vegetative_bud_break_probability_min = 1.0
    main.vegetative_bud_break_time = 2
    main.phyllochron_min = 2
    main.elongation_rate_max = 0.1
    main.girth_area_factor = f32(3.0)
    main.gravitropic_curvature = 300
    main.tortuosity = 125
    main.internode_length_max = 0.04
    main.internode_length_decay_rate = 0.005
    main.internode_length_min = 0.01
    main.insertion_angle_tip = 75
    main.insertion_angle_decay_rate = 10
    main.flowers_require_dormancy = True
    main.growth_requires_dormancy = True
    main.determinate_shoot_growth = False
    main.max_terminal_floral_buds = 0
    main.flower_bud_break_probability = 0.8
    main.fruit_set_probability = 0.3
    main.max_nodes = 25
    main.max_nodes_per_season = 10
    main.base_roll = 90

    trunk = main.copy()
    trunk.girth_area_factor = f32(1.13)
    trunk.phytomer_parameters.internode.pitch = 0
    trunk.phytomer_parameters.internode.radial_subdivisions = 15
    trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0
    trunk.insertion_angle_tip = 60
    trunk.max_nodes = 75
    trunk.max_nodes_per_season = 10
    trunk.tortuosity = 37.5
    trunk.defineChildShootTypes(["eastern_redbud_shoot"], [1.0])

    pa.defineShootType("eastern_redbud_trunk", trunk)
    pa.defineShootType("eastern_redbud_shoot", main)


@ports("PlantArchitecture::buildEasternRedbudPlant")
def buildEasternRedbudPlant(pa, base_position):
    """PL:5334-5364."""
    if not pa.shoot_types:
        initializeEasternRedbudShoots(pa)
    ctx = pa.context_ptr
    plantID = pa.addPlantInstance(base_position, 0)
    # Context::randu(int, int) (see the module docstring): roll, yaw, then pitch = randu(0, 0) = 0
    roll = f32(ctx.randu(0, 6))
    yaw = f32(ctx.randu(0, 6))
    pitch = f32(ctx.randu(0, 0))
    uID_stem = pa.addBaseStemShoot(plantID, 16, make_AxisRotation(pitch, yaw, roll), f32(0.0075), f32(0.05), f32(1),
                                   f32(1), f32(0.4), "eastern_redbud_trunk")
    from ..seasons import makePlantDormant
    makePlantDormant(pa, plantID)
    pa.breakPlantDormancy(plantID)
    for phytomer in pa.plant_instances[plantID].shoot_tree[uID_stem].phytomers:
        if int(phytomer.shoot_index[0]) < 12:
            for petiole in phytomer.axillary_vegetative_buds:
                for vbud in petiole:
                    phytomer.setVegetativeBudState(BUD_DEAD, vbud)
    pa.setPlantPhenologicalThresholds(plantID, 165, -1, 3, 7, 30, 200)
    pa.plant_instances[plantID].max_age = f32(1460)
    return plantID


@RedbudFlowerPrototype.implement
@ports("RedbudFlowerPrototype")
def _redbud_flower_prototype(context, subdivisions, flower_is_open, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/RedbudFlower_open.obj")


@RedbudFruitPrototype.implement
@ports("RedbudFruitPrototype")
def _redbud_fruit_prototype(context, subdivisions, build_parameters=None):
    from .bean import load_obj_black
    return load_obj_black(context, "assets/obj/RedbudPod.obj")


@RedbudPhytomerCreationFunction.implement
@ports("RedbudPhytomerCreationFunction")
def _redbud_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    return None


@RedbudPhytomerCallbackFunction.implement
@ports("RedbudPhytomerCallbackFunction")
def _redbud_phytomer_callback(phytomer):
    """AS:2288-2297: on a dormant phytomer, the last 2-4 nodes keep their vegetative buds and lose their floral
    ones; the nodes below them keep only their floral buds (``helios::randu``, the process-wide generator)."""
    if phytomer.isdormant:
        Nchild_shoots = global_randu(2, 4)
        if int(phytomer.shoot_index[0]) < int(phytomer.shoot_index[1]) - Nchild_shoots:
            phytomer.setVegetativeBudState(BUD_DEAD)
        else:
            phytomer.setFloralBudState(BUD_DEAD)


__all__ = ["initializeEasternRedbudShoots", "buildEasternRedbudPlant"]
