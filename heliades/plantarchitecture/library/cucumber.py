# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:6689-6824 and src/Assets.cpp:4262-4458, with
# PlantArchitecture::enableAttractionPoints / updateAttractionPoints (PA:13455-13511)
# (Copyright (C) 2016-2026 Brian Bailey).
"""High-wire greenhouse cucumber (``initializeCucumberShoots`` / ``buildCucumberPlant`` and its Assets.cpp functions).

Line for line with the C++, literals with their C++ types (see ``tomato.py``). One unbranched ``main`` shoot type
(125 nodes, indeterminate, tortuosity 5), trained up a string of attraction points to a wire 3.8 m above the base;
``CucumberMainPhytomerCallbackFunction`` picks every full-size fruit and, once the head reaches the wire, lowers and
leans the stem (the older stem laid along +y at 0.3 m, the head held at the wire, the string slid with it).
``CucumberMainPhytomerCreationFunction`` is the tomato's own creation function plus a bump map on the new leaves.
Flowers and fruit are baked OBJ variants picked with one Context ``randu`` each.

RNG on load: the tomato's four identifier draws. Build: one ``randu(0, 2*pi)`` (the base yaw).
"""
from __future__ import annotations

import math

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from .. import hooks
from .strawberry import named_hook
from ..architecture import context_generator
from ..params import BUD_DEAD, BUD_FRUITING, LeafPrototype, PhytomerParameters, ShootParameters, make_AxisRotation, \
    make_RGBcolor
from ..random_parameter import PlantArchitectureError

f32 = np.float32


def _sub(a, b):
    return np.asarray(a, f32) - np.asarray(b, f32)


def _add(a, b):
    return np.asarray(a, f32) + np.asarray(b, f32)

CucumberFruitPrototype = named_hook("CucumberFruitPrototype")                                   # AS:4265
CucumberFlowerPrototype = named_hook("CucumberFlowerPrototype")                                 # AS:4276
CucumberMainPhytomerCreationFunction = named_hook("CucumberMainPhytomerCreationFunction")       # AS:4286
CucumberMainPhytomerCallbackFunction = named_hook("CucumberMainPhytomerCallbackFunction")       # AS:4369

# High-wire training constants (AS:4307-4313), all float
cucumber_wire_height = f32(3.8)
cucumber_string_top = f32(cucumber_wire_height + f32(0.6))
cucumber_string_point_spacing = f32(0.05)
cucumber_laid_stem_height = f32(0.3)
cucumber_bend_radius = f32(0.2)


@ports("PlantArchitecture::initializeCucumberShoots")
def initializeCucumberShoots(pa):
    gen = context_generator(pa.context_ptr)

    lp = LeafPrototype(gen)                                                      # identifier draw 1
    lp.build_petiolule = False
    lp.flexibility = 0.0
    lp.flexibility_aging = 0.0
    lp.flexibility_aging_max = 4.0
    lp.flexibility_taper = 1.0
    lp.lateral_curvature.uniformDistribution(-0.15, 0.05)
    lp.leaf_aspect_ratio = 0.8917609
    lp.leaf_offset = (f32(-0.2771165), f32(0.0), f32(0.0))
    lp.leaf_texture_file[0] = "CucumberLeaf_0.png"
    lp.longitudinal_curvature.uniformDistribution(-0.25, 0.0)
    lp.longitudinal_curvature_exponent = 4.0
    lp.midrib_fold_fraction.uniformDistribution(0.0, 0.15)
    lp.petiole_roll = 0.0
    lp.petiolule_length = {0: f32(0.085)}
    lp.roll_fraction = 0.0
    lp.subdivisions = 10
    lp.unique_prototypes = 10
    lp.wave_amplitude.uniformDistribution(0.0, 0.04)
    lp.wave_period = 0.3

    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3
    pp.inflorescence.flower_offset = 0.0
    pp.inflorescence.flower_prototype_function = CucumberFlowerPrototype
    pp.inflorescence.flower_prototype_scale.normalDistribution(0.11, 0.01)
    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.fruit_gravity_factor_fraction = 0.8
    pp.inflorescence.fruit_prototype_function = CucumberFruitPrototype
    pp.inflorescence.fruit_prototype_scale.normalDistribution(0.3, 0.015)
    pp.inflorescence.inflorescence_maturity_period = -1.0
    pp.inflorescence.pitch = 20.0
    pp.inflorescence.roll = 180.0
    pp.inflorescence.unique_prototypes = 6
    pp.internode.color = make_RGBcolor(0.3, 0.45, 0.16)
    pp.internode.length_segments = 1
    pp.internode.max_floral_buds_per_petiole = 1
    pp.internode.max_vegetative_buds_per_petiole = 0
    pp.internode.phyllotactic_angle.normalDistribution(144.0, 12.0)
    pp.internode.pitch = 0.0
    pp.internode.radial_subdivisions = 7
    pp.internode.radius_initial = 0.003
    pp.leaf.intercalary_leaflet_scale = 0.0
    pp.leaf.leaflet_offset = 0.0
    pp.leaf.leaflet_scale = 1.0
    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.normalDistribution(-35.0, 12.0)
    pp.leaf.prototype_scale.normalDistribution(0.264, 0.019)
    pp.leaf.roll.normalDistribution(0.0, 8.0)
    pp.leaf.yaw.normalDistribution(0.0, 8.0)
    pp.peduncle.color = make_RGBcolor(0.3, 0.45, 0.16)
    pp.peduncle.curvature = -300.0
    pp.peduncle.length.normalDistribution(0.035, 0.008)
    pp.peduncle.length_segments = 5
    pp.peduncle.pitch = 60.0
    pp.peduncle.radial_subdivisions = 8
    pp.peduncle.radius = 0.003
    pp.peduncle.roll = 0.0
    pp.peduncle.yaw = 0.0
    pp.petiole.color = make_RGBcolor(0.32, 0.5, 0.18)
    pp.petiole.curvature.uniformDistribution(-120.0, -40.0)
    pp.petiole.flexibility = 0.0
    pp.petiole.flexibility_aging = 0.0
    pp.petiole.length.normalDistribution(0.19, 0.025)
    pp.petiole.length_segments = 5
    pp.petiole.petioles_per_internode = 1
    pp.petiole.pitch.normalDistribution(50.0, 10.0)
    pp.petiole.radial_subdivisions = 7
    pp.petiole.radius = 0.0035
    pp.petiole.taper = 0.4
    pp.leaf.prototype = lp                                                       # identifier draw 4

    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.phytomer_parameters.phytomer_creation_function = CucumberMainPhytomerCreationFunction
    sp.phytomer_parameters.phytomer_callback_function = CucumberMainPhytomerCallbackFunction
    sp.base_roll = 90.0
    sp.base_yaw.uniformDistribution(-20.0, 20.0)
    sp.determinate_shoot_growth = False
    sp.elongation_rate_max = 0.2
    sp.flower_bud_break_probability = 0.6
    sp.flowers_require_dormancy = False
    sp.fruit_set_probability = 0.7
    sp.girth_area_factor = 0.8
    sp.gravitropic_curvature = 0.0
    sp.growth_requires_dormancy = False
    sp.insertion_angle_decay_rate = 0.0
    sp.insertion_angle_tip = 30.0
    sp.internode_length_decay_rate = 0.0
    sp.internode_length_max.normalDistribution(0.1, 0.009)
    sp.internode_length_min = 0.0
    sp.leaf_expansion_rate_max = -1.0
    sp.max_nodes = 125
    sp.max_nodes_per_season = 9999
    sp.max_terminal_floral_buds = 0
    sp.phyllochron_min.normalDistribution(1.43, 0.1)
    sp.tortuosity = 5.0
    sp.tortuosity_persistence_length = 0.5
    sp.vegetative_bud_break_probability_decay_rate = -0.25
    sp.vegetative_bud_break_probability_max = 0.0
    sp.vegetative_bud_break_probability_min = 0.0
    sp.vegetative_bud_break_time = 30.0
    pa.defineShootType("main", sp)


@ports("PlantArchitecture::buildCucumberPlant")
def buildCucumberPlant(pa, base_position):
    """PL:6800-6824."""
    if not pa.shoot_types:
        initializeCucumberShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    # make_AxisRotation(0, randu(0.f, 2.f * float(M_PI)), 0): 2.f * float(M_PI) is float
    base_rotation = make_AxisRotation(0, pa.context_ptr.randu(f32(0), f32(f32(2) * f32(math.pi))), 0)
    internode_length = f32(0.1)
    pa.addBaseStemShoot(plantID, 2, base_rotation, 0.003, internode_length, 0.1, 0.1, 0.0, "main")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0.0, 12.0, 6.0, 2.0, 14.0, 1000.0, 45.0, False)
    pa.plant_instances[plantID].max_age = f32(165)
    enableAttractionPoints(pa, plantID, CucumberStringAttractionPoints(base_position), 80.0, 0.15, 0.8)
    return plantID


# ---- attraction points (PA:13455-13511) ---------------------------------------------------------------

@ports("PlantArchitecture::enableAttractionPoints(uint,const std::vector<helios::vec3>&,float,float,float)")
def enableAttractionPoints(pa, plantID, attraction_points, view_half_angle_deg, look_ahead_distance, attraction_weight):
    """Per-plant attraction points; unlike ``setPlantAttractionPoints`` the obstacle reduction factor is left as it is."""
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::enableAttractionPoints): Plant with ID {plantID} does not exist.")
    view_half_angle_deg, look_ahead_distance = f32(view_half_angle_deg), f32(look_ahead_distance)
    if view_half_angle_deg <= 0 or view_half_angle_deg > 180:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enableAttractionPoints): view_half_angle_deg must be between 0 "
                                     "and 180 degrees.")
    if look_ahead_distance <= 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enableAttractionPoints): look_ahead_distance must be greater "
                                     "than 0.")
    if len(attraction_points) == 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::enableAttractionPoints): attraction_points cannot be empty.")
    plant = pa.plant_instances[plantID]
    plant.attraction_points_enabled = True
    plant.attraction_points = np.asarray(attraction_points, f32).reshape(-1, 3).copy()
    plant.attraction_cone_half_angle_rad = V.deg2rad(view_half_angle_deg)
    plant.attraction_cone_height = look_ahead_distance
    plant.attraction_weight = f32(attraction_weight)


@ports("PlantArchitecture::updateAttractionPoints")
def updateAttractionPoints(pa, plantID, attraction_points):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::updateAttractionPoints): Plant with ID {plantID} does not exist.")
    plant = pa.plant_instances[plantID]
    if not plant.attraction_points_enabled:
        raise PlantArchitectureError("ERROR (PlantArchitecture::updateAttractionPoints): Attraction points must be enabled for "
                                     f"plant {plantID} before updating positions.")
    if len(attraction_points) == 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::updateAttractionPoints): attraction_points cannot be empty.")
    plant.attraction_points = np.asarray(attraction_points, f32).reshape(-1, 3).copy()


# ---- Assets.cpp ---------------------------------------------------------------------------------------

def _load_variant(context, filename):
    """``loadOBJ(resolvePluginAsset(.../Cucumber/<file>), (0,0,0), 0, nullrotation, RGB::black, "ZUP", true)``."""
    from .tomato_inflorescence import _plugin_asset
    uu = context.load_obj(_plugin_asset("assets/obj/Cucumber/" + filename), (0.0, 0.0, 0.0), height=f32(0),
                          rotation=(0.0, 0.0), color=(0.0, 0.0, 0.0), upaxis="ZUP", silent=True)
    return context.add_polymesh_object(list(uu))


@CucumberFruitPrototype.implement
@ports("CucumberFruitPrototype")
def _cucumber_fruit(context, subdivisions, build_parameters=None) -> int:
    """AS:4265-4271: one of six baked variants, picked with ``randu(0, 5)`` (int)."""
    pick = int(context.randu(0, 5))
    return _load_variant(context, f"fruit_{pick:02d}.obj")


@CucumberFlowerPrototype.implement
@ports("CucumberFlowerPrototype")
def _cucumber_flower(context, subdivisions, flower_is_open, build_parameters=None) -> int:
    """AS:4276-4284."""
    pick = int(context.randu(0, 5))
    return _load_variant(context, f"flower_{'open' if flower_is_open else 'closed'}_{pick:02d}.obj")


@CucumberMainPhytomerCreationFunction.implement
@ports("CucumberMainPhytomerCreationFunction")
def _cucumber_main_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:4286-4301: the tomato's creation function, then the leaf bump map (strength 12.58861f)."""
    hooks.TomatoPhytomerCreationFunction(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age)
    context = phytomer.parent_shoot_ptr.context_ptr
    leaf_UUIDs = []
    for petiole in phytomer.leaf_objIDs:
        for objID in petiole:
            leaf_UUIDs.extend(context.object_primitive_uuids(objID))
    if leaf_UUIDs:
        from .tomato_inflorescence import _plugin_asset
        context.set_primitive_bump_map(leaf_UUIDs, _plugin_asset("assets/textures/CucumberLeaf_0_bump.png"), f32(12.58861))


@ports("cucumberLoweredStemPath")
def cucumberLoweredStemPath(laid_length, t):
    """AS:4321-4357 (file-static): the (y, z) point at arclength ``t`` along the lowered stem path. ``std::cos`` /
    ``std::sin`` / ``std::min`` / ``std::max`` on floats are the float overloads; ``0.5f * float(M_PI)`` is float."""
    laid_length, t = f32(laid_length), f32(t)
    radius = cucumber_bend_radius
    rise = f32(cucumber_laid_stem_height - radius)
    if t <= rise or laid_length <= rise:
        return (f32(0), t)
    bend_angle = min(f32(f32(0.5) * f32(math.pi)), f32(f32(laid_length - rise) / f32(f32(2) * radius)))
    bend_length = f32(radius * bend_angle)
    run_length = max(f32(0), f32(f32(laid_length - rise) - f32(f32(2) * bend_length)))

    s = f32(t - rise)
    if s <= bend_length:
        heading = f32(s / radius)
        return (f32(radius * f32(f32(1) - V.cosf(heading))), f32(rise + f32(radius * V.sinf(heading))))
    first_bend_end = (f32(radius * f32(f32(1) - V.cosf(bend_angle))), f32(rise + f32(radius * V.sinf(bend_angle))))
    first_bend_heading = (V.sinf(bend_angle), V.cosf(bend_angle))
    s = f32(s - bend_length)

    if s <= run_length:
        return (f32(first_bend_end[0] + f32(first_bend_heading[0] * s)), f32(first_bend_end[1] + f32(first_bend_heading[1] * s)))
    run_end = (f32(first_bend_end[0] + f32(first_bend_heading[0] * run_length)),
               f32(first_bend_end[1] + f32(first_bend_heading[1] * run_length)))
    s = f32(s - run_length)

    c = (f32(-V.cosf(bend_angle) * radius), f32(V.sinf(bend_angle) * radius))
    second_bend_center = (f32(run_end[0] + c[0]), f32(run_end[1] + c[1]))
    if s <= bend_length:
        heading = f32(bend_angle - f32(s / radius))
        return (f32(second_bend_center[0] + f32(V.cosf(heading) * radius)),
                f32(second_bend_center[1] + f32(-V.sinf(heading) * radius)))
    second_bend_end = (f32(second_bend_center[0] + radius), f32(second_bend_center[1] + f32(0)))
    return (f32(second_bend_end[0] + f32(0)), f32(second_bend_end[1] + f32(s - bend_length)))


@ports("CucumberStringAttractionPoints")
def CucumberStringAttractionPoints(string_base):
    """AS:4359-4367: points every 5 cm from 5 cm above ``string_base`` to the string top (4.4 m): 88 points.
    ``std::round(float)`` (half away from zero)."""
    q = float(f32(cucumber_string_top / cucumber_string_point_spacing))
    point_count = int(math.floor(abs(q) + 0.5)) * (1 if q >= 0 else -1)
    b = np.asarray(string_base, f32)
    return [(f32(b[0] + f32(0)), f32(b[1] + f32(0)), f32(b[2] + f32(f32(i) * cucumber_string_point_spacing)))
            for i in range(1, point_count + 1)]


@CucumberMainPhytomerCallbackFunction.implement
@ports("CucumberMainPhytomerCallbackFunction")
def _cucumber_main_phytomer_callback(phytomer):
    """AS:4369-4458: harvest full-size fruit; lower and lean the main stem from its tip phytomer."""
    for petiole in phytomer.floral_buds:
        for fbud in petiole:
            if fbud.state == BUD_FRUITING and fbud.current_fruit_scale_factor >= f32(1):
                phytomer.setFloralBudState(BUD_DEAD, fbud)

    stem = phytomer.parent_shoot_ptr
    if stem.parent_shoot_ID >= 0 or not stem.phytomers or phytomer is not stem.phytomers[-1]:
        return

    node_index = [(p, s) for p in range(len(stem.shoot_internode_vertices))
                  for s in range(len(stem.shoot_internode_vertices[p]))]
    if len(node_index) < 2:
        return

    def node(i):
        p, s = node_index[i]
        return stem.shoot_internode_vertices[p][s]

    def set_node(i, v):
        p, s = node_index[i]
        stem.shoot_internode_vertices[p][s] = np.asarray(v, f32).copy()

    tip = len(node_index) - 1
    base = np.asarray(node(0), f32).copy()

    upright_base_height = f32(cucumber_laid_stem_height + cucumber_bend_radius)
    upright_rise = f32(cucumber_wire_height - upright_base_height)
    upright_base = tip
    while upright_base > 0 and f32(node(tip)[2] - node(upright_base)[2]) < upright_rise:
        upright_base -= 1
    if upright_base == 0:
        return

    segment_length = [f32(0)] * (upright_base + 1)
    laid_length = f32(0)
    for i in range(1, upright_base + 1):
        segment_length[i] = V.magnitude(_sub(node(i), node(i - 1)))
        laid_length = f32(laid_length + segment_length[i])

    def path_point(t):
        y, z = cucumberLoweredStemPath(laid_length, t)
        return _add(base, np.array([f32(0), y, z], f32))

    laid_nodes = [None] * (upright_base + 1)
    laid_nodes[0] = base
    t_previous = f32(0)
    for i in range(1, upright_base + 1):
        t_low = f32(t_previous + segment_length[i])
        t_high = f32(t_previous + f32(f32(2) * segment_length[i]))
        for _ in range(40):
            t_mid = f32(f32(0.5) * f32(t_low + t_high))
            if V.magnitude(_sub(path_point(t_mid), laid_nodes[i - 1])) < segment_length[i]:
                t_low = t_mid
            else:
                t_high = t_mid
        t_previous = f32(f32(0.5) * f32(t_low + t_high))
        laid_nodes[i] = path_point(t_previous)

    upright_shift = _sub(laid_nodes[-1], node(upright_base))
    for i in range(1, upright_base + 1):
        set_node(i, laid_nodes[i])
    for i in range(upright_base + 1, tip + 1):
        set_node(i, _add(node(i), upright_shift))
    stem.updateShootNodes(True)

    if f32(laid_nodes[-1][1] - base[1]) > f32(1e-4):
        updateAttractionPoints(stem.plantarchitecture_ptr, stem.plantID,
                               CucumberStringAttractionPoints((base[0], laid_nodes[-1][1], base[2])))


__all__ = ["initializeCucumberShoots", "buildCucumberPlant", "CucumberFruitPrototype", "CucumberFlowerPrototype",
           "CucumberMainPhytomerCreationFunction", "CucumberMainPhytomerCallbackFunction",
           "CucumberStringAttractionPoints", "cucumberLoweredStemPath", "enableAttractionPoints",
           "updateAttractionPoints"]
