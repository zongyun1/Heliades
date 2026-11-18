# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:5366-5508 and src/Assets.cpp:2302-2405, 3555-3947
# (Copyright (C) 2016-2026 Brian Bailey).
"""Rice shoot parameters, builder, phytomer creation function and the procedural panicle (W5-A).

A grass like the maize (``library/maize.py``): sessile flexible strap leaves whose texture ages through three images
(``leaf_texture_series`` with stage ages 0 / 55 / 72 days: green, yellowing, straw), 14 main-stem phytomers, and a hill
of tillers (shoot type ``tiller``; build parameter ``tiller_count`` 0-18, default 12) placed at the positions of
``rice_tiller_positions`` by ``RicePhytomerCreationFunction``, which also sets each blade's length fraction, runs the
internodes from collar to collar (culm internode plus own sheath less the sheath below, except in the crown), sets a
tiller's leaf number from its position, and turns the upper three leaves (and the crown leaves) with
``setGrassFlagLeafAngle`` at a drawn angle. The panicle is procedural (``buildRicePanicle``) in three stages: heading
(closed flower), anthesis (open flower, with anthers) and ripening (fruit), shaped by ``RicePanicleParameters`` from the
plant's organ build parameters.

RNG: build ``make_AxisRotation(0, randu(0, 2 pi), randu(0, 2 pi))``, right to left (roll, then yaw).
"""
from __future__ import annotations

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, deg2rad, make_AxisRotation, make_RGBcolor
from ..random_parameter import PlantArchitectureError
from . import grass_inflorescence as GI
from .build_parameters import (BuildParameterDeclaration, _std_to_string, getBuildParameterIntegerValue,
                               registerPlantModelBuildParameters)
from .maize import setGrassFlagLeafAngle, setInternodeMaxLength
from .strawberry import named_hook

f32 = np.float32
cross, dot, clamp = GI.cross, GI.dot, V.clamp
sinf, cosf = V.sinf, V.cosf
one = f32(1)

#: PlantArchitectureInternal.h: ``rice_tiller_count``
rice_tiller_count = BuildParameterDeclaration("tiller_count", 0.0, 18.0, True, "number of tillers of a rice plant")


# ---- panicle parameters (PlantArchitectureInternal.h:507-625, AS:3555-3666) --------------------------------------

class RicePanicleParameters:
    def __init__(self):
        self.length = f32(0.24)
        self.length_variation = f32(0.06)
        self.rachis_droop = f32(75)
        self.rachis_radius = f32(0.0011)
        self.rachis_tip_radius = f32(0.0004)
        self.primary_branch_count = 10
        self.primary_branch_count_variation = f32(0.12)
        self.branch_length_base = f32(0.115)
        self.branch_length_tip = f32(0.045)
        self.branch_length_variation = f32(0.12)
        self.branch_angle_base = f32(24)
        self.branch_angle_tip = f32(12)
        self.branch_hang = f32(0.85)
        self.branch_bare_fraction = f32(0.07)
        self.branch_radius = f32(0.00035)
        self.secondary_branches_per_primary = f32(2.5)
        self.secondary_zone_fraction = f32(0.5)
        self.secondary_branch_length = f32(0.017)
        self.spikelets_per_secondary = 3
        self.spikelet_spacing = f32(0.0055)
        self.spikelet_length = f32(0.0076)
        self.spikelet_width = f32(0.0028)
        self.spikelet_thickness = f32(0.002)
        self.spikelet_size_variation = f32(0.05)
        self.spikelet_angle = f32(12)
        self.pedicel_length = f32(0.0015)
        self.spikelet_subdivisions = 6
        self.ripeness = f32(0.85)
        self.hull_green_color = GI.rgb(0.42, 0.54, 0.22)
        self.hull_ripe_color = GI.rgb(0.72, 0.60, 0.36)
        self.hull_color_variation = f32(0.08)
        self.anther_fraction = f32(0.3)
        self.anther_length = f32(0.0022)
        self.anther_width = f32(0.0005)
        self.filament_length = f32(0.004)
        self.anther_color = GI.rgb(0.85, 0.78, 0.42)
        self.anther_color_variation = f32(0.08)
        self.branch_color = GI.rgb(0.42, 0.50, 0.22)
        self.branch_radial_subdivisions = 3
        self.rachis_radial_subdivisions = 8

    @staticmethod
    @ports("RicePanicleParameters::erect")
    def erect():
        p = RicePanicleParameters()
        p.length = f32(0.19)
        p.rachis_droop = f32(22)
        p.primary_branch_count = 13
        p.branch_length_base = f32(0.085)
        p.branch_length_tip = f32(0.04)
        p.branch_angle_base = f32(14)
        p.branch_angle_tip = f32(8)
        p.branch_hang = f32(0.1)
        p.secondary_branches_per_primary = f32(2.5)
        p.secondary_branch_length = f32(0.016)
        p.spikelet_spacing = f32(0.0052)
        return p

    @staticmethod
    @ports("RicePanicleParameters::lax")
    def lax():
        p = RicePanicleParameters()
        p.length = f32(0.30)
        p.rachis_droop = f32(125)
        p.primary_branch_count = 10
        p.branch_length_base = f32(0.15)
        p.branch_length_tip = f32(0.06)
        p.branch_angle_base = f32(32)
        p.branch_angle_tip = f32(16)
        p.branch_hang = f32(0.9)
        p.secondary_branches_per_primary = f32(2)
        p.secondary_branch_length = f32(0.028)
        p.spikelet_spacing = f32(0.0085)
        return p

    @staticmethod
    @ports("RicePanicleParameters::fromBuildParameters")
    def fromBuildParameters(build_parameters):
        return GI.fromBuildParameters(RicePanicleParameters, RICE_PANICLE_KEY_TABLES, rice_panicle_type,
                                      rice_spikelet_subdivisions, "spikelet_subdivisions", RicePanicleParameters.erect,
                                      RicePanicleParameters.lax, build_parameters)

    @staticmethod
    @ports("RicePanicleParameters::buildParameterDeclarations")
    def buildParameterDeclarations():
        declarations = [rice_panicle_type, rice_spikelet_subdivisions]
        GI.appendOrganBuildParameterDeclarations(RICE_PANICLE_KEY_TABLES, declarations)
        return declarations


RICE_PANICLE_KEY_TABLES = GI.OrganKeyTables(
    [("panicle_length", "length", 0.05, 0.6, "rice panicle length in meters"),
     ("panicle_length_variation", "length_variation", 0.0, 0.3, "coefficient of variation of rice panicle length"),
     ("rachis_droop", "rachis_droop", 0.0, 180.0, "angle through which the rachis of a ripe rice panicle turns from base to tip in degrees"),
     ("branch_count_variation", "primary_branch_count_variation", 0.0, 0.5, "coefficient of variation of the number of rice panicle primary branches"),
     ("branch_length_base", "branch_length_base", 0.01, 0.4, "length of the lowest rice panicle primary branches in meters"),
     ("branch_length_tip", "branch_length_tip", 0.005, 0.4, "length of the uppermost rice panicle primary branch in meters"),
     ("branch_length_variation", "branch_length_variation", 0.0, 0.5, "coefficient of variation of rice panicle primary branch length"),
     ("branch_angle_base", "branch_angle_base", 0.0, 90.0, "insertion angle of the lowest rice panicle primary branch in degrees from the rachis"),
     ("branch_angle_tip", "branch_angle_tip", 0.0, 90.0, "insertion angle of the uppermost rice panicle primary branch in degrees from the rachis"),
     ("branch_hang", "branch_hang", 0.0, 1.0, "how far a primary branch of a ripe rice panicle turns toward hanging, from 0 to 1"),
     ("branch_bare_fraction", "branch_bare_fraction", 0.0, 0.9, "fraction of each rice panicle primary branch that bears nothing"),
     ("secondary_branches_per_primary", "secondary_branches_per_primary", 0.0, 10.0, "mean number of secondary branches on the longest rice panicle primary branches"),
     ("secondary_zone_fraction", "secondary_zone_fraction", 0.05, 1.0, "fraction of a rice panicle primary branch over which its secondary branches arise"),
     ("secondary_branch_length", "secondary_branch_length", 0.003, 0.1, "length of a rice panicle secondary branch in meters"),
     ("spikelet_spacing", "spikelet_spacing", 0.002, 0.03, "distance along a rice panicle primary branch between successive spikelets in meters"),
     ("spikelet_length", "spikelet_length", 0.004, 0.015, "rice spikelet length in meters"),
     ("spikelet_width", "spikelet_width", 0.001, 0.006, "rice spikelet width in meters"),
     ("spikelet_thickness", "spikelet_thickness", 0.001, 0.005, "rice spikelet thickness in meters"),
     ("spikelet_size_variation", "spikelet_size_variation", 0.0, 0.3, "coefficient of variation of rice spikelet size"),
     ("spikelet_angle", "spikelet_angle", 0.0, 60.0, "angle of a rice spikelet from the branch bearing it in degrees"),
     ("pedicel_length", "pedicel_length", 0.0, 0.015, "length of a rice pedicel in meters"),
     ("ripeness", "ripeness", 0.0, 1.0, "ripeness of the hulls of a rice panicle, from 0 (green) to 1 (fully ripe)"),
     ("hull_color_variation", "hull_color_variation", 0.0, 0.5, "coefficient of variation of rice hull brightness"),
     ("anther_fraction", "anther_fraction", 0.0, 1.0, "fraction of rice spikelets open at anthesis"),
     ("anther_length", "anther_length", 0.0005, 0.006, "rice anther length in meters"),
     ("anther_width", "anther_width", 0.0002, 0.002, "rice anther width in meters"),
     ("filament_length", "filament_length", 0.0, 0.015, "rice filament length in meters"),
     ("anther_color_variation", "anther_color_variation", 0.0, 0.5, "coefficient of variation of rice anther brightness")],
    [("branch_count", "primary_branch_count", 1, 40, "mean number of rice panicle primary branches"),
     ("spikelets_per_secondary", "spikelets_per_secondary", 1, 12, "number of spikelets on a rice panicle secondary branch")],
    [("hull_green_color_r", "hull_green_color", 0, "red component of the hull color of an unripe rice spikelet"),
     ("hull_green_color_g", "hull_green_color", 1, "green component of the hull color of an unripe rice spikelet"),
     ("hull_green_color_b", "hull_green_color", 2, "blue component of the hull color of an unripe rice spikelet"),
     ("hull_ripe_color_r", "hull_ripe_color", 0, "red component of the hull color of a ripe rice spikelet"),
     ("hull_ripe_color_g", "hull_ripe_color", 1, "green component of the hull color of a ripe rice spikelet"),
     ("hull_ripe_color_b", "hull_ripe_color", 2, "blue component of the hull color of a ripe rice spikelet"),
     ("anther_color_r", "anther_color", 0, "red component of the rice anther color"),
     ("anther_color_g", "anther_color", 1, "green component of the rice anther color"),
     ("anther_color_b", "anther_color", 2, "blue component of the rice anther color")])

rice_panicle_type = BuildParameterDeclaration("panicle_type", 0.0, 1.0, False,
                                              "rice panicle type from 0 (erect) through 0.5 (the default) to 1 (lax)")
rice_spikelet_subdivisions = BuildParameterDeclaration("spikelet_subdivisions", 3.0, 12.0, True,
                                                       "number of facets around a rice spikelet")


def _validate_rice_panicle_parameters(parameters):
    RicePanicleParameters.fromBuildParameters(parameters)


def register_build_parameters(pa):
    """PL:531-532."""
    registerPlantModelBuildParameters(pa, "rice", [rice_tiller_count], RicePanicleParameters.buildParameterDeclarations(),
                                      _validate_rice_panicle_parameters)


# ---- shoots and builder (PL:5366-5508) ----------------------------------------------------------------------------

@ports("PlantArchitecture::initializeRiceShoots")
def initializeRiceShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.leaf_texture_series[0] = [["RiceLeaf_A_0.png", "RiceLeaf_A_1.png", "RiceLeaf_A_2.png"],
                                             ["RiceLeaf_B_0.png", "RiceLeaf_B_1.png", "RiceLeaf_B_2.png"]]
    leaf_prototype.leaf_texture_stage_ages = [f32(0), f32(55), f32(72)]
    leaf_prototype.leaf_aspect_ratio = f32(0.04)
    leaf_prototype.midrib_fold_fraction = f32(0.15)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.06, -0.02)
    leaf_prototype.longitudinal_curvature_exponent = f32(3)
    leaf_prototype.flexibility_taper = f32(150)
    leaf_prototype.flexibility_aging = f32(11)
    leaf_prototype.flexibility_aging_max = f32(5)
    leaf_prototype.lateral_curvature = f32(-0.2)
    leaf_prototype.petiole_roll = f32(0)
    leaf_prototype.wave_period = f32(0.1)
    leaf_prototype.wave_amplitude = f32(0.02)
    leaf_prototype.flexibility.uniformDistribution(1.0, 1.6)
    leaf_prototype.subdivisions = 20
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)
    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(170, 190)
    pp.internode.radius_initial = 0.002
    pp.internode.color = make_RGBcolor(0.25, 0.42, 0.16)
    pp.internode.length_segments = 3
    pp.internode.radial_subdivisions = 8
    pp.internode.max_floral_buds_per_petiole = 0
    pp.internode.max_vegetative_buds_per_petiole = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.length = 0

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-28, -14)
    pp.leaf.yaw = 0
    pp.leaf.roll.uniformDistribution(-10, 10)
    pp.leaf.prototype_scale = 0.43
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length.uniformDistribution(0.04, 0.09)
    pp.peduncle.radius = 0.0011
    pp.peduncle.pitch.uniformDistribution(3, 10)
    pp.peduncle.color = make_RGBcolor(*RicePanicleParameters().branch_color)
    pp.peduncle.curvature = 0
    pp.peduncle.radial_subdivisions = 8
    pp.peduncle.length_segments = 8

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.flower_prototype_scale = 1
    pp.inflorescence.flower_prototype_function = named_hook("RiceFloweringPaniclePrototype")
    pp.inflorescence.fruit_prototype_scale = 1
    pp.inflorescence.fruit_prototype_function = named_hook("RicePaniclePrototype")
    pp.inflorescence.inflorescence_maturity_period = 0
    pp.inflorescence.unique_prototypes = 3

    pp.phytomer_creation_function = named_hook("RicePhytomerCreationFunction")

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.vegetative_bud_break_probability_min = 1
    sp.vegetative_bud_break_time = 18
    sp.flower_bud_break_probability = 1
    sp.phyllochron_min = 6
    sp.elongation_rate_max = 0.05
    sp.leaf_expansion_rate_max = 0.1
    sp.girth_area_factor = f32(0.8)
    sp.gravitropic_curvature.uniformDistribution(-300, -100)
    sp.internode_length_max = 0.05
    sp.internode_length_decay_rate = 0
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False
    sp.fruit_set_probability = 1.0
    sp.defineChildShootTypes(["tiller"], [1.0])
    sp.max_nodes = 14
    sp.max_terminal_floral_buds = 1

    st = sp.copy()                                                               # copy ctor: no draw
    st.defineChildShootTypes(["tiller"], [1.0])
    st.insertion_angle_tip.uniformDistribution(20, 32)
    st.base_yaw.uniformDistribution(-30, 30)
    st.base_roll.uniformDistribution(-20, 20)
    st.gravitropic_curvature.uniformDistribution(4000, 6000)

    pa.defineShootType("mainstem", sp)
    pa.defineShootType("tiller", st)


@ports("riceTillerCount")
def riceTillerCount(build_parameters) -> int:
    return getBuildParameterIntegerValue(build_parameters, rice_tiller_count, 12)


@ports("PlantArchitecture::buildRicePlant")
def buildRicePlant(pa, base_position):
    """PL:5485-5508."""
    if not pa.shoot_types:
        initializeRiceShoots(pa)
    riceTillerCount(pa.current_build_parameters)
    plantID = pa.addPlantInstance(np.asarray(base_position, f32) - np.array([0, 0, 0.01], f32), 0)
    ctx = pa.context_ptr
    internode_length_max = pa.shoot_types["mainstem"].internode_length_max.val()
    roll = ctx.randu(f32(0), f32(2.0 * np.pi))                                  # arguments right to left
    yaw = ctx.randu(f32(0), f32(2.0 * np.pi))
    pa.addBaseStemShoot(plantID, 1, make_AxisRotation(f32(0), yaw, roll), 0.002, internode_length_max, 0.01, 0.01, 0,
                        "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, 4, 3, 7, 30, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


# ---- phytomer creation function (AS:2302-2405) --------------------------------------------------------------------

_BLADE_FRACTION = [f32(x) for x in (0.78, 1.0, 0.99, 0.93, 0.84, 0.74, 0.63, 0.52)]
_SHEATH_LENGTH = [f32(x) for x in (0.24, 0.25, 0.24, 0.22, 0.20, 0.17, 0.14, 0.11)]
_CULM_LENGTH = [f32(x) for x in (0.18, 0.10, 0.055, 0.035, 0.012)]
_UPPER_LEAF_ANGLE = [f32(x) for x in (9.4, 9.3, 12.4)]

#: (main_stem_leaf, primary_tiller_leaf) in emergence order (AS:2342)
rice_tiller_positions = ((2, 0), (3, 0), (4, 0), (5, 0), (2, 1), (6, 0), (2, 2), (3, 1), (7, 0), (2, 3), (3, 2), (4, 1),
                         (2, 4), (3, 3), (4, 2), (5, 1), (3, 4), (4, 3))


def _ranks_from_top(node_index, shoot_max_nodes):
    return max(0, int(shoot_max_nodes) - 1 - int(node_index))


def riceBladeLengthFraction(node_index, shoot_max_nodes) -> np.float32:
    r = _ranks_from_top(node_index, shoot_max_nodes)
    if r < len(_BLADE_FRACTION):
        return _BLADE_FRACTION[r]
    return max(f32(0.2), f32(_BLADE_FRACTION[-1] - f32(f32(0.1) * f32(r - len(_BLADE_FRACTION) + 1))))


def riceSheathLength(node_index, shoot_max_nodes) -> np.float32:
    r = _ranks_from_top(node_index, shoot_max_nodes)
    if r < len(_SHEATH_LENGTH):
        return _SHEATH_LENGTH[r]
    return max(f32(0.03), f32(_SHEATH_LENGTH[-1] - f32(f32(0.02) * f32(r - len(_SHEATH_LENGTH) + 1))))


def riceCulmInternodeLength(node_index, shoot_max_nodes) -> np.float32:
    r = _ranks_from_top(node_index, shoot_max_nodes)
    return _CULM_LENGTH[r] if r < len(_CULM_LENGTH) else f32(0.003)


@named_hook("RicePhytomerCreationFunction").implement
@ports("RicePhytomerCreationFunction")
def RicePhytomerCreationFunction(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    """AS:2348-2405."""
    shoot_node_index, shoot_max_nodes = int(shoot_node_index), int(shoot_max_nodes)
    parent_shoot_node_index = int(parent_shoot_node_index)
    shoot = phytomer.parent_shoot_ptr
    pa = shoot.plantarchitecture_ptr
    tiller_count = riceTillerCount(pa.getPlantBuildParameters(phytomer.plantID))
    positions = rice_tiller_positions[:tiller_count]

    if shoot_node_index == 0 and phytomer.rank > 0:
        parent_shoot = pa.plant_instances[phytomer.plantID].shoot_tree[shoot.parent_shoot_ID]
        tiller_max_nodes = max(3, int(parent_shoot.shoot_parameters.max_nodes.val()) - (parent_shoot_node_index + 1) - 2)
        shoot.shoot_parameters.max_nodes = tiller_max_nodes
        phytomer.shoot_index[2] = tiller_max_nodes
        shoot_max_nodes = tiller_max_nodes

    crown_top = 7 if phytomer.rank == 0 else (4 if phytomer.rank == 1 else 0)

    def node_is_in_crown(node_index):
        return node_index + 1 <= crown_top

    bud_makes_tiller = False
    if phytomer.rank == 0:
        bud_makes_tiller = any(t == 0 and m == shoot_node_index + 1 for m, t in positions)
    elif phytomer.rank == 1:
        main_stem_leaf = parent_shoot_node_index + 1
        bud_makes_tiller = any(m == main_stem_leaf and t == shoot_node_index + 1 for m, t in positions)
    if not bud_makes_tiller:
        phytomer.setVegetativeBudState(BUD_DEAD)

    phytomer.scaleLeafPrototypeScale(riceBladeLengthFraction(shoot_node_index, shoot_max_nodes))

    model_internode_length = f32(0.004)
    if not node_is_in_crown(shoot_node_index):
        if shoot_node_index == 0 or node_is_in_crown(shoot_node_index - 1):
            sheath_length_below = f32(0)
        else:
            sheath_length_below = riceSheathLength(shoot_node_index - 1, shoot_max_nodes)
        model_internode_length = max(f32(0.002), f32(f32(riceCulmInternodeLength(shoot_node_index, shoot_max_nodes)
                                                         + riceSheathLength(shoot_node_index, shoot_max_nodes))
                                                     - sheath_length_below))
    setInternodeMaxLength(phytomer, model_internode_length)

    ctx = pa.context_ptr
    leaf_ranks_from_top = shoot_max_nodes - 1 - shoot_node_index
    if shoot_max_nodes > 0 and leaf_ranks_from_top < len(_UPPER_LEAF_ANGLE):
        setGrassFlagLeafAngle(phytomer, f32(_UPPER_LEAF_ANGLE[leaf_ranks_from_top] + ctx.randu(f32(-3), f32(5))))
    elif node_is_in_crown(shoot_node_index):
        setGrassFlagLeafAngle(phytomer, ctx.randu(f32(3), f32(12)))


# ---- panicle (AS:3668-3947) ----------------------------------------------------------------------------------------

HEADING, ANTHESIS, RIPENING = "heading", "anthesis", "ripening"


@ports("addRiceSpikelet")
def addRiceSpikelet(ctx, center, spikelet_axis, width_axis, length, width, thickness, facets, color, mesh):
    """AS:3668-3712."""
    thickness_axis = cross(spikelet_axis, width_axis)
    half_length = f32(f32(0.5) * length)
    ring_height = (f32(-0.80), f32(-0.10), f32(0.66))
    ring_profile = (f32(0.70), f32(1), f32(0.80))
    half_w, half_t = f32(f32(0.5) * width), f32(f32(0.5) * thickness)
    base_pole = mesh.addVertex(center - spikelet_axis * half_length)
    apex_pole = mesh.addVertex(center + spikelet_axis * half_length)
    rings = ([], [], [])
    for ring in range(3):
        along = spikelet_axis * f32(ring_height[ring] * half_length)
        wr, tr = f32(half_w * ring_profile[ring]), f32(half_t * ring_profile[ring])
        for facet in range(facets):
            angle = f32(f32(GI.TWO_PI_F * f32(f32(facet) + f32(f32(0.5) * f32(ring % 2)))) / f32(facets))
            rings[ring].append(mesh.addVertex(((center + along) + width_axis * f32(wr * cosf(angle)))
                                              + thickness_axis * f32(tr * sinf(angle))))
    lemma_color = GI.rgb_scale(color, 1.05)
    palea_color = GI.rgb_scale(color, 0.88)
    apex_color = GI.rgb_scale(color, 0.86)
    for facet in range(facets):
        nxt = (facet + 1) % facets
        body = lemma_color if facet < (facets + 1) // 2 else palea_color
        mesh.addTriangle(ctx, base_pole, rings[0][nxt], rings[0][facet], body)
        mesh.addTriangle(ctx, rings[0][facet], rings[0][nxt], rings[1][facet], body)
        mesh.addTriangle(ctx, rings[1][facet], rings[0][nxt], rings[1][nxt], body)
        mesh.addTriangle(ctx, rings[1][facet], rings[2][nxt], rings[2][facet], body)
        mesh.addTriangle(ctx, rings[1][facet], rings[1][nxt], rings[2][nxt], body)
        mesh.addTriangle(ctx, apex_pole, rings[2][facet], rings[2][nxt], apex_color)
    sterile_lemma_color = GI.rgb_scale(color, 1.2)
    base = center - spikelet_axis * half_length
    t_off = thickness_axis * f32(f32(0.12) * thickness)
    tip_off = spikelet_axis * f32(f32(0.25) * length)
    for side in (f32(-1), f32(1)):
        outer = width_axis * f32(f32(side * f32(0.28)) * width)
        inner = base + outer * f32(0.2)
        mesh.addIsolatedTriangle(ctx, inner - t_off, inner + t_off, base + outer + tip_off, sterile_lemma_color)


@ports("buildRicePanicle")
def buildRicePanicle(ctx, parameters, stage):
    """AS:3714-3939. Draws in C++ statement order; arguments and ``make_vec3(0, randu, randu)`` right to left."""
    p = parameters
    if (p.length <= 0 or p.spikelet_length <= 0 or p.spikelet_width <= 0 or p.spikelet_thickness <= 0
            or p.rachis_radius <= 0 or p.rachis_tip_radius <= 0 or p.branch_radius <= 0):
        raise PlantArchitectureError("ERROR (buildRicePanicle): Panicle length, rachis and branch radii, and spikelet "
                                     "dimensions must all be positive.")
    elif p.primary_branch_count < 1:
        raise PlantArchitectureError("ERROR (buildRicePanicle): primary_branch_count must be at least 1.")
    elif p.spikelet_spacing <= 0:
        raise PlantArchitectureError("ERROR (buildRicePanicle): spikelet_spacing must be positive.")
    elif p.branch_length_base <= 0 or p.branch_length_tip <= 0:
        raise PlantArchitectureError("ERROR (buildRicePanicle): Primary branch lengths must be positive.")
    elif p.branch_bare_fraction < 0 or p.branch_bare_fraction >= 1:
        raise PlantArchitectureError("ERROR (buildRicePanicle): branch_bare_fraction must be in the range [0,1), but "
                                     f"{_std_to_string(p.branch_bare_fraction)} was given.")
    elif p.secondary_branches_per_primary < 0 or p.secondary_branch_length <= 0 or p.spikelets_per_secondary < 1:
        raise PlantArchitectureError("ERROR (buildRicePanicle): secondary_branches_per_primary cannot be negative, "
                                     "secondary_branch_length must be positive, and spikelets_per_secondary must be at "
                                     "least 1.")
    elif p.secondary_zone_fraction <= 0 or p.secondary_zone_fraction > 1:
        raise PlantArchitectureError("ERROR (buildRicePanicle): secondary_zone_fraction must be greater than 0 and at most "
                                     f"1, but {_std_to_string(p.secondary_zone_fraction)} was given.")
    elif not (0 <= p.branch_hang <= 1 and 0 <= p.ripeness <= 1 and 0 <= p.anther_fraction <= 1):
        raise PlantArchitectureError("ERROR (buildRicePanicle): branch_hang, ripeness and anther_fraction must each be in "
                                     "the range [0,1].")
    elif p.anther_fraction > 0 and (p.anther_length <= 0 or p.anther_width <= 0):
        raise PlantArchitectureError("ERROR (buildRicePanicle): anther_length and anther_width must be positive when "
                                     "there are anthers.")
    elif p.filament_length < 0 or p.pedicel_length < 0:
        raise PlantArchitectureError("ERROR (buildRicePanicle): filament_length and pedicel_length cannot be negative.")
    elif p.spikelet_subdivisions < 3 or p.branch_radial_subdivisions < 3 or p.rachis_radial_subdivisions < 3:
        raise PlantArchitectureError("ERROR (buildRicePanicle): spikelet_subdivisions, branch_radial_subdivisions and "
                                     "rachis_radial_subdivisions must each be at least 3.")

    droop_fraction = hang_fraction = spread_fraction = one
    if stage == HEADING:
        droop_fraction, hang_fraction, spread_fraction = f32(0.1), f32(0.1), f32(0.35)
    elif stage == ANTHESIS:
        droop_fraction, hang_fraction, spread_fraction = f32(0.3), f32(0.25), f32(0.75)

    size_factor = clamp(f32(one + f32(p.length_variation * ctx.randn())), f32(0.7), f32(1.3))
    panicle_length = f32(p.length * size_factor)
    rachis_bend = deg2rad(f32(p.rachis_droop * droop_fraction))
    branch_hang = f32(p.branch_hang * hang_fraction)
    gravity = GI.vec(-1, 0, 0)
    binormal = GI.vec(0, 1, 0)

    # ---- rachis ----
    rachis_segments = 20
    nseg = f32(rachis_segments)
    exponent = f32(1.4)

    def bend_angle(s):
        return f32(rachis_bend * GI.powf(clamp(f32(s / panicle_length), f32(0), one), exponent))

    def rachis_tangent(s):
        a = bend_angle(s)
        return GI.vec(cosf(a), 0, -sinf(a))

    def rachis_normal(s):
        a = bend_angle(s)
        return GI.vec(sinf(a), 0, cosf(a))

    radius_span = f32(p.rachis_radius - p.rachis_tip_radius)

    def rachis_radius_at(s):
        return f32(p.rachis_tip_radius + f32(radius_span * f32(one - f32(s / panicle_length))))

    rachis_nodes = [GI.vec(0, 0, 0)]
    rachis_radii = [p.rachis_radius]
    segment_length = f32(panicle_length / nseg)
    for node in range(1, rachis_segments + 1):
        rachis_nodes.append(rachis_nodes[node - 1]
                            + rachis_tangent(f32(f32(f32(node) - f32(0.5)) * segment_length)) * segment_length)
        rachis_radii.append(rachis_radius_at(f32(f32(node) * segment_length)))

    def rachis_position(s):
        position = f32(clamp(f32(s / panicle_length), f32(0), one) * nseg)
        segment = min(int(position), rachis_segments - 1)
        return rachis_nodes[segment] + (rachis_nodes[segment + 1] - rachis_nodes[segment]) * f32(position - f32(segment))

    mesh = GI.GrassInflorescenceMesh()
    mesh.addTube(ctx, GI.context_add_tube(ctx, p.rachis_radial_subdivisions, rachis_nodes, rachis_radii,
                                          [p.branch_color] * len(rachis_nodes)))

    # ---- spikelets ----
    ripeness_step = f32(0.2)
    unripe = f32(one - p.ripeness)
    hull_delta = p.hull_ripe_color - p.hull_green_color

    def hull_color_at(height_on_panicle):
        ripe_weight = f32(0)
        if stage == RIPENING:
            lag = f32(f32(one - f32(height_on_panicle)) * unripe)
            v = clamp(f32(f32(p.ripeness - lag) + ctx.randu(f32(-0.1), f32(0.1))), f32(0), one)
            ripe_weight = f32(GI.roundf(f32(v / ripeness_step)) * ripeness_step)
        c = p.hull_green_color + hull_delta * ripe_weight
        c = np.where(c < 0, f32(0), np.where(c > 1, f32(1), c)).astype(f32)          # make_RGBcolor clamps
        return GI.rgb_scale(c, GI.quantizedBrightness(ctx, p.hull_color_variation))

    pedicel_half = f32(f32(0.5) * f32(0.0005))

    def add_spikelet(attachment, spikelet_axis, height_on_panicle):
        scale = clamp(f32(one + f32(p.spikelet_size_variation * ctx.randn())), f32(0.8), f32(1.2))
        length = f32(p.spikelet_length * scale)
        hull_base = attachment + spikelet_axis * p.pedicel_length
        center = hull_base + spikelet_axis * f32(f32(0.5) * length)
        reference = GI.perpendicularTo(spikelet_axis)
        twist = ctx.randu(f32(0), GI.TWO_PI_F)
        width_axis = GI.normalize(reference * cosf(twist) + cross(spikelet_axis, reference) * sinf(twist))
        if p.pedicel_length > 0:
            mesh.addIsolatedTriangle(ctx, attachment - width_axis * pedicel_half, attachment + width_axis * pedicel_half,
                                     hull_base, p.branch_color)
        color = hull_color_at(height_on_panicle)
        addRiceSpikelet(ctx, center, spikelet_axis, width_axis, length, f32(p.spikelet_width * scale),
                        f32(p.spikelet_thickness * scale), int(p.spikelet_subdivisions), color, mesh)
        if stage == ANTHESIS and ctx.randu() < p.anther_fraction:
            floret_point = center + spikelet_axis * f32(f32(0.25) * length)
            for _anther in range(6):
                sz = ctx.randu(f32(-0.35), f32(0.35))
                sy = ctx.randu(f32(-0.35), f32(0.35))
                hang_direction = GI.normalize(gravity + GI.vec(0, sy, sz))
                anther_color = GI.rgb_scale(p.anther_color, GI.quantizedBrightness(ctx, p.anther_color_variation))
                anther_length = f32(p.anther_length * ctx.randu(f32(0.9), f32(1.1)))
                filament_length = f32(p.filament_length * ctx.randu(f32(0.7), f32(1.3)))
                GI.addGrassAnther(ctx, floret_point, hang_direction, hang_direction, filament_length, anther_length,
                                  p.anther_width, anther_color, mesh)

    # ---- branches ----
    def add_branch_curve(base, initial_direction, length, base_radius, hang, segments):
        nodes, tangents = [base], []
        fs = f32(segments)
        step = f32(length / fs)
        for segment in range(segments):
            position = f32(f32(f32(segment) + f32(0.5)) / fs)
            weight = f32(hang * GI.powf(position, f32(0.8)))
            tangents.append(GI.normalize(initial_direction * f32(one - weight) + gravity * weight))
            nodes.append(nodes[segment] + tangents[segment] * step)
        radii = [f32(base_radius * f32(one - f32(f32(f32(0.5) * f32(node)) / fs))) for node in range(segments + 1)]
        mesh.addTube(ctx, GI.context_add_tube(ctx, p.branch_radial_subdivisions, nodes, radii,
                                              [p.branch_color] * len(nodes)))
        return nodes, tangents, f32(length)

    def point_on_curve(curve, arc_length):
        nodes, tangents, length = curve
        segments = len(tangents)
        position = f32(clamp(f32(arc_length / length), f32(0), one) * f32(segments))
        segment = min(int(position), segments - 1)
        point = nodes[segment] + (nodes[segment + 1] - nodes[segment]) * f32(position - f32(segment))
        return point, tangents[segment]

    spikelet_angle = deg2rad(p.spikelet_angle)

    def add_branch_spikelets(curve, first_arc, last_arc, count, side_axis, height_on_panicle):
        span = f32(last_arc - first_arc)
        for spikelet in range(count):
            is_terminal = spikelet == count - 1
            fraction = f32(f32(spikelet) / f32(count - 1)) if count > 1 else one
            point, tangent = point_on_curve(curve, f32(first_arc + f32(span * fraction)))
            side = one if spikelet % 2 == 0 else f32(-1)
            jitter = deg2rad(ctx.randu(f32(-6), f32(6)))
            lateral = GI.normalize(side_axis - tangent * dot(side_axis, tangent)) * side
            if is_terminal:
                spikelet_axis = tangent
            else:
                a = f32(spikelet_angle + jitter)
                spikelet_axis = GI.normalize(tangent * cosf(a) + lateral * sinf(a))
            add_spikelet(point, spikelet_axis, height_on_panicle)

    bc = f32(f32(p.primary_branch_count) * f32(one + f32(p.primary_branch_count_variation * ctx.randn())))
    primary_branch_count = max(1, int(GI.roundf(bc)))
    azimuth_offset = ctx.randu(f32(0), GI.TWO_PI_F)
    golden = deg2rad(f32(137.5))
    for b in range(primary_branch_count):
        position = f32(f32(b) / f32(primary_branch_count - 1)) if primary_branch_count > 1 else one
        crowded_position = GI.powf(position, f32(1.25))
        if b + 1 == primary_branch_count:
            jittered_position = one
        else:
            jittered_position = clamp(f32(crowded_position + f32(ctx.randu(f32(-0.2), f32(0.2)) / f32(primary_branch_count))),
                                      f32(0), one)
        arc_length = f32(jittered_position * panicle_length)

        length = f32(f32(size_factor * f32(p.branch_length_base + f32(f32(p.branch_length_tip - p.branch_length_base) * position)))
                     * clamp(f32(one + f32(p.branch_length_variation * ctx.randn())), f32(0.6), f32(1.4)))
        insertion_angle = deg2rad(clamp(f32(f32(f32(p.branch_angle_base + f32(f32(p.branch_angle_tip - p.branch_angle_base)
                                                                              * position)) * spread_fraction)
                                            + ctx.randu(f32(-4), f32(4))), f32(0), f32(90)))
        azimuth = f32(f32(azimuth_offset + f32(golden * f32(b))) + deg2rad(ctx.randu(f32(-15), f32(15))))

        tangent = rachis_tangent(arc_length)
        outward = rachis_normal(arc_length) * cosf(azimuth) + binormal * sinf(azimuth)
        initial_direction = GI.normalize(tangent * cosf(insertion_angle) + outward * sinf(insertion_angle))
        base = rachis_position(arc_length) + outward * f32(f32(0.7) * rachis_radius_at(arc_length))
        hang = f32(branch_hang * ctx.randu(f32(0.8), f32(1.2)))
        primary = add_branch_curve(base, initial_direction, length, p.branch_radius, clamp(hang, f32(0), one), 8)

        branch_side = GI.normalize(cross(initial_direction, outward))

        bare_length = f32(p.branch_bare_fraction * length)
        zone_end = max(bare_length, f32(p.secondary_zone_fraction * length))
        sc = f32(f32(f32(p.secondary_branches_per_primary * length) / f32(size_factor * p.branch_length_base))
                 * ctx.randu(f32(0.8), f32(1.2)))
        secondary_count = int(GI.roundf(sc))
        zone = f32(zone_end - bare_length)
        for s in range(secondary_count):
            arc = f32(bare_length + f32(f32(zone * f32(f32(s) + f32(0.5))) / f32(secondary_count)))
            point, primary_tangent = point_on_curve(primary, arc)
            side = one if s % 2 == 0 else f32(-1)
            lateral = GI.normalize(branch_side - primary_tangent * dot(branch_side, primary_tangent)) * side
            secondary_angle = deg2rad(f32(ctx.randu(f32(14), f32(28)) * spread_fraction))
            secondary_direction = GI.normalize(primary_tangent * cosf(secondary_angle) + lateral * sinf(secondary_angle))
            secondary_length = f32(f32(p.secondary_branch_length * size_factor) * ctx.randu(f32(0.8), f32(1.2)))
            secondary = add_branch_curve(point, secondary_direction, secondary_length, f32(f32(0.6) * p.branch_radius),
                                         clamp(hang, f32(0), one), 3)
            add_branch_spikelets(secondary, f32(f32(0.15) * secondary[2]), secondary[2], int(p.spikelets_per_secondary),
                                 cross(secondary_direction, lateral), jittered_position)

        direct_start = zone_end if secondary_count > 0 else bare_length
        direct_count = max(1, int(GI.roundf(f32(f32(length - direct_start) / p.spikelet_spacing))) + 1)
        add_branch_spikelets(primary, direct_start, length, direct_count, branch_side, jittered_position)

    return mesh.finish(ctx)


@named_hook("RicePaniclePrototype").implement
@ports("RicePaniclePrototype")
def RicePaniclePrototype(scene, subdivisions, build_parameters=None):
    """AS:3941-3943: the ripening panicle (fruit)."""
    return buildRicePanicle(scene, RicePanicleParameters.fromBuildParameters(build_parameters or {}), RIPENING)


@named_hook("RiceFloweringPaniclePrototype").implement
@ports("RiceFloweringPaniclePrototype")
def RiceFloweringPaniclePrototype(scene, subdivisions, flower_is_open, build_parameters=None):
    """AS:3945-3947: anthesis when open, heading when closed."""
    return buildRicePanicle(scene, RicePanicleParameters.fromBuildParameters(build_parameters or {}),
                            ANTHESIS if flower_is_open else HEADING)
