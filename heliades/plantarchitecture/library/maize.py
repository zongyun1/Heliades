# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:4574-4692 and src/Assets.cpp:2007-2110, 3086-3600
# (Copyright (C) 2016-2026 Brian Bailey).
"""Maize shoot parameters, builder and phytomer creation function.

``initializeMaizeShoots`` / ``buildMaizePlant`` / ``MaizePhytomerCreationFunction`` line for line, literals with
their C++ types. A maize phytomer has a sessile strap leaf (``petiole.length = 0``: the blade sits on the culm
at its drawn leaf pitch), a flexible blade that droops under its own weight (``LeafPrototype::flexibility``
U(1.5, 2.3), with taper and ageing), a midrib class map (``SorghumLeaf_classes.png``), and at most one floral
bud: the ear at node ``max_nodes - 6`` (made ACTIVE by the creation function, set as fruit after 4 days) and the
tassel as the shoot's terminal bud (``max_terminal_floral_buds = 1``). The placement of ear and tassel is the
inflorescence machinery's (W3-D); their prototype functions are bound here. The tassel is procedural
(``buildMaizeTassel``, AS:3086-3553, W5-A): axis, central spike, primary and secondary branches, spikelets and anthers,
shaped by ``MaizeTasselParameters`` from the plant's organ build parameters (``tassel_*`` keys, registered by
``register_build_parameters``).

RNG on load: four identifier draws, like the tomato. Build: ``make_AxisRotation(randu(0, 0.035 pi), randu(0, 2 pi),
randu(0, 2 pi))`` evaluated right to left (roll, yaw, then pitch); ``internode_length_max.val()`` is constant.
"""
from __future__ import annotations

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_ACTIVE, BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, deg2rad, make_AxisRotation, make_RGBcolor
from ..random_parameter import PlantArchitectureError
from . import grass_inflorescence as GI
from .build_parameters import BuildParameterDeclaration, _std_to_string, registerPlantModelBuildParameters
from .strawberry import named_hook, obj_prototype

f32 = np.float32
cross, dot, clamp = GI.cross, GI.dot, V.clamp
sinf, cosf = V.sinf, V.cosf

#: MaizeTasselParameters::branch_color (PlantArchitectureInternal.h:465)
MAIZE_TASSEL_BRANCH_COLOR = (0.34, 0.42, 0.20)


@ports("PlantArchitecture::initializeMaizeShoots")
def initializeMaizeShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)                                          # identifier draw 1
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png"
    leaf_prototype.leaf_class_map_file[0] = "SorghumLeaf_classes.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.15)
    leaf_prototype.midrib_fold_fraction = f32(0.3)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.07, -0.04)
    leaf_prototype.longitudinal_curvature_exponent = f32(3)
    leaf_prototype.flexibility_taper = f32(150)
    leaf_prototype.flexibility_aging = f32(11)
    leaf_prototype.flexibility_aging_max = f32(6)
    leaf_prototype.lateral_curvature = f32(-0.3)
    leaf_prototype.petiole_roll = f32(0)
    leaf_prototype.wave_period = f32(0.1)
    leaf_prototype.wave_amplitude = f32(0.06)
    leaf_prototype.flexibility.uniformDistribution(1.5, 2.3)
    leaf_prototype.subdivisions = 50
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)                                                 # identifier draws 2, 3

    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(170, 190)
    pp.internode.radius_initial = 0.0075
    pp.internode.color = make_RGBcolor(0.126, 0.182, 0.084)
    pp.internode.length_segments = 2
    pp.internode.radial_subdivisions = 10
    pp.internode.max_floral_buds_per_petiole = 1
    pp.internode.max_vegetative_buds_per_petiole = 0

    pp.petiole.petioles_per_internode = 1
    pp.petiole.length = 0

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-30, -16)
    pp.leaf.yaw = 0
    pp.leaf.roll = 0
    pp.leaf.prototype_scale = 0.9
    pp.leaf.prototype = leaf_prototype                                           # identifier draw 4

    pp.peduncle.length = f32(0.14)
    pp.peduncle.radius = 0.004
    pp.peduncle.curvature = 0
    pp.peduncle.color = make_RGBcolor(*(f32(c) for c in MAIZE_TASSEL_BRANCH_COLOR))   # MaizeTasselParameters().branch_color
    pp.peduncle.radial_subdivisions = 10
    pp.peduncle.length_segments = 8

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.fruit_prototype_scale = 1
    pp.inflorescence.fruit_prototype_function = named_hook("MaizeTasselPrototype")
    pp.inflorescence.unique_prototypes = 3
    pp.inflorescence.inflorescence_maturity_period = 6

    pp.phytomer_creation_function = named_hook("MaizePhytomerCreationFunction")

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.vegetative_bud_break_probability_min = 0.5
    sp.flower_bud_break_probability = 1
    sp.phyllochron_min = 3
    sp.elongation_rate_max = 0.1
    sp.girth_area_factor = f32(6)
    sp.gravitropic_curvature.uniformDistribution(-250, 0)
    sp.internode_length_max = 0.1
    sp.tortuosity = 10
    sp.internode_length_decay_rate = 0
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False
    sp.flower_bud_break_probability = 1.0
    sp.fruit_set_probability = 1.0
    sp.defineChildShootTypes(["mainstem"], [1.0])
    sp.max_nodes = 20
    sp.max_terminal_floral_buds = 1

    pa.defineShootType("mainstem", sp)


@ports("PlantArchitecture::buildMaizePlant")
def buildMaizePlant(pa, base_position):
    """PL:4672-4692. The base rotation's three ``randu`` are drawn right to left (GCC): roll, yaw, pitch.
    ``0.035f * M_PI`` and ``2.f * M_PI`` are double products narrowed by ``randu(float, float)``."""
    if not pa.shoot_types:
        initializeMaizeShoots(pa)
    plantID = pa.addPlantInstance(base_position, 0)
    ctx = pa.context_ptr
    # addBaseStemShoot's arguments, right to left: the constant internode_length_max.val() makes no draw
    internode_length_max = pa.shoot_types["mainstem"].internode_length_max.val()
    roll = ctx.randu(f32(0), f32(2.0 * np.pi))
    yaw = ctx.randu(f32(0), f32(2.0 * np.pi))
    pitch = ctx.randu(f32(0), f32(float(f32(0.035)) * np.pi))
    pa.addBaseStemShoot(plantID, 1, make_AxisRotation(pitch, yaw, roll), 0.003, internode_length_max, 0.01, 0.01, 0.2,
                        "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 58, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


# ---- phytomer creation function (AS:2014-2110) -----------------------------------------------------

def setGrassFlagLeafAngle(phytomer, angle_off_culm_degrees):
    """AS:2014-2027 (file-static): turn the flag leaf to ``angle`` off the culm with ``Phytomer::rotateLeaf``."""
    if not phytomer.leaf_objIDs or not phytomer.leaf_objIDs[0]:
        return
    drawn_angle_off_culm = abs(f32(phytomer.petiole_pitch[0]))                       # std::fabs(float)
    phytomer.rotateLeaf(0, 0, make_AxisRotation(f32(drawn_angle_off_culm - deg2rad(f32(angle_off_culm_degrees))), 0, 0))


@ports("Phytomer::setInternodeMaxLength")
def setInternodeMaxLength(phytomer, internode_length_max_new):
    """PA:4360-4363: ``scaleInternodeMaxLength(new / internode_length_max)`` (a free function here so that the shared
    ``phytomer.py`` needs no edit; used by the wheat, rice and sorghum creation functions)."""
    phytomer.scaleInternodeMaxLength(f32(f32(internode_length_max_new) / phytomer.internode_length_max))


@named_hook("MaizePhytomerCreationFunction").implement
@ports("MaizePhytomerCreationFunction")
def MaizePhytomerCreationFunction(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    shoot_node_index, shoot_max_nodes = int(shoot_node_index), int(shoot_max_nodes)
    scale = f32(1)
    if shoot_node_index <= 5:
        # fmin(1.f, 0.7 + 0.3 * float(i) / 5.f): double arithmetic, double fmin, narrowed
        scale = f32(min(1.0, 0.7 + 0.3 * float(f32(shoot_node_index)) / 5.0))
        phytomer.scaleInternodeMaxLength(scale)

    ranks_from_top = f32(int(phytomer.shoot_index[2]) - 1 - shoot_node_index)
    inner = f32(f32(0.475) + f32(f32(0.092) * ranks_from_top))
    inner = f32(min(1.0, float(inner)))                                              # fmin(1.f, float): exact
    scale = f32(min(float(scale), float(inner)))
    phytomer.scaleLeafPrototypeScale(scale)

    if phytomer.shoot_index[2] > 1:
        rank_fraction = f32(f32(shoot_node_index) / f32(int(phytomer.shoot_index[2]) - 1))
        # leaf_flexibility *= 1.f - 0.75f * fmax(0.f, rank_fraction - 0.45f) / 0.55f: C fmax -> double chain
        fm = max(0.0, float(f32(rank_fraction - f32(0.45))))
        factor = 1.0 - float(f32(0.75)) * fm / float(f32(0.55))
        phytomer.leaf_flexibility = f32(float(phytomer.leaf_flexibility) * factor)

    nodes_below_tassel = 6
    apical_ear_node = shoot_max_nodes - nodes_below_tassel
    ears_per_plant = 1
    is_ear = shoot_max_nodes > 0 and (shoot_node_index == apical_ear_node
                                      or (ears_per_plant >= 2 and shoot_node_index == apical_ear_node - 1))

    pp = phytomer.phytomer_parameters
    if is_ear:
        pp.inflorescence.flowers_per_peduncle = 1
        pp.inflorescence.fruit_prototype_function = named_hook("MaizeEarPrototype")
        pp.inflorescence.fruit_prototype_scale = 0.2
        pp.inflorescence.pitch.uniformDistribution(0, 30)
        pp.inflorescence.unique_prototypes = 1
        pp.peduncle.color = pp.internode.color
        pp.peduncle.length = f32(0.05)
        pp.peduncle.radius = 0.01
        pp.peduncle.pitch = 5
        phytomer.setFloralBudState(BUD_ACTIVE)
    else:
        pp.inflorescence.fruit_prototype_function = named_hook("MaizeTasselPrototype")
        phytomer.setFloralBudState(BUD_DEAD)

    if shoot_max_nodes > 0 and shoot_node_index + 1 == shoot_max_nodes:
        setGrassFlagLeafAngle(phytomer, 23.0)


# ---- organ prototypes ------------------------------------------------------------------------------

@named_hook("MaizeEarPrototype").implement
@ports("MaizeEarPrototype")
def MaizeEarPrototype(scene, subdivisions, build_parameters=None):
    """AS:2007-2011."""
    return obj_prototype(scene, "MaizeEar.obj")





# ---- procedural tassel (AS:3086-3553) --------------------------------------------------------------

class MaizeTasselParameters:
    """``MaizeTasselParameters`` (PlantArchitectureInternal.h:373-440): in-class defaults."""

    def __init__(self):
        self.length = f32(0.387)
        self.length_variation = f32(0.06)
        self.central_spike_fraction = f32(0.69)
        self.axis_curvature = f32(8)
        self.axis_radius = f32(0.00675)
        self.axis_tip_radius = f32(0.0008)
        self.primary_branch_count = 10
        self.primary_branch_count_variation = f32(0.15)
        self.secondary_branch_count = 2
        self.branch_length = f32(0.20)
        self.branch_length_tip_fraction = f32(0.5)
        self.branch_length_variation = f32(0.1)
        self.branch_angle_base = f32(38)
        self.branch_angle_tip = f32(20)
        self.branch_droop_base = f32(30)
        self.branch_droop_tip = f32(5)
        self.branch_bare_fraction = f32(0.12)
        self.branch_radius = f32(0.00095)
        self.spike_spikelet_density = f32(14.1)
        self.branch_spikelet_density = f32(5.2)
        self.spikelet_length = f32(0.0102)
        self.spikelet_width = f32(0.0032)
        self.spikelet_thickness = f32(0.0026)
        self.spikelet_size_variation = f32(0.08)
        self.spikelet_angle = f32(16)
        self.spikelet_subdivisions = 4
        self.glume_color = GI.rgb(0.40, 0.46, 0.24)
        self.glume_tip_color = GI.rgb(0.50, 0.46, 0.27)
        self.glume_color_variation = f32(0.1)
        self.anther_fraction = f32(0.5)
        self.anther_length = f32(0.0055)
        self.anther_width = f32(0.0013)
        self.filament_length = f32(0.004)
        self.anther_color = GI.rgb(0.76, 0.60, 0.30)
        self.anther_color_variation = f32(0.1)
        self.branch_color = GI.rgb(*MAIZE_TASSEL_BRANCH_COLOR)
        self.branch_radial_subdivisions = 3
        self.axis_radial_subdivisions = 10

    @staticmethod
    @ports("MaizeTasselParameters::fewBranched")
    def fewBranched():
        p = MaizeTasselParameters()
        p.length = f32(0.36)
        p.central_spike_fraction = f32(0.8)
        p.primary_branch_count = 6
        p.secondary_branch_count = 0
        p.branch_length = f32(0.17)
        p.branch_angle_base = f32(24)
        p.branch_angle_tip = f32(14)
        p.branch_droop_base = f32(12)
        p.branch_droop_tip = f32(3)
        return p

    @staticmethod
    @ports("MaizeTasselParameters::manyBranched")
    def manyBranched():
        p = MaizeTasselParameters()
        p.length = f32(0.40)
        p.central_spike_fraction = f32(0.55)
        p.primary_branch_count = 22
        p.secondary_branch_count = 8
        p.branch_length = f32(0.23)
        p.branch_length_tip_fraction = f32(0.45)
        p.branch_angle_base = f32(54)
        p.branch_angle_tip = f32(28)
        p.branch_droop_base = f32(45)
        p.branch_droop_tip = f32(10)
        return p

    @staticmethod
    @ports("MaizeTasselParameters::fromBuildParameters")
    def fromBuildParameters(build_parameters):
        return GI.fromBuildParameters(MaizeTasselParameters, MAIZE_TASSEL_KEY_TABLES, maize_tassel_type,
                                      maize_spikelet_subdivisions, "spikelet_subdivisions",
                                      MaizeTasselParameters.fewBranched, MaizeTasselParameters.manyBranched,
                                      build_parameters)

    @staticmethod
    @ports("MaizeTasselParameters::buildParameterDeclarations")
    def buildParameterDeclarations():
        declarations = [maize_tassel_type, maize_spikelet_subdivisions]
        GI.appendOrganBuildParameterDeclarations(MAIZE_TASSEL_KEY_TABLES, declarations)
        return declarations


MAIZE_TASSEL_KEY_TABLES = GI.OrganKeyTables(
    [("tassel_length", "length", 0.05, 0.8, "maize tassel length in meters, from its lowest branch to its tip"),
     ("tassel_length_variation", "length_variation", 0.0, 0.3, "coefficient of variation of maize tassel length"),
     ("tassel_central_spike_fraction", "central_spike_fraction", 0.1, 1.0, "fraction of the maize tassel length spanned by the central spike"),
     ("tassel_axis_curvature", "axis_curvature", 0.0, 90.0, "total bend of the maize tassel axis in degrees"),
     ("tassel_branch_count_variation", "primary_branch_count_variation", 0.0, 0.5, "coefficient of variation of the number of maize tassel primary branches"),
     ("tassel_branch_length", "branch_length", 0.02, 0.5, "length of the longest maize tassel primary branches in meters"),
     ("tassel_branch_length_tip_fraction", "branch_length_tip_fraction", 0.1, 1.0, "length of the uppermost maize tassel primary branch as a fraction of the longest"),
     ("tassel_branch_length_variation", "branch_length_variation", 0.0, 0.5, "coefficient of variation of maize tassel primary branch length"),
     ("tassel_branch_angle_base", "branch_angle_base", 0.0, 100.0, "insertion angle of the lowest maize tassel primary branch in degrees from the axis"),
     ("tassel_branch_angle_tip", "branch_angle_tip", 0.0, 100.0, "insertion angle of the uppermost maize tassel primary branch in degrees from the axis"),
     ("tassel_branch_droop_base", "branch_droop_base", 0.0, 120.0, "downward arch of the lowest maize tassel primary branch in degrees"),
     ("tassel_branch_droop_tip", "branch_droop_tip", 0.0, 120.0, "downward arch of the uppermost maize tassel primary branch in degrees"),
     ("tassel_branch_bare_fraction", "branch_bare_fraction", 0.0, 0.9, "fraction of each maize tassel branch that bears no spikelets"),
     ("tassel_branch_radius", "branch_radius", 0.0002, 0.003, "radius of a maize tassel primary branch in meters"),
     ("tassel_spike_spikelet_density", "spike_spikelet_density", 1.0, 40.0, "spikelets per centimeter of maize central spike"),
     ("tassel_branch_spikelet_density", "branch_spikelet_density", 1.0, 20.0, "spikelets per centimeter of maize tassel branch"),
     ("spikelet_length", "spikelet_length", 0.003, 0.02, "maize spikelet length in meters"),
     ("spikelet_width", "spikelet_width", 0.001, 0.008, "maize spikelet width in meters"),
     ("spikelet_thickness", "spikelet_thickness", 0.001, 0.008, "maize spikelet thickness in meters"),
     ("spikelet_size_variation", "spikelet_size_variation", 0.0, 0.3, "coefficient of variation of maize spikelet size"),
     ("spikelet_angle", "spikelet_angle", 0.0, 60.0, "angle of a maize spikelet from the axis or branch bearing it in degrees"),
     ("glume_color_variation", "glume_color_variation", 0.0, 0.5, "coefficient of variation of maize glume brightness"),
     ("anther_fraction", "anther_fraction", 0.0, 1.0, "fraction of maize spikelets with a floret at anthesis"),
     ("anther_length", "anther_length", 0.001, 0.012, "maize anther length in meters"),
     ("anther_width", "anther_width", 0.0003, 0.004, "maize anther width in meters"),
     ("filament_length", "filament_length", 0.0, 0.02, "maize filament length in meters"),
     ("anther_color_variation", "anther_color_variation", 0.0, 0.5, "coefficient of variation of maize anther brightness")],
    [("tassel_branch_count", "primary_branch_count", 0, 60, "mean number of maize tassel primary branches"),
     ("tassel_secondary_branch_count", "secondary_branch_count", 0, 40, "number of maize tassel secondary branches")],
    [("glume_color_r", "glume_color", 0, "red component of the maize glume color"),
     ("glume_color_g", "glume_color", 1, "green component of the maize glume color"),
     ("glume_color_b", "glume_color", 2, "blue component of the maize glume color"),
     ("glume_tip_color_r", "glume_tip_color", 0, "red component of the maize glume tip color"),
     ("glume_tip_color_g", "glume_tip_color", 1, "green component of the maize glume tip color"),
     ("glume_tip_color_b", "glume_tip_color", 2, "blue component of the maize glume tip color"),
     ("anther_color_r", "anther_color", 0, "red component of the maize anther color"),
     ("anther_color_g", "anther_color", 1, "green component of the maize anther color"),
     ("anther_color_b", "anther_color", 2, "blue component of the maize anther color")])

maize_tassel_type = BuildParameterDeclaration(
    "tassel_type", 0.0, 1.0, False, "maize tassel type from 0 (few-branched) through 0.5 (the default) to 1 (many-branched)")
maize_spikelet_subdivisions = BuildParameterDeclaration("spikelet_subdivisions", 3.0, 12.0, True,
                                                        "number of facets around a maize spikelet")


def _validate_maize_tassel_parameters(parameters):
    MaizeTasselParameters.fromBuildParameters(parameters)


def register_build_parameters(pa):
    """PL:528-529: no structure parameters; the tassel keys as organ parameters."""
    registerPlantModelBuildParameters(pa, "maize", [], MaizeTasselParameters.buildParameterDeclarations(),
                                      _validate_maize_tassel_parameters)


@ports("addMaizeSpikelet")
def addMaizeSpikelet(ctx, center, spikelet_axis, width_axis, length, width, thickness, facets, color, tip_color, mesh):
    """AS:3243-3268."""
    thickness_axis = cross(spikelet_axis, width_axis)
    half_length = f32(f32(0.5) * length)
    ring_height = (f32(f32(-0.45) * half_length), f32(f32(0.35) * half_length))
    ring_profile = (f32(1), f32(0.72))
    half_w, half_t = f32(f32(0.5) * width), f32(f32(0.5) * thickness)
    base_pole = mesh.addVertex(center - spikelet_axis * half_length)
    apex_pole = mesh.addVertex(center + spikelet_axis * half_length)
    rings = ([], [])
    for ring in range(2):
        along = spikelet_axis * ring_height[ring]
        wr, tr = f32(half_w * ring_profile[ring]), f32(half_t * ring_profile[ring])
        for facet in range(facets):
            angle = f32(f32(GI.TWO_PI_F * f32(f32(facet) + f32(f32(0.5) * f32(ring)))) / f32(facets))
            rings[ring].append(mesh.addVertex(((center + along) + width_axis * f32(wr * cosf(angle)))
                                              + thickness_axis * f32(tr * sinf(angle))))
    light, dark = GI.rgb_scale(color, 1.08), GI.rgb_scale(color, 0.92)
    for facet in range(facets):
        nxt = (facet + 1) % facets
        body = light if facet % 2 == 0 else dark
        mesh.addTriangle(ctx, base_pole, rings[0][nxt], rings[0][facet], body)
        mesh.addTriangle(ctx, rings[0][facet], rings[0][nxt], rings[1][facet], body)
        mesh.addTriangle(ctx, rings[1][facet], rings[0][nxt], rings[1][nxt], body)
        mesh.addTriangle(ctx, apex_pole, rings[1][facet], rings[1][nxt], tip_color)


@ports("buildMaizeTassel")
def buildMaizeTassel(ctx, parameters):
    """AS:3302-3547. Every draw is the Context's, in the C++ statement order; ``make_vec3(0, randu(), randu())`` and
    function arguments are evaluated right to left (GCC)."""
    p = parameters
    if (p.length <= 0 or p.spikelet_length <= 0 or p.spikelet_width <= 0 or p.spikelet_thickness <= 0
            or p.axis_radius <= 0 or p.axis_tip_radius <= 0 or p.branch_radius <= 0):
        raise PlantArchitectureError("ERROR (buildMaizeTassel): Tassel length, axis and branch radii, and spikelet "
                                     "dimensions must all be positive.")
    elif p.central_spike_fraction <= 0 or p.central_spike_fraction > 1:
        raise PlantArchitectureError("ERROR (buildMaizeTassel): central_spike_fraction must be greater than 0 and at most "
                                     f"1, but {_std_to_string(p.central_spike_fraction)} was given.")
    elif p.primary_branch_count < 0 or p.secondary_branch_count < 0:
        raise PlantArchitectureError("ERROR (buildMaizeTassel): primary_branch_count and secondary_branch_count cannot be "
                                     "negative.")
    elif p.spike_spikelet_density <= 0 or p.branch_spikelet_density <= 0:
        raise PlantArchitectureError("ERROR (buildMaizeTassel): spike_spikelet_density and branch_spikelet_density must "
                                     "be positive.")
    elif p.branch_bare_fraction < 0 or p.branch_bare_fraction >= 1:
        raise PlantArchitectureError("ERROR (buildMaizeTassel): branch_bare_fraction must be in the range [0,1), but "
                                     f"{_std_to_string(p.branch_bare_fraction)} was given.")
    elif p.anther_fraction < 0 or p.anther_fraction > 1:
        raise PlantArchitectureError("ERROR (buildMaizeTassel): anther_fraction must be in the range [0,1], but "
                                     f"{_std_to_string(p.anther_fraction)} was given.")
    elif p.anther_fraction > 0 and (p.anther_length <= 0 or p.anther_width <= 0):
        raise PlantArchitectureError("ERROR (buildMaizeTassel): anther_length and anther_width must be positive when "
                                     "there are anthers.")
    elif p.filament_length < 0:
        raise PlantArchitectureError("ERROR (buildMaizeTassel): filament_length cannot be negative.")
    elif p.spikelet_subdivisions < 3 or p.branch_radial_subdivisions < 3 or p.axis_radial_subdivisions < 3:
        raise PlantArchitectureError("ERROR (buildMaizeTassel): spikelet_subdivisions, branch_radial_subdivisions and "
                                     "axis_radial_subdivisions must each be at least 3.")

    one = f32(1)
    size_factor = clamp(f32(one + f32(p.length_variation * ctx.randn())), f32(0.7), f32(1.3))
    tassel_length = f32(p.length * size_factor)
    if p.primary_branch_count == 0:
        primary_branch_count = 0
    else:
        bc = f32(f32(p.primary_branch_count) * f32(one + f32(p.primary_branch_count_variation * ctx.randn())))
        primary_branch_count = max(1, int(GI.roundf(bc)))
    branched_length = f32(0) if primary_branch_count == 0 else f32(f32(one - p.central_spike_fraction) * tassel_length)

    gravity = GI.vec(-1, 0, 0)

    # ---- axis ----
    total_bend = deg2rad(p.axis_curvature)
    bend_azimuth = ctx.randu(f32(0), GI.TWO_PI_F)
    axis_base_direction = GI.vec(1, 0, 0)
    bend_direction = GI.vec(0, cosf(bend_azimuth), sinf(bend_azimuth))
    binormal = cross(axis_base_direction, bend_direction)
    straight = abs(total_bend) < f32(1e-4)
    bend_radius = None if straight else f32(tassel_length / total_bend)

    def axis_position(s):
        if straight:
            return axis_base_direction * s
        angle = f32(s / bend_radius)
        return axis_base_direction * f32(bend_radius * sinf(angle)) + bend_direction * f32(bend_radius * f32(one - cosf(angle)))

    def axis_tangent(s):
        angle = f32(f32(total_bend * s) / tassel_length)
        return axis_base_direction * cosf(angle) + bend_direction * sinf(angle)

    def axis_normal(s):
        angle = f32(f32(total_bend * s) / tassel_length)
        return bend_direction * cosf(angle) - axis_base_direction * sinf(angle)

    radius_span = f32(p.axis_radius - p.axis_tip_radius)

    def axis_radius_at(s):
        return f32(p.axis_tip_radius + f32(radius_span * GI.powf(f32(one - f32(s / tassel_length)), f32(1.5))))

    mesh = GI.GrassInflorescenceMesh()
    axis_segments = 16
    axis_nodes, axis_radii = [], []
    for node in range(axis_segments + 1):
        s = f32(f32(tassel_length * f32(node)) / f32(axis_segments))
        axis_nodes.append(axis_position(s))
        axis_radii.append(axis_radius_at(s))
    mesh.addTube(ctx, GI.context_add_tube(ctx, p.axis_radial_subdivisions, axis_nodes, axis_radii,
                                          [p.branch_color] * len(axis_nodes)))

    # ---- spikelets ----
    def add_spikelet(surface_point, spikelet_axis, outward, taper, hang_bias):
        scale = f32(taper * clamp(f32(one + f32(p.spikelet_size_variation * ctx.randn())), f32(0.7), f32(1.3)))
        length = f32(p.spikelet_length * scale)
        width = f32(p.spikelet_width * scale)
        thickness = f32(p.spikelet_thickness * scale)
        width_axis = GI.normalize(cross(spikelet_axis, outward))
        spikelet_center = (surface_point + outward * f32(f32(0.45) * thickness)) + spikelet_axis * f32(f32(0.5) * length)
        brightness = GI.quantizedBrightness(ctx, p.glume_color_variation)
        body_color = GI.rgb_scale(p.glume_color, brightness)
        tip_color = GI.rgb_scale(p.glume_tip_color, brightness)
        addMaizeSpikelet(ctx, spikelet_center, spikelet_axis, width_axis, length, width, thickness,
                         int(p.spikelet_subdivisions), body_color, tip_color, mesh)
        if ctx.randu() < p.anther_fraction:
            floret_point = (spikelet_center + spikelet_axis * f32(f32(0.3) * length)) + outward * f32(f32(0.5) * thickness)
            for _anther in range(3):
                sz = ctx.randu(f32(-0.25), f32(0.25))
                sy = ctx.randu(f32(-0.25), f32(0.25))
                hang_direction = GI.normalize((gravity + hang_bias) + GI.vec(0, sy, sz))
                tz = ctx.randu(f32(-0.2), f32(0.2))
                ty = ctx.randu(f32(-0.2), f32(0.2))
                anther_axis = GI.normalize(hang_direction + GI.vec(0, ty, tz))
                anther_color = GI.rgb_scale(p.anther_color, GI.quantizedBrightness(ctx, p.anther_color_variation))
                anther_length = f32(p.anther_length * ctx.randu(f32(0.85), f32(1.15)))
                filament_length = f32(p.filament_length * ctx.randu(f32(0.7), f32(1.3)))
                GI.addGrassAnther(ctx, floret_point, hang_direction, anther_axis, filament_length, anther_length,
                                  p.anther_width, anther_color, mesh)

    # ---- central spike ----
    pairs_per_whorl = 2
    spike_pair_step = f32(f32(0.5) * GI.PI_F)
    spike_length = f32(tassel_length - branched_length)
    whorl_spacing = f32(f32(f32(0.01) * f32(2 * pairs_per_whorl)) / p.spike_spikelet_density)
    whorl_count = max(1, int(GI.roundf(f32(spike_length / whorl_spacing))))
    spike_azimuth_offset = ctx.randu(f32(0), GI.TWO_PI_F)
    spikelet_angle = deg2rad(p.spikelet_angle)
    cos_sa, sin_sa = cosf(spikelet_angle), sinf(spikelet_angle)
    arc_cap = f32(tassel_length - f32(f32(0.5) * p.spikelet_length))
    pedicel_lift_up = f32(f32(0.4) * whorl_spacing)
    for whorl in range(whorl_count):
        fraction = f32(f32(f32(whorl) + f32(0.5)) / f32(whorl_count))
        arc_length = min(f32(branched_length + f32(fraction * spike_length)), arc_cap)
        taper = one if fraction < f32(0.8) else f32(one - f32(f32(f32(0.45) * f32(fraction - f32(0.8))) / f32(0.2)))
        center = axis_position(arc_length)
        tangent = axis_tangent(arc_length)
        normal = axis_normal(arc_length)
        radius = axis_radius_at(arc_length)
        pair_half_angle = f32(f32(f32(0.5) * p.spikelet_width) / f32(radius + f32(f32(0.5) * p.spikelet_thickness)))
        for pair in range(pairs_per_whorl):
            pair_azimuth = f32(f32(f32(spike_azimuth_offset + f32(spike_pair_step * f32(whorl))) + f32(GI.PI_F * f32(pair)))
                               + deg2rad(ctx.randu(f32(-12), f32(12))))
            for side in (f32(-1), f32(1)):
                azimuth = f32(pair_azimuth + f32(side * pair_half_angle))
                outward = normal * cosf(azimuth) + binormal * sinf(azimuth)
                spikelet_axis = GI.normalize(tangent * cos_sa + outward * sin_sa)
                pedicel_lift = pedicel_lift_up if side > 0 else f32(0)
                add_spikelet((center + tangent * pedicel_lift) + outward * radius, spikelet_axis, outward, taper,
                             outward * f32(0.3))

    # ---- branches ----
    branch_segments = 8
    nseg = f32(branch_segments)

    def add_branch(base, parent_tangent, outward, length, insertion_angle, droop, base_radius, bare_fraction):
        nodes = [base]
        tangents = []
        seg_len = f32(length / nseg)
        for segment in range(branch_segments):
            position = f32(f32(f32(segment) + f32(0.5)) / nseg)
            angle = f32(insertion_angle + f32(droop * GI.powf(position, f32(1.5))))
            tangents.append(GI.normalize(parent_tangent * cosf(angle) + outward * sinf(angle)))
            nodes.append(nodes[segment] + tangents[segment] * seg_len)
        radii = [f32(base_radius * f32(one - f32(f32(f32(0.6) * f32(node)) / nseg))) for node in range(branch_segments + 1)]
        mesh.addTube(ctx, GI.context_add_tube(ctx, p.branch_radial_subdivisions, nodes, radii,
                                              [p.branch_color] * len(nodes)))
        pair_spacing = f32(f32(f32(0.01) * f32(2)) / p.branch_spikelet_density)
        fertile_length = f32(f32(one - bare_fraction) * length)
        pair_count = max(1, int(GI.roundf(f32(fertile_length / pair_spacing))))
        plane_normal = GI.normalize(cross(parent_tangent, outward))
        bare_length = f32(bare_fraction * length)
        lift = f32(f32(0.25) * pair_spacing)
        for pair in range(pair_count):
            fraction = f32(f32(f32(pair) + f32(0.5)) / f32(pair_count))
            arc_length = f32(bare_length + f32(fraction * fertile_length))
            seg_f = f32(f32(nseg * arc_length) / length)
            segment = min(int(seg_f), branch_segments - 1)
            local = f32(seg_f - f32(segment))
            point = nodes[segment] + (nodes[segment + 1] - nodes[segment]) * local
            tangent = tangents[segment]
            radius = f32(radii[segment] + f32(f32(radii[segment + 1] - radii[segment]) * local))
            cpt = cross(plane_normal, tangent)
            upper = GI.normalize(cpt * (one if dot(cpt, parent_tangent) >= 0 else f32(-1)))
            row_side = one if pair % 2 == 0 else f32(-1)
            taper = one if fraction < f32(0.85) else f32(one - f32(f32(f32(0.4) * f32(fraction - f32(0.85))) / f32(0.15)))
            sessile_elevation = deg2rad(ctx.randu(f32(-30), f32(0)))
            pedicellate_elevation = deg2rad(ctx.randu(f32(25), f32(55)))
            for member in range(2):
                elevation = sessile_elevation if member == 0 else pedicellate_elevation
                flank = GI.normalize(plane_normal * f32(row_side * cosf(elevation)) + upper * sinf(elevation))
                spikelet_axis = GI.normalize(tangent * cos_sa + flank * sin_sa)
                surface_point = (point + flank * radius) + tangent * (lift if member == 1 else f32(0))
                if member == 1:
                    pedicel_side = spikelet_axis * f32(0.0004)
                    mesh.addIsolatedTriangle(ctx, point - pedicel_side, point + pedicel_side, surface_point, p.branch_color)
                add_spikelet(surface_point, spikelet_axis, flank, taper, flank * f32(0.25))
        return nodes

    primary = []          # (nodes, parent_tangent, outward, length)
    branch_azimuth_offset = ctx.randu(f32(0), GI.TWO_PI_F)
    golden = deg2rad(f32(137.5))
    for b in range(primary_branch_count):
        position = f32(f32(b) / f32(primary_branch_count - 1)) if primary_branch_count > 1 else f32(0)
        jittered = clamp(f32(position + f32(ctx.randu(f32(-0.25), f32(0.25)) / f32(primary_branch_count))), f32(0), one)
        arc_length = f32(jittered * branched_length)
        if position < f32(0.3):
            length_profile = one
        else:
            length_profile = f32(one + f32(f32(f32(p.branch_length_tip_fraction - one) * f32(position - f32(0.3))) / f32(0.7)))
        length = f32(f32(f32(p.branch_length * size_factor) * length_profile)
                     * clamp(f32(one + f32(p.branch_length_variation * ctx.randn())), f32(0.5), f32(1.5)))
        insertion_angle = deg2rad(clamp(f32(f32(p.branch_angle_base + f32(f32(p.branch_angle_tip - p.branch_angle_base) * position))
                                            + ctx.randu(f32(-5), f32(5))), f32(0), f32(120)))
        droop = deg2rad(f32(f32(p.branch_droop_base + f32(f32(p.branch_droop_tip - p.branch_droop_base) * position))
                            * ctx.randu(f32(0.7), f32(1.3))))
        azimuth = f32(f32(branch_azimuth_offset + f32(golden * f32(b))) + deg2rad(ctx.randu(f32(-20), f32(20))))
        parent_tangent = axis_tangent(arc_length)
        outward = axis_normal(arc_length) * cosf(azimuth) + binormal * sinf(azimuth)
        base = axis_position(arc_length) + outward * f32(f32(0.8) * axis_radius_at(arc_length))
        nodes = add_branch(base, parent_tangent, outward, length, insertion_angle, droop, p.branch_radius,
                           p.branch_bare_fraction)
        primary.append((nodes, parent_tangent, outward, length))

    # ---- secondary branches ----
    s = 0
    while s < p.secondary_branch_count and primary_branch_count > 0:
        nodes, parent_tangent, parent_outward, parent_length = primary[s % max(1, primary_branch_count // 2)]
        side = one if s % 2 == 0 else f32(-1)
        parent_direction = GI.normalize(nodes[1] - nodes[0])
        plane_normal = GI.normalize(cross(parent_tangent, parent_outward))
        base = nodes[0] + (nodes[1] - nodes[0]) * f32(0.6)
        length = f32(f32(f32(0.45) * parent_length) * ctx.randu(f32(0.8), f32(1.2)))
        droop = deg2rad(ctx.randu(f32(0), f32(15)))                     # arguments right to left
        insertion_angle = deg2rad(ctx.randu(f32(20), f32(35)))
        add_branch(base, parent_direction, plane_normal * side, length, insertion_angle, droop,
                   f32(f32(0.65) * p.branch_radius), f32(0.1))
        s += 1

    return mesh.finish(ctx)


@named_hook("MaizeTasselPrototype").implement
@ports("MaizeTasselPrototype")
def MaizeTasselPrototype(scene, subdivisions, build_parameters=None):
    """AS:3549-3551."""
    return buildMaizeTassel(scene, MaizeTasselParameters.fromBuildParameters(build_parameters or {}))
