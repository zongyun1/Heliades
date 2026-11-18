# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:5590-5743 and src/Assets.cpp:2419-3084, 3949-4029
# (Copyright (C) 2016-2026 Brian Bailey).
"""Sorghum shoot parameters, builder, phytomer creation function and procedural panicle (W5-A).

A grass like the maize (``library/maize.py``): 16 main-stem phytomers with sessile flexible blades, a 6-internode
crown, the panicle as the terminal fruit. Tillers (shoot type ``tiller``) grow from the axils of the lowest leaves
when the build parameter ``tiller_count`` (0-6, default 0) asks for them, in the order T3, T4, T2, T1, T5, T6.

The panicle (``buildSorghumPanicle``) is a rachis with whorls of primary branches bearing racemes of grains (each an
obovoid of ``4 * grain_subdivisions`` triangles with two glumes), packed by an overlap-resolution pass. Its shape comes
from ``SorghumPanicleParameters``, set by the plant's organ build parameters (``panicle_*``, ``grain_*``, ``glume_*``).
The packing pass is vectorised over grain pairs, but every float sum is accumulated in the C++ loop order (see
``_resolve_overlaps``).

RNG on build: ``randu(1, 6)`` (lean), ``randu(0, 2 pi)`` (lean azimuth), ``randu(0, 2 pi)`` (phyllotaxis azimuth), then
``internode_length_max.val()`` (no draw) and ``internode_length_max.resample()`` (one draw for the next plant).
"""
from __future__ import annotations

import math

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
sinf, cosf, clamp = V.sinf, V.cosf, V.clamp
cross, dot, normalize = GI.cross, GI.dot, GI.normalize

#: PlantArchitectureInternal.h: ``sorghum_tiller_count``
sorghum_tiller_count = BuildParameterDeclaration("tiller_count", 0.0, 6.0, True, "number of tillers of a sorghum plant")


# ---- panicle parameters (PlantArchitectureInternal.h:234-305; AS:2419-2614) ------------------------------------------

class SorghumPanicleParameters:
    def __init__(self):
        self.length = f32(0.25)
        self.length_variation = f32(0.08)
        self.rachis_length_fraction = f32(0.85)
        self.rachis_curvature = f32(0)
        self.compactness = f32(0.8)
        self.rachis_radius = f32(0.0046)
        self.primary_branch_count = 70
        self.branch_length_base = f32(0.2)
        self.branch_length_middle = f32(0.21)
        self.branch_length_tip = f32(0.12)
        self.branch_length_variation = f32(0.2)
        self.branch_angle_base = f32(45)
        self.branch_angle_tip = f32(25)
        self.branch_upcurve = f32(15)
        self.branch_bare_fraction = f32(0.15)
        self.splayed_branch_fraction = f32(0.04)
        self.splayed_branch_angle = f32(65)
        self.branch_radius = f32(0.0007)
        self.grains_per_raceme = 5
        self.raceme_length_fraction = f32(0.2)
        self.raceme_angle = f32(35)
        self.grain_count = 2000
        self.grain_length = f32(0.0045)
        self.grain_width = f32(0.0040)
        self.grain_thickness = f32(0.0034)
        self.grain_size_variation = f32(0.08)
        self.grain_color = GI.rgb(0.78, 0.62, 0.42)
        self.grain_color_variation = f32(0.08)
        self.grain_subdivisions = 4
        self.glume_length_fraction = f32(0.55)
        self.glume_spread = f32(5)
        self.glume_color = GI.rgb(0.45, 0.30, 0.18)
        self.branch_color = GI.rgb(0.42, 0.45, 0.28)
        self.branch_radial_subdivisions = 3
        self.rachis_radial_subdivisions = 10

    @staticmethod
    @ports("SorghumPanicleParameters::compact")
    def compact():
        p = SorghumPanicleParameters()
        p.length = f32(0.22)
        p.compactness = f32(0.9)
        p.primary_branch_count = 80
        p.branch_length_base = f32(0.15)
        p.branch_length_middle = f32(0.16)
        p.branch_length_tip = f32(0.1)
        p.branch_angle_base = f32(40)
        p.branch_angle_tip = f32(20)
        p.branch_upcurve = f32(10)
        p.branch_bare_fraction = f32(0.05)
        p.splayed_branch_fraction = f32(0.02)
        p.raceme_length_fraction = f32(0.25)
        p.grain_count = 2000
        return p

    @staticmethod
    @ports("SorghumPanicleParameters::loose")
    def loose():
        p = SorghumPanicleParameters()
        p.length = f32(0.30)
        p.compactness = f32(0)
        p.rachis_length_fraction = f32(0.8)
        p.primary_branch_count = 50
        p.branch_length_base = f32(0.35)
        p.branch_length_middle = f32(0.25)
        p.branch_length_tip = f32(0.12)
        p.branch_length_variation = f32(0.25)
        p.branch_angle_base = f32(50)
        p.branch_angle_tip = f32(25)
        p.branch_upcurve = f32(30)
        p.branch_bare_fraction = f32(0.35)
        p.splayed_branch_fraction = f32(0)
        p.raceme_length_fraction = f32(0.15)
        p.grains_per_raceme = 4
        p.grain_count = 1100
        return p

    @staticmethod
    @ports("SorghumPanicleParameters::fromBuildParameters")
    def fromBuildParameters(build_parameters):
        return GI.fromBuildParameters(SorghumPanicleParameters, SORGHUM_PANICLE_KEY_TABLES, sorghum_panicle_type,
                                      sorghum_grain_subdivisions, "grain_subdivisions", SorghumPanicleParameters.compact,
                                      SorghumPanicleParameters.loose, build_parameters)

    @staticmethod
    @ports("SorghumPanicleParameters::buildParameterDeclarations")
    def buildParameterDeclarations():
        declarations = [sorghum_panicle_type, sorghum_grain_subdivisions]
        GI.appendOrganBuildParameterDeclarations(SORGHUM_PANICLE_KEY_TABLES, declarations)
        return declarations


SORGHUM_PANICLE_KEY_TABLES = GI.OrganKeyTables(
    [("panicle_length", "length", 0.05, 0.6, "sorghum panicle length in meters"),
     ("panicle_length_variation", "length_variation", 0.0, 0.3, "coefficient of variation of sorghum panicle length"),
     ("panicle_rachis_length_fraction", "rachis_length_fraction", 0.3, 1.0, "fraction of the sorghum panicle length spanned by the rachis"),
     ("panicle_rachis_curvature", "rachis_curvature", 0.0, 180.0, "total bend of the sorghum rachis in degrees"),
     ("panicle_compactness", "compactness", 0.0, 1.0, "compactness of the sorghum panicle"),
     ("panicle_branch_length_base", "branch_length_base", 0.02, 0.6, "length of the basal sorghum primary branches as a fraction of panicle length"),
     ("panicle_branch_length_middle", "branch_length_middle", 0.02, 0.6, "length of the middle sorghum primary branches as a fraction of panicle length"),
     ("panicle_branch_length_tip", "branch_length_tip", 0.02, 0.6, "length of the apical sorghum primary branches as a fraction of panicle length"),
     ("panicle_branch_length_variation", "branch_length_variation", 0.0, 0.5, "coefficient of variation of sorghum primary branch length"),
     ("panicle_branch_angle_base", "branch_angle_base", 0.0, 90.0, "insertion angle of the basal sorghum primary branches in degrees"),
     ("panicle_branch_angle_tip", "branch_angle_tip", 0.0, 90.0, "insertion angle of the apical sorghum primary branches in degrees"),
     ("panicle_branch_upcurve", "branch_upcurve", 0.0, 90.0, "upward turn of the sorghum primary branches in degrees"),
     ("panicle_branch_bare_fraction", "branch_bare_fraction", 0.0, 0.9, "fraction of each sorghum primary branch that bears no racemes"),
     ("panicle_splayed_branch_fraction", "splayed_branch_fraction", 0.0, 1.0, "fraction of sorghum primary branches that splay out from the head"),
     ("panicle_splayed_branch_angle", "splayed_branch_angle", 0.0, 120.0, "insertion angle of a splayed sorghum primary branch in degrees"),
     ("panicle_branch_radius", "branch_radius", 0.0002, 0.003, "radius of a sorghum primary branch in meters"),
     ("panicle_raceme_length_fraction", "raceme_length_fraction", 0.02, 1.0, "sorghum raceme length as a fraction of its primary branch"),
     ("panicle_raceme_angle", "raceme_angle", 0.0, 90.0, "angle of a sorghum raceme from its primary branch in degrees"),
     ("grain_length", "grain_length", 0.001, 0.01, "sorghum grain length in meters"),
     ("grain_width", "grain_width", 0.001, 0.01, "sorghum grain width in meters"),
     ("grain_thickness", "grain_thickness", 0.001, 0.01, "sorghum grain thickness in meters"),
     ("grain_size_variation", "grain_size_variation", 0.0, 0.3, "coefficient of variation of sorghum grain size"),
     ("grain_color_variation", "grain_color_variation", 0.0, 0.5, "coefficient of variation of sorghum grain brightness"),
     ("glume_length_fraction", "glume_length_fraction", 0.0, 1.2, "sorghum glume length as a fraction of grain length"),
     ("glume_spread", "glume_spread", 0.0, 90.0, "gape of the sorghum glumes in degrees")],
    [("panicle_branch_count", "primary_branch_count", 5, 200, "number of sorghum primary branches"),
     ("panicle_grains_per_raceme", "grains_per_raceme", 1, 20, "mean number of grains on a sorghum raceme"),
     ("grain_count", "grain_count", 100, 10000, "mean number of grains on a sorghum panicle")],
    [("grain_color_r", "grain_color", 0, "red component of the sorghum grain color"),
     ("grain_color_g", "grain_color", 1, "green component of the sorghum grain color"),
     ("grain_color_b", "grain_color", 2, "blue component of the sorghum grain color"),
     ("glume_color_r", "glume_color", 0, "red component of the sorghum glume color"),
     ("glume_color_g", "glume_color", 1, "green component of the sorghum glume color"),
     ("glume_color_b", "glume_color", 2, "blue component of the sorghum glume color")])

sorghum_panicle_type = BuildParameterDeclaration(
    "panicle_type", 0.0, 1.0, False, "sorghum panicle type from 0 (compact) through 0.5 (semi-compact) to 1 (loose)")
sorghum_grain_subdivisions = BuildParameterDeclaration("grain_subdivisions", 3.0, 12.0, True,
                                                       "number of facets around a sorghum grain")


def _validate_sorghum_panicle_parameters(parameters):
    SorghumPanicleParameters.fromBuildParameters(parameters)


def register_build_parameters(pa):
    """PL:525-526."""
    registerPlantModelBuildParameters(pa, "sorghum", [sorghum_tiller_count],
                                      SorghumPanicleParameters.buildParameterDeclarations(),
                                      _validate_sorghum_panicle_parameters)


# ---- shoots and builder (PL:5590-5743) --------------------------------------------------------------------------------

@ports("PlantArchitecture::initializeSorghumShoots")
def initializeSorghumShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.leaf_texture_file[0] = "SorghumLeaf.png"
    leaf_prototype.leaf_class_map_file[0] = "SorghumLeaf_classes.png"
    leaf_prototype.leaf_aspect_ratio = f32(0.15)
    leaf_prototype.midrib_fold_fraction = f32(0.45)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.07, -0.04)
    leaf_prototype.longitudinal_curvature_exponent = f32(3)
    leaf_prototype.flexibility_taper = f32(150)
    leaf_prototype.flexibility_aging = f32(11)
    leaf_prototype.flexibility_aging_max = f32(6)
    leaf_prototype.lateral_curvature = f32(-0.3)
    leaf_prototype.petiole_roll = f32(0)
    leaf_prototype.wave_period = f32(0.1)
    leaf_prototype.wave_amplitude = f32(0.06)
    leaf_prototype.flexibility.uniformDistribution(1.2, 1.8)
    leaf_prototype.subdivisions = 50
    leaf_prototype.unique_prototypes = 10

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)
    pp.internode.pitch = 0
    pp.internode.phyllotactic_angle.uniformDistribution(155, 205)
    pp.internode.radius_initial = 0.003
    pp.internode.color = make_RGBcolor(0.09, 0.13, 0.06)
    pp.internode.length_segments = 2
    pp.internode.radial_subdivisions = 10
    pp.internode.max_floral_buds_per_petiole = 0
    pp.internode.max_vegetative_buds_per_petiole = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.length = 0

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-25, -12)
    pp.leaf.yaw = 0
    pp.leaf.roll.uniformDistribution(-12, 12)
    pp.leaf.prototype_scale = 0.65
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length = 0.3
    pp.peduncle.radius = 0.008
    pp.peduncle.color = make_RGBcolor(*SorghumPanicleParameters().branch_color)
    pp.peduncle.length_segments = 16
    pp.peduncle.radial_subdivisions = 10
    pp.peduncle.pitch.uniformDistribution(0, 6)
    pp.peduncle.yaw.uniformDistribution(0, 360)
    pp.peduncle.curvature.uniformDistribution(-40, 0)

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.roll = 0
    pp.inflorescence.fruit_prototype_scale = 1
    pp.inflorescence.fruit_prototype_function = named_hook("SorghumPaniclePrototype")
    pp.inflorescence.unique_prototypes = 3

    pp.phytomer_creation_function = named_hook("SorghumPhytomerCreationFunction")

    # ---- Shoot Parameters ---- //
    sp = ShootParameters(gen)
    sp.phytomer_parameters = pp
    sp.vegetative_bud_break_probability_min = 1
    sp.vegetative_bud_break_time = 9
    sp.flower_bud_break_probability = 1
    sp.phyllochron_min = 3.5
    sp.elongation_rate_max = 0.05
    sp.girth_area_factor = f32(5)
    sp.gravitropic_curvature = 0
    sp.internode_length_max.uniformDistribution(0.0675, 0.0825)
    sp.internode_length_decay_rate = 0
    sp.flowers_require_dormancy = False
    sp.growth_requires_dormancy = False
    sp.determinate_shoot_growth = False
    sp.flower_bud_break_probability = 1.0
    sp.fruit_set_probability = 1.0
    sp.defineChildShootTypes(["tiller"], [1.0])
    sp.max_nodes = 16
    sp.max_terminal_floral_buds = 1

    st = sp.copy()                                                               # copy ctor: no draw
    st.phytomer_parameters.internode.max_vegetative_buds_per_petiole = 0
    st.vegetative_bud_break_probability_min = 0
    st.defineChildShootTypes(["tiller"], [1.0])
    st.insertion_angle_tip.uniformDistribution(15, 30)
    st.base_yaw.uniformDistribution(-20, 20)
    st.base_roll.uniformDistribution(-20, 20)
    st.gravitropic_curvature.uniformDistribution(200, 400)

    pa.defineShootType("mainstem", sp)
    pa.defineShootType("tiller", st)


@ports("sorghumTillerCount")
def sorghumTillerCount(build_parameters) -> int:
    return getBuildParameterIntegerValue(build_parameters, sorghum_tiller_count, 0)


@ports("PlantArchitecture::buildSorghumPlant")
def buildSorghumPlant(pa, base_position):
    """PL:5708-5743."""
    if not pa.shoot_types:
        initializeSorghumShoots(pa)
    sorghumTillerCount(pa.current_build_parameters)
    plantID = pa.addPlantInstance(np.asarray(base_position, f32) - np.array([0, 0, 0.01], f32), 0)
    ctx = pa.context_ptr
    lean = deg2rad(ctx.randu(f32(1), f32(6)))
    lean_azimuth = ctx.randu(f32(0), f32(2.0 * np.pi))
    phyllotaxis_azimuth = ctx.randu(f32(0), f32(2.0 * np.pi))
    internode_length_max = pa.shoot_types["mainstem"].internode_length_max.val()
    pa.shoot_types["mainstem"].internode_length_max.resample()
    pa.addBaseStemShoot(plantID, 1, make_AxisRotation(lean, lean_azimuth, phyllotaxis_azimuth), 0.003,
                        internode_length_max, 0.01, 0.01, 0, "mainstem")
    pa.breakPlantDormancy(plantID)
    pa.setPlantPhenologicalThresholds(plantID, 0, -1, -1, 4, 35, 1000)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


# ---- phytomer creation function (AS:3953-4029) --------------------------------------------------------------------

_TILLER_LEAF_NUMBERS = (3, 4, 2, 1, 5, 6)


@named_hook("SorghumPhytomerCreationFunction").implement
@ports("SorghumPhytomerCreationFunction")
def SorghumPhytomerCreationFunction(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    shoot_node_index, shoot_max_nodes = int(shoot_node_index), int(shoot_max_nodes)
    shoot = phytomer.parent_shoot_ptr
    pa = shoot.plantarchitecture_ptr
    if phytomer.rank == 0:
        tiller_count = sorghumTillerCount(pa.getPlantBuildParameters(phytomer.plantID))
        if shoot_node_index + 1 not in _TILLER_LEAF_NUMBERS[:tiller_count]:
            phytomer.setVegetativeBudState(BUD_DEAD)
    elif shoot_node_index == 0:
        main_stem = pa.plant_instances[phytomer.plantID].shoot_tree[shoot.parent_shoot_ID]
        tiller_max_nodes = max(1, int(main_stem.shoot_parameters.max_nodes.val()) - int(parent_shoot_node_index + 1))
        shoot.shoot_parameters.max_nodes = tiller_max_nodes
        phytomer.shoot_index[2] = tiller_max_nodes
        shoot_max_nodes = tiller_max_nodes

    one = f32(1)
    leaf_scale = min(one, f32(f32(0.82) + f32(f32(f32(0.18) * f32(shoot_node_index)) / f32(3))))
    if shoot_max_nodes > 0:
        largest_leaf_rank = f32(f32(4.64) + f32(f32(0.46) * f32(shoot_max_nodes)))
        ranks_above_largest = f32(f32(shoot_node_index + 1) - largest_leaf_rank)
        if ranks_above_largest > 0:
            ranks_to_flag = max(one, f32(f32(shoot_max_nodes) - largest_leaf_rank))
            leaf_scale = f32(leaf_scale * max(f32(0.80), f32(one - f32(f32(f32(0.20) * ranks_above_largest) / ranks_to_flag))))

    if phytomer.leaf_size_max and phytomer.leaf_size_max[0]:
        leaf_size_target = f32(leaf_scale * phytomer.phytomer_parameters.leaf.prototype_scale.val())
        leaf_size_current = f32(phytomer.leaf_size_max[0][0])
        if leaf_size_current > 0 and abs(f32(leaf_size_target - leaf_size_current)) > f32(1e-5):
            phytomer.scaleLeafPrototypeScale(f32(leaf_size_target / leaf_size_current))

    crown_internodes = 6
    if shoot_node_index < crown_internodes:
        setInternodeMaxLength(phytomer, f32(0.008))
    else:
        internode_scale = min(one, f32(f32(0.60) + f32(f32(0.20) * f32(shoot_node_index - crown_internodes))))
        phytomer.scaleInternodeMaxLength(internode_scale)

    if shoot_max_nodes > 0 and shoot_node_index + 1 == shoot_max_nodes:
        setGrassFlagLeafAngle(phytomer, f32(30))
        setInternodeMaxLength(phytomer, phytomer.phytomer_parameters.peduncle.length.val())


# ---- panicle geometry (AS:2616-3084) -----------------------------------------------------------------------------------

def interpolateRachisProfile(base_value, middle_value, tip_value, position) -> np.float32:
    """AS:2616-2618."""
    p = f32(position)
    one, two = f32(1), f32(2)
    a = f32(f32(base_value * f32(one - p)) * f32(one - f32(two * p)))
    b = f32(f32(f32(middle_value * f32(4)) * p) * f32(one - p))
    c = f32(f32(tip_value * p) * f32(f32(two * p) - one))
    return f32(f32(a + b) + c)


@ports("addSorghumGrain")
def addSorghumGrain(ctx, center, grain_axis, width_axis, length, width, thickness, facets, color, UUIDs):
    """AS:2628-2654."""
    thickness_axis = cross(grain_axis, width_axis)
    half_length = f32(f32(0.5) * length)
    ring_height = (f32(f32(-0.4) * half_length), f32(f32(0.25) * half_length))
    base_pole = center - grain_axis * half_length
    apex_pole = center + grain_axis * half_length
    hl2 = f32(half_length * half_length)
    half_w, half_t = f32(f32(0.5) * width), f32(f32(0.5) * thickness)
    rings = ([], [])
    for ring in range(2):
        rh = ring_height[ring]
        profile = f32(np.sqrt(f32(f32(1) - f32(f32(rh * rh) / hl2))))
        along = center + grain_axis * rh
        wp, tp = f32(half_w * profile), f32(half_t * profile)
        for facet in range(facets):
            angle = f32(f32(GI.TWO_PI_F * f32(f32(facet) + f32(f32(0.5) * f32(ring)))) / f32(facets))
            rings[ring].append((along + width_axis * f32(wp * cosf(angle))) + thickness_axis * f32(tp * sinf(angle)))
    for facet in range(facets):
        nxt = (facet + 1) % facets
        UUIDs.append(GI.add_triangle(ctx, base_pole, rings[0][nxt], rings[0][facet], color))
        UUIDs.append(GI.add_triangle(ctx, rings[0][facet], rings[0][nxt], rings[1][facet], color))
        UUIDs.append(GI.add_triangle(ctx, rings[1][facet], rings[0][nxt], rings[1][nxt], color))
        UUIDs.append(GI.add_triangle(ctx, apex_pole, rings[1][facet], rings[1][nxt], color))


@ports("addSorghumGlumes")
def addSorghumGlumes(ctx, center, grain_axis, width_axis, length, width, thickness, glume_length_fraction, glume_spread,
                     color, UUIDs):
    """AS:2656-2677. ``std::sin(float)`` is ``sinf``; ``std::sqrt(float)`` is ``sqrtf``."""
    thickness_axis = cross(grain_axis, width_axis)
    half_length = f32(f32(0.5) * length)
    glume_length = f32(glume_length_fraction * length)
    hl2 = f32(half_length * half_length)

    def grain_profile(height):
        return f32(np.sqrt(max(f32(0), f32(f32(1) - f32(f32(height * height) / hl2)))))

    tip_height = min(f32(-half_length + glume_length), f32(f32(1.05) * half_length))
    side_height = f32(-half_length + f32(f32(0.55) * glume_length))
    base = center - grain_axis * f32(f32(1.05) * half_length)
    spread_term = f32(f32(f32(0.5) * sinf(deg2rad(glume_spread))) * glume_length)
    for face in (f32(-1), f32(1)):
        tip_offset = f32(f32(f32(f32(0.5) * thickness) * f32(f32(f32(1.05) * grain_profile(tip_height)) + f32(0.1)))
                         + spread_term)
        tip = (center + grain_axis * tip_height) + thickness_axis * f32(face * tip_offset)
        side_offset = thickness_axis * f32(f32(f32(face * f32(0.3)) * thickness) * grain_profile(side_height))
        side_center = (center + grain_axis * side_height) + side_offset
        side_extent = width_axis * f32(f32(f32(f32(0.48) * width) * grain_profile(side_height)) + f32(f32(0.05) * width))
        UUIDs.append(GI.add_triangle(ctx, base, side_center + side_extent, tip, color))
        UUIDs.append(GI.add_triangle(ctx, base, tip, side_center - side_extent, color))


def _vec_trig(fn, angles):
    """glibc sinf / cosf of a float32 array, element by element (0 maps to the exact 0 / 1 glibc returns)."""
    a = np.asarray(angles, f32)
    out = np.empty_like(a)
    for k, x in enumerate(a.tolist()):
        out[k] = fn(x)
    return out


class _Rachis:
    """The rachis curve and frames of AS:2706-2746, scalar and vectorised over arc lengths."""

    def __init__(self, rachis_length, total_bend, bend_direction):
        self.L = f32(rachis_length)
        self.bend = f32(total_bend)
        self.axis = GI.vec(1, 0, 0)
        self.bd = bend_direction
        self.straight = abs(self.bend) < f32(1e-4)
        self.R = None if self.straight else f32(self.L / self.bend)

    def position(self, s):
        s = f32(s)
        if self.straight:
            return self.axis * s
        angle = f32(s / self.R)
        return self.axis * f32(self.R * sinf(angle)) + self.bd * f32(self.R * f32(f32(1) - cosf(angle)))

    def tangent(self, s):
        angle = f32(f32(self.bend * f32(s)) / self.L)
        return self.axis * cosf(angle) + self.bd * sinf(angle)

    def normal(self, s):
        angle = f32(f32(self.bend * f32(s)) / self.L)
        return self.bd * cosf(angle) - self.axis * sinf(angle)

    # vectorised (s: float32 array of n) -> (n, 3)
    def positions(self, s):
        s = np.asarray(s, f32)
        if self.straight:
            return s[:, None] * self.axis[None, :]
        angle = (s / self.R).astype(f32)
        a = (self.R * _vec_trig(sinf, angle)).astype(f32)
        b = (self.R * (f32(1) - _vec_trig(cosf, angle))).astype(f32)
        return self.axis[None, :] * a[:, None] + self.bd[None, :] * b[:, None]

    def tangents(self, s):
        angle = ((self.bend * np.asarray(s, f32)) / self.L).astype(f32)
        return self.axis[None, :] * _vec_trig(cosf, angle)[:, None] + self.bd[None, :] * _vec_trig(sinf, angle)[:, None]


def _dot_rows(a, b):
    return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) + a[:, 2] * b[:, 2]


def _magnitude_rows(a):
    return np.sqrt(_dot_rows(a, a))


def _offset_from_axis(rachis, points, hints):
    """AS:2731-2745 for n points at once: the component of each point's offset from the axis (the rachis continued
    straight past its tip) perpendicular to it."""
    points = np.asarray(points, f32).reshape(-1, 3)
    hints = np.asarray(hints, f32).reshape(-1)
    arc = hints + _dot_rows(points - rachis.positions(hints), rachis.tangents(hints))
    arc = np.maximum(f32(0), arc).astype(f32)
    inside = arc <= rachis.L
    L = np.full(len(arc), rachis.L, f32)
    axis_point = np.where(inside[:, None], rachis.positions(np.where(inside, arc, L)), 0).astype(f32)
    tangent = rachis.tangents(np.where(inside, arc, L))
    if not inside.all():
        out = ~inside
        tip = rachis.positions(L[out]) + tangent[out] * (arc[out] - rachis.L)[:, None]
        axis_point[out] = tip
    offset = points - axis_point
    return offset - tangent * _dot_rows(offset, tangent)[:, None]


def _cell_key(ix, iy, iz):
    """AS:2900: ``(ix + 2^20) << 42 | (iy + 2^20) << 21 | (iz + 2^20)`` in int64."""
    off = np.int64(1 << 20)
    return ((ix.astype(np.int64) + off) << 42) | ((iy.astype(np.int64) + off) << 21) | (iz.astype(np.int64) + off)


_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def _resolve_overlaps(centers, scale, contact_radius, cell_size):
    """One overlap pass of AS:2966-3000, vectorised.

    The C++ visits, for each grain i ascending, the 27 cells around i's cell (dx, dy, dz from -1 to 1, nested in that
    order), and in each cell the grains j > i in the order they were entered (ascending index); an overlapping pair
    adds its push to displacement[i] and subtracts it from displacement[j]. So grain k receives, in this order: the
    subtractions from every pair (i, k) with i < k, i ascending; then the additions from its own pairs (k, j) in visit
    order. Pairs are found with sorted cell keys, the pushes computed in float32 as in the C++, and each grain's
    displacement is then accumulated term by term in that order (one masked add per rank)."""
    n = len(centers)
    cells = np.floor(centers / cell_size).astype(np.int64)                 # floor(position / cell_size), float division
    keys = _cell_key(cells[:, 0], cells[:, 1], cells[:, 2])
    order = np.argsort(keys, kind="stable")                                # each cell's grains in ascending index
    sorted_keys = keys[order]
    I_all, J_all, T_off, T_pos = [], [], [], []
    idx = np.arange(n)
    for o, (dx, dy, dz) in enumerate(_OFFSETS):
        nk = _cell_key(cells[:, 0] + dx, cells[:, 1] + dy, cells[:, 2] + dz)
        lo = np.searchsorted(sorted_keys, nk, side="left")
        hi = np.searchsorted(sorted_keys, nk, side="right")
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            continue
        ii = np.repeat(idx, cnt)
        start = np.repeat(lo, cnt)
        pos = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        jj = order[start + pos]
        keep = jj > ii
        I_all.append(ii[keep])
        J_all.append(jj[keep])
        T_off.append(np.full(int(keep.sum()), o, np.int64))
        T_pos.append(pos[keep])
    disp = np.zeros((n, 3), f32)
    if not I_all:
        return centers + disp
    I = np.concatenate(I_all)
    J = np.concatenate(J_all)
    Toff = np.concatenate(T_off)
    Tpos = np.concatenate(T_pos)
    sep = centers[I] - centers[J]
    distance = _magnitude_rows(sep)
    overlap = (contact_radius * (scale[I] + scale[J])).astype(f32) - distance
    hit = (overlap > 0) & (distance > f32(1e-9))
    I, J, Toff, Tpos, sep, distance, overlap = I[hit], J[hit], Toff[hit], Tpos[hit], sep[hit], distance[hit], overlap[hit]
    push = sep * ((f32(0.5) * overlap) / distance)[:, None]
    # contributions: (target, phase, i, offset, pos, sign); phase 0 = as j (from i < k), 1 = as i (own pairs)
    target = np.concatenate([J, I])
    phase = np.concatenate([np.zeros(len(I), np.int64), np.ones(len(I), np.int64)])
    t_i = np.concatenate([I, I])
    t_o = np.concatenate([Toff, Toff])
    t_p = np.concatenate([Tpos, Tpos])
    contrib = np.concatenate([push, push])
    negative = np.concatenate([np.ones(len(I), bool), np.zeros(len(I), bool)])
    srt = np.lexsort((t_p, t_o, t_i, phase, target))
    target, contrib, negative = target[srt], contrib[srt], negative[srt]
    first = np.r_[True, target[1:] != target[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(target)), 0))
    rank = np.arange(len(target)) - group_start
    for r in range(int(rank.max()) + 1 if len(rank) else 0):
        m = rank == r
        t = target[m]
        c = contrib[m]
        neg = negative[m]
        cur = disp[t]
        disp[t] = np.where(neg[:, None], cur - c, cur + c)
    return centers + disp


@ports("buildSorghumPanicle")
def buildSorghumPanicle(ctx, parameters):
    """AS:2679-3080."""
    p = parameters
    if (p.length <= 0 or p.grain_length <= 0 or p.grain_width <= 0 or p.grain_thickness <= 0 or p.rachis_radius <= 0
            or p.branch_radius <= 0):
        raise PlantArchitectureError("ERROR (buildSorghumPanicle): Panicle length, rachis and branch radii, and grain "
                                     "dimensions must all be positive.")
    elif p.rachis_length_fraction <= 0 or p.rachis_length_fraction > 1:
        raise PlantArchitectureError("ERROR (buildSorghumPanicle): rachis_length_fraction must be greater than 0 and at "
                                     f"most 1, but {_std_to_string(p.rachis_length_fraction)} was given.")
    elif p.primary_branch_count < 1 or p.grain_count < 1 or p.grains_per_raceme < 1:
        raise PlantArchitectureError("ERROR (buildSorghumPanicle): primary_branch_count, grain_count and "
                                     "grains_per_raceme must each be at least 1.")
    elif p.branch_bare_fraction < 0 or p.branch_bare_fraction >= 1:
        raise PlantArchitectureError("ERROR (buildSorghumPanicle): branch_bare_fraction must be in the range [0,1), but "
                                     f"{_std_to_string(p.branch_bare_fraction)} was given.")
    elif p.compactness < 0 or p.compactness > 1:
        raise PlantArchitectureError("ERROR (buildSorghumPanicle): compactness must be in the range [0,1], but "
                                     f"{_std_to_string(p.compactness)} was given.")
    elif p.glume_length_fraction < 0:
        raise PlantArchitectureError("ERROR (buildSorghumPanicle): glume_length_fraction cannot be negative.")
    elif p.grain_subdivisions < 3 or p.branch_radial_subdivisions < 3 or p.rachis_radial_subdivisions < 3:
        raise PlantArchitectureError("ERROR (buildSorghumPanicle): grain_subdivisions, branch_radial_subdivisions and "
                                     "rachis_radial_subdivisions must each be at least 3.")

    one = f32(1)
    size_factor = clamp(f32(one + f32(p.length_variation * ctx.randn())), f32(0.7), f32(1.3))
    panicle_length = f32(p.length * size_factor)
    grain_count = max(1, int(GI.roundf(f32(f32(p.grain_count) * size_factor))))

    # ---- rachis ----
    rachis_length = f32(p.rachis_length_fraction * panicle_length)
    total_bend = deg2rad(p.rachis_curvature)
    bend_azimuth = ctx.randu(f32(0), GI.TWO_PI_F)
    bend_direction = GI.vec(0, cosf(bend_azimuth), sinf(bend_azimuth))
    rachis = _Rachis(rachis_length, total_bend, bend_direction)
    binormal = cross(rachis.axis, bend_direction)

    # ---- primary branches ----
    branch_segments = 5
    nseg = f32(branch_segments)
    nb = p.primary_branch_count
    br_nodes, br_length, br_arc, br_outward = [], [], [], []
    azimuth_offset = ctx.randu(f32(0), GI.TWO_PI_F)
    golden = deg2rad(f32(137.5))
    for b in range(nb):
        position = clamp(f32(f32(f32(f32(b) + f32(0.5)) + ctx.randu(f32(-0.3), f32(0.3))) / f32(nb)), f32(0), one)
        arc = f32(position * rachis_length)
        length_fraction = max(f32(0.02), interpolateRachisProfile(p.branch_length_base, p.branch_length_middle,
                                                                  p.branch_length_tip, position))
        length = f32(f32(length_fraction * panicle_length)
                     * clamp(f32(one + f32(p.branch_length_variation * ctx.randn())), f32(0.4), f32(1.6)))
        insertion_angle = f32(f32(p.branch_angle_base + f32(f32(p.branch_angle_tip - p.branch_angle_base) * position))
                              + ctx.randu(f32(-5), f32(5)))
        upcurve = p.branch_upcurve
        if ctx.randu() < p.splayed_branch_fraction:
            insertion_angle = p.splayed_branch_angle
            upcurve = f32(upcurve * f32(0.5))
        insertion_angle = deg2rad(clamp(insertion_angle, f32(0), f32(90)))
        upcurve = deg2rad(upcurve)
        azimuth = f32(f32(azimuth_offset + f32(golden * f32(b))) + ctx.randu(f32(-0.3), f32(0.3)))
        tangent = rachis.tangent(arc)
        outward = rachis.normal(arc) * cosf(azimuth) + binormal * sinf(azimuth)
        nodes = [rachis.position(arc)]
        segment_length = f32(length / nseg)
        for segment in range(branch_segments):
            angle = max(f32(0), f32(insertion_angle - f32(f32(upcurve * f32(f32(segment) + f32(0.5))) / nseg)))
            direction = tangent * cosf(angle) + outward * sinf(angle)
            nodes.append(nodes[segment] + direction * segment_length)
        br_nodes.append(nodes)
        br_length.append(length)
        br_arc.append(arc)
        br_outward.append(outward)

    def branch_point(b, fraction):
        scaled = f32(clamp(f32(fraction), f32(0), one) * nseg)
        segment = min(int(scaled), branch_segments - 1)
        local = f32(scaled - f32(segment))
        n0, n1 = br_nodes[b][segment], br_nodes[b][segment + 1]
        return n0 + (n1 - n0) * local, normalize(n1 - n0)

    # ---- racemes and grains ----
    raceme_count = max(1, int(GI.roundf(f32(f32(grain_count) / f32(p.grains_per_raceme)))))
    fertile = []
    total_fertile = f32(0)
    for b in range(nb):
        fertile.append(f32(f32(one - p.branch_bare_fraction) * br_length[b]))
        total_fertile = f32(total_fertile + fertile[b])
    branch_raceme_count, remainders, assigned = [], [], 0
    for b in range(nb):
        share = f32(f32(fertile[b] / total_fertile) * f32(raceme_count))
        fl = f32(math.floor(float(share)))
        branch_raceme_count.append(int(fl))
        remainders.append((float(f32(share - fl)), b))
        assigned += int(fl)
    remainders.sort(reverse=True)                       # std::sort(rbegin, rend) of pair<float, int>
    for i in range(raceme_count - assigned):
        branch_raceme_count[remainders[i][1]] += 1

    racemes = []          # (branch, fraction, first_grain, grain_count)
    g_anchor, g_axis, g_width, g_scale, g_arc = [], [], [], [], []
    raceme_angle = deg2rad(p.raceme_angle)
    cos_ra, sin_ra = cosf(raceme_angle), sinf(raceme_angle)
    per, extra = grain_count // raceme_count, grain_count % raceme_count
    half_gw = f32(f32(0.5) * p.grain_width)
    for b in range(nb):
        nr = branch_raceme_count[b]
        for r in range(nr):
            raceme_index = len(racemes)
            raceme_grains = per + (1 if raceme_index < extra else 0)
            fraction = f32(p.branch_bare_fraction + f32(f32(f32(one - p.branch_bare_fraction)
                                                            * f32(f32(r) + ctx.randu(f32(0.2), f32(0.8)))) / f32(nr)))
            racemes.append((b, fraction, len(g_anchor), raceme_grains))
            base, branch_direction = branch_point(b, fraction)
            side_reference = GI.perpendicularTo(branch_direction)
            side_angle = ctx.randu(f32(0), GI.TWO_PI_F)
            side = side_reference * cosf(side_angle) + cross(branch_direction, side_reference) * sinf(side_angle)
            so = dot(side, br_outward[b])
            if so < 0:
                side = side - br_outward[b] * f32(f32(1.5) * so)
                side = normalize(side - branch_direction * dot(side, branch_direction))
            raceme_direction = branch_direction * cos_ra + side * sin_ra
            raceme_length = max(f32(p.raceme_length_fraction * br_length[b]),
                                f32(f32(f32(0.6) * f32(raceme_grains)) * p.grain_length))
            alternation_axis = normalize(cross(raceme_direction, side))
            for g in range(raceme_grains):
                along = f32(f32(f32(g) + f32(0.75)) / f32(raceme_grains))
                alternation = one if g % 2 == 0 else f32(-1)
                lateral = normalize(side + alternation_axis * f32(f32(0.8) * alternation))
                s = clamp(f32(one + f32(p.grain_size_variation * ctx.randn())), f32(0.7), f32(1.3))
                anchor = (base + raceme_direction * f32(raceme_length * along)) + lateral * f32(half_gw * s)
                axis = normalize(raceme_direction * f32(0.8) + lateral * f32(0.6))
                g_anchor.append(anchor)
                g_axis.append(axis)
                g_width.append(normalize(cross(axis, lateral)))
                g_scale.append(s)
                g_arc.append(br_arc[b])

    anchor = np.array(g_anchor, f32).reshape(-1, 3)
    center = anchor.copy()
    scale = np.array(g_scale, f32)
    garc = np.array(g_arc, f32)
    ng = len(anchor)

    # ---- packing ----
    packing_iterations, settling_iterations = 24, 10
    compaction_rate = f32(0.25)
    anchor_spring = f32(f32(0.1) * f32(one - p.compactness))
    contact_radius = f32(f32(0.25) * f32(p.grain_width + p.grain_thickness))
    layer_depth = f32(f32(3) * contact_radius)
    cell_size = f32(f32(f32(2) * f32(1.3)) * contact_radius)

    envelope_bands = 24
    axial = garc + _dot_rows(anchor - rachis.positions(garc), rachis.tangents(garc))
    band = np.array([int(v) for v in ((f32(envelope_bands) * axial) / panicle_length).astype(f32).tolist()])
    band = np.clip(band, 0, envelope_bands - 1)
    anchor_radius = _magnitude_rows(_offset_from_axis(rachis, anchor, garc))
    band_radii = [sorted(anchor_radius[band == k].tolist()) for k in range(envelope_bands)]
    band_envelope = [f32(0)] * envelope_bands
    for k in range(envelope_bands):
        radii = band_radii[k]
        if radii:
            band_envelope[k] = f32(radii[min(len(radii) - 1, int(f32(f32(0.85) * f32(len(radii)))))])
    envelope = list(band_envelope)
    for k in range(envelope_bands):
        total, count = f32(0), 0
        for nbh in range(max(0, k - 1), min(envelope_bands - 1, k + 1) + 1):
            if band_radii[nbh]:
                total = f32(total + band_envelope[nbh])
                count += 1
        envelope[k] = f32(total / f32(count)) if count > 0 else f32(0)
    outer = np.array(envelope, f32)[band]
    inner = np.maximum(f32(p.rachis_radius + f32(0)) + contact_radius * scale, outer - layer_depth).astype(f32)
    rate = f32(p.compactness * compaction_rate)

    for iteration in range(packing_iterations + settling_iterations):
        if iteration < packing_iterations:
            radial = _offset_from_axis(rachis, center, garc)
            radial_distance = _magnitude_rows(radial)
            target = np.where(radial_distance > outer, outer, np.where(radial_distance < inner, inner, radial_distance))
            move = radial_distance > f32(1e-6)
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = (rate * ((target / radial_distance) - one)).astype(f32)
            center = np.where(move[:, None], center + radial * factor[:, None], center).astype(f32)
            center = center + (anchor - center) * anchor_spring
        for _pass in range(3):
            center = _resolve_overlaps(center, scale, contact_radius, cell_size)

    # ---- branch geometry ----
    UUIDs: list = []
    rachis_segments = 10
    rachis_nodes, rachis_radii = [], []
    for node in range(rachis_segments + 1):
        fraction = f32(f32(node) / f32(rachis_segments))
        rachis_nodes.append(rachis.position(f32(fraction * rachis_length)))
        rachis_radii.append(f32(p.rachis_radius * f32(one - f32(f32(0.67) * fraction))))
    UUIDs += GI.context_add_tube(ctx, p.rachis_radial_subdivisions, rachis_nodes, rachis_radii,
                                 [p.branch_color] * len(rachis_nodes))

    final_radius = _magnitude_rows(_offset_from_axis(rachis, center, garc))
    anchor_sum = [f32(0)] * nb
    final_sum = [f32(0)] * nb
    for b_, _fr, first, count in racemes:
        for g in range(first, first + count):
            anchor_sum[b_] = f32(anchor_sum[b_] + anchor_radius[g])
            final_sum[b_] = f32(final_sum[b_] + final_radius[g])
    half = f32(0.5)
    for b in range(nb):
        if anchor_sum[b] > 0:
            radial_ratio = min(one, f32(final_sum[b] / anchor_sum[b]))
            pull = f32(one - radial_ratio)
            for node in range(1, branch_segments + 1):
                radial = _offset_from_axis(rachis, br_nodes[b][node][None, :], [br_arc[b]])[0]
                br_nodes[b][node] = br_nodes[b][node] - radial * pull
        radii = [f32(p.branch_radius * f32(one - f32(f32(half * f32(node)) / nseg))) for node in range(branch_segments + 1)]
        UUIDs += GI.context_add_tube(ctx, p.branch_radial_subdivisions, br_nodes[b], radii,
                                     [p.branch_color] * len(br_nodes[b]))

    raceme_radii = [f32(f32(0.6) * p.branch_radius), f32(f32(0.4) * p.branch_radius)]
    for b_, fraction, first, count in racemes:
        if count == 0:
            continue
        base, _direction = branch_point(b_, fraction)
        tip = center[first + count - 1]
        if V.magnitude(tip - base) < f32(1e-5):
            continue
        UUIDs += GI.context_add_tube(ctx, p.branch_radial_subdivisions, [base, tip], raceme_radii, [p.branch_color] * 2)

    # ---- grain geometry ----
    half_cv = f32(f32(0.5) * p.grain_color_variation)
    for g in range(ng):
        brightness = clamp(f32(one + f32(p.grain_color_variation * ctx.randn())), f32(0.6), f32(1.4))
        grain_color = GI.rgb_scale(p.grain_color, brightness)
        length = f32(p.grain_length * scale[g])
        width = f32(p.grain_width * scale[g])
        thickness = f32(p.grain_thickness * scale[g])
        addSorghumGrain(ctx, center[g], g_axis[g], g_width[g], length, width, thickness, int(p.grain_subdivisions),
                        grain_color, UUIDs)
        if p.glume_length_fraction > 0:
            glume_color = GI.rgb_scale(p.glume_color, clamp(f32(one + f32(half_cv * ctx.randn())), f32(0.6), f32(1.4)))
            addSorghumGlumes(ctx, center[g], g_axis[g], g_width[g], length, width, thickness, p.glume_length_fraction,
                             p.glume_spread, glume_color, UUIDs)

    ctx.set_primitive_data(UUIDs, "object_label", "fruit")
    return ctx.add_polymesh_object(UUIDs)


@named_hook("SorghumPaniclePrototype").implement
@ports("SorghumPaniclePrototype")
def SorghumPaniclePrototype(scene, subdivisions, build_parameters=None):
    """AS:3082-3084."""
    return buildSorghumPanicle(scene, SorghumPanicleParameters.fromBuildParameters(build_parameters or {}))
