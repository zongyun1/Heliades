# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:24-58, 100-181, 399-433, 521-524, 3567-4030 and
# src/Assets.cpp:1760-2005 (Copyright (C) 2016-2026 Brian Bailey).
"""Grapevine, vertical-shoot-positioned trellis (``grapevine_VSP``), W4-D.

* ``initializeGrapevineVSPShoots`` (PL:3567-3785): five shoot types (trunk, cane, shoot, shoot_mirrored,
  lateral). Literals keep their C++ types: ``0.08 * 0.079 / 0.14`` is a double expression narrowed once,
  ``180.f - 30.f`` is float, ``uniformDistribution(-0.4, 0.4)`` narrows each bound.
* ``buildGrapevineVSP`` (PL:3787-4030) with ``BoundedWander`` (PL:24-58) and ``grapevineTrunkPath``
  (PL:132-181): a prescribed trunk and two prescribed canes (``addShootFromNodePositions``), catch wires as
  per-plant attraction points, trained buds woken, trained leaves removed.
* ``GrapevinePhytomerCreationFunction`` / ``GrapevinePhytomerCallbackFunction`` (AS:1943-2005).
* ``GrapeClusterParameters`` and the build-parameter declarations (PL:399, 432; AS:1760-1772).
* ``GrapevineFruitPrototype`` (AS:1774-1935) with a local ``Context::addSphere``; called through W3-D's
  inflorescence machinery (``fruit_prototype_function(ctx, 1, organ_build_parameters)``).

Float maths: ``std::sin`` / ``std::cos`` / ``std::exp`` of a float are glibc ``sinf`` / ``cosf`` / ``expf``
(not correctly rounded, called through ctypes). Function arguments are evaluated right to left (GCC), so
``addComponent(randu(a), randu(b), randu(c))`` draws the phase first.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..hooks import Hook
from ..params import BUD_ACTIVE, BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_RGBcolor
from .build_parameters import (BuildParameterDeclaration, getBuildParameterValue, registerPlantModelBuildParameters)

f32 = np.float32
PI_F = f32(math.pi)

# ---- function-pointer hooks (Assets.cpp) -------------------------------------------------------------
GrapevineFruitPrototype = Hook("GrapevineFruitPrototype")                       # AS:1774
GrapevinePhytomerCreationFunction = Hook("GrapevinePhytomerCreationFunction")   # AS:1943
GrapevinePhytomerCallbackFunction = Hook("GrapevinePhytomerCallbackFunction")   # AS:1987


def _libm():
    m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    fns = {}
    for name in ("expf", "sinf", "cosf", "cbrtf", "powf"):
        fn = getattr(m, name)
        fn.restype = ctypes.c_float
        fn.argtypes = [ctypes.c_float, ctypes.c_float] if name == "powf" else [ctypes.c_float]
        fns[name] = fn
    return fns


_M = _libm()


def expf(x) -> np.float32:
    """``std::exp(float)`` = glibc ``expf``."""
    return f32(_M["expf"](float(f32(x))))


sinf, cosf = V.sinf, V.cosf


# ---- build parameters (PL:399, 432; AS:1760-1772) -------------------------------------------------------
grapevine_vine_spacing = BuildParameterDeclaration("vine_spacing", 0.5, 5.0, False, "plant-to-plant spacing in meters")
grapevine_VSP_trunk_height = BuildParameterDeclaration("trunk_height", 0.05, 1.0, False, "height of the fruiting wire in meters")
grape_berry_diameter = BuildParameterDeclaration("berry_diameter", 0.004, 0.03, False, "grape berry diameter in meters")
grape_cluster_berry_count = BuildParameterDeclaration("cluster_berry_count", 10.0, 400.0, False,
                                                      "mean number of berries in a grape cluster")


class GrapeClusterParameters:
    """GrapeClusterParameters (Assets.h): in-class defaults, overridden by the organ build parameters."""

    def __init__(self):
        self.berry_diameter = f32(0.012)
        self.cluster_berry_count = f32(90)

    @staticmethod
    @ports("GrapeClusterParameters::fromBuildParameters")
    def fromBuildParameters(build_parameters: dict) -> "GrapeClusterParameters":
        p = GrapeClusterParameters()
        p.berry_diameter = getBuildParameterValue(build_parameters, grape_berry_diameter, p.berry_diameter)
        p.cluster_berry_count = getBuildParameterValue(build_parameters, grape_cluster_berry_count, p.cluster_berry_count)
        return p

    @staticmethod
    def buildParameterDeclarations():
        return [grape_berry_diameter, grape_cluster_berry_count]


def _validate_grape_cluster_parameters(parameters):
    GrapeClusterParameters.fromBuildParameters(parameters)


def register_build_parameters(pa):
    """PL:521-522."""
    registerPlantModelBuildParameters(pa, "grapevine_VSP", [grapevine_vine_spacing, grapevine_VSP_trunk_height],
                                      GrapeClusterParameters.buildParameterDeclarations(),
                                      _validate_grape_cluster_parameters)


# ---- BoundedWander (PL:24-58) ----------------------------------------------------------------------------

class BoundedWander:
    def __init__(self):
        self.amplitudes, self.wavelengths, self.phases = [], [], []

    def addComponent(self, amplitude, wavelength, phase):
        self.amplitudes.append(f32(amplitude))
        self.wavelengths.append(f32(wavelength))
        self.phases.append(f32(phase))

    def evaluate(self, distance) -> np.float32:
        """``departure += a * std::sin(2.f * PI_F * distance / wl + phase)`` in float, left to right."""
        distance = f32(distance)
        two_pi = f32(f32(2) * PI_F)
        departure = f32(0)
        for a, wl, ph in zip(self.amplitudes, self.wavelengths, self.phases):
            departure = f32(departure + f32(a * sinf(f32(f32(f32(two_pi * distance) / wl) + ph))))
        return departure


def _add_component(ctx, wander, amp_range, wl_range, phase_range):
    """``wander.addComponent(randu(amp), randu(wl), randu(phase))``: GCC evaluates the arguments right to left."""
    phase = ctx.randu(f32(phase_range[0]), f32(phase_range[1]))
    wavelength = ctx.randu(f32(wl_range[0]), f32(wl_range[1]))
    amplitude = ctx.randu(f32(amp_range[0]), f32(amp_range[1]))
    wander.addComponent(amplitude, wavelength, phase)


TWO_PI_F = f32(f32(2) * PI_F)


@ports("grapevineTrunkPath")
def grapevineTrunkPath(ctx, base_position, head_height, nominal_radius):
    """PL:147-181. Returns (trunk_nodes, node_positions, node_radii)."""
    base_position = np.asarray(base_position, f32)
    head_height, nominal_radius = f32(head_height), f32(nominal_radius)
    trunk_nodes = max(6, int(np.ceil(f32(head_height / f32(0.04)))))
    trunk_node_spacing = f32(head_height / f32(trunk_nodes))

    sway_x, sway_y = BoundedWander(), BoundedWander()
    for sway in (sway_x, sway_y):
        _add_component(ctx, sway, (0.015, 0.04), (1.2, 2.2), (0.0, TWO_PI_F))
        _add_component(ctx, sway, (0.002, 0.006), (0.3, 0.6), (0.0, TWO_PI_F))
    head_offset_x = ctx.randu(f32(-0.015), f32(0.015))
    trunk_lean_x = f32(f32(head_offset_x - f32(sway_x.evaluate(head_height) - sway_x.evaluate(0))) / head_height)
    trunk_lean_y = ctx.randu(f32(-0.06), f32(0.06))

    trunk_radius = f32(nominal_radius * ctx.randu(f32(0.85), f32(1.15)))
    head_swelling = ctx.randu(f32(0.3), f32(0.6))
    lumps = BoundedWander()
    _add_component(ctx, lumps, (0.03, 0.06), (0.12, 0.3), (0.0, TWO_PI_F))

    dome_angles_degrees = (f32(30), f32(60), f32(85))
    positions, radii = [], []
    sx0, sy0 = sway_x.evaluate(0), sway_y.evaluate(0)
    for node in range(trunk_nodes + len(dome_angles_degrees) + 1):
        height = f32(f32(node) * trunk_node_spacing)
        dome_radius_fraction = f32(1)
        if node > trunk_nodes:
            dome_angle = V.deg2rad(dome_angles_degrees[node - trunk_nodes - 1])
            height = f32(head_height + f32(radii[trunk_nodes] * sinf(dome_angle)))
            dome_radius_fraction = cosf(dome_angle)
        girth_height = min(height, head_height)
        root_flare = f32(f32(0.35) * expf(f32(-girth_height / f32(0.05))))
        taper = f32(f32(f32(-0.18) * girth_height) / head_height)
        head_distance = f32(f32(girth_height - f32(head_height - f32(0.02))) / f32(0.06))
        head = f32(head_swelling * expf(f32(-head_distance * head_distance)))
        radius = f32(f32(trunk_radius * f32(f32(f32(f32(1) + root_flare) + taper) + head))
                     * f32(f32(1) + lumps.evaluate(girth_height)))
        offset_x = f32(f32(f32(trunk_lean_x * height) + sway_x.evaluate(height)) - sx0)
        offset_y = f32(f32(f32(trunk_lean_y * height) + sway_y.evaluate(height)) - sy0)
        positions.append(base_position + np.array([offset_x, offset_y, height], f32))
        radii.append(f32(radius * dome_radius_fraction))
    return trunk_nodes, positions, radii


# ---- shoot types (PL:3567-3785) -----------------------------------------------------------------------

@ports("PlantArchitecture::initializeGrapevineVSPShoots")
def initializeGrapevineVSPShoots(pa):
    gen = context_generator(pa.context_ptr)

    # ---- Leaf Prototype ---- //
    leaf_prototype = LeafPrototype(gen)
    leaf_prototype.leaf_texture_file[0] = "GrapeLeaf.png"
    leaf_prototype.leaf_aspect_ratio = f32(1)
    leaf_prototype.midrib_fold_fraction = f32(0.3)
    leaf_prototype.longitudinal_curvature.uniformDistribution(-0.4, 0.4)
    leaf_prototype.lateral_curvature = 0
    leaf_prototype.wave_period = f32(0.3)
    leaf_prototype.wave_amplitude = f32(0.1)
    leaf_prototype.subdivisions = 5
    leaf_prototype.unique_prototypes = 10
    leaf_prototype.leaf_offset = (-0.3, 0, 0)

    # ---- Phytomer Parameters ---- //
    pp = PhytomerParameters(gen)
    pp.internode.pitch = 20
    pp.internode.phyllotactic_angle.uniformDistribution(170, 190)
    pp.internode.radius_initial = 0.003
    pp.internode.color = make_RGBcolor(0.23, 0.13, 0.062)
    pp.internode.length_segments = 1
    pp.internode.max_floral_buds_per_petiole = 1
    pp.internode.max_vegetative_buds_per_petiole = 1

    pp.petiole.petioles_per_internode = 1
    pp.petiole.color = make_RGBcolor(0.35, 0.38, 0.15)
    pp.petiole.pitch.uniformDistribution(45, 70)
    pp.petiole.radius = 0.0025
    pp.petiole.length = 0.08
    pp.petiole.taper = 0
    pp.petiole.curvature = 0
    pp.petiole.length_segments = 1

    pp.leaf.leaves_per_petiole = 1
    pp.leaf.pitch.uniformDistribution(-110, -80)
    pp.leaf.yaw.uniformDistribution(-20, 20)
    pp.leaf.roll.uniformDistribution(-5, 5)
    pp.leaf.prototype_scale = 0.14
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length = 0.08
    pp.peduncle.pitch.uniformDistribution(50, 90)
    pp.peduncle.yaw = 180
    pp.peduncle.color = make_RGBcolor(0.32, 0.05, 0.13)

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.flower_prototype_scale = 0.04
    pp.inflorescence.fruit_prototype_function = GrapevineFruitPrototype
    pp.inflorescence.fruit_prototype_scale = 1
    pp.inflorescence.unique_prototypes = 5
    pp.inflorescence.fruit_gravity_factor_fraction = 0.7

    pp.phytomer_creation_function = GrapevinePhytomerCreationFunction
    pp.phytomer_callback_function = GrapevinePhytomerCallbackFunction

    # ---- Shoot Parameters ---- //
    main = ShootParameters(gen)
    main.phytomer_parameters = pp
    main.vegetative_bud_break_probability_min = 0.30
    main.vegetative_bud_break_probability_max = 0.30
    main.vegetative_bud_break_probability_decay_rate = 0.
    main.vegetative_bud_break_time = 30
    main.phyllochron_min.uniformDistribution(1.75, 2.25)
    main.elongation_rate_max = 0.15
    main.girth_area_factor = f32(1)
    main.gravitropic_curvature = 300
    main.tortuosity = 226
    main.internode_length_max.uniformDistribution(0.048, 0.058)
    main.internode_length_decay_rate = 0
    main.insertion_angle_tip.uniformDistribution(55, 125)
    main.insertion_angle_decay_rate = 0
    main.flowers_require_dormancy = False
    main.growth_requires_dormancy = False
    main.determinate_shoot_growth = False
    main.max_terminal_floral_buds = 0
    main.flower_bud_break_probability = 0.5
    main.fruit_set_probability = 1.0
    main.max_nodes = 20
    main.base_roll.uniformDistribution(90 - 15, 90 + 15)
    main.base_yaw.uniformDistribution(-30, 30)

    cane = main.copy()                                       # ShootParameters cane = main (copy constructor)
    cane.phytomer_parameters.internode.pitch = 0
    cane.phytomer_parameters.internode.radial_subdivisions = 15
    cane.phytomer_parameters.internode.max_floral_buds_per_petiole = 0
    cane.phytomer_parameters.internode.phyllotactic_angle = 0
    cane.insertion_angle_tip.uniformDistribution(60, 120)
    cane.girth_area_factor = f32(0.7)
    cane.max_nodes = 25
    cane.tortuosity = 0
    cane.gravitropic_curvature = 0
    cane.vegetative_bud_break_probability_min = 0.9
    cane.defineChildShootTypes(["grapevine_shoot"], [1.0])

    trunk = main.copy()
    trunk.phytomer_parameters.internode.image_texture = "GrapeBark.jpg"
    trunk.phytomer_parameters.internode.pitch = 0
    trunk.phytomer_parameters.internode.phyllotactic_angle = 180
    trunk.phytomer_parameters.internode.radius_initial = 0.04
    trunk.phytomer_parameters.internode.radial_subdivisions = 25
    trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0
    trunk.phyllochron_min = 2.5
    trunk.insertion_angle_tip = 90
    trunk.girth_area_factor = 0
    trunk.max_nodes = 40
    trunk.tortuosity = 0
    trunk.vegetative_bud_break_probability_min = 0
    trunk.defineChildShootTypes(["grapevine_shoot"], [1.0])

    lateral = main.copy()
    lateral.phytomer_parameters.leaf.prototype_scale = 0.079
    lateral.phytomer_parameters.petiole.length = 0.08 * 0.079 / 0.14          # double, narrowed once
    lateral.phytomer_parameters.petiole.radius = 0.0025 * 0.079 / 0.14
    lateral.phytomer_parameters.internode.max_floral_buds_per_petiole = 0
    lateral.insertion_angle_tip.uniformDistribution(10, 30)
    lateral.base_yaw.uniformDistribution(-15, 15)
    lateral.max_nodes = 9
    lateral.vegetative_bud_break_probability_min = 0
    lateral.vegetative_bud_break_probability_max = 0
    lateral.defineChildShootTypes(["grapevine_lateral"], [1.0])

    main.defineChildShootTypes(["grapevine_lateral"], [1.0])

    mirrored = main.copy()
    mirrored.base_yaw.uniformDistribution(f32(f32(180) - f32(30)), f32(f32(180) + f32(30)))
    mirrored.defineChildShootTypes(["grapevine_lateral"], [1.0])

    pa.defineShootType("grapevine_trunk", trunk)
    pa.defineShootType("grapevine_cane", cane)
    pa.defineShootType("grapevine_shoot", main)
    pa.defineShootType("grapevine_shoot_mirrored", mirrored)
    pa.defineShootType("grapevine_lateral", lateral)


# ---- builder (PL:3787-4030) ------------------------------------------------------------------------------

def _linspace3(a, b, num):
    """helios::linspace(vec3, vec3, int): ``start + step * float(i)`` with ``step = (end - start) / float(num-1)``
    (vec3 / float multiplies by the reciprocal); the last point is ``end`` exactly."""
    a, b = np.asarray(a, f32), np.asarray(b, f32)
    if num == 1:
        return [a.copy()]
    step = V.div(b - a, f32(num - 1))
    out = [a + step * f32(i) for i in range(num)]
    out[-1] = b.copy()
    return out


@ports("PlantArchitecture::buildGrapevineVSP")
def buildGrapevineVSP(pa, base_position):
    from ..shoot import addShootFromNodePositions, removeShootLeaves, setPlantAttractionPoints
    if not pa.shoot_types:
        initializeGrapevineVSPShoots(pa)
    ctx = pa.context_ptr
    base_position = np.asarray(base_position, f32).reshape(3).copy()

    vine_spacing = getBuildParameterValue(pa.current_build_parameters, grapevine_vine_spacing, f32(2.4))
    trunk_height = getBuildParameterValue(pa.current_build_parameters, grapevine_VSP_trunk_height, f32(0.8))

    cane_internode_length = f32(0.10)
    cane_total_length = f32(vine_spacing / f32(2))
    cane_nodes = int(f32(cane_total_length / cane_internode_length))
    if cane_nodes < 1:
        cane_nodes = 1

    # catch wires (attraction points)
    catch_wire_half_separation = f32(0.075)
    catch_wire_pairs = 4
    catch_wire_lowest = f32(0.35)
    catch_wire_spacing = f32(0.20)
    catch_wire_samples = max(2, int(f32(vine_spacing / f32(0.06))))
    trellis_points = []
    for wire in range(catch_wire_pairs):
        wire_height = f32(f32(trunk_height + catch_wire_lowest) + f32(f32(wire) * catch_wire_spacing))
        for side in (-catch_wire_half_separation, catch_wire_half_separation):
            half = f32(f32(0.5) * vine_spacing)
            trellis_points.append(_linspace3((side, -half, wire_height), (side, half, wire_height), catch_wire_samples))
    trellis_points = [[p + base_position for p in wire] for wire in trellis_points]

    plantID = pa.addPlantInstance(base_position, 0)
    setPlantAttractionPoints(pa, plantID, [p for wire in trellis_points for p in wire], 70.0, 0.35, 0.15)

    fruiting_wire_height = f32(base_position[2] + trunk_height)

    # ---- Trunk ---- //
    head_height = f32(trunk_height * f32(f32(1) - ctx.randu(f32(0.08), f32(0.16))))
    nominal = pa.shoot_types["grapevine_trunk"].phytomer_parameters.internode.radius_initial.val()
    trunk_nodes, trunk_node_positions, trunk_node_radii = grapevineTrunkPath(ctx, base_position, head_height, nominal)
    uID_stem = addShootFromNodePositions(pa, plantID, -1, 0, trunk_node_positions, trunk_node_radii, "grapevine_trunk")

    # ---- Canes ---- //
    tree = pa.plant_instances[plantID].shoot_tree
    trunk_phytomers = tree[uID_stem].phytomers
    head_phytomer = trunk_nodes - 1
    head_bud_faces_negative_y = bool(trunk_phytomers[head_phytomer].getPetioleAxisVector(f32(0), 0)[1] < 0)

    uID_cane_L = uID_cane_R = 0
    for direction in (f32(-1), f32(1)):
        attachment_phytomer = head_phytomer if ((direction < 0) == head_bud_faces_negative_y) else head_phytomer - 1
        attachment_node = trunk_node_positions[attachment_phytomer + 1]

        rise = f32(fruiting_wire_height - attachment_node[2])
        bend_length = f32(rise * ctx.randu(f32(1.8), f32(3)))

        wander_x, wander_z = BoundedWander(), BoundedWander()
        _add_component(ctx, wander_x, (0.008, 0.018), (0.4, 0.8), (0.0, TWO_PI_F))
        _add_component(ctx, wander_x, (0.003, 0.006), (0.15, 0.3), (0.0, TWO_PI_F))
        _add_component(ctx, wander_z, (0.006, 0.015), (0.5, 0.9), (0.0, TWO_PI_F))
        _add_component(ctx, wander_z, (0.003, 0.006), (0.18, 0.3), (0.0, TWO_PI_F))

        free_tip_length = f32(0.2)
        free_tip_deflection = ctx.randu(f32(-0.03), f32(0.02))
        cane_length = f32(cane_total_length * ctx.randu(f32(0.93), f32(1)))

        lengths = []
        unscaled = f32(0)
        for internode in range(cane_nodes):
            basal_shortening = min(f32(1), f32(f32(0.4) + f32(f32(0.2) * f32(internode))))
            lengths.append(f32(basal_shortening * ctx.randu(f32(0.8), f32(1.2))))
            unscaled = f32(unscaled + lengths[-1])
        ratio = f32(cane_length / unscaled)
        lengths = [f32(L * ratio) for L in lengths]

        ax_minus_bx = f32(attachment_node[0] - base_position[0])

        def cane_position(row_distance, _bl=bend_length, _cl=cane_length, _rise=rise, _wx=wander_x, _wz=wander_z,
                          _ftd=free_tip_deflection, _an=attachment_node, _dir=direction, _axbx=ax_minus_bx):
            row_distance = f32(row_distance)
            bend_fraction = min(f32(row_distance / _bl), f32(1))
            on_wire = f32(f32(bend_fraction * bend_fraction) * f32(f32(3) - f32(f32(2) * bend_fraction)))
            tip_fraction = max(f32(0), f32(f32(row_distance - f32(_cl - free_tip_length)) / free_tip_length))
            x = f32(f32(f32(f32(1) - on_wire) * _axbx) + f32(on_wire * _wx.evaluate(row_distance)))
            one_bf = f32(f32(1) - bend_fraction)
            z = f32(f32(f32(f32(-_rise * one_bf) * one_bf) + f32(on_wire * _wz.evaluate(row_distance)))
                    + f32(f32(_ftd * tip_fraction) * tip_fraction))
            return np.array([f32(base_position[0] + x), f32(_an[1] + f32(_dir * row_distance)),
                             f32(fruiting_wire_height + z)], f32)

        cane_node_positions, cane_node_radii = [], []
        cane_radius = f32(f32(0.0055) * ctx.randu(f32(0.9), f32(1.15)))
        row_distance = f32(0)
        path_distance = f32(0)
        for node in range(cane_nodes + 1):
            node_position = cane_position(row_distance)
            if node > 0:
                zigzag = f32((f32(1) if node % 2 == 0 else f32(-1)) * ctx.randu(f32(0.001), f32(0.004)))
                node_position[0] = f32(node_position[0] + zigzag)
            cane_node_positions.append(node_position)
            tip_taper = f32(f32(1) - f32(f32(f32(0.4) * f32(node)) / f32(cane_nodes)))
            base_thickening = f32(f32(1) + f32(f32(0.8) * expf(f32(-path_distance / f32(0.06)))))
            cane_node_radii.append(f32(f32(cane_radius * tip_taper) * base_thickening))
            if node < cane_nodes:
                slope_step = f32(0.005)
                d = cane_position(f32(row_distance + slope_step)) - cane_position(row_distance)
                path_length_per_row_distance = f32(V.magnitude(d) / slope_step)
                row_distance = f32(row_distance + f32(lengths[node] / path_length_per_row_distance))
                path_distance = f32(path_distance + lengths[node])

        uID_cane = addShootFromNodePositions(pa, plantID, uID_stem, attachment_phytomer, cane_node_positions,
                                             cane_node_radii, "grapevine_cane", "grapevine_cane")
        if direction < 0:
            uID_cane_L = uID_cane
        else:
            uID_cane_R = uID_cane

    tree[uID_cane_L].meristem_is_alive = False
    tree[uID_cane_R].meristem_is_alive = False

    for uID_trained in (uID_stem, uID_cane_L, uID_cane_R):
        trained = tree[uID_trained]
        trained.isdormant = False
        bears_shoots = uID_trained != uID_stem
        for phytomer in trained.phytomers:
            phytomer.isdormant = False
            for petiole in phytomer.axillary_vegetative_buds:
                for vbud in petiole:
                    phytomer.setVegetativeBudState(BUD_ACTIVE if bears_shoots else BUD_DEAD, vbud)
                    hangs_below = bool(phytomer.getPetioleAxisVector(f32(0), 0)[2] < 0)
                    vbud.shoot_type_label = "grapevine_shoot_mirrored" if hangs_below else "grapevine_shoot"

    tree[uID_stem].meristem_is_alive = False

    removeShootLeaves(pa, plantID, uID_stem)
    removeShootLeaves(pa, plantID, uID_cane_L)
    removeShootLeaves(pa, plantID, uID_cane_R)

    pa.setPlantPhenologicalThresholds(plantID, 165, -1, -1, 45, 45, 200)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


# ---- creation / callback functions (AS:1943-2005) --------------------------------------------------------

@GrapevinePhytomerCreationFunction.implement
@ports("GrapevinePhytomerCreationFunction")
def _grapevine_phytomer_creation(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
    is_cluster_node = shoot_node_index in (2, 3, 5)
    if not is_cluster_node:
        phytomer.setFloralBudState(BUD_DEAD)
    if phytomer.rank >= 1 and shoot_node_index >= 8:
        phytomer.setVegetativeBudState(BUD_DEAD)
    shoot = phytomer.parent_shoot_ptr
    if shoot.parent_shoot_ID >= 0:
        bearer = shoot.plantarchitecture_ptr.plant_instances[shoot.plantID].shoot_tree[shoot.parent_shoot_ID]
        if bearer.shoot_type_label in ("grapevine_shoot", "grapevine_shoot_mirrored", "grapevine_lateral"):
            phytomer.setFloralBudState(BUD_DEAD)
    if shoot.shoot_type_label == "grapevine_lateral":
        phytomer.setVegetativeBudState(BUD_DEAD)


grapevine_leaf_pulling_nodes = 6
grapevine_leaf_pulling_shoot_age = f32(45)


@GrapevinePhytomerCallbackFunction.implement
@ports("GrapevinePhytomerCallbackFunction")
def _grapevine_phytomer_callback(phytomer):
    shoot = phytomer.parent_shoot_ptr
    if shoot.shoot_type_label not in ("grapevine_shoot", "grapevine_shoot_mirrored"):
        return
    if shoot.parent_shoot_ID >= 0:
        bearer = shoot.plantarchitecture_ptr.plant_instances[shoot.plantID].shoot_tree[shoot.parent_shoot_ID]
        if bearer.shoot_type_label in ("grapevine_shoot", "grapevine_shoot_mirrored"):
            return
    if (int(phytomer.shoot_index[0]) < grapevine_leaf_pulling_nodes and phytomer.hasLeaf()
            and shoot.phytomers[0].age >= grapevine_leaf_pulling_shoot_age):
        phytomer.removeLeaf()


@GrapevineFruitPrototype.implement
@ports("GrapevineFruitPrototype")
def _grapevine_fruit_prototype(context_ptr, subdivisions, build_parameters=None):
    """AS:1774-1935: a seeded, packed grape cluster built along +x, in meters.

    libm choices (compile probe of a TU with Assets.cpp's includes and ``using namespace helios``): the
    unqualified ``sin`` / ``cos`` of a float are the C double functions; ``std::pow(float, int)`` is double;
    ``std::pow(float, float)`` is ``powf``; ``std::cbrt(float)`` is ``cbrtf``; ``std::round(float)`` is ``roundf``.
    The overlap pass is vectorised over berries but accumulates each berry's displacement in the C++ pair order
    (pushes from earlier partners first, then later ones), so every float sum is the C++ one."""
    ctx = context_ptr
    cp = GrapeClusterParameters.fromBuildParameters(build_parameters or {})
    berry_diameter, cluster_berry_count = cp.berry_diameter, cp.cluster_berry_count
    reference_berry_count = 90
    berry_radius_mean = f32(0.05)
    berry_radius_cv = f32(0.07)
    berry_subdivisions = 8
    meters_per_unit = f32(f32(f32(0.5) * berry_diameter) / berry_radius_mean)

    def roundf(x):
        x = float(x)
        return math.copysign(math.floor(abs(x) + 0.5), x)

    lo = int(roundf(f32(f32(cluster_berry_count * f32(75)) / f32(90))))
    hi = int(roundf(f32(f32(cluster_berry_count * f32(105)) / f32(90))))
    berry_count = int(ctx.randu(lo, hi))
    cluster_length = f32(_M["cbrtf"](float(f32(f32(berry_count) / f32(reference_berry_count)))))

    shoulder_radius = f32(f32(0.33) * cluster_length)
    primary_branch_count = int(ctx.randu(13, 17))
    first_branch_position = f32(f32(0.06) * cluster_length)
    last_branch_position = f32(f32(0.85) * cluster_length)

    branch_position, branch_length, branch_berry_weight, branch_direction = [], [], [], []
    has_wing = ctx.randu() < f32(0.5)
    azimuth_offset = ctx.randu(f32(0), TWO_PI_F)
    span = f32(last_branch_position - first_branch_position)
    for branch in range(primary_branch_count):
        pos = f32(first_branch_position + f32(f32(span * f32(f32(branch) + f32(f32(0.5) * ctx.randu())))
                                             / f32(primary_branch_count)))
        branch_position.append(pos)
        reach = f32(f32(shoulder_radius - berry_radius_mean)
                    * f32(f32(1) - f32(f32(pos - first_branch_position) / f32(cluster_length - first_branch_position))))
        reach = f32(reach * f32(f32(1) + f32(f32(0.15) * ctx.randn())))
        reach = max(reach, f32(f32(0.3) * berry_radius_mean))
        insertion_angle = V.deg2rad(ctx.randu(f32(45), f32(65)))
        azimuth = f32(f32(f32(V.deg2rad(f32(137.5)) * f32(branch)) + ctx.randu(f32(-0.3), f32(0.3))) + azimuth_offset)
        length = f32(float(reach) / math.sin(float(insertion_angle)))
        berry_weight = f32(float(f32(reach * reach)) + (float(f32(f32(1.2) * berry_radius_mean))) ** 2)
        if has_wing and branch == 0:
            insertion_angle = V.deg2rad(f32(22))
            length = f32(f32(0.4) * cluster_length)
            berry_weight = f32(berry_weight * f32(2.5))
        branch_length.append(length)
        branch_berry_weight.append(berry_weight)
        ia, az = float(insertion_angle), float(azimuth)
        branch_direction.append(np.array([f32(math.cos(ia)), f32(math.sin(ia) * math.cos(az)),
                                          f32(math.sin(ia) * math.sin(az))], f32))

    branch_berries_total = berry_count - 1
    total_weight = f32(0)
    for w in branch_berry_weight:
        total_weight = f32(total_weight + w)
    branch_berry_count, remainders, assigned = [], [], 0
    for branch in range(primary_branch_count):
        share = f32(f32(branch_berry_weight[branch] / total_weight) * f32(branch_berries_total))
        fl = f32(np.floor(share))
        branch_berry_count.append(int(fl))
        remainders.append((f32(share - fl), branch))
        assigned += int(fl)
    remainders.sort(reverse=True)                       # std::sort(rbegin, rend) of pair<float, int>
    for i in range(branch_berries_total - assigned):
        branch_berry_count[remainders[i][1]] += 1

    anchors = [np.array([f32(last_branch_position + berry_radius_mean), 0, 0], f32)]
    for branch in range(primary_branch_count):
        d = branch_direction[branch]
        bl = branch_length[branch]
        for _ in range(branch_berry_count[branch]):
            along = f32(f32(1) - f32(f32(0.75) * f32(_M["powf"](float(ctx.randu()), 1.5))))
            sz = ctx.randn()                             # make_vec3(randn(), randn(), randn()): right to left
            sy = ctx.randn()
            sx = ctx.randn()
            scatter = np.array([sx, sy, sz], f32)
            scatter = scatter - d * V.dot(scatter, d)
            anchor = (np.array([branch_position[branch], 0, 0], f32) + d * f32(bl * along)) \
                + scatter * f32(f32(f32(0.35) * bl) * along)
            anchors.append(anchor.astype(f32))

    radii = np.empty(berry_count, f32)
    for b in range(berry_count):
        radii[b] = f32(berry_radius_mean * V.clamp(f32(f32(1) + f32(berry_radius_cv * ctx.randn())), f32(0.8),
                                                    f32(1.2)))

    A = np.array(anchors, f32)
    C = A.copy()
    n = berry_count
    keep = f32(f32(1) - f32(0.012))
    shift_x = f32(f32(f32(0.3) * f32(0.012)) * berry_radius_mean)
    spring = f32(0.02)
    rsum = radii[:, None] + radii[None, :]
    upper = np.triu(np.ones((n, n), bool), 1)
    x_hi = (cluster_length - f32(0.5) * radii).astype(f32)
    for iteration in range(400):
        if iteration < 300:
            C[:, 1] = C[:, 1] * keep
            C[:, 2] = C[:, 2] * keep
            C[:, 0] = C[:, 0] + shift_x
            C = C + (A - C) * spring
        for _pass in range(2):
            S = C[:, None, :] - C[None, :, :]                                     # separation(i, j) = c_i - c_j
            dist = np.sqrt((S[..., 0] * S[..., 0] + S[..., 1] * S[..., 1]) + S[..., 2] * S[..., 2])
            overlap = rsum - dist
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (f32(0.5) * overlap) / dist
                P = S * s[..., None]
            mask = upper & (overlap > 0) & (dist > 0)
            disp = np.zeros((n, 3), f32)
            for i in range(n - 1):                     # disp_k -= push(i, k), i ascending, for every k > i
                m = mask[i, i + 1:]
                if m.any():
                    sub = disp[i + 1:]
                    sub[m] = sub[m] - P[i, i + 1:][m]
            for j in range(1, n):                      # then disp_k += push(k, j), j ascending, for every k < j
                m = mask[:j, j]
                if m.any():
                    sub = disp[:j]
                    sub[m] = sub[m] + P[:j, j][m]
            C = C + disp
        cx = C[:, 0]
        cx = np.where(cx < radii, radii, np.where(cx > x_hi, x_hi, cx))
        C[:, 0] = cx

    color = make_RGBcolor(0.053, 0.076, 0.098)
    UUIDs = []
    for b in range(n):
        UUIDs += _add_sphere(ctx, berry_subdivisions, meters_per_unit * C[b], f32(meters_per_unit * radii[b]), color)
    ctx.set_primitive_data(UUIDs, "object_label", "fruit")
    return ctx.add_polymesh_object(UUIDs)


def _add_sphere(ctx, ndivs, center, radius, color):
    """``Context::addSphere(Ndivs, center, radius, color)`` (Context.cpp:1815-1853): triangles, no object. Not in
    the core yet (it has only addSphereObject, whose vertices differ: radius applied after sphere2cart, top cap
    wound the other way)."""
    center = np.asarray(center, f32)
    radius = f32(radius)
    dtheta = f32(PI_F / f32(ndivs))
    dphi = f32(f32(f32(2) * PI_F) / f32(ndivs))
    half = f32(f32(0.5) * PI_F)

    def p(el, az):
        el, az = f32(el), f32(az)
        rc = f32(radius * cosf(el))
        return center + np.array([f32(rc * sinf(az)), f32(rc * cosf(az)), f32(radius * sinf(el))], f32)

    uu = []
    for j in range(ndivs):
        uu.append(ctx.add_triangle(p(-half, 0), p(f32(-half + dtheta), f32(f32(j) * dphi)),
                                   p(f32(-half + dtheta), f32(f32(j + 1) * dphi)), color))
    for j in range(ndivs):
        v0 = p(half, 0)
        v1 = p(f32(half - dtheta), f32(f32(j) * dphi))
        v2 = p(f32(half - dtheta), f32(f32(j + 1) * dphi))
        uu.append(ctx.add_triangle(v2, v1, v0, color))
    for j in range(ndivs):
        for i in range(1, ndivs - 1):
            a, b = f32(-half + f32(f32(i) * dtheta)), f32(-half + f32(f32(i + 1) * dtheta))
            v0 = p(a, f32(f32(j) * dphi))
            v1 = p(b, f32(f32(j) * dphi))
            v2 = p(b, f32(f32(j + 1) * dphi))
            v3 = p(a, f32(f32(j + 1) * dphi))
            uu.append(ctx.add_triangle(v0, v1, v2, color))
            uu.append(ctx.add_triangle(v0, v2, v3, color))
    return uu


__all__ = ["initializeGrapevineVSPShoots", "buildGrapevineVSP", "grapevineTrunkPath", "BoundedWander",
           "GrapeClusterParameters", "GrapevineFruitPrototype", "GrapevinePhytomerCreationFunction",
           "GrapevinePhytomerCallbackFunction", "register_build_parameters"]

#: True while GrapevineFruitPrototype only raises (the parity gate for cluster-bearing ages reads it)
FRUIT_PROTOTYPE_IS_STUB = False
