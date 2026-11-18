# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:111-130, 433-435, 523-524, 4033-4457
# (Copyright (C) 2016-2026 Brian Bailey).
"""Grapevine on a Wye (open gable) trellis, quadrilateral cordons (``grapevine_Wye``), W5-C.

Reuses the VSP machinery of ``grapevine.py`` (W4-D): ``BoundedWander``, ``grapevineTrunkPath``, the prescribed
shoots (``addShootFromNodePositions``), per-plant attraction points, the creation / callback functions and the
grape cluster prototype. New here:

* ``initializeGrapevineWyeShoots`` (PL:4033-4219): five shoot types (``grapevine_trunk`` (trunk and arms),
  ``grapevine_cordon``, ``grapevine_shoot``, ``grapevine_shoot_mirrored``, ``grapevine_lateral``);
* ``buildGrapevineWye`` (PL:4221-4457): a prescribed trunk to a head below the cordon wires, two prescribed arms
  forking from a hidden node inside the head, four prescribed cordons (two per arm, one each way along the wire)
  forking from a hidden node at the top of each arm, and the gable wires as attraction points;
* ``appendWoodDome`` (PL:120-130), which closes each arm with three dome nodes;
* the build parameters ``trunk_height`` [0.5, 2], ``cordon_spacing`` [0.2, 2], ``vine_spacing`` [0.5, 5],
  ``catch_wire_height`` [0.5, 4] plus the grape cluster parameters, and the
  ``catch_wire_height <= trunk_height`` rejection.

Float maths as in ``grapevine.py``: ``std::sin`` / ``std::exp`` of a float are glibc ``sinf`` / ``expf``,
``std::round(float)`` is ``roundf``; arguments are evaluated right to left (GCC).
"""
from __future__ import annotations

import math

import numpy as np

from ...core import vec as V
from ...migration_registry import ports
from ..architecture import context_generator
from ..params import BUD_ACTIVE, BUD_DEAD, LeafPrototype, PhytomerParameters, ShootParameters, make_RGBcolor
from ..random_parameter import PlantArchitectureError
from .build_parameters import BuildParameterDeclaration, _std_to_string, getBuildParameterValue, registerPlantModelBuildParameters
from .grapevine import (PI_F, TWO_PI_F, BoundedWander, GrapeClusterParameters, GrapevineFruitPrototype,
                        GrapevinePhytomerCallbackFunction, GrapevinePhytomerCreationFunction, _add_component,
                        _linspace3, _validate_grape_cluster_parameters, expf, grapevine_vine_spacing,
                        grapevineTrunkPath, sinf, cosf)

f32 = np.float32

# ---- build parameters (PL:433-435, 523-524) ----------------------------------------------------------------
grapevine_Wye_trunk_height = BuildParameterDeclaration("trunk_height", 0.5, 2.0, False, "height of the cordon wires in meters")
grapevine_Wye_cordon_spacing = BuildParameterDeclaration("cordon_spacing", 0.2, 2.0, False,
                                                         "distance between the two cordon wires across the row in meters")
grapevine_Wye_catch_wire_height = BuildParameterDeclaration("catch_wire_height", 0.5, 4.0, False,
                                                            "height of the top gable wires in meters")


def register_build_parameters(pa):
    """PL:523-524."""
    registerPlantModelBuildParameters(pa, "grapevine_Wye", [grapevine_Wye_trunk_height, grapevine_Wye_cordon_spacing,
                                                            grapevine_vine_spacing, grapevine_Wye_catch_wire_height],
                                      GrapeClusterParameters.buildParameterDeclarations(),
                                      _validate_grape_cluster_parameters)


def _roundf(x) -> np.float32:
    """``std::round(float)``: half away from zero (exact in double for a float argument)."""
    x = float(f32(x))
    return f32(math.copysign(math.floor(abs(x) + 0.5), x))


# ---- shoot types (PL:4033-4219) ----------------------------------------------------------------------------

@ports("PlantArchitecture::initializeGrapevineWyeShoots")
def initializeGrapevineWyeShoots(pa):
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
    pp.internode.phyllotactic_angle.uniformDistribution(160, 200)
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
    pp.leaf.prototype_scale = 0.126
    pp.leaf.prototype = leaf_prototype

    pp.peduncle.length = 0.04
    pp.peduncle.radius = 0.005
    pp.peduncle.color = make_RGBcolor(0.13, 0.125, 0.03)
    pp.peduncle.pitch.uniformDistribution(80, 100)
    pp.peduncle.yaw = 180

    pp.inflorescence.flowers_per_peduncle = 1
    pp.inflorescence.pitch = 0
    pp.inflorescence.flower_prototype_scale = 0.04
    pp.inflorescence.fruit_prototype_function = GrapevineFruitPrototype
    pp.inflorescence.fruit_prototype_scale = 1
    pp.inflorescence.unique_prototypes = 5
    pp.inflorescence.fruit_gravity_factor_fraction = 0.9

    pp.phytomer_creation_function = GrapevinePhytomerCreationFunction
    pp.phytomer_callback_function = GrapevinePhytomerCallbackFunction

    # ---- Shoot Parameters ---- //
    main = ShootParameters(gen)
    main.phytomer_parameters = pp
    main.vegetative_bud_break_probability_min = 0.25
    main.vegetative_bud_break_probability_max = 0.25
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
    main.max_nodes.uniformDistribution(21, 25)
    main.base_roll.uniformDistribution(90 - 15, 90 + 15)
    main.base_yaw.uniformDistribution(-40, 40)

    cordon = main.copy()
    cordon.phytomer_parameters.internode.image_texture = "GrapeBark.jpg"
    cordon.phytomer_parameters.internode.pitch = 0
    cordon.phytomer_parameters.internode.radial_subdivisions = 15
    cordon.phytomer_parameters.internode.max_floral_buds_per_petiole = 0
    cordon.phytomer_parameters.internode.phyllotactic_angle = 0
    cordon.girth_area_factor = 0
    cordon.max_nodes = 40
    cordon.tortuosity = 0
    cordon.gravitropic_curvature = 0
    cordon.base_yaw = 0
    cordon.defineChildShootTypes(["grapevine_shoot"], [1.0])

    trunk = main.copy()
    trunk.phytomer_parameters.internode.image_texture = "GrapeBark.jpg"
    trunk.phytomer_parameters.internode.pitch = 0
    trunk.phytomer_parameters.internode.phyllotactic_angle = 0
    trunk.phytomer_parameters.internode.radius_initial = 0.04
    trunk.phytomer_parameters.internode.radial_subdivisions = 25
    trunk.phytomer_parameters.internode.max_floral_buds_per_petiole = 0
    trunk.phyllochron_min = 2.5
    trunk.insertion_angle_tip = 90
    trunk.girth_area_factor = 0
    trunk.max_nodes = 60
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
    mirrored.base_yaw.uniformDistribution(f32(f32(180) - f32(40)), f32(f32(180) + f32(40)))
    mirrored.defineChildShootTypes(["grapevine_lateral"], [1.0])

    pa.defineShootType("grapevine_trunk", trunk)
    pa.defineShootType("grapevine_cordon", cordon)
    pa.defineShootType("grapevine_shoot", main)
    pa.defineShootType("grapevine_shoot_mirrored", mirrored)
    pa.defineShootType("grapevine_lateral", lateral)


# ---- builder (PL:4221-4457) -------------------------------------------------------------------------------

_DOME_ANGLES_DEGREES = (f32(30), f32(60), f32(85))


@ports("appendWoodDome")
def appendWoodDome(node_positions, node_radii):
    """PL:120-130: three nodes past the last one, along its heading, radius falling as a quarter circle."""
    end_direction = V.normalize(node_positions[-1] - node_positions[-2])
    end_position = node_positions[-1].copy()
    end_radius = f32(node_radii[-1])
    for deg in _DOME_ANGLES_DEGREES:
        dome_angle = V.deg2rad(deg)
        node_positions.append(end_position + end_direction * f32(end_radius * sinf(dome_angle)))
        node_radii.append(f32(end_radius * cosf(dome_angle)))


@ports("PlantArchitecture::buildGrapevineWye")
def buildGrapevineWye(pa, base_position):
    from ..shoot import addShootFromNodePositions, removeShootLeaves, setPlantAttractionPoints
    if not pa.shoot_types:
        initializeGrapevineWyeShoots(pa)
    ctx = pa.context_ptr
    base_position = np.asarray(base_position, f32).reshape(3).copy()
    bp = pa.current_build_parameters

    trunk_height = getBuildParameterValue(bp, grapevine_Wye_trunk_height, f32(1.4))
    cordon_spacing = getBuildParameterValue(bp, grapevine_Wye_cordon_spacing, f32(0.6))
    vine_spacing = getBuildParameterValue(bp, grapevine_vine_spacing, f32(1.8))
    catch_wire_height = getBuildParameterValue(bp, grapevine_Wye_catch_wire_height, f32(trunk_height + f32(0.6)))
    if catch_wire_height <= trunk_height:
        raise PlantArchitectureError(
            f"ERROR (PlantArchitecture::buildGrapevineWye): build parameter 'catch_wire_height' "
            f"({_std_to_string(catch_wire_height)} m) must be greater than 'trunk_height' ({_std_to_string(trunk_height)} "
            "m). The gable wires that carry the canopy rise from the cordon wires, whose height is 'trunk_height'.")

    cordon_internode_length = f32(0.08)
    cordon_total_length = f32(f32(0.5) * vine_spacing)
    cordon_nodes = max(1, int(f32(cordon_total_length / cordon_internode_length)))

    # trellis wires (attraction points)
    gable_rise = f32(catch_wire_height - trunk_height)
    gable_spread_per_rise = f32(0.6)
    gable_wire_levels = max(1, int(_roundf(f32(gable_rise / f32(0.2)))))
    wire_samples = max(2, int(f32(vine_spacing / f32(0.06))))

    trellis_points = []
    half = f32(f32(0.5) * vine_spacing)
    for level in range(gable_wire_levels + 1):
        height_above_cordon = f32(f32(gable_rise * f32(level)) / f32(gable_wire_levels))
        wire_offset = f32(f32(f32(0.5) * cordon_spacing) + f32(gable_spread_per_rise * height_above_cordon))
        z = f32(trunk_height + height_above_cordon)
        for side in (f32(-1), f32(1)):
            x = f32(side * wire_offset)
            trellis_points.append(_linspace3((x, -half, z), (x, half, z), wire_samples))
    trellis_points = [[p + base_position for p in wire] for wire in trellis_points]

    plantID = pa.addPlantInstance(base_position, 0)
    setPlantAttractionPoints(pa, plantID, [p for wire in trellis_points for p in wire], 70.0, 0.35, 0.15)

    cordon_wire_height = f32(base_position[2] + trunk_height)

    # ---- Trunk ---- //
    arm_rise = f32(min(f32(f32(0.5) * cordon_spacing), f32(f32(0.35) * trunk_height)) * ctx.randu(f32(0.9), f32(1.1)))
    head_height = f32(trunk_height - arm_rise)

    nominal = pa.shoot_types["grapevine_trunk"].phytomer_parameters.internode.radius_initial.val()
    trunk_nodes, trunk_node_positions, trunk_node_radii = grapevineTrunkPath(ctx, base_position, head_height, nominal)

    head_radius = f32(trunk_node_radii[trunk_nodes])
    trunk_node_positions.append(trunk_node_positions[trunk_nodes] - np.array([0, 0, f32(f32(0.4) * head_radius)], f32))
    trunk_node_radii.append(f32(0.002))
    uID_stem = addShootFromNodePositions(pa, plantID, -1, 0, trunk_node_positions, trunk_node_radii, "grapevine_trunk")

    tree = pa.plant_instances[plantID].shoot_tree

    # ---- Arms and cordons ---- //
    uID_arms, uID_cordons = [], []
    for side in (f32(-1), f32(1)):
        arm_attachment_phytomer = len(trunk_node_positions) - 2
        arm_base = trunk_node_positions[-1].copy()
        half_spacing_side = f32(f32(side * f32(0.5)) * cordon_spacing)
        arm_top = np.array([f32(base_position[0] + half_spacing_side),
                            f32(arm_base[1] + ctx.randu(f32(-0.02), f32(0.02))), cordon_wire_height], f32)
        arm_length = V.magnitude(arm_top - arm_base)
        arm_nodes = max(4, int(np.ceil(f32(arm_length / f32(0.04)))))

        arm_bow = f32(ctx.randu(f32(0.05), f32(0.15)) * arm_length)
        arm_radius = f32(f32(f32(0.55) * head_radius) * ctx.randu(f32(0.9), f32(1.1)))

        arm_node_positions, arm_node_radii = [], []
        delta = arm_top - arm_base
        side_half_bow = f32(f32(side * f32(0.5)) * arm_bow)
        for node in range(arm_nodes + 1):
            along = f32(f32(node) / f32(arm_nodes))
            node_position = arm_base + delta * along
            s = sinf(f32(PI_F * along))
            node_position[2] = f32(node_position[2] - f32(arm_bow * s))
            node_position[0] = f32(node_position[0] + f32(side_half_bow * s))
            arm_node_positions.append(node_position)
            e = expf(f32(f32(f32(-along) * arm_length) / f32(0.05)))
            arm_node_radii.append(f32(arm_radius * f32(f32(f32(1) + f32(f32(0.5) * e)) - f32(f32(0.2) * along))))
        appendWoodDome(arm_node_positions, arm_node_radii)
        arm_top_direction = V.normalize(arm_node_positions[arm_nodes] - arm_node_positions[arm_nodes - 1])
        arm_node_positions.append(arm_node_positions[arm_nodes]
                                  - arm_top_direction * f32(f32(0.4) * arm_node_radii[arm_nodes]))
        arm_node_radii.append(f32(0.002))

        uID_arm = addShootFromNodePositions(pa, plantID, uID_stem, arm_attachment_phytomer, arm_node_positions,
                                            arm_node_radii, "grapevine_trunk")
        uID_arms.append(uID_arm)

        # -- Cordons -- //
        for direction in (f32(-1), f32(1)):
            attachment_phytomer = len(arm_node_positions) - 2
            attachment_node = np.asarray(tree[uID_arm].shoot_internode_vertices[attachment_phytomer][-1], f32).copy()

            rise = f32(cordon_wire_height - attachment_node[2])
            inset = f32(side * f32(f32(base_position[0] + half_spacing_side) - attachment_node[0]))
            bend_length = min(f32(0.25), max(f32(0.08), f32(max(f32(abs(rise)), f32(abs(inset)))
                                                            * ctx.randu(f32(2), f32(3)))))

            wander_x, wander_z = BoundedWander(), BoundedWander()
            _add_component(ctx, wander_x, (0.008, 0.018), (0.4, 0.8), (0.0, TWO_PI_F))
            _add_component(ctx, wander_x, (0.003, 0.006), (0.15, 0.3), (0.0, TWO_PI_F))
            _add_component(ctx, wander_z, (0.006, 0.015), (0.5, 0.9), (0.0, TWO_PI_F))
            _add_component(ctx, wander_z, (0.003, 0.006), (0.18, 0.3), (0.0, TWO_PI_F))

            cordon_reach = f32(f32(cordon_total_length * ctx.randu(f32(0.93), f32(1)))
                               - f32(direction * f32(attachment_node[1] - base_position[1])))
            cordon_length = max(cordon_internode_length, cordon_reach)

            lengths = []
            unscaled = f32(0)
            for _ in range(cordon_nodes):
                lengths.append(ctx.randu(f32(0.8), f32(1.2)))
                unscaled = f32(unscaled + lengths[-1])
            ratio = f32(cordon_length / unscaled)
            lengths = [f32(L * ratio) for L in lengths]

            neg_side_inset = f32(f32(-side) * inset)
            wire_x = f32(base_position[0] + half_spacing_side)

            def cordon_position(row_distance, _bl=bend_length, _wx=wander_x, _wz=wander_z, _rise=rise,
                                _nsi=neg_side_inset, _an=attachment_node, _dir=direction, _wire_x=wire_x):
                row_distance = f32(row_distance)
                bend_fraction = min(f32(row_distance / _bl), f32(1))
                on_wire = f32(f32(bend_fraction * bend_fraction) * f32(f32(3) - f32(f32(2) * bend_fraction)))
                one_ow = f32(f32(1) - on_wire)
                x = f32(f32(_nsi * one_ow) + f32(on_wire * _wx.evaluate(row_distance)))
                z = f32(f32(f32(-_rise) * one_ow) + f32(on_wire * _wz.evaluate(row_distance)))
                return np.array([f32(_wire_x + x), f32(_an[1] + f32(_dir * row_distance)),
                                 f32(cordon_wire_height + z)], f32)

            cordon_node_positions, cordon_node_radii = [], []
            arm_top_radius = f32(arm_node_radii[arm_nodes])
            cordon_radius = f32(f32(f32(0.75) * arm_top_radius) * ctx.randu(f32(0.92), f32(1.08)))
            junction_flare = max(f32(0), f32(f32(f32(f32(0.95) * arm_top_radius) / cordon_radius) - f32(1)))
            row_distance = f32(0)
            path_distance = f32(0)
            for node in range(cordon_nodes + 1):
                cordon_node_positions.append(cordon_position(row_distance))
                tip_taper = f32(f32(1) - f32(f32(f32(0.2) * f32(node)) / f32(cordon_nodes)))
                flare = f32(f32(1) + f32(junction_flare * expf(f32(-path_distance / f32(0.05)))))
                cordon_node_radii.append(f32(f32(cordon_radius * tip_taper) * flare))
                if node < cordon_nodes:
                    slope_step = f32(0.005)
                    d = cordon_position(f32(row_distance + slope_step)) - cordon_position(row_distance)
                    path_length_per_row_distance = f32(V.magnitude(d) / slope_step)
                    row_distance = f32(row_distance + f32(lengths[node] / path_length_per_row_distance))
                    path_distance = f32(path_distance + lengths[node])
            uID_cordons.append(addShootFromNodePositions(pa, plantID, uID_arm, attachment_phytomer, cordon_node_positions,
                                                         cordon_node_radii, "grapevine_cordon"))

    # wake the trained wood, set its buds, stop it extending
    for uID in [uID_stem] + uID_arms + uID_cordons:
        trained = tree[uID]
        trained.isdormant = False
        trained.meristem_is_alive = False
        bears_shoots = trained.shoot_type_label == "grapevine_cordon"
        for phytomer in trained.phytomers:
            phytomer.isdormant = False
            for petiole in phytomer.axillary_vegetative_buds:
                for vbud in petiole:
                    phytomer.setVegetativeBudState(BUD_ACTIVE if bears_shoots else BUD_DEAD, vbud)
                    below = bool(phytomer.getPetioleAxisVector(f32(0), 0)[2] < 0)
                    vbud.shoot_type_label = "grapevine_shoot_mirrored" if below else "grapevine_shoot"
        removeShootLeaves(pa, plantID, uID)

    pa.setPlantPhenologicalThresholds(plantID, 165, -1, -1, 45, 45, 200)
    pa.plant_instances[plantID].max_age = f32(365)
    return plantID


__all__ = ["initializeGrapevineWyeShoots", "buildGrapevineWye", "appendWoodDome", "register_build_parameters"]
