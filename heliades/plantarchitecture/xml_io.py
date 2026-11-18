# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/InputOutput.cpp:519-1066 (Copyright (C) 2016-2026 Brian Bailey).
"""``writePlantStructureXML`` for the plants Heliades can grow (W3-C).

Text-identical to Helios for the tomato path: every number goes through :func:`cout`, which is what
``std::ostream << float`` prints (default precision 6, ``%g``), integers and booleans as C++ streams them. It is
the shoot / phytomer / bud state oracle of ``parity.suites.plantarch.cases_growth``. Branches for features Heliades does not
grow (prescribed shoots and petioles, reshaped or posed leaves, shed petioles, peduncle swings, build
parameters, leaf-angle tracking) raise ``NotImplementedError`` when they would write something.
"""
from __future__ import annotations

import math

import numpy as np

from ..migration_registry import ports
from .params import rad2deg

f32 = np.float32


def cout(v) -> str:
    """``std::ostream << float`` (or ``<< double`` for a Python float), default flags."""
    x = float(v)
    if math.isnan(x):
        return "-nan" if math.copysign(1.0, x) < 0 else "nan"
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return "%g" % x


def _i(v) -> str:
    return str(int(v))


@ports("PlantArchitecture::writePlantStructureXML")
def plant_structure_xml(pa, plantID: int) -> str:
    if plantID not in pa.plant_instances:
        raise ValueError(f"ERROR (PlantArchitecture::writePlantStructureXML): Plant ID {plantID} does not exist.")
    inst = pa.plant_instances[plantID]
    o: list[str] = []
    w = o.append
    w('<?xml version="1.0" encoding="UTF-8"?>')
    w("<helios>")
    w(f'\t<plant_instance ID="{plantID}">')
    bp = inst.base_position
    w(f"\t\t<base_position> {cout(bp[0])} {cout(bp[1])} {cout(bp[2])} </base_position>")
    w(f"\t\t<plant_age> {cout(inst.current_age)} </plant_age>")
    w(f"\t\t<time_since_dormancy> {cout(inst.time_since_dormancy)} </time_since_dormancy>")
    for tag in ("dd_to_dormancy_break", "dd_to_flower_initiation", "dd_to_flower_opening", "dd_to_fruit_set",
                "dd_to_fruit_maturity", "dd_to_dormancy", "max_leaf_lifespan"):
        w(f"\t\t<{tag}> {cout(getattr(inst, tag))} </{tag}>")
    w(f"\t\t<is_evergreen> {_i(inst.is_evergreen)} </is_evergreen>")
    if inst.minimum_leaf_texture_stage > 0:
        w(f"\t\t<minimum_leaf_texture_stage> {_i(inst.minimum_leaf_texture_stage)} </minimum_leaf_texture_stage>")
    w(f"\t\t<max_age> {cout(inst.max_age)} </max_age>")
    if inst.build_parameters or inst.organ_build_parameters:
        # W5-D (IO:576-600): structure parameters at max_digits10 (9) significant digits
        if any(getattr(d, "type", "constant") != "constant" for d in inst.organ_build_parameters.values()):
            raise NotImplementedError("writing organ build-parameter distributions is not ported")
        w(f"\t\t<plant_model> {inst.plant_name} </plant_model>")
        for name in sorted(inst.build_parameters):
            value = "%.9g" % float(f32(inst.build_parameters[name]))
            w(f'\t\t<build_parameter name="{name}"> {value} </build_parameter>')
    if getattr(inst, "leaf_angle_tracking_enabled", False) and inst.leaf_angle_tracker.track_elevation:   # W5-B (IO:612)
        t = inst.leaf_angle_tracker
        w(f"\t\t<leaf_inclination_Beta_distribution> {cout(t.Beta_mu_inclination)} {cout(t.Beta_nu_inclination)} "
          "</leaf_inclination_Beta_distribution>")

    for shoot in inst.shoot_tree:
        if shoot.isPruned():
            continue
        w(f'\t\t<shoot ID="{shoot.ID}">')
        w(f"\t\t\t<shoot_type_label> {shoot.shoot_type_label} </shoot_type_label>")
        w(f"\t\t\t<parent_shoot_ID> {_i(shoot.parent_shoot_ID)} </parent_shoot_ID>")
        w(f"\t\t\t<parent_node_index> {_i(shoot.parent_node_index)} </parent_node_index>")
        w(f"\t\t\t<parent_petiole_index> {_i(shoot.parent_petiole_index)} </parent_petiole_index>")
        w(f"\t\t\t<is_dormant> {_i(shoot.isdormant)} </is_dormant>")
        w(f"\t\t\t<dormancy_cycles> {_i(shoot.dormancy_cycles)} </dormancy_cycles>")
        w(f"\t\t\t<meristem_is_alive> {_i(shoot.meristem_is_alive)} </meristem_is_alive>")
        w(f"\t\t\t<phyllochron_counter> {cout(shoot.phyllochron_counter)} </phyllochron_counter>")
        w(f"\t\t\t<nodes_this_season> {_i(shoot.nodes_this_season)} </nodes_this_season>")
        r = shoot.base_rotation
        w(f"\t\t\t<base_rotation> {cout(rad2deg(r.pitch))} {cout(rad2deg(r.yaw))} {cout(rad2deg(r.roll))} </base_rotation>")
        if shoot.is_axis_continuation:
            w("\t\t\t<is_axis_continuation> 1 </is_axis_continuation>")
        gs = shoot.growth_state
        if gs.growth_type_label:
            w(f"\t\t\t<growth_shoot_type_label> {gs.growth_type_label} </growth_shoot_type_label>")
        if gs.max_nodes_growth_cap >= 0:
            w(f"\t\t\t<max_nodes_growth_cap> {_i(gs.max_nodes_growth_cap)} </max_nodes_growth_cap>")
        if gs.max_nodes_per_season_growth_cap >= 0:
            w(f"\t\t\t<max_nodes_per_season_growth_cap> {_i(gs.max_nodes_per_season_growth_cap)} "
              "</max_nodes_per_season_growth_cap>")
        if gs.internode_radius_cap > 0:
            w(f"\t\t\t<internode_radius_cap> {cout(gs.internode_radius_cap)} </internode_radius_cap>")
        if gs.geometry_is_prescribed:
            # W5-D (IO:673-700): one node per phytomer (the base node, then each phytomer's last node)
            nodes = [shoot.shoot_internode_vertices[0][0]] + [row[-1] for row in shoot.shoot_internode_vertices]
            radii = [shoot.shoot_internode_radii[0][0]] + [row[-1] for row in shoot.shoot_internode_radii]
            w("\t\t\t<prescribed_internode_nodes>" + "".join(f" {cout(n[0])} {cout(n[1])} {cout(n[2])}" for n in nodes)
              + " </prescribed_internode_nodes>")
            w("\t\t\t<prescribed_internode_radii>" + "".join(f" {cout(r)}" for r in radii)
              + " </prescribed_internode_radii>")

        for phytomer_index, ph in enumerate(shoot.phytomers):
            _write_phytomer(w, shoot, ph, phytomer_index)
        w("\t\t</shoot>")
    w("\t</plant_instance>")
    w("</helios>")
    return "\n".join(o) + "\n"


def _write_floral_buds(w, ph, petiole):
    if petiole >= len(ph.floral_buds):
        return
    pp = ph.phytomer_parameters
    for bud, fbud in enumerate(ph.floral_buds[petiole]):
        w("\t\t\t\t\t\t<floral_bud>")
        w(f"\t\t\t\t\t\t\t<bud_state>{_i(fbud.state)}</bud_state>")
        w(f"\t\t\t\t\t\t\t<parent_index>{_i(fbud.parent_index)}</parent_index>")
        w(f"\t\t\t\t\t\t\t<bud_index>{_i(fbud.bud_index)}</bud_index>")
        w(f"\t\t\t\t\t\t\t<is_terminal>{_i(fbud.isterminal)}</is_terminal>")
        w(f"\t\t\t\t\t\t\t<current_fruit_scale_factor>{cout(fbud.current_fruit_scale_factor)}</current_fruit_scale_factor>")
        w(f"\t\t\t\t\t\t\t<bud_time_counter>{cout(fbud.time_counter)}</bud_time_counter>")
        w(f"\t\t\t\t\t\t\t<bud_age>{cout(fbud.age)}</bud_age>")
        w(f"\t\t\t\t\t\t\t<previous_fruit_scale_factor>{cout(fbud.previous_fruit_scale_factor)}</previous_fruit_scale_factor>")
        w("\t\t\t\t\t\t\t<peduncle>")
        if petiole < len(ph.peduncle_length) and bud < len(ph.peduncle_length[petiole]):
            vals = (ph.peduncle_length[petiole][bud], ph.peduncle_radius[petiole][bud], ph.peduncle_pitch[petiole][bud],
                    ph.peduncle_curvature[petiole][bud], ph.peduncle_roll[petiole][bud], ph.peduncle_yaw[petiole][bud])
        else:
            ped = pp.peduncle
            vals = (ped.length.val(), ped.radius.val(), ped.pitch.val(), ped.curvature.val(), ped.roll.val(),
                    ped.yaw.val())
        for tag, v in zip(("length", "radius", "pitch", "curvature", "roll", "yaw"), vals):
            w(f"\t\t\t\t\t\t\t\t<{tag}>{cout(v)}</{tag}>")
        w("\t\t\t\t\t\t\t</peduncle>")
        if f32(fbud.contact_swing_angle) > 0:
            raise NotImplementedError("writing peduncle swings is not ported")
        w("\t\t\t\t\t\t\t<inflorescence>")
        w(f"\t\t\t\t\t\t\t\t<flower_offset>{cout(pp.inflorescence.flower_offset.val())}</flower_offset>")
        for i in range(len(fbud.inflorescence_bases)):                    # InputOutput.cpp:760-784 (W3-D)
            w("\t\t\t\t\t\t\t\t<flower>")
            if i < len(fbud.inflorescence_rotation):
                r = fbud.inflorescence_rotation[i]
                w(f"\t\t\t\t\t\t\t\t\t<flower_pitch>{cout(rad2deg(r.pitch))}</flower_pitch>")
                w(f"\t\t\t\t\t\t\t\t\t<flower_yaw>{cout(rad2deg(r.yaw))}</flower_yaw>")
                w(f"\t\t\t\t\t\t\t\t\t<flower_roll>{cout(rad2deg(r.roll))}</flower_roll>")
                w(f"\t\t\t\t\t\t\t\t\t<flower_azimuth>{cout(rad2deg(r.azimuth))}</flower_azimuth>")
                if f32(r.gravity_fraction) > 0:
                    w(f"\t\t\t\t\t\t\t\t\t<flower_gravity_fraction>{cout(r.gravity_fraction)}</flower_gravity_fraction>")
                if f32(r.contact_tilt_angle) > 0:
                    raise NotImplementedError("writing fruit contact tilts is not ported")
            if i < len(fbud.inflorescence_base_scales):
                w(f"\t\t\t\t\t\t\t\t\t<flower_base_scale>{cout(fbud.inflorescence_base_scales[i])}</flower_base_scale>")
            w("\t\t\t\t\t\t\t\t</flower>")
        w("\t\t\t\t\t\t\t</inflorescence>")
        w("\t\t\t\t\t\t</floral_bud>")


def _write_phytomer(w, shoot, ph, phytomer_index):
    pp = ph.phytomer_parameters
    w("\t\t\t<phytomer>")
    w(f"\t\t\t\t<phytomer_age>{cout(ph.age)}</phytomer_age>")
    w(f"\t\t\t\t<phytomer_is_dormant>{_i(ph.isdormant)}</phytomer_is_dormant>")
    states = "".join("".join(f" {_i(b.state)}" for b in petiole) + " ;" for petiole in ph.axillary_vegetative_buds)
    w(f"\t\t\t\t<vegetative_bud_states>{states} </vegetative_bud_states>")
    w(f"\t\t\t\t<leaf_flexibility>{cout(ph.leaf_flexibility)}</leaf_flexibility>")
    w("\t\t\t\t<internode>")
    w(f"\t\t\t\t\t<internode_length>{cout(ph.getInternodeLength())}</internode_length>")
    w(f"\t\t\t\t\t<internode_radius>{cout(ph.getInternodeRadius())}</internode_radius>")
    w(f"\t\t\t\t\t<internode_pitch>{cout(rad2deg(ph.internode_pitch))}</internode_pitch>")
    w(f"\t\t\t\t\t<internode_phyllotactic_angle>{cout(rad2deg(ph.internode_phyllotactic_angle))}"
      "</internode_phyllotactic_angle>")
    w(f"\t\t\t\t\t<internode_length_max>{cout(ph.internode_length_max)}</internode_length_max>")
    w(f"\t\t\t\t\t<internode_curvature_step>{cout(ph.internode_curvature_arc_length_step)}</internode_curvature_step>")
    nseg = len(ph.internode_curvature_perturbations)
    if nseg > 0:
        w(f"\t\t\t\t\t<internode_length_segments>{nseg}</internode_length_segments>")
    elif phytomer_index < len(shoot.shoot_internode_vertices):
        n = len(shoot.shoot_internode_vertices[phytomer_index]) - (1 if phytomer_index == 0 else 0)
        w(f"\t\t\t\t\t<internode_length_segments>{n}</internode_length_segments>")
    for tag, arr in (("curvature_perturbations", ph.internode_curvature_perturbations),
                     ("yaw_perturbations", ph.internode_yaw_perturbations),
                     ("azimuthal_restoring_angles", ph.internode_azimuthal_restoring_angles),
                     ("gravitropic_angles", ph.internode_gravitropic_angles)):
        if arr:
            w(f"\t\t\t\t\t<{tag}>" + ";".join(cout(v) for v in arr) + f"</{tag}>")

    for petiole in range(len(ph.petiole_length)):
        w("\t\t\t\t\t<petiole>")
        w(f"\t\t\t\t\t\t<petiole_length>{cout(ph.petiole_length[petiole])}</petiole_length>")
        w(f"\t\t\t\t\t\t<petiole_radius>{cout(ph.petiole_radii[petiole][0])}</petiole_radius>")
        w(f"\t\t\t\t\t\t<petiole_pitch>{cout(rad2deg(ph.petiole_pitch[petiole]))}</petiole_pitch>")
        w(f"\t\t\t\t\t\t<petiole_curvature>{cout(ph.petiole_curvature[petiole])}</petiole_curvature>")
        if ph.leaf_is_sessile:
            w("\t\t\t\t\t\t<leaf_is_sessile>1</leaf_is_sessile>")
        w(f"\t\t\t\t\t\t<current_leaf_scale_factor>{cout(ph.current_leaf_scale_factor[petiole])}</current_leaf_scale_factor>")
        w(f"\t\t\t\t\t\t<current_petiole_scale_factor>{cout(ph.current_petiole_scale_factor[petiole])}"
          "</current_petiole_scale_factor>")
        w(f"\t\t\t\t\t\t<petiole_taper>{cout(ph.petiole_taper[petiole])}</petiole_taper>")
        w(f"\t\t\t\t\t\t<petiole_length_segments>{_i(pp.petiole.length_segments)}</petiole_length_segments>")
        w(f"\t\t\t\t\t\t<petiole_radial_subdivisions>{_i(pp.petiole.radial_subdivisions)}</petiole_radial_subdivisions>")
        a, b = ph.petiole_axis_initial[petiole], ph.petiole_rotation_axis[petiole]
        w(f"\t\t\t\t\t\t<petiole_axis_initial>{cout(a[0])} {cout(a[1])} {cout(a[2])}</petiole_axis_initial>")
        w(f"\t\t\t\t\t\t<petiole_rotation_axis>{cout(b[0])} {cout(b[1])} {cout(b[2])}</petiole_rotation_axis>")
        if petiole < len(ph.petiole_path_prescribed) and ph.petiole_path_prescribed[petiole]:
            raise NotImplementedError("writing prescribed petioles is not ported")
        sizes = ph.leaf_size_max[petiole]
        if len(ph.leaf_rotation[petiole]) <= 1 or not sizes:
            w(f"\t\t\t\t\t\t<leaflet_scale>{cout(1.0)}</leaflet_scale>")
        else:
            tip_ind = int(math.floor(float(f32(f32(len(ph.leaf_rotation[petiole]) - 1) / f32(2)))))
            lateral_ind = min(max(0, tip_ind - 1), len(sizes) - 1)
            mx = max(sizes)
            scale = f32(sizes[lateral_ind] / mx) if mx > 0 else f32(1)
            w(f"\t\t\t\t\t\t<leaflet_scale>{cout(scale)}</leaflet_scale>")
        w(f"\t\t\t\t\t\t<leaflet_offset>{cout(pp.leaf.leaflet_offset.val())}</leaflet_offset>")
        for leaf, rot in enumerate(ph.leaf_rotation[petiole]):
            w("\t\t\t\t\t\t<leaf>")
            w(f"\t\t\t\t\t\t\t<leaf_scale>{cout(f32(sizes[leaf] * ph.current_leaf_scale_factor[petiole]))}</leaf_scale>")
            w(f"\t\t\t\t\t\t\t<leaf_pitch>{cout(rad2deg(rot.pitch))}</leaf_pitch>")
            w(f"\t\t\t\t\t\t\t<leaf_yaw>{cout(rad2deg(rot.yaw))}</leaf_yaw>")
            w(f"\t\t\t\t\t\t\t<leaf_roll>{cout(rad2deg(rot.roll))}</leaf_roll>")
            if petiole < len(ph.leaflet_angle_offsets) and leaf < len(ph.leaflet_angle_offsets[petiole]):
                raise NotImplementedError("writing leaflet angle offsets is not ported")
            if petiole < len(ph.leaf_prototype_index) and leaf < len(ph.leaf_prototype_index[petiole]):
                w(f"\t\t\t\t\t\t\t<leaf_prototype>{_i(ph.leaf_prototype_index[petiole][leaf])}</leaf_prototype>")
            if petiole < len(ph.leaf_objIDs) and leaf < len(ph.leaf_objIDs[petiole]):
                st = ph.leaf_shape_state.get(ph.leaf_objIDs[petiole][leaf])
                if st is not None and getattr(st, "reshaped", False):
                    raise NotImplementedError("writing reshaped leaves is not ported")
            if (petiole < len(ph.leaf_pose_prescribed) and leaf < len(ph.leaf_pose_prescribed[petiole])
                    and ph.leaf_pose_prescribed[petiole][leaf]):
                # W5-B (IO:1002-1011): a leaf steered by setLeafNormal (no stalk base of its own)
                lb = ph.leaf_bases[petiole][leaf]
                w("\t\t\t\t\t\t\t<leaf_prescribed>1</leaf_prescribed>")
                w(f"\t\t\t\t\t\t\t<leaf_base>{cout(lb[0])} {cout(lb[1])} {cout(lb[2])}</leaf_base>")
                if ph.leaf_objIDs[petiole][leaf] in getattr(ph, "leaf_stalk_base_placed", {}):
                    raise NotImplementedError("writing placed leaf stalk bases is not ported")
            w("\t\t\t\t\t\t</leaf>")
        _write_floral_buds(w, ph, petiole)
        w("\t\t\t\t\t</petiole>")

    if not ph.petiole_length and ph.petiole_vertices:
        # removeLeaf() keeps the centerlines and floral buds: one <shed_petiole> per petiole index (W4-A)
        for petiole, centerline in enumerate(ph.petiole_vertices):
            w("\t\t\t\t\t<shed_petiole>")
            if len(centerline):
                offs = [np.asarray(n, f32) - np.asarray(centerline[0], f32) for n in centerline]
                w("\t\t\t\t\t\t<centerline_offsets>" + ";".join(f"{cout(d[0])} {cout(d[1])} {cout(d[2])}" for d in offs)
                  + "</centerline_offsets>")
            _write_floral_buds(w, ph, petiole)
            w("\t\t\t\t\t</shed_petiole>")
    entry_count = len(ph.petiole_vertices) if not ph.petiole_length else len(ph.petiole_length)
    for entry in range(entry_count, len(ph.floral_buds)):
        w("\t\t\t\t\t<internode_floral_buds>")
        _write_floral_buds(w, ph, entry)
        w("\t\t\t\t\t</internode_floral_buds>")
    w("\t\t\t\t</internode>")
    w("\t\t\t</phytomer>")


__all__ = ["plant_structure_xml", "cout"]
