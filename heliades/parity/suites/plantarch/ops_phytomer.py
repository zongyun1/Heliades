# SPDX-License-Identifier: LGPL-2.1-or-later
"""Phytomer-geometry operations for both backends (W3-B).

* ``pa_age0_xml_values``: the per-phytomer values ``writePlantStructureXML`` writes for an age-0 plant, as the
  numbers printed by ``std::ostream << float`` (Helios: parse the XML; Heliades: format the ported state the way
  the writer computes it, InputOutput.cpp:519-1066);
* ``pa_age0_full_values``: the same state at full float32 precision through the PyHelios getters
  (``getInternodeNodePositions``, ``getPetioleVertices``, ``getPetioleRadii``, ``getLeafBasePosition``,
  ``getPetioleAxisVector``, ``getInternodeAxisVector``);
* ``pa_tube_prims``: the primitives labelled ``shoot`` / ``petiole`` (``object_label`` primitive data): UUID,
  vertices and color, sorted by UUID;
* ``pa_tube_materials``: the material label of each of those primitives (``nameOrganMaterials``);
* ``pa_leaf_frames``: the leaflet objects' transformation matrices. On Heliades the 35 hidden blades are
  stand-ins (``stub_generic_leaf_prototype``) whose object frames are copied from the oracle's hidden
  prototypes of the same seed, so the comparison isolates the placement chain (copy, scale, orient, translate,
  ``scaleLeafPrototypeScale``) from the blade geometry (W3-A).
"""
from __future__ import annotations

import numpy as np

from ...ops import op

f32 = np.float32


def _cout(v) -> float:
    """``std::ostream << float`` (precision 6), read back."""
    return float("%g" % float(f32(v)))


def _xml_floats(text):
    return [float(x) for x in text.replace(";", " ").split()]


# ---- XML values -----------------------------------------------------------------------------------

@op("pa_age0_xml_values", "helios")
def _(B):
    root = B.plantarch_xml()
    shoot = root.find("plant_instance/shoot")
    ph = shoot.find("phytomer")
    it = ph.find("internode")
    pet = it.find("petiole")
    out = _xml_floats(shoot.find("base_rotation").text)
    out += _xml_floats(ph.find("vegetative_bud_states").text)
    for tag in ("internode_length", "internode_radius", "internode_pitch", "internode_phyllotactic_angle",
                "internode_length_max", "internode_curvature_step", "internode_length_segments"):
        out += _xml_floats(it.find(tag).text)
    for tag in ("curvature_perturbations", "yaw_perturbations", "azimuthal_restoring_angles", "gravitropic_angles"):
        out += _xml_floats(it.find(tag).text)
    for tag in ("petiole_length", "petiole_radius", "petiole_pitch", "petiole_curvature", "current_leaf_scale_factor",
                "current_petiole_scale_factor", "petiole_taper", "petiole_length_segments",
                "petiole_radial_subdivisions", "petiole_axis_initial", "petiole_rotation_axis", "leaflet_scale",
                "leaflet_offset"):
        out += _xml_floats(pet.find(tag).text)
    for lf in pet.findall("leaf"):
        for tag in ("leaf_scale", "leaf_pitch", "leaf_yaw", "leaf_roll", "leaf_prototype"):
            out += _xml_floats(lf.find(tag).text)
    for fb in pet.findall("floral_bud"):
        for tag in ("bud_state", "parent_index", "bud_index", "is_terminal"):
            out += _xml_floats(fb.find(tag).text)
    return out


@op("pa_age0_xml_values", "heliades")
def _(B):
    """What InputOutput.cpp:519-1066 prints for shoot 0 / phytomer 0 / petiole 0."""
    from ....plantarchitecture.params import rad2deg
    shoot = B.pa.plant_instances[B.pid].shoot_tree[0]
    ph = shoot.phytomers[0]
    r = shoot.base_rotation
    out = [_cout(rad2deg(r.pitch)), _cout(rad2deg(r.yaw)), _cout(rad2deg(r.roll))]
    out += [float(int(b.state)) for petiole in ph.axillary_vegetative_buds for b in petiole]
    nseg = len(ph.internode_curvature_perturbations)
    out += [_cout(ph.getInternodeLength()), _cout(ph.getInternodeRadius()), _cout(rad2deg(ph.internode_pitch)),
            _cout(rad2deg(ph.internode_phyllotactic_angle)), _cout(ph.internode_length_max),
            _cout(ph.internode_curvature_arc_length_step), float(nseg)]
    for arr in (ph.internode_curvature_perturbations, ph.internode_yaw_perturbations,
                ph.internode_azimuthal_restoring_angles, ph.internode_gravitropic_angles):
        out += [_cout(v) for v in arr]
    p = 0
    pp = ph.phytomer_parameters
    out += [_cout(ph.petiole_length[p]), _cout(ph.petiole_radii[p][0]), _cout(rad2deg(ph.petiole_pitch[p])),
            _cout(ph.petiole_curvature[p]), _cout(ph.current_leaf_scale_factor[p]),
            _cout(ph.current_petiole_scale_factor[p]), _cout(ph.petiole_taper[p]),
            float(pp.petiole.length_segments), float(pp.petiole.radial_subdivisions)]
    out += [_cout(x) for x in ph.petiole_axis_initial[p]] + [_cout(x) for x in ph.petiole_rotation_axis[p]]
    sizes = ph.leaf_size_max[p]
    if len(ph.leaf_rotation[p]) <= 1 or not sizes:
        out.append(1.0)
    else:
        tip_ind = (len(ph.leaf_rotation[p]) - 1) // 2
        lateral_ind = min(max(0, tip_ind - 1), len(sizes) - 1)
        mx = max(sizes)
        out.append(_cout(f32(sizes[lateral_ind] / mx) if mx > 0 else f32(1)))
    out.append(_cout(pp.leaf.leaflet_offset.val()))
    for leaf, rot in enumerate(ph.leaf_rotation[p]):
        out += [_cout(f32(sizes[leaf] * ph.current_leaf_scale_factor[p])), _cout(rad2deg(rot.pitch)),
                _cout(rad2deg(rot.yaw)), _cout(rad2deg(rot.roll)), float(ph.leaf_prototype_index[p][leaf])]
    for fb in ph.floral_buds[p]:
        out += [float(int(fb.state)), float(fb.parent_index), float(fb.bud_index), float(fb.isterminal)]
    return out


# ---- full-precision state through the PyHelios getters ----------------------------------------------

@op("pa_age0_full_values", "helios")
def _(B):
    pa, pid = B.pa, B.pid
    out = []
    for v in pa.getInternodeNodePositions(pid, 0, 0):
        out += [v.x, v.y, v.z]
    for v in pa.getPetioleVertices(pid, 0, 0, 0):
        out += [v.x, v.y, v.z]
    out += list(pa.getPetioleRadii(pid, 0, 0, 0))
    for leaf in range(7):
        v = pa.getLeafBasePosition(pid, 0, 0, 0, leaf)
        out += [v.x, v.y, v.z]
    for frac in (0.0, 0.5, 1.0):
        v = pa.getPetioleAxisVector(pid, 0, 0, frac, 0)
        out += [v.x, v.y, v.z]
        v = pa.getInternodeAxisVector(pid, 0, 0, frac)
        out += [v.x, v.y, v.z]
    out.append(pa.getPetioleLength(pid, 0, 0, 0))
    return [float(f32(x)) for x in out]


@op("pa_age0_full_values", "heliades")
def _(B):
    ph = B.pa.plant_instances[B.pid].shoot_tree[0].phytomers[0]
    out = []
    for v in ph.getInternodeNodePositions():
        out += list(v)
    for v in ph.petiole_vertices[0]:
        out += list(v)
    out += list(ph.petiole_radii[0])
    for b in ph.leaf_bases[0]:
        out += list(b)
    for frac in (0.0, 0.5, 1.0):
        out += list(ph.getPetioleAxisVector(f32(frac), 0))
        out += list(ph.getInternodeAxisVector(f32(frac)))
    out.append(ph.petiole_length[0])
    return [float(f32(x)) for x in out]


# ---- tubes ----------------------------------------------------------------------------------------

_TUBE_LABELS = ("shoot", "petiole")


@op("pa_tube_prims", "helios")
def _(B):
    ctx = B.ctx
    uu = sorted(int(u) for u in ctx.getAllUUIDs())
    out = []
    for u in uu:
        if not ctx.doesPrimitiveDataExist(u, "object_label"):
            continue
        label = ctx.getPrimitiveData(u, "object_label")
        if label not in _TUBE_LABELS:
            continue
        flat, _off = ctx.getPrimitiveVertices([u])
        c = ctx.getPrimitiveColor(u)
        out += [float(u), float(_TUBE_LABELS.index(label))] + [float(x) for x in np.asarray(flat, f32).reshape(-1)]
        out += [float(c.r), float(c.g), float(c.b)]
    return out


@op("pa_tube_prims", "heliades")
def _(B):
    s = B.scene
    out = []
    for u in s.all_uuids():
        if not s.does_primitive_data_exist(u, "object_label"):
            continue
        label = s.get_primitive_data(u, "object_label")
        if label not in _TUBE_LABELS:
            continue
        c = s.primitives[u].color
        out += [float(u), float(_TUBE_LABELS.index(label))] + [float(x) for x in s.primitive_vertices(u).reshape(-1)]
        out += [float(c[0]), float(c[1]), float(c[2])]
    return out


@op("pa_tube_materials", "helios")
def _(B):
    ctx = B.ctx
    out = []
    for u in sorted(int(u) for u in ctx.getAllUUIDs()):
        if ctx.doesPrimitiveDataExist(u, "object_label") and ctx.getPrimitiveData(u, "object_label") in _TUBE_LABELS:
            out.append(ctx.getPrimitiveMaterialLabel(u))
    return out


@op("pa_tube_materials", "heliades")
def _(B):
    s = B.scene
    return [s.get_primitive_material_label(u) for u in s.all_uuids()
            if s.does_primitive_data_exist(u, "object_label") and s.get_primitive_data(u, "object_label") in _TUBE_LABELS]


# ---- leaflet frames (placement chain against oracle prototype frames) -------------------------------

def _oracle_prototype_frames(seed):
    """Translation of each hidden leaf prototype's object frame in an oracle build of the same seed, in
    creation order (objects between the petiole tube and the first leaflet copy)."""
    from pyhelios import Context, PlantArchitecture
    from pyhelios.types import vec3
    ctx = Context()
    ctx.seedRandomGenerator(int(seed))
    pa = PlantArchitecture(ctx)
    try:
        pa.disableMessages()
    except Exception:
        pass
    pa.loadPlantModelFromLibrary("tomato")
    pid = pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), 0.0)
    first_leaf = min(min(p) for p in pa.getPhytomerLeafObjectIDs(pid, 0, 0))
    frames = []
    for oid in range(1, first_leaf):
        if ctx.doesObjectExist(oid) and ctx.isObjectHidden(oid):
            T = np.asarray(ctx.getObjectTransformationMatrix(oid), f32).reshape(16)
            frames.append((T[3], T[7], T[11]))
    return frames


@op("pa_build_with_oracle_prototype_frames", "heliades")
def _(B, seed):
    """Build at age 0 with stand-in blades placed in the oracle prototypes' frames."""
    from ....plantarchitecture.phytomer import stub_generic_leaf_prototype
    frames = iter(_oracle_prototype_frames(seed))
    B.pa.leaf_prototype_function_override = (
        lambda ctx, proto, idx: stub_generic_leaf_prototype(ctx, proto, idx, first_vertex=next(frames)))
    return B.plantarch_build((0.0, 0.0, 0.0), 0.0)


@op("pa_build_with_oracle_prototype_frames", "helios")
def _(B, seed):
    return B.plantarch_build((0.0, 0.0, 0.0), 0.0)


@op("pa_leaf_frames", "helios")
def _(B):
    out = []
    for petiole in B.pa.getPhytomerLeafObjectIDs(B.pid, 0, 0):
        for oid in petiole:
            out.append(float(oid))
            out += [float(x) for x in np.asarray(B.ctx.getObjectTransformationMatrix(oid), f32).reshape(16)]
    return out


@op("pa_leaf_frames", "heliades")
def _(B):
    out = []
    ph = B.pa.plant_instances[B.pid].shoot_tree[0].phytomers[0]
    for petiole in ph.leaf_objIDs:
        for oid in petiole:
            out.append(float(oid))
            out += [float(x) for x in B.scene.object_transform(oid).reshape(16)]
    return out
