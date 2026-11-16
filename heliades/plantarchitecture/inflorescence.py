# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp (Copyright (C) 2016-2026 Brian Bailey):
#   makeClosedFruitSurfacesOneSided                          PA:54-179
#   recordFruitClassMap                                      PA:790-796
#   Phytomer::getFloralBudStorageIndex                       PA:1295-1305
#   Phytomer::setFloralBudState(BudState, FloralBud&)        PA:1491-1547
#   Phytomer::createInflorescenceGeometry                    PA:3623-3731
#   PlantArchitecture::ensureInflorescencePrototypesInitialized  PA:3733-3773
#   PlantArchitecture::sampleOrganBuildParameters            PA:3775-3781
#   Phytomer::getReconciledPeduncleRadius                    PA:3783-3802
#   Phytomer::getPeduncleNodeColors                          PA:3804-3818
#   Phytomer::bendPeduncleAxis                               PA:3820-3845
#   Phytomer::peduncleShouldMatchCulm                        PA:3847-3858
#   Phytomer::updatePeduncleRadii                            PA:3860-3932
#   Phytomer::updateInflorescence                            PA:3934-4197
#   Phytomer::setInflorescenceScaleFraction                  PA:6744-6759
"""Inflorescence geometry: peduncle tube, flower / fruit placement and the floral-bud geometry side (W3-D).

Every function takes the objects the C++ member function would see: ``phytomer`` is a
:class:`~heliades.plantarchitecture.phytomer.Phytomer` (or any object with the same members: see
:func:`required_phytomer_members`), ``fbud`` one of its :class:`~heliades.plantarchitecture.params.FloralBud`
entries (identity matters: :func:`get_floral_bud_storage_index` finds it with ``is``).

Entry points for the growth code (W3-C), matching the C++ call sites:

* :func:`set_floral_bud_state` -- ``Phytomer::setFloralBudState(BudState, FloralBud&)`` (the whole function,
  including the DORMANT/ACTIVE shortcut; the carbon model raises ``NotImplementedError``);
* :func:`update_inflorescence` -- ``Phytomer::updateInflorescence(FloralBud&)`` (``appendPhytomerToShoot``'s
  terminal buds);
* :func:`set_inflorescence_scale_fraction` -- fruit growth in ``advanceTime``;
* :func:`update_peduncle_radii` -- the end of ``incrementPhytomerInternodeGirth``.

Float / double, as in the C++ (``using namespace helios`` only, so unqualified ``fabs`` is the C double one;
``std::atan2`` / ``std::acos`` / ``std::abs`` on floats are the float overloads = glibc ``atan2f`` / ``acosf``):

* ``fabs(fruit - float(n-1)/float(p))``: float argument, exact in double, narrowed back: identical to float;
* ``floor(0.5*float(n-1)) + 1`` in the equal-spacing branch is double;
* ``std::abs(angle) > 1e-6``: float magnitude compared in double (exact).

RNG: tomato draws ``internode.phyllotactic_angle.resample()`` once per flower (6 per rebuild);
``randu(0, unique_prototypes - 1)`` is ``randu(0, 0)`` (no draw); every other resample is of a constant.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import vec as V
from ..migration_registry import ports
from .params import BUD_ACTIVE, BUD_DEAD, BUD_DORMANT, BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING, AxisRotation
from .random_parameter import PlantArchitectureError
from .tube_utils import PI_F, clamp_offset, get_axis_vector, interpolate_tube, interpolate_tube_float

f32 = np.float32
NULLORIGIN = np.zeros(3, f32)
_TINY = 1e-6          # double literal in `std::abs(x) > 1e-6`


def _v(x) -> np.ndarray:
    return np.asarray(x, f32).reshape(3).copy()


def _rot(p, axis, theta) -> np.ndarray:
    """``rotatePointAboutLine(p, nullorigin, axis, theta)``."""
    return V.rotate_point_about_line(p, NULLORIGIN, axis, theta)


def _atan2f(y, x) -> np.float32:
    from .phytomer import atan2f
    return atan2f(y, x)


def _name_organ_materials(pa, objID, name):
    from .phytomer import name_organ_materials
    name_organ_materials(pa, objID, name)


def required_phytomer_members():
    """What these functions read from / write to ``phytomer`` (for stand-ins in tests and parity ops)."""
    return ("plantarchitecture_ptr", "context_ptr", "plantID", "phytomer_parameters", "parent_shoot_ptr",
            "shoot_index", "rank", "petiole_vertices", "floral_buds", "inflorescence_bending_axis",
            "build_context_geometry_peduncle", "peduncle_vertices", "peduncle_radii", "peduncle_length",
            "peduncle_radius", "peduncle_pitch", "peduncle_curvature", "peduncle_roll", "peduncle_yaw",
            "getInternodeNodePositions()", "getPetioleAxisVector(frac, petiole)")


# ---- per-PlantArchitecture caches (H:6700-6704) -----------------------------------------------------

def _caches(pa):
    """``unique_{closed,open}_flower_prototype_objIDs`` / ``unique_fruit_prototype_objIDs``: maps keyed by
    ``(function pointer, organ build parameters)``. Created on first use (W3-C/W3-E may move them into
    ``shoot_min.init_plant_state``; the attribute names are the C++ member names)."""
    for name in ("unique_closed_flower_prototype_objIDs", "unique_open_flower_prototype_objIDs",
                 "unique_fruit_prototype_objIDs"):
        if not hasattr(pa, name):
            setattr(pa, name, {})
    return pa


def _bind_library_hooks():
    """Bind the ported flower / fruit prototype functions (``hooks.TomatoFlowerPrototype`` / ``...Fruit...``)."""
    from .library import tomato_inflorescence  # noqa: F401


def _build_parameters_key(build_parameters) -> tuple:
    """``std::map<std::string, BuildParameterDistribution>`` as a hashable, ordered key."""
    if not build_parameters:
        return ()
    return tuple(sorted((k, repr(v)) for k, v in dict(build_parameters).items()))


def _prototype_key(fn, build_parameters):
    return (fn, _build_parameters_key(build_parameters))


@ports("PlantArchitecture::sampleOrganBuildParameters")
def sample_organ_build_parameters(pa, organ_build_parameters) -> dict:
    """``std::map`` order; ``distribution.sample(*generator)`` per entry (none for tomato)."""
    out = {}
    for name in sorted(organ_build_parameters or {}):
        dist = organ_build_parameters[name]
        sample = getattr(dist, "sample", None)
        if sample is None:
            raise NotImplementedError("BuildParameterDistribution.sample is not ported (no library model on the "
                                      "ported path declares organ build parameters)")
        out[name] = f32(sample(pa.generator))
    return out


# ---- fruit prototype post-processing -----------------------------------------------------------------

@ports("makeClosedFruitSurfacesOneSided")
def make_closed_fruit_surfaces_one_sided(ctx, objID: int, fruit_material_name: str):
    """PA:54-179. Weld vertices within ``max(1e-9f, 1e-4f*|bbox|)``, split into connected parts, and give the
    primitives of every closed, consistently oriented, outward-facing part a one-sided material copy."""
    UUIDs = list(ctx.object_primitive_uuids(objID))
    if not UUIDs:
        return
    object_min, object_max = ctx.object_bounding_box(objID)
    tolerance = max(f32(1e-9), f32(f32(1e-4) * V.magnitude(_v(object_max) - _v(object_min))))
    tolerance = f32(tolerance)

    welded: list = []
    vertex_cells: dict = {}

    def weld(vertex):
        cell = (int(math.floor(float(f32(vertex[0] / tolerance)))), int(math.floor(float(f32(vertex[1] / tolerance)))),
                int(math.floor(float(f32(vertex[2] / tolerance)))))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    lst = vertex_cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
                    if lst is None:
                        continue
                    for index in lst:
                        if V.magnitude(welded[index] - vertex) <= tolerance:
                            return index
        welded.append(_v(vertex))
        vertex_cells.setdefault(cell, []).append(len(welded) - 1)
        return len(welded) - 1

    faces = []
    for u in UUIDs:
        faces.append([weld(_v(v)) for v in ctx.primitive_vertices(u)])

    parent = list(range(len(faces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    face_of_vertex: dict = {}
    for i, f in enumerate(faces):
        for vertex in f:
            if vertex in face_of_vertex:
                parent[root(i)] = root(face_of_vertex[vertex])
            else:
                face_of_vertex[vertex] = i
    parts: dict = {}
    for i in range(len(faces)):
        parts.setdefault(root(i), []).append(i)

    for part_root in sorted(parts):                       # std::map<int, ...> order
        part = parts[part_root]
        directed_edges: dict = {}
        volume = 0.0
        for face in part:
            f = faces[face]
            for k in range(len(f)):
                e = (f[k], f[(k + 1) % len(f)])
                directed_edges[e] = directed_edges.get(e, 0) + 1
            for k in range(1, len(f) - 1):
                volume += float(V.dot(welded[f[0]], V.cross(welded[f[k]], welded[f[k + 1]]))) / 6.0
        closed_and_oriented = True
        for edge in sorted(directed_edges):               # std::map order (only the outcome matters)
            n = directed_edges[edge]
            rev = directed_edges.get((edge[1], edge[0]))
            if n != 1 or rev is None or rev != 1:
                closed_and_oriented = False
                break
        if not closed_and_oriented or volume <= 0.0:
            continue
        for face in part:
            UUID = UUIDs[face]
            material = ctx.get_primitive_material_label(UUID)
            if ctx.get_material_twosided_flag(material) == 0:
                continue
            base_label = (fruit_material_name if material[:2] == "__" else material) + "_closed"

            def same_appearance(label):
                # normal-map members are not modelled by the core material table (equal by construction)
                return (ctx.get_material_twosided_flag(label) == 0
                        and np.array_equal(np.asarray(ctx.get_material_color(label), f32),
                                           np.asarray(ctx.get_material_color(material), f32))
                        and ctx.get_material_texture(label) == ctx.get_material_texture(material)
                        and ctx.is_material_texture_color_overridden(label)
                        == ctx.is_material_texture_color_overridden(material))

            closed_material = base_label
            suffix = 1
            while ctx.does_material_exist(closed_material) and not same_appearance(closed_material):
                closed_material = f"{base_label}_{suffix}"
                suffix += 1
            if not ctx.does_material_exist(closed_material):
                ctx.add_material(closed_material)
                ctx.set_material_color(closed_material, ctx.get_material_color(material))
                texture = ctx.get_material_texture(material)
                if texture:
                    ctx.set_material_texture(closed_material, texture)
                ctx.set_material_texture_color_override(closed_material,
                                                        ctx.is_material_texture_color_overridden(material))
                ctx.set_material_twosided_flag(closed_material, 0)
            ctx.assign_material_to_primitive(UUID, closed_material)


@ports("recordFruitClassMap")
def record_fruit_class_map(ctx, objID_fruit: int, params):
    if not params.inflorescence.fruit_class_map_file:
        return
    raise NotImplementedError("fruit spectral class maps (recordSpectralClassMap) are not ported")


# ---- prototypes ----------------------------------------------------------------------------------------

@ports("PlantArchitecture::ensureInflorescencePrototypesInitialized")
def ensure_inflorescence_prototypes_initialized(pa, params, plantID: int):
    """PA:3733-3773: on first use for a (function, build parameters) key, build ``unique_prototypes`` closed
    flowers, then open flowers, then fruit; name their materials, make the fruit's closed surfaces one-sided,
    and hide them. Shared by every plant of this PlantArchitecture."""
    _caches(pa)
    _bind_library_hooks()
    ctx = pa.context_ptr
    infl = params.inflorescence
    if infl.unique_prototypes <= 0:
        return
    inst = pa.plant_instances[plantID]
    plant_name = inst.plant_name
    build_parameters = inst.organ_build_parameters
    flower_key = _prototype_key(infl.flower_prototype_function, build_parameters)
    if infl.flower_prototype_function is not None and flower_key not in pa.unique_closed_flower_prototype_objIDs:
        prototypes = pa.unique_closed_flower_prototype_objIDs.setdefault(flower_key, [])
        prototypes[:] = [0] * int(infl.unique_prototypes)
        for i in range(len(prototypes)):
            prototypes[i] = infl.flower_prototype_function(ctx, 1, False,
                                                           sample_organ_build_parameters(pa, build_parameters))
            _name_organ_materials(pa, prototypes[i], plant_name + "_flower_closed")
            ctx.hide_object(prototypes[i])
    if infl.flower_prototype_function is not None and flower_key not in pa.unique_open_flower_prototype_objIDs:
        prototypes = pa.unique_open_flower_prototype_objIDs.setdefault(flower_key, [])
        prototypes[:] = [0] * int(infl.unique_prototypes)
        for i in range(len(prototypes)):
            prototypes[i] = infl.flower_prototype_function(ctx, 1, True,
                                                           sample_organ_build_parameters(pa, build_parameters))
            _name_organ_materials(pa, prototypes[i], plant_name + "_flower_open")
            ctx.hide_object(prototypes[i])
    fruit_key = _prototype_key(infl.fruit_prototype_function, build_parameters)
    if infl.fruit_prototype_function is not None and fruit_key not in pa.unique_fruit_prototype_objIDs:
        prototypes = pa.unique_fruit_prototype_objIDs.setdefault(fruit_key, [])
        prototypes[:] = [0] * int(infl.unique_prototypes)
        for i in range(len(prototypes)):
            prototypes[i] = infl.fruit_prototype_function(ctx, 1, sample_organ_build_parameters(pa, build_parameters))
            _name_organ_materials(pa, prototypes[i], plant_name + "_fruit")
            make_closed_fruit_surfaces_one_sided(ctx, prototypes[i], plant_name + "_fruit")
            record_fruit_class_map(ctx, prototypes[i], params)
            ctx.hide_object(prototypes[i])


# ---- floral bud bookkeeping ------------------------------------------------------------------------

@ports("Phytomer::getFloralBudStorageIndex")
def get_floral_bud_storage_index(phytomer, fbud) -> tuple[int, int]:
    for entry, buds in enumerate(phytomer.floral_buds):
        for position, b in enumerate(buds):
            if b is fbud:
                return entry, position
    raise PlantArchitectureError("ERROR (Phytomer::getFloralBudStorageIndex): The floral bud given is not one of "
                                 "this phytomer's floral buds.")


# ---- peduncle helpers ------------------------------------------------------------------------------

@ports("Phytomer::peduncleShouldMatchCulm")
def peduncle_should_match_culm(phytomer) -> bool:
    if phytomer.parent_shoot_ptr.shoot_parameters.phytomer_parameters.internode.image_texture:
        return False
    return phytomer.shoot_index[0] + 1 == phytomer.shoot_index[2]


def _internode_radius_at(phytomer, stem_fraction) -> np.float32:
    """``Phytomer::getInternodeRadius(float)`` (PA:1375): interpolateTube over this phytomer's radii."""
    s = phytomer.parent_shoot_ptr
    with np.errstate(divide="ignore"):   # one-node radius lists: 1.f/0 = inf in C++ too, loop body never runs
        return interpolate_tube_float(s.shoot_internode_radii[phytomer.shoot_index[0]], stem_fraction)


@ports("Phytomer::getReconciledPeduncleRadius")
def get_reconciled_peduncle_radius(phytomer) -> np.float32:
    culm_tip_radius = _internode_radius_at(phytomer, f32(1))
    if culm_tip_radius <= 0 or not peduncle_should_match_culm(phytomer):
        return phytomer.phytomer_parameters.peduncle.radius.val()
    return f32(f32(0.9) * culm_tip_radius)


@ports("Phytomer::getPeduncleNodeColors")
def get_peduncle_node_colors(phytomer, node_count: int) -> list:
    pp = phytomer.phytomer_parameters
    pc = np.asarray(pp.peduncle.color, f32)[:3]
    colors = [pc.copy() for _ in range(node_count)]
    if node_count < 3 or not peduncle_should_match_culm(phytomer):
        return colors
    cc = np.asarray(pp.internode.color, f32)[:3]
    for node in range(node_count - 1):
        blend = f32(f32(node) / f32(node_count - 2))
        colors[node] = np.array([f32(cc[k] + f32(f32(pc[k] - cc[k]) * blend)) for k in range(3)], f32)
    return colors


@ports("Phytomer::bendPeduncleAxis")
def bend_peduncle_axis(peduncle_axis, curvature_degrees_per_meter, segment_length) -> np.ndarray:
    peduncle_axis = _v(peduncle_axis)
    curvature_degrees_per_meter = f32(curvature_degrees_per_meter)
    if curvature_degrees_per_meter == 0:
        return peduncle_axis
    target = _v((0, 0, 1)) if curvature_degrees_per_meter > 0 else _v((0, 0, -1))
    horizontal_bending_axis = V.cross(peduncle_axis, (0, 0, 1))
    axis_magnitude = V.magnitude(horizontal_bending_axis)
    if axis_magnitude < f32(1e-6):
        return target if V.dot(peduncle_axis, target) > 0 else peduncle_axis
    horizontal_bending_axis = V.div(horizontal_bending_axis, axis_magnitude)
    theta_curvature = V.deg2rad(f32(curvature_degrees_per_meter * f32(segment_length)))
    c = V.dot(peduncle_axis, target)
    c = f32(-1) if c < -1 else (f32(1) if c > 1 else c)          # std::clamp(x, -1.f, 1.f)
    theta_from_target = V.acosf(c)                                 # std::acos(float) = acosf
    if abs(theta_curvature) >= theta_from_target:
        return target
    bent_axis = _rot(peduncle_axis, horizontal_bending_axis, theta_curvature)
    return V.normalize(bent_axis)


@ports("Phytomer::updatePeduncleRadii")
def update_peduncle_radii(phytomer):
    """PA:3860-3932: match the peduncle to the culm (``setTubeRadii``, path dependent) and re-anchor it to the
    bud base (``setTubeNodes``) when that moved."""
    ctx = phytomer.context_ptr
    if not any(fb.peduncle_objIDs for petiole in phytomer.floral_buds for fb in petiole):
        return
    radius = get_reconciled_peduncle_radius(phytomer)
    for petiole_idx, buds in enumerate(phytomer.floral_buds):
        for fbud in buds:
            for objID in fbud.peduncle_objIDs:
                if not ctx.does_object_exist(objID):
                    continue
                existing_radii = list(ctx.tube_object_node_radii(objID))
                if not existing_radii:
                    continue
                if any(abs(f32(f32(r) - radius)) > f32(1e-9) for r in existing_radii):
                    ctx.set_tube_radii(objID, [radius] * len(existing_radii))
                nodes = [_v(n) for n in ctx.tube_object_nodes(objID)]
                if nodes:
                    node_base = nodes[0].copy()
                    base = _v(fbud.base_position)
                    anchor_shift = base - node_base
                    if V.magnitude(anchor_shift) > f32(1e-6):
                        nodes = [base + (n - node_base) for n in nodes]
                        ctx.set_tube_nodes(objID, nodes)
            if petiole_idx < len(phytomer.peduncle_radius) and fbud.bud_index < len(phytomer.peduncle_radius[petiole_idx]):
                phytomer.peduncle_radius[petiole_idx][fbud.bud_index] = radius


def _resize(lst, n, fill):
    while len(lst) < n:
        lst.append(fill() if callable(fill) else fill)


# ---- flowers and fruit -----------------------------------------------------------------------------

@ports("Phytomer::createInflorescenceGeometry")
def create_inflorescence_geometry(phytomer, fbud, fruit_base, peduncle_axis, pitch, roll, azimuth, yaw_compound,
                                  scale_factor, is_open_flower: bool, gravity_fraction):
    pa, ctx = phytomer.plantarchitecture_ptr, phytomer.context_ptr
    pa.invalidateOrganIndex()
    _caches(pa)
    _bind_library_hooks()
    pp = phytomer.phytomer_parameters
    infl = pp.inflorescence
    inst = pa.plant_instances[phytomer.plantID]
    build_parameters = inst.organ_build_parameters
    pitch, roll, azimuth, yaw_compound = f32(pitch), f32(roll), f32(azimuth), f32(yaw_compound)
    scale_factor, gravity_fraction = f32(scale_factor), f32(gravity_fraction)
    fruit_base, peduncle_axis = _v(fruit_base), _v(peduncle_axis)

    if fbud.state == BUD_FRUITING:
        if infl.unique_prototypes > 0:
            prototype = ctx.randu(0, int(infl.unique_prototypes) - 1)
            objID_fruit = ctx.copy_object(
                pa.unique_fruit_prototype_objIDs[_prototype_key(infl.fruit_prototype_function, build_parameters)][prototype])
        else:
            objID_fruit = infl.fruit_prototype_function(ctx, 1, sample_organ_build_parameters(pa, build_parameters))
            fruit_material_name = inst.plant_name + "_fruit"
            _name_organ_materials(pa, objID_fruit, fruit_material_name)
            make_closed_fruit_surfaces_one_sided(ctx, objID_fruit, fruit_material_name)
            record_fruit_class_map(ctx, objID_fruit, pp)
    else:
        if infl.unique_prototypes > 0:
            prototype = ctx.randu(0, int(infl.unique_prototypes) - 1)
            key = _prototype_key(infl.flower_prototype_function, build_parameters)
            cache = pa.unique_open_flower_prototype_objIDs if is_open_flower else pa.unique_closed_flower_prototype_objIDs
            objID_fruit = ctx.copy_object(cache[key][prototype])
        else:
            objID_fruit = infl.flower_prototype_function(ctx, 1, is_open_flower,
                                                         sample_organ_build_parameters(pa, build_parameters))
            _name_organ_materials(pa, objID_fruit,
                                  inst.plant_name + ("_flower_open" if is_open_flower else "_flower_closed"))

    # Step 2: scale; Step 3: roll, pitch, azimuth at the origin; Step 4: translate; Step 5: compound yaw
    ctx.scale_object(objID_fruit, (scale_factor, scale_factor, scale_factor))
    if abs(float(roll)) > _TINY:
        ctx.rotate_object(objID_fruit, roll, "x")
    if abs(float(pitch)) > _TINY:
        ctx.rotate_object(objID_fruit, pitch, "y")
    if abs(float(azimuth)) > _TINY:
        ctx.rotate_object(objID_fruit, azimuth, "z")
    ctx.translate_object(objID_fruit, fruit_base)
    if abs(float(yaw_compound)) > _TINY:
        ctx.rotate_object(objID_fruit, yaw_compound, peduncle_axis, origin=fruit_base)

    # Step 6: gravity (off for tomato; not oracle-tested)
    if gravity_fraction > 0:
        from ..core.object_transforms import _xform_point, make_rotation_matrix_axis
        from ..core.primitives import make_rotation_matrix
        fruit_axis = _v((1, 0, 0))
        fruit_axis = _xform_point(make_rotation_matrix(roll, "x"), fruit_axis)
        fruit_axis = _xform_point(make_rotation_matrix(pitch, "y"), fruit_axis)
        fruit_axis = _xform_point(make_rotation_matrix(azimuth, "z"), fruit_axis)
        if abs(float(yaw_compound)) > _TINY:
            fruit_axis = _xform_point(make_rotation_matrix_axis(yaw_compound, peduncle_axis), fruit_axis)
        fruit_axis = V.normalize(fruit_axis)
        down = _v((0, 0, -1))
        angle_from_down = V.acos_safe(V.dot(fruit_axis, down))
        hinge = V.cross(fruit_axis, down)
        if V.magnitude(hinge) < f32(1e-6):
            hinge = _v((-V.sinf(azimuth), V.cosf(azimuth), 0))
        hinge = V.normalize(hinge)
        if angle_from_down > f32(1e-6):
            ctx.rotate_object(objID_fruit, f32(gravity_fraction * angle_from_down), hinge, origin=fruit_base)

    # Step 7: records
    fbud.inflorescence_objIDs.append(objID_fruit)
    fbud.inflorescence_bases.append(fruit_base.copy())
    rot = AxisRotation()
    rot.pitch = pitch
    rot.yaw = yaw_compound
    rot.roll = roll
    rot.azimuth = azimuth
    rot.peduncle_axis = tuple(peduncle_axis)
    rot.gravity_fraction = gravity_fraction
    fbud.inflorescence_rotation.append(rot)
    fbud.inflorescence_base_scales.append(scale_factor)
    return objID_fruit


@ports("Phytomer::updateInflorescence")
def update_inflorescence(pa, phytomer, fbud):
    """PA:3934-4197: rebuild the peduncle tube from the bud base and place the flowers / fruit on it.

    ``pa`` is ``phytomer.plantarchitecture_ptr`` (kept as an argument so the call sites read like W3-C's)."""
    ctx = phytomer.context_ptr
    pa.invalidateOrganIndex()
    pp = phytomer.phytomer_parameters
    ped, infl = pp.peduncle, pp.inflorescence
    fbud.contact_swing_axis = (0, 0, 1)
    fbud.contact_swing_angle = 0
    phytomer.build_context_geometry_peduncle = pa.build_context_geometry_peduncle

    Ndiv_peduncle_length = max(1, int(ped.length_segments))
    Ndiv_peduncle_radius = max(3, int(ped.radial_subdivisions))
    if ped.length_segments == 0 or ped.radial_subdivisions < 3:
        phytomer.build_context_geometry_peduncle = False

    peduncle_length = ped.length.val()
    dr_peduncle = f32(peduncle_length / f32(Ndiv_peduncle_length))

    nseg = int(ped.length_segments)
    peduncle_vertices = [np.zeros(3, f32) for _ in range(nseg + 1)]
    peduncle_vertices[0] = _v(fbud.base_position)
    peduncle_radii = [f32(0)] * (nseg + 1)
    peduncle_radii[0] = get_reconciled_peduncle_radius(phytomer)
    peduncle_colors = get_peduncle_node_colors(phytomer, nseg + 1)

    peduncle_axis = get_axis_vector(f32(1), phytomer.getInternodeNodePositions())
    inflorescence_bending_axis_actual = _v(phytomer.inflorescence_bending_axis)

    if ped.pitch.val() != 0 or f32(fbud.base_rotation.pitch) != 0:
        peduncle_axis = _rot(peduncle_axis, inflorescence_bending_axis_actual,
                             f32(V.deg2rad(ped.pitch.val()) + f32(fbud.base_rotation.pitch)))

    internode_axis = get_axis_vector(f32(1), phytomer.getInternodeNodePositions())
    if not phytomer.petiole_vertices:
        parent_petiole_base_axis = internode_axis
    else:
        parent_petiole_base_axis = phytomer.getPetioleAxisVector(f32(0), fbud.parent_index)
    parent_petiole_azimuth = f32(-_atan2f(parent_petiole_base_axis[1], parent_petiole_base_axis[0]))
    current_peduncle_azimuth = f32(-_atan2f(peduncle_axis[1], peduncle_axis[0]))
    azimuthal_rotation = f32(current_peduncle_azimuth - parent_petiole_azimuth)
    peduncle_axis = _rot(peduncle_axis, internode_axis, azimuthal_rotation)
    inflorescence_bending_axis_actual = _rot(inflorescence_bending_axis_actual, internode_axis, azimuthal_rotation)

    peduncle_yaw = ped.yaw.val()
    if peduncle_yaw != 0:
        peduncle_axis = _rot(peduncle_axis, internode_axis, V.deg2rad(peduncle_yaw))
        inflorescence_bending_axis_actual = _rot(inflorescence_bending_axis_actual, internode_axis,
                                                 V.deg2rad(peduncle_yaw))

    # float theta_base = fabs(cart2sphere(peduncle_axis).zenith): only read by fruit collision avoidance
    if getattr(pa, "fruit_collision_detection_enabled", False):
        raise NotImplementedError("fruit-branch collision avoidance is not ported")

    peduncle_curvature = ped.curvature.val()
    ped.curvature.resample()
    peduncle_roll = ped.roll.val()

    petiole_idx, bud_idx = get_floral_bud_storage_index(phytomer, fbud)
    if petiole_idx >= len(phytomer.peduncle_length):
        for name in ("peduncle_vertices", "peduncle_radii", "peduncle_length", "peduncle_radius", "peduncle_pitch",
                     "peduncle_curvature", "peduncle_roll", "peduncle_yaw"):
            _resize(getattr(phytomer, name), petiole_idx + 1, list)
    if len(phytomer.peduncle_length[petiole_idx]) <= bud_idx:
        for name in ("peduncle_length", "peduncle_radius", "peduncle_pitch", "peduncle_curvature", "peduncle_roll",
                     "peduncle_yaw"):
            _resize(getattr(phytomer, name)[petiole_idx], bud_idx + 1, f32(0))
    phytomer.peduncle_length[petiole_idx][bud_idx] = peduncle_length
    phytomer.peduncle_radius[petiole_idx][bud_idx] = ped.radius.val()
    phytomer.peduncle_pitch[petiole_idx][bud_idx] = ped.pitch.val()
    phytomer.peduncle_curvature[petiole_idx][bud_idx] = peduncle_curvature
    phytomer.peduncle_roll[petiole_idx][bud_idx] = peduncle_roll
    phytomer.peduncle_yaw[petiole_idx][bud_idx] = peduncle_yaw

    for i in range(1, nseg + 1):
        peduncle_axis = bend_peduncle_axis(peduncle_axis, peduncle_curvature, dr_peduncle)
        peduncle_vertices[i] = peduncle_vertices[i - 1] + dr_peduncle * peduncle_axis
        peduncle_radii[i] = peduncle_radii[0]

    if phytomer.build_context_geometry_peduncle:
        objID = ctx.add_tube_object(Ndiv_peduncle_radius, peduncle_vertices, peduncle_radii, peduncle_colors)
        fbud.peduncle_objIDs.append(objID)
        ctx.set_primitive_data(ctx.object_primitive_uuids(objID), "object_label", "peduncle")
        plant_name = pa.plant_instances[phytomer.plantID].plant_name
        _name_organ_materials(pa, objID, f"{plant_name}_{phytomer.parent_shoot_ptr.shoot_type_label}_peduncle")

    _resize(phytomer.peduncle_vertices[petiole_idx], bud_idx + 1, list)
    phytomer.peduncle_vertices[petiole_idx][bud_idx] = [v.copy() for v in peduncle_vertices]
    _resize(phytomer.peduncle_radii[petiole_idx], bud_idx + 1, list)
    phytomer.peduncle_radii[petiole_idx][bud_idx] = list(peduncle_radii)

    ped.length.resample()
    ped.radius.resample()
    ped.pitch.resample()
    ped.yaw.resample()

    ensure_inflorescence_prototypes_initialized(pa, pp, phytomer.plantID)

    flowers_per_peduncle = int(infl.flowers_per_peduncle.val())
    flower_offset_val = clamp_offset(flowers_per_peduncle, infl.flower_offset.val())
    petioles_per_internode = f32(int(pp.petiole.petioles_per_internode))
    for fruit in range(flowers_per_peduncle):
        if fbud.state == BUD_FRUITING:
            scale_factor = infl.fruit_prototype_scale.val()
            infl.fruit_prototype_scale.resample()
        else:
            scale_factor = infl.flower_prototype_scale.val()
            infl.flower_prototype_scale.resample()

        ind_from_tip = f32(abs(f32(f32(fruit) - f32(f32(flowers_per_peduncle - 1) / petioles_per_internode))))

        fruit_base = peduncle_vertices[-1].copy()
        frac = f32(1)
        if flowers_per_peduncle > 1 and flower_offset_val > 0:
            if ind_from_tip != 0:
                offset = f32(f32(f32(ind_from_tip - f32(0.5)) * flower_offset_val) * ped.length.val())
                if ped.length.val() > 0:
                    frac = f32(f32(1) - f32(offset / ped.length.val()))
                fruit_base = interpolate_tube(peduncle_vertices, frac)

        compound_rotation = f32(0)
        if flowers_per_peduncle > 1:
            if flower_offset_val == 0:
                dphi = f32(float(PI_F) / (math.floor(0.5 * float(f32(flowers_per_peduncle - 1))) + 1))
                compound_rotation = f32(-PI_F + f32(dphi * f32(f32(fruit) + f32(0.5))))
            else:
                compound_rotation = f32(f32(V.deg2rad(pp.internode.phyllotactic_angle.val()) * ind_from_tip)
                                        + f32(f32(f32(f32(2) * PI_F) / petioles_per_internode) * f32(fruit)))
                pp.internode.phyllotactic_angle.resample()

        axis = get_axis_vector(frac, peduncle_vertices)

        applied_roll = V.deg2rad(infl.roll.val())
        infl.roll.resample()
        applied_pitch_param = V.deg2rad(infl.pitch.val())
        infl.pitch.resample()

        pitch_inflorescence = f32(-V.asin_safe(axis[2]) + applied_pitch_param)
        gravity_fraction = f32(0)
        if fbud.state == BUD_FRUITING:
            gravity_fraction = infl.fruit_gravity_factor_fraction.val()
        infl.fruit_gravity_factor_fraction.resample()

        azimuth = _atan2f(axis[1], axis[0])
        yaw_compound = f32(V.deg2rad(peduncle_roll) + compound_rotation)
        is_open_flower = fbud.state == BUD_FLOWER_OPEN
        create_inflorescence_geometry(phytomer, fbud, fruit_base, axis, pitch_inflorescence, applied_roll, azimuth,
                                      yaw_compound, scale_factor, is_open_flower, gravity_fraction)
    infl.flowers_per_peduncle.resample()
    ped.roll.resample()

    od = pa.output_object_data
    if od.get("age"):
        ctx.set_object_data(fbud.inflorescence_objIDs, "age", f32(fbud.age))
        ctx.set_object_data(fbud.peduncle_objIDs, "age", f32(fbud.age))
    if od.get("rank"):
        ctx.set_object_data(fbud.peduncle_objIDs, "rank", int(phytomer.rank))
        ctx.set_object_data(fbud.inflorescence_objIDs, "rank", int(phytomer.rank))
    if od.get("plant_name"):
        name = pa.plant_instances[phytomer.plantID].plant_name
        ctx.set_object_data(fbud.peduncle_objIDs, "plant_name", name)
        ctx.set_object_data(fbud.inflorescence_objIDs, "plant_name", name)
    if od.get("peduncleID"):
        for objID in fbud.peduncle_objIDs:
            ctx.set_object_data(objID, "peduncleID", int(objID))
    for objID in fbud.inflorescence_objIDs:
        if fbud.state == BUD_FLOWER_CLOSED and od.get("closedflowerID"):
            ctx.set_object_data(objID, "closedflowerID", int(objID))
        elif fbud.state == BUD_FLOWER_OPEN and od.get("openflowerID"):
            ctx.clear_object_data(objID, "closedflowerID")
            ctx.set_object_data(objID, "openflowerID", int(objID))
        elif od.get("fruitID"):
            ctx.set_object_data(objID, "fruitID", int(objID))


@ports("Phytomer::setInflorescenceScaleFraction")
def set_inflorescence_scale_fraction(phytomer, fbud, inflorescence_scale_factor_fraction):
    frac = f32(inflorescence_scale_factor_fraction)
    assert 0 <= frac <= 1
    if frac == f32(fbud.current_fruit_scale_factor):
        return
    delta_scale = f32(frac / f32(fbud.current_fruit_scale_factor))
    fbud.current_fruit_scale_factor = frac
    for objID, base in zip(fbud.inflorescence_objIDs, fbud.inflorescence_bases):
        phytomer.context_ptr.scale_object_about_point(objID, (delta_scale, delta_scale, delta_scale), base)


@ports("Phytomer::setFloralBudState(BudState,FloralBud&)")
def set_floral_bud_state(pa, phytomer, fbud, state):
    """PA:1491-1547 (geometry side; the carbon model is not ported)."""
    ctx = phytomer.context_ptr
    pa.invalidateOrganIndex()
    if fbud.state == state:
        return
    if state in (BUD_DORMANT, BUD_ACTIVE):
        fbud.state = state
        return
    if pa.carbon_model_enabled:
        raise NotImplementedError("carbohydrate model (flower/fruit construction costs) is not ported")
    ctx.delete_object(list(fbud.inflorescence_objIDs))
    fbud.inflorescence_objIDs = []
    fbud.inflorescence_bases = []
    fbud.inflorescence_rotation = []
    fbud.inflorescence_base_scales = []
    if phytomer.build_context_geometry_peduncle:
        ctx.delete_object(list(fbud.peduncle_objIDs))
        fbud.peduncle_objIDs = []
    fbud.contact_swing_axis = (0, 0, 1)
    fbud.contact_swing_angle = 0
    fbud.state = state
    if state != BUD_DEAD:
        update_inflorescence(pa, phytomer, fbud)
        fbud.time_counter = 0
        if fbud.state == BUD_FRUITING and f32(fbud.dd_to_inflorescence_maturity) != 0:
            set_inflorescence_scale_fraction(phytomer, fbud, f32(0.25))


def fruit_growth_fraction(time_counter, dd_to_maturity) -> np.float32:
    """``fmin(1, 0.25f + 0.75f * fbud.time_counter / dd_to_maturity)`` (PA:12155): float expression, double
    ``fmin``, narrowed. For W3-C's fruit-growth loop and the parity replay."""
    x = f32(f32(0.25) + f32(f32(f32(0.75) * f32(time_counter)) / f32(dd_to_maturity)))
    return f32(min(1.0, float(x)))


__all__ = ["set_floral_bud_state", "update_inflorescence", "create_inflorescence_geometry",
           "ensure_inflorescence_prototypes_initialized", "set_inflorescence_scale_fraction", "update_peduncle_radii",
           "bend_peduncle_axis", "get_reconciled_peduncle_radius", "get_peduncle_node_colors",
           "peduncle_should_match_culm", "get_floral_bud_storage_index", "make_closed_fruit_surfaces_one_sided",
           "sample_organ_build_parameters", "fruit_growth_fraction"]
