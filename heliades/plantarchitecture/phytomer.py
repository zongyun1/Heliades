# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp (Copyright (C) 2016-2026 Brian Bailey):
#   Phytomer::Phytomer                      PA:2694-3601
#   createPetioleFloralBuds                 PA:2666-2691
#   Phytomer::getInternodeNodePositions/... PA:1239-1296
#   Phytomer::setVegetativeBudState/...     PA:1449-1547
#   Phytomer::getLeafArea                   PA:1379-1394
#   Phytomer::scaleInternodeMaxLength       PA:4349-4358
#   PlantArchitecture::getCachedLeafPrototypeObjID / recordLeafPrototypeRestGeometry  PA:4532-4595
#   Phytomer::buildLeafPrototype / labelLeafPrototype / assignPetioluleMaterial      PA:4904-5014
#   Phytomer::orientLeaf                    PA:5016-5197
#   Phytomer::recordPetioleRestShape / bendPetioleUnderLeafWeight (rigid guard)      PA:5416-5480
#   Phytomer::scaleLeafPrototypeScale       PA:5846-5874
#   Phytomer::getLeafShape                  PA:6225-6250
#   PlantArchitecture::nameOrganMaterials   PA:181-264
#   PlantArchitecture::getLeafBladeArea     PA:8799-8810
"""The ``Phytomer``: one internode, its petiole(s), buds and leaves (W3-B).

Everything here is float32 op for op in C++ order. The constructor is a line-by-line port of PA:2694-3601;
the branches Helios takes only with collision avoidance, attraction points, prescribed (measured) internodes,
textured internodes or flexible petioles raise ``NotImplementedError`` (none of them is on the tomato path).

**Leaf geometry interface (W3-A).** The constructor builds the 35 hidden leaflet blades through
``Phytomer.buildLeafPrototype``, which calls ``phytomer_parameters.leaf.prototype.prototype_function`` -- the
``hooks.GenericLeafPrototype`` hook -- as ``fn(scene, prototype: LeafPrototype, compound_leaf_index: int) -> objID``
(Assets.cpp:640). W3-A binds that hook (``hooks.GenericLeafPrototype.implement(GenericLeafPrototype)``) when
``heliades.plantarchitecture.leaf_prototype`` is imported. The prototype function must

* draw only from the ``RandomParameter`` members of ``prototype`` (their generator is the private
  ``MinstdRand0(identifier)`` while the cache is built), in Helios order, including the trailing resamples;
* set ``prototype.last_built_shape`` / ``last_built_shape_valid``;
* return a polymesh object ID (``addPolymeshObject``), petiolule triangles labelled ``"petiolule"``.

Labelling, petiolule material, rest-geometry record, ``nameOrganMaterials`` and ``hideObject`` are done here,
in the C++ order (PA:3477-3494). Placement of every leaflet copy (``copyObject`` -> ``scaleObject`` ->
``orientLeaf`` rotations -> ``translateObject``) is :meth:`Phytomer._place_leaflet`. Until W3-A lands, the
hook falls back to :func:`stub_generic_leaf_prototype` (same RNG behaviour, placeholder triangles), and
``PlantArchitecture.leaf_geometry_is_stub`` is set.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
from dataclasses import dataclass, field

import numpy as np

from ..core import vec as V
from ..migration_registry import ports
from . import hooks
from .params import (BUD_ACTIVE, BUD_DEAD, BUD_DORMANT, AxisRotation, FloralBud, LeafShapeValues,
                     PhytomerParameters, VegetativeBud, deg2rad, make_AxisRotation)
from .random_parameter import PlantArchitectureError
from .tube_utils import (MIN_TUBE_LENGTH_FOR_GEOMETRY, MIN_TUBE_RADIUS_FOR_GEOMETRY, NO_PETIOLE_OBJID, PI_F,
                         clamp_offset, compound_leaf_rotation, distribution_mean, get_axis_vector, interpolate_tube,
                         leaflet_size_max, make_petiole_tube, petiole_radius_fraction)

f32 = np.float32
NULLORIGIN = np.zeros(3, f32)


def _load_atan2f():
    try:
        m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        fn = m.atan2f
        fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float, ctypes.c_float]
        return fn
    except (OSError, AttributeError):
        return None


_ATAN2F = _load_atan2f()


def atan2f(y, x) -> np.float32:
    """``std::atan2(float, float)`` = glibc ``atan2f``."""
    if _ATAN2F is not None:
        return f32(_ATAN2F(float(f32(y)), float(f32(x))))
    return f32(math.atan2(float(f32(y)), float(f32(x))))


def _v(x) -> np.ndarray:
    return np.asarray(x, f32).reshape(3).copy()


def _vec_eq(a, b) -> bool:
    return bool(a[0] == b[0] and a[1] == b[1] and a[2] == b[2])


def _rot(p, axis, theta) -> np.ndarray:
    """``rotatePointAboutLine(p, nullorigin, axis, theta)``."""
    return V.rotate_point_about_line(p, NULLORIGIN, axis, theta)


def _smul(s, v) -> np.ndarray:
    return f32(s) * np.asarray(v, f32)


# ---- materials (PlantArchitecture::nameOrganMaterials) ---------------------------------------------

@dataclass
class OrganMaterial:
    """PlantArchitecture::OrganMaterial (label + the appearance it was named for). The core material table
    has no normal maps yet, so those members are not compared (they are equal by construction)."""
    label: str
    color: np.ndarray
    texture_file: str = ""
    texture_color_overridden: bool = False


@ports("PlantArchitecture::nameOrganMaterials")
def name_organ_materials(pa, objID: int, desired_base_name: str):
    ctx = pa.context_ptr
    groups: dict[str, list[int]] = {}
    for u in ctx.object_primitive_uuids(objID):
        label = ctx.get_primitive_material_label(u)
        if label[:7] == "__auto_":
            groups.setdefault(label, []).append(u)
    for automatic_label in sorted(groups):                    # std::map<std::string, ...> order
        prims = groups[automatic_label]
        key = (automatic_label, desired_base_name)
        named = pa.organ_material_labels.get(key)
        if named is not None and ctx.does_material_exist(named.label):
            ctx.assign_material_to_primitive(prims, named.label)
            continue
        automatic = ctx.get_material(ctx.get_primitive_material_id(prims[0]))
        acol = np.asarray(automatic.color, f32)
        same = None
        for other_key in sorted(pa.organ_material_labels):    # std::map<pair<string,string>, ...> order
            other = pa.organ_material_labels[other_key]
            if (other_key[1] != desired_base_name or not ctx.does_material_exist(other.label)
                    or other.texture_file != automatic.texture_file
                    or other.texture_color_overridden != automatic.texture_color_overridden):
                continue
            if all(abs(f32(other.color[k] - acol[k])) < f32(1e-4) for k in range(4)):
                same = other
                break
        if same is not None:
            ctx.assign_material_to_primitive(prims, same.label)
            pa.organ_material_labels[key] = same
            continue
        label = desired_base_name
        suffix = 1
        while ctx.does_material_exist(label):
            label = f"{desired_base_name}_{suffix}"
            suffix += 1
        if automatic.reference_count == len(prims):
            ctx.rename_material(automatic_label, label)
        else:
            ctx.add_material(label)
            ctx.set_material_color(label, acol)
            if automatic.texture_file:
                ctx.set_material_texture(label, automatic.texture_file)
            ctx.set_material_texture_color_override(label, automatic.texture_color_overridden)
            ctx.set_material_twosided_flag(label, automatic.twosided_flag)
            ctx.assign_material_to_primitive(prims, label)
        pa.organ_material_labels[key] = OrganMaterial(label, acol.copy(), automatic.texture_file,
                                                      automatic.texture_color_overridden)


@ports("PlantArchitecture::getLeafBladeArea")
def get_leaf_blade_area(pa, leaf_objID: int) -> np.float32:
    ctx = pa.context_ptr
    if not ctx.does_object_exist(leaf_objID):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::getLeafBladeArea): Leaf object with ID of {leaf_objID} "
                                     "does not exist.")
    area = ctx.object_area(leaf_objID)
    for u in ctx.filter_primitives_by_data(ctx.object_primitive_uuids(leaf_objID), "object_label", "petiolule"):
        area = f32(area - ctx.primitive_area(u))
    return f32(area)


# ---- leaf prototype cache records ------------------------------------------------------------------

@dataclass
class LeafRestGeometry:
    """PlantArchitecture::LeafRestGeometry (only what a rigid leaf records)."""
    has_shape: bool = False
    shape: LeafShapeValues | None = None
    lattice_frame_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, f32))
    vertices: list = field(default_factory=list)
    appended_vertices: list = field(default_factory=list)
    subdivisions_x: int = 0
    subdivisions_y: int = 0


@ports("PlantArchitecture::recordLeafPrototypeRestGeometry")
def record_leaf_prototype_rest_geometry(pa, prototype_params, leaves_per_petiole: int, prototype_index: int,
                                        objID_leaf: int, leaf_flexibility):
    rest = LeafRestGeometry()
    rest.has_shape = bool(prototype_params.last_built_shape_valid)
    if rest.has_shape:
        rest.shape = prototype_params.last_built_shape.copy()
    T = pa.context_ptr.object_transform(objID_leaf)
    rest.lattice_frame_offset = np.array([T[3], T[7], T[11]], f32)
    if f32(leaf_flexibility) > 0:
        raise NotImplementedError("recordLeafPrototypeRestGeometry for a flexible leaf (W3-A / later)")
    key = (int(prototype_params.unique_prototype_identifier), max(int(leaves_per_petiole), 0))
    pa.unique_leaf_prototype_rest_geometry[key][prototype_index].append(rest)


@ports("PlantArchitecture::getCachedLeafPrototypeObjID")
def get_cached_leaf_prototype_objID(pa, prototype_identifier, leaves_per_petiole, prototype_index, leaf_index,
                                    calling_function: str) -> int:
    cached = pa.unique_leaf_prototype_objIDs.get((int(prototype_identifier), int(leaves_per_petiole)))
    if cached is None:
        raise PlantArchitectureError(f"ERROR ({calling_function}): No leaf blade prototypes have been cached for a petiole "
                                     f"of {leaves_per_petiole} leaflets.")
    if len(cached) <= prototype_index:
        raise PlantArchitectureError(f"ERROR ({calling_function}): Leaf prototype {prototype_index} was requested, but "
                                     f"only {len(cached)} unique leaf prototypes were cached.")
    if len(cached[prototype_index]) <= leaf_index:
        raise PlantArchitectureError(f"ERROR ({calling_function}): Leaflet {leaf_index} was requested, but the cached leaf "
                                     f"prototype holds blades for only {len(cached[prototype_index])} leaflet positions.")
    return cached[prototype_index][leaf_index]


# ---- leaf prototype function: W3-A hook with a stand-in -------------------------------------------

_SHAPE_READ_ORDER = ("leaf_aspect_ratio", "roll_fraction", "midrib_fold_fraction", "longitudinal_curvature",
                     "longitudinal_curvature_exponent", "lateral_curvature", "petiole_roll", "wave_period",
                     "wave_amplitude")
_SHAPE_RESAMPLE_ORDER = ("leaf_aspect_ratio", "midrib_fold_fraction", "longitudinal_curvature",
                         "longitudinal_curvature_exponent", "lateral_curvature", "roll_fraction", "petiole_roll")


def stub_generic_leaf_prototype(context, prototype, compound_leaf_index, first_vertex=(0.0, 0.0, 0.0)) -> int:
    """Stand-in for ``GenericLeafPrototype`` (AS:640) until W3-A: the same parameter reads and resamples on the
    prototype's (private) stream, in the same order (AS:675-701, 867-878), and the same ``last_built_shape``
    record, but two placeholder triangles instead of the lattice blade. ``first_vertex`` sets the polymesh
    frame (``addPolymeshObject`` translates to the first vertex), so a test can give it the oracle's frame."""
    p = prototype
    Nx = int(p.subdivisions)
    Ny = int(math.ceil(float(f32(p.leaf_aspect_ratio.val() * f32(Nx)))))
    if Ny % 2:
        Ny += 1
    for name in _SHAPE_READ_ORDER:
        getattr(p, name).val()
    shape = LeafShapeValues()
    shape.aspect_ratio = p.leaf_aspect_ratio.val()
    shape.midrib_fold_fraction = p.midrib_fold_fraction.val()
    shape.longitudinal_curvature = p.longitudinal_curvature.val()
    shape.longitudinal_curvature_exponent = p.longitudinal_curvature_exponent.val()
    shape.lateral_curvature = p.lateral_curvature.val()
    shape.petiole_roll = p.petiole_roll.val()
    shape.wave_period = p.wave_period.val()
    shape.wave_amplitude = p.wave_amplitude.val()
    shape.roll_fraction = p.roll_fraction.val()
    shape.subdivisions_x = Nx
    shape.subdivisions_y = Ny
    o = _v(first_vertex)
    a = f32(f32(0.5) * shape.aspect_ratio)
    u0 = context.add_triangle(o, o + _v((1, a, 0)), o + _v((1, -a, 0)))
    u1 = context.add_triangle(o + _v((1, a, 0)), o + _v((2, 0, 0)), o + _v((1, -a, 0)))
    for name in _SHAPE_RESAMPLE_ORDER:
        getattr(p, name).resample()
    p._raw_set("last_built_shape", shape)
    p._raw_set("last_built_shape_valid", True)
    p.wave_period.resample()
    p.wave_amplitude.resample()
    return context.add_polymesh_object([u0, u1])


def _resolve_prototype_function(pa, fn):
    """The prototype function to call: an explicit per-PlantArchitecture override, the bound W3-A hook, or
    the stub (recorded in ``pa.leaf_geometry_is_stub``)."""
    override = getattr(pa, "leaf_prototype_function_override", None)
    if override is not None:
        pa.leaf_geometry_is_stub = True
        return override
    if fn is hooks.GenericLeafPrototype and fn.impl is None:
        try:
            from . import leaf_prototype  # noqa: F401  (W3-A binds the hook on import)
        except ImportError:
            pass
        if fn.impl is None:
            pa.leaf_geometry_is_stub = True
            return stub_generic_leaf_prototype
    return fn


# ---- floral buds ---------------------------------------------------------------------------------

@ports("createPetioleFloralBuds")
def create_petiole_floral_buds(max_floral_buds_per_petiole, parent_index: int, petiole_count: int, base_position,
                               bending_axis) -> list:
    """PA:2666-2691; ``max_floral_buds_per_petiole`` is resampled for the next petiole."""
    n = int(max_floral_buds_per_petiole.val())
    max_floral_buds_per_petiole.resample()
    buds = []
    for bud_index in range(max(n, 0)):
        fbud = FloralBud()
        fbud.bud_index = bud_index
        fbud.parent_index = parent_index
        pitch_adjustment = f32(f32(f32(f32(bud_index) * f32(0.1)) * PI_F) / f32(petiole_count))
        yaw_adjustment = f32(f32(f32(-0.25) * PI_F)
                             + f32(f32(f32(f32(bud_index) * f32(0.5)) * PI_F) / f32(petiole_count)))
        fbud.base_rotation = make_AxisRotation(pitch_adjustment, yaw_adjustment, 0)
        fbud.base_position = tuple(_v(base_position))
        fbud.bending_axis = tuple(_v(bending_axis))
        buds.append(fbud)
    return buds


class Phytomer:
    """helios Phytomer (geometry part, W3-B)."""

    no_petiole_objID = NO_PETIOLE_OBJID

    @ports("Phytomer::Phytomer")
    def __init__(self, params: PhytomerParameters, parent_shoot, phytomer_index: int, parent_internode_axis,
                 parent_petiole_axis, internode_base_origin, shoot_base_rotation: AxisRotation, internode_radius,
                 internode_length_max, internode_length_scale_factor_fraction, leaf_scale_factor_fraction, rank: int,
                 plantarchitecture_ptr, context_ptr, prescribed_internode=None):
        if prescribed_internode is not None:
            raise NotImplementedError("Phytomer from a prescribed internode (addShootFromNodePositions) is not ported")
        pa, ctx = plantarchitecture_ptr, context_ptr
        self.rank = rank
        self.context_ptr = ctx
        self.plantarchitecture_ptr = pa
        pa.invalidateOrganIndex()
        # this->phytomer_parameters = params (default member, then the implicit operator=), then resample
        self.phytomer_parameters = PhytomerParameters()
        self.phytomer_parameters._assign_from(params)
        self.phytomer_parameters.resample()
        pp = self.phytomer_parameters

        parent_shoot_parameters = parent_shoot.shoot_parameters.copy()

        internode_radius = f32(internode_radius)
        internode_length_max = f32(internode_length_max)
        internode_length_scale_factor_fraction = f32(internode_length_scale_factor_fraction)
        leaf_scale_factor_fraction = f32(leaf_scale_factor_fraction)

        self.internode_radius_initial = internode_radius
        self.internode_length_max = internode_length_max
        self.shoot_index = [int(phytomer_index), int(parent_shoot.current_node_number),
                            int(parent_shoot_parameters.max_nodes.val())]
        self.rank = parent_shoot.rank
        self.plantID = parent_shoot.plantID
        self.parent_shoot_ID = parent_shoot.ID
        self.parent_shoot_ptr = parent_shoot

        # state with in-class defaults (H:1691-2966) used on this path
        self.age = f32(0)
        self.isdormant = False
        self.downstream_leaf_area = f32(0)
        self.leaf_texture_stage = 0
        self.internode_curvature_arc_length_step = f32(0)
        self.leaf_shape_state: dict = {}
        self.leaf_stalk_base_placed: dict = {}
        self.last_built_leaf_shape_valid = False
        self.last_built_leaf_shape = None
        self.leaf_is_sessile = False

        build_context_geometry_internode = pa.build_context_geometry_internode
        self.build_context_geometry_petiole = pa.build_context_geometry_petiole
        self.build_context_geometry_peduncle = pa.build_context_geometry_peduncle

        Ndiv_internode_length = max(1, int(pp.internode.length_segments))
        Ndiv_internode_radius = max(3, int(pp.internode.radial_subdivisions))
        Ndiv_petiole_length = max(1, int(pp.petiole.length_segments))
        Ndiv_petiole_radius = max(3, int(pp.petiole.radial_subdivisions))

        if pp.internode.length_segments == 0 or pp.internode.radial_subdivisions < 3:
            build_context_geometry_internode = False
        if pp.petiole.length_segments == 0 or pp.petiole.radial_subdivisions < 3:
            self.build_context_geometry_petiole = False
        if pp.petiole.petioles_per_internode == 0:
            self.build_context_geometry_petiole = False
            pp.leaf.leaves_per_petiole = 0
        npet = int(pp.petiole.petioles_per_internode)

        self.current_internode_scale_factor = internode_length_scale_factor_fraction
        self.current_leaf_scale_factor = [leaf_scale_factor_fraction] * npet
        self.current_petiole_scale_factor = [leaf_scale_factor_fraction] * npet

        if internode_radius == 0:
            internode_radius = MIN_TUBE_RADIUS_FOR_GEOMETRY

        # ---- internode variables
        internode_length = f32(internode_length_scale_factor_fraction * internode_length_max)
        dr_internode = f32(internode_length / f32(pp.internode.length_segments))
        dr_internode_max = f32(internode_length_max / f32(pp.internode.length_segments))
        phytomer_internode_vertices = [np.zeros(3, f32) for _ in range(Ndiv_internode_length + 1)]
        phytomer_internode_vertices[0] = _v(internode_base_origin)
        phytomer_internode_radii = [f32(0)] * (Ndiv_internode_length + 1)
        phytomer_internode_radii[0] = internode_radius
        self.internode_pitch = deg2rad(pp.internode.pitch.val())
        self.internode_phyllotactic_angle = deg2rad(pp.internode.phyllotactic_angle.val())
        if phytomer_index == 0 or not parent_shoot.phytomers:
            self.internode_phyllotactic_azimuth = f32(0)
        else:
            self.internode_phyllotactic_azimuth = f32(parent_shoot.phytomers[-1].internode_phyllotactic_azimuth
                                                      + self.internode_phyllotactic_angle)

        # ---- petiole variables
        self.petiole_length = [f32(0)] * npet
        self.petiole_length_max = [f32(0)] * npet
        self.petiole_vertices = [[] for _ in range(npet)]
        self.petiole_radii = [[] for _ in range(npet)]
        self.peduncle_vertices = [[] for _ in range(npet)]
        self.peduncle_radii = [[] for _ in range(npet)]
        self.peduncle_length = [[] for _ in range(npet)]
        self.peduncle_radius = [[] for _ in range(npet)]
        self.peduncle_pitch = [[] for _ in range(npet)]
        self.peduncle_curvature = [[] for _ in range(npet)]
        self.peduncle_roll = [[] for _ in range(npet)]
        self.peduncle_yaw = [[] for _ in range(npet)]
        self.petiole_pitch = [f32(0)] * npet
        self.petiole_curvature = [f32(0)] * npet
        self.petiole_taper = [f32(0)] * npet
        self.petiole_axis_initial = [np.zeros(3, f32) for _ in range(npet)]
        self.petiole_rotation_axis = [np.zeros(3, f32) for _ in range(npet)]
        dr_petiole = [f32(0)] * npet
        dr_petiole_max = [f32(0)] * npet
        suppress_petiole_geometry = [False] * npet
        self.leaf_is_sessile = bool(pp.petiole.length.val() == 0)
        for p in range(npet):
            self.petiole_vertices[p] = [np.zeros(3, f32) for _ in range(Ndiv_petiole_length + 1)]
            self.petiole_radii[p] = [f32(0)] * (Ndiv_petiole_length + 1)
            suppress_petiole_geometry[p] = bool(pp.petiole.radius.val() <= 0 or pp.petiole.length.val() <= 0)
            self.petiole_length_max[p] = pp.petiole.length.val()
            self.petiole_length[p] = f32(leaf_scale_factor_fraction * self.petiole_length_max[p])
            if self.petiole_length[p] <= 0:
                self.petiole_length[p] = MIN_TUBE_LENGTH_FOR_GEOMETRY
                self.petiole_length_max[p] = MIN_TUBE_LENGTH_FOR_GEOMETRY
                self.current_petiole_scale_factor[p] = f32(1)
            dr_petiole[p] = f32(self.petiole_length[p] / f32(pp.petiole.length_segments))
            dr_petiole_max[p] = f32(pp.petiole.length.val() / f32(pp.petiole.length_segments))
            self.petiole_radii[p][0] = f32(petiole_radius_fraction(leaf_scale_factor_fraction) * pp.petiole.radius.val())
            if self.petiole_radii[p][0] <= 0:
                self.petiole_radii[p][0] = MIN_TUBE_RADIUS_FOR_GEOMETRY
        self.petiole_objIDs = [NO_PETIOLE_OBJID] * npet

        # ---- leaf variables
        self.leaf_bases = [[] for _ in range(npet)]
        self.leaf_objIDs = [[] for _ in range(npet)]
        self.leaf_prototype_index = [[] for _ in range(npet)]
        self.leaf_pose_prescribed = [[] for _ in range(npet)]
        self.leaflet_angle_offsets = [[] for _ in range(npet)]
        self.leaf_angle_steering = [[] for _ in range(npet)]
        self.leaf_last_deformed_scale = [[] for _ in range(npet)]
        self.petiole_path_prescribed = [False] * npet

        self.leaf_flexibility = pp.leaf.prototype.resolveFlexibility()
        self.petiole_flexibility = f32(0) if self.leaf_is_sessile else max(f32(0), pp.petiole.flexibility.val())
        self.petiole_rest_offsets = [[] for _ in range(npet)]
        self.petiole_bend_state = [np.array([-1, 0, 0], f32) for _ in range(npet)]
        self.leaf_size_max = [[] for _ in range(npet)]
        self.leaf_rotation = [[] for _ in range(npet)]
        leaves_per_petiole = int(pp.leaf.leaves_per_petiole.val())
        if self.leaf_is_sessile and leaves_per_petiole > 1:
            raise PlantArchitectureError(
                f"ERROR (PlantArchitecture::Phytomer): Shoot type {parent_shoot.shoot_type_label} has a petiole length of 0, "
                f"meaning its leaves have no petiole, but {leaves_per_petiole} leaves per petiole.")
        leaflet_offset_val = clamp_offset(leaves_per_petiole, pp.leaf.leaflet_offset.val())
        for p in range(npet):
            self.leaf_size_max[p] = [f32(0)] * max(leaves_per_petiole, 0)
            self.leaf_rotation[p] = [AxisRotation() for _ in range(max(leaves_per_petiole, 0))]

        self.internode_colors = [(f32(0), f32(0), f32(0))] * (Ndiv_internode_length + 1)
        self.internode_colors[0] = pp.internode.color
        self.petiole_colors = [(f32(0), f32(0), f32(0))] * (Ndiv_petiole_length + 1)
        self.petiole_colors[0] = pp.petiole.color

        parent_internode_axis = _v(parent_internode_axis)
        parent_petiole_axis = _v(parent_petiole_axis)
        internode_axis = parent_internode_axis.copy()

        petiole_rotation_axis = V.cross(parent_internode_axis, parent_petiole_axis)
        if _vec_eq(petiole_rotation_axis, (0, 0, 0)):
            petiole_rotation_axis = _v((1, 0, 0))

        if phytomer_index == 0:
            # first phytomer of a shoot: the origin rotation about the parent axis (PA:2900-2955)
            if self.internode_pitch != 0:
                internode_axis = _rot(internode_axis, petiole_rotation_axis, f32(f32(0.5) * self.internode_pitch))
            roll_nudge = f32(0)
            if shoot_base_rotation.roll != 0 or roll_nudge != 0:
                a = f32(shoot_base_rotation.roll + roll_nudge)
                petiole_rotation_axis = _rot(petiole_rotation_axis, parent_internode_axis, a)
                internode_axis = _rot(internode_axis, parent_internode_axis, a)
            base_pitch_axis = _smul(-1, V.cross(parent_internode_axis, parent_petiole_axis))
            if shoot_base_rotation.pitch != 0:
                petiole_rotation_axis = _rot(petiole_rotation_axis, base_pitch_axis, f32(-shoot_base_rotation.pitch))
                internode_axis = _rot(internode_axis, base_pitch_axis, f32(-shoot_base_rotation.pitch))
            if shoot_base_rotation.yaw != 0:
                petiole_rotation_axis = _rot(petiole_rotation_axis, parent_internode_axis, shoot_base_rotation.yaw)
                internode_axis = _rot(internode_axis, parent_internode_axis, shoot_base_rotation.yaw)
            parent_shoot.radial_outward_axis = _rot(internode_axis, petiole_rotation_axis, f32(f32(0.5) * PI_F))
        else:
            if self.internode_pitch != 0:
                internode_axis = _rot(internode_axis, petiole_rotation_axis, f32(f32(-1.25) * self.internode_pitch))

        if V.magnitude(parent_shoot.initial_heading_horizontal) < f32(1e-6):
            heading = _v((internode_axis[0], internode_axis[1], 0))
            if V.magnitude(heading) > f32(1e-4):
                parent_shoot.initial_heading_horizontal = V.normalize(heading)

        shoot_bending_axis = V.cross(internode_axis, (0, 0, 1))
        internode_axis = V.normalize(internode_axis)
        if _vec_eq(internode_axis, (0, 0, 1)):
            shoot_bending_axis = _v((0, 1, 0))

        if pa.collision_detection_enabled or pa.attraction_points_enabled or pa.solid_obstacle_avoidance_enabled:
            raise NotImplementedError("collision / attraction-point / solid-obstacle guidance is not ported")

        self.internode_curvature_perturbations = [f32(0)] * Ndiv_internode_length
        self.internode_yaw_perturbations = [f32(0)] * Ndiv_internode_length
        self.internode_azimuthal_restoring_angles = [f32(0)] * Ndiv_internode_length
        self.internode_gravitropic_angles = [f32(0)] * Ndiv_internode_length

        # ---- internode segments: gravitropism + tortuosity (OU on curvature) + azimuthal restoring force
        for inode_segment in range(1, Ndiv_internode_length + 1):
            if ((abs(float(parent_shoot.gravitropic_curvature)) > 0 or parent_shoot_parameters.tortuosity.val() > 0)
                    and self.shoot_index[0] > 0):
                ds = dr_internode_max
                self.internode_curvature_arc_length_step = ds
                current_curvature_fact = f32(f32(0.5) - f32(internode_axis[2] / f32(2)))
                if internode_axis[2] < 0:
                    current_curvature_fact = f32(current_curvature_fact * f32(2))
                gravitropic_angle = f32(f32(parent_shoot.gravitropic_curvature * current_curvature_fact) * ds)
                persistence_length = parent_shoot_parameters.tortuosity_persistence_length.val()
                if persistence_length <= 0:
                    persistence_length = f32(0.5)
                tortuosity_amplitude = parent_shoot_parameters.tortuosity.val()
                ou_decay = f32(ds / persistence_length)
                ou_drive = f32(tortuosity_amplitude * f32(np.sqrt(f32(f32(2) * ou_decay))))
                cp = f32(parent_shoot.curvature_perturbation)
                parent_shoot.curvature_perturbation = f32(cp + f32(f32(-cp * ou_decay) + f32(ou_drive * f32(ctx.randn()))))
                yp = f32(parent_shoot.yaw_perturbation)
                parent_shoot.yaw_perturbation = f32(yp + f32(f32(-yp * ou_decay) + f32(ou_drive * f32(ctx.randn()))))

                azimuthal_restoring_angle = f32(0)
                if V.magnitude(parent_shoot.initial_heading_horizontal) > f32(1e-6):
                    current_heading = _v((internode_axis[0], internode_axis[1], 0))
                    if V.magnitude(current_heading) > f32(1e-4):
                        current_heading = V.normalize(current_heading)
                        target = parent_shoot.initial_heading_horizontal
                        cos_deviation = V.dot(current_heading, target)
                        cos_deviation = f32(-1) if cos_deviation < -1 else (f32(1) if cos_deviation > 1 else cos_deviation)
                        deviation = f32(f32(V.acosf(cos_deviation) * f32(180)) / f32(math.pi))
                        cross_z = f32(f32(current_heading[0] * target[1]) - f32(current_heading[1] * target[0]))
                        sense = f32(1) if cross_z >= 0 else f32(-1)
                        ratio = f32(ds / persistence_length)
                        azimuthal_restoring_angle = f32(f32(sense * deviation) * (ratio if ratio < 1 else f32(1)))

                curvature_perturbation_angle = f32(parent_shoot.curvature_perturbation * ds)
                yaw_perturbation_angle = f32(parent_shoot.yaw_perturbation * ds)
                self.internode_curvature_perturbations[inode_segment - 1] = curvature_perturbation_angle
                self.internode_yaw_perturbations[inode_segment - 1] = yaw_perturbation_angle
                self.internode_azimuthal_restoring_angles[inode_segment - 1] = azimuthal_restoring_angle
                self.internode_gravitropic_angles[inode_segment - 1] = gravitropic_angle

                bending_axis_normalized = shoot_bending_axis.copy()
                if V.magnitude(bending_axis_normalized) > f32(1e-4):
                    bending_axis_normalized = V.normalize(bending_axis_normalized)
                    bending_angle = deg2rad(f32(gravitropic_angle + curvature_perturbation_angle))
                    internode_axis = _rot(internode_axis, bending_axis_normalized, bending_angle)
                    transverse_axis = V.cross(internode_axis, bending_axis_normalized)
                    if V.magnitude(transverse_axis) > f32(1e-4):
                        transverse_axis = V.normalize(transverse_axis)
                        internode_axis = _rot(internode_axis, transverse_axis, deg2rad(yaw_perturbation_angle))
                    if abs(azimuthal_restoring_angle) > f32(1e-6):
                        internode_axis = _rot(internode_axis, (0, 0, 1), deg2rad(azimuthal_restoring_angle))
                    internode_axis = V.normalize(internode_axis)

            # solid obstacle avoidance / attraction / soft collision: all disabled (checked above)
            phytomer_internode_vertices[inode_segment] = (phytomer_internode_vertices[inode_segment - 1]
                                                          + _smul(dr_internode, internode_axis))
            phytomer_internode_radii[inode_segment] = internode_radius
            self.internode_colors[inode_segment] = pp.internode.color

        if self.shoot_index[0] == 0:
            parent_shoot.shoot_internode_vertices.append([v.copy() for v in phytomer_internode_vertices])
            parent_shoot.shoot_internode_radii.append(list(phytomer_internode_radii))
        else:
            parent_shoot.shoot_internode_vertices.append([v.copy() for v in phytomer_internode_vertices[1:]])
            parent_shoot.shoot_internode_radii.append(list(phytomer_internode_radii[1:]))

        # ---- internode Context geometry
        if build_context_geometry_internode:
            if pp.internode.image_texture:
                raise NotImplementedError("textured internode tubes are not ported (tomato has none)")
            plant_name = pa.plant_instances[self.plantID].plant_name
            stem_material_name = f"{plant_name}_{parent_shoot.shoot_type_label}_stem"
            if not ctx.does_object_exist(parent_shoot.internode_tube_objID):
                parent_shoot.internode_tube_objID = ctx.add_tube_object(Ndiv_internode_radius, phytomer_internode_vertices,
                                                                        phytomer_internode_radii, self.internode_colors)
            else:
                for inode_segment in range(1, Ndiv_internode_length + 1):
                    ctx.append_tube_segment(parent_shoot.internode_tube_objID, phytomer_internode_vertices[inode_segment],
                                            phytomer_internode_radii[inode_segment], self.internode_colors[inode_segment])
            ctx.set_primitive_data(ctx.object_primitive_uuids(parent_shoot.internode_tube_objID), "object_label", "shoot")
            name_organ_materials(pa, parent_shoot.internode_tube_objID, stem_material_name)

        # ---- petioles
        self.inflorescence_bending_axis = np.zeros(3, f32)
        self.axillary_vegetative_buds = []
        self.floral_buds = []
        for petiole in range(npet):
            petiole_axis = internode_axis.copy()
            if self.leaf_is_sessile:
                leaf_pitch_degrees = pp.leaf.pitch.val()
                pp.leaf.pitch.resample()
                if leaf_pitch_degrees > 0:
                    raise PlantArchitectureError(
                        f"ERROR (PlantArchitecture::Phytomer): Shoot type {parent_shoot.shoot_type_label} has leaves without "
                        f"a petiole (petiole length of 0) and a leaf pitch of {float(leaf_pitch_degrees)} degrees.")
                self.petiole_pitch[petiole] = f32(-deg2rad(leaf_pitch_degrees))
            elif self.shoot_index[1] + 1 == self.shoot_index[2]:
                self.petiole_pitch[petiole] = deg2rad(f32(5))
            else:
                self.petiole_pitch[petiole] = deg2rad(pp.petiole.pitch.val())
                pp.petiole.pitch.resample()
                if abs(self.petiole_pitch[petiole]) < deg2rad(f32(5)):
                    self.petiole_pitch[petiole] = deg2rad(f32(5))
            petiole_axis = _rot(petiole_axis, petiole_rotation_axis, abs(self.petiole_pitch[petiole]))

            if phytomer_index != 0 and self.internode_phyllotactic_angle != 0:
                petiole_axis = _rot(petiole_axis, internode_axis, self.internode_phyllotactic_angle)
                petiole_rotation_axis = _rot(petiole_rotation_axis, internode_axis, self.internode_phyllotactic_angle)

            if self.leaf_is_sessile:
                self.petiole_curvature[petiole] = f32(0)
            else:
                self.petiole_curvature[petiole] = pp.petiole.curvature.val()
                pp.petiole.curvature.resample()

            petiole_rotation_axis_actual = petiole_rotation_axis.copy()
            petiole_axis_actual = petiole_axis.copy()
            if petiole > 0:
                budrot = f32(f32(f32(f32(petiole) * f32(2)) * PI_F) / f32(npet))
                petiole_axis_actual = _rot(petiole_axis_actual, internode_axis, budrot)
                petiole_rotation_axis_actual = _rot(petiole_rotation_axis_actual, internode_axis, budrot)
            self.petiole_axis_initial[petiole] = petiole_axis_actual.copy()
            self.petiole_rotation_axis[petiole] = petiole_rotation_axis_actual.copy()

            if pa.petiole_collision_detection_enabled:
                raise NotImplementedError("petiole collision avoidance is not ported")

            self.petiole_taper[petiole] = pp.petiole.taper.val()
            pp.petiole.taper.resample()

            pv = self.petiole_vertices[petiole]
            pv[0] = phytomer_internode_vertices[-1].copy()
            radius_fraction = petiole_radius_fraction(leaf_scale_factor_fraction)
            for j in range(1, Ndiv_petiole_length + 1):
                if abs(float(self.petiole_curvature[petiole])) > 0:
                    petiole_axis_actual = _rot(petiole_axis_actual, petiole_rotation_axis_actual,
                                               f32(-deg2rad(f32(self.petiole_curvature[petiole] * dr_petiole_max[petiole]))))
                pv[j] = pv[j - 1] + _smul(dr_petiole[petiole], petiole_axis_actual)
                taper_term = f32(f32(1) - f32(f32(self.petiole_taper[petiole] / f32(Ndiv_petiole_length)) * f32(j)))
                self.petiole_radii[petiole][j] = f32(f32(radius_fraction * pp.petiole.radius.val()) * taper_term)
                self.petiole_colors[j] = pp.petiole.color
            self.recordPetioleRestShape(petiole)

            if self.build_context_geometry_petiole and not suppress_petiole_geometry[petiole]:
                self.petiole_objIDs[petiole] = make_petiole_tube(Ndiv_petiole_radius, pv, self.petiole_radii[petiole],
                                                                 self.petiole_colors, ctx)
                if ctx.does_object_exist(self.petiole_objIDs[petiole]):
                    ctx.set_primitive_data(ctx.object_primitive_uuids(self.petiole_objIDs[petiole]), "object_label", "petiole")
                    plant_name = pa.plant_instances[self.plantID].plant_name
                    name_organ_materials(pa, self.petiole_objIDs[petiole],
                                         f"{plant_name}_{parent_shoot.shoot_type_label}_petiole")

            # ---- buds (data only)
            nveg = int(pp.internode.max_vegetative_buds_per_petiole.val())
            pp.internode.max_vegetative_buds_per_petiole.resample()
            self.axillary_vegetative_buds.append([VegetativeBud() for _ in range(max(nveg, 0))])
            self.floral_buds.append(create_petiole_floral_buds(pp.internode.max_floral_buds_per_petiole, petiole,
                                                               len(self.axillary_vegetative_buds),
                                                               phytomer_internode_vertices[-1], shoot_bending_axis))

            # ---- leaves
            if pp.leaf.prototype.prototype_function is None:
                raise PlantArchitectureError("ERROR (PlantArchitecture::Phytomer): Leaf prototype function was not defined "
                                             f"for shoot type {parent_shoot.shoot_type_label}.")
            petiole_tip_axis = self.getPetioleAxisVector(f32(1), petiole)

            proto = pp.leaf.prototype
            assert proto.unique_prototype_identifier != 0
            key = (int(proto.unique_prototype_identifier), max(leaves_per_petiole, 0))
            if proto.unique_prototypes > 0 and key not in pa.unique_leaf_prototype_objIDs:
                self._build_leaf_prototype_cache(key, leaves_per_petiole)

            for leaf in range(max(leaves_per_petiole, 0)):
                self._place_leaflet(petiole, leaf, leaves_per_petiole, leaflet_offset_val, petiole_tip_axis,
                                    leaf_scale_factor_fraction)

            self.bendPetioleUnderLeafWeight(petiole)
            pp.leaf.prototype_scale.resample()

            self.inflorescence_bending_axis = V.cross(parent_internode_axis, petiole_axis_actual)
            if _vec_eq(self.inflorescence_bending_axis, (0, 0, 0)):
                self.inflorescence_bending_axis = _v((1, 0, 0))

        if npet == 0:
            nveg = int(pp.internode.max_vegetative_buds_per_petiole.val())
            pp.internode.max_vegetative_buds_per_petiole.resample()
            self.axillary_vegetative_buds.append([VegetativeBud() for _ in range(max(nveg, 0))])
            self.inflorescence_bending_axis = V.cross(
                parent_internode_axis, self.getPetioleAxisVectorOrGhost(0, internode_axis, phytomer_index))
            if V.magnitude(self.inflorescence_bending_axis) < f32(1e-6):
                self.inflorescence_bending_axis = _v((1, 0, 0))
            self.floral_buds.append(create_petiole_floral_buds(pp.internode.max_floral_buds_per_petiole, 0,
                                                               len(self.axillary_vegetative_buds),
                                                               phytomer_internode_vertices[-1], shoot_bending_axis))

        self.seated_node_position = phytomer_internode_vertices[-1].copy()

    # ---- leaf cache and placement ----------------------------------------------------------------

    def _build_leaf_prototype_cache(self, key, leaves_per_petiole: int):
        """PA:3460-3497: the unique blades from a private ``minstd_rand0(identifier)``, hidden."""
        from ..core.random import MinstdRand0
        pa, ctx, pp = self.plantarchitecture_ptr, self.context_ptr, self.phytomer_parameters
        proto = pp.leaf.prototype
        nproto = int(proto.unique_prototypes)
        pa.unique_leaf_prototype_objIDs[key] = [[] for _ in range(nproto)]
        pa.unique_leaf_prototype_rest_geometry[key] = [[] for _ in range(nproto)]
        prototype_generator = MinstdRand0(int(proto.unique_prototype_identifier))
        context_generator = proto.setRandomGenerator(prototype_generator)
        plant_name = pa.plant_instances[self.plantID].plant_name
        material_base_name = f"{plant_name}_{self.parent_shoot_ptr.shoot_type_label}_leaf"
        for prototype in range(nproto):
            for leaf in range(leaves_per_petiole):
                ind_from_tip = f32(f32(leaf) - f32(f32(leaves_per_petiole - 1) / f32(2)))
                objID_leaf = self.buildLeafPrototype(leaves_per_petiole, ind_from_tip, prototype)
                self.labelLeafPrototype(objID_leaf)
                pa.unique_leaf_prototype_objIDs[key][prototype].append(objID_leaf)
                record_leaf_prototype_rest_geometry(pa, proto, leaves_per_petiole, prototype, objID_leaf,
                                                    self.leaf_flexibility)
                name_organ_materials(pa, objID_leaf, material_base_name)
                ctx.hide_object(objID_leaf)
        proto.setRandomGenerator(context_generator)

    def _place_leaflet(self, petiole: int, leaf: int, leaves_per_petiole: int, leaflet_offset_val, petiole_tip_axis,
                       leaf_scale_factor_fraction):
        """One leaflet of the constructor's leaf loop (PA:3499-3570): choose a cached prototype (1 ``randu``),
        ``copyObject`` it, ``scaleObject``, sample roll/pitch/yaw, ``orientLeaf``, ``translateObject``."""
        pa, ctx, pp = self.plantarchitecture_ptr, self.context_ptr, self.phytomer_parameters
        proto = pp.leaf.prototype
        ind_from_tip = f32(f32(leaf) - f32(f32(leaves_per_petiole - 1) / f32(2)))
        leaf_prototype_source = -1
        if proto.unique_prototypes > 0:
            prototype = int(ctx.randu(0, int(proto.unique_prototypes) - 1))
            leaf_prototype_source = prototype
            objID_leaf = ctx.copy_object(get_cached_leaf_prototype_objID(
                pa, proto.unique_prototype_identifier, leaves_per_petiole, prototype, leaf, "PlantArchitecture::Phytomer"))
        else:
            raise NotImplementedError("unique_prototypes == 0 (a fresh blade per leaf) is not ported")

        if leaves_per_petiole > 0:
            self.leaf_size_max[petiole][leaf] = leaflet_size_max(ind_from_tip, pp.leaf.leaflet_scale.val(),
                                                                 pp.leaf.intercalary_leaflet_scale.val(),
                                                                 pp.leaf.prototype_scale.val())
        else:
            self.leaf_size_max[petiole][leaf] = pp.leaf.prototype_scale.val()
        s = f32(f32(leaf_scale_factor_fraction) * self.leaf_size_max[petiole][leaf])
        ctx.scale_object(objID_leaf, (s, s, s))

        compound_rotation = compound_leaf_rotation(leaves_per_petiole, leaf, leaflet_offset_val)

        leaf_roll_angle = deg2rad(pp.leaf.roll.val())
        pp.leaf.roll.resample()
        leaf_pitch_angle = f32(0)
        if not self.leaf_is_sessile:
            leaf_pitch_angle = deg2rad(pp.leaf.pitch.val())
            pp.leaf.pitch.resample()
        leaf_yaw_angle = f32(0)
        if ind_from_tip != 0:
            # -compound_rotation / fabs(compound_rotation): C fabs -> double, exact +-1 either way
            yaw_sign = f32(-float(compound_rotation) / abs(float(compound_rotation)))
            leaf_yaw_angle = f32(yaw_sign * deg2rad(pp.leaf.yaw.val()))
            pp.leaf.yaw.resample()

        self.orientLeaf(objID_leaf, petiole, leaf, leaves_per_petiole, ind_from_tip, compound_rotation, petiole_tip_axis,
                        leaf_roll_angle, leaf_pitch_angle, leaf_yaw_angle)

        leaf_base = self.petiole_vertices[petiole][-1].copy()
        if leaves_per_petiole > 1 and leaflet_offset_val > 0:
            if ind_from_tip != 0:
                # (fabs(ind) - 0.5f) * offset * length: C fabs promotes the product to double
                length = pp.petiole.length.val()
                offset = f32((abs(float(ind_from_tip)) - 0.5) * float(leaflet_offset_val) * float(length))
                leaf_base = interpolate_tube(self.petiole_vertices[petiole], f32(f32(1) - f32(offset / length)))
        ctx.translate_object(objID_leaf, leaf_base)

        self.leaf_objIDs[petiole].append(objID_leaf)
        self.leaf_bases[petiole].append(np.asarray(leaf_base, f32).copy())
        self.leaf_prototype_index[petiole].append(leaf_prototype_source)
        self.leaf_pose_prescribed[petiole].append(False)
        self.leaf_angle_steering[petiole].append(None)
        self.leaf_last_deformed_scale[petiole].append(f32(-1))

    @ports("Phytomer::buildLeafPrototype")
    def buildLeafPrototype(self, leaves_per_petiole: int, ind_from_tip, prototype_index: int, leaflet_size_own=0.0,
                           petiole_radius_own=0.0, petiolule_length_own=0.0) -> int:
        pa, pp = self.plantarchitecture_ptr, self.phytomer_parameters
        prototype = pp.leaf.prototype
        prototype.validateTextureSeries(self.parent_shoot_ptr.shoot_type_label)
        prototype.texture_series_selector = prototype_index
        leaflet_size_own, petiole_radius_own = f32(leaflet_size_own), f32(petiole_radius_own)
        petiolule_length_own = f32(petiolule_length_own)
        if prototype.build_petiolule and leaflet_size_own > 0 and petiole_radius_own > 0:
            prototype.petiolule_radius_fraction = f32(petiole_radius_own / leaflet_size_own)
        elif prototype.build_petiolule:
            label = self.parent_shoot_ptr.shoot_type_label
            if label not in pa.shoot_types:
                raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::buildLeafPrototype): Shoot type '{label}' "
                                             "is not defined, but its leaf prototype builds a petiolule whose radius is "
                                             "taken from the shoot type's petiole.")
            tp = pa.shoot_types[label].phytomer_parameters
            attachment_fraction = f32(1)
            offset_val = clamp_offset(leaves_per_petiole, distribution_mean(tp.leaf.leaflet_offset))
            if leaves_per_petiole > 1 and offset_val > 0 and ind_from_tip != 0:
                attachment_fraction = f32(f32(1) - f32(f32(abs(f32(ind_from_tip)) - f32(0.5)) * offset_val))
            petiole_radius = f32(distribution_mean(tp.petiole.radius)
                                 * f32(f32(1) - f32(distribution_mean(tp.petiole.taper) * attachment_fraction)))
            leaflet_size = leaflet_size_max(ind_from_tip, distribution_mean(tp.leaf.leaflet_scale),
                                            distribution_mean(tp.leaf.intercalary_leaflet_scale),
                                            distribution_mean(tp.leaf.prototype_scale))
            if petiole_radius <= 0 or leaflet_size <= 0:
                raise PlantArchitectureError("ERROR (PlantArchitecture::Phytomer::buildLeafPrototype): petiolule radius "
                                             "or leaflet size is not positive.")
            prototype.petiolule_radius_fraction = f32(petiole_radius / leaflet_size)
        petiolule_length_species = dict(prototype.petiolule_length)
        own_stalk = bool(prototype.build_petiolule and petiolule_length_own > 0 and leaflet_size_own > 0)
        if own_stalk:
            prototype.petiolule_length = {0: f32(petiolule_length_own / leaflet_size_own)}
        prototype.last_built_shape_valid = False
        fn = _resolve_prototype_function(pa, prototype.prototype_function)
        objID_leaf = fn(self.context_ptr, prototype, int(ind_from_tip))
        if own_stalk:
            prototype.petiolule_length = petiolule_length_species
        self.last_built_leaf_shape_valid = bool(prototype.last_built_shape_valid)
        if self.last_built_leaf_shape_valid:
            self.last_built_leaf_shape = prototype.last_built_shape.copy()
        return objID_leaf

    @ports("Phytomer::labelLeafPrototype")
    def labelLeafPrototype(self, objID_leaf: int):
        ctx = self.context_ptr
        if self.phytomer_parameters.leaf.prototype.prototype_function is hooks.GenericLeafPrototype:
            uu = ctx.object_primitive_uuids(objID_leaf)
            keep = set(ctx.filter_primitives_by_data(uu, "object_label", "petiolule"))
            blade = [u for u in uu if u not in keep]
            if blade:
                ctx.set_primitive_data(blade, "object_label", "leaf")
        self.assignPetioluleMaterial(objID_leaf)

    @ports("Phytomer::assignPetioluleMaterial")
    def assignPetioluleMaterial(self, objID_leaf: int):
        ctx, pa = self.context_ptr, self.plantarchitecture_ptr
        petiolule = ctx.filter_primitives_by_data(ctx.object_primitive_uuids(objID_leaf), "object_label", "petiolule")
        if not petiolule:
            return
        material_label = ""
        for objID_petiole in self.petiole_objIDs:
            if ctx.does_object_exist(objID_petiole) and ctx.object_primitive_uuids(objID_petiole):
                material_label = ctx.get_primitive_material_label(ctx.object_primitive_uuids(objID_petiole)[0])
                break
        if not material_label:
            material_label = (f"{pa.plant_instances[self.plantID].plant_name}_"
                              f"{self.parent_shoot_ptr.shoot_type_label}_petiole")
            if not ctx.does_material_exist(material_label):
                ctx.add_material(material_label)
                c = self.phytomer_parameters.petiole.color
                rgba = np.array([c[0], c[1], c[2], 1], f32)
                ctx.set_material_color(material_label, rgba)
                pa.organ_material_labels[("", material_label)] = OrganMaterial(material_label, rgba, "", False)
        ctx.assign_material_to_primitive(petiolule, material_label)

    # ---- orientation ----------------------------------------------------------------------------

    @ports("Phytomer::orientLeaf")
    def orientLeaf(self, objID_leaf, petiole_index, leaf_index, leaves_per_petiole, ind_from_tip, compound_rotation,
                   petiole_tip_axis, leaf_roll_angle, leaf_pitch_angle, leaf_yaw_angle, prescribed=False,
                   terminal_roll_angle=0.0):
        ctx = self.context_ptr
        ind_from_tip, compound_rotation = f32(ind_from_tip), f32(compound_rotation)
        leaf_roll_angle, leaf_pitch_angle, leaf_yaw_angle = f32(leaf_roll_angle), f32(leaf_pitch_angle), f32(leaf_yaw_angle)
        terminal_roll_angle = f32(terminal_roll_angle)
        tip = _v(petiole_tip_axis)
        rec = self.leaf_rotation[petiole_index][leaf_index]

        if prescribed:
            rec.roll, rec.pitch, rec.yaw = leaf_roll_angle, leaf_pitch_angle, leaf_yaw_angle
            tip_axis = V.normalize(tip)
            elevation = V.asin_safe(tip_axis[2])
            horizontal = f32(np.sqrt(f32(f32(tip_axis[0] * tip_axis[0]) + f32(tip_axis[1] * tip_axis[1]))))
            azimuth = f32(0) if horizontal < f32(1e-6) else atan2f(tip_axis[1], tip_axis[0])
            ctx.rotate_object(objID_leaf, leaf_roll_angle, "x")
            ctx.rotate_object(objID_leaf, f32(-leaf_pitch_angle), "y")
            ctx.rotate_object(objID_leaf, leaf_yaw_angle, "z")
            ctx.rotate_object(objID_leaf, compound_rotation, "z")
            ctx.rotate_object(objID_leaf, f32(-elevation), "y")
            ctx.rotate_object(objID_leaf, azimuth, "z")
            return

        if self.leaf_is_sessile:
            sign = 1 if self.shoot_index[0] % 2 == 0 else -1
            rec.roll, rec.pitch, rec.yaw = leaf_roll_angle, leaf_pitch_angle, f32(0)
            ctx.rotate_object(objID_leaf, f32(f32(-leaf_roll_angle) * f32(sign)), "x")
            ctx.rotate_object(objID_leaf, f32(-leaf_pitch_angle), "y")
            leaf_axis = V.normalize(tip)
            stem_axis = V.normalize(self.getInternodeAxisVector(f32(1)))
            toward_stem = stem_axis - _smul(V.dot(stem_axis, leaf_axis), leaf_axis)
            if V.magnitude(toward_stem) < f32(1e-4):
                toward_stem = _rot(leaf_axis, self.petiole_rotation_axis[petiole_index], f32(f32(-0.5) * PI_F))
            toward_stem = V.normalize(toward_stem)
            blade_axis = _v((1, 0, 0))
            blade_upper_face = _v((0, 0, 1))
            alignment_axis = V.cross(blade_axis, leaf_axis)
            alignment_angle = V.acos_safe(V.dot(blade_axis, leaf_axis))
            if V.magnitude(alignment_axis) < f32(1e-6):
                alignment_axis = blade_upper_face.copy()
            alignment_axis = V.normalize(alignment_axis)
            if alignment_angle != 0:
                ctx.rotate_object(objID_leaf, alignment_angle, alignment_axis)
                blade_upper_face = _rot(blade_upper_face, alignment_axis, alignment_angle)
            face_angle = atan2f(V.dot(V.cross(blade_upper_face, toward_stem), leaf_axis), V.dot(blade_upper_face, toward_stem))
            if face_angle != 0:
                ctx.rotate_object(objID_leaf, face_angle, leaf_axis)
            return

        petiole_elevation = V.asin_safe(tip[2])
        petiole_azimuth = atan2f(tip[1], tip[0])
        posed_past_vertical = False
        stem_axis = V.normalize(self.getInternodeAxisVector(f32(1)))
        if stem_axis[2] > V.cosf(f32(f32(0.25) * PI_F)):
            leaf_side = tip - _smul(V.dot(tip, stem_axis), stem_axis)
            leaf_side_horizontal = _v((leaf_side[0], leaf_side[1], 0))
            petiole_horizontal = _v((tip[0], tip[1], 0))
            if V.magnitude(leaf_side_horizontal) > f32(1e-6) and V.dot(petiole_horizontal, leaf_side_horizontal) < 0:
                posed_past_vertical = True
                petiole_elevation = f32(PI_F - petiole_elevation)
                petiole_azimuth = f32(petiole_azimuth + PI_F)

        if leaves_per_petiole == 1:
            sign = 1 if self.shoot_index[0] % 2 == 0 else -1
            roll_rot = f32(f32(-leaf_roll_angle) * f32(sign))
        elif ind_from_tip != 0:
            roll_rot = f32(f32(f32(leaf_roll_angle - petiole_elevation) * compound_rotation) / abs(compound_rotation))
        else:
            roll_rot = terminal_roll_angle
        rec.roll = leaf_roll_angle
        ctx.rotate_object(objID_leaf, roll_rot, "x")

        rec.pitch = leaf_pitch_angle
        pitch_rot = leaf_pitch_angle
        if ind_from_tip == 0:
            pitch_rot = f32(pitch_rot + petiole_elevation)
        ctx.rotate_object(objID_leaf, f32(-pitch_rot), "y")

        rec.yaw = leaf_yaw_angle
        if leaf_yaw_angle != 0:
            ctx.rotate_object(objID_leaf, leaf_yaw_angle, "z")

        ctx.rotate_object(objID_leaf, f32(petiole_azimuth + compound_rotation), "z")

        if leaves_per_petiole == 1:
            sign = 1 if self.shoot_index[0] % 2 == 0 else -1
            r_h = f32(np.sqrt(f32(f32(tip[0] * tip[0]) + f32(tip[1] * tip[1]))))
            if r_h > f32(1e-4):
                blade_correction = atan2f(f32(tip[2] * r_h), f32(r_h * r_h))
                if posed_past_vertical:
                    blade_correction = f32(PI_F - blade_correction)
                petiole_len = self.petiole_length[petiole_index]
                ref = self.leaf_size_max[petiole_index][leaf_index]
                leaf_size_ref = ref if ref > f32(1e-6) else f32(1e-6)
                ratio = f32(petiole_len / leaf_size_ref)
                length_ratio = ratio if ratio < 1 else f32(1)
                blade_correction = f32(blade_correction * length_ratio)
                max_correction = f32(f32(f32(0.5) * PI_F) - deg2rad(f32(1)))
                if blade_correction > max_correction:
                    blade_correction = max_correction
                if blade_correction < -max_correction:
                    blade_correction = f32(-max_correction)
                ctx.rotate_object(objID_leaf, f32(blade_correction * f32(sign)), tip)

    compoundLeafRotation = staticmethod(compound_leaf_rotation)

    # ---- geometry queries -----------------------------------------------------------------------

    @ports("Phytomer::getInternodeNodePositions")
    def getInternodeNodePositions(self) -> list:
        s = self.parent_shoot_ptr
        nodes = [v.copy() for v in s.shoot_internode_vertices[self.shoot_index[0]]]
        if self.shoot_index[0] > 0:
            nodes.insert(0, s.shoot_internode_vertices[self.shoot_index[0] - 1][-1].copy())
        return nodes

    @ports("Phytomer::getInternodeNodeRadii")
    def getInternodeNodeRadii(self) -> list:
        s = self.parent_shoot_ptr
        radii = list(s.shoot_internode_radii[self.shoot_index[0]])
        if self.shoot_index[0] > 0:
            radii.insert(0, s.shoot_internode_radii[self.shoot_index[0] - 1][-1])
        return radii

    @ports("Phytomer::getInternodeAxisVector")
    def getInternodeAxisVector(self, stem_fraction) -> np.ndarray:
        return get_axis_vector(stem_fraction, self.getInternodeNodePositions())

    @ports("Phytomer::getPetioleAxisVector")
    def getPetioleAxisVector(self, stem_fraction, petiole_index: int) -> np.ndarray:
        if petiole_index >= len(self.petiole_vertices):
            raise PlantArchitectureError("ERROR (Phytomer::getPetioleAxisVector): Petiole index out of range.")
        return get_axis_vector(stem_fraction, self.petiole_vertices[petiole_index])

    @ports("Phytomer::getPetioleAxisVectorOrGhost")
    def getPetioleAxisVectorOrGhost(self, petiole_index: int, internode_axis, phytomer_index: int) -> np.ndarray:
        if self.petiole_vertices:
            return self.getPetioleAxisVector(f32(0), petiole_index)
        ghost = V.cross(internode_axis, (0, 0, 1))
        if V.magnitude(ghost) < f32(0.01):
            ghost = _v((0, 1, 0))
        ghost = V.normalize(ghost)
        return _rot(ghost, internode_axis, self.internode_phyllotactic_azimuth)

    @ports("Phytomer::getInternodeLength")
    def getInternodeLength(self) -> np.float32:
        nodes = self.getInternodeNodePositions()
        length = f32(0)
        for i in range(len(nodes) - 1):
            length = f32(length + V.magnitude(nodes[i + 1] - nodes[i]))
        return length

    @ports("Phytomer::getInternodeRadius")
    def getInternodeRadius(self) -> np.float32:
        return self.parent_shoot_ptr.shoot_internode_radii[self.shoot_index[0]][0]

    @ports("Phytomer::getLeafArea")
    def getLeafArea(self) -> np.float32:
        leaf_area = f32(0)
        for p, petiole in enumerate(self.leaf_objIDs):
            for objID in petiole:
                if self.context_ptr.does_object_exist(objID):
                    obj_area = get_leaf_blade_area(self.plantarchitecture_ptr, objID)
                    sf = self.current_leaf_scale_factor[p]
                    leaf_area = f32(leaf_area + f32(obj_area / f32(sf * sf)))     # powi(sf, 2)
        return leaf_area

    @ports("Phytomer::getLeafShape")
    def getLeafShape(self, petiole_index: int, leaf_index: int) -> LeafShapeValues:
        if petiole_index >= len(self.leaf_objIDs) or leaf_index >= len(self.leaf_objIDs[petiole_index]):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::getLeafShape): Leaf {leaf_index} of petiole "
                                         f"{petiole_index} does not exist on this phytomer.")
        objID = self.leaf_objIDs[petiole_index][leaf_index]
        if objID in self.leaf_shape_state:
            return self.leaf_shape_state[objID]
        prototype = self.leaf_prototype_index[petiole_index][leaf_index]
        if prototype >= 0:
            key = (int(self.phytomer_parameters.leaf.prototype.unique_prototype_identifier),
                   len(self.leaf_prototype_index[petiole_index]))
            cache = self.plantarchitecture_ptr.unique_leaf_prototype_rest_geometry.get(key)
            if cache is not None and len(cache) > prototype and len(cache[prototype]) > leaf_index \
                    and cache[prototype][leaf_index].has_shape:
                return cache[prototype][leaf_index].shape
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::Phytomer::getLeafShape): The blade shape of leaf object "
                                     f"{objID} was not recorded.")

    # ---- petiole rest shape / bending ---------------------------------------------------------

    @ports("Phytomer::recordPetioleRestShape")
    def recordPetioleRestShape(self, petiole_index: int):
        if petiole_index >= len(self.petiole_vertices):
            raise PlantArchitectureError("ERROR (PlantArchitecture::Phytomer::recordPetioleRestShape): Petiole index out "
                                         "of range.")
        while len(self.petiole_rest_offsets) < len(self.petiole_vertices):
            self.petiole_rest_offsets.append([])
        while len(self.petiole_bend_state) < len(self.petiole_vertices):
            self.petiole_bend_state.append(np.array([-1, 0, 0], f32))
        nodes = self.petiole_vertices[petiole_index]
        self.petiole_rest_offsets[petiole_index] = [np.asarray(n, f32) - np.asarray(nodes[0], f32) for n in nodes]
        self.petiole_bend_state[petiole_index] = np.array([-1, 0, 0], f32)

    @ports("Phytomer::bendPetioleUnderLeafWeight")
    def bendPetioleUnderLeafWeight(self, petiole_index: int):
        if self.petiole_flexibility <= 0:
            return
        raise NotImplementedError("bendPetioleUnderLeafWeight for a flexible petiole is not ported (tomato is rigid)")

    # ---- buds -----------------------------------------------------------------------------------

    @ports("Phytomer::setVegetativeBudState")
    def setVegetativeBudState(self, state, *where):
        if not where:
            for petiole in self.axillary_vegetative_buds:
                for bud in petiole:
                    bud.state = state
            return
        if len(where) == 1:
            where[0].state = state
            return
        petiole_index, bud_index = where
        if petiole_index >= len(self.axillary_vegetative_buds):
            raise PlantArchitectureError("ERROR (Phytomer::setVegetativeBudState): Petiole index out of range.")
        if bud_index >= len(self.axillary_vegetative_buds[petiole_index]):
            raise PlantArchitectureError("ERROR (Phytomer::setVegetativeBudState): Bud index out of range.")
        self.axillary_vegetative_buds[petiole_index][bud_index].state = state

    @ports("Phytomer::setFloralBudState")
    def setFloralBudState(self, state, *where):
        if not where:
            for petiole in self.floral_buds:
                for fbud in petiole:
                    if not fbud.isterminal:
                        self._set_floral_bud_state(state, fbud)
            return
        if len(where) == 1:
            self._set_floral_bud_state(state, where[0])
            return
        petiole_index, bud_index = where
        if petiole_index >= len(self.floral_buds):
            raise PlantArchitectureError("ERROR (Phytomer::setFloralBudState): Petiole index out of range.")
        if bud_index >= len(self.floral_buds[petiole_index]):
            raise PlantArchitectureError("ERROR (Phytomer::setFloralBudState): Bud index out of range.")
        self._set_floral_bud_state(state, self.floral_buds[petiole_index][bud_index])

    def _set_floral_bud_state(self, state, fbud):
        """PA:1491-1547 (carbon model off). Reproductive states rebuild the inflorescence (W3-D)."""
        pa, ctx = self.plantarchitecture_ptr, self.context_ptr
        pa.invalidateOrganIndex()
        if fbud.state == state:
            return
        if state in (BUD_DORMANT, BUD_ACTIVE):
            fbud.state = state
            return
        if pa.carbon_model_enabled:
            raise NotImplementedError("carbohydrate model is not ported")
        ctx.delete_object(list(fbud.inflorescence_objIDs))
        fbud.inflorescence_objIDs = []
        fbud.inflorescence_bases = []
        fbud.inflorescence_rotation = []
        fbud.inflorescence_base_scales = []
        if self.build_context_geometry_peduncle:
            ctx.delete_object(list(fbud.peduncle_objIDs))
            fbud.peduncle_objIDs = []
        fbud.contact_swing_axis = (0, 0, 1)
        fbud.contact_swing_angle = 0
        fbud.state = state
        if state != BUD_DEAD:
            raise NotImplementedError("updateInflorescence (flower/fruit geometry) is W3-D")

    # ---- scaling called by phytomer creation functions -------------------------------------------

    @ports("Phytomer::scaleLeafPrototypeScale(uint,float)", "Phytomer::scaleLeafPrototypeScale(float)")
    def scaleLeafPrototypeScale(self, *args):
        if len(args) == 1:
            for petiole_index in range(len(self.leaf_objIDs)):
                self.scaleLeafPrototypeScale(petiole_index, args[0])
            return
        petiole_index, scale_factor = args
        if len(self.leaf_objIDs) <= petiole_index:
            raise PlantArchitectureError("ERROR (PlantArchitecture::Phytomer): Invalid petiole index for leaf prototype "
                                         "scale.")
        scale_factor = f32(scale_factor)
        if scale_factor < 0:
            scale_factor = f32(0)
        for leaf in range(len(self.leaf_objIDs[petiole_index])):
            self.leaf_size_max[petiole_index][leaf] = f32(self.leaf_size_max[petiole_index][leaf] * scale_factor)
            self.context_ptr.scale_object_about_point(self.leaf_objIDs[petiole_index][leaf],
                                                      (scale_factor, scale_factor, scale_factor),
                                                      self.leaf_bases[petiole_index][leaf])

    @ports("Phytomer::scaleInternodeMaxLength")
    def scaleInternodeMaxLength(self, scale_factor):
        scale_factor = f32(scale_factor)
        self.internode_length_max = f32(self.internode_length_max * scale_factor)
        self.current_internode_scale_factor = f32(self.current_internode_scale_factor / scale_factor)
        if self.current_internode_scale_factor >= 1:
            self.setInternodeLengthScaleFraction(f32(1), True)
            self.current_internode_scale_factor = f32(1)

    def setInternodeLengthScaleFraction(self, internode_scale_factor_fraction, update_context_geometry):
        raise NotImplementedError("Phytomer::setInternodeLengthScaleFraction is W3-C (growth)")


__all__ = ["Phytomer", "create_petiole_floral_buds", "name_organ_materials", "get_leaf_blade_area",
           "stub_generic_leaf_prototype", "record_leaf_prototype_rest_geometry", "get_cached_leaf_prototype_objID",
           "OrganMaterial", "LeafRestGeometry", "atan2f"]
