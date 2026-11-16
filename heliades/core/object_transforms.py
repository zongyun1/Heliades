# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context.cpp, Context_object.cpp, Context_primitive.cpp, global.cpp;
# Copyright (C) 2016-2026 Brian Bailey).
"""Copy, delete and object-level translate/rotate/scale (mixed into Scene).

Helios keeps every compound object as (member primitives, a 4x4 object transform, an origin, and
type-specific data stored in the object's *local* frame, e.g. Tube nodes and radii). An object
transform left-multiplies the same matrix onto the object transform and onto each member
primitive's transform; the type-specific data is never touched, its getters apply the object
transform on read. The functions below reproduce that, float32 op for op.
"""
from __future__ import annotations

import copy

import numpy as np

from . import vec as V
from .primitives import PATCH, VOXEL, make_rotation_matrix, make_translation_matrix, matmult
from ..migration_registry import ports

f32 = np.float32

_OBJECT_TILE, _OBJECT_TUBE, _OBJECT_POLYMESH = 0, 2, 5   # helios::ObjectType (objects.py order)


# ---- float32 matrix helpers (global.cpp) -----------------------------------------------------

def _identity() -> np.ndarray:
    return np.eye(4, dtype=f32).reshape(16).copy()


def _axis_rotation_entries(rotation, axis, T: np.ndarray):
    """The nine Rodrigues entries shared by both vec3 makeRotationMatrix overloads."""
    u = V.normalize(V.v3(axis))                       # vec3::normalize (member, guarded)
    ux, uy, uz = u[0], u[1], u[2]
    sx, cx = V.sin_d(rotation), V.cos_d(rotation)  # global.cpp: float sx = sin(rotation) (double sin)
    omc = f32(f32(1.0) - cx)
    T[0] = cx + f32(ux * ux) * omc
    T[1] = f32(f32(ux * uy) * omc) - f32(uz * sx)
    T[2] = f32(f32(ux * uz) * omc) + f32(uy * sx)
    T[3] = 0
    T[4] = f32(f32(uy * ux) * omc) + f32(uz * sx)
    T[5] = cx + f32(uy * uy) * omc
    T[6] = f32(f32(uy * uz) * omc) - f32(ux * sx)
    T[7] = 0
    T[8] = f32(f32(uz * ux) * omc) - f32(uy * sx)
    T[9] = f32(f32(uz * uy) * omc) + f32(ux * sx)
    T[10] = cx + f32(uz * uz) * omc
    T[11] = 0


@ports("helios::makeRotationMatrix(float,const vec3&)")
def make_rotation_matrix_axis(rotation, axis) -> np.ndarray:
    """Rotation by ``rotation`` radians about ``axis`` through the world origin."""
    T = np.zeros(16, f32)
    _axis_rotation_entries(rotation, axis, T)
    T[15] = 1
    return T


@ports("helios::makeRotationMatrix(float,const vec3&,const vec3&)")
def make_rotation_matrix_about(rotation, origin, axis) -> np.ndarray:
    """Rotation about the line through ``origin`` along ``axis``: Ttrans(+o) * (Trot * Ttrans(-o))."""
    o = V.v3(origin)
    Ttrans = _identity()
    Ttrans[3], Ttrans[7], Ttrans[11] = -o[0], -o[1], -o[2]
    Trot = _identity()
    _axis_rotation_entries(rotation, axis, Trot)
    T = matmult(Trot, Ttrans)
    Ttrans[3], Ttrans[7], Ttrans[11] = o[0], o[1], o[2]
    return matmult(Ttrans, T)


@ports("helios::makeScaleMatrix(const vec3&,const vec3&)")
def make_scale_matrix_about_point(scale, point) -> np.ndarray:
    s, p = V.v3(scale), V.v3(point)
    T = np.zeros(16, f32)
    T[0], T[5], T[10], T[15] = s[0], s[1], s[2], 1
    T[3] = p[0] * f32(f32(1) - s[0])
    T[7] = p[1] * f32(f32(1) - s[1])
    T[11] = p[2] * f32(f32(1) - s[2])
    return T


def _apply_translation(shift, M: np.ndarray) -> np.ndarray:
    """applyTranslationToMatrix (Context_object.cpp) / translateObjectPrimitives_private: add the
    shift to the translation column of an affine matrix, else fall back on T*M."""
    s = V.v3(shift)
    if M[12] == 0 and M[13] == 0 and M[14] == 0 and M[15] == 1:
        M = M.copy()
        M[3] = M[3] + s[0]
        M[7] = M[7] + s[1]
        M[11] = M[11] + s[2]
        return M
    return matmult(make_translation_matrix(s), M)


def _xform_point(T, p) -> np.ndarray:
    """T[0]*x + T[1]*y + T[2]*z + T[3] per row (Tube::getNodes, Tile::getVertices, vecmult)."""
    x, y, z = V.v3(p)
    return np.array([f32(f32(f32(T[4 * r] * x) + f32(T[4 * r + 1] * y)) + f32(T[4 * r + 2] * z)) + T[4 * r + 3]
                     for r in range(3)], f32)


def _ids(x) -> list[int]:
    return [int(x)] if np.isscalar(x) else [int(i) for i in x]


class ObjectTransformsMixin:
    """Context copy/delete/transform API for primitives and compound objects."""

    # ---- helpers -----------------------------------------------------------------------------

    def _object(self, oid: int, who: str):
        o = self.objects.get(int(oid))
        if o is None:
            raise ValueError(f"{who}: Object ID of {oid} not found in the context.")
        return o

    def _transform_locked(self, p) -> bool:
        """Primitive::isTransformLockedByParentObject: members of a non-deformable object (anything
        but a Polymesh) ignore individual transforms."""
        o = self.objects.get(p.parent_object) if p.parent_object != 0 else None
        return o is not None and o.type != _OBJECT_POLYMESH

    def _left_multiply_members(self, obj, T):
        obj.transform = matmult(T, obj.transform)
        for u in obj.uuids:
            p = self.primitives.get(u)
            if p is not None:
                p.transform = matmult(T, p.transform)

    # ---- primitives --------------------------------------------------------------------------

    @ports("Context::copyPrimitive(uint)", "Context::copyPrimitive(const std::vector<uint>&)")
    def copy_primitive(self, uuids):
        """Copy one primitive (int -> int) or several (list -> list), in order. The copy keeps every
        Primitive field and its primitive data, but belongs to no object."""
        if np.isscalar(uuids):
            return self._copy_primitive_one(int(uuids))
        return [self._copy_primitive_one(int(u)) for u in uuids]

    def _copy_primitive_one(self, uuid: int) -> int:
        src = self.primitives.get(uuid)
        if src is None:
            raise ValueError(f"Context::copyPrimitive: UUID of {uuid} not found in the context.")
        new = self.current_uuid
        p = copy.deepcopy(src)                            # every field, including ones added later
        p.uuid, p.parent_object = new, 0
        if p.type == PATCH and getattr(src, "texture_file", None):
            # A textured patch is rebuilt from solid_fraction = getArea() / (size.x * size.y), which is
            # not always bit-identical to the source's stored fraction.
            sz = src.size()
            p.solid_fraction = float(f32(src.area() / f32(sz[0] * sz[1])))
        self.primitives[new] = p
        self.copy_primitive_data(uuid, new)
        self.current_uuid += 1
        return new

    @ports("Context::copyPrimitiveData")
    def copy_primitive_data(self, src: int, dst: int):
        """Replace all of dst's primitive data by (deep copies of) src's."""
        for d in self.primitive_data.values():
            d.pop(dst, None)
            if src in d:
                d[dst] = copy.deepcopy(d[src])

    @ports("Context::deletePrimitive(const std::vector<uint>&)", "Context::deletePrimitive(uint)")
    def delete_primitives(self, uuids):
        """Delete primitives, detaching them from their parent objects first; an object left with no
        members is deleted too. Raises on a UUID that does not exist (after deleting those before it,
        as Helios does)."""
        uuids = _ids(uuids)
        by_parent: dict[int, list[int]] = {}
        for u in uuids:
            p = self.primitives.get(u)
            if p is None:
                continue
            if p.parent_object != 0 and p.parent_object in self.objects:
                by_parent.setdefault(p.parent_object, []).append(u)
        for oid in sorted(by_parent):                     # std::map iteration order
            obj = self.objects[oid]
            gone = set(by_parent[oid])
            obj.uuids = [u for u in obj.uuids if u not in gone]   # CompoundObject::deleteChildPrimitive
            for u in by_parent[oid]:
                self.primitives[u].parent_object = 0
            if not obj.uuids:
                del self.objects[oid]
        for u in uuids:
            p = self.primitives.get(u)
            if p is None:
                raise ValueError(f"Context::deletePrimitive: UUID of {u} not found in the context.")
            if p.parent_object != 0 and p.parent_object in self.objects:
                obj = self.objects[p.parent_object]
                obj.uuids = [x for x in obj.uuids if x != u]
                if not obj.uuids:
                    del self.objects[p.parent_object]
            del self.primitives[u]
            for d in self.primitive_data.values():
                d.pop(u, None)

    @ports("Context::rotatePrimitive(uint,float,const vec3&)", "Context::rotatePrimitive(const std::vector<uint>&,float,const vec3&)",
           "Context::rotatePrimitive(uint,float,const vec3&,const vec3&)",
           "Context::rotatePrimitive(const std::vector<uint>&,float,const vec3&,const vec3&)",
           "Patch::rotate(float,const vec3&)", "Patch::rotate(float,const vec3&,const vec3&)",
           "Triangle::rotate(float,const vec3&)", "Triangle::rotate(float,const vec3&,const vec3&)")
    def rotate_primitive_about_axis(self, uuids, radians, axis, origin=None):
        """Rotate primitives about ``axis`` through the world origin, or through ``origin`` if given.
        Members of a non-deformable object and voxels are left unchanged, as in Helios."""
        if f32(radians) == 0:
            return
        T = make_rotation_matrix_axis(radians, axis) if origin is None else make_rotation_matrix_about(radians, origin, axis)
        for u in _ids(uuids):
            p = self.primitives[u]
            if p.type == VOXEL or self._transform_locked(p):
                continue
            p.transform = matmult(T, p.transform)

    @ports("Context::scalePrimitiveAboutPoint", "Primitive::scale(const vec3&,const vec3&)")
    def scale_primitive_about_point(self, uuids, scale, point):
        s = V.v3(scale)
        if s[0] == 1 and s[1] == 1 and s[2] == 1:
            return
        T = None
        for u in _ids(uuids):
            p = self.primitives[u]
            if self._transform_locked(p):
                continue
            if s[0] == 0 or s[1] == 0 or s[2] == 0:
                raise ValueError("Primitive::scale: Scaling factor cannot be zero.")
            T = make_scale_matrix_about_point(s, point) if T is None else T
            p.transform = matmult(T, p.transform)

    # ---- objects -----------------------------------------------------------------------------

    @ports("Context::copyObject(uint)", "Context::copyObject(const std::vector<uint>&)")
    def copy_object(self, oids):
        if np.isscalar(oids):
            return self._copy_object_one(int(oids))
        return [self._copy_object_one(int(o)) for o in oids]

    def _copy_object_one(self, oid: int) -> int:
        src = self._object(oid, "Context::copyObject")
        uu = self.copy_primitive(list(src.uuids))
        new = self.current_object_id
        for u in uu:
            self.primitives[u].parent_object = new
        # Type-specific data (tile subdivisions, tube nodes/radii in the local frame, ...) is copied
        # as stored; the transform is copied verbatim. The origin is NOT copied: Helios constructs the
        # new object with a default (0,0,0) object_origin (see migration/notes/W1-B.md).
        obj = copy.deepcopy(src)
        obj.oid, obj.uuids, obj.origin = new, list(uu), np.zeros(3, f32)
        self.objects[new] = obj
        od = getattr(self, "object_data", None)          # Context::copyObjectData, once object data exists
        if isinstance(od, dict):
            for d in od.values():
                d.pop(new, None)
                if oid in d:
                    d[new] = copy.deepcopy(d[oid])
        self.current_object_id += 1
        return new

    @ports("Context::deleteObject(uint)", "Context::deleteObject(const std::vector<uint>&)")
    def delete_object(self, oids):
        for oid in _ids(oids):
            obj = self._object(oid, "Context::deleteObject")
            od = getattr(self, "object_data", None)
            if isinstance(od, dict):
                for d in od.values():
                    d.pop(oid, None)
            del self.objects[oid]
            self.delete_primitives(list(obj.uuids))

    @ports("Context::translateObject(uint,const vec3&)", "Context::translateObject(const std::vector<uint>&,const vec3&)",
           "CompoundObject::translate")
    def translate_object(self, oids, shift):
        s = V.v3(shift)
        for oid in _ids(oids):
            obj = self._object(oid, "Context::translateObject")
            if s[0] == 0 and s[1] == 0 and s[2] == 0:
                continue
            obj.transform = _apply_translation(s, obj.transform)
            for u in obj.uuids:
                p = self.primitives.get(u)
                if p is not None:
                    p.transform = _apply_translation(s, p.transform)

    @ports("Context::rotateObject(uint,float,const char*)", "Context::rotateObject(const std::vector<uint>&,float,const char*)",
           "Context::rotateObject(uint,float,const vec3&)", "Context::rotateObject(const std::vector<uint>&,float,const vec3&)",
           "Context::rotateObject(uint,float,const vec3&,const vec3&)",
           "Context::rotateObject(const std::vector<uint>&,float,const vec3&,const vec3&)",
           "CompoundObject::rotate(float,const char*)", "CompoundObject::rotate(float,const vec3&)",
           "CompoundObject::rotate(float,const vec3&,const vec3&)")
    def rotate_object(self, oids, radians, axis, origin=None):
        """``axis`` is "x"/"y"/"z" or a vec3. A vec3 axis passes through the world origin, or through
        ``origin`` when given (it is NOT the object center)."""
        if isinstance(axis, str) and origin is not None:
            raise ValueError("rotateObject: an origin needs a vec3 axis")
        for oid in _ids(oids):
            obj = self._object(oid, "Context::rotateObject")
            if f32(radians) == 0:
                continue
            if isinstance(axis, str):
                if axis not in ("x", "y", "z"):
                    raise ValueError("CompoundObject::rotate: Rotation axis should be one of x, y, or z.")
                T = make_rotation_matrix(radians, axis)
            elif origin is None:
                T = make_rotation_matrix_axis(radians, axis)
            else:
                T = make_rotation_matrix_about(radians, origin, axis)
            self._left_multiply_members(obj, T)

    @ports("Context::rotateObjectAboutOrigin(uint,float,const vec3&)",
           "Context::rotateObjectAboutOrigin(const std::vector<uint>&,float,const vec3&)")
    def rotate_object_about_origin(self, oids, radians, axis):
        """Rotate about the object's stored origin (its creation center; (0,0,0) for a copy)."""
        for oid in _ids(oids):
            self.rotate_object(oid, radians, axis, self._object(oid, "Context::rotateObjectAboutOrigin").origin)

    @ports("Context::scaleObjectAboutPoint(uint,const vec3&,const vec3&)",
           "Context::scaleObjectAboutPoint(const std::vector<uint>&,const vec3&,const vec3&)",
           "CompoundObject::scaleAboutPoint")
    def scale_object_about_point(self, oids, scale, point):
        s = V.v3(scale)
        for oid in _ids(oids):
            obj = self._object(oid, "Context::scaleObjectAboutPoint")
            if s[0] == 1 and s[1] == 1 and s[2] == 1:
                continue
            self._left_multiply_members(obj, make_scale_matrix_about_point(s, point))

    @ports("Context::scaleObject(uint,const vec3&)", "Context::scaleObject(const std::vector<uint>&,const vec3&)",
           "CompoundObject::scale")
    def scale_object(self, oids, scale):
        """Scale about the world origin (Helios: scaleAboutPoint(scale, nullorigin))."""
        for oid in _ids(oids):
            self.scale_object_about_point(oid, scale, (0.0, 0.0, 0.0))

    @ports("Context::scaleObjectAboutCenter(uint,const vec3&)", "Context::scaleObjectAboutCenter(const std::vector<uint>&,const vec3&)",
           "CompoundObject::scaleAboutCenter")
    def scale_object_about_center(self, oids, scale):
        for oid in _ids(oids):
            self._object(oid, "Context::scaleObjectAboutCenter")
            self.scale_object_about_point(oid, scale, self.object_center(oid))

    @ports("Context::scaleObjectAboutOrigin(uint,const vec3&)", "Context::scaleObjectAboutOrigin(const std::vector<uint>&,const vec3&)")
    def scale_object_about_origin(self, oids, scale):
        for oid in _ids(oids):
            self.scale_object_about_point(oid, scale, self._object(oid, "Context::scaleObjectAboutOrigin").origin)

    # ---- type-specific stored data, read through the object transform -------------------------

    @ports("Context::getTubeObjectNodes", "Tube::getNodes")
    def tube_object_nodes(self, oid: int) -> np.ndarray:
        obj = self._object(oid, "Context::getTubeObjectNodes")
        if obj.type != _OBJECT_TUBE:
            raise ValueError(f"getTubeObjectNodes: object {oid} is not a tube")
        return np.array([_xform_point(obj.transform, n) for n in obj.extra["nodes"]], f32).reshape(-1, 3)

    @ports("Context::getTubeObjectNodeRadii", "Tube::getNodeRadii")
    def tube_object_node_radii(self, oid: int) -> np.ndarray:
        obj = self._object(oid, "Context::getTubeObjectNodeRadii")
        if obj.type != _OBJECT_TUBE:
            raise ValueError(f"getTubeObjectNodeRadii: object {oid} is not a tube")
        T = obj.transform
        n0 = _xform_point(T, (0, 0, 0))
        return np.array([V.magnitude(_xform_point(T, (r, 0, 0)) - n0) for r in obj.extra["radius"]], f32)

    @ports("Context::getTileObjectVertices", "Tile::getVertices")
    def tile_object_vertices(self, oid: int) -> np.ndarray:
        obj = self._object(oid, "Context::getTileObjectVertices")
        if obj.type != _OBJECT_TILE:
            raise ValueError(f"getTileObjectVertices: object {oid} is not a tile")
        Y = ((-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0))
        return np.array([_xform_point(obj.transform, y) for y in Y], f32)

    @ports("Context::getTileObjectCenter", "Tile::getCenter")
    def tile_object_center(self, oid: int) -> np.ndarray:
        obj = self._object(oid, "Context::getTileObjectCenter")
        if obj.type != _OBJECT_TILE:
            raise ValueError(f"getTileObjectCenter: object {oid} is not a tile")
        return _xform_point(obj.transform, (0, 0, 0))
