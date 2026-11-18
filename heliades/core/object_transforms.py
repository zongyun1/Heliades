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
from .store import matmult_rows
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
    s = shift if isinstance(shift, np.ndarray) and shift.dtype == f32 and shift.shape == (3,) else V.v3(shift)
    if (M[12:16] == _AFFINE_ROW).all():
        M = M.copy()
        M[3:12:4] = M[3:12:4] + s          # M[3] += s.x, M[7] += s.y, M[11] += s.z
        return M
    return matmult(make_translation_matrix(s), M)


_AFFINE_ROW = np.array([0, 0, 0, 1], f32)


def _xform_point(T, p) -> np.ndarray:
    """T[0]*x + T[1]*y + T[2]*z + T[3] per row (Tube::getNodes, Tile::getVertices, vecmult)."""
    x, y, z = V.v3(p)
    return np.array([f32(f32(f32(T[4 * r] * x) + f32(T[4 * r + 1] * y)) + f32(T[4 * r + 2] * z)) + T[4 * r + 3]
                     for r in range(3)], f32)


def _ids(x) -> list[int]:
    return [int(x)] if np.isscalar(x) else [int(i) for i in x]


_IMMUTABLE = (np.generic, str, bool, int, float)


def _copy_datum(v):
    """Copy of a data.TypedData for copyPrimitiveData / copyObjectData. Entries are never modified in place
    (setters replace them, getters return copies), so an entry holding only immutable values (numpy scalars,
    str, bool) is shared; one holding arrays is deep-copied."""
    vals = v.values
    if isinstance(vals, list) and all(isinstance(x, _IMMUTABLE) for x in vals):
        return v
    return copy.deepcopy(v)


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

    def _alive_members(self, obj):
        """The object's existing member rows (slice or index array); Helios skips members that no longer
        exist (doesPrimitiveExist)."""
        st = self._store
        if obj._alive_gen == st.deaths:          # no primitive deleted since the last check (rows never come back)
            return obj._alive_ix
        ix = obj.member_index()
        out = self._alive_members_uncached(ix)
        if out is ix:                            # every member exists: cache until the next deletion
            obj._alive_ix, obj._alive_gen = out, st.deaths
        return out

    def _alive_members_uncached(self, ix):
        st = self._store
        if isinstance(ix, slice):
            if ix.stop <= st.n and st.alive[ix].all():
                return ix
            ix = np.arange(ix.start, ix.stop)
        ok = (ix >= 0) & (ix < st.n)
        if ok.all() and st.alive[ix].all():
            return ix
        ix = ix[ok]
        return ix[st.alive[ix]]

    def _left_multiply_members(self, obj, T):
        """T * M for the object matrix and each member matrix, as one batch (the object matrix is the last row)."""
        ix = self._alive_members(obj)
        st = self._store
        R = np.concatenate([st.T[ix], np.asarray(obj.transform, f32).reshape(1, 16)])
        out = matmult_rows(np.asarray(T, f32).reshape(16), R)
        st.T[ix] = out[:-1]
        obj.transform = np.array(out[-1], f32)

    def _batch(self, ids):
        """(objects, member rows) when one call on several objects can be done as one row operation: they all
        exist, are distinct and share no member primitive. Else None (then they are done one by one)."""
        if len(ids) < 2:
            return None
        objs = [self.objects.get(o) for o in ids]
        if any(o is None for o in objs) or len(set(ids)) != len(ids):
            return None
        parts = []
        for o in objs:
            ix = self._alive_members(o)
            parts.append(np.arange(ix.start, ix.stop) if isinstance(ix, slice) else ix)
        rows = np.concatenate(parts)
        if len(rows) > 1:
            srt = np.sort(rows)
            if (srt[1:] == srt[:-1]).any():
                return None
        return objs, rows

    def _left_multiply_objects(self, objs, rows, T):
        """T * M for each object transform and every member primitive (CompoundObject::rotate/scaleAboutPoint)."""
        OT = matmult_rows(T, np.stack([o.transform for o in objs]).astype(f32, copy=False))
        for o, t in zip(objs, OT):
            o.transform = t.copy()
        self._store.left_multiply_rows(rows, T)

    # ---- primitives --------------------------------------------------------------------------

    @ports("Context::copyPrimitive(uint)", "Context::copyPrimitive(const std::vector<uint>&)")
    def copy_primitive(self, uuids):
        """Copy one primitive (int -> int) or several (list -> list), in order. The copy keeps every
        Primitive field and its primitive data, but belongs to no object."""
        if np.isscalar(uuids):
            return self._copy_primitives([int(uuids)])[0]
        return self._copy_primitives([int(u) for u in uuids])

    def _copy_primitive_one(self, uuid: int) -> int:
        """One copyPrimitive, with the Scene bookkeeping (used when a batch has a missing UUID, so the
        copies made before the error are exactly Helios's)."""
        if uuid not in self.primitives:
            raise ValueError(f"Context::copyPrimitive: UUID of {uuid} not found in the context.")
        return self._copy_primitives([uuid])[0]

    def _copy_primitives(self, uuids: list[int]) -> list[int]:
        """copyPrimitive for a list, as array row copies. Same result as copying them one by one."""
        if not uuids:
            return []
        st = self._store
        src = np.asarray(uuids, np.int64)
        ok = (src >= 0) & (src < st.n)
        if not ok.all() or not st.alive[src].all():
            return [self._copy_primitive_one(u) for u in uuids]     # raises at the first missing UUID
        new0 = self.current_uuid
        dst = st.copy_rows(src, new0)
        # A textured patch is rebuilt from solid_fraction = getArea() / (size.x * size.y), which is not
        # always bit-identical to the source's stored fraction.
        tp = np.flatnonzero((st.ptype[src] == PATCH) & np.array([bool(t) for t in st.tex[src]], bool))
        if len(tp):
            s_ = src[tp]
            sz = st.patch_sizes_rows(s_)
            with np.errstate(divide="ignore", invalid="ignore"):
                st.sf[dst[tp]] = st.areas_rows(s_) / (sz[:, 0] * sz[:, 1])
        self.current_uuid = new0 + len(src)
        self._copy_primitive_data_rows(src.tolist(), dst.tolist())
        out = dst.tolist()
        self._after_copy_primitives(out)
        return out

    def _after_copy_primitives(self, new_uuids: list[int]):
        """Hook for the Scene's per-copy bookkeeping (iteration order, material references)."""

    def _copy_primitive_data_rows(self, src: list[int], dst: list[int]):
        """copyPrimitiveData(src[i], dst[i]) for fresh dst UUIDs (no data of their own yet)."""
        for d in self.primitive_data.values():
            if not d:
                continue
            for s_, t_ in zip(src, dst):
                v = d.get(s_)
                if v is not None:
                    d[t_] = _copy_datum(v)

    @ports("Context::copyPrimitiveData")
    def copy_primitive_data(self, src: int, dst: int):
        """Replace all of dst's primitive data by (deep copies of) src's."""
        for d in self.primitive_data.values():
            d.pop(dst, None)
            if src in d:
                d[dst] = _copy_datum(d[src])

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
        if uu:
            self._store.parent[np.asarray(uu, np.int64)] = new
        # Type-specific data (tile subdivisions, tube nodes/radii in the local frame, ...) is copied
        # as stored; the transform is copied verbatim. The origin is NOT copied: Helios constructs the
        # new object with a default (0,0,0) object_origin (see migration/notes/W1-B.md).
        obj = type(src)(new, src.type, uu, src.transform.copy(), np.zeros(3, f32), copy.deepcopy(src.extra))
        self.objects[new] = obj
        od = getattr(self, "object_data", None)          # Context::copyObjectData, once object data exists
        if isinstance(od, dict):
            for d in od.values():
                d.pop(new, None)
                if oid in d:
                    d[new] = _copy_datum(d[oid])
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
        ids = _ids(oids)
        zero = not s.any()                      # shift == nullorigin (-0 counts as 0, NaN does not)
        b = self._batch(ids)
        if b is not None:
            if zero:
                return
            for obj in b[0]:
                obj.transform = _apply_translation(s, obj.transform)
            self._store.translate_rows(b[1], s)
            return
        for oid in ids:
            obj = self._object(oid, "Context::translateObject")
            if zero:
                continue
            obj.transform = _apply_translation(s, obj.transform)
            self._store.translate_rows(self._alive_members(obj), s)

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
        ids = _ids(oids)
        b = self._batch(ids)
        if b is not None:                               # one matrix, as each object would build it
            if f32(radians) == 0:
                return
            if isinstance(axis, str):
                if axis not in ("x", "y", "z"):
                    raise ValueError("CompoundObject::rotate: Rotation axis should be one of x, y, or z.")
                T = make_rotation_matrix(radians, axis)
            elif origin is None:
                T = make_rotation_matrix_axis(radians, axis)
            else:
                T = make_rotation_matrix_about(radians, origin, axis)
            self._left_multiply_objects(*b, T)
            return
        for oid in ids:
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
        ids = _ids(oids)
        b = self._batch(ids)
        if b is not None:
            if s[0] == 1 and s[1] == 1 and s[2] == 1:
                return
            self._left_multiply_objects(*b, make_scale_matrix_about_point(s, point))
            return
        for oid in ids:
            obj = self._object(oid, "Context::scaleObjectAboutPoint")
            if s[0] == 1 and s[1] == 1 and s[2] == 1:
                continue
            self._left_multiply_members(obj, make_scale_matrix_about_point(s, point))

    @ports("Context::scaleObject(uint,const vec3&)", "Context::scaleObject(const std::vector<uint>&,const vec3&)",
           "CompoundObject::scale")
    def scale_object(self, oids, scale):
        """Scale about the world origin (Helios: scaleAboutPoint(scale, nullorigin))."""
        ids = _ids(oids)
        if all(o in self.objects for o in ids) and len(set(ids)) == len(ids):
            self.scale_object_about_point(ids, scale, (0.0, 0.0, 0.0))    # batches when it can
            return
        for oid in ids:
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
