# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_primitive.cpp, global.cpp; Copyright (C) 2016-2026 Brian Bailey).
"""Scene: the Heliades counterpart of helios::Context (geometry part).

Primitives keep Helios's representation exactly: a float32 4x4 affine ``transform`` applied to a
canonical shape (unit patch or generic triangle). All arithmetic below is float32, op for op, in the
same order as the C++ so results agree with Helios to the last bit or within an ulp.

Host-side (numpy) by design: generation is branchy and sequential. Bulk consumers (rendering,
ray tracing, physics) use ``snapshot()`` / ``heliades.core.device`` to get dense GPU arrays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import vec as V
from .random import ContextRandom
from ..migration_registry import ports

f32 = np.float32
PATCH, TRIANGLE, VOXEL = 0, 1, 2

_PATCH_Y = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]], f32)
_TRI_Y = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], f32)


# ---- float32 matrix helpers (global.cpp) -----------------------------------------------------

@ports("helios::matmult")
def matmult(ML: np.ndarray, MR: np.ndarray) -> np.ndarray:
    """M[i,j] = sum_k ML[i,k] MR[k,j], accumulated from 0 in k order, float32 each step."""
    ML, MR = ML.reshape(4, 4), MR.reshape(4, 4)
    M = np.zeros((4, 4), f32)
    for k in range(4):
        M = M + ML[:, k:k + 1] * MR[k:k + 1, :]
    return M.reshape(16)


@ports("helios::makeScaleMatrix")
def make_scale_matrix(s) -> np.ndarray:
    T = np.zeros(16, f32)
    T[0], T[5], T[10], T[15] = s[0], s[1], s[2], 1
    return T


@ports("helios::makeTranslationMatrix")
def make_translation_matrix(t) -> np.ndarray:
    T = np.eye(4, dtype=f32).reshape(16).copy()
    T[3], T[7], T[11] = t[0], t[1], t[2]
    return T


@ports("helios::makeRotationMatrix(float,const char*)")
def make_rotation_matrix(rotation: float, axis: str) -> np.ndarray:
    r = float(f32(rotation))
    sx, cx = f32(math.sin(r)), f32(math.cos(r))
    T = np.zeros(16, f32)
    if axis == "x":
        T[0], T[5], T[6], T[9], T[10] = 1, cx, -sx, sx, cx
    elif axis == "y":
        T[0], T[2], T[5], T[8], T[10] = cx, sx, 1, -sx, cx
    elif axis == "z":
        T[0], T[1], T[4], T[5], T[10] = cx, -sx, sx, cx, 1
    else:
        raise ValueError("makeRotationMatrix: axis must be x, y or z")
    T[15] = 1
    return T


def _apply(T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """v = T[0]*y.x + T[1]*y.y + T[2]*y.z + T[3] (left to right, float32)."""
    T = T.reshape(4, 4)
    out = np.empty_like(Y)
    for r in range(3):
        out[:, r] = ((T[r, 0] * Y[:, 0] + T[r, 1] * Y[:, 1]) + T[r, 2] * Y[:, 2]) + T[r, 3]
    return out


def _norm3(v):
    """helios::normalize (free function): no zero guard, multiply by 1/sqrtf."""
    v = np.asarray(v, f32)
    m2 = f32(f32(v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])
    with np.errstate(divide="ignore", invalid="ignore"):
        return v * f32(f32(1.0) / f32(np.sqrt(m2)))


_cross = V.cross
_mag = V.magnitude


# generic triangle (0,0,0),(0,1,0),(1,1,0) padded with ones; its inverse in double (constant)
_V1 = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], np.float64).reshape(4, 4)
_INV_V1_F32 = np.linalg.inv(_V1).astype(f32)  # exact small rationals, so any inversion method agrees


@dataclass
class Primitive:
    uuid: int
    type: int
    transform: np.ndarray                      # float32[16], row-major 4x4
    color: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1], f32))
    parent_object: int = 0
    solid_fraction: float = 1.0
    texture_file: str | None = None
    uv: np.ndarray | None = None

    def vertices(self) -> np.ndarray:
        return _apply(self.transform, _PATCH_Y if self.type == PATCH else _TRI_Y)

    def normal(self) -> np.ndarray:
        if self.type == PATCH:
            T = self.transform
            return _norm3([T[2], T[6], T[10]])
        v = self.vertices()
        return _norm3(_cross(v[1] - v[0], v[2] - v[1]))

    def size(self) -> np.ndarray:
        v = self.vertices()
        return np.array([_mag(v[1] - v[0]), _mag(v[3] - v[0])], f32)

    def area(self) -> np.float32:
        if self.type == PATCH:
            s = self.size()
            return f32(f32(s[0] * s[1]) * f32(self.solid_fraction))
        v = self.vertices()
        return f32(f32(f32(0.5) * _mag(_cross(v[1] - v[0], v[2] - v[0]))) * f32(self.solid_fraction))

    # transforms (Patch::rotate / Primitive::scale / Primitive::translate)
    def scale(self, s):
        s = np.asarray(s, f32)
        if np.any(s == 0):
            raise ValueError("scale factor cannot be zero")
        if np.all(s == 1):
            return
        self.transform = matmult(make_scale_matrix(s), self.transform)

    def rotate(self, radians: float, axis: str):
        if f32(radians) == 0:
            return
        self.transform = matmult(make_rotation_matrix(radians, axis), self.transform)

    def translate(self, t):
        t = np.asarray(t, f32)
        if np.all(t == 0):
            return
        self.transform = matmult(make_translation_matrix(t), self.transform)


@ports("Triangle::makeTransformationMatrix")
def triangle_transform(v0, v1, v2) -> np.ndarray:
    V2 = np.ones((4, 4), f32)
    V2[:3, 0], V2[:3, 1], V2[:3, 2] = np.asarray(v0, f32), np.asarray(v1, f32), np.asarray(v2, f32)
    T = np.zeros((4, 4), f32)
    for k in range(4):  # transform[j+4i] += V2[k+4i] * float(invV1[j+4k])
        T = T + V2[:, k:k + 1] * _INV_V1_F32[k:k + 1, :]
    return T.reshape(16)


class Scene:
    """Geometry + data container (helios::Context counterpart)."""

    def __init__(self):
        self.primitives: dict[int, Primitive] = {}
        self.current_uuid = 0
        self.objects: dict[int, CompoundObject] = {}
        self.current_object_id = 1  # object ID 0 is reserved for "no parent object"
        self.primitive_data: dict[str, dict[int, object]] = {}
        self.rng = ContextRandom()

    # ---- random ----
    @ports("Context::seedRandomGenerator")
    def seed_random_generator(self, seed: int):
        self.rng.seed(seed)

    @ports("Context::randu")
    def randu(self, lo=None, hi=None):
        return self.rng.randu(lo, hi)

    @ports("Context::randn")
    def randn(self, mean=None, stddev=None):
        return self.rng.randn(mean, stddev)

    # ---- primitives ----
    def _new(self, ptype, transform, color) -> int:
        c = np.ones(4, f32)
        if color is not None:
            c[: len(color)] = color
        else:
            c[:3] = 0
        u = self.current_uuid
        self.primitives[u] = Primitive(u, ptype, transform.astype(f32), c)
        self.current_uuid += 1
        return u

    @ports("Context::addPatch")
    def add_patch(self, center=(0, 0, 0), size=(1, 1), rotation=(0.0, 0.0), color=None) -> int:
        """rotation = (elevation, azimuth) in radians, as helios::SphericalCoord."""
        size = np.asarray(size, f32)
        if size[0] < f32(1e-6) or size[1] < f32(1e-6):
            raise ValueError("addPatch: size must be > 1e-6")
        u = self._new(PATCH, np.eye(4, dtype=f32).reshape(16), color)
        p = self.primitives[u]
        p.scale((size[0], size[1], 1))
        el, az = f32(rotation[0]), f32(rotation[1])
        if el != 0:
            p.rotate(-el, "x")
        if az != 0:
            p.rotate(-az, "z")
        p.translate(center)
        return u

    @ports("Context::addTriangle")
    def add_triangle(self, v0, v1, v2, color=None) -> int:
        return self._new(TRIANGLE, triangle_transform(v0, v1, v2), color)

    @ports("Context::deletePrimitive")
    def delete_primitive(self, uuid: int):
        del self.primitives[uuid]
        for d in self.primitive_data.values():
            d.pop(uuid, None)

    @ports("Context::getAllUUIDs")
    def all_uuids(self) -> list[int]:
        return sorted(self.primitives)

    @ports("Context::getPrimitiveVertices")
    def primitive_vertices(self, uuid: int) -> np.ndarray:
        return self.primitives[uuid].vertices()

    @ports("Context::getPrimitiveArea")
    def primitive_area(self, uuid: int):
        return self.primitives[uuid].area()

    @ports("Context::getPrimitiveNormal")
    def primitive_normal(self, uuid: int):
        return self.primitives[uuid].normal()

    @ports("Context::getPrimitiveType")
    def primitive_type(self, uuid: int) -> int:
        return self.primitives[uuid].type

    @ports("Context::translatePrimitive")
    def translate_primitive(self, uuid: int, shift):
        self.primitives[uuid].translate(shift)

    @ports("Context::rotatePrimitive")
    def rotate_primitive(self, uuid: int, radians: float, axis: str):
        self.primitives[uuid].rotate(radians, axis)

    @ports("Context::scalePrimitive")
    def scale_primitive(self, uuid: int, s):
        self.primitives[uuid].scale(s)

    # ---- primitive data ----
    @ports("Context::setPrimitiveData")
    def set_primitive_data(self, uuid: int, label: str, value):
        self.primitive_data.setdefault(label, {})[uuid] = value

    @ports("Context::getPrimitiveData")
    def get_primitive_data(self, uuid: int, label: str):
        return self.primitive_data[label][uuid]

    @ports("Context::doesPrimitiveDataExist")
    def does_primitive_data_exist(self, uuid: int, label: str) -> bool:
        return uuid in self.primitive_data.get(label, {})

    # ---- compound objects (Context_object.cpp) ----
    def _register_object(self, otype: int, uuids: list[int], **kw) -> int:
        oid = self.current_object_id
        for u in uuids:
            self.primitives[u].parent_object = oid
        self.objects[oid] = CompoundObject(oid, otype, list(uuids), **kw)
        self.current_object_id += 1
        return oid

    @ports("Context::addTileObject")
    def add_tile_object(self, center, size, rotation=(0.0, 0.0), subdiv=(1, 1), color=(0.0, 0.75, 0.0)) -> int:
        size = V.v3(size[0], size[1], 0)[:2]
        if size[0] == 0 or size[1] == 0:
            raise ValueError("addTileObject: size must be > 0")
        if subdiv[0] < 1 or subdiv[1] < 1:
            raise ValueError("addTileObject: subdivisions must be > 0")
        sub = np.array([size[0] / f32(subdiv[0]), size[1] / f32(subdiv[1])], f32)
        el, az = f32(rotation[0]), f32(rotation[1])
        uuids = []
        for j in range(subdiv[1]):
            for i in range(subdiv[0]):
                sc = (f32(-0.5) * size[0] + f32(f32(i) + f32(0.5)) * sub[0],
                      f32(-0.5) * size[1] + f32(f32(j) + f32(0.5)) * sub[1], 0.0)
                u = self.add_patch(sc, sub, (0.0, 0.0), color)
                p = self.primitives[u]
                if el != 0:
                    p.rotate(-el, "x")
                if az != 0:
                    p.rotate(-az, "z")
                p.translate(center)
                uuids.append(u)
        T = np.eye(4, dtype=f32).reshape(16)
        T = matmult(make_scale_matrix((size[0], size[1], 1)), T)
        T = matmult(make_rotation_matrix(-el, "x"), T)
        T = matmult(make_rotation_matrix(-az, "z"), T)
        T = matmult(make_translation_matrix(center), T)
        return self._register_object(OBJECT_TILE, uuids, transform=T, origin=V.v3(center),
                                     extra={"subdiv": tuple(int(s) for s in subdiv)})

    @ports("Context::addTubeObject")
    def add_tube_object(self, radial_subdivisions: int, nodes, radius, colors=None) -> int:
        nodes = [V.v3(n) for n in nodes]
        radius = [f32(r) for r in radius]
        n = len(nodes)
        if n == 0:
            raise ValueError("addTubeObject: node and radius arrays are empty")
        if n != len(radius):
            raise ValueError("addTubeObject: nodes and radius must agree in size")
        colors = [(0.0, 0.75, 0.0)] * n if colors is None else list(colors)
        if n != len(colors):
            raise ValueError("addTubeObject: nodes and color must agree in size")
        rc = [f32(1e-5) if (r < f32(1e-5) and r >= 0) else r for r in radius]
        nr = int(radial_subdivisions)
        cf = [V.cosf(f32(f32(f32(2.0) * V.PI_F) * f32(j)) / f32(nr)) for j in range(nr + 1)]
        sf = [V.sinf(f32(f32(f32(2.0) * V.PI_F) * f32(j)) / f32(nr)) for j in range(nr + 1)]
        X, Y, Z = V.v3(1, 0, 0), V.v3(0, 1, 0), V.v3(0, 0, 1)
        initial_radial = X.copy()
        prev_axial = prev_radial = None
        ring = [[None] * (nr + 1) for _ in range(n)]

        def fallback(axial):
            fr = X.copy()
            if abs(V.dot(axial, fr)) > f32(0.95):
                fr = Y.copy()
            if abs(axial[2]) > f32(0.95):
                fr = X.copy()
            return V.normalize(V.cross(axial, fr))

        for i in range(n):
            if radius[i] < 0:
                raise ValueError("addTubeObject: radius must be positive")
            if i == 0:
                axial = nodes[i + 1] - nodes[i]  # n == 1 raises IndexError, as Helios reads past the end
                mag = V.magnitude(axial)
                axial = Z.copy() if mag < f32(1e-6) else V.div(axial, mag)
                if abs(V.dot(axial, initial_radial)) > f32(0.95):
                    initial_radial = Y.copy()
                if abs(axial[2]) > f32(0.95):
                    initial_radial = X.copy()
                prev_radial = V.normalize(V.cross(axial, initial_radial))
            else:
                if i == n - 1:
                    axial = nodes[i] - nodes[i - 1]
                else:
                    axial = f32(0.5) * ((nodes[i] - nodes[i - 1]) + (nodes[i + 1] - nodes[i]))
                mag = V.magnitude(axial)
                axial = Z.copy() if mag < f32(1e-6) else V.div(axial, mag)
                rot_axis = V.cross(prev_axial, axial)
                if float(V.magnitude(rot_axis)) > 1e-5:
                    angle = V.acosf(np.clip(V.dot(prev_axial, axial), f32(-1), f32(1)))
                    prev_radial = V.rotate_point_about_line(prev_radial, V.v3(0, 0, 0), rot_axis, angle)
                else:
                    prev_radial = fallback(axial)
            prev_axial = axial
            radial = prev_radial
            ortho = V.normalize(V.cross(radial, axial))
            for j in range(nr + 1):
                normal = (f32(cf[j] * rc[i]) * radial) + (f32(sf[j] * rc[i]) * ortho)
                ring[i][j] = nodes[i] + normal
        uuids = []
        for j in range(nr):
            for i in range(n - 1):
                uuids.append(self.add_triangle(ring[i][j], ring[i + 1][j + 1], ring[i][j + 1], colors[i]))
                uuids.append(self.add_triangle(ring[i][j], ring[i + 1][j], ring[i + 1][j + 1], colors[i]))
        oid = self._register_object(OBJECT_TUBE, uuids, transform=np.eye(4, dtype=f32).reshape(16),
                                    extra={"nodes": np.array(nodes), "radius": np.array(radius), "subdiv": nr})
        self.objects[oid].origin = self.object_center(oid)
        return oid

    @ports("Context::addPolymeshObject")
    def add_polymesh_object(self, uuids) -> int:
        uuids = list(uuids)
        if not uuids:
            raise ValueError("addPolymeshObject: UUIDs array is empty")
        if any(u not in self.primitives for u in uuids):
            raise ValueError("addPolymeshObject: a UUID does not exist")
        first = self.primitives[uuids[0]].parent_object
        if first != 0 and first in self.objects and self.objects[first].type == OBJECT_POLYMESH \
                and all(self.primitives[u].parent_object == first for u in uuids):
            return first
        free = [u for u in uuids if self.primitives[u].parent_object == 0]
        if not free:
            raise ValueError("addPolymeshObject: all primitives already belong to another object")
        T = matmult(make_translation_matrix(self.primitives[free[0]].vertices()[0]), np.eye(4, dtype=f32).reshape(16))
        oid = self._register_object(OBJECT_POLYMESH, free, transform=T)
        self.objects[oid].origin = self.object_center(oid)
        return oid

    @ports("Context::getAllObjectIDs")
    def all_object_ids(self) -> list[int]:
        return sorted(self.objects)

    @ports("Context::getObjectType")
    def object_type(self, oid: int) -> int:
        return self.objects[oid].type

    @ports("Context::getObjectPrimitiveUUIDs")
    def object_primitive_uuids(self, oid: int) -> list[int]:
        return list(self.objects[oid].uuids)

    @ports("Context::getPrimitiveParentObjectID")
    def primitive_parent_object(self, uuid: int) -> int:
        return self.primitives[uuid].parent_object

    @ports("Context::getObjectTransformationMatrix")
    def object_transform(self, oid: int) -> np.ndarray:
        return self.objects[oid].transform.copy()

    @ports("Context::getObjectBoundingBox")
    def object_bounding_box(self, oid: int):
        V_ = np.concatenate([self.primitives[u].vertices() for u in self.objects[oid].uuids])
        return V_.min(0), V_.max(0)

    @ports("Context::getObjectCenter")
    def object_center(self, oid: int) -> np.ndarray:
        lo, hi = self.object_bounding_box(oid)
        return f32(0.5) * (lo + hi)

    # ---- bulk ----
    def snapshot(self) -> "Snapshot":
        from ..parity.snapshot import ObjectTable, Snapshot
        uu = self.all_uuids()
        n = len(uu)
        Vt, nv = np.zeros((n, 4, 3), f32), np.zeros(n, np.int32)
        A, N, C, T = np.zeros(n, f32), np.zeros((n, 3), f32), np.zeros((n, 4), f32), np.zeros(n, np.int32)
        P = np.zeros(n, np.int64)
        for i, u in enumerate(uu):
            p = self.primitives[u]
            v = p.vertices()
            Vt[i, : len(v)], nv[i] = v, len(v)
            A[i], N[i], C[i], T[i], P[i] = p.area(), p.normal(), p.color, p.type, p.parent_object
        oids = self.all_object_ids()
        objs = ObjectTable(np.array(oids, np.int64),
                           np.array([self.objects[o].type for o in oids], np.int32),
                           [np.array(self.objects[o].uuids, np.int64) for o in oids],
                           np.array([self.object_center(o) for o in oids], f32).reshape(-1, 3),
                           np.array([self.objects[o].transform for o in oids], f32).reshape(-1, 16))
        return Snapshot(np.array(uu, np.int64), T, Vt, nv, A, N, C, P, objs)


OBJECT_TILE, OBJECT_SPHERE, OBJECT_TUBE, OBJECT_BOX, OBJECT_DISK, OBJECT_POLYMESH, OBJECT_CONE = range(7)


@dataclass
class CompoundObject:
    oid: int
    type: int
    uuids: list[int]
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=f32).reshape(16))
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, f32))
    extra: dict = field(default_factory=dict)
