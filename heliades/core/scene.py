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
    v = np.asarray(v, f32)
    m = f32(math.sqrt(float(f32(f32(v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]))))
    return v / m if m > 0 else v


def _cross(a, b):
    a, b = np.asarray(a, f32), np.asarray(b, f32)
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], f32)


def _mag(v):
    v = np.asarray(v, f32)
    return f32(math.sqrt(float(f32(f32(v[0] * v[0] + v[1] * v[1]) + v[2] * v[2]))))


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

    # ---- bulk ----
    def snapshot(self) -> "Snapshot":
        from ..parity.snapshot import Snapshot
        uu = self.all_uuids()
        n = len(uu)
        V, nv = np.zeros((n, 4, 3), f32), np.zeros(n, np.int32)
        A, N, C, T = np.zeros(n, f32), np.zeros((n, 3), f32), np.zeros((n, 4), f32), np.zeros(n, np.int32)
        for i, u in enumerate(uu):
            p = self.primitives[u]
            v = p.vertices()
            V[i, : len(v)], nv[i] = v, len(v)
            A[i], N[i], C[i], T[i] = p.area(), p.normal(), p.color, p.type
        return Snapshot(np.array(uu, np.int64), T, V, nv, A, N, C)
