# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_primitive.cpp, global.cpp; Copyright (C) 2016-2026 Brian Bailey).
"""Primitives (patch, triangle): float32 transforms applied to canonical shapes, op for op as Helios."""
from __future__ import annotations

import math
import numpy as np

from . import vec as V
from ..migration_registry import ports

f32 = np.float32

PATCH, TRIANGLE, VOXEL = 0, 1, 2

from .store import INV_V1_F32 as _INV_V1_F32, PATCH_Y as _PATCH_Y, TRI_Y as _TRI_Y  # noqa: E402


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


class Primitive:
    """One primitive, as a light view of its row in the Scene's :class:`~heliades.core.store.PrimitiveStore`.

    Reading a field returns a copy (``transform``, ``color``, ``uv``), so a Primitive behaves like the value
    object it replaces; assigning a field writes the row. The proxy holds no state of its own, so two proxies
    of one UUID always agree. Bulk code should use the store's batched row functions instead."""

    __slots__ = ("_s", "uuid")

    def __init__(self, store, uuid: int):
        _setattr(self, "_s", store)
        _setattr(self, "uuid", uuid)

    def __repr__(self):
        return f"Primitive(uuid={self.uuid}, type={self.type}, parent_object={self.parent_object})"

    # Attributes other than the fields below (added by other packages) live in the store's per-UUID
    # ``extra`` dict; copies deep-copy them.
    def __getattr__(self, name):
        if name in ("_s", "uuid"):
            raise AttributeError(name)
        d = self._s.extra.get(self.uuid)
        if d is not None and name in d:
            return d[name]
        raise AttributeError(f"'Primitive' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in _PRIMITIVE_OWN:
            object.__setattr__(self, name, value)
        else:
            self._s.extra.setdefault(self.uuid, {})[name] = value

    # ---- fields ----
    @property
    def type(self) -> int:
        return int(self._s.ptype[self.uuid])

    @property
    def transform(self) -> np.ndarray:                 # float32[16], row-major 4x4
        return self._s.T[self.uuid].copy()

    @transform.setter
    def transform(self, T):
        self._s.T[self.uuid] = np.asarray(T, f32).reshape(16)

    @property
    def color(self) -> np.ndarray:                     # RGBA float32
        return self._s.color[self.uuid].copy()

    @color.setter
    def color(self, c):
        self._s.color[self.uuid] = c

    @property
    def parent_object(self) -> int:
        return int(self._s.parent[self.uuid])

    @parent_object.setter
    def parent_object(self, oid):
        self._s.parent[self.uuid] = oid

    @property
    def solid_fraction(self) -> np.float32:            # Primitive::solid_fraction (from the texture's alpha mask)
        return self._s.sf[self.uuid]

    @solid_fraction.setter
    def solid_fraction(self, x):
        self._s.sf[self.uuid] = x

    @property
    def texture_file(self) -> str | None:              # the material's texture_file, verbatim as passed to addPatch etc.
        return self._s.tex[self.uuid]

    @texture_file.setter
    def texture_file(self, t):
        self._s.tex[self.uuid] = t

    @property
    def uv(self) -> np.ndarray | None:                 # (k,2) float32 texture (u,v) per vertex; None = Helios's empty vector
        k = int(self._s.nuv[self.uuid])
        return None if k == 0 else self._s.uv[self.uuid, :k].copy()

    @uv.setter
    def uv(self, uv):
        if uv is None or len(uv) == 0:
            self._s.nuv[self.uuid] = 0
            return
        uv = np.asarray(uv, f32).reshape(-1, 2)
        self._s.uv[self.uuid] = 0
        self._s.uv[self.uuid, : len(uv)] = uv
        self._s.nuv[self.uuid] = len(uv)

    @property
    def material_id(self) -> int | None:
        m = int(self._s.mat[self.uuid])
        return None if m < 0 else m

    @material_id.setter
    def material_id(self, m):
        self._s.mat[self.uuid] = -1 if m is None else m

    @ports("Primitive::hasTexture")
    def has_texture(self) -> bool:
        return bool(self.texture_file)

    @ports("Primitive::getTextureUV")
    def texture_uv(self) -> np.ndarray:
        uv = self.uv
        return np.zeros((0, 2), f32) if uv is None else uv

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
        self.transform = scaled(self._s.T[self.uuid], s)

    def rotate(self, radians: float, axis: str):
        self.transform = rotated(self._s.T[self.uuid], radians, axis)

    def translate(self, t):
        self.transform = translated(self._s.T[self.uuid], t)


_setattr = object.__setattr__
_PRIMITIVE_OWN = frozenset({"_s", "uuid"} | {k for k, v in vars(Primitive).items() if isinstance(v, property)})


# The three Primitive transforms as functions of a matrix (used to build a new primitive's matrix before it is
# stored, and by the proxy above).

def scaled(T, s) -> np.ndarray:
    s = np.asarray(s, f32)
    if np.any(s == 0):
        raise ValueError("scale factor cannot be zero")
    if np.all(s == 1):
        return T
    return matmult(make_scale_matrix(s), T)


def rotated(T, radians: float, axis: str) -> np.ndarray:
    if f32(radians) == 0:
        return T
    return matmult(make_rotation_matrix(radians, axis), T)


def translated(T, t) -> np.ndarray:
    t = np.asarray(t, f32)
    if np.all(t == 0):
        return T
    return matmult(make_translation_matrix(t), T)


@ports("Triangle::makeTransformationMatrix")
def triangle_transform(v0, v1, v2) -> np.ndarray:
    V2 = np.ones((4, 4), f32)
    V2[:3, 0], V2[:3, 1], V2[:3, 2] = np.asarray(v0, f32), np.asarray(v1, f32), np.asarray(v2, f32)
    T = np.zeros((4, 4), f32)
    for k in range(4):  # transform[j+4i] += V2[k+4i] * float(invV1[j+4k])
        T = T + V2[:, k:k + 1] * _INV_V1_F32[k:k + 1, :]
    return T.reshape(16)
