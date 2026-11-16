# SPDX-License-Identifier: LGPL-2.1-or-later
"""Two implementations behind one small API, so a parity case is written once and run on both.

``HeliosBackend`` drives the pinned Helios build through PyHelios (the oracle).
``ForgeBackend`` drives Heliades. Add a method to both when a newly ported feature needs a case.
"""
from __future__ import annotations

import numpy as np

from .snapshot import Snapshot


class ForgeBackend:
    name = "heliades"

    def __init__(self):
        from ..core.scene import Scene
        self.scene = Scene()

    def seed(self, s):
        self.scene.seed_random_generator(s)

    def randu(self, *a):
        return self.scene.randu(*a)

    def randn(self, *a):
        return self.scene.randn(*a)

    def add_patch(self, center, size, rotation=(0.0, 0.0), color=None):
        return self.scene.add_patch(center, size, rotation, color)

    def add_triangle(self, v0, v1, v2, color=None):
        return self.scene.add_triangle(v0, v1, v2, color)

    def translate(self, uuid, shift):
        self.scene.translate_primitive(uuid, shift)

    def rotate(self, uuid, radians, axis):
        self.scene.rotate_primitive(uuid, radians, axis)

    def scale(self, uuid, s):
        self.scene.scale_primitive(uuid, s)

    def snapshot(self) -> Snapshot:
        return self.scene.snapshot()


class HeliosBackend:
    name = "helios"

    def __init__(self):
        from pyhelios import Context
        from pyhelios import DataTypes as D
        self.D = D
        self.ctx = Context()

    def seed(self, s):
        self.ctx.seedRandomGenerator(int(s))

    def randu(self, *a):
        if len(a) == 2 and all(isinstance(x, (int, np.integer)) for x in a):
            return self.ctx.randu(int(a[0]), int(a[1]))
        return self.ctx.randu(*[float(x) for x in a])

    def randn(self, *a):
        return self.ctx.randn(*[float(x) for x in a])

    def _c(self, color):
        return None if color is None else self.D.RGBcolor(*[float(c) for c in color[:3]])

    def add_patch(self, center, size, rotation=(0.0, 0.0), color=None):
        D = self.D
        # PyHelios substitutes white for color=None; the C++ default is black (HELIOS_ISSUES.md #1)
        return self.ctx.addPatch(D.vec3(*map(float, center)), D.vec2(*map(float, size)),
                                 D.SphericalCoord(1.0, float(rotation[0]), float(rotation[1])),
                                 self._c(color if color is not None else (0.0, 0.0, 0.0)))

    def add_triangle(self, v0, v1, v2, color=None):
        D = self.D
        return self.ctx.addTriangle(D.vec3(*map(float, v0)), D.vec3(*map(float, v1)), D.vec3(*map(float, v2)),
                                    self._c(color))

    def translate(self, uuid, shift):
        self.ctx.translatePrimitive(uuid, self.D.vec3(*map(float, shift)))

    def rotate(self, uuid, radians, axis):
        self.ctx.rotatePrimitive(uuid, float(radians), axis)

    def scale(self, uuid, s):
        self.ctx.scalePrimitive(uuid, self.D.vec3(*map(float, s)))

    def snapshot(self) -> Snapshot:
        return helios_snapshot(self.ctx)


def helios_snapshot(ctx) -> Snapshot:
    """Dense dump of a PyHelios Context via its bulk API, sorted by UUID."""
    uu = np.array(ctx.getAllUUIDs(), np.int64)
    flat, off = ctx.getAllPrimitiveVertices()
    flat = np.asarray(flat, np.float32).reshape(-1, 3)
    off = np.asarray(off, np.int64) // 3
    n = len(uu)
    V, nv = np.zeros((n, 4, 3), np.float32), np.zeros(n, np.int32)
    ends = np.r_[off[1:], len(flat)]
    for i in range(n):
        v = flat[off[i]:ends[i]]
        V[i, : len(v)], nv[i] = v, len(v)
    T = np.asarray(ctx.getAllPrimitiveTypes(), np.int32)
    A = np.asarray(ctx.getAllPrimitiveAreas(), np.float32)
    N = np.asarray(ctx.getAllPrimitiveNormals(), np.float32).reshape(n, 3)
    C = np.asarray(ctx.getAllPrimitiveColors(), np.float32).reshape(n, -1)
    if C.shape[1] == 3:
        C = np.c_[C, np.ones(n, np.float32)]
    o = np.argsort(uu)
    return Snapshot(uu[o], T[o], V[o], nv[o], A[o], N[o], C[o])
