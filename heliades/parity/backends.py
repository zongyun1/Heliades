# SPDX-License-Identifier: LGPL-2.1-or-later
"""Two implementations behind one small API, so a parity case is written once and run on both.

``HeliosBackend`` drives the pinned Helios build through PyHelios (the oracle).
``ForgeBackend`` drives Heliades. Add a method to both when a newly ported feature needs a case.
"""
from __future__ import annotations

import numpy as np

from .snapshot import Snapshot


class _OpsDispatch:
    """Falls back to operations registered with ``@op`` in ops_* modules."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        from .ops import OPS, load_plugins
        load_plugins()
        impl = OPS.get(name, {}).get(self.name)
        if impl is None:
            raise AttributeError(f"{type(self).__name__} has no operation '{name}' (register it with @op in an ops_* module)")
        return lambda *a, **k: impl(self, *a, **k)


class ForgeBackend(_OpsDispatch):
    name = "heliades"

    def __init__(self, mode: str = "exact"):
        from ..core.scene import Scene
        self.scene = Scene(mode=mode)

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

    def add_tile(self, center, size, rotation, subdiv, color=(0.0, 0.75, 0.0)):
        return self.scene.add_tile_object(center, size, rotation, subdiv, color)

    def add_tube(self, ndiv, nodes, radius, colors=None):
        return self.scene.add_tube_object(ndiv, nodes, radius, colors)

    def add_polymesh(self, uuids):
        return self.scene.add_polymesh_object(uuids)

    def snapshot(self, include_hidden: bool = False) -> Snapshot:
        return self.scene.snapshot(include_hidden)


class HeliosBackend(_OpsDispatch):
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

    def add_tile(self, center, size, rotation, subdiv, color=(0.0, 0.75, 0.0)):
        D = self.D
        return self.ctx.addTileObject(D.vec3(*map(float, center)), D.vec2(*map(float, size)),
                                      D.SphericalCoord(1.0, float(rotation[0]), float(rotation[1])),
                                      D.int2(int(subdiv[0]), int(subdiv[1])), self._c(color))

    def add_tube(self, ndiv, nodes, radius, colors=None):
        D = self.D
        cols = None if colors is None else [self._c(c) for c in colors]
        return self.ctx.addTubeObject(int(ndiv), [D.vec3(*map(float, p)) for p in nodes],
                                      [float(r) for r in radius], cols)

    def add_polymesh(self, uuids):
        return self.ctx.addPolymeshObject([int(u) for u in uuids])

    def snapshot(self, include_hidden: bool = False) -> Snapshot:
        return helios_snapshot(self.ctx, include_hidden)


def _all_ids_including_hidden(count, exists, start):
    """getAllUUIDs/getAllObjectIDs skip hidden entries, so enumerate by probing IDs upward until
    ``count`` (which includes hidden ones) existing IDs are found."""
    out, i = [], start
    while len(out) < count:
        if exists(i):
            out.append(i)
        i += 1
    return out


def helios_snapshot(ctx, include_hidden: bool = False) -> Snapshot:
    """Dense dump of a PyHelios Context, sorted by UUID. By default it covers what the bulk getters
    see (getAllUUIDs, which skips hidden primitives); with include_hidden every primitive and object,
    with its hidden flag."""
    if include_hidden:
        uu = np.array(_all_ids_including_hidden(ctx.getPrimitiveCount(), ctx.doesPrimitiveExist, 0), np.int64)
    else:
        uu = np.sort(np.array(ctx.getAllUUIDs(), np.int64))
    n = len(uu)
    ul = [int(u) for u in uu]
    V, nv = np.zeros((n, 4, 3), np.float32), np.zeros(n, np.int32)
    if n:
        flat, off = ctx.getPrimitiveVertices(ul)
        flat = np.asarray(flat, np.float32).reshape(-1, 3)
        off = np.asarray(off, np.int64) // 3
        ends = np.r_[off[1:], len(flat)]
        for i in range(n):
            v = flat[off[i]:ends[i]]
            V[i, : len(v)], nv[i] = v, len(v)
        T = np.asarray(ctx.getPrimitiveType(ul), np.int32).reshape(n)
        A = np.asarray(ctx.getPrimitiveArea(ul), np.float32).reshape(n)
        N = np.asarray(ctx.getPrimitiveNormal(ul), np.float32).reshape(n, 3)
        C = np.asarray(ctx.getPrimitiveColor(ul), np.float32).reshape(n, -1)
    else:
        T, A, N, C = np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    if C.shape[1] == 3:
        C = np.c_[C, np.ones(n, np.float32)]
    o = np.arange(n)
    P = np.array([ctx.getPrimitiveParentObjectID(int(u)) for u in uu[o]], np.int64)
    # textures: batch UV / solid fraction for the explicit UUID list (getAllUUIDs is not sorted). Texture
    # files are read one by one: the batch getter slices the decoded str with C byte offsets, which is
    # wrong for non-ASCII paths (migration/notes/W1-A.md).
    us = [int(u) for u in uu[o]]
    UV, nuv = np.zeros((n, 4, 2), np.float32), np.zeros(n, np.int32)
    SF = np.ones(n, np.float32)
    tex = [ctx.getPrimitiveTextureFile(u) for u in us]
    if n:
        flat, uoff = ctx.getPrimitiveTextureUV(us)
        flat, uoff = np.asarray(flat, np.float32), np.asarray(uoff, np.int64)
        for i in range(n):
            k = (uoff[i + 1] - uoff[i]) // 2
            UV[i, :k], nuv[i] = flat[uoff[i]:uoff[i + 1]].reshape(-1, 2), k
        SF = np.asarray(ctx.getPrimitiveSolidFraction(us), np.float32).reshape(n)
    if include_hidden:
        oids = _all_ids_including_hidden(ctx.getObjectCount(), ctx.doesObjectExist, 1)
    else:
        oids = sorted(ctx.getAllObjectIDs())
    from .snapshot import ObjectTable
    objs = ObjectTable(
        np.array(oids, np.int64),
        np.array([ctx.getObjectType(i) for i in oids], np.int32),
        [np.array(ctx.getObjectPrimitiveUUIDs(i), np.int64) for i in oids],
        np.array([[*_vec(ctx.getObjectCenter(i))] for i in oids], np.float32).reshape(-1, 3),
        np.array([np.asarray(ctx.getObjectTransformationMatrix(i), np.float32).reshape(16) for i in oids],
                 np.float32).reshape(-1, 16),
        hidden=np.array([bool(ctx.isObjectHidden(i)) for i in oids], bool))
    hidden = np.array([bool(ctx.isPrimitiveHidden(u)) for u in us], bool)
    return Snapshot(uu[o], T[o], V[o], nv[o], A[o], N[o], C[o], P, objs, uv=UV, nuv=nuv, texture=tex,
                    solid_fraction=SF, hidden=hidden)


def _vec(v):
    return (v.x, v.y, v.z)
