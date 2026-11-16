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


import numpy as np

from .fileio import FileIOMixin
from .object_transforms import ObjectTransformsMixin
from .objects import CompoundObject, ObjectsMixin  # noqa: F401
from .primitives import PATCH, TRIANGLE, VOXEL, Primitive, triangle_transform  # noqa: F401
from .random import ContextRandom
from .shapes import ShapesMixin
from ..migration_registry import ports

f32 = np.float32


class Scene(ObjectsMixin, ShapesMixin, ObjectTransformsMixin, FileIOMixin):
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
