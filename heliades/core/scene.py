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
from .texture import Texture, does_texture_file_exist, get_file_extension, resolve_file_path
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
        self.textures: dict[str, Texture] = {}   # Context::textures, keyed by the file string as passed

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

    # ---- textures ----
    @ports("Context::addTexture")
    def add_texture(self, texture_file: str):
        if texture_file not in self.textures:
            ext = get_file_extension(texture_file)
            if ext not in (".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"):
                raise ValueError(f"ERROR (Context::addTexture): Texture file {texture_file} is not PNG or JPEG format.")
            if not does_texture_file_exist(texture_file):
                raise FileNotFoundError(f"ERROR (Context::addTexture): Texture file {texture_file} does not exist.")
            self.textures[texture_file] = Texture(resolve_file_path(texture_file))

    def _new_textured(self, ptype, transform, texture_file: str, uv) -> int:
        """Patch/Triangle(texturefile, uv, textures, ...): clamp uv to <= 1, solid fraction from the
        (cached) texture, material color black. The texture must already be registered."""
        uv = np.minimum(np.asarray(uv, f32).reshape(-1, 2), f32(1.0))   # std::min(uv, 1.f); lower end unclamped
        sf = self.textures[texture_file].get_solid_fraction(uv)
        u = self._new(ptype, transform, None)
        p = self.primitives[u]
        p.texture_file, p.uv, p.solid_fraction = texture_file, uv, f32(sf)
        return u

    def _place_patch(self, p, size, rotation, center):
        p.scale((size[0], size[1], 1))
        el, az = f32(rotation[0]), f32(rotation[1])
        if el != 0:
            p.rotate(-el, "x")
        if az != 0:
            p.rotate(-az, "z")
        p.translate(center)

    @ports("Context::addPatch(texture)", "Context::addPatch(texture,uv_center,uv_size)")
    def add_patch_textured(self, center, size, rotation, texture_file: str, uv_center=None, uv_size=None) -> int:
        """Textured patch. Without uv_center/uv_size the size is only assert()ed in Helios (not checked
        in release builds); with them it is checked like the colored overload."""
        size = np.asarray(size, f32)
        if uv_center is None:
            self.add_texture(texture_file)
            uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], f32)
        else:
            if size[0] < f32(1e-6) or size[1] < f32(1e-6):
                raise ValueError("ERROR (Context::addPatch): Size of patch must be greater than 1e-6 to avoid numerical precision issues.")
            uc, us = np.asarray(uv_center, f32), np.asarray(uv_size, f32)
            # bounds test is in double: uv_center.x - 0.5 * uv_size.x < -1e-3 ...
            cx, cy, sx, sy = (float(x) for x in (uc[0], uc[1], us[0], us[1]))
            if cx - 0.5 * sx < -1e-3 or cy - 0.5 * sy < -1e-3 or cx + 0.5 * sx - 1.0 > 1e-3 or cy + 0.5 * sy - 1.0 > 1e-3:
                raise ValueError("ERROR (Context::addPatch): Invalid texture coordinates. uv_center-0.5*uv_size should be >=0 and uv_center+0.5*uv_size should be <=1.")
            self.add_texture(texture_file)
            hx, hy = f32(f32(0.5) * us[0]), f32(f32(0.5) * us[1])
            uv = np.array([[uc[0] + -hx, uc[1] + -hy], [uc[0] + hx, uc[1] + -hy],
                           [uc[0] + hx, uc[1] + hy], [uc[0] + -hx, uc[1] + hy]], f32)
        u = self._new_textured(PATCH, np.eye(4, dtype=f32).reshape(16), texture_file, uv)
        self._place_patch(self.primitives[u], size, rotation, center)
        return u

    @ports("Context::addTriangle(texture,uv0,uv1,uv2)")
    def add_triangle_textured(self, v0, v1, v2, texture_file: str, uv0, uv1, uv2) -> int:
        self.add_texture(texture_file)
        return self._new_textured(TRIANGLE, triangle_transform(v0, v1, v2), texture_file, [uv0, uv1, uv2])

    @ports("Context::getPrimitiveTextureFile")
    def primitive_texture_file(self, uuid: int) -> str:
        return self.primitives[uuid].texture_file or ""

    @ports("Context::getPrimitiveTextureUV")
    def primitive_texture_uv(self, uuid: int) -> np.ndarray:
        return self.primitives[uuid].texture_uv()

    @ports("Context::getPrimitiveSolidFraction")
    def primitive_solid_fraction(self, uuid: int) -> np.float32:
        return f32(self.primitives[uuid].solid_fraction)

    @ports("Primitive::hasTexture")
    def primitive_has_texture(self, uuid: int) -> bool:
        return self.primitives[uuid].has_texture()

    @ports("Context::getPrimitiveTextureSize")
    def primitive_texture_size(self, uuid: int) -> tuple[int, int]:
        t = self.primitives[uuid].texture_file
        return self.textures[t].get_image_resolution() if t and t in self.textures else (0, 0)

    @ports("Context::primitiveTextureHasTransparencyChannel")
    def primitive_texture_has_transparency_channel(self, uuid: int) -> bool:
        t = self.primitives[uuid].texture_file
        return bool(t) and t in self.textures and self.textures[t].has_transparency()

    @ports("Context::getPrimitiveTextureTransparencyData")
    def primitive_texture_transparency_data(self, uuid: int) -> np.ndarray:
        if not self.primitive_texture_has_transparency_channel(uuid):
            raise ValueError(f"ERROR (Context::getPrimitiveTransparencyData): Texture transparency data does not exist for primitive {uuid}.")
        return self.textures[self.primitives[uuid].texture_file].transparency_data

    @ports("Context::deletePrimitive")
    def delete_primitive(self, uuid: int):
        """Also drops it from its parent object and deletes the object once empty (W1-B)."""
        self.delete_primitives(uuid)

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
        # members of tiles/tubes (etc.) are locked in Helios: warning + no-op
        if not self._transform_locked(self.primitives[uuid]):
            self.primitives[uuid].translate(shift)

    @ports("Context::rotatePrimitive")
    def rotate_primitive(self, uuid: int, radians: float, axis: str):
        if not self._transform_locked(self.primitives[uuid]):
            self.primitives[uuid].rotate(radians, axis)

    @ports("Context::scalePrimitive")
    def scale_primitive(self, uuid: int, s):
        if not self._transform_locked(self.primitives[uuid]):
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
        UV, nuv, SF = np.zeros((n, 4, 2), f32), np.zeros(n, np.int32), np.ones(n, f32)
        tex = []
        for i, u in enumerate(uu):
            p = self.primitives[u]
            v = p.vertices()
            Vt[i, : len(v)], nv[i] = v, len(v)
            A[i], N[i], C[i], T[i], P[i] = p.area(), p.normal(), p.color, p.type, p.parent_object
            if p.uv is not None:
                UV[i, : len(p.uv)], nuv[i] = p.uv, len(p.uv)
            SF[i] = p.solid_fraction
            tex.append(p.texture_file or "")
        oids = self.all_object_ids()
        objs = ObjectTable(np.array(oids, np.int64),
                           np.array([self.objects[o].type for o in oids], np.int32),
                           [np.array(self.objects[o].uuids, np.int64) for o in oids],
                           np.array([self.object_center(o) for o in oids], f32).reshape(-1, 3),
                           np.array([self.objects[o].transform for o in oids], f32).reshape(-1, 16))
        return Snapshot(np.array(uu, np.int64), T, Vt, nv, A, N, C, P, objs,
                        uv=UV, nuv=nuv, texture=tex, solid_fraction=SF)
