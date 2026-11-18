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

from .data import DataMixin
from .fileio import FileIOMixin
from .materials import MaterialsMixin
from .object_transforms import ObjectTransformsMixin
from .objects import CompoundObject, ObjectsMixin, TubeEditMixin  # noqa: F401
from .primitives import PATCH, TRIANGLE, VOXEL, Primitive, triangle_transform  # noqa: F401
from .random import ContextRandom
from .texture import Texture, does_texture_file_exist, get_file_extension, resolve_file_path, try_resolve_file_path
from .shapes import ShapesMixin
from ..migration_registry import ports

f32 = np.float32


class Scene(MaterialsMixin, DataMixin, TubeEditMixin, ObjectsMixin, ShapesMixin, ObjectTransformsMixin, FileIOMixin):
    """Geometry + data container (helios::Context counterpart)."""

    def __init__(self):
        self.primitives: dict[int, Primitive] = {}
        self.current_uuid = 0
        self.objects: dict[int, CompoundObject] = {}
        self.current_object_id = 1  # object ID 0 is reserved for "no parent object"
        self.primitive_data: dict[str, dict[int, object]] = {}   # label -> {UUID: data.TypedData}
        self.object_data: dict[str, dict[int, object]] = {}      # label -> {objID: data.TypedData}
        self.rng = ContextRandom()
        self.textures: dict[str, Texture] = {}   # Context::textures, keyed by the file string as passed
        # Primitive::ishidden / CompoundObject::ishidden. Kept outside the Primitive so that copies (which
        # Helios creates visible) do not inherit it; IDs are never reused, so a stale entry is harmless.
        self.hidden_primitives: set[int] = set()
        self.hidden_objects: set[int] = set()
        self._init_materials()

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
    def _new(self, ptype, transform, color, automatic_material: bool = True) -> int:
        c = np.ones(4, f32)
        if color is not None:
            c[: len(color)] = color
        else:
            c[:3] = 0
        u = self.current_uuid
        self.primitives[u] = Primitive(u, ptype, transform.astype(f32), c)
        self.primitive_order.insert(u)
        if automatic_material:
            self._assign_automatic_material(self.primitives[u])
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
        u = self._new(ptype, transform, None, automatic_material=False)
        p = self.primitives[u]
        p.texture_file, p.uv, p.solid_fraction = texture_file, uv, f32(sf)
        self._assign_automatic_material(p)     # black + texture
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

    # ---- color textures (W5-A, rice leaf texture stages; Context.cpp:3366-3436) ----
    @ports("Context::validateColorTexture")
    def validate_color_texture(self, shape_texture_file: str, color_texture_file: str):
        """The color image must match the shape texture's resolution and transparency channel (checked once per pair)."""
        done = self.__dict__.setdefault("_validated_color_textures", set())
        if (shape_texture_file, color_texture_file) in done:
            return
        self.add_texture(shape_texture_file)
        self.add_texture(color_texture_file)
        shape, color = self.textures[shape_texture_file], self.textures[color_texture_file]
        prefix = (f"ERROR (Context::validateColorTexture): The image '{color_texture_file}' cannot give color to primitives "
                  f"whose shape comes from the texture '{shape_texture_file}' because ")
        ss, cs = shape.get_image_resolution(), color.get_image_resolution()
        if tuple(ss) != tuple(cs):
            raise ValueError(prefix + f"their resolutions differ ({cs[0]}x{cs[1]} and {ss[0]}x{ss[1]}). Give it the same "
                             "resolution.")
        if shape.has_transparency() != color.has_transparency():
            raise ValueError(prefix + "only one of them has a transparency channel. Save both in a format with the same "
                             "transparency, for example PNG.")
        if shape.has_transparency() and not np.array_equal(shape.transparency_data, color.transparency_data):
            raise ValueError(prefix + "their transparency differs, so they have different outlines. Give it exactly the "
                             "same transparency channel. To change a primitive's shape, change its texture instead.")
        done.add((shape_texture_file, color_texture_file))

    @ports("Context::setPrimitiveColorTexture")
    def set_primitive_color_texture(self, uuids, color_texture_file: str):
        """Stored on the primitive as ``color_texture_file`` ("" = none; copied with the primitive like Helios's
        ``color_texture_ID``). Geometry, material and solid fraction are unchanged."""
        uuids = [int(u) for u in (uuids if isinstance(uuids, (list, tuple, np.ndarray)) else [uuids])]
        if color_texture_file:
            for u in uuids:
                shape = self.primitives[u].texture_file or ""
                if not shape:
                    raise ValueError(f"ERROR (Context::setPrimitiveColorTexture): Primitive {u} has no texture, so there is "
                                     f"no shape for the image '{color_texture_file}' to match. Give the primitive a "
                                     "texture, or use Context::setPrimitiveTextureFile() or a material to give it this "
                                     "image as its texture.")
                self.validate_color_texture(shape, color_texture_file)
        for u in uuids:
            self.primitives[u].color_texture_file = color_texture_file

    @ports("Context::getPrimitiveColorTextureFile")
    def primitive_color_texture_file(self, uuid: int) -> str:
        return getattr(self.primitives[uuid], "color_texture_file", "") or self.primitive_texture_file(uuid)

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
    def all_uuids(self, include_hidden: bool = False) -> list[int]:
        """Sorted UUIDs. Hidden primitives are skipped, as Helios does (include_hidden is a Heliades
        extension for snapshots of prototype geometry)."""
        hid = self.hidden_primitives
        return sorted(u for u in self.primitives if include_hidden or u not in hid)

    @ports("Context::getPrimitiveCount")
    def primitive_count(self, include_hidden: bool = True) -> int:
        if include_hidden:
            return len(self.primitives)
        return sum(1 for u in self.primitives if u not in self.hidden_primitives)

    @ports("Context::doesPrimitiveExist")
    def does_primitive_exist(self, uuid: int) -> bool:
        return int(uuid) in self.primitives

    # ---- hidden primitives and objects ----
    @ports("Context::hidePrimitive(uint)", "Context::hidePrimitive(const std::vector<uint>&)")
    def hide_primitive(self, uuids):
        for u in _ids(uuids):
            if u not in self.primitives:                 # primitives.at() throws (no debug check)
                raise ValueError(f"ERROR (Context::hidePrimitive): UUID of {u} does not exist in the Context.")
            self.hidden_primitives.add(u)

    @ports("Context::showPrimitive(uint)", "Context::showPrimitive(const std::vector<uint>&)")
    def show_primitive(self, uuids):
        for u in _ids(uuids):
            if u not in self.primitives:
                raise ValueError(f"ERROR (Context::showPrimitive): UUID of {u} does not exist in the Context.")
            self.hidden_primitives.discard(u)

    @ports("Context::isPrimitiveHidden")
    def is_primitive_hidden(self, uuid: int) -> bool:
        if int(uuid) not in self.primitives:
            raise ValueError(f"ERROR (Context::isPrimitiveHidden): UUID of {uuid} does not exist in the Context.")
        return int(uuid) in self.hidden_primitives

    @ports("Context::hideObject(uint)", "Context::hideObject(const std::vector<uint>&)")
    def hide_object(self, oids):
        """Hides the object and every member primitive. Hidden objects still occupy their IDs and
        UUIDs, can be copied (the copies are visible) and transformed."""
        for o in _ids(oids):
            obj = self.objects.get(o)
            if obj is None:
                raise ValueError(f"ERROR (Context::hideObject): Object ID of {o} does not exist in the Context.")
            self.hidden_objects.add(o)
            for u in obj.uuids:
                if u not in self.primitives:
                    raise ValueError(f"ERROR (Context::hideObject): Primitive UUID of {u} does not exist in the Context.")
                self.hidden_primitives.add(u)

    @ports("Context::showObject(uint)", "Context::showObject(const std::vector<uint>&)")
    def show_object(self, oids):
        for o in _ids(oids):
            obj = self.objects.get(o)
            if obj is None:
                raise ValueError(f"ERROR (Context::showObject): Object ID of {o} does not exist in the Context.")
            self.hidden_objects.discard(o)
            for u in obj.uuids:
                if u not in self.primitives:
                    raise ValueError(f"ERROR (Context::showObject): Primitive UUID of {u} does not exist in the Context.")
                self.hidden_primitives.discard(u)

    @ports("Context::isObjectHidden")
    def is_object_hidden(self, oid: int) -> bool:
        if int(oid) not in self.objects:
            raise ValueError(f"ERROR (Context::isObjectHidden): Object ID of {oid} does not exist in the Context.")
        return int(oid) in self.hidden_objects

    # ---- bookkeeping wrapped around W1-B's copy / delete ----
    def _copy_primitive_one(self, uuid: int) -> int:
        new = super()._copy_primitive_one(uuid)
        self.primitive_order.insert(new)
        self._material_reference_copied(self.primitives[new])     # same material, one more user
        return new

    def delete_primitives(self, uuids):
        ids = _ids(uuids)
        alive = {u: self.primitives[u] for u in ids if u in self.primitives}
        try:
            super().delete_primitives(uuids)
        finally:
            for u, p in alive.items():
                if u not in self.primitives:
                    self._material_reference_deleted(p)
                    self.primitive_order.erase(u)
                    self.hidden_primitives.discard(u)

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

    # ---- bulk ----
    def snapshot(self, include_hidden: bool = False) -> "Snapshot":
        """Dense dump sorted by UUID: by default what Helios's bulk getters see (getAllUUIDs /
        getAllObjectIDs skip hidden primitives and objects); with include_hidden, everything, with the
        ``hidden`` flags set."""
        from ..parity.snapshot import ObjectTable, Snapshot
        uu = self.all_uuids(include_hidden)
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
        oids = self.all_object_ids(include_hidden)
        objs = ObjectTable(np.array(oids, np.int64),
                           np.array([self.objects[o].type for o in oids], np.int32),
                           [np.array(self.objects[o].uuids, np.int64) for o in oids],
                           np.array([self.object_center(o) for o in oids], f32).reshape(-1, 3),
                           np.array([self.objects[o].transform for o in oids], f32).reshape(-1, 16),
                           hidden=np.array([o in self.hidden_objects for o in oids], bool))
        return Snapshot(np.array(uu, np.int64), T, Vt, nv, A, N, C, P, objs,
                        uv=UV, nuv=nuv, texture=tex, solid_fraction=SF,
                        hidden=np.array([u in self.hidden_primitives for u in uu], bool))


def _ids(x) -> list[int]:
    return [int(x)] if np.isscalar(x) else [int(i) for i in x]


@ports("helios::resolvePluginAsset")
def resolve_plugin_asset(plugin_name: str, asset_path: str) -> str:
    """resolveFilePath("plugins/<plugin>/<asset>"): cwd first, then the build directory ($HELIOS_BUILD,
    else cwd or its parent if it contains plugins/). Raises if the file is not found."""
    return resolve_file_path(f"plugins/{plugin_name}/{asset_path}")


@ports("helios::tryResolvePluginAsset")
def try_resolve_plugin_asset(plugin_name: str, asset_path: str) -> str:
    """Like resolve_plugin_asset, but '' instead of raising."""
    return try_resolve_file_path(f"plugins/{plugin_name}/{asset_path}")
