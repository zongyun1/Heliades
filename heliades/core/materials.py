# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context.cpp "MATERIAL MANAGEMENT METHODS", Context_primitive.cpp
# Primitive::setColor, Context_object.cpp keepNamedTubeMaterial; Copyright (C) 2016-2026 Brian Bailey).
"""A minimal material table (mixed into Scene).

In Helios a primitive's color and texture live in its *material*. Every new primitive gets an
automatic material, labelled ``"__auto_<hash of its appearance>"`` and shared by every primitive of
the same appearance. Named materials are created with ``addMaterial`` or by renaming an automatic
one (plantarchitecture's ``nameOrganMaterials``). Editing a named material re-colors every primitive
that uses it, and editing a primitive's color moves it to another material: the automatic material of
its new appearance, or a ``"__copy_<ID>_<UUID>"`` copy when its named material is shared.

Heliades keeps ``Primitive.color`` / ``Primitive.texture_file`` as the drawn appearance, which
everything else reads, and stores the material ID in ``Primitive.material_id``. This module keeps the
two consistent: every change of a material's color or texture is written to each primitive using it.
Reference counts are kept incrementally as in Helios (creation, copy, delete, reassignment).

Covered: material IDs and labels (automatic labels hash exactly as libstdc++'s ``std::hash`` does, so
``getPrimitiveMaterialLabel`` matches Helios), color, texture (with the solid-fraction update), texture
color override flag, two-sided flag, reference counts, rename, delete, assignment to primitives and
objects, and ``setPrimitiveColor``. Not covered: normal/bump maps, material data, color textures,
spectral textures and the revision counters.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..migration_registry import ports

f32 = np.float32

DEFAULT_MATERIAL_LABEL = "__default__"
_M64 = (1 << 64) - 1
_MUL = (0xC6A4A793 << 32) + 0x5BD1E995
_SEED = 0xC70F6907


def _shift_mix(v: int) -> int:
    return v ^ (v >> 47)


def hash_bytes(b: bytes, seed: int = _SEED) -> int:
    """libstdc++ ``std::_Hash_bytes`` for a 64-bit size_t (MurmurHash64A variant, hash_bytes.cc)."""
    n = len(b)
    aligned = n & ~7
    h = (seed ^ ((n * _MUL) & _M64)) & _M64
    for i in range(0, aligned, 8):
        d = int.from_bytes(b[i:i + 8], "little")
        d = (_shift_mix((d * _MUL) & _M64) * _MUL) & _M64
        h = ((h ^ d) * _MUL) & _M64
    if n & 7:
        d = int.from_bytes(b[aligned:], "little")
        h = ((h ^ d) * _MUL) & _M64
    h = (_shift_mix(h) * _MUL) & _M64
    return _shift_mix(h)


def _hash_float(x) -> int:
    """std::hash<float>: 0 for +-0, else _Hash_bytes of the 4 bytes."""
    x = f32(x)
    return 0 if x == 0 else hash_bytes(struct.pack("<f", float(x)))


@ports("Context::generateMaterialLabel")
def generate_material_label(color, texture_file: str = "", texture_override: bool = False) -> str:
    """``"__auto_" + std::to_string(hash)`` of (r, g, b, a, texture, override), combined with the
    boost-style ``hash ^= h + 0x9e3779b9 + (hash << 6) + (hash >> 2)`` (no normal map)."""
    h = 0
    c = np.asarray(color, f32)
    for v in (_hash_float(c[0]), _hash_float(c[1]), _hash_float(c[2]), _hash_float(c[3]),
              hash_bytes(texture_file.encode("utf-8")), int(bool(texture_override))):
        h ^= (v + 0x9E3779B9 + ((h << 6) & _M64) + (h >> 2)) & _M64
    return "__auto_" + str(h)


def _rgba(color) -> np.ndarray:
    """RGBAcolor(r, g, b, a) (alpha 1 for an RGB color); the constructor clamps each component to [0, 1]."""
    c = np.ones(4, f32)
    v = np.asarray(color, f32).reshape(-1)
    c[: len(v)] = v
    return np.where(c < 0, f32(0), np.where(c > 1, f32(1), c)).astype(f32)


def _rgba_equal(a, b) -> bool:
    """RGBAcolor::operator==: every component within 1e-6f (float subtraction)."""
    return all(abs(f32(f32(x) - f32(y))) < f32(1e-6) for x, y in zip(a, b))


@dataclass
class Material:
    id: int
    label: str
    color: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1], f32))   # RGBA float32
    texture_file: str = ""
    texture_color_overridden: bool = False
    twosided_flag: int = 1
    reference_count: int = 0


def _uuid_list(x) -> list[int]:
    return [int(x)] if np.isscalar(x) else [int(i) for i in x]


class MaterialsMixin:
    """Context material API (subset); ``_init_materials`` is called by ``Scene.__init__``."""

    def _init_materials(self):
        self.materials: dict[int, Material] = {0: Material(0, DEFAULT_MATERIAL_LABEL)}
        self.material_label_to_id: dict[str, int] = {DEFAULT_MATERIAL_LABEL: 0}
        self.current_material_id = 1
        self.materials_renamed_from_automatic: set[int] = set()
        self._auto_label_cache: dict[tuple, str] = {}

    # ---- internals ---------------------------------------------------------------------------

    def _auto_label(self, color, texture: str, override: bool) -> str:
        c = _rgba(color)
        key = (c.tobytes(), texture, bool(override))
        lbl = self._auto_label_cache.get(key)
        if lbl is None:
            lbl = self._auto_label_cache[key] = generate_material_label(c, texture, override)
        return lbl

    def _material_id(self, label: str, who: str = "getMaterialIDFromLabel") -> int:
        i = self.material_label_to_id.get(label)
        if i is None:
            raise ValueError(f"ERROR (Context::{who}): Material with label '{label}' does not exist.")
        return i

    @ports("Context::addMaterial_internal")
    def _add_material_internal(self, label: str, color, texture: str) -> int:
        if not label:
            raise ValueError("ERROR (Context::addMaterial_internal): Material label cannot be empty.")
        i = self.material_label_to_id.get(label)
        if i is not None:            # replaced in place, keeping the ID and the reference count
            rc = self.materials[i].reference_count
            old_tex = self.materials[i].texture_file
            self.materials[i] = Material(i, label, _rgba(color), texture, False, 1, rc)
            self._refresh_material_users(i, old_tex != texture)
            return i
        i = self.current_material_id
        self.current_material_id += 1
        self.materials[i] = Material(i, label, _rgba(color), texture, False)
        self.material_label_to_id[label] = i
        return i

    def _assign_automatic_material(self, p):
        """What addPatch/addTriangle do right after creating a primitive: reuse or create the automatic
        material of its appearance and count the reference."""
        tex = p.texture_file or ""
        lbl = self._auto_label_cache.get((p.color.tobytes(), tex, False))     # fast path (p.color is RGBA f32)
        if lbl is None:
            lbl = self._auto_label(p.color, tex, False)
        i = self.material_label_to_id.get(lbl)
        if i is None:
            i = self._add_material_internal(lbl, p.color, tex)
        p.material_id = i
        self.materials[i].reference_count += 1

    def _primitive_material_id(self, p) -> int:
        i = getattr(p, "material_id", None)
        if i is None:              # created by a path that bypassed Scene._new: give it its automatic material
            self._assign_automatic_material(p)
            i = p.material_id
        return i

    def _sync_appearance(self, p):
        m = self.materials[p.material_id]
        p.color = m.color.copy()
        p.texture_file = m.texture_file or None

    @ports("Context::updatePrimitiveSolidFraction")
    def _update_solid_fraction(self, p):
        from .primitives import PATCH
        tex = self.materials[p.material_id].texture_file
        if not tex:
            p.solid_fraction = f32(1.0)
            return
        self.add_texture(tex)
        uv = p.texture_uv()
        if len(uv) == 0 and p.type == PATCH:          # a patch given no (u,v) maps the whole image
            uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], f32)
        p.solid_fraction = f32(self.textures[tex].get_solid_fraction(uv))

    @ports("Context::markMaterialPrimitivesModified")
    def _refresh_material_users(self, mid: int, texture_changed: bool):
        if self.materials[mid].reference_count == 0:
            return
        for p in self.primitives.values():
            if getattr(p, "material_id", None) == mid:
                self._sync_appearance(p)
                if texture_changed:
                    self._update_solid_fraction(p)

    @ports("Context::setPrimitiveMaterial")
    def _set_primitive_material(self, p, mid: int):
        old = self._primitive_material_id(p)
        if old == mid:
            return
        if old in self.materials:
            self.materials[old].reference_count -= 1
        self.materials[mid].reference_count += 1
        p.material_id = mid
        old_tex = self.materials[old].texture_file if old in self.materials else ""
        self._sync_appearance(p)
        if old_tex != self.materials[mid].texture_file:
            self._update_solid_fraction(p)

    @ports("Context::copyMaterialForPrimitive")
    def _copy_material_for_primitive(self, p) -> int:
        old = self.materials[p.material_id]
        new = self._add_material_internal(f"__copy_{self.current_material_id}_{p.uuid}", old.color, old.texture_file)
        m = self.materials[new]
        m.texture_color_overridden, m.twosided_flag = old.texture_color_overridden, old.twosided_flag
        old.reference_count -= 1
        m.reference_count = 1
        p.material_id = new
        return new

    @ports("Context::setPrimitiveAppearance")
    def _set_primitive_appearance(self, p, color, texture: str, override: bool):
        cur = self.materials[self._primitive_material_id(p)]
        automatic = cur.label.startswith("__auto_") or cur.label == DEFAULT_MATERIAL_LABEL
        if automatic:                 # (material data is not modelled, so an automatic material carries none)
            lbl = self._auto_label(color, texture, override)
            i = self.material_label_to_id.get(lbl)
            if i is None:
                i = self._add_material_internal(lbl, color, texture)
                self.materials[i].texture_color_overridden = bool(override)
            self._set_primitive_material(p, i)
            return
        if p.material_id in self.materials_renamed_from_automatic or cur.reference_count > 1:
            self._copy_material_for_primitive(p)
        m = self.materials[p.material_id]
        texture_changed = m.texture_file != texture
        m.color, m.texture_file, m.texture_color_overridden = _rgba(color), texture, bool(override)
        self._sync_appearance(p)
        if texture_changed:
            self._update_solid_fraction(p)

    def _material_reference_deleted(self, p):
        i = getattr(p, "material_id", None)
        if i is not None and i in self.materials:
            self.materials[i].reference_count -= 1

    def _material_reference_copied(self, p):
        i = getattr(p, "material_id", None)
        if i is not None and i in self.materials:
            self.materials[i].reference_count += 1

    # ---- public API --------------------------------------------------------------------------

    @ports("Context::addMaterial")
    def add_material(self, label: str):
        if not label:
            raise ValueError("ERROR (Context::addMaterial): Material label cannot be empty.")
        if label.startswith("__") and label != DEFAULT_MATERIAL_LABEL:
            raise ValueError("ERROR (Context::addMaterial): Material labels starting with '__' are reserved for internal use.")
        i = self.material_label_to_id.get(label)
        if i is not None:            # overwritten in place (black, no texture), keeping ID and reference count
            m = self.materials[i]
            texture_changed = bool(m.texture_file)
            self.materials[i] = Material(i, label, reference_count=m.reference_count)
            self._refresh_material_users(i, texture_changed)
            return
        i = self.current_material_id
        self.current_material_id += 1
        self.materials[i] = Material(i, label)
        self.material_label_to_id[label] = i

    @ports("Context::doesMaterialExist")
    def does_material_exist(self, label: str) -> bool:
        return label in self.material_label_to_id

    @ports("Context::listMaterials")
    def list_materials(self) -> list[str]:
        """Named (and copied) materials; sorted, as Helios iterates an unordered_map."""
        return sorted(lbl for lbl in self.material_label_to_id
                      if lbl != DEFAULT_MATERIAL_LABEL and not lbl.startswith("__auto_"))

    @ports("Context::getMaterialCount")
    def get_material_count(self) -> int:
        return len(self.list_materials())

    @ports("Context::getMaterialIDFromLabel")
    def get_material_id_from_label(self, label: str) -> int:
        return self._material_id(label)

    @ports("Context::getMaterial")
    def get_material(self, material_id: int) -> Material:
        m = self.materials.get(int(material_id))
        if m is None:
            raise ValueError(f"ERROR (Context::getMaterial): Material ID {material_id} does not exist.")
        return m

    @ports("Context::getMaterialColor")
    def get_material_color(self, label: str) -> np.ndarray:
        return self.materials[self._material_id(label)].color.copy()

    @ports("Context::getMaterialTexture")
    def get_material_texture(self, label: str) -> str:
        return self.materials[self._material_id(label)].texture_file

    @ports("Context::isMaterialTextureColorOverridden")
    def is_material_texture_color_overridden(self, label: str) -> bool:
        return self.materials[self._material_id(label)].texture_color_overridden

    @ports("Context::getMaterialTwosidedFlag")
    def get_material_twosided_flag(self, label: str) -> int:
        return self.materials[self._material_id(label)].twosided_flag

    @ports("Context::setMaterialTwosidedFlag")
    def set_material_twosided_flag(self, label: str, flag: int):
        self.materials[self._material_id(label)].twosided_flag = int(flag)

    @ports("Context::setMaterialColor")
    def set_material_color(self, label: str, color):
        i = self._material_id(label)
        c = _rgba(color)
        if _rgba_equal(self.materials[i].color, c):
            return
        self.materials[i].color = c
        self._refresh_material_users(i, False)

    @ports("Context::setMaterialTexture")
    def set_material_texture(self, label: str, texture_file: str):
        i = self._material_id(label)
        if texture_file:
            self.add_texture(texture_file)
        if self.materials[i].texture_file == texture_file:
            return
        self.materials[i].texture_file = texture_file
        self._refresh_material_users(i, True)

    @ports("Context::setMaterialTextureColorOverride")
    def set_material_texture_color_override(self, label: str, override: bool):
        i = self._material_id(label)
        if self.materials[i].texture_color_overridden == bool(override):
            return
        self.materials[i].texture_color_overridden = bool(override)

    @ports("Context::renameMaterial")
    def rename_material(self, old_label: str, new_label: str):
        if old_label not in self.material_label_to_id:
            raise ValueError(f"ERROR (Context::renameMaterial): Material with label '{old_label}' does not exist.")
        if not new_label:
            raise ValueError("ERROR (Context::renameMaterial): New material label cannot be empty.")
        if new_label.startswith("__"):
            raise ValueError("ERROR (Context::renameMaterial): Material labels starting with '__' are reserved for internal use.")
        if new_label in self.material_label_to_id:
            raise ValueError(f"ERROR (Context::renameMaterial): Material with label '{new_label}' already exists.")
        i = self.material_label_to_id.pop(old_label)
        self.materials[i].label = new_label
        if old_label.startswith("__auto_"):
            self.materials_renamed_from_automatic.add(i)
        self.material_label_to_id[new_label] = i

    @ports("Context::deleteMaterial")
    def delete_material(self, label: str):
        if label == DEFAULT_MATERIAL_LABEL:
            raise ValueError("ERROR (Context::deleteMaterial): Cannot delete the default material.")
        i = self._material_id(label, "deleteMaterial")
        for u in self.get_primitives_using_material(label):
            self._set_primitive_material(self.primitives[u], 0)
        del self.materials[i]
        del self.material_label_to_id[label]
        self.materials_renamed_from_automatic.discard(i)

    @ports("Context::assignMaterialToPrimitive(uint,const std::string&)",
           "Context::assignMaterialToPrimitive(const std::vector<uint>&,const std::string&)")
    def assign_material_to_primitive(self, uuids, label: str):
        i = self._material_id(label)
        ps = []
        for u in _uuid_list(uuids):              # every primitive is checked before any is changed
            p = self.primitives.get(u)
            if p is None:
                raise ValueError(f"ERROR (Context::assignMaterialToPrimitive): UUID of {u} does not exist in the Context.")
            ps.append(p)
        for p in ps:
            self._set_primitive_material(p, i)

    @ports("Context::assignMaterialToObject(uint,const std::string&)",
           "Context::assignMaterialToObject(const std::vector<uint>&,const std::string&)")
    def assign_material_to_object(self, oids, label: str):
        for o in _uuid_list(oids):
            if o not in self.objects:
                raise ValueError(f"ERROR (Context::getObjectPrimitiveUUIDs): Object ID of {o} not found in the context.")
            self.assign_material_to_primitive(list(self.objects[o].uuids), label)

    @ports("Context::getPrimitiveMaterialLabel")
    def get_primitive_material_label(self, uuid: int) -> str:
        i = self._primitive_material_id(self.primitives[int(uuid)])
        m = self.materials.get(i)
        return m.label if m is not None else DEFAULT_MATERIAL_LABEL

    @ports("Context::getPrimitiveMaterialID")
    def get_primitive_material_id(self, uuid: int) -> int:
        return self._primitive_material_id(self.primitives[int(uuid)])

    @ports("Context::getPrimitivesUsingMaterial")
    def get_primitives_using_material(self, label: str) -> list[int]:
        """Sorted by UUID (Helios iterates its primitive hash map)."""
        i = self._material_id(label)
        return sorted(u for u, p in self.primitives.items() if self._primitive_material_id(p) == i)

    @ports("Context::setPrimitiveColor(uint,const RGBcolor&)", "Context::setPrimitiveColor(const std::vector<uint>&,const RGBcolor&)",
           "Context::setPrimitiveColor(uint,const RGBAcolor&)", "Context::setPrimitiveColor(const std::vector<uint>&,const RGBAcolor&)",
           "Primitive::setColor")
    def set_primitive_color(self, uuids, color):
        """Primitive::setColor: the primitive's appearance with a new color (alpha 1 for an RGB color)."""
        for u in _uuid_list(uuids):
            p = self.primitives.get(u)
            if p is None:
                raise ValueError(f"ERROR (Context::setPrimitiveColor): UUID of {u} does not exist in the Context.")
            m = self.materials[self._primitive_material_id(p)]
            self._set_primitive_appearance(p, _rgba(color), m.texture_file, m.texture_color_overridden)

    @ports("Context::getPrimitiveColor", "Context::getPrimitiveColorRGBA")
    def get_primitive_color(self, uuid: int) -> np.ndarray:
        """RGBA of the primitive's material."""
        return self.materials[self._primitive_material_id(self.primitives[int(uuid)])].color.copy()

    @ports("keepNamedTubeMaterial (Context_object.cpp)")
    def _keep_named_tube_material(self, earlier_uuids, all_uuids, appended_texture: str, appended_color):
        """Give the triangles just appended to a tube the named material all its earlier triangles share,
        when they were appended with that material's texture (or, untextured, its RGB color)."""
        import os
        if not earlier_uuids:
            return
        label = self.get_primitive_material_label(earlier_uuids[0])
        if label.startswith("__"):
            return
        for u in earlier_uuids:
            if self.get_primitive_material_label(u) != label:
                return
        tex = self.get_material_texture(label)
        if bool(tex) != bool(appended_texture):
            return
        if tex:
            same = tex == appended_texture
            if not same:
                try:
                    same = os.path.samefile(tex, appended_texture)      # std::filesystem::equivalent
                except OSError:
                    same = False
            if not same:
                return
        else:
            mc, ac = self.get_material_color(label), np.asarray(appended_color, f32)
            if mc[0] != ac[0] or mc[1] != ac[1] or mc[2] != ac[2]:
                return
        earlier = set(earlier_uuids)
        self.assign_material_to_primitive([u for u in all_uuids if u not in earlier], label)
