# SPDX-License-Identifier: LGPL-2.1-or-later
"""Material-table operations for both backends (W2-E): the rest of the material API, and a full dump
of the material state (labels, IDs, reference counts, users, per-primitive appearance).

PyHelios exposes no reference count, so the Helios side reads ``Material::reference_count`` from the
``const Material&`` that ``Context::getMaterial(uint)`` returns, through ctypes. The field offset (140
on x86-64 libstdc++: uint ID, std::string label, RGBAcolor, std::string texture, bool, uint twosided,
std::string normal map, enum kind, float strength, bool flip) is checked on every read against the
material ID at offset 0 and the color at offset 40.
"""
from __future__ import annotations

import ctypes
import struct

import numpy as np

from ...ops import op

_RC_OFFSET, _COLOR_OFFSET = 140, 40
_get_material = None


def _ids(x):
    return int(x) if np.isscalar(x) else [int(i) for i in x]


def _codes(s: str) -> list[float]:
    return [float(len(s))] + [float(b) for b in s.encode("utf-8")]


def _base(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _helios_reference_count(B, mid: int, color) -> int:
    global _get_material
    if _get_material is None:
        from pyhelios.wrappers import UContextWrapper as W
        f = W.helios_lib._ZNK6helios7Context11getMaterialEj
        f.restype, f.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_uint]
        _get_material = f
    raw = ctypes.string_at(_get_material(ctypes.cast(B.ctx.context, ctypes.c_void_p).value, int(mid)), 144)
    if struct.unpack_from("<I", raw, 0)[0] != mid or struct.unpack_from("<4f", raw, _COLOR_OFFSET) != tuple(color):
        raise RuntimeError("helios::Material layout differs from the one ops_materials assumes")
    return struct.unpack_from("<I", raw, _RC_OFFSET)[0]


# ---- the rest of the material API ------------------------------------------------------------

@op("delete_material", "helios")
def _(B, label):
    B.ctx.deleteMaterial(label)


@op("delete_material", "heliades")
def _(B, label):
    B.scene.delete_material(label)


@op("set_material_texture_color_override", "helios")
def _(B, label, flag):
    B.ctx.setMaterialTextureColorOverride(label, bool(flag))


@op("set_material_texture_color_override", "heliades")
def _(B, label, flag):
    B.scene.set_material_texture_color_override(label, bool(flag))


@op("set_material_twosided_flag", "helios")
def _(B, label, flag):
    B.ctx.setMaterialTwosidedFlag(label, int(flag))


@op("set_material_twosided_flag", "heliades")
def _(B, label, flag):
    B.scene.set_material_twosided_flag(label, int(flag))


@op("override_primitive_texture_color", "helios")
def _(B, uuids):
    B.ctx.overridePrimitiveTextureColor(_ids(uuids))


@op("override_primitive_texture_color", "heliades")
def _(B, uuids):
    B.scene.override_primitive_texture_color(_ids(uuids))


@op("use_primitive_texture_color", "helios")
def _(B, uuids):
    B.ctx.usePrimitiveTextureColor(_ids(uuids))


@op("use_primitive_texture_color", "heliades")
def _(B, uuids):
    B.scene.use_primitive_texture_color(_ids(uuids))


@op("material_list_raw", "helios")
def _(B):
    """listMaterials in the order Helios returns it (std::map: bytewise label order)."""
    return list(B.ctx.listMaterials())


@op("material_list_raw", "heliades")
def _(B):
    return B.scene.list_materials()


# ---- full state dump -------------------------------------------------------------------------

def _state(B, extra_labels, *, uuids, prim_label, prim_id, prim_override, prim_texture, listing, count, exists,
           mat_id, mat_color, mat_texture, mat_override, mat_twosided, users, refcount):
    out = []
    labels_by_id = {0: "__default__"}
    for u in uuids:                                        # per primitive: material ID, label, override, texture
        i, lbl = prim_id(u), prim_label(u)
        labels_by_id.setdefault(i, lbl)
        out += [float(u), float(i), float(prim_override(u))] + _codes(lbl) + _codes(_base(prim_texture(u)))
    lst = listing()
    out += [-1.0, float(count()), float(len(lst))]
    for lbl in lst:
        out += _codes(lbl)
        labels_by_id.setdefault(mat_id(lbl), lbl)
    out.append(-2.0)
    for lbl in extra_labels:                               # labels seen earlier in the run (automatic ones too)
        e = exists(lbl)
        out.append(float(e))
        if e:
            labels_by_id.setdefault(mat_id(lbl), lbl)
    out.append(-3.0)
    for i in sorted(labels_by_id):                         # per reachable material
        lbl = labels_by_id[i]
        c = [float(x) for x in mat_color(lbl)]
        us = users(lbl)
        out += [float(i), float(mat_id(lbl))] + _codes(lbl) + c + _codes(_base(mat_texture(lbl)))
        out += [float(mat_override(lbl)), float(mat_twosided(lbl)), float(refcount(i, c)), float(len(us))]
        out += [float(u) for u in us]                      # in Helios's (hash-map) order
    return out


@op("material_state", "helios")
def _(B, extra_labels=()):
    c = B.ctx

    def color(lbl):
        k = c.getMaterialColor(lbl)
        return [k.r, k.g, k.b, k.a]

    return _state(B, extra_labels, uuids=sorted(int(u) for u in c.getAllUUIDs()),
                  prim_label=c.getPrimitiveMaterialLabel, prim_id=lambda u: int(c.getPrimitiveMaterialID(u)),
                  prim_override=c.isPrimitiveTextureColorOverridden, prim_texture=c.getPrimitiveTextureFile,
                  listing=c.listMaterials, count=c.getMaterialCount, exists=c.doesMaterialExist,
                  mat_id=lambda lbl: int(c.getMaterialIDFromLabel(lbl)), mat_color=color,
                  mat_texture=c.getMaterialTexture, mat_override=c.isMaterialTextureColorOverridden,
                  mat_twosided=c.getMaterialTwosidedFlag, users=lambda lbl: [int(u) for u in c.getPrimitivesUsingMaterial(lbl)],
                  refcount=lambda i, col: _helios_reference_count(B, i, np.asarray(col, np.float32).tolist()))


@op("material_state", "heliades")
def _(B, extra_labels=()):
    S = B.scene
    return _state(B, extra_labels, uuids=S.all_uuids(),
                  prim_label=S.get_primitive_material_label, prim_id=S.get_primitive_material_id,
                  prim_override=S.is_primitive_texture_color_overridden, prim_texture=S.primitive_texture_file,
                  listing=S.list_materials, count=S.get_material_count, exists=S.does_material_exist,
                  mat_id=S.get_material_id_from_label, mat_color=S.get_material_color,
                  mat_texture=S.get_material_texture, mat_override=S.is_material_texture_color_overridden,
                  mat_twosided=S.get_material_twosided_flag, users=S.get_primitives_using_material,
                  refcount=lambda i, col: S.get_material(i).reference_count)
