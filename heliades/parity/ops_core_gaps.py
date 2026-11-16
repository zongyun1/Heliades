# SPDX-License-Identifier: LGPL-2.1-or-later
"""Backend operations for W2-D: tube editing, hidden primitives/objects, typed primitive/object data,
object area and the material table.

Getters return plain Python values; typed data is returned through ``encode_data`` so that a stream
comparison is exact for every type (doubles and 32-bit ints are split into 16-bit chunks, strings into
their bytes).
"""
from __future__ import annotations

import struct

import numpy as np

from .ops import op


def _ids(x):
    return int(x) if np.isscalar(x) else [int(i) for i in x]


def _v(B, p):
    return B.D.vec3(*map(float, p))


# ---- tube editing ---------------------------------------------------------------------------

@op("append_tube_segment", "helios")
def _(B, oid, node, radius, color):
    B.ctx.appendTubeSegment(int(oid), _v(B, node), float(radius), color=B.D.RGBcolor(*map(float, color[:3])))


@op("append_tube_segment", "heliades")
def _(B, oid, node, radius, color):
    B.scene.append_tube_segment(int(oid), node, np.float32(radius), tuple(np.float32(c) for c in color[:3]))


@op("append_tube_segment_textured", "helios")
def _(B, oid, node, radius, texture, ufrac):
    B.ctx.appendTubeSegment(int(oid), _v(B, node), float(radius), texture_file=texture,
                            uv=B.D.vec2(float(ufrac[0]), float(ufrac[1])))


@op("append_tube_segment_textured", "heliades")
def _(B, oid, node, radius, texture, ufrac):
    B.scene.append_tube_segment_textured(int(oid), node, np.float32(radius), texture, ufrac)


@op("set_tube_nodes", "helios")
def _(B, oid, nodes):
    B.ctx.setTubeNodes(int(oid), [_v(B, p) for p in nodes])


@op("set_tube_nodes", "heliades")
def _(B, oid, nodes):
    B.scene.set_tube_nodes(int(oid), nodes)


@op("set_tube_radii", "helios")
def _(B, oid, radii):
    B.ctx.setTubeRadii(int(oid), [float(r) for r in radii])


@op("set_tube_radii", "heliades")
def _(B, oid, radii):
    B.scene.set_tube_radii(int(oid), [np.float32(r) for r in radii])


@op("scale_tube_girth", "helios")
def _(B, oid, s):
    B.ctx.scaleTubeGirth(int(oid), float(s))


@op("scale_tube_girth", "heliades")
def _(B, oid, s):
    B.scene.scale_tube_girth(int(oid), s)


@op("scale_tube_length", "helios")
def _(B, oid, s):
    B.ctx.scaleTubeLength(int(oid), float(s))


@op("scale_tube_length", "heliades")
def _(B, oid, s):
    B.scene.scale_tube_length(int(oid), s)


@op("prune_tube_nodes", "helios")
def _(B, oid, k):
    B.ctx.pruneTubeNodes(int(oid), int(k))


@op("prune_tube_nodes", "heliades")
def _(B, oid, k):
    B.scene.prune_tube_nodes(int(oid), int(k))


@op("tube_info", "helios")
def _(B, oid):
    """Node count, subdivisions, colors, volume and every segment volume (float32 values)."""
    c, o = B.ctx, int(oid)
    n = c.getTubeObjectNodeCount(o)
    out = [n, c.getTubeObjectSubdivisionCount(o)]
    out += [x for col in c.getTubeObjectNodeColors(o) for x in (col.r, col.g, col.b)]
    out += [c.getTubeObjectVolume(o)] + [c.getTubeObjectSegmentVolume(o, k) for k in range(n - 1)]
    return out


@op("tube_info", "heliades")
def _(B, oid):
    S, o = B.scene, int(oid)
    n = S.tube_object_node_count(o)
    out = [n, S.tube_object_subdivision_count(o)]
    out += S.tube_object_node_colors(o).reshape(-1).tolist()
    out += [S.tube_object_volume(o)] + [S.tube_object_segment_volume(o, k) for k in range(n - 1)]
    return [float(x) for x in out]


# ---- hidden ---------------------------------------------------------------------------------

@op("hide_object", "helios")
def _(B, oids):
    B.ctx.hideObject(_ids(oids))


@op("hide_object", "heliades")
def _(B, oids):
    B.scene.hide_object(_ids(oids))


@op("show_object", "helios")
def _(B, oids):
    B.ctx.showObject(_ids(oids))


@op("show_object", "heliades")
def _(B, oids):
    B.scene.show_object(_ids(oids))


@op("hide_primitive", "helios")
def _(B, uuids):
    B.ctx.hidePrimitive(_ids(uuids))


@op("hide_primitive", "heliades")
def _(B, uuids):
    B.scene.hide_primitive(_ids(uuids))


@op("show_primitive", "helios")
def _(B, uuids):
    B.ctx.showPrimitive(_ids(uuids))


@op("show_primitive", "heliades")
def _(B, uuids):
    B.scene.show_primitive(_ids(uuids))


@op("is_primitive_hidden", "helios")
def _(B, u):
    return bool(B.ctx.isPrimitiveHidden(int(u)))


@op("is_primitive_hidden", "heliades")
def _(B, u):
    return B.scene.is_primitive_hidden(int(u))


@op("is_object_hidden", "helios")
def _(B, o):
    return bool(B.ctx.isObjectHidden(int(o)))


@op("is_object_hidden", "heliades")
def _(B, o):
    return B.scene.is_object_hidden(int(o))


@op("counts", "helios")
def _(B):
    """(primitive count incl. hidden, object count incl. hidden)."""
    return [B.ctx.getPrimitiveCount(), B.ctx.getObjectCount()]


@op("counts", "heliades")
def _(B):
    return [B.scene.primitive_count(), B.scene.object_count()]


@op("object_area", "helios")
def _(B, o):
    return float(B.ctx.getObjectArea(int(o)))


@op("object_area", "heliades")
def _(B, o):
    return float(B.scene.object_area(int(o)))


@op("primitive_area", "helios")
def _(B, u):
    return float(B.ctx.getPrimitiveArea(int(u)))


@op("primitive_area", "heliades")
def _(B, u):
    return float(B.scene.primitive_area(int(u)))


# ---- typed data -----------------------------------------------------------------------------

_SETTER = {"int": "Int", "uint": "UInt", "float": "Float", "double": "Double", "string": "String", "vec2": "Vec2",
           "vec3": "Vec3", "vec4": "Vec4", "int2": "Int2", "int3": "Int3", "int4": "Int4"}


def _helios_value(B, value, dtype):
    D = B.D
    if dtype in ("vec2", "vec3", "vec4"):
        return getattr(D, dtype)(*map(float, value))
    if dtype in ("int2", "int3", "int4"):
        return getattr(D, dtype)(*map(int, value))
    if dtype in ("float", "double"):
        return float(value)
    if dtype == "string":
        return str(value)
    return int(value)


def _helios_getter_type(B, dtype):
    D = B.D
    return {"int": int, "uint": "uint", "float": float, "double": "double", "string": str, "vec2": D.vec2,
            "vec3": D.vec3, "vec4": D.vec4, "int2": D.int2, "int3": D.int3, "int4": D.int4}[dtype]


def _comps(v):
    """A PyHelios vector type or a numpy array as a list."""
    if hasattr(v, "to_list"):
        return list(v.to_list())
    for names in (("x", "y", "z", "w"),):
        if hasattr(v, "x"):
            return [getattr(v, n) for n in names if hasattr(v, n)]
    return list(np.asarray(v).reshape(-1))


def encode_data(value, dtype) -> list:
    """Exact float32-representable encoding of a typed value."""
    if dtype == "float":
        return [float(np.float32(value))]
    if dtype == "double":
        b = struct.pack("<d", float(value))
        return [float(int.from_bytes(b[i:i + 2], "little")) for i in range(0, 8, 2)]
    if dtype in ("int", "uint"):
        x = int(value) & 0xFFFFFFFF
        return [float(x & 0xFFFF), float(x >> 16)]
    if dtype == "string":
        return [float(len(value))] + [float(c) for c in str(value).encode("utf-8")]
    if dtype in ("vec2", "vec3", "vec4"):
        return [float(np.float32(c)) for c in _comps(value)]
    out = []
    for c in _comps(value):
        out += encode_data(int(c), "int")
    return out


@op("set_data", "helios")
def _(B, kind, ids, label, value, dtype):
    """kind: 'primitive' or 'object'."""
    fn = getattr(B.ctx, f"set{'Primitive' if kind == 'primitive' else 'Object'}Data{_SETTER[dtype]}")
    fn(_ids(ids), label, _helios_value(B, value, dtype))


@op("set_data", "heliades")
def _(B, kind, ids, label, value, dtype):
    fn = B.scene.set_primitive_data if kind == "primitive" else B.scene.set_object_data
    fn(_ids(ids), label, value, dtype)


@op("get_data", "helios")
def _(B, kind, i, label, dtype):
    c = B.ctx
    if kind == "primitive":
        if not c.doesPrimitiveDataExist(int(i), label):
            return None
        return encode_data(c.getPrimitiveData(int(i), label, _helios_getter_type(B, dtype)), dtype)
    if not c.doesObjectDataExist(int(i), label):
        return None
    return encode_data(c.getObjectData(int(i), label, _helios_getter_type(B, dtype)), dtype)


_CTYPE = {"int": np.int32, "uint": np.uint32, "float": np.float32, "double": np.float64, "string": str}


def _stored(v, dtype):
    """The encoded value, or [-99] if Heliades holds it with the wrong C++ type (the encoding alone
    would hide e.g. an uncast float stored under an int label)."""
    t = _CTYPE.get(dtype)
    if t is not None and type(v) is not t:
        return [-99.0]
    if dtype.startswith(("vec", "int")) and dtype not in ("int",):
        if not isinstance(v, np.ndarray) or v.dtype != (np.float32 if dtype.startswith("vec") else np.int32):
            return [-99.0]
    return encode_data(v, dtype)


@op("get_data", "heliades")
def _(B, kind, i, label, dtype):
    S = B.scene
    if kind == "primitive":
        if not S.does_primitive_data_exist(int(i), label):
            return None
        return _stored(S.get_primitive_data(int(i), label), dtype)
    if not S.does_object_data_exist(int(i), label):
        return None
    return _stored(S.get_object_data(int(i), label), dtype)


@op("data_type", "helios")
def _(B, kind, i, label):
    if kind == "primitive":
        return int(B.ctx.getPrimitiveDataType(int(i), label))
    return int(B.ctx.getObjectDataType(int(i), label))


@op("data_type", "heliades")
def _(B, kind, i, label):
    if kind == "primitive":
        return int(B.scene.get_primitive_data_type(int(i), label))
    return int(B.scene.get_object_data_type(int(i), label))


@op("filter_data", "helios")
def _(B, kind, ids, label, value, dtype, comparator="=="):
    """Calls the C wrappers directly: PyHelios's default comparator '=' is rejected by Helios 1.3.98."""
    from ..parity import backends  # noqa: F401
    from pyhelios.wrappers import UContextWrapper as W
    ids = [int(i) for i in ids]
    if not ids:
        return []
    T = {"float": "Float", "double": "Double", "int": "Int", "uint": "UInt", "string": "String"}[dtype]
    if kind == "primitive":
        fn = getattr(W, f"filterPrimitivesByData{T}Wrapper")
    else:
        fn = getattr(W, f"filterObjectsByData{T}Wrapper")
    if dtype == "string":
        return list(fn(B.ctx.context, ids, label, str(value)))
    v = float(value) if dtype in ("float", "double") else int(value)
    return list(fn(B.ctx.context, ids, label, v, comparator))


@op("filter_data", "heliades")
def _(B, kind, ids, label, value, dtype, comparator="=="):
    fn = B.scene.filter_primitives_by_data if kind == "primitive" else B.scene.filter_objects_by_data
    if dtype == "float":
        value = np.float32(value)
    return list(fn([int(i) for i in ids], label, value, comparator, dtype))


# ---- materials ------------------------------------------------------------------------------

@op("add_material", "helios")
def _(B, label):
    B.ctx.addMaterial(label)


@op("add_material", "heliades")
def _(B, label):
    B.scene.add_material(label)


@op("set_material_color", "helios")
def _(B, label, rgba):
    B.ctx.setMaterialColor(label, tuple(float(c) for c in rgba))


@op("set_material_color", "heliades")
def _(B, label, rgba):
    B.scene.set_material_color(label, np.asarray(rgba, np.float32))


@op("set_material_texture", "helios")
def _(B, label, tex):
    B.ctx.setMaterialTexture(label, tex)


@op("set_material_texture", "heliades")
def _(B, label, tex):
    B.scene.set_material_texture(label, tex)


@op("rename_material", "helios")
def _(B, old, new):
    B.ctx.renameMaterial(old, new)


@op("rename_material", "heliades")
def _(B, old, new):
    B.scene.rename_material(old, new)


@op("assign_material_to_primitive", "helios")
def _(B, uuids, label):
    B.ctx.assignMaterialToPrimitive(_ids(uuids), label)


@op("assign_material_to_primitive", "heliades")
def _(B, uuids, label):
    B.scene.assign_material_to_primitive(_ids(uuids), label)


@op("assign_material_to_object", "helios")
def _(B, oids, label):
    B.ctx.assignMaterialToObject(_ids(oids), label)


@op("assign_material_to_object", "heliades")
def _(B, oids, label):
    B.scene.assign_material_to_object(_ids(oids), label)


@op("set_primitive_color", "helios")
def _(B, uuids, rgb):
    B.ctx.setPrimitiveColor(_ids(uuids), B.D.RGBcolor(*map(float, rgb[:3])))


@op("set_primitive_color", "heliades")
def _(B, uuids, rgb):
    B.scene.set_primitive_color(_ids(uuids), np.asarray(rgb[:3], np.float32))


@op("material_labels", "helios")
def _(B, uuids):
    return [B.ctx.getPrimitiveMaterialLabel(int(u)) for u in uuids]


@op("material_labels", "heliades")
def _(B, uuids):
    return [B.scene.get_primitive_material_label(int(u)) for u in uuids]


@op("material_ids", "helios")
def _(B, uuids):
    return [int(B.ctx.getPrimitiveMaterialID(int(u))) for u in uuids]


@op("material_ids", "heliades")
def _(B, uuids):
    return [B.scene.get_primitive_material_id(int(u)) for u in uuids]


@op("material_query", "helios")
def _(B, label):
    """[exists, color rgba, texture basename, users] for one label (exists = 0 and nothing else if not)."""
    c = B.ctx
    if not c.doesMaterialExist(label):
        return [0]
    col = c.getMaterialColor(label)
    tex = c.getMaterialTexture(label)
    return [1, col.r, col.g, col.b, col.a, tex.replace("\\", "/").rsplit("/", 1)[-1],
            [int(u) for u in c.getPrimitivesUsingMaterial(label)]]      # Helios's (hash-map) order


@op("material_query", "heliades")
def _(B, label):
    S = B.scene
    if not S.does_material_exist(label):
        return [0]
    col = S.get_material_color(label)
    tex = S.get_material_texture(label)
    return [1, *map(float, col), tex.replace("\\", "/").rsplit("/", 1)[-1], S.get_primitives_using_material(label)]


@op("list_materials", "helios")
def _(B):
    return list(B.ctx.listMaterials())          # Helios's own order (std::map: bytewise by label)


@op("list_materials", "heliades")
def _(B):
    return B.scene.list_materials()
