# SPDX-License-Identifier: LGPL-2.1-or-later
"""loadOBJ and polymesh-topology operations for both backends (W2-A).

The Helios side calls the PyHelios ctypes wrappers directly: ``Context.loadOBJ`` picks the overload
without upaxis whenever origin/height/rotation/color are all given, so its "YUP" never reaches Helios.
"""
from __future__ import annotations

import numpy as np

from ...ops import op


def _f(xs):
    return [float(x) for x in xs]


def _codes(s: str) -> list[float]:
    """A string as a float stream: length, then code points (exact in float32 below 2**24)."""
    return [float(len(s))] + [float(ord(c)) for c in s]


# ---- Helios (PyHelios) ---------------------------------------------------------------------

@op("load_obj", "helios")
def _(B, filename, origin=(0.0, 0.0, 0.0), height=None, rotation=(0.0, 0.0), color=None, upaxis=None, scale=None,
      overload=None):
    """overload: "simple" (file, silent), "height" (no upaxis: ZUP), "height_up", or "scale" (vec3, upaxis)."""
    from pyhelios.wrappers import UContextWrapper as W
    c = B.ctx.context
    rot = [1.0, float(rotation[0]), float(rotation[1])]           # SphericalCoord(radius, elevation, azimuth)
    col = _f(color) if color is not None else [0.0, 0.0, 1.0]
    if overload == "simple":
        return W.loadOBJ(c, filename, True)
    if overload == "height":
        return W.loadOBJWithOriginHeightRotationColor(c, filename, _f(origin), float(height), rot, col, True)
    if overload == "height_up":
        return W.loadOBJWithOriginHeightRotationColorUpaxis(c, filename, _f(origin), float(height), rot, col, upaxis, True)
    if overload == "scale":
        return W.loadOBJWithOriginScaleRotationColorUpaxis(c, filename, _f(origin), _f(scale), rot, col, upaxis, True)
    raise ValueError(overload)


@op("obj_queries", "helios")
def _(B):
    """Per primitive (sorted): texture-color override, material ID and label (automatic ones included),
    object_label; then per polymesh
    object: normal source, counts, vertices, faces, normals, UVs and the face -> UUID map."""
    ctx = B.ctx
    out = []
    for u in sorted(ctx.getAllUUIDs()):
        u = int(u)
        out += [float(ctx.isPrimitiveTextureColorOverridden(u)), float(ctx.getPrimitiveMaterialID(u))]
        out += _codes(ctx.getPrimitiveMaterialLabel(u))
        out += _codes(ctx.getPrimitiveData(u, "object_label")) if ctx.doesPrimitiveDataExist(u, "object_label") else [-1.0]
    for o in sorted(ctx.getAllObjectIDs()):
        if ctx.getObjectType(o) != 5:
            continue
        nf = ctx.getPolymeshObjectFaceCount(o)
        out += [float(int(ctx.getPolymeshObjectVertexNormalSource(o))), float(ctx.doesPolymeshObjectHaveVertexNormals(o)),
                float(ctx.getPolymeshObjectVertexCount(o)), float(nf)]
        out += [x for v in ctx.getPolymeshObjectVertices(o) for x in (v.x, v.y, v.z)]
        out += [float(x) for f in ctx.getPolymeshObjectFaces(o) for x in (f.x, f.y, f.z)]
        out += [x for v in ctx.getPolymeshObjectVertexNormals(o) for x in (v.x, v.y, v.z)]
        out += [x for v in ctx.getPolymeshObjectVertexUV(o) for x in (v.x, v.y)]
        out += [float(ctx.getPolymeshObjectPrimitiveUUIDForFace(o, f)) for f in range(nf)]
    return out


@op("compute_polymesh_normals", "helios")
def _(B, oid, crease):
    B.ctx.computePolymeshObjectVertexNormals(int(oid), float(crease))


@op("set_polymesh_topology", "helios")
def _(B, oid, vertices, faces, face_uuids, normals=None, uv=None, source=0):
    from pyhelios.wrappers import UContextWrapper as W
    W.setPolymeshObjectTopologyWrapper(B.ctx.context, int(oid), [tuple(_f(v)) for v in vertices],
                                       [tuple(int(x) for x in f) for f in faces], [int(u) for u in face_uuids],
                                       [tuple(_f(v)) for v in normals] if normals is not None else [],
                                       [tuple(_f(v)) for v in uv] if uv is not None else [], int(source))


# ---- Heliades ------------------------------------------------------------------------------

@op("load_obj", "heliades")
def _(B, filename, origin=(0.0, 0.0, 0.0), height=None, rotation=(0.0, 0.0), color=None, upaxis=None, scale=None,
      overload=None):
    S = B.scene
    if overload == "simple":
        return S.load_obj(filename)
    if overload == "height":
        return S.load_obj(filename, origin, height, rotation, color, "ZUP")
    if overload == "height_up":
        return S.load_obj(filename, origin, height, rotation, color, upaxis)
    if overload == "scale":
        return S.load_obj(filename, origin, None, rotation, color, upaxis, scale=scale)
    raise ValueError(overload)


@op("obj_queries", "heliades")
def _(B):
    S = B.scene
    out = []
    for u in S.all_uuids():
        out += [float(S.is_primitive_texture_color_overridden(u)), float(S.get_primitive_material_id(u))]
        out += _codes(S.get_primitive_material_label(u))
        out += _codes(S.get_primitive_data(u, "object_label")) if S.does_primitive_data_exist(u, "object_label") else [-1.0]
    for o in S.all_object_ids():
        if S.object_type(o) != 5:
            continue
        nf = S.polymesh_object_face_count(o)
        out += [float(S.polymesh_object_vertex_normal_source(o)), float(S.does_polymesh_object_have_vertex_normals(o)),
                float(S.polymesh_object_vertex_count(o)), float(nf)]
        out += list(np.asarray(S.polymesh_object_vertices(o), np.float32).reshape(-1))
        out += [float(x) for x in np.asarray(S.polymesh_object_faces(o)).reshape(-1)]
        out += list(np.asarray(S.polymesh_object_vertex_normals(o), np.float32).reshape(-1))
        out += list(np.asarray(S.polymesh_object_vertex_uv(o), np.float32).reshape(-1))
        out += [float(S.polymesh_object_primitive_uuid_for_face(o, f)) for f in range(nf)]
    return out


@op("compute_polymesh_normals", "heliades")
def _(B, oid, crease):
    B.scene.compute_polymesh_object_vertex_normals(int(oid), crease)


@op("set_polymesh_topology", "heliades")
def _(B, oid, vertices, faces, face_uuids, normals=None, uv=None, source=0):
    B.scene.set_polymesh_object_topology(int(oid), vertices, faces, face_uuids,
                                         () if normals is None else normals, () if uv is None else uv, source)
