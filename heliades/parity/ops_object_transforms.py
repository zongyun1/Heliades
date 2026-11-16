# SPDX-License-Identifier: LGPL-2.1-or-later
"""Backend operations for W1-B: copy / delete / object transforms / arbitrary-axis primitive rotation.

Every operation takes a single ID (int) or a list of IDs, and routes a list to the Helios vector
overload. Getter ops return flat float lists so cases can compare them as streams.
"""
from __future__ import annotations

import numpy as np

from .ops import op


def _ids(x):
    return int(x) if np.isscalar(x) else [int(i) for i in x]


def _v(B, p):
    return B.D.vec3(*map(float, p))


# ---- copy -------------------------------------------------------------------------------------

@op("copy_primitive", "helios")
def _(B, uuids):
    return B.ctx.copyPrimitive(_ids(uuids))


@op("copy_primitive", "heliades")
def _(B, uuids):
    return B.scene.copy_primitive(_ids(uuids))


@op("copy_object", "helios")
def _(B, oids):
    return B.ctx.copyObject(_ids(oids))


@op("copy_object", "heliades")
def _(B, oids):
    return B.scene.copy_object(_ids(oids))


# ---- delete -----------------------------------------------------------------------------------

@op("delete_primitive", "helios")
def _(B, uuids):
    B.ctx.deletePrimitive(_ids(uuids))


@op("delete_primitive", "heliades")
def _(B, uuids):
    B.scene.delete_primitives(_ids(uuids))


@op("delete_object", "helios")
def _(B, oids):
    B.ctx.deleteObject(_ids(oids))


@op("delete_object", "heliades")
def _(B, oids):
    B.scene.delete_object(_ids(oids))


# ---- object transforms ------------------------------------------------------------------------

@op("translate_object", "helios")
def _(B, oids, shift):
    B.ctx.translateObject(_ids(oids), _v(B, shift))


@op("translate_object", "heliades")
def _(B, oids, shift):
    B.scene.translate_object(_ids(oids), shift)


@op("rotate_object", "helios")
def _(B, oids, radians, axis, origin=None):
    ax = axis if isinstance(axis, str) else _v(B, axis)
    if origin is None:
        B.ctx.rotateObject(_ids(oids), float(radians), ax)
    else:
        B.ctx.rotateObject(_ids(oids), float(radians), ax, origin=_v(B, origin))


@op("rotate_object", "heliades")
def _(B, oids, radians, axis, origin=None):
    B.scene.rotate_object(_ids(oids), radians, axis, origin)


@op("rotate_object_about_origin", "helios")
def _(B, oids, radians, axis):
    B.ctx.rotateObject(_ids(oids), float(radians), _v(B, axis), about_origin=True)


@op("rotate_object_about_origin", "heliades")
def _(B, oids, radians, axis):
    B.scene.rotate_object_about_origin(_ids(oids), radians, axis)


@op("scale_object", "helios")
def _(B, oids, s, mode="world", point=None):
    """mode: 'world' (scaleObject), 'center', 'point', 'origin' (the object's stored origin)."""
    ids, sv = _ids(oids), _v(B, s)
    if mode == "world":
        B.ctx.scaleObject(ids, sv)
    elif mode == "center":
        B.ctx.scaleObject(ids, sv, about_center=True)
    elif mode == "point":
        B.ctx.scaleObject(ids, sv, point=_v(B, point))
    elif mode == "origin":
        B.ctx.scaleObject(ids, sv, about_origin=True)
    else:
        raise ValueError(mode)


@op("scale_object", "heliades")
def _(B, oids, s, mode="world", point=None):
    S, ids = B.scene, _ids(oids)
    if mode == "world":
        S.scale_object(ids, s)
    elif mode == "center":
        S.scale_object_about_center(ids, s)
    elif mode == "point":
        S.scale_object_about_point(ids, s, point)
    elif mode == "origin":
        S.scale_object_about_origin(ids, s)
    else:
        raise ValueError(mode)


# ---- primitive transforms about an arbitrary axis ---------------------------------------------

@op("rotate_primitive_axis", "helios")
def _(B, uuids, radians, axis, origin=None):
    if origin is None:
        B.ctx.rotatePrimitive(_ids(uuids), float(radians), _v(B, axis))
    else:
        B.ctx.rotatePrimitive(_ids(uuids), float(radians), _v(B, axis), _v(B, origin))


@op("rotate_primitive_axis", "heliades")
def _(B, uuids, radians, axis, origin=None):
    B.scene.rotate_primitive_about_axis(_ids(uuids), radians, axis, origin)


@op("scale_primitive_about_point", "helios")
def _(B, uuids, s, point):
    B.ctx.scalePrimitive(_ids(uuids), _v(B, s), point=_v(B, point))


@op("scale_primitive_about_point", "heliades")
def _(B, uuids, s, point):
    B.scene.scale_primitive_about_point(_ids(uuids), s, point)


# ---- queries ----------------------------------------------------------------------------------

@op("object_ids", "helios")
def _(B):
    return sorted(B.ctx.getAllObjectIDs())


@op("object_ids", "heliades")
def _(B):
    return B.scene.all_object_ids()


@op("all_uuids", "helios")
def _(B):
    return sorted(B.ctx.getAllUUIDs())


@op("all_uuids", "heliades")
def _(B):
    return B.scene.all_uuids()


@op("object_uuids", "helios")
def _(B, oid):
    return list(B.ctx.getObjectPrimitiveUUIDs(int(oid)))


@op("object_uuids", "heliades")
def _(B, oid):
    return B.scene.object_primitive_uuids(int(oid))


@op("object_type", "helios")
def _(B, oid):
    return int(B.ctx.getObjectType(int(oid)))


@op("object_type", "heliades")
def _(B, oid):
    return int(B.scene.object_type(int(oid)))


@op("primitive_transform", "helios")
def _(B, uuid):
    return np.asarray(B.ctx.getPrimitiveTransformationMatrix(int(uuid)), np.float32).reshape(16).tolist()


@op("primitive_transform", "heliades")
def _(B, uuid):
    return B.scene.primitives[int(uuid)].transform.tolist()


@op("tube_nodes", "helios")
def _(B, oid):
    return [c for v in B.ctx.getTubeObjectNodes(int(oid)) for c in (v.x, v.y, v.z)]


@op("tube_nodes", "heliades")
def _(B, oid):
    return B.scene.tube_object_nodes(int(oid)).reshape(-1).tolist()


@op("tube_radii", "helios")
def _(B, oid):
    return list(B.ctx.getTubeObjectNodeRadii(int(oid)))


@op("tube_radii", "heliades")
def _(B, oid):
    return B.scene.tube_object_node_radii(int(oid)).tolist()


@op("tile_vertices", "helios")
def _(B, oid):
    return [c for v in B.ctx.getTileObjectVertices(int(oid)) for c in (v.x, v.y, v.z)]


@op("tile_vertices", "heliades")
def _(B, oid):
    return B.scene.tile_object_vertices(int(oid)).reshape(-1).tolist()


@op("tile_center", "helios")
def _(B, oid):
    v = B.ctx.getTileObjectCenter(int(oid))
    return [v.x, v.y, v.z]


@op("tile_center", "heliades")
def _(B, oid):
    return B.scene.tile_object_center(int(oid)).tolist()


@op("set_pdata_float", "helios")
def _(B, uuid, label, value):
    B.ctx.setPrimitiveDataFloat(int(uuid), label, float(value))


@op("set_pdata_float", "heliades")
def _(B, uuid, label, value):
    B.scene.set_primitive_data(int(uuid), label, np.float32(value))


@op("get_pdata_float", "helios")
def _(B, uuid, label):
    return float(B.ctx.getPrimitiveDataFloat(int(uuid), label)) if B.ctx.doesPrimitiveDataExist(int(uuid), label) else None


@op("get_pdata_float", "heliades")
def _(B, uuid, label):
    S = B.scene
    return float(S.get_primitive_data(int(uuid), label)) if S.does_primitive_data_exist(int(uuid), label) else None
