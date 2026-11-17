# SPDX-License-Identifier: LGPL-2.1-or-later
"""Backend ops for sphere/disk/cone/box objects (W1-C).

The Helios side calls PyHelios's ctypes wrappers (``pyhelios.wrappers.UContextWrapper``) directly so
each C++ overload is chosen explicitly. ``color=None`` selects the colorless C++ overload (its default
color), never a PyHelios substitute. The direct calls also bypass PyHelios's Python-side validation
(e.g. sphere ``ndivs >= 3``) so Helios's own edge behaviour is reachable.
"""
from __future__ import annotations

import numpy as np

from ...ops import op

_GREEN = (0.0, 0.75, 0.0)


def _W():
    from pyhelios.wrappers import UContextWrapper as W
    return W


def _f(v):
    return [float(x) for x in np.asarray(v, float).ravel()]


# ---- sphere ----------------------------------------------------------------------------------

@op("add_sphere", "helios")
def _(B, ndivs, center, radius, color=None):
    W, ctx = _W(), B.ctx.context
    if np.ndim(radius) == 0:
        if color is None:
            return W.addSphereObject_basic(ctx, int(ndivs), _f(center), float(radius))
        return W.addSphereObject_color(ctx, int(ndivs), _f(center), float(radius), _f(color)[:3])
    if color is None:
        return W.addSphereObject_ellipsoid(ctx, int(ndivs), _f(center), _f(radius))
    return W.addSphereObject_ellipsoid_color(ctx, int(ndivs), _f(center), _f(radius), _f(color)[:3])


@op("add_sphere", "heliades")
def _(B, ndivs, center, radius, color=None):
    return B.scene.add_sphere_object(ndivs, center, radius, _GREEN if color is None else tuple(color)[:3])


# ---- disk ------------------------------------------------------------------------------------

@op("add_disk", "helios")
def _(B, ndivs, center, size, rotation=None, color=None):
    """ndivs int -> uint overloads; (nx, ny) -> int2 overloads (which have no colorless form)."""
    W, ctx = _W(), B.ctx.context
    rot = None if rotation is None else [1.0, float(rotation[0]), float(rotation[1])]
    if np.ndim(ndivs) == 0:
        n = int(ndivs)
        if color is None:
            if rot is None:
                return W.addDiskObject_basic(ctx, n, _f(center), _f(size))
            return W.addDiskObject_rotation(ctx, n, _f(center), _f(size), rot)
        rot = rot or [1.0, 0.0, 0.0]
        if len(color) == 4:
            return W.addDiskObject_rgba(ctx, n, _f(center), _f(size), rot, _f(color))
        return W.addDiskObject_color(ctx, n, _f(center), _f(size), rot, _f(color))
    if color is None:
        raise ValueError("int2 addDiskObject has no colorless overload")
    nd, rot = [int(ndivs[0]), int(ndivs[1])], rot or [1.0, 0.0, 0.0]
    if len(color) == 4:
        return W.addDiskObject_polar_rgba(ctx, nd, _f(center), _f(size), rot, _f(color))
    return W.addDiskObject_polar_color(ctx, nd, _f(center), _f(size), rot, _f(color))


@op("add_disk", "heliades")
def _(B, ndivs, center, size, rotation=None, color=None):
    kw = {} if color is None else {"color": tuple(color)}
    return B.scene.add_disk_object(ndivs, center, size, (0.0, 0.0) if rotation is None else rotation, **kw)


# ---- cone ------------------------------------------------------------------------------------

@op("add_cone", "helios")
def _(B, ndivs, node0, node1, radius0, radius1, color=None):
    W, ctx = _W(), B.ctx.context
    if color is None:
        return W.addConeObject_basic(ctx, int(ndivs), _f(node0), _f(node1), float(radius0), float(radius1))
    return W.addConeObject_color(ctx, int(ndivs), _f(node0), _f(node1), float(radius0), float(radius1), _f(color)[:3])


@op("add_cone", "heliades")
def _(B, ndivs, node0, node1, radius0, radius1, color=None):
    return B.scene.add_cone_object(ndivs, node0, node1, radius0, radius1, _GREEN if color is None else tuple(color)[:3])


# ---- box -------------------------------------------------------------------------------------

@op("add_box", "helios")
def _(B, center, size, subdiv, color=None, reverse_normals=False):
    W, ctx = _W(), B.ctx.context
    sd = [int(s) for s in subdiv]
    if reverse_normals:  # only the color+bool overload exists; pass the default green explicitly
        return W.addBoxObject_color_reverse(ctx, _f(center), _f(size), sd, _f(_GREEN if color is None else color)[:3], True)
    if color is None:
        return W.addBoxObject_basic(ctx, _f(center), _f(size), sd)
    return W.addBoxObject_color(ctx, _f(center), _f(size), sd, _f(color)[:3])


@op("add_box", "heliades")
def _(B, center, size, subdiv, color=None, reverse_normals=False):
    return B.scene.add_box_object(center, size, subdiv, _GREEN if color is None else tuple(color)[:3], reverse_normals)


# ---- type-specific getters -------------------------------------------------------------------
# shape_getters(oid) -> flat list of every getter PyHelios exposes for the object's type, in a fixed order.

@op("shape_getters", "helios")
def _(B, oid):
    W, ctx, oid = _W(), B.ctx.context, int(oid)
    t = B.ctx.getObjectType(oid)
    if t == 1:
        return [*W.getSphereObjectCenterWrapper(ctx, oid), *W.getSphereObjectRadiusWrapper(ctx, oid),
                W.getSphereObjectSubdivisionCountWrapper(ctx, oid), W.getSphereObjectVolumeWrapper(ctx, oid)]
    if t == 3:
        return [*W.getBoxObjectCenterWrapper(ctx, oid), *W.getBoxObjectSizeWrapper(ctx, oid),
                *W.getBoxObjectSubdivisionCountWrapper(ctx, oid), W.getBoxObjectVolumeWrapper(ctx, oid)]
    if t == 4:
        return [*W.getDiskObjectCenterWrapper(ctx, oid), *W.getDiskObjectSizeWrapper(ctx, oid),
                W.getDiskObjectSubdivisionCountWrapper(ctx, oid)]
    if t == 6:
        return [W.getConeObjectSubdivisionCountWrapper(ctx, oid),
                *[x for n in W.getConeObjectNodesWrapper(ctx, oid) for x in n], *W.getConeObjectNodeRadiiWrapper(ctx, oid),
                *W.getConeObjectNodeWrapper(ctx, oid, 0), *W.getConeObjectNodeWrapper(ctx, oid, 1),
                W.getConeObjectNodeRadiusWrapper(ctx, oid, 0), W.getConeObjectNodeRadiusWrapper(ctx, oid, 1),
                *W.getConeObjectAxisUnitVectorWrapper(ctx, oid), W.getConeObjectLengthWrapper(ctx, oid),
                W.getConeObjectVolumeWrapper(ctx, oid)]
    raise ValueError(f"object {oid} has type {t}, not a W1-C shape")


@op("shape_getters", "heliades")
def _(B, oid):
    S = B.scene
    t = S.object_type(oid)
    if t == 1:
        return [*S.sphere_object_center(oid), *S.sphere_object_radius(oid),
                S.sphere_object_subdivision_count(oid), S.sphere_object_volume(oid)]
    if t == 3:
        return [*S.box_object_center(oid), *S.box_object_size(oid),
                *S.box_object_subdivision_count(oid), S.box_object_volume(oid)]
    if t == 4:
        return [*S.disk_object_center(oid), *S.disk_object_size(oid), S.disk_object_subdivision_count(oid)]
    if t == 6:
        return [S.cone_object_subdivision_count(oid), *S.cone_object_nodes(oid).ravel(), *S.cone_object_node_radii(oid),
                *S.cone_object_node(oid, 0), *S.cone_object_node(oid, 1),
                S.cone_object_node_radius(oid, 0), S.cone_object_node_radius(oid, 1),
                *S.cone_object_axis_unit_vector(oid), S.cone_object_length(oid), S.cone_object_volume(oid)]
    raise ValueError(f"object {oid} has type {t}, not a W1-C shape")



# ---- snapshot with alpha ---------------------------------------------------------------------
# helios_snapshot reads getAllPrimitiveColors, which is RGB only, and pads alpha with 1. The RGBA
# disk overloads need alpha compared, so refill it from getPrimitiveColorRGBA.

@op("snapshot_rgba", "helios")
def _(B):
    W, ctx = _W(), B.ctx.context
    s = B.snapshot()
    for i, u in enumerate(s.uuid):
        s.color[i] = W.getPrimitiveColorRGBA(ctx, int(u))[:4]
    return s


@op("snapshot_rgba", "heliades")
def _(B):
    return B.snapshot()
