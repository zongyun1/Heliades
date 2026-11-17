# SPDX-License-Identifier: LGPL-2.1-or-later
"""Textured primitive and object operations for both backends (W1-A).

Texture strings are passed through unchanged on both sides. PyHelios itself makes patch/triangle
paths absolute and leaves tile/tube paths alone, which is why snapshots compare texture basenames
(``compare.texture_key``).
"""
from __future__ import annotations

from ...ops import op


def _f(xs):
    return [float(x) for x in xs]


# ---- Helios (PyHelios) ---------------------------------------------------------------------

@op("add_patch_textured", "helios")
def _(B, center, size, rotation, texture_file, uv_center=None, uv_size=None):
    D = B.D
    kw = {}
    if uv_center is not None:
        kw = dict(uv_center=D.vec2(*_f(uv_center)), uv_size=D.vec2(*_f(uv_size)))
    return B.ctx.addPatchTextured(D.vec3(*_f(center)), D.vec2(*_f(size)), texture_file,
                                  rotation=D.SphericalCoord(1.0, float(rotation[0]), float(rotation[1])), **kw)


@op("add_triangle_textured", "helios")
def _(B, v0, v1, v2, texture_file, uv0, uv1, uv2):
    D = B.D
    return B.ctx.addTriangleTextured(D.vec3(*_f(v0)), D.vec3(*_f(v1)), D.vec3(*_f(v2)), texture_file,
                                     D.vec2(*_f(uv0)), D.vec2(*_f(uv1)), D.vec2(*_f(uv2)))


@op("add_tile_textured", "helios")
def _(B, center, size, rotation, subdiv, texture_file, texture_repeat=None):
    D = B.D
    rep = None if texture_repeat is None else D.int2(int(texture_repeat[0]), int(texture_repeat[1]))
    return B.ctx.addTileObject(center=D.vec3(*_f(center)), size=D.vec2(*_f(size)),
                               rotation=D.SphericalCoord(1.0, float(rotation[0]), float(rotation[1])),
                               subdiv=D.int2(int(subdiv[0]), int(subdiv[1])), texturefile=texture_file,
                               texture_repeat=rep)


@op("add_tube_textured", "helios")
def _(B, ndiv, nodes, radius, texture_file, textureuv_ufrac=None):
    D = B.D
    uf = None if textureuv_ufrac is None else _f(textureuv_ufrac)
    return B.ctx.addTubeObject(int(ndiv), [D.vec3(*_f(p)) for p in nodes], _f(radius), texturefile=texture_file,
                               texture_uv=uf)


@op("texture_queries", "helios")
def _(B):
    """Per primitive (sorted UUIDs): hasTransparencyChannel, texture width, height."""
    out = []
    for u in sorted(B.ctx.getAllUUIDs()):
        s = B.ctx.getPrimitiveTextureSize(int(u))
        out += [float(B.ctx.primitiveTextureHasTransparencyChannel(int(u))), float(s.x), float(s.y)]
    return out


# ---- Heliades ------------------------------------------------------------------------------

@op("add_patch_textured", "heliades")
def _(B, center, size, rotation, texture_file, uv_center=None, uv_size=None):
    return B.scene.add_patch_textured(center, size, rotation, texture_file, uv_center, uv_size)


@op("add_triangle_textured", "heliades")
def _(B, v0, v1, v2, texture_file, uv0, uv1, uv2):
    return B.scene.add_triangle_textured(v0, v1, v2, texture_file, uv0, uv1, uv2)


@op("add_tile_textured", "heliades")
def _(B, center, size, rotation, subdiv, texture_file, texture_repeat=None):
    return B.scene.add_tile_object_textured(center, size, rotation, subdiv, texture_file,
                                            (1, 1) if texture_repeat is None else texture_repeat)


@op("add_tube_textured", "heliades")
def _(B, ndiv, nodes, radius, texture_file, textureuv_ufrac=None):
    return B.scene.add_tube_object_textured(ndiv, nodes, radius, texture_file, textureuv_ufrac)


@op("texture_queries", "heliades")
def _(B):
    out = []
    for u in B.scene.all_uuids():
        w, h = B.scene.primitive_texture_size(u)
        out += [float(B.scene.primitive_texture_has_transparency_channel(u)), float(w), float(h)]
    return out
