# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_object.cpp; Copyright (C) 2016-2026 Brian Bailey).
"""Compound objects: tile, tube, polymesh, and object queries (mixed into Scene)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import vec as V
from ..migration_registry import ports

f32 = np.float32

from .primitives import make_rotation_matrix, make_scale_matrix, make_translation_matrix, matmult

OBJECT_TILE, OBJECT_SPHERE, OBJECT_TUBE, OBJECT_BOX, OBJECT_DISK, OBJECT_POLYMESH, OBJECT_CONE = range(7)


@dataclass
class CompoundObject:
    oid: int
    type: int
    uuids: list[int]
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=f32).reshape(16))
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3, f32))
    extra: dict = field(default_factory=dict)


class ObjectsMixin:
    def _register_object(self, otype: int, uuids: list[int], **kw) -> int:
        oid = self.current_object_id
        for u in uuids:
            self.primitives[u].parent_object = oid
        self.objects[oid] = CompoundObject(oid, otype, list(uuids), **kw)
        self.current_object_id += 1
        return oid

    @ports("Context::addTileObject")
    def add_tile_object(self, center, size, rotation=(0.0, 0.0), subdiv=(1, 1), color=(0.0, 0.75, 0.0)) -> int:
        size = V.v3(size[0], size[1], 0)[:2]
        if size[0] == 0 or size[1] == 0:
            raise ValueError("addTileObject: size must be > 0")
        if subdiv[0] < 1 or subdiv[1] < 1:
            raise ValueError("addTileObject: subdivisions must be > 0")
        sub = np.array([size[0] / f32(subdiv[0]), size[1] / f32(subdiv[1])], f32)
        el, az = f32(rotation[0]), f32(rotation[1])
        uuids = []
        for j in range(subdiv[1]):
            for i in range(subdiv[0]):
                sc = (f32(-0.5) * size[0] + f32(f32(i) + f32(0.5)) * sub[0],
                      f32(-0.5) * size[1] + f32(f32(j) + f32(0.5)) * sub[1], 0.0)
                u = self.add_patch(sc, sub, (0.0, 0.0), color)
                p = self.primitives[u]
                if el != 0:
                    p.rotate(-el, "x")
                if az != 0:
                    p.rotate(-az, "z")
                p.translate(center)
                uuids.append(u)
        T = np.eye(4, dtype=f32).reshape(16)
        T = matmult(make_scale_matrix((size[0], size[1], 1)), T)
        T = matmult(make_rotation_matrix(-el, "x"), T)
        T = matmult(make_rotation_matrix(-az, "z"), T)
        T = matmult(make_translation_matrix(center), T)
        return self._register_object(OBJECT_TILE, uuids, transform=T, origin=V.v3(center),
                                     extra={"subdiv": tuple(int(s) for s in subdiv)})

    @ports("Context::addTubeObject")
    def add_tube_object(self, radial_subdivisions: int, nodes, radius, colors=None) -> int:
        nodes = [V.v3(n) for n in nodes]
        radius = [f32(r) for r in radius]
        n = len(nodes)
        if n == 0:
            raise ValueError("addTubeObject: node and radius arrays are empty")
        if n != len(radius):
            raise ValueError("addTubeObject: nodes and radius must agree in size")
        colors = [(0.0, 0.75, 0.0)] * n if colors is None else list(colors)
        if n != len(colors):
            raise ValueError("addTubeObject: nodes and color must agree in size")
        nr = int(radial_subdivisions)
        ring = _tube_rings(nr, nodes, radius)
        uuids = []
        for j in range(nr):
            for i in range(n - 1):
                uuids.append(self.add_triangle(ring[i][j], ring[i + 1][j + 1], ring[i][j + 1], colors[i]))
                uuids.append(self.add_triangle(ring[i][j], ring[i + 1][j], ring[i + 1][j + 1], colors[i]))
        oid = self._register_object(OBJECT_TUBE, uuids, transform=np.eye(4, dtype=f32).reshape(16),
                                    extra={"nodes": np.array(nodes), "radius": np.array(radius), "subdiv": nr})
        self.objects[oid].origin = self.object_center(oid)
        return oid

    # ---- textured overloads ----
    @ports("Context::addTileObject(texture)", "Context::addTileObject(texture,texture_repeat)")
    def add_tile_object_textured(self, center, size, rotation, subdiv, texture_file: str, texture_repeat=(1, 1)) -> int:
        from .primitives import PATCH
        from .texture import does_texture_file_exist, validate_texture_file_extension
        if not validate_texture_file_extension(texture_file):
            raise ValueError(f"ERROR (Context::addTileObject): Texture file {texture_file} is not PNG or JPEG format.")
        if not does_texture_file_exist(texture_file):
            raise FileNotFoundError(f"ERROR (Context::addTileObject): Texture file {texture_file} does not exist.")
        size = V.v3(size[0], size[1], 0)[:2]
        sdx, sdy = int(subdiv[0]), int(subdiv[1])
        if size[0] == 0 or size[1] == 0:
            raise ValueError("addTileObject: size must be > 0")
        if sdx < 1 or sdy < 1:
            raise ValueError("addTileObject: subdivisions must be > 0")
        if texture_repeat[0] < 1 or texture_repeat[1] < 1:
            raise ValueError("addTileObject: number of texture repeats must be > 0")
        rx, ry = _correct_texture_repeat((sdx, sdy), texture_repeat)
        sub = np.array([size[0] / f32(sdx), size[1] / f32(sdy)], f32)
        spx, spy = sdx // rx, sdy // ry
        uvx, uvy = f32(1.0) / f32(spx), f32(1.0) / f32(spy)
        self.add_texture(texture_file)
        W, H = self.textures[texture_file].get_image_resolution()
        if sdx >= rx * W or sdy >= ry * H:
            raise ValueError(f"ERROR (Context::addTileObject): The resolution of the texture image '{texture_file}' is lower than "
                             "the number of tile subdivisions. Increase resolution of the texture image.")
        el, az = f32(rotation[0]), f32(rotation[1])
        uuids = []
        for j in range(sdy):
            for i in range(sdx):
                sc = (f32(-0.5) * size[0] + f32(f32(i) + f32(0.5)) * sub[0],
                      f32(-0.5) * size[1] + f32(f32(j) + f32(0.5)) * sub[1], 0.0)
                il, jl = i % spx, j % spy
                uv = np.array([[f32(il) * uvx, f32(jl) * uvy], [f32(il + 1) * uvx, f32(jl) * uvy],
                               [f32(il + 1) * uvx, f32(jl + 1) * uvy], [f32(il) * uvx, f32(jl + 1) * uvy]], f32)
                u = self._new_textured(PATCH, np.eye(4, dtype=f32).reshape(16), texture_file, uv)
                p = self.primitives[u]
                p.scale((sub[0], sub[1], 1))
                p.translate(sc)
                if el != 0:
                    p.rotate(-el, "x")
                if az != 0:
                    p.rotate(-az, "z")
                p.translate(center)
                uuids.append(u)
        T = np.eye(4, dtype=f32).reshape(16)
        T = matmult(make_scale_matrix((size[0], size[1], 1)), T)
        T = matmult(make_rotation_matrix(-el, "x"), T)
        T = matmult(make_rotation_matrix(-az, "z"), T)
        T = matmult(make_translation_matrix(center), T)
        # the Tile stores the requested repeat, not the corrected one (Context_object.cpp)
        return self._register_object(OBJECT_TILE, uuids, transform=T, origin=V.v3(center),
                                     extra={"subdiv": (sdx, sdy), "texture_file": texture_file,
                                            "texture_repeat": (int(texture_repeat[0]), int(texture_repeat[1]))})

    @ports("Context::addTubeObject(texture)", "Context::addTubeObject(texture,textureuv_ufrac)")
    def add_tube_object_textured(self, radial_subdivisions: int, nodes, radius, texture_file: str, textureuv_ufrac=None) -> int:
        """Like the colored overload, but triangles whose area (times solid fraction) is not > 0 are
        deleted right after creation, so their UUIDs are skipped."""
        from .texture import does_texture_file_exist, validate_texture_file_extension
        n = len(nodes)
        if textureuv_ufrac is None:
            textureuv_ufrac = [f32(i) / f32(n - 1) for i in range(n)]   # n == 1 gives 0/0 = NaN, as in Helios
        if not validate_texture_file_extension(texture_file):
            raise ValueError(f"ERROR (Context::addTubeObject): Texture file {texture_file} is not PNG or JPEG format.")
        if not does_texture_file_exist(texture_file):
            raise FileNotFoundError(f"ERROR (Context::addTubeObject): Texture file {texture_file} does not exist.")
        nodes = [V.v3(p) for p in nodes]
        radius = [f32(r) for r in radius]
        ufrac = [f32(x) for x in textureuv_ufrac]
        if n == 0:
            raise ValueError("addTubeObject: node and radius arrays are empty")
        if n != len(radius):
            raise ValueError("addTubeObject: nodes and radius must agree in size")
        if n != len(ufrac):
            raise ValueError("addTubeObject: nodes and textureuv_ufrac must agree in size")
        nr = int(radial_subdivisions)
        ring = _tube_rings(nr, nodes, radius)
        uvv = [f32(j) / f32(nr) for j in range(nr + 1)]

        def uv(i, j):
            return (ufrac[i], uvv[j])

        uuids = []
        for j in range(nr):
            for i in range(n - 1):
                for (a, b, c) in (((i, j), (i + 1, j + 1), (i, j + 1)), ((i, j), (i + 1, j), (i + 1, j + 1))):
                    u = self.add_triangle_textured(ring[a[0]][a[1]], ring[b[0]][b[1]], ring[c[0]][c[1]], texture_file,
                                                   uv(*a), uv(*b), uv(*c))
                    if self.primitives[u].area() > 0:
                        uuids.append(u)
                    else:
                        self.delete_primitive(u)
        oid = self._register_object(OBJECT_TUBE, uuids, transform=np.eye(4, dtype=f32).reshape(16),
                                    extra={"nodes": np.array(nodes), "radius": np.array(radius), "subdiv": nr,
                                           "texture_file": texture_file, "textureuv_ufrac": np.array(ufrac, f32)})
        self.objects[oid].origin = self.object_center(oid)
        return oid

    @ports("Context::addPolymeshObject")
    def add_polymesh_object(self, uuids) -> int:
        uuids = list(uuids)
        if not uuids:
            raise ValueError("addPolymeshObject: UUIDs array is empty")
        if any(u not in self.primitives for u in uuids):
            raise ValueError("addPolymeshObject: a UUID does not exist")
        first = self.primitives[uuids[0]].parent_object
        if first != 0 and first in self.objects and self.objects[first].type == OBJECT_POLYMESH \
                and all(self.primitives[u].parent_object == first for u in uuids):
            return first
        free = [u for u in uuids if self.primitives[u].parent_object == 0]
        if not free:
            raise ValueError("addPolymeshObject: all primitives already belong to another object")
        T = matmult(make_translation_matrix(self.primitives[free[0]].vertices()[0]), np.eye(4, dtype=f32).reshape(16))
        oid = self._register_object(OBJECT_POLYMESH, free, transform=T)
        self.objects[oid].origin = self.object_center(oid)
        return oid

    @ports("Context::getAllObjectIDs")
    def all_object_ids(self) -> list[int]:
        return sorted(self.objects)

    @ports("Context::getObjectType")
    def object_type(self, oid: int) -> int:
        return self.objects[oid].type

    @ports("Context::getObjectPrimitiveUUIDs")
    def object_primitive_uuids(self, oid: int) -> list[int]:
        return list(self.objects[oid].uuids)

    @ports("Context::getPrimitiveParentObjectID")
    def primitive_parent_object(self, uuid: int) -> int:
        return self.primitives[uuid].parent_object

    @ports("Context::getObjectTransformationMatrix")
    def object_transform(self, oid: int) -> np.ndarray:
        return self.objects[oid].transform.copy()

    @ports("Context::getObjectBoundingBox")
    def object_bounding_box(self, oid: int):
        V_ = np.concatenate([self.primitives[u].vertices() for u in self.objects[oid].uuids])
        return V_.min(0), V_.max(0)

    @ports("Context::getObjectCenter")
    def object_center(self, oid: int) -> np.ndarray:
        lo, hi = self.object_bounding_box(oid)
        return f32(0.5) * (lo + hi)


def _tube_rings(nr: int, nodes, radius):
    """Vertex rings of addTubeObject (shared by the colored and textured overloads, which are
    identical up to the triangles). ``ring[i][j]`` for node i and radial index j in 0..nr."""
    n = len(nodes)
    rc = [f32(1e-5) if (r < f32(1e-5) and r >= 0) else r for r in radius]
    cf = [V.cosf(f32(f32(f32(2.0) * V.PI_F) * f32(j)) / f32(nr)) for j in range(nr + 1)]
    sf = [V.sinf(f32(f32(f32(2.0) * V.PI_F) * f32(j)) / f32(nr)) for j in range(nr + 1)]
    X, Y, Z = V.v3(1, 0, 0), V.v3(0, 1, 0), V.v3(0, 0, 1)
    initial_radial = X.copy()
    prev_axial = prev_radial = None
    ring = [[None] * (nr + 1) for _ in range(n)]

    def fallback(axial):
        fr = X.copy()
        if abs(V.dot(axial, fr)) > f32(0.95):
            fr = Y.copy()
        if abs(axial[2]) > f32(0.95):
            fr = X.copy()
        return V.normalize(V.cross(axial, fr))

    for i in range(n):
        if radius[i] < 0:
            raise ValueError("addTubeObject: radius must be positive")
        if i == 0:
            axial = nodes[i + 1] - nodes[i]  # n == 1 raises IndexError, as Helios reads past the end
            mag = V.magnitude(axial)
            axial = Z.copy() if mag < f32(1e-6) else V.div(axial, mag)
            if abs(V.dot(axial, initial_radial)) > f32(0.95):
                initial_radial = Y.copy()
            if abs(axial[2]) > f32(0.95):
                initial_radial = X.copy()
            prev_radial = V.normalize(V.cross(axial, initial_radial))
        else:
            if i == n - 1:
                axial = nodes[i] - nodes[i - 1]
            else:
                axial = f32(0.5) * ((nodes[i] - nodes[i - 1]) + (nodes[i + 1] - nodes[i]))
            mag = V.magnitude(axial)
            axial = Z.copy() if mag < f32(1e-6) else V.div(axial, mag)
            rot_axis = V.cross(prev_axial, axial)
            if float(V.magnitude(rot_axis)) > 1e-5:
                angle = V.acos_d(np.clip(V.dot(prev_axial, axial), f32(-1), f32(1)))
                prev_radial = V.rotate_point_about_line(prev_radial, V.v3(0, 0, 0), rot_axis, angle)
            else:
                prev_radial = fallback(axial)
        prev_axial = axial
        radial = prev_radial
        ortho = V.normalize(V.cross(radial, axial))
        for j in range(nr + 1):
            normal = (f32(cf[j] * rc[i]) * radial) + (f32(sf[j] * rc[i]) * ortho)
            ring[i][j] = nodes[i] + normal
    return ring


def _correct_texture_repeat(subdiv, texture_repeat):
    """correctTextureRepeat (Context_object.cpp): clamp to subdiv, then shrink until it divides it."""
    rx, ry = min(int(subdiv[0]), int(texture_repeat[0])), min(int(subdiv[1]), int(texture_repeat[1]))
    while subdiv[0] % rx != 0:
        rx -= 1
    while subdiv[1] % ry != 0:
        ry -= 1
    return rx, ry
