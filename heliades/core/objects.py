# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_object.cpp; Copyright (C) 2016-2026 Brian Bailey).
"""Compound objects: tile, tube, polymesh, and object queries (mixed into Scene)."""
from __future__ import annotations

import functools
import math

import numpy as np

from . import vec as V
from ..migration_registry import ports

f32 = np.float32

from .primitives import make_rotation_matrix, make_scale_matrix, make_translation_matrix, matmult

OBJECT_TILE, OBJECT_SPHERE, OBJECT_TUBE, OBJECT_BOX, OBJECT_DISK, OBJECT_POLYMESH, OBJECT_CONE = range(7)


class CompoundObject:
    """oid, type, member UUIDs (in object order), the object transform, origin and type-specific data
    (``extra``, in the object's local frame). ``member_index()`` caches the members as a slice (when they
    are consecutive UUIDs, the usual case) or an index array, for the batched row operations."""

    __slots__ = ("oid", "type", "_uuids", "transform", "origin", "extra", "_ix", "_alive_ix", "_alive_gen")

    def __init__(self, oid: int, type: int, uuids, transform=None, origin=None, extra=None):
        self.oid = oid
        self.type = type
        self.uuids = uuids
        self.transform = np.eye(4, dtype=f32).reshape(16) if transform is None else transform
        self.origin = np.zeros(3, f32) if origin is None else origin
        self.extra = {} if extra is None else extra

    @property
    def uuids(self) -> list[int]:
        return self._uuids

    @uuids.setter
    def uuids(self, uu):
        self._uuids = list(uu)
        self._ix = None
        self._alive_gen = -1          # store.deaths when _alive_ix (the existing members) was last computed

    def member_index(self):
        """The member UUIDs as ``slice(a, b)`` if they are a, a+1, ..., b-1, else an int64 array."""
        if self._ix is None:
            ix = np.asarray(self._uuids, np.int64)
            n = len(ix)
            if n and ix[0] >= 0 and ix[-1] - ix[0] == n - 1 and (n == 1 or bool((ix[1:] - ix[:-1] == 1).all())):
                self._ix = slice(int(ix[0]), int(ix[-1]) + 1)
            else:
                self._ix = ix
        return self._ix

    def __repr__(self):
        return f"CompoundObject(oid={self.oid}, type={self.type}, n={len(self._uuids)})"


class ObjectsMixin:
    def _tube_ring_list(self, nr: int, nodes, radius):
        """The vertex rings of a new tube (``_tube_rings``; fast mode overrides it)."""
        return _tube_rings(nr, nodes, radius)

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
        ring = self._tube_ring_list(nr, nodes, radius)
        uuids = []
        for j in range(nr):
            for i in range(n - 1):
                uuids.append(self.add_triangle(ring[i][j], ring[i + 1][j + 1], ring[i][j + 1], colors[i]))
                uuids.append(self.add_triangle(ring[i][j], ring[i + 1][j], ring[i + 1][j + 1], colors[i]))
        oid = self._register_object(OBJECT_TUBE, uuids, transform=np.eye(4, dtype=f32).reshape(16),
                                    extra={"nodes": np.array(nodes), "radius": np.array(radius), "subdiv": nr,
                                           "triangle_vertices": _ring_array(ring),
                                           "colors": np.array([np.asarray(c, f32)[:3] for c in colors], f32).reshape(-1, 3)})
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
        ring = self._tube_ring_list(nr, nodes, radius)
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
                                           "texture_file": texture_file, "textureuv_ufrac": np.array(ufrac, f32),
                                           "triangle_vertices": _ring_array(ring), "colors": np.zeros((n, 3), f32)})
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
    def all_object_ids(self, include_hidden: bool = False) -> list[int]:
        """Sorted object IDs. Hidden objects are skipped, as Helios does (include_hidden is a Heliades
        extension for snapshots of prototype geometry)."""
        hid = getattr(self, "hidden_objects", ())
        return sorted(o for o in self.objects if include_hidden or o not in hid)

    @ports("Context::getObjectCount")
    def object_count(self) -> int:
        """All objects, hidden ones included (objects.size())."""
        return len(self.objects)

    @ports("Context::doesObjectExist")
    def does_object_exist(self, oid: int) -> bool:
        return int(oid) in self.objects

    @ports("Context::getObjectArea", "CompoundObject::getArea")
    def object_area(self, oid: int) -> np.float32:
        """Sum of the member primitives' areas (hidden ones too) accumulated in double, then float."""
        obj = self.objects.get(int(oid))
        if obj is None:
            raise ValueError(f"ERROR (Context::getObjectPointer): Object ID of {oid} does not exist in the Context.")
        ix = self._alive_members(obj)
        rows = np.arange(ix.start, ix.stop) if isinstance(ix, slice) else ix
        if not len(rows):
            return f32(0.0)
        # sum in member order with a double accumulator from 0 (accumulate is sequential; 0 + a0 == a0)
        return f32(np.add.accumulate(self._store.areas_rows(rows).astype(np.float64))[-1])

    @ports("Context::getObjectPrimitiveCount")
    def object_primitive_count(self, oid: int) -> int:
        return len(self.objects[int(oid)].uuids)

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
        obj = self.objects[oid]
        st = self._store
        ix = np.asarray(obj.uuids, np.int64)
        if not len(ix) or not ((ix >= 0) & (ix < st.n)).all() or not st.alive[ix].all():
            V_ = np.concatenate([self.primitives[u].vertices() for u in obj.uuids])   # raises like before
            return V_.min(0), V_.max(0)
        V_, nv = st.vertices_rows(ix)
        V_ = V_[np.arange(4)[None, :] < nv[:, None]]
        return V_.min(0), V_.max(0)

    @ports("Context::getObjectCenter")
    def object_center(self, oid: int) -> np.ndarray:
        lo, hi = self.object_bounding_box(oid)
        return f32(0.5) * (lo + hi)


@functools.lru_cache(maxsize=64)
def _tube_factors(nr: int):
    """cfact/sfact: cosf/sinf(2.f * PI_F * float(j) / float(nr)) for j in 0..nr (tuples; cached per nr)."""
    cf = tuple(V.cosf(f32(f32(f32(2.0) * V.PI_F) * f32(j)) / f32(nr)) for j in range(nr + 1))
    sf = tuple(V.sinf(f32(f32(f32(2.0) * V.PI_F) * f32(j)) / f32(nr)) for j in range(nr + 1))
    return cf, sf


@functools.lru_cache(maxsize=256)
def _tube_triangle_index(nr: int, n: int):
    """Flat ring-vertex indices (into triangle_vertices.reshape(-1, 3)) of the tube triangles in member order
    (radial slot j slowest, segment i fastest, two per quad): (i,j),(i+1,j+1),(i,j+1) and (i,j),(i+1,j),(i+1,j+1)."""
    J, I = np.meshgrid(np.arange(nr), np.arange(n - 1), indexing="ij")
    J, I = J.reshape(-1), I.reshape(-1)
    w = nr + 1
    m = 2 * len(J)
    i0, i1, i2 = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m, np.int64)
    i0[0::2] = i0[1::2] = I * w + J
    i1[0::2], i2[0::2] = (I + 1) * w + J + 1, I * w + J + 1
    i1[1::2], i2[1::2] = (I + 1) * w + J, (I + 1) * w + J + 1
    for a in (i0, i1, i2):
        a.setflags(write=False)
    return i0, i1, i2


def _tube_frames(nodes, radius=None):
    """(radial, orthogonal) unit vectors per node: the parallel-transport frame shared verbatim by
    addTubeObject, Tube::appendTubeSegment and Tube::recomputeCrossSections. ``radius`` (if given) is
    checked for negative entries, as the first two do inside the loop (the frame has no side effects, so
    checking first raises the same error).

    Everything that does not depend on the transported radial vector (axial directions, rotation axes and
    angles, the radial-independent parts of rotatePointAboutLine) is computed for all nodes at once with the
    scalar code's float32 op sequence; only the radial chain itself runs node by node."""
    n = len(nodes)
    if radius is not None:
        for i in range(n):
            if radius[i] < 0:
                raise ValueError("addTubeObject: radius must be positive")
    if n < 2:
        return _tube_frames_scalar(nodes, radius)       # n == 1 raises IndexError, as Helios reads past the end
    N = np.array([np.asarray(p, f32).reshape(3) for p in nodes], f32)
    d = N[1:] - N[:-1]
    A = np.empty((n, 3), f32)
    A[0] = d[0]
    A[n - 1] = d[n - 2]
    if n > 2:
        A[1:n - 1] = f32(0.5) * (d[:-1] + d[1:])
    mag = np.sqrt((A[:, 0] * A[:, 0] + A[:, 1] * A[:, 1]) + A[:, 2] * A[:, 2])
    small = mag < f32(1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        A = A * (f32(1.0) / mag)[:, None]
    A[small] = (0, 0, 1)
    X, Y = V.v3(1, 0, 0), V.v3(0, 1, 0)
    # node 0
    a0 = A[0]
    initial_radial = X
    if abs(V.dot(a0, initial_radial)) > f32(0.95):
        initial_radial = Y
    if abs(a0[2]) > f32(0.95):
        initial_radial = X
    R = np.empty((n, 3), f32)
    R[0] = V.normalize(V.cross(a0, initial_radial))
    # nodes 1..n-1: rotation axis cross(prev_axial, axial), angle acos(clamp(dot)) (double acos)
    P, Q = A[:-1], A[1:]
    rot = np.empty((n - 1, 3), f32)
    rot[:, 0] = P[:, 1] * Q[:, 2] - P[:, 2] * Q[:, 1]
    rot[:, 1] = P[:, 2] * Q[:, 0] - P[:, 0] * Q[:, 2]
    rot[:, 2] = P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]
    rmag = np.sqrt((rot[:, 0] * rot[:, 0] + rot[:, 1] * rot[:, 1]) + rot[:, 2] * rot[:, 2])
    turn = rmag.astype(np.float64) > 1e-5
    dots = np.clip((P[:, 0] * Q[:, 0] + P[:, 1] * Q[:, 1]) + P[:, 2] * Q[:, 2], f32(-1), f32(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        U = rot * (f32(1.0) / rmag)[:, None]                 # div(line_direction, mag)
    u_, v_, w_ = U[:, 0], U[:, 1], U[:, 2]
    zero = np.zeros(n - 1, f32)                           # line_base = (0, 0, 0)
    with np.errstate(invalid="ignore"):
        KX, KY, KZ = zero * (v_ * v_ + w_ * w_), zero * (u_ * u_ + w_ * w_), zero * (u_ * u_ + v_ * v_)
        BX, BY, BZ = zero * v_ + zero * w_, zero * u_ + zero * w_, zero * u_ + zero * v_
        CX, CY, CZ = -zero * v_ + zero * w_, zero * u_ - zero * w_, -zero * u_ + zero * v_
    cols = [c.tolist() for c in (KX, KY, KZ, BX, BY, BZ, CX, CY, CZ)]
    uu, vv, ww = u_.tolist(), v_.tolist(), w_.tolist()
    rm, tl, dl = rmag.tolist(), turn.tolist(), dots.tolist()
    x, y, z = (f32(c) for c in R[0])
    for i in range(1, n):
        k = i - 1
        if tl[k]:
            theta = f32(math.acos(dl[k]))
            if theta != 0 and not f32(rm[k]) < f32(1e-6):     # else rotatePointAboutLine returns the point
                st, ct = f32(math.sin(float(theta))), f32(math.cos(float(theta)))
                one_ct = f32(1) - ct
                u, v, w = f32(uu[k]), f32(vv[k]), f32(ww[k])
                kx, ky, kz, bx, by, bz, cx, cy, cz = (f32(c[k]) for c in cols)
                ix = ((bx - u * x) - v * y) - w * z
                iy = ((by - u * x) - v * y) - w * z
                iz = ((bz - u * x) - v * y) - w * z
                px = ((kx - u * ix) * one_ct + x * ct) + ((cx - w * y) + v * z) * st
                py = ((ky - v * iy) * one_ct + y * ct) + ((cy + w * x) - u * z) * st
                pz = ((kz - w * iz) * one_ct + z * ct) + ((cz - v * x) + u * y) * st
                x, y, z = px, py, pz
        else:
            fr = X
            ax = A[i]
            if abs(V.dot(ax, fr)) > f32(0.95):
                fr = Y
            if abs(ax[2]) > f32(0.95):
                fr = X
            x, y, z = (f32(c) for c in V.normalize(V.cross(ax, fr)))
        R[i] = (x, y, z)
    # orthogonal = normalize(cross(radial, axial)) (vec3::normalize keeps a zero vector)
    C = np.empty((n, 3), f32)
    C[:, 0] = R[:, 1] * A[:, 2] - R[:, 2] * A[:, 1]
    C[:, 1] = R[:, 2] * A[:, 0] - R[:, 0] * A[:, 2]
    C[:, 2] = R[:, 0] * A[:, 1] - R[:, 1] * A[:, 0]
    m2 = (C[:, 0] * C[:, 0] + C[:, 1] * C[:, 1]) + C[:, 2] * C[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        O = np.where((m2 <= 0)[:, None], C, C * (f32(1.0) / np.sqrt(m2))[:, None])
    return [(R[i], O[i]) for i in range(n)]


def _tube_frames_scalar(nodes, radius=None):
    """Node-by-node reference of :func:`_tube_frames` (used for n < 2 and by the tests)."""
    n = len(nodes)
    X, Y, Z = V.v3(1, 0, 0), V.v3(0, 1, 0), V.v3(0, 0, 1)
    initial_radial = X.copy()
    prev_axial = prev_radial = None
    frames = []

    def fallback(axial):
        fr = X.copy()
        if abs(V.dot(axial, fr)) > f32(0.95):
            fr = Y.copy()
        if abs(axial[2]) > f32(0.95):
            fr = X.copy()
        return V.normalize(V.cross(axial, fr))

    for i in range(n):
        if radius is not None and radius[i] < 0:
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
        frames.append((radial, V.normalize(V.cross(radial, axial))))
    return frames


def _clamp_radius(radius):
    """radius_clamped: r < 1e-5f and r >= 0 -> 1e-5f."""
    return [f32(1e-5) if (r < f32(1e-5) and r >= 0) else f32(r) for r in radius]


def _tube_rings(nr: int, nodes, radius):
    """Vertex rings of addTubeObject (shared by the colored and textured overloads, which are
    identical up to the triangles). ``ring[i][j]`` for node i and radial index j in 0..nr."""
    rc = _clamp_radius(radius)
    cf, sf = _tube_factors(nr)
    frames = _tube_frames(nodes, radius)
    ring = [[None] * (nr + 1) for _ in range(len(nodes))]
    for i, (radial, ortho) in enumerate(frames):
        for j in range(nr + 1):
            normal = (f32(cf[j] * rc[i]) * radial) + (f32(sf[j] * rc[i]) * ortho)
            ring[i][j] = nodes[i] + normal
    return ring


def _ring_array(ring) -> np.ndarray:
    """Tube::triangle_vertices as a float32 (nodes, nr+1, 3) array (local frame)."""
    return np.array([[np.asarray(v, f32) for v in r] for r in ring], f32).reshape(len(ring), -1, 3)


def _correct_texture_repeat(subdiv, texture_repeat):
    """correctTextureRepeat (Context_object.cpp): clamp to subdiv, then shrink until it divides it."""
    rx, ry = min(int(subdiv[0]), int(texture_repeat[0])), min(int(subdiv[1]), int(texture_repeat[1]))
    while subdiv[0] % rx != 0:
        rx -= 1
    while subdiv[1] % ry != 0:
        ry -= 1
    return rx, ry


# ---- tube editing (Context_object.cpp, Tube class) ------------------------------------------

_IDENTITY16 = np.eye(4, dtype=f32).reshape(16)


def _vecmult(M, v) -> np.ndarray:
    """helios::vecmult(M, vec3): V[i] = 0.f, then V[i] += M[4i+k] * v[k] for k = 0..3 (v[3] = 1)."""
    x = (f32(v[0]), f32(v[1]), f32(v[2]), f32(1.0))
    out = np.empty(3, f32)
    for i in range(3):
        acc = f32(0.0)
        for k in range(4):
            acc = f32(acc + f32(M[4 * i + k] * x[k]))
        out[i] = acc
    return out


def _vecmult_rows(M, P: np.ndarray) -> np.ndarray:
    """_vecmult for every point of P (..., 3): V[i] = 0.f, then V[i] += M[4i+k] * v[k], k = 0..3 (v[3] = 1)."""
    out = np.empty(P.shape, f32)
    one = f32(1.0)
    for i in range(3):
        acc = np.zeros(P.shape[:-1], f32)
        for k in range(3):
            acc = acc + M[4 * i + k] * P[..., k]
        out[..., i] = acc + M[4 * i + 3] * one
    return out


class TubeEditMixin:
    """Tube object setters and getters. Tube data (nodes, radii, colors and the ring vertices
    ``triangle_vertices``) is stored in the object's local frame, as in Helios; every setter first
    bakes a non-identity object transform into it (``Tube::bakeTransform``)."""

    def _tube(self, oid: int, who: str):
        obj = self.objects.get(int(oid))
        if obj is None or obj.type != OBJECT_TUBE:
            raise ValueError(f"ERROR (Context::{who}): ObjectID {oid} is not a tube object or does not exist.")
        return obj

    @ports("Tube::bakeTransform")
    def _bake_tube_transform(self, obj):
        T = obj.transform
        if np.array_equal(T, _IDENTITY16):            # std::equal on floats (-0 == 0)
            return
        nodes_world = self.tube_object_nodes(obj.oid)
        radius_world = self.tube_object_node_radii(obj.oid)
        tv = obj.extra["triangle_vertices"]
        obj.extra["triangle_vertices"] = _vecmult_rows(T, np.asarray(tv, f32))
        obj.extra["nodes"] = np.asarray(nodes_world, f32).reshape(-1, 3)
        obj.extra["radius"] = np.asarray(radius_world, f32).reshape(-1)
        obj.transform = _IDENTITY16.copy()

    @ports("Tube::updateTriangleVertices")
    def _update_tube_triangles(self, obj):
        """Rebuild every member triangle from the rings, by position in the UUID list (radial slot
        slowest, segment fastest): Context::setTriangleVertices -> Triangle::makeTransformationMatrix."""
        from .primitives import TRIANGLE, triangle_transform
        from .store import triangle_transform_rows
        tv, nr, n = obj.extra["triangle_vertices"], int(obj.extra["subdiv"]), len(obj.extra["nodes"])
        uu = obj.uuids

        def put(k, v0, v1, v2):
            if k >= len(uu):        # UUIDs.at(k): a textured tube that dropped zero-area triangles ends here
                raise IndexError(f"ERROR (Tube::updateTriangleVertices): vector::_M_range_check: __n (which is {k}) "
                                 f">= this->size() (which is {len(uu)})")
            p = self.primitives.get(uu[k])
            if p is None or p.type != TRIANGLE:
                raise ValueError(f"ERROR (Context::setTriangleVertices): UUID {uu[k]} is not a triangle in the Context.")
            p.transform = triangle_transform(v0, v1, v2)

        m = 2 * nr * max(n - 1, 0)
        st = self._store
        ix = obj.member_index() if m == len(uu) else np.asarray(uu[:m], np.int64)
        ok = m > 0 and len(uu) >= m
        if ok:
            ok = ix.stop <= st.n if isinstance(ix, slice) else bool(((ix >= 0) & (ix < st.n)).all())
        if ok and st.alive[ix].all() and (st.ptype[ix] == TRIANGLE).all():
            # batched (every row the same op sequence as triangle_transform)
            A = np.asarray(tv, f32).reshape(-1, 3)
            i0, i1, i2 = _tube_triangle_index(nr, n)
            st.T[ix] = triangle_transform_rows(A[i0], A[i1], A[i2])
            return
        ii = 0
        for j in range(nr):
            for i in range(n - 1):       # one triangle at a time, so a failure leaves the same partial update
                put(ii, tv[i][j], tv[i + 1][j + 1], tv[i][j + 1])
                put(ii + 1, tv[i][j], tv[i + 1][j], tv[i + 1][j + 1])
                ii += 2

    def _append_rings(self, obj, node, radius, color):
        """Shared first half of both appendTubeSegment overloads: bake, push the node, recompute the
        frames and the last two rings (with the stored, unclamped radii)."""
        r = f32(radius)
        if r < 0:
            raise ValueError("ERROR (Tube::appendTubeSegment): Node radius must be positive.")
        r = r if f32(1e-5) < r else f32(1e-5)         # std::max((float) 1e-5, node_radius)
        self._bake_tube_transform(obj)
        nr = int(obj.extra["subdiv"])
        cf, sf = _tube_factors(nr)
        tv = [list(ring) for ring in obj.extra["triangle_vertices"]] + [[np.zeros(3, f32) for _ in range(nr + 1)]]
        nodes = [V.v3(x) for x in obj.extra["nodes"]] + [V.v3(node)]
        rad = [f32(x) for x in obj.extra["radius"]] + [r]
        obj.extra["colors"] = np.vstack([obj.extra["colors"].reshape(-1, 3), np.asarray(color, f32)[:3]]).astype(f32)
        n = len(nodes)
        try:
            frames = _tube_frames(nodes, rad)
        except ValueError:
            raise ValueError("ERROR (Context::addTubeObject): Radius of tube must be positive.") from None
        for i in range(max(n - 2, 0), n):
            radial, ortho = frames[i]
            for j in range(nr + 1):
                normal = (f32(cf[j] * rad[i]) * radial) + (f32(sf[j] * rad[i]) * ortho)
                tv[i][j] = nodes[i] + normal
        obj.extra["triangle_vertices"] = np.array(tv, f32).reshape(n, nr + 1, 3)
        obj.extra["nodes"] = np.array(nodes, f32).reshape(-1, 3)
        obj.extra["radius"] = np.array(rad, f32)
        return n, nr

    def _append_reorder(self, obj, n, nr, make_pair):
        """UUIDs stay ordered radial slot slowest, segment fastest: the new segment's two triangles are
        inserted at the end of each radial slot's block."""
        earlier = list(obj.uuids)
        out, k = [], 0
        for j in range(nr):
            for _ in range(n - 2):
                out += [earlier[k], earlier[k + 1]]
                k += 2
            out += make_pair(j)
        obj.uuids = out
        for u in out:
            self.primitives[u].parent_object = obj.oid
        return earlier

    @ports("Context::appendTubeSegment(uint,const vec3&,float,const RGBcolor&)",
           "Tube::appendTubeSegment(const vec3&,float,const RGBcolor&)")
    def append_tube_segment(self, oid: int, node_position, node_radius, node_color):
        obj = self._tube(oid, "appendTubeSegment")
        n, nr = self._append_rings(obj, node_position, node_radius, node_color)
        tv = obj.extra["triangle_vertices"]
        col = tuple(np.asarray(node_color, f32)[:3])

        def pair(j):
            return [self.add_triangle(tv[n - 2][j], tv[n - 1][j + 1], tv[n - 2][j + 1], col),
                    self.add_triangle(tv[n - 2][j], tv[n - 1][j], tv[n - 1][j + 1], col)]

        earlier = self._append_reorder(obj, n, nr, pair)
        self._keep_named_tube_material(earlier, obj.uuids, "", col)
        self._update_tube_triangles(obj)

    @ports("Context::appendTubeSegment(uint,const vec3&,float,const char*,const vec2&)",
           "Tube::appendTubeSegment(const vec3&,float,const char*,const vec2&)")
    def append_tube_segment_textured(self, oid: int, node_position, node_radius, texture_file: str, textureuv_ufrac):
        obj = self._tube(oid, "appendTubeSegment")
        uf = (f32(textureuv_ufrac[0]), f32(textureuv_ufrac[1]))
        if f32(node_radius) < 0:
            raise ValueError("ERROR (Tube::appendTubeSegment): Node radius must be positive.")
        if uf[0] < 0 or uf[1] < 0 or uf[0] > 1 or uf[1] > 1:
            raise ValueError("ERROR (Tube::appendTubeSegment): Texture U fraction must be between 0 and 1.")
        n, nr = self._append_rings(obj, node_position, node_radius, (0.0, 0.0, 0.0))
        tv = obj.extra["triangle_vertices"]
        vv = [f32(j) / f32(nr) for j in range(nr + 1)]

        def pair(j):
            a, b = n - 2, n - 1
            return [self.add_triangle_textured(tv[a][j], tv[b][j + 1], tv[a][j + 1], texture_file,
                                               (uf[0], vv[j]), (uf[1], vv[j + 1]), (uf[0], vv[j + 1])),
                    self.add_triangle_textured(tv[a][j], tv[b][j], tv[b][j + 1], texture_file,
                                               (uf[0], vv[j]), (uf[1], vv[j]), (uf[1], vv[j + 1]))]

        earlier = self._append_reorder(obj, n, nr, pair)
        self._keep_named_tube_material(earlier, obj.uuids, texture_file, (0.0, 0.0, 0.0))
        self._update_tube_triangles(obj)

    @ports("Context::setTubeRadii", "Tube::setTubeRadii")
    def set_tube_radii(self, oid: int, node_radii):
        """Path dependent: each ring vertex is moved along its *current* offset from the node, so a
        radius-only update keeps whatever cross-section shape earlier updates left."""
        obj = self._tube(oid, "setTubeRadii")
        radii = [f32(r) for r in node_radii]
        if len(radii) != len(obj.extra["nodes"]):
            raise ValueError("ERROR (Tube::setTubeRadii): Number of radii in input vector must match number of tube nodes.")
        self._bake_tube_transform(obj)
        nodes = obj.extra["nodes"]
        tv = np.asarray(obj.extra["triangle_vertices"], f32)
        nd = np.asarray(nodes, f32).reshape(-1, 1, 3)
        d = tv - nd                                  # per vertex: nodes[seg] + normalize(tv - nodes[seg]) * r
        m2 = (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = f32(1.0) / np.sqrt(m2)
            axis = np.where((m2 <= 0)[..., None], d, d * inv[..., None])   # vec3::normalize keeps a zero vector
        obj.extra["triangle_vertices"] = (nd + axis * np.asarray(radii, f32).reshape(-1, 1, 1)).astype(f32)
        obj.extra["radius"] = np.array(radii, f32)
        self._update_tube_triangles(obj)

    @ports("Context::setTubeNodes", "Tube::setTubeNodes", "Tube::recomputeCrossSections")
    def set_tube_nodes(self, oid: int, node_xyz):
        obj = self._tube(oid, "setTubeNodes")
        nodes = [V.v3(p) for p in node_xyz]
        if len(nodes) != len(obj.extra["nodes"]):
            raise ValueError("ERROR (Tube::setTubeNodes): Number of nodes in input vector must match number of tube nodes.")
        self._bake_tube_transform(obj)
        nr = int(obj.extra["subdiv"])
        rc = _clamp_radius(obj.extra["radius"])
        cf, sf = _tube_factors(nr)
        fr = _tube_frames(nodes)
        radial = np.array([f[0] for f in fr], f32).reshape(-1, 1, 3)
        ortho = np.array([f[1] for f in fr], f32).reshape(-1, 1, 3)
        r_ = np.array(rc, f32).reshape(-1, 1)
        a = np.array(cf, f32).reshape(1, -1) * r_          # cfact * r, sfact * r per (node, radial slot)
        b = np.array(sf, f32).reshape(1, -1) * r_
        # nodes[i] + cfact*r*radial + sfact*r*orthogonal, left to right
        tv = (np.array(nodes, f32).reshape(-1, 1, 3) + a[..., None] * radial) + b[..., None] * ortho
        obj.extra["nodes"] = np.array(nodes, f32).reshape(-1, 3)
        obj.extra["triangle_vertices"] = tv
        self._update_tube_triangles(obj)

    @ports("Context::scaleTubeGirth", "Tube::scaleTubeGirth")
    def scale_tube_girth(self, oid: int, S):
        obj = self._tube(oid, "scaleTubeGirth")
        S = f32(S)
        self._bake_tube_transform(obj)
        nodes, tv = obj.extra["nodes"], obj.extra["triangle_vertices"].copy()
        rad = obj.extra["radius"].copy()
        with np.errstate(invalid="ignore"):          # a ring collapsed onto its node gives 0 * inf = NaN, as in C++
            for seg in range(len(tv)):
                for k in range(tv.shape[1]):
                    axis = tv[seg][k] - nodes[seg]
                    cur = V.magnitude(axis)
                    axis = V.div(axis, cur)
                    tv[seg][k] = nodes[seg] + (axis * cur) * S
                rad[seg] = f32(rad[seg] * S)
        obj.extra["triangle_vertices"], obj.extra["radius"] = tv, rad
        self._update_tube_triangles(obj)

    @ports("Context::scaleTubeLength", "Tube::scaleTubeLength")
    def scale_tube_length(self, oid: int, S):
        obj = self._tube(oid, "scaleTubeLength")
        S = f32(S)
        self._bake_tube_transform(obj)
        nodes, tv = obj.extra["nodes"].copy(), obj.extra["triangle_vertices"].copy()
        with np.errstate(invalid="ignore"):          # coincident nodes give NaN, as in C++
            for seg in range(len(tv) - 1):
                axis = nodes[seg + 1] - nodes[seg]
                L = V.magnitude(axis)
                axis = V.div(axis, L)
                dL = (axis * L) * f32(f32(1) - S)
                for d in range(seg + 1, len(tv)):
                    nodes[d] = nodes[d] - dL
                    tv[d] = tv[d] - dL
        obj.extra["nodes"], obj.extra["triangle_vertices"] = nodes, tv
        self._update_tube_triangles(obj)

    @ports("Context::getTubeObjectSubdivisionCount")
    def tube_object_subdivision_count(self, oid: int) -> int:
        return int(self._tube(oid, "getTubeObjectSubdivisionCount").extra["subdiv"])

    @ports("Context::getTubeObjectNodeCount")
    def tube_object_node_count(self, oid: int) -> int:
        return len(self._tube(oid, "getTubeObjectNodeCount").extra["nodes"])

    @ports("Context::getTubeObjectNodeColors")
    def tube_object_node_colors(self, oid: int) -> np.ndarray:
        return self._tube(oid, "getTubeObjectNodeColors").extra["colors"].copy()

    @ports("Tube::getTriangleVertices")
    def tube_object_triangle_vertices(self, oid: int) -> np.ndarray:
        """Local-frame ring vertices (no object transform applied), as Helios returns them."""
        return self._tube(oid, "getTubeObjectTriangleVertices").extra["triangle_vertices"].copy()

    @staticmethod
    def _frustum_volume(L, r0, r1) -> np.float32:
        """PI_F * segment_length / 3.f * (r0 * r0 + r0 * r1 + r1 * r1), left to right in float."""
        return f32(f32(f32(V.PI_F * L) / f32(3.0)) * f32(f32(f32(r0 * r0) + f32(r0 * r1)) + f32(r1 * r1)))

    @ports("Context::getTubeObjectVolume", "Tube::getVolume")
    def tube_object_volume(self, oid: int) -> np.float32:
        """World radii (getNodeRadii) but local-frame segment lengths, as Helios computes it."""
        obj = self._tube(oid, "getTubeObjectVolume")
        radii, nodes = self.tube_object_node_radii(oid), obj.extra["nodes"]
        vol = f32(0)
        for i in range(len(radii) - 1):
            vol = f32(vol + self._frustum_volume(V.magnitude(nodes[i + 1] - nodes[i]), f32(radii[i]), f32(radii[i + 1])))
        return vol

    @ports("Context::getTubeObjectSegmentVolume", "Tube::getSegmentVolume")
    def tube_object_segment_volume(self, oid: int, segment_index: int) -> np.float32:
        """Local-frame nodes and radii (no object transform), as Helios computes it."""
        obj = self._tube(oid, "getTubeObjectSegmentVolume")
        nodes, rad = obj.extra["nodes"], obj.extra["radius"]
        k = int(segment_index)
        if k >= len(nodes) - 1:
            raise ValueError("ERROR (Tube::getSegmentVolume): Segment index out of bounds.")
        return self._frustum_volume(V.magnitude(nodes[k + 1] - nodes[k]), f32(rad[k]), f32(rad[k + 1]))

    @ports("Context::pruneTubeNodes", "Tube::pruneTubeNodes")
    def prune_tube_nodes(self, oid: int, node_index: int):
        obj = self._tube(oid, "pruneTubeNodes")
        n = len(obj.extra["nodes"])
        k = int(node_index)
        if k >= n:
            raise ValueError(f"ERROR (Tube::pruneTubeNodes): Node index of {k} is out of bounds.")
        if k <= 1:
            self.delete_object(oid)
            return
        nseg, keep = n - 1, k - 1
        kept, gone, ii = [], [], 0
        for _ in range(int(obj.extra["subdiv"])):
            for i in range(nseg):
                (kept if i < keep else gone).extend(obj.uuids[ii:ii + 2])
                ii += 2
        for key in ("nodes", "triangle_vertices", "radius", "colors"):
            obj.extra[key] = obj.extra[key][:k].copy()
        obj.uuids = kept
        for u in gone:
            self.primitives[u].parent_object = 0   # no longer a member: deleting it must not touch the object
            self.delete_primitives(u)
