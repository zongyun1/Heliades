# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_object.cpp, Context.cpp, global.cpp; Copyright (C) 2016-2026 Brian Bailey).
"""Sphere, disk, cone, box objects (colored overloads) and their type-specific getters. Mixed into Scene.

Every vertex is computed with Helios's float32 operation order. Vertex grids are evaluated with numpy
float32 elementwise ops (each op rounds like the scalar C++ expression), and sinf/cosf are a float64
libm call rounded to float32, as in ``vec.py``.
"""
from __future__ import annotations

import numpy as np

from . import vec as V
from .objects import OBJECT_BOX, OBJECT_CONE, OBJECT_DISK, OBJECT_SPHERE
from .primitives import make_scale_matrix, make_translation_matrix, matmult
from ..migration_registry import ports

f32 = np.float32
_GREEN = (0.0, 0.75, 0.0)
_RED_RGBA = (1.0, 0.0, 0.0, 1.0)
_NAMES = {OBJECT_SPHERE: "Sphere", OBJECT_DISK: "Disk", OBJECT_CONE: "Cone", OBJECT_BOX: "Box"}


def _load_libm():
    """glibc's sinf/cosf are not correctly rounded: about 1 argument in 40 of the form 2*PI_F*j/N
    (N < 64) differs by 1 ulp from the double result rounded to float. Helios calls these exact
    functions, so call them too (falls back to rounded double where no libm is found)."""
    import ctypes
    import ctypes.util
    try:
        m = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        for fn in (m.sinf, m.cosf):
            fn.restype, fn.argtypes = ctypes.c_float, [ctypes.c_float]
        return m.sinf, m.cosf
    except (OSError, AttributeError):
        return None, None


_LIBM_SINF, _LIBM_COSF = _load_libm()


def _libm(fn, dfn, x):
    x = np.asarray(x, f32)
    if fn is None:
        return dfn(x.astype(np.float64)).astype(f32)
    return np.array([fn(float(v)) for v in x.ravel()], f32).reshape(x.shape)


def _sinf(x):
    """C sinf, elementwise."""
    return _libm(_LIBM_SINF, np.sin, x)


def _cosf(x):
    """C cosf, elementwise."""
    return _libm(_LIBM_COSF, np.cos, x)


def _eye():
    return np.eye(4, dtype=f32).reshape(16)


@ports("helios::vecmult")
def _vecmult(M, v) -> np.ndarray:
    """V[i] += M[4i+k] * v[k] for k = 0..3, starting from 0.f, with v[3] = 1."""
    M = np.asarray(M, f32).reshape(4, 4)
    v4 = np.array([v[0], v[1], v[2], 1.0], f32)
    out = np.zeros(4, f32)
    for k in range(4):
        out = out + M[:, k] * v4[k]
    return out[:3].copy()


def _xform_point(T, p) -> np.ndarray:
    """transform[0]*p.x + transform[1]*p.y + transform[2]*p.z + transform[3] (left to right)."""
    T = np.asarray(T, f32).reshape(4, 4)
    p = np.asarray(p, f32)
    return ((T[:3, 0] * p[0] + T[:3, 1] * p[1]) + T[:3, 2] * p[2]) + T[:3, 3]


def _powf2(x) -> np.float32:
    """powf(x, 2): the exact double square rounded once to float, which equals x*x in float32."""
    return f32(float(x) * float(x))


def _powf_half(x) -> np.float32:
    """powf(x, 0.5): correctly rounded sqrt."""
    return f32(np.sqrt(np.float64(f32(x))))


class ShapesMixin:
    # ---- helpers -------------------------------------------------------------------------------

    def _shape(self, oid: int, otype: int):
        o = self.objects.get(oid)
        if o is None or o.type != otype:
            raise ValueError(f"get{_NAMES[otype]}ObjectPointer: ObjectID {oid} is not a {_NAMES[otype].lower()} object")
        return o

    @ports("Context::addTile")
    def _add_tile_prims(self, center, size, rotation, subdiv, color) -> list[int]:
        """Context::addTile: loose patches (no object). Same op order as addTileObject's patch loop."""
        size = np.asarray(size, f32)
        sub = np.array([size[0] / f32(subdiv[0]), size[1] / f32(subdiv[1])], f32)
        el, az = f32(rotation[0]), f32(rotation[1])
        center = V.v3(center)
        uuids = []
        for j in range(int(subdiv[1])):
            for i in range(int(subdiv[0])):
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
        return uuids

    @ports("Context::addTile(texture)", "Context::addTile(texture,texture_repeat)")
    def add_tile_textured(self, center, size, rotation, subdiv, texture_file: str, texture_repeat=(1, 1)) -> list[int]:
        """Context::addTile with a texture (Context.cpp:2005-2104; W5-E, olive leaves): loose textured patches (no
        object). A sub-patch whose solid fraction is 0 is dropped before it takes a UUID."""
        from .objects import _correct_texture_repeat
        from .primitives import PATCH, rotated, scaled, translated
        from .texture import does_texture_file_exist, validate_texture_file_extension
        if not validate_texture_file_extension(texture_file):
            raise ValueError(f"ERROR (Context::addTile): Texture file {texture_file} is not PNG or JPEG format.")
        if not does_texture_file_exist(texture_file):
            raise FileNotFoundError(f"ERROR (Context::addTile): Texture file {texture_file} does not exist.")
        if texture_repeat[0] < 1 or texture_repeat[1] < 1:
            raise ValueError("ERROR (Context::addTile): Number of texture repeats must be greater than 0.")
        sdx, sdy = int(subdiv[0]), int(subdiv[1])
        rx, ry = _correct_texture_repeat((sdx, sdy), texture_repeat)
        size = np.asarray(size, f32)
        sub = np.array([size[0] / f32(sdx), size[1] / f32(sdy)], f32)
        spx, spy = sdx // rx, sdy // ry
        uvx, uvy = f32(1.0) / f32(spx), f32(1.0) / f32(spy)
        self.add_texture(texture_file)
        W, H = self.textures[texture_file].get_image_resolution()
        if sdx >= rx * W or sdy >= ry * H:
            raise ValueError(f"ERROR (Context::addTile): The resolution of the texture image '{texture_file}' is lower than "
                             "the number of tile subdivisions. Increase resolution of the texture image.")
        el, az = f32(rotation[0]), f32(rotation[1])
        center = V.v3(center)
        uuids = []
        for j in range(sdy):
            for i in range(sdx):
                sc = (f32(-0.5) * size[0] + f32(f32(i) + f32(0.5)) * sub[0],
                      f32(-0.5) * size[1] + f32(f32(j) + f32(0.5)) * sub[1], 0.0)
                il, jl = i % spx, j % spy
                uv = np.array([[f32(il) * uvx, f32(jl) * uvy], [f32(il + 1) * uvx, f32(jl) * uvy],
                               [f32(il + 1) * uvx, f32(jl + 1) * uvy], [f32(il) * uvx, f32(jl + 1) * uvy]], f32)
                if self.textures[texture_file].get_solid_fraction(np.minimum(uv, f32(1.0))) == 0:
                    continue
                u = self._new_textured(PATCH, np.eye(4, dtype=f32).reshape(16), texture_file, uv)
                # Primitive::scale / translate / rotate in order, on a local matrix written to the store once
                T = translated(scaled(self._store.T[u], (sub[0], sub[1], 1)), sc)
                if el != 0:
                    T = rotated(T, -el, "x")
                if az != 0:
                    T = rotated(T, -az, "z")
                self._store.T[u] = translated(T, center)
                uuids.append(u)
        return uuids

    # ---- sphere --------------------------------------------------------------------------------

    @ports("Context::addSphereObject")
    def add_sphere_object(self, ndivs: int, center, radius, color=_GREEN) -> int:
        """Scalar ``radius`` gives a sphere, a 3-vector an ellipsoid. ``color`` is RGB (default green)."""
        r = np.asarray(radius, f32)
        radius = np.full(3, r, f32) if r.ndim == 0 else V.v3(r)
        if radius[0] <= 0 or radius[1] <= 0 or radius[2] <= 0:
            raise ValueError("addSphereObject: Radius of sphere must be positive.")
        N = int(ndivs)
        if N < 1:
            # Helios (uint Ndivs = 0) builds an object with no primitives; we refuse instead (W1-C notes)
            raise ValueError("addSphereObject: Ndivs must be >= 1")
        center = V.v3(center)
        half_pi = f32(f32(0.5) * V.PI_F)
        dtheta = f32(V.PI_F / f32(N))
        dphi = f32(f32(f32(2.0) * V.PI_F) / f32(N))
        az = np.arange(N + 1, dtype=f32) * dphi                       # float(j) * dphi

        def pts(el, a):
            """center + (cart.x*r.x, cart.y*r.y, cart.z*r.z), cart = sphere2cart(1, el, a)."""
            el, a = np.broadcast_arrays(np.asarray(el, f32), np.asarray(a, f32))
            ce = f32(1.0) * _cosf(el)
            cart = np.stack([ce * _sinf(a), ce * _cosf(a), f32(1.0) * _sinf(el)], -1)
            return center + cart * radius

        bottom = pts(-half_pi, f32(0))
        top = pts(half_pi, f32(0))
        ring_b = pts(-half_pi + dtheta, az)
        ring_t = pts(half_pi - dtheta, az)
        uuids = []
        for j in range(N):
            uuids.append(self.add_triangle(bottom, ring_b[j], ring_b[j + 1], color))
        for j in range(N):
            uuids.append(self.add_triangle(ring_t[j + 1], ring_t[j], top, color))
        if N > 2:
            el = -half_pi + np.arange(N, dtype=f32) * dtheta          # -0.5f*PI_F + float(i)*dtheta
            G = pts(el[:, None], az[None, :])                          # G[i, j]
            for j in range(N):
                for i in range(1, N - 1):
                    v0, v1, v2, v3 = G[i, j], G[i + 1, j], G[i + 1, j + 1], G[i, j + 1]
                    uuids.append(self.add_triangle(v0, v1, v2, color))
                    uuids.append(self.add_triangle(v0, v2, v3, color))
        T = matmult(make_scale_matrix(radius), _eye())
        T = matmult(make_translation_matrix(center), T)
        # addSphereObject never assigns object_origin, so it keeps vec3's default (0,0,0)
        return self._register_object(OBJECT_SPHERE, uuids, transform=T, origin=np.zeros(3, f32),
                                     extra={"subdiv": N})

    @ports("Context::getSphereObjectCenter", "Sphere::getCenter")
    def sphere_object_center(self, oid: int) -> np.ndarray:
        return _xform_point(self._shape(oid, OBJECT_SPHERE).transform, (0, 0, 0))

    @ports("Context::getSphereObjectRadius", "Sphere::getRadius")
    def sphere_object_radius(self, oid: int) -> np.ndarray:
        T = self._shape(oid, OBJECT_SPHERE).transform
        n0 = _vecmult(T, (0, 0, 0))
        return np.array([V.magnitude(_vecmult(T, e) - n0) for e in np.eye(3, dtype=f32)], f32)

    @ports("Context::getSphereObjectSubdivisionCount")
    def sphere_object_subdivision_count(self, oid: int) -> int:
        return self._shape(oid, OBJECT_SPHERE).extra["subdiv"]

    @ports("Context::getSphereObjectVolume", "Sphere::getVolume")
    def sphere_object_volume(self, oid: int) -> np.float32:
        r = self.sphere_object_radius(oid)
        return f32(f32(f32(f32(f32(f32(4.0) / f32(3.0)) * V.PI_F) * r[0]) * r[1]) * r[2])

    # ---- disk ----------------------------------------------------------------------------------

    @ports("Context::addDiskObject")
    def add_disk_object(self, ndivs, center, size, rotation=(0.0, 0.0), color=_RED_RGBA) -> int:
        """``ndivs`` is an int (Ndivs, 1) or a pair (polar Ndivs.x, radial Ndivs.y).
        ``rotation`` = (elevation, azimuth) in radians; ``color`` RGB or RGBA (default opaque red)."""
        nx, ny = (int(ndivs), 1) if np.ndim(ndivs) == 0 else (int(ndivs[0]), int(ndivs[1]))
        if nx < 1 or ny < 1:
            # Helios has no check: Ndivs.y = 0 sizes a vector with a negative count, Ndivs.x = 0 builds
            # an empty object; we refuse instead (W1-C notes)
            raise ValueError("addDiskObject: subdivisions must be >= 1")
        center = V.v3(center)
        size = np.asarray(size, f32).reshape(2)
        el, az = f32(rotation[0]), f32(rotation[1])
        dtheta = f32(f32(f32(2.0) * V.PI_F) / f32(nx))
        th = dtheta * np.arange(nx + 1, dtype=f32)                     # dtheta * float(t)
        ct, st = _cosf(th), _sinf(th)
        rs = np.arange(ny + 1, dtype=f32)
        rx = f32(size[0] / f32(ny)) * rs                               # size.x / float(Ndivs.y) * float(r)
        ry = f32(size[1] / f32(ny)) * rs

        def p(r, t):
            return (rx[r] * ct[t], ry[r] * st[t], 0.0)

        uuids = []
        for r in range(ny):
            for t in range(nx):
                if r == 0:
                    new = [self.add_triangle((0.0, 0.0, 0.0), p(1, t), p(1, t + 1), color)]
                else:
                    new = [self.add_triangle(p(r, t + 1), p(r, t), p(r + 1, t), color),
                           self.add_triangle(p(r, t + 1), p(r + 1, t), p(r + 1, t + 1), color)]
                for u in new:
                    prim = self.primitives[u]
                    prim.rotate(el, "y")
                    prim.rotate(az, "z")
                    prim.translate(center)
                uuids += new
        # the object transform carries scale and translation only: the rotation is not recorded (W1-C notes)
        T = matmult(make_scale_matrix((size[0], size[1], 1.0)), _eye())
        T = matmult(make_translation_matrix(center), T)
        return self._register_object(OBJECT_DISK, uuids, transform=T, origin=center.copy(),
                                     extra={"subdiv": (nx, ny)})

    @ports("Context::getDiskObjectCenter", "Disk::getCenter")
    def disk_object_center(self, oid: int) -> np.ndarray:
        return _xform_point(self._shape(oid, OBJECT_DISK).transform, (0, 0, 0))

    @ports("Context::getDiskObjectSize", "Disk::getSize")
    def disk_object_size(self, oid: int) -> np.ndarray:
        T = self._shape(oid, OBJECT_DISK).transform
        n0 = _vecmult(T, (0, 0, 0))
        return np.array([V.magnitude(_vecmult(T, (1, 0, 0)) - n0), V.magnitude(_vecmult(T, (0, 1, 0)) - n0)], f32)

    @ports("Context::getDiskObjectSubdivisionCount")
    def disk_object_subdivision_count(self, oid: int) -> int:
        """Only Ndivs.x, as in Helios."""
        return self._shape(oid, OBJECT_DISK).extra["subdiv"][0]

    # ---- cone ----------------------------------------------------------------------------------

    @ports("Context::addConeObject")
    def add_cone_object(self, ndivs: int, node0, node1, radius0, radius1, color=_GREEN) -> int:
        N = int(ndivs)
        if N < 1:
            # Helios (uint Ndivs = 0) divides 0/0 and builds an object with no primitives; we refuse (W1-C notes)
            raise ValueError("addConeObject: Ndivs must be >= 1")
        nodes = [V.v3(node0), V.v3(node1)]
        radii = [f32(radius0), f32(radius1)]
        ang = f32(f32(f32(2.0) * V.PI_F) * np.arange(N + 1, dtype=f32)) / f32(N)   # 2.f*PI_F*float(j)/float(Ndivs)
        cfact, sfact = _cosf(ang), _sinf(ang)
        nvec = V.v3(0.1817, 0.6198, 0.7634)
        X, Y, Z = V.v3(1, 0, 0), V.v3(0, 1, 0), V.v3(0, 0, 1)
        xyz = [None, None]
        for i in range(2):
            vec = nodes[1] - nodes[0]
            if V.magnitude(vec) < f32(1e-6):
                vec = Z.copy()
            convec = V.cross(nvec, vec)
            norm = V.magnitude(convec)
            if norm < f32(1e-6):
                convec = V.cross(vec, X if abs(vec[0]) < f32(0.9) else Y)
                norm = max(V.magnitude(convec), f32(1e-6))
            convec = V.div(convec, norm)
            nvec = V.cross(vec, convec)
            norm = V.magnitude(nvec)
            if norm < f32(1e-6):
                nvec = V.cross(convec, vec)
                norm = max(V.magnitude(nvec), f32(1e-6))
            nvec = V.div(nvec, norm)
            a = (cfact * radii[i])[:, None]                                 # cfact[j] * radii[i]
            b = (sfact * radii[i])[:, None]
            xyz[i] = nodes[i] + (a * nvec + b * convec)
        uuids = []
        for j in range(N):
            uuids.append(self.add_triangle(xyz[0][j], xyz[1][j + 1], xyz[0][j + 1], color))
            uuids.append(self.add_triangle(xyz[0][j], xyz[1][j], xyz[1][j + 1], color))
        oid = self._register_object(OBJECT_CONE, uuids, transform=_eye(),
                                    extra={"nodes": np.array(nodes), "radii": np.array(radii, f32), "subdiv": N})
        self.objects[oid].origin = self.object_center(oid)
        return oid

    @ports("Context::getConeObjectNodes", "Cone::getNodeCoordinates")
    def cone_object_nodes(self, oid: int) -> np.ndarray:
        o = self._shape(oid, OBJECT_CONE)
        return np.array([_xform_point(o.transform, n) for n in o.extra["nodes"]], f32)

    @ports("Context::getConeObjectNode", "Cone::getNodeCoordinate")
    def cone_object_node(self, oid: int, number: int) -> np.ndarray:
        if number not in (0, 1):
            raise ValueError("Cone::getNodeCoordinate: node number must be 0 or 1.")
        return self.cone_object_nodes(oid)[number]

    @ports("Cone::transformedRadius")
    def _cone_transformed_radius(self, T, r) -> np.float32:
        return V.magnitude(_vecmult(T, (r, 0, 0)) - _vecmult(T, (0, 0, 0)))

    @ports("Context::getConeObjectNodeRadii", "Cone::getNodeRadii")
    def cone_object_node_radii(self, oid: int) -> np.ndarray:
        o = self._shape(oid, OBJECT_CONE)
        return np.array([self._cone_transformed_radius(o.transform, r) for r in o.extra["radii"]], f32)

    @ports("Context::getConeObjectNodeRadius", "Cone::getNodeRadius")
    def cone_object_node_radius(self, oid: int, number: int) -> np.float32:
        if number not in (0, 1):
            raise ValueError("Cone::getNodeRadius: node number must be 0 or 1.")
        return self.cone_object_node_radii(oid)[number]

    @ports("Context::getConeObjectSubdivisionCount")
    def cone_object_subdivision_count(self, oid: int) -> int:
        return self._shape(oid, OBJECT_CONE).extra["subdiv"]

    @ports("Context::getConeObjectAxisUnitVector", "Cone::getAxisUnitVector")
    def cone_object_axis_unit_vector(self, oid: int) -> np.ndarray:
        n = self.cone_object_nodes(oid)
        d = n[1] - n[0]
        length = _powf_half(f32(f32(_powf2(d[0]) + _powf2(d[1])) + _powf2(d[2])))
        with np.errstate(invalid="ignore"):
            return V.div(d, length)

    @ports("Context::getConeObjectLength", "Cone::getLength")
    def cone_object_length(self, oid: int) -> np.float32:
        n = self.cone_object_nodes(oid)
        d = n[1] - n[0]
        return _powf_half(f32(f32(_powf2(d[0]) + _powf2(d[1])) + _powf2(d[2])))

    @ports("Context::getConeObjectVolume", "Cone::getVolume")
    def cone_object_volume(self, oid: int) -> np.float32:
        r0, r1 = self.cone_object_node_radii(oid)
        h = self.cone_object_length(oid)
        return f32(f32(f32(V.PI_F * h) / f32(3.0)) * f32(f32(f32(r0 * r0) + f32(r0 * r1)) + f32(r1 * r1)))

    # ---- box -----------------------------------------------------------------------------------

    @ports("Context::addBoxObject")
    def add_box_object(self, center, size, subdiv, color=_GREEN, reverse_normals: bool = False) -> int:
        center, size = V.v3(center), V.v3(size)
        if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
            raise ValueError("addBoxObject: Size of box must be positive.")
        sx, sy, sz = (int(s) for s in subdiv)
        if sx < 1 or sy < 1 or sz < 1:
            raise ValueError("addBoxObject: Number of box subdivisions must be positive.")
        h = f32(0.5)
        hp, p, p15 = f32(h * V.PI_F), V.PI_F, f32(f32(1.5) * V.PI_F)
        dy, dx, dz = V.v3(0, h * size[1], 0), V.v3(h * size[0], 0, 0), V.v3(0, 0, h * size[2])
        xz, yz, xy = (size[0], size[2]), (size[1], size[2]), (size[0], size[1])
        if reverse_normals:
            faces = [(center + dy, xz, (hp, p), (sx, sz)), (center - dy, xz, (hp, 0), (sx, sz)),
                     (center + dx, yz, (hp, p15), (sy, sz)), (center - dx, yz, (hp, hp), (sy, sz)),
                     (center + dz, xy, (p, 0), (sx, sy)), (center - dz, xy, (0, 0), (sx, sy))]
        else:
            faces = [(center + dy, xz, (hp, 0), (sx, sz)), (center - dy, xz, (hp, p), (sx, sz)),
                     (center + dx, yz, (hp, hp), (sy, sz)), (center - dx, yz, (hp, p15), (sy, sz)),
                     (center + dz, xy, (0, 0), (sx, sy)), (center - dz, xy, (p, 0), (sx, sy))]
        uuids = []
        for c, s, rot, sub in faces:
            uuids += self._add_tile_prims(c, s, rot, sub, color)
        T = matmult(make_scale_matrix(size), _eye())
        T = matmult(make_translation_matrix(center), T)
        return self._register_object(OBJECT_BOX, uuids, transform=T, origin=center.copy(),
                                     extra={"subdiv": (sx, sy, sz)})

    @ports("Context::getBoxObjectCenter", "Box::getCenter")
    def box_object_center(self, oid: int) -> np.ndarray:
        return _xform_point(self._shape(oid, OBJECT_BOX).transform, (0, 0, 0))

    @ports("Context::getBoxObjectSize", "Box::getSize")
    def box_object_size(self, oid: int) -> np.ndarray:
        T = self._shape(oid, OBJECT_BOX).transform
        n0 = _vecmult(T, (0, 0, 0))
        return np.array([V.magnitude(_vecmult(T, e) - n0) for e in np.eye(3, dtype=f32)], f32)

    @ports("Context::getBoxObjectSubdivisionCount")
    def box_object_subdivision_count(self, oid: int) -> tuple[int, int, int]:
        return self._shape(oid, OBJECT_BOX).extra["subdiv"]

    @ports("Context::getBoxObjectVolume", "Box::getVolume")
    def box_object_volume(self, oid: int) -> np.float32:
        s = self.box_object_size(oid)
        return f32(f32(s[0] * s[1]) * s[2])
