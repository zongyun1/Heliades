# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_object.cpp: Polymesh::setTopology, Polymesh::computeVertexNormals and the
# topology getters, makeNormalMatrix, transformNormalByMatrix; global.cpp: deg2rad, acos_safe;
# Copyright (C) 2016-2026 Brian Bailey).
"""Polymesh indexed face set (shared vertices, faces, vertex normals/UVs), float32 op for op.

Like Helios, the face set is stored in the object-local frame (``CompoundObject.extra["topology"]``) so
object transforms only touch ``CompoundObject.transform``; the getters apply it on read.

Faces refer to their primitives by *position in the object's member list* rather than by UUID. Helios
keys them by UUID and remaps the keys in ``copyObject``; ``copyPrimitive`` keeps member order, so a
position is invariant under the deep copy that ``ObjectTransformsMixin.copy_object`` makes and no
remapping hook is needed there. Deleting a member primitive (``Polymesh::onChildPrimitiveDeleted``,
which compacts the face set) is not ported: the getters raise once the member list has changed size.
"""
from __future__ import annotations

import math

import numpy as np

from . import vec as V
from ..migration_registry import ports

f32 = np.float32

NORMAL_SOURCE_NONE, NORMAL_SOURCE_AUTHORED, NORMAL_SOURCE_COMPUTED = 0, 1, 2


@ports("helios::deg2rad")
def deg2rad(deg) -> np.float32:
    return f32(f32(f32(deg) * f32(math.pi)) / f32(180))


@ports("helios::acos_safe")
def acos_safe(x) -> np.float32:
    x = f32(x)
    if x < -1.0:
        x = f32(-1)
    elif x > 1.0:
        x = f32(1)
    return V.acosf(x)


def make_normal_matrix(T):
    """makeNormalMatrix (Context_object.cpp): inverse transpose of the 3x3 block, or None if singular."""
    T = np.asarray(T, f32)
    a, b, c, d, e, f, g, h, i = T[0], T[1], T[2], T[4], T[5], T[6], T[8], T[9], T[10]
    det = f32(f32(f32(a * f32(e * i - f * h)) - f32(b * f32(d * i - f * g))) + f32(c * f32(d * h - e * g)))
    if abs(float(det)) < float(f32(1e-20)):
        return None
    inv = f32(f32(1) / det)
    N = np.empty(9, f32)
    N[0] = f32(e * i - f * h) * inv
    N[3] = -f32(b * i - c * h) * inv
    N[6] = f32(b * f - c * e) * inv
    N[1] = -f32(d * i - f * g) * inv
    N[4] = f32(a * i - c * g) * inv
    N[7] = -f32(a * f - c * d) * inv
    N[2] = f32(d * h - e * g) * inv
    N[5] = -f32(a * h - b * g) * inv
    N[8] = f32(a * e - b * d) * inv
    return N


def _normalize_cols(X: np.ndarray) -> np.ndarray:
    """n = (m > 0) ? n / m : n with vec3::operator/ (multiply by 1/m), on (N,3) rows."""
    m = np.sqrt((X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]) + X[:, 2] * X[:, 2])
    out = X.copy()
    pos = m > f32(0)
    with np.errstate(divide="ignore"):
        inv = f32(1) / m
    out[pos] = X[pos] * inv[pos, None]
    return out


def _empty_topology():
    return dict(vertices=np.zeros((0, 3), f32), faces=np.zeros((0, 3), np.int64), face_members=np.zeros(0, np.int64),
                n_members=0, normals=np.zeros((0, 3), f32), uv=np.zeros((0, 2), f32), normal_source=NORMAL_SOURCE_NONE)


def topology(obj) -> dict:
    t = obj.extra.get("topology")
    if t is None:
        return _empty_topology()
    if t["n_members"] != len(obj.uuids) and len(t["faces"]):
        raise NotImplementedError(f"Polymesh object {obj.oid}: members changed since its topology was set "
                                  "(Polymesh::onChildPrimitiveDeleted is not ported)")
    return t


@ports("Polymesh::setTopology")
def set_topology(obj, vertices, faces, face_uuids, vertex_normals=(), vertex_uv=(), normal_source=NORMAL_SOURCE_NONE):
    """Store a global-frame face set in the object's local frame (object transform inverted)."""
    Vg = np.asarray(vertices, f32).reshape(-1, 3)
    F = np.asarray(faces, np.int64).reshape(-1, 3)
    fu = [int(u) for u in face_uuids]
    Ng = np.asarray(vertex_normals, f32).reshape(-1, 3)
    UV = np.asarray(vertex_uv, f32).reshape(-1, 2)
    nv = len(Vg)
    if len(F) != len(fu):
        raise ValueError(f"ERROR (Polymesh::setTopology): The face table has {len(F)} entries but {len(fu)} primitive UUIDs were given. "
                         "Every face must correspond to exactly one primitive.")
    if len(Ng) and len(Ng) != nv:
        raise ValueError(f"ERROR (Polymesh::setTopology): {len(Ng)} vertex normals were given for {nv} vertices.")
    if len(UV) and len(UV) != nv:
        raise ValueError(f"ERROR (Polymesh::setTopology): {len(UV)} texture coordinates were given for {nv} vertices.")
    if not len(Ng) and normal_source != NORMAL_SOURCE_NONE:
        raise ValueError("ERROR (Polymesh::setTopology): A vertex normal source was specified but no vertex normals were given.")
    if len(F) and ((F < 0).any() or (F >= nv).any()):
        raise ValueError(f"ERROR (Polymesh::setTopology): Face references vertex indices out of range; the mesh has only {nv} vertices.")
    T = obj.transform
    Ti = make_normal_matrix(T)
    if Ti is None:
        raise ValueError("ERROR (Polymesh::setTopology): The object transformation matrix is singular and cannot be inverted.")
    sx, sy, sz = Vg[:, 0] - T[3], Vg[:, 1] - T[7], Vg[:, 2] - T[11]
    loc = np.empty_like(Vg)
    loc[:, 0] = (Ti[0] * sx + Ti[3] * sy) + Ti[6] * sz
    loc[:, 1] = (Ti[1] * sx + Ti[4] * sy) + Ti[7] * sz
    loc[:, 2] = (Ti[2] * sx + Ti[5] * sy) + Ti[8] * sz
    ln = np.empty_like(Ng)
    ln[:, 0] = (T[0] * Ng[:, 0] + T[4] * Ng[:, 1]) + T[8] * Ng[:, 2]
    ln[:, 1] = (T[1] * Ng[:, 0] + T[5] * Ng[:, 1]) + T[9] * Ng[:, 2]
    ln[:, 2] = (T[2] * Ng[:, 0] + T[6] * Ng[:, 1]) + T[10] * Ng[:, 2]
    pos = {u: k for k, u in enumerate(obj.uuids)}
    obj.extra["topology"] = dict(vertices=loc, faces=F.copy(), face_members=np.array([pos[u] for u in fu], np.int64),
                                 n_members=len(obj.uuids), normals=_normalize_cols(ln), uv=UV.copy(),
                                 normal_source=int(normal_source))


@ports("Polymesh::getVertices")
def vertices_global(obj) -> np.ndarray:
    L = topology(obj)["vertices"]
    T = obj.transform
    out = np.empty_like(L)
    for r in range(3):
        out[:, r] = ((T[4 * r] * L[:, 0] + T[4 * r + 1] * L[:, 1]) + T[4 * r + 2] * L[:, 2]) + T[4 * r + 3]
    return out


@ports("Polymesh::getVertexNormals")
def vertex_normals_global(obj) -> np.ndarray:
    L = topology(obj)["normals"]
    if not len(L):
        return np.zeros((0, 3), f32)
    N = make_normal_matrix(obj.transform)
    if N is None:
        raise ValueError("ERROR (Polymesh::getVertexNormals): The object transformation matrix is singular, so surface normals are undefined.")
    out = np.empty_like(L)
    for r in range(3):
        out[:, r] = (N[3 * r] * L[:, 0] + N[3 * r + 1] * L[:, 1]) + N[3 * r + 2] * L[:, 2]
    return _normalize_cols(out)


@ports("Polymesh::computeVertexNormals")
def compute_vertex_normals(obj, crease_angle_degrees):
    t = topology(obj)
    F = t["faces"]
    if not len(F):
        raise ValueError(f"ERROR (Polymesh::computeVertexNormals): Polymesh object {obj.oid} carries no mesh topology, so vertex normals cannot be computed.")
    crease_angle_degrees = f32(crease_angle_degrees)
    if crease_angle_degrees < f32(0) or crease_angle_degrees > f32(180):
        raise ValueError(f"ERROR (Polymesh::computeVertexNormals): Crease angle of {crease_angle_degrees} degrees is out of range.")
    crease = deg2rad(crease_angle_degrees)
    Vl = t["vertices"]
    v0, v1, v2 = Vl[F[:, 0]], Vl[F[:, 1]], Vl[F[:, 2]]
    a, b = v1 - v0, v2 - v0
    cr = np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                   a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], 1).astype(f32)
    cm = np.sqrt((cr[:, 0] * cr[:, 0] + cr[:, 1] * cr[:, 1]) + cr[:, 2] * cr[:, 2])
    areas = f32(0.5) * cm
    fn = np.zeros_like(cr)
    pos = cm > f32(0)
    with np.errstate(divide="ignore"):
        inv = f32(1) / cm
    fn[pos] = cr[pos] * inv[pos, None]

    incident = [[] for _ in range(len(Vl))]
    for f in range(len(F)):
        for c in range(3):
            incident[F[f, c]].append(f)
    has_uv = len(t["uv"]) > 0
    sv, sn, suv = [], [], []
    newF = F.copy()
    fnl = [fn[f] for f in range(len(F))]
    for v in range(len(Vl)):
        inc = incident[v]
        if not inc:
            sv.append(Vl[v]); sn.append(np.zeros(3, f32))
            if has_uv:
                suv.append(t["uv"][v])
            continue
        groups: list[list[int]] = []
        for fi in inc:
            placed = False
            for g in groups:
                for m in g:
                    if acos_safe(V.dot(fnl[fi], fnl[m])) <= crease:
                        g.append(fi)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                groups.append([fi])
        for g in groups:
            acc = np.zeros(3, f32)
            for m in g:
                acc = acc + fnl[m] * areas[m]
            mag = V.magnitude(acc)
            acc = acc * f32(f32(1) / mag) if mag > f32(0) else fnl[g[0]].copy()
            k = len(sv)
            sv.append(Vl[v]); sn.append(acc)
            if has_uv:
                suv.append(t["uv"][v])
            for m in g:
                for c in range(3):
                    if F[m, c] == v:
                        newF[m, c] = k
    t["vertices"] = np.array(sv, f32).reshape(-1, 3)
    t["normals"] = np.array(sn, f32).reshape(-1, 3)
    t["uv"] = np.array(suv, f32).reshape(-1, 2)
    t["faces"] = newF
    t["normal_source"] = NORMAL_SOURCE_COMPUTED
    obj.extra["topology"] = t
