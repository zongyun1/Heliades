# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_primitive.cpp, global.cpp; Copyright (C) 2016-2026 Brian Bailey).
"""Structure-of-arrays primitive storage and the batched float32 kernels that read and transform it.

Helios keeps one heap object per primitive (``Patch``/``Triangle`` with a ``float transform[16]``). Heliades
keeps every primitive field in one growable numpy array per field, indexed by UUID (UUIDs are handed out
in increasing order and never reused, so the UUID *is* the row). ``Scene.primitives`` is a dict-like view
(:class:`PrimitiveMap`) that hands out light :class:`~heliades.core.primitives.Primitive` proxies for code
that works on one primitive at a time; bulk code (object transforms, copies, snapshots, areas) works on
rows directly.

**Bit-exactness.** Each batched kernel below performs, for every row, exactly the float32 operation
sequence of the scalar C++ code it replaces, in the same order; it only vectorises *across* rows.
numpy's elementwise float32 ``+ - * / sqrt`` are IEEE correctly rounded, like the SSE code Helios is built
to, so the results are bit-identical. Never replace these loops by ``@``/``einsum``/``sum`` (different
summation order, possible FMA) and keep every operand float32 (no float64 promotion).
See ``docs/core/PERFORMANCE.md``.
"""
from __future__ import annotations

import copy

import numpy as np

f32 = np.float32

PATCH, TRIANGLE, VOXEL = 0, 1, 2

PATCH_Y = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]], f32)
TRI_Y = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], f32)

# generic triangle (0,0,0),(0,1,0),(1,1,0) padded with ones; its inverse in double (constant)
_V1 = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], np.float64).reshape(4, 4)
INV_V1_F32 = np.linalg.inv(_V1).astype(f32)  # exact small rationals, so any inversion method agrees


# ---- batched float32 kernels -----------------------------------------------------------------------

def matmult_rows(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """helios::matmult(L, R[n]) for a batch: ``L`` is one (16,) matrix or (n,16) matrices, ``R`` is (n,16).
    M[4i+j] = 0.f, then M[4i+j] += L[4i+k] * R[4k+j] for k = 0..3 (float32 after every step)."""
    if L.ndim == 1 and len(R) > 256:
        # same per-entry sequence, on contiguous (n,) columns (faster than (n,4,4) broadcasting for large n)
        Rt = np.ascontiguousarray(R.reshape(-1, 16).T)
        M = np.zeros((16, len(Rt[0])), f32)
        tmp = np.empty(len(Rt[0]), f32)
        for i in range(4):
            for j in range(4):
                m = M[4 * i + j]
                for k in range(4):
                    np.multiply(Rt[4 * k + j], L[4 * i + k], out=tmp)
                    m += tmp
        return M.T
    R3 = R.reshape(-1, 4, 4)
    if L.ndim == 1:
        L3 = L.reshape(1, 4, 4)
    else:
        L3 = L.reshape(-1, 4, 4)
    M = np.zeros(R3.shape, f32)
    for k in range(4):
        M = M + L3[:, :, k:k + 1] * R3[:, k:k + 1, :]
    return M.reshape(-1, 16)


def apply_rows(T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Patch/Triangle::getVertices for a batch: (n,16) transforms on the (k,3) canonical shape ->
    (n,k,3), each coordinate ``T[4r]*y.x + T[4r+1]*y.y + T[4r+2]*y.z + T[4r+3]`` left to right."""
    out = np.empty((len(T), len(Y), 3), f32)
    for r in range(3):
        a, b, c, d = T[:, 4 * r, None], T[:, 4 * r + 1, None], T[:, 4 * r + 2, None], T[:, 4 * r + 3, None]
        out[:, :, r] = ((a * Y[None, :, 0] + b * Y[None, :, 1]) + c * Y[None, :, 2]) + d
    return out


def triangle_transform_rows(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Triangle::makeTransformationMatrix for (n,3) vertex arrays -> (n,16):
    transform[j+4i] += V2[k+4i] * float(invV1[j+4k]) for k = 0..3, from 0."""
    n = len(v0)
    V2 = np.ones((n, 4, 4), f32)
    V2[:, :3, 0], V2[:, :3, 1], V2[:, :3, 2] = v0, v1, v2
    T = np.zeros((n, 4, 4), f32)
    for k in range(4):
        T = T + V2[:, :, k:k + 1] * INV_V1_F32[None, k:k + 1, :]
    return T.reshape(n, 16)


def _mag_rows(v: np.ndarray) -> np.ndarray:
    """vec3::magnitude per row: sqrtf((x*x + y*y) + z*z)."""
    return np.sqrt((v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1]) + v[..., 2] * v[..., 2])


def _cross_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(a.shape, f32)
    out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return out


def _norm_rows(v: np.ndarray) -> np.ndarray:
    """helios::normalize (free function): no zero guard, v * (1/sqrtf(m2))."""
    m2 = (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + v[:, 2] * v[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return v * (f32(1.0) / np.sqrt(m2))[:, None]


# ---- storage -----------------------------------------------------------------------------------------

_AFFINE_ROW = np.array([0, 0, 0, 1], f32)


class PrimitiveStore:
    """One row per UUID. ``alive`` marks existing primitives (deleted and skipped UUIDs are dead rows)."""

    _FIELDS = (("T", (16,), f32, 0), ("ptype", (), np.int8, -1), ("color", (4,), f32, 0), ("parent", (), np.int64, 0),
               ("sf", (), f32, 1), ("mat", (), np.int64, -1), ("alive", (), np.bool_, False),
               ("uv", (4, 2), f32, 0), ("nuv", (), np.int8, 0))

    def __init__(self, cap: int = 1024):
        self.cap = 0
        self.n = 0              # rows in use: max UUID + 1
        self.count = 0          # alive rows
        for name, shape, dt, fill in self._FIELDS:
            setattr(self, name, np.full((0, *shape), fill, dt))
        self.tex = np.empty(0, object)          # texture file (str) or None
        self.extra: dict[int, dict] = {}        # other per-primitive attributes (rare), by UUID
        self.reserve(cap)

    def reserve(self, need: int):
        if need <= self.cap:
            return
        cap = max(int(need), 2 * self.cap, 1024)
        for name, shape, dt, fill in self._FIELDS:
            old = getattr(self, name)
            new = np.full((cap, *shape), fill, dt)
            new[: self.cap] = old
            setattr(self, name, new)
        tex = np.empty(cap, object)
        tex[: self.cap] = self.tex
        self.tex = tex
        self.cap = cap

    def exists(self, u) -> bool:
        try:
            u = int(u)
        except (TypeError, ValueError):
            return False
        return 0 <= u < self.n and bool(self.alive[u])

    def add(self, u: int, ptype: int, T, color):
        self.reserve(u + 1)
        self.T[u] = T
        self.ptype[u] = ptype
        self.color[u] = color
        self.parent[u] = 0
        self.sf[u] = 1
        self.mat[u] = -1
        self.tex[u] = None
        self.nuv[u] = 0
        self.extra.pop(u, None)
        if not self.alive[u]:
            self.alive[u] = True
            self.count += 1
        self.n = max(self.n, u + 1)

    def add_rows(self, u0: int, ptype: int, T: np.ndarray, color) -> np.ndarray:
        """add() for the fresh rows u0 .. u0+len(T)-1, all of one type and color."""
        m = len(T)
        self.reserve(u0 + m)
        sl = slice(u0, u0 + m)
        if self.alive[sl].any():
            raise ValueError("add_rows: rows already in use")
        self.T[sl] = T
        self.ptype[sl] = ptype
        self.color[sl] = color
        self.parent[sl] = 0
        self.sf[sl] = 1
        self.mat[sl] = -1
        self.tex[sl] = None
        self.nuv[sl] = 0
        for u in range(u0, u0 + m):
            self.extra.pop(u, None)
        self.alive[sl] = True
        self.count += m
        self.n = max(self.n, u0 + m)
        return np.arange(u0, u0 + m)

    def kill(self, u: int):
        if self.alive[u]:
            self.alive[u] = False
            self.count -= 1

    def copy_rows(self, src: np.ndarray, dst0: int) -> np.ndarray:
        """Copy rows ``src`` (alive) to the fresh rows dst0..dst0+len(src)-1, every field; parent = 0."""
        m = len(src)
        dst = np.arange(dst0, dst0 + m)
        self.reserve(dst0 + m)
        for name, _, _, _ in self._FIELDS:
            a = getattr(self, name)
            a[dst0:dst0 + m] = a[src]
        self.tex[dst0:dst0 + m] = self.tex[src]
        self.parent[dst0:dst0 + m] = 0
        if self.extra:
            for s_, d_ in zip(src.tolist(), dst.tolist()):
                e = self.extra.get(s_)
                if e is not None:
                    self.extra[d_] = copy.deepcopy(e)
        self.count += m                         # fresh rows, src rows are alive
        self.n = max(self.n, dst0 + m)
        return dst

    def alive_uuids(self) -> np.ndarray:
        return np.flatnonzero(self.alive[: self.n])

    # ---- batched getters (rows must be alive) ----
    def vertices_rows(self, idx: np.ndarray):
        """(m,4,3) padded vertices and (m,) vertex counts (4 patch, 3 triangle, 0 voxel)."""
        idx = np.asarray(idx, np.int64)
        V = np.zeros((len(idx), 4, 3), f32)
        nv = np.zeros(len(idx), np.int32)
        t = self.ptype[idx]
        for ty, Y in ((PATCH, PATCH_Y), (TRIANGLE, TRI_Y)):
            sel = np.flatnonzero(t == ty)
            if len(sel):
                V[sel, : len(Y)] = apply_rows(self.T[idx[sel]], Y)
                nv[sel] = len(Y)
        return V, nv

    def areas_rows(self, idx: np.ndarray, V=None) -> np.ndarray:
        """Primitive::getArea per row: patch (|v1-v0| * |v3-v0|) * sf, triangle (0.5 * |e1 x e2|) * sf."""
        idx = np.asarray(idx, np.int64)
        if V is None:
            V, _ = self.vertices_rows(idx)
        t = self.ptype[idx]
        sf = self.sf[idx]
        A = np.zeros(len(idx), f32)
        p = np.flatnonzero(t == PATCH)
        if len(p):
            v = V[p]
            A[p] = (_mag_rows(v[:, 1] - v[:, 0]) * _mag_rows(v[:, 3] - v[:, 0])) * sf[p]
        q = np.flatnonzero(t == TRIANGLE)
        if len(q):
            v = V[q]
            A[q] = (f32(0.5) * _mag_rows(_cross_rows(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]))) * sf[q]
        return A

    def normals_rows(self, idx: np.ndarray, V=None) -> np.ndarray:
        """Primitive::getNormal per row: patch normalize(T[2],T[6],T[10]); triangle normalize((v1-v0) x (v2-v1))."""
        idx = np.asarray(idx, np.int64)
        if V is None:
            V, _ = self.vertices_rows(idx)
        t = self.ptype[idx]
        N = np.zeros((len(idx), 3), f32)
        p = np.flatnonzero(t == PATCH)
        if len(p):
            T = self.T[idx[p]]
            N[p] = _norm_rows(np.stack([T[:, 2], T[:, 6], T[:, 10]], 1))
        q = np.flatnonzero(t == TRIANGLE)
        if len(q):
            v = V[q]
            N[q] = _norm_rows(_cross_rows(v[:, 1] - v[:, 0], v[:, 2] - v[:, 1]))
        return N

    def patch_sizes_rows(self, idx: np.ndarray) -> np.ndarray:
        """Patch::getSize per row: (|v1-v0|, |v3-v0|)."""
        v = apply_rows(self.T[np.asarray(idx, np.int64)], PATCH_Y)
        return np.stack([_mag_rows(v[:, 1] - v[:, 0]), _mag_rows(v[:, 3] - v[:, 0])], 1)

    # ---- batched transforms ----
    def translate_rows(self, idx, shift):
        """translateObjectPrimitives_private: add the shift to the translation column of affine rows,
        T(shift) * M for the others. ``idx`` is a slice or an index array."""
        M = self.T
        sub = M[idx]
        s = np.array([shift[0], shift[1], shift[2]], f32)
        if (sub[:, 12:16] == _AFFINE_ROW).all():
            # columns 3, 7, 11 (the translation column) each += its shift component
            if isinstance(idx, slice):
                sub[:, 3:12:4] += s
            else:
                M[idx, 3:12:4] = sub[:, 3:12:4] + s
            return
        s0, s1, s2 = s[0], s[1], s[2]
        aff = (sub[:, 12] == 0) & (sub[:, 13] == 0) & (sub[:, 14] == 0) & (sub[:, 15] == 1)
        rows = np.arange(len(M))[idx]
        a, na = rows[aff], rows[~aff]
        M[a, 3] = M[a, 3] + s0
        M[a, 7] = M[a, 7] + s1
        M[a, 11] = M[a, 11] + s2
        Tt = np.eye(4, dtype=f32).reshape(16)
        Tt[3], Tt[7], Tt[11] = s0, s1, s2
        M[na] = matmult_rows(Tt, M[na])

    def left_multiply_rows(self, idx, T: np.ndarray):
        """M = T * M (helios::matmult) for every row in ``idx`` (a slice or an index array)."""
        self.T[idx] = matmult_rows(np.asarray(T, f32).reshape(16), self.T[idx])


class PrimitiveMap:
    """Dict-like ``UUID -> Primitive`` view of a :class:`PrimitiveStore` (``Scene.primitives``). Iteration is
    in increasing UUID order, which is creation order, as the dict it replaces iterated."""

    __slots__ = ("_s",)

    def __init__(self, store: PrimitiveStore):
        self._s = store

    def _proxy(self, u: int):
        from .primitives import Primitive
        return Primitive(self._s, u)

    def __getitem__(self, u):
        if not self._s.exists(u):
            raise KeyError(u)
        return self._proxy(int(u))

    def get(self, u, default=None):
        if not self._s.exists(u):
            return default
        return self._proxy(int(u))

    def __contains__(self, u) -> bool:
        return self._s.exists(u)

    def __len__(self) -> int:
        return self._s.count

    def __iter__(self):
        return iter(self._s.alive_uuids().tolist())

    def keys(self):
        return self._s.alive_uuids().tolist()

    def values(self):
        return [self._proxy(u) for u in self.keys()]

    def items(self):
        return [(u, self._proxy(u)) for u in self.keys()]

    def __delitem__(self, u):
        if not self._s.exists(u):
            raise KeyError(u)
        self._s.kill(int(u))
