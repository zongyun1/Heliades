# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios core (Context_object.cpp, Context_primitive.cpp, global.cpp; Copyright (C) 2016-2026 Brian Bailey).
"""Fast mode: the Scene with deferred, composed object transforms and deferred tube rebuilds.

``Scene(mode="fast")`` returns a :class:`FastScene`. It has the whole Scene API and makes exactly the same
random draws, but drops the bit-exactness of the member geometry (see ``docs/core/FAST_MODE.md``):

* **Object transforms** (``translateObject``, ``rotateObject``, ``scaleObject*``) update the object matrix
  eagerly with Helios's float32 operations (so the object matrices, and the tube nodes derived from them, stay
  bit-identical to exact mode), but only compose the operation into a pending per-object float64 matrix ``P``.
  The member primitives' matrices are updated once, ``T <- float32(P @ T)``, when something reads them.
* **Tube edits** (``setTubeNodes``, ``setTubeRadii``, ``appendTubeSegment``) keep the nodes and a per-node
  ring radius (the "recipe"); the ring vertices and the member triangles are rebuilt once, on read.
  Ring vertices are always ``node + c * (cos * radial + sin * orthogonal)`` with the parallel-transport frame
  of the current nodes, which is what the eager Helios sequence produces up to rounding (``setTubeRadii``
  rescales along the current ring offsets, ``appendTubeSegment`` recomputes the last two rings from the frames
  of the extended node list, earlier rings depend only on earlier nodes). A tube whose rings stop being of that
  form (rotated by an object transform, ``scaleTubeGirth`` / ``scaleTubeLength``, ``pruneTubeNodes``) falls back
  to editing its stored rings, as exact mode does.
* Float math on those paths is float64 numpy / Python floats (any order, no libm-by-ctypes).

**Flushing.** Pending work is applied

* for one object or the parents of some primitives, by the object / primitive getters (areas, vertices,
  normals, bounding boxes, centers, copies of single primitives, tube ring vertices) and by the primitive proxies
  (``scene.primitives[u].transform`` etc.);
* for everything, by any other read of the primitive matrices: :class:`FastPrimitiveStore` makes ``T`` a property
  that flushes all pending work before it hands out the array (so ``snapshot()``, XML/OBJ export and any code that
  reads ``scene._store.T`` see current geometry without knowing about fast mode). Store methods that only append
  rows (``add``, ``add_rows``, ``copy_rows``, ``reserve``) and the fast paths below use the raw array.

Copies of an object carry its pending matrix and tube recipe, so copying does not force a flush.
"""
from __future__ import annotations

import math

import numpy as np

from . import vec as V
from .objects import (OBJECT_TUBE, _clamp_radius, _tube_factors, _tube_frames, _tube_triangle_index, _vecmult_rows)
from .object_transforms import (_ids, make_rotation_matrix, make_rotation_matrix_about, make_rotation_matrix_axis,
                                make_scale_matrix_about_point, matmult)
from .primitives import PATCH, TRIANGLE, Primitive
from .scene import Scene
from .materials import UnorderedUIntMapOrder
from .store import INV_V1_F32, PrimitiveMap, PrimitiveStore

f32 = np.float32
f64 = np.float64

_I4 = np.eye(4)
_INV_V1_64 = INV_V1_F32.astype(f64)
_FLUSH_CHUNK = 262144          # rows per batched flush product (bounds the float64 temporaries)


# ---- store and primitive proxies ---------------------------------------------------------------------------

class FastPrimitiveStore(PrimitiveStore):
    """PrimitiveStore whose ``T`` property flushes the scene's pending transforms before any read."""

    def __init__(self, cap: int = 1024):
        self._quiet = 0
        self._dirty = False
        self.on_read = None                     # Scene._flush_all
        self.flush_reads = 0                    # statistics: global flushes forced through ``T``
        super().__init__(cap)

    @property
    def T(self):
        if self._dirty and not self._quiet:
            self.flush_reads += 1
            self.on_read()
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    def reserve(self, need: int):
        self._quiet += 1
        try:
            super().reserve(need)
        finally:
            self._quiet -= 1

    def add(self, u, ptype, T, color):
        self._quiet += 1
        try:
            super().add(u, ptype, T, color)
        finally:
            self._quiet -= 1

    def add_rows(self, u0, ptype, T, color):
        self._quiet += 1
        try:
            return super().add_rows(u0, ptype, T, color)
        finally:
            self._quiet -= 1

    def copy_rows(self, src, dst0):
        """Raw row copy: the caller has flushed the source rows or carries their pending state to the copies."""
        self._quiet += 1
        try:
            return super().copy_rows(src, dst0)
        finally:
            self._quiet -= 1


class _Quiet:
    """``with _Quiet(store):`` reads of ``store.T`` do not flush (the caller flushed what it reads)."""

    __slots__ = ("s",)

    def __init__(self, store):
        self.s = store

    def __enter__(self):
        self.s._quiet += 1

    def __exit__(self, *exc):
        self.s._quiet -= 1


class FastPrimitive(Primitive):
    """Primitive proxy that flushes only its parent object before reading or writing its matrix."""

    __slots__ = ()

    def _sync(self):
        st = self._s
        if st._dirty:
            st.on_object(int(st.parent[self.uuid]))

    @property
    def transform(self) -> np.ndarray:
        self._sync()
        return self._s._T[self.uuid].copy()

    @transform.setter
    def transform(self, T):
        self._sync()
        self._s._T[self.uuid] = np.asarray(T, f32).reshape(16)

    def scale(self, s):
        from .primitives import scaled
        self.transform = scaled(self.transform, s)

    def rotate(self, radians, axis):
        from .primitives import rotated
        self.transform = rotated(self.transform, radians, axis)

    def translate(self, t):
        from .primitives import translated
        self.transform = translated(self.transform, t)


class FastPrimitiveMap(PrimitiveMap):
    __slots__ = ()

    def _proxy(self, u: int):
        return FastPrimitive(self._s, u)


# ---- float64 helpers -----------------------------------------------------------------------------------------

def _tube_frames_fast(N: np.ndarray):
    """(radial, orthogonal) per node, (n, 3) float64 each: :func:`heliades.core.objects._tube_frames` with the
    branch decisions taken on the same float32 quantities, but the transported radial chain and the
    normalisations in float64 (Python floats in the sequential loop)."""
    n = len(N)
    N = np.asarray(N, f32).reshape(n, 3)
    if n < 2:
        fr = _tube_frames([N[i] for i in range(n)])
        return np.array([f[0] for f in fr], f64).reshape(-1, 3), np.array([f[1] for f in fr], f64).reshape(-1, 3)
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
    P, Q = A[:-1], A[1:]
    rot = np.empty((n - 1, 3), f32)
    rot[:, 0] = P[:, 1] * Q[:, 2] - P[:, 2] * Q[:, 1]
    rot[:, 1] = P[:, 2] * Q[:, 0] - P[:, 0] * Q[:, 2]
    rot[:, 2] = P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]
    rmag = np.sqrt((rot[:, 0] * rot[:, 0] + rot[:, 1] * rot[:, 1]) + rot[:, 2] * rot[:, 2])
    turn = (rmag.astype(f64) > 1e-5).tolist()
    dots = np.clip((P[:, 0] * Q[:, 0] + P[:, 1] * Q[:, 1]) + P[:, 2] * Q[:, 2], f32(-1), f32(1))
    tiny = (rmag < f32(1e-6)).tolist()
    with np.errstate(divide="ignore", invalid="ignore"):
        U = (rot.astype(f64) / rmag.astype(f64)[:, None]).tolist()
    dl = dots.astype(f64).tolist()
    Al = A.astype(f64).tolist()

    def fallback(a):
        fx, fy, fz = 1.0, 0.0, 0.0
        if abs(a[0]) > 0.95:          # |dot(axial, X)| > 0.95f (the float comparison: a[0] is an exact float)
            fx, fy = 0.0, 1.0
        if abs(a[2]) > 0.95:
            fx, fy = 1.0, 0.0
        cx, cy, cz = a[1] * fz - a[2] * fy, a[2] * fx - a[0] * fz, a[0] * fy - a[1] * fx
        m = math.sqrt(cx * cx + cy * cy + cz * cz)
        return (cx / m, cy / m, cz / m) if m > 0 else (cx, cy, cz)

    R = np.empty((n, 3), f64)
    x, y, z = fallback(Al[0])
    R[0] = (x, y, z)
    for i in range(1, n):
        k = i - 1
        if turn[k]:
            c_ = dl[k]
            if c_ < 1.0 and not tiny[k]:               # theta = acos(c) != 0
                theta = math.acos(c_)
                s, c = math.sin(theta), math.cos(theta)
                oc = 1.0 - c
                u, v, w = U[k]
                dd = u * x + v * y + w * z
                x, y, z = (u * dd * oc + x * c + (v * z - w * y) * s,
                           v * dd * oc + y * c + (w * x - u * z) * s,
                           w * dd * oc + z * c + (u * y - v * x) * s)
        else:
            x, y, z = fallback(Al[i])
        R[i] = (x, y, z)
    A64 = A.astype(f64)
    C = np.cross(R, A64)
    m = np.sqrt((C * C).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        O = np.where((m > 0)[:, None], C / m[:, None], C)
    return R, O


def _rings_fast(nr: int, nodes: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(n, nr+1, 3) float32 ring vertices ``node + (cf*c) * radial + (sf*c) * orthogonal``."""
    N = np.asarray(nodes, f32).reshape(-1, 3)
    R, O = _tube_frames_fast(N)
    cf, sf = _tube_factors(nr)
    cf = np.asarray(cf, f64)
    sf = np.asarray(sf, f64)
    c = np.asarray(c, f64).reshape(-1, 1)
    a, b = cf[None, :] * c, sf[None, :] * c                   # (n, nr+1)
    tv = N.astype(f64)[:, None, :] + a[..., None] * R[:, None, :] + b[..., None] * O[:, None, :]
    return tv.astype(f32)


def _triangle_rows_fast(v0, v1, v2) -> np.ndarray:
    """Triangle::makeTransformationMatrix for (m,3) vertices, as one float64 product -> (m,16) float32."""
    m = len(v0)
    V2 = np.ones((m, 4, 4), f64)
    V2[:, :3, 0], V2[:, :3, 1], V2[:, :3, 2] = v0, v1, v2
    return (V2 @ _INV_V1_64).astype(f32).reshape(m, 16)


def _rows_of(ix) -> np.ndarray:
    return np.arange(ix.start, ix.stop) if isinstance(ix, slice) else np.asarray(ix, np.int64)


# ---- lazy primitive iteration order ---------------------------------------------------------------------------

class LazyMapOrder(UnorderedUIntMapOrder):
    """``Context::primitives`` hash order, replayed only when someone iterates it: insertions and erasures are
    logged and applied, in order, on the first iteration after them (the same final order as eager updates)."""

    def __init__(self):
        super().__init__()
        self._log: list[int] = []

    def insert(self, k: int):
        self._log.append(int(k))

    def insert_many(self, ks):
        self._log.extend(ks)

    def erase(self, k: int):
        self._log.append(~int(k))

    def _sync(self):
        log = self._log
        if log:
            self._log = []
            ins, era = super().insert, super().erase
            for k in log:
                if k >= 0:
                    ins(k)
                else:
                    era(~k)

    def __iter__(self):
        self._sync()
        return super().__iter__()


# ---- the scene -----------------------------------------------------------------------------------------------

class FastSceneMixin:
    """Deferred object transforms and tube rebuilds (see the module docstring). Mixed in before Scene."""

    mode = "fast"
    _store_cls = FastPrimitiveStore
    _map_cls = FastPrimitiveMap

    def __init__(self, mode: str = "fast"):
        super().__init__("fast")
        self._pend: dict[int, np.ndarray] = {}      # oid -> float64 4x4 not yet applied to the member rows
        self._tdirty: set[int] = set()              # tubes whose member triangles must be rebuilt from the rings
        self._recipe: dict[int, np.ndarray] = {}    # tube oid -> per-node ring radius c (rings are frame rings)
        self._stale: set[int] = set()               # tubes (with a recipe) whose stored ring vertices are out of date
        self.primitive_order = LazyMapOrder()
        st = self._store
        st.on_read = self._flush_all
        st.on_object = self._flush_object
        self.fast_stats = {"flush_all": 0, "flush_objects": 0, "rows_flushed": 0, "tubes_rebuilt": 0, "rings": 0}

    # ---- bookkeeping -------------------------------------------------------------------------------------
    def _mark(self):
        self._store._dirty = bool(self._pend or self._tdirty)

    def _pending_rows(self, oids) -> None:
        """Apply the pending work of the given objects (tube rebuild first, then the pending matrix)."""
        st = self._store
        objects = self.objects
        st._quiet += 1
        try:
            tubes = [o for o in oids if o in self._tdirty]
            if tubes:
                self._rebuild_tubes(tubes)
            groups, mats = [], []
            pend = self._pend
            for o in oids:
                P = pend.pop(o, None)
                if P is None:
                    continue
                obj = objects.get(o)
                if obj is None:
                    continue
                ix = self._alive_members(obj)
                rows = _rows_of(ix)
                if len(rows):
                    groups.append(rows)
                    mats.append(P)
            if groups:
                self._apply_matrices(groups, mats)
        finally:
            st._quiet -= 1
            self._mark()

    def _apply_matrices(self, groups, mats):
        st = self._store
        Tm = st._T
        counts = np.array([len(g) for g in groups])
        rows = np.concatenate(groups)
        Ps = np.stack(mats)                                  # (k,4,4)
        trans_only = np.all(Ps[:, :3, :3] == _I4[:3, :3], axis=(1, 2))
        self.fast_stats["rows_flushed"] += len(rows)
        owner = np.repeat(np.arange(len(groups)), counts)
        tsel = trans_only[owner]
        if tsel.any():
            r = rows[tsel]
            t = Ps[owner[tsel], :3, 3]
            Tm[r, 3:12:4] = (Tm[r, 3:12:4].astype(f64) + t).astype(f32)
        if not tsel.all():
            r_all, o_all = rows[~tsel], owner[~tsel]
            for a in range(0, len(r_all), _FLUSH_CHUNK):
                r, o = r_all[a:a + _FLUSH_CHUNK], o_all[a:a + _FLUSH_CHUNK]
                M = Tm[r].astype(f64).reshape(-1, 4, 4)
                Tm[r] = (Ps[o] @ M).astype(f32).reshape(-1, 16)

    def _flush_all(self):
        self.fast_stats["flush_all"] += 1
        self._pending_rows(sorted(self._tdirty | set(self._pend)))

    def _flush_object(self, oid: int):
        if oid in self._pend or oid in self._tdirty:
            self.fast_stats["flush_objects"] += 1
            self._pending_rows([oid])

    def _flush_objects(self, oids):
        if not self._store._dirty:
            return
        todo = [o for o in oids if o in self._pend or o in self._tdirty]
        if todo:
            self.fast_stats["flush_objects"] += 1
            self._pending_rows(todo)

    def _flush_parents(self, uuids):
        st = self._store
        if not st._dirty:
            return
        ix = np.asarray(uuids, np.int64).reshape(-1)
        ix = ix[(ix >= 0) & (ix < st.n)]
        if not len(ix):
            return
        par = st.parent[ix]
        if len(par) > 1:
            par = np.unique(par)
        self._flush_objects([int(p) for p in par.tolist() if p != 0])

    def flush(self):
        """Apply every pending transform and tube rebuild now (reads do this on their own)."""
        if self._store._dirty:
            self._flush_all()

    # ---- object transforms ------------------------------------------------------------------------------
    def _compose(self, obj, T64):
        P = self._pend.get(obj.oid)
        self._pend[obj.oid] = T64.copy() if P is None else T64 @ P

    def translate_object(self, oids, shift):
        s = V.v3(shift)
        if not s.any():
            for oid in _ids(oids):
                self._object(oid, "Context::translateObject")
            return
        s64 = s.astype(f64)
        pend = self._pend
        for oid in _ids(oids):
            obj = self._object(oid, "Context::translateObject")
            M = obj.transform
            if (M[12:16] == (0, 0, 0, 1)).all():
                M = M.copy()
                M[3:12:4] = M[3:12:4] + s
                obj.transform = M
            else:
                from .primitives import make_translation_matrix
                obj.transform = matmult(make_translation_matrix(s), M)
            P = pend.get(oid)
            if P is None:
                P = _I4.copy()
                pend[oid] = P
            P[:3, 3] += s64                                   # T(s) @ P for an affine P
        self._store._dirty = True

    def _left_multiply_lazy(self, ids, T, who):
        T = np.asarray(T, f32).reshape(16)
        T64 = T.astype(f64).reshape(4, 4)
        for oid in ids:
            obj = self._object(oid, who)
            obj.transform = matmult(T, obj.transform)
            self._compose(obj, T64)
        self._store._dirty = True

    def rotate_object(self, oids, radians, axis, origin=None):
        if isinstance(axis, str) and origin is not None:
            raise ValueError("rotateObject: an origin needs a vec3 axis")
        ids = _ids(oids)
        for oid in ids:
            self._object(oid, "Context::rotateObject")
        if f32(radians) == 0 or not ids:
            return
        if isinstance(axis, str):
            if axis not in ("x", "y", "z"):
                raise ValueError("CompoundObject::rotate: Rotation axis should be one of x, y, or z.")
            T = make_rotation_matrix(radians, axis)
        elif origin is None:
            T = make_rotation_matrix_axis(radians, axis)
        else:
            T = make_rotation_matrix_about(radians, origin, axis)
        self._left_multiply_lazy(ids, T, "Context::rotateObject")

    def scale_object_about_point(self, oids, scale, point):
        s = V.v3(scale)
        ids = _ids(oids)
        for oid in ids:
            self._object(oid, "Context::scaleObjectAboutPoint")
        if (s[0] == 1 and s[1] == 1 and s[2] == 1) or not ids:
            return
        self._left_multiply_lazy(ids, make_scale_matrix_about_point(s, point), "Context::scaleObjectAboutPoint")

    def scale_object(self, oids, scale):
        self.scale_object_about_point(oids, scale, (0.0, 0.0, 0.0))

    # ---- copies and deletions ------------------------------------------------------------------------------
    def _copy_object_one(self, oid: int) -> int:
        src = self._object(oid, "Context::copyObject")
        self._copy_carry = True
        try:
            new = super()._copy_object_one(oid)
        finally:
            self._copy_carry = False
        P = self._pend.get(oid)
        if P is not None:
            self._pend[new] = P.copy()
        if oid in self._tdirty:
            self._tdirty.add(new)
        c = self._recipe.get(oid)
        if c is not None:
            self._recipe[new] = c.copy()
            if oid in self._stale:
                self._stale.add(new)
        del src
        self._mark()
        return new

    _copy_carry = False

    def _copy_primitives(self, uuids):
        """copyPrimitive as raw row copies. A textured patch keeps its stored solid fraction (exact mode
        recomputes it as area / (size.x * size.y), which equals it up to rounding)."""
        if not uuids:
            return []
        st = self._store
        src = np.asarray(uuids, np.int64)
        ok = (src >= 0) & (src < st.n)
        if not ok.all() or not st.alive[src].all():
            return [self._copy_primitive_one(u) for u in uuids]
        if not self._copy_carry:
            self._flush_parents(src)
        new0 = self.current_uuid
        dst = st.copy_rows(src, new0)
        self.current_uuid = new0 + len(src)
        self._copy_primitive_data_rows(src.tolist(), dst.tolist())
        out = dst.tolist()
        self._after_copy_primitives(out)
        return out

    def _after_copy_primitives(self, new_uuids):
        """Scene._after_copy_primitives with the reference counts added per material (only a material with the
        texture-colour override needs the per-copy path, in order)."""
        self.primitive_order.insert_many(new_uuids)
        st, mats = self._store, self.materials
        m = st.mat[np.asarray(new_uuids, np.int64)]
        um, cnt = np.unique(m, return_counts=True)
        if any(mats.get(int(x)) is not None and mats[int(x)].texture_color_overridden for x in um.tolist()):
            for u, mid in zip(new_uuids, m.tolist()):
                mm = mats.get(mid) if mid >= 0 else None
                if mm is None:
                    continue
                if mm.texture_color_overridden:
                    self._material_reference_copied(FastPrimitive(st, u))
                else:
                    mm.reference_count += 1
            return
        for x, c in zip(um.tolist(), cnt.tolist()):
            mm = mats.get(x) if x >= 0 else None
            if mm is not None:
                mm.reference_count += c

    def _copy_primitive_data_rows(self, src, dst):
        """copyPrimitiveData for fresh rows, sharing every entry: data entries are never modified in place
        (setters replace them, getters return copies), so a shared entry behaves like a copy."""
        for d in self.primitive_data.values():
            if not d:
                continue
            get = d.get
            d.update([(t_, v) for t_, v in zip(dst, map(get, src)) if v is not None])

    def delete_object(self, oids):
        """deleteObject; the members of each object are deleted in one row operation when they are all live,
        distinct and belong to no other object (else the per-primitive path)."""
        for oid in _ids(oids):
            obj = self._object(oid, "Context::deleteObject")
            if not self._delete_members_bulk(obj):
                super().delete_object(oid)
            self._forget([oid])

    def _delete_members_bulk(self, obj) -> bool:
        st = self._store
        uu = obj.uuids
        ix = np.asarray(uu, np.int64)
        if len(ix):
            if not (((ix >= 0) & (ix < st.n)).all() and st.alive[ix].all()) or len(np.unique(ix)) != len(ix):
                return False
            par = st.parent[ix]
            if not (par == obj.oid).all():
                return False
        oid = obj.oid
        for d in self.object_data.values():
            d.pop(oid, None)
        del self.objects[oid]
        if not len(ix):
            return True
        st.alive[ix] = False
        st.count -= len(ix)
        st.deaths += 1
        mats = self.materials
        m = st.mat[ix]
        um, cnt = np.unique(m[m >= 0], return_counts=True)
        for x, c in zip(um.tolist(), cnt.tolist()):
            mm = mats.get(x)
            if mm is not None:
                mm.reference_count -= c
        for d in self.primitive_data.values():
            if d:
                pop = d.pop
                for u in uu:
                    pop(u, None)
        self.primitive_order.insert_many([~u for u in uu])
        if self.hidden_primitives:
            self.hidden_primitives.difference_update(uu)
        return True

    def _forget(self, oids):
        """Drop the deferred state of deleted objects (IDs are never reused; a stale entry would only cost a
        no-op flush)."""
        for o in oids:
            if o not in self.objects:
                self._pend.pop(o, None)
                self._recipe.pop(o, None)
                self._tdirty.discard(o)
                self._stale.discard(o)
        self._mark()

    # ---- targeted geometry reads ---------------------------------------------------------------------------
    def object_area(self, oid):
        self._flush_object(int(oid))
        with _Quiet(self._store):
            return super().object_area(oid)

    def object_bounding_box(self, oid):
        self._flush_object(int(oid))
        with _Quiet(self._store):
            return super().object_bounding_box(oid)

    def primitive_vertices_rows(self, uuids):
        uu = _ids(uuids)
        self._flush_parents(uu)
        with _Quiet(self._store):
            return super().primitive_vertices_rows(uu)

    def primitive_areas(self, uuids):
        uu = _ids(uuids)
        self._flush_parents(uu)
        with _Quiet(self._store):
            return super().primitive_areas(uu)

    def get_primitive_material_labels(self, uuids):
        with _Quiet(self._store):
            return super().get_primitive_material_labels(uuids)

    # ---- tubes ----------------------------------------------------------------------------------------------
    def _tube_ring_list(self, nr, nodes, radius):
        rc = _clamp_radius(radius)
        if len(nodes) < 2:
            return super()._tube_ring_list(nr, nodes, radius)
        tv = _rings_fast(nr, np.array(nodes, f32), np.array(rc, f32))
        self.fast_stats["rings"] += 1
        return [[tv[i, j] for j in range(nr + 1)] for i in range(len(nodes))]

    def add_tube_object(self, radial_subdivisions, nodes, radius, colors=None):
        nodes_l = [V.v3(n) for n in nodes]
        radius_l = [f32(r) for r in radius]
        n = len(nodes_l)
        same = colors is None or all(np.array_equal(np.asarray(c, f32), np.asarray(colors[0], f32)) for c in colors)
        if n < 2 or n != len(radius_l) or (colors is not None and len(colors) != n) or not same:
            return super().add_tube_object(radial_subdivisions, nodes, radius, colors)
        for r in radius_l:
            if r < 0:
                raise ValueError("addTubeObject: radius must be positive")
        nr = int(radial_subdivisions)
        col = (0.0, 0.75, 0.0) if colors is None else colors[0]
        rc = np.array(_clamp_radius(radius_l), f32)
        N = np.array(nodes_l, f32)
        tv = _rings_fast(nr, N, rc)
        self.fast_stats["rings"] += 1
        i0, i1, i2 = _tube_triangle_index(nr, n)
        A = tv.reshape(-1, 3)
        uuids = self._add_triangle_rows(_triangle_rows_fast(A[i0], A[i1], A[i2]), col)
        allc = [(0.0, 0.75, 0.0)] * n if colors is None else list(colors)
        oid = self._register_object(OBJECT_TUBE, uuids, transform=np.eye(4, dtype=f32).reshape(16),
                                    extra={"nodes": N.copy(), "radius": np.array(radius_l, f32), "subdiv": nr,
                                           "triangle_vertices": tv,
                                           "colors": np.array([np.asarray(c, f32)[:3] for c in allc], f32).reshape(-1, 3)})
        self._recipe[oid] = rc.copy()
        self.objects[oid].origin = self.object_center(oid)
        return oid

    def add_tube_object_textured(self, radial_subdivisions, nodes, radius, texture_file, textureuv_ufrac=None):
        oid = super().add_tube_object_textured(radial_subdivisions, nodes, radius, texture_file, textureuv_ufrac)
        if len(nodes) >= 2:
            self._recipe[oid] = np.array(_clamp_radius([f32(r) for r in radius]), f32)
        return oid

    def add_triangle(self, v0, v1, v2, color=None):
        """addTriangle with the transform from one float64 product and the RGBA color cached per color value."""
        key = None if color is None else tuple(float(c) for c in color)
        cache = self.__dict__.setdefault("_rgba_cache", {})
        c = cache.get(key)
        if c is None:
            from .scene import _color_rgba
            c = cache[key] = _color_rgba(color)
        V2 = np.ones((4, 4), f64)
        V2[:3, 0], V2[:3, 1], V2[:3, 2] = v0, v1, v2
        st = self._store
        u = self.current_uuid
        st.add(u, TRIANGLE, (V2 @ _INV_V1_64).astype(f32).reshape(16), c)
        self.primitive_order.insert(u)
        self._assign_automatic_material(FastPrimitive(st, u))
        self.current_uuid = u + 1
        return u

    def _add_triangle_rows(self, Trows: np.ndarray, color) -> list[int]:
        """add_triangles with ready-made transforms (same UUIDs, materials and iteration order)."""
        from .scene import _color_rgba
        n = len(Trows)
        if n == 0:
            return []
        c = _color_rgba(color)
        u0 = self.current_uuid
        st = self._store
        rows = st.add_rows(u0, TRIANGLE, Trows, c)
        self.primitive_order.insert_many(range(u0, u0 + n))
        self._assign_automatic_material(FastPrimitive(st, u0))
        m = int(st.mat[u0])
        st.mat[rows[1:]] = m
        self.materials[m].reference_count += n - 1
        self.current_uuid = u0 + n
        return rows.tolist()

    def _tube_valid(self, obj) -> bool:
        """The batched rebuild applies (every member a live triangle, one per ring triangle)."""
        nr, n = int(obj.extra["subdiv"]), len(obj.extra["nodes"])
        m = 2 * nr * max(n - 1, 0)
        uu = obj.uuids
        if m == 0 or m != len(uu):
            return False
        st = self._store
        ix = obj.member_index()
        if isinstance(ix, slice):
            if ix.stop > st.n:
                return False
        elif not ((ix >= 0) & (ix < st.n)).all():
            return False
        return bool(st.alive[ix].all() and (st.ptype[ix] == TRIANGLE).all())

    def _materialize(self, obj):
        """Bring the stored ring vertices up to date with the recipe."""
        oid = obj.oid
        if oid in self._stale:
            self._stale.discard(oid)
            obj.extra["triangle_vertices"] = _rings_fast(int(obj.extra["subdiv"]), obj.extra["nodes"], self._recipe[oid])
            self.fast_stats["rings"] += 1

    def _rebuild_tubes(self, oids):
        """Rebuild the member triangles of dirty tubes from their rings, as one batch."""
        parts, rowsets = [], []
        for oid in oids:
            self._tdirty.discard(oid)
            obj = self.objects.get(oid)
            if obj is None:
                continue
            self._materialize(obj)
            nr, n = int(obj.extra["subdiv"]), len(obj.extra["nodes"])
            if not self._tube_valid(obj):
                with _Quiet(self._store):
                    super()._update_tube_triangles(obj)
                continue
            i0, i1, i2 = _tube_triangle_index(nr, n)
            A = np.asarray(obj.extra["triangle_vertices"], f32).reshape(-1, 3)
            parts.append((A[i0], A[i1], A[i2]))
            rowsets.append(_rows_of(obj.member_index()))
        if parts:
            v0 = np.concatenate([p[0] for p in parts])
            v1 = np.concatenate([p[1] for p in parts])
            v2 = np.concatenate([p[2] for p in parts])
            self._store._T[np.concatenate(rowsets)] = _triangle_rows_fast(v0, v1, v2)
            self.fast_stats["tubes_rebuilt"] += len(parts)

    def _update_tube_triangles(self, obj):
        """Deferred: the member triangles are rebuilt on the next read (they then replace the member rows,
        so a pending matrix of the object is void)."""
        if not self._tube_valid(obj):
            self._flush_object(obj.oid)
            self._materialize(obj)
            with _Quiet(self._store):
                return super()._update_tube_triangles(obj)
        self._pend.pop(obj.oid, None)
        self._tdirty.add(obj.oid)
        self._store._dirty = True

    def _bake_tube_transform(self, obj):
        """Tube::bakeTransform. Nodes and radii are baked with exact mode's float32 operations (the plant code
        reads them back); the ring vertices are only moved when they are current."""
        T = obj.transform
        if np.array_equal(T, np.eye(4, dtype=f32).reshape(16)):
            return
        lin = T.reshape(4, 4)[:3, :3]
        if obj.oid in self._recipe and np.array_equal(lin, np.eye(3, dtype=f32)) and (T[12:16] == (0, 0, 0, 1)).all():
            # a pure translation keeps the rings frame rings
            if obj.oid not in self._stale:
                obj.extra["triangle_vertices"] = _vecmult_rows(T, np.asarray(obj.extra["triangle_vertices"], f32))
        else:
            self._materialize(obj)
            self._recipe.pop(obj.oid, None)
            obj.extra["triangle_vertices"] = _vecmult_rows(T, np.asarray(obj.extra["triangle_vertices"], f32))
        N = np.asarray(obj.extra["nodes"], f32).reshape(-1, 3)
        Nw = np.empty_like(N)
        for r in range(3):
            Nw[:, r] = ((T[4 * r] * N[:, 0] + T[4 * r + 1] * N[:, 1]) + T[4 * r + 2] * N[:, 2]) + T[4 * r + 3]
        rad = np.asarray(obj.extra["radius"], f32).reshape(-1)
        zero = f32(0)
        d = np.empty((len(rad), 3), f32)
        for r in range(3):
            n0 = ((T[4 * r] * zero + T[4 * r + 1] * zero) + T[4 * r + 2] * zero) + T[4 * r + 3]
            d[:, r] = (((T[4 * r] * rad + T[4 * r + 1] * zero) + T[4 * r + 2] * zero) + T[4 * r + 3]) - n0
        obj.extra["nodes"] = Nw
        obj.extra["radius"] = np.sqrt((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]).astype(f32)
        obj.transform = np.eye(4, dtype=f32).reshape(16)

    def set_tube_nodes(self, oid, node_xyz):
        obj = self._tube(oid, "setTubeNodes")
        nodes = np.array([V.v3(p) for p in node_xyz], f32).reshape(-1, 3)
        if len(nodes) != len(obj.extra["nodes"]):
            raise ValueError("ERROR (Tube::setTubeNodes): Number of nodes in input vector must match number of tube nodes.")
        if len(nodes) < 2:
            self._materialize(obj)
            self._recipe.pop(obj.oid, None)
            return super().set_tube_nodes(oid, node_xyz)
        self._bake_tube_transform(obj)
        obj.extra["nodes"] = nodes
        self._recipe[obj.oid] = np.array(_clamp_radius(obj.extra["radius"]), f32)
        self._stale.add(obj.oid)
        self._update_tube_triangles(obj)

    def set_tube_radii(self, oid, node_radii):
        obj = self._tube(oid, "setTubeRadii")
        radii = np.array([f32(r) for r in node_radii], f32)
        if len(radii) != len(obj.extra["nodes"]):
            raise ValueError("ERROR (Tube::setTubeRadii): Number of radii in input vector must match number of tube nodes.")
        self._bake_tube_transform(obj)
        c = self._recipe.get(obj.oid)
        if c is not None:
            # node + normalize(ring - node) * r: the ring offset direction is the frame one, times sign(c)
            self._recipe[obj.oid] = (np.sign(c) * radii).astype(f32)
            self._stale.add(obj.oid)
            obj.extra["radius"] = radii
            self._update_tube_triangles(obj)
            return
        super().set_tube_radii(oid, node_radii)

    def _append_rings(self, obj, node, radius, color):
        c = self._recipe.get(obj.oid)
        if c is None:
            return super()._append_rings(obj, node, radius, color)
        r = f32(radius)
        if r < 0:
            raise ValueError("ERROR (Tube::appendTubeSegment): Node radius must be positive.")
        r = r if f32(1e-5) < r else f32(1e-5)
        self._bake_tube_transform(obj)
        if obj.oid not in self._recipe:                     # the bake rotated the rings
            return super()._append_rings(obj, node, radius, color)
        nr = int(obj.extra["subdiv"])
        rad = np.append(np.asarray(obj.extra["radius"], f32), r).astype(f32)
        if (rad < 0).any():
            raise ValueError("ERROR (Context::addTubeObject): Radius of tube must be positive.")
        nodes = np.vstack([np.asarray(obj.extra["nodes"], f32).reshape(-1, 3), V.v3(node)]).astype(f32)
        obj.extra["colors"] = np.vstack([obj.extra["colors"].reshape(-1, 3), np.asarray(color, f32)[:3]]).astype(f32)
        n = len(nodes)
        c = np.append(self._recipe[obj.oid], r).astype(f32)
        if n >= 2:
            c[n - 2] = rad[n - 2]                           # the last two rings are recomputed with the stored radii
        self._recipe[obj.oid] = c
        obj.extra["nodes"], obj.extra["radius"] = nodes, rad
        # the ring vertices are only needed by the new triangles' creation, which the rebuild overwrites
        obj.extra["triangle_vertices"] = np.zeros((n, nr + 1, 3), f32)
        self._stale.add(obj.oid)
        return n, nr

    def append_tube_segment(self, oid, node_position, node_radius, node_color):
        obj = self._tube(oid, "appendTubeSegment")
        if obj.oid not in self._recipe:
            return super().append_tube_segment(oid, node_position, node_radius, node_color)
        n, nr = self._append_rings(obj, node_position, node_radius, node_color)
        col = tuple(np.asarray(node_color, f32)[:3])
        if obj.oid not in self._recipe:
            # baked a rotation: exact-mode triangles from the stored rings
            tv = obj.extra["triangle_vertices"]

            def pair(j):
                return [self.add_triangle(tv[n - 2][j], tv[n - 1][j + 1], tv[n - 2][j + 1], col),
                        self.add_triangle(tv[n - 2][j], tv[n - 1][j], tv[n - 1][j + 1], col)]
            earlier = self._append_reorder(obj, n, nr, pair)
        else:
            new = iter(self._add_triangle_rows(np.tile(np.eye(4, dtype=f32).reshape(1, 16), (2 * nr, 1)), col))
            earlier = self._append_reorder(obj, n, nr, lambda j: [next(new), next(new)])
        self._keep_named_tube_material(earlier, obj.uuids, "", col)
        self._update_tube_triangles(obj)

    def append_tube_segment_textured(self, oid, node_position, node_radius, texture_file, textureuv_ufrac):
        obj = self._tube(oid, "appendTubeSegment")
        self._materialize(obj)
        return super().append_tube_segment_textured(oid, node_position, node_radius, texture_file, textureuv_ufrac)

    def _append_reorder(self, obj, n, nr, make_pair):
        earlier = list(obj.uuids)
        out, k = [], 0
        for j in range(nr):
            for _ in range(n - 2):
                out += [earlier[k], earlier[k + 1]]
                k += 2
            out += make_pair(j)
        obj.uuids = out
        self._store.parent[np.asarray(out, np.int64)] = obj.oid
        return earlier

    def scale_tube_girth(self, oid, S):
        obj = self._tube(oid, "scaleTubeGirth")
        self._materialize(obj)
        self._recipe.pop(obj.oid, None)
        return super().scale_tube_girth(oid, S)

    def scale_tube_length(self, oid, S):
        obj = self._tube(oid, "scaleTubeLength")
        self._materialize(obj)
        self._recipe.pop(obj.oid, None)
        return super().scale_tube_length(oid, S)

    def prune_tube_nodes(self, oid, node_index):
        obj = self._tube(oid, "pruneTubeNodes")
        self._flush_object(obj.oid)
        self._materialize(obj)
        self._recipe.pop(obj.oid, None)
        with _Quiet(self._store):
            out = super().prune_tube_nodes(oid, node_index)
        self._forget([obj.oid])
        return out

    def tube_object_triangle_vertices(self, oid):
        obj = self._tube(oid, "getTubeObjectTriangleVertices")
        self._materialize(obj)
        return super().tube_object_triangle_vertices(oid)


class FastScene(FastSceneMixin, Scene):
    """``Scene(mode="fast")``."""


__all__ = ["FastScene", "FastPrimitiveStore", "FastPrimitive"]
