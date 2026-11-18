# SPDX-License-Identifier: LGPL-2.1-or-later
"""Batched primitive areas and normals (W5-D), bit-identical to ``Primitive.area()`` / ``Primitive.normal()``.

The season bookkeeping (``sumPlantLeafArea``, ``captureSeasonPeakLeaves``) and leaf-angle tracking read the area
and normal of every leaf facet on every growth sub-step. One ``Primitive.vertices()`` call costs ~20 us, which
made those reads the dominant cost of growing a tree. Here the vertices of many primitives are computed in one
vectorised pass with the same float32 operations in the same order (``T[r,0]*y.x + T[r,1]*y.y + T[r,2]*y.z +
T[r,3]``, then the cross product, magnitude and ``1/sqrtf`` normalisation), so every value equals the scalar one.
Sums over the primitives of one object are done column by column over a zero-padded table, which keeps the C++
left-to-right accumulation order (adding 0 to a partial sum is exact).
"""
from __future__ import annotations

import numpy as np

from ..core.primitives import PATCH, TRIANGLE

f32 = np.float32

_Y = {TRIANGLE: np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], f32)}


def _vertices(T, Y):
    """T: (N, 16) float32; Y: (k, 3) -> (N, k, 3), as ``primitives._apply`` per primitive."""
    T = T.reshape(-1, 4, 4)
    out = np.empty((T.shape[0], Y.shape[0], 3), f32)
    for r in range(3):
        out[:, :, r] = ((T[:, r, 0:1] * Y[None, :, 0] + T[:, r, 1:2] * Y[None, :, 1]) + T[:, r, 2:3] * Y[None, :, 2]) \
            + T[:, r, 3:4]
    return out


def _cross(a, b):
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], axis=1)


def _m2(v):
    return (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1]) + v[:, 2] * v[:, 2]


def areas_normals(ctx, uuids, want_normals=True):
    """(areas float32[N], normals float32[N, 3] or None) of the given primitives."""
    n = len(uuids)
    areas = np.empty(n, f32)
    normals = np.empty((n, 3), f32) if want_normals else None
    prims = [ctx.primitives[u] for u in uuids]
    tri = [i for i, p in enumerate(prims) if p.type == TRIANGLE]
    if len(tri) != n:
        for i, p in enumerate(prims):
            if p.type != TRIANGLE:
                areas[i] = p.area()
                if want_normals:
                    normals[i] = p.normal()
    if tri:
        T = np.stack([prims[i].transform for i in tri]).astype(f32, copy=False)
        sf = np.array([prims[i].solid_fraction for i in tri], f32)
        v = _vertices(T, _Y[TRIANGLE])
        v0, v1, v2 = v[:, 0], v[:, 1], v[:, 2]
        c = _cross(v1 - v0, v2 - v0)
        with np.errstate(invalid="ignore", over="ignore"):
            a = (f32(0.5) * np.sqrt(_m2(c))) * sf
        areas[tri] = a
        if want_normals:
            nn = _cross(v1 - v0, v2 - v1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                inv = f32(1.0) / np.sqrt(_m2(nn))
                normals[tri] = nn * inv[:, None]
    return areas, normals


def padded(groups, values, dtype=f32):
    """Zero-padded (len(groups), max group size) table of ``values`` split into consecutive groups of the given
    sizes (column j holds every group's j-th value)."""
    sizes = np.asarray(groups, np.int64)
    width = int(sizes.max()) if len(sizes) else 0
    table = np.zeros((len(sizes), width), dtype)
    if len(sizes):
        mask = np.arange(width)[None, :] < sizes[:, None]
        table[mask] = np.asarray(values, dtype)
    return table


def sequential_row_sums(table, start=None):
    """Row sums accumulated left to right in the table's dtype (``acc = acc + x`` per column)."""
    acc = np.zeros(table.shape[0], table.dtype) if start is None else np.asarray(start, table.dtype).copy()
    for j in range(table.shape[1]):
        acc = acc + table[:, j]
    return acc


__all__ = ["areas_normals", "padded", "sequential_row_sums", "PATCH", "TRIANGLE"]
