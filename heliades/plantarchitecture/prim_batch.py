# SPDX-License-Identifier: LGPL-2.1-or-later
"""Batched primitive areas and normals (W5-D), bit-identical to ``Primitive.area()`` / ``Primitive.normal()``.

The season bookkeeping (``sumPlantLeafArea``, ``captureSeasonPeakLeaves``) and leaf-angle tracking read the area
and normal of every leaf facet on every growth sub-step. One ``Primitive.vertices()`` call costs ~20 us, which
made those reads the dominant cost of growing a tree. Here the vertices of many primitives are computed in one
vectorised pass with the same float32 operations in the same order (``T[r,0]*y.x + T[r,1]*y.y + T[r,2]*y.z +
T[r,3]``, then the cross product, magnitude and ``1/sqrtf`` normalisation), so every value equals the scalar one.
Since P1 the vertex / area / normal kernels are the core's (``heliades.core.store``); this module keeps the
grouping helpers.
Sums over the primitives of one object are done column by column over a zero-padded table, which keeps the C++
left-to-right accumulation order (adding 0 to a partial sum is exact).
"""
from __future__ import annotations

import numpy as np

from ..core.primitives import PATCH, TRIANGLE

f32 = np.float32

def areas_normals(ctx, uuids, want_normals=True):
    """(areas float32[N], normals float32[N, 3] or None) of the given primitives: the core's batched row kernels
    (``PrimitiveStore.areas_rows`` / ``normals_rows``, the same op sequence as ``Primitive.area()`` / ``normal()``)."""
    ix = np.asarray(list(uuids), np.int64)
    if not len(ix):
        return np.empty(0, f32), (np.empty((0, 3), f32) if want_normals else None)
    V, _ = ctx.primitive_vertices_rows(ix)          # raises on a UUID that does not exist
    st = ctx._store
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        areas = st.areas_rows(ix, V)
        normals = st.normals_rows(ix, V) if want_normals else None
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
