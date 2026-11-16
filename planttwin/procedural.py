"""Procedural potted plant: a stand-in for the reconstruction front end.

Gives a ground-truth asset with known topology so the sim / Real2Sim back end can be
developed and validated before the 3DGS -> structure step exists.
"""
from __future__ import annotations

import numpy as np

from .asset import Leaf, PhysicsParams, PlantAsset, StemGraph

UP = np.array([0.0, 0.0, 1.0])


def _unit(v):
    return v / np.linalg.norm(v)


def leaf_mesh(base, direction, length, width, rows=5, cols=5, droop=0.15):
    """Elliptic leaf blade: base vertex, `rows` x `cols` grid, tip vertex."""
    d = _unit(direction)
    s = np.cross(d, UP)
    s = _unit(s) if np.linalg.norm(s) > 1e-6 else np.array([1.0, 0, 0])
    verts = [base]
    us = (np.arange(rows) + 1) / (rows + 1)
    vs = np.linspace(-1, 1, cols)
    for u in us:
        half_w = 0.5 * width * np.sin(np.pi * u) ** 0.8
        for v in vs:
            # blade curls down a little along the midrib and at the margins
            sag = droop * length * (u**2 + 0.3 * v**2 * u)
            verts.append(base + d * length * u + s * half_w * v - UP * sag)
    verts.append(base + d * length - UP * droop * length)
    verts = np.array(verts)

    idx = lambda r, c: 1 + r * cols + c
    tip = len(verts) - 1
    faces = [(0, idx(0, c), idx(0, c + 1)) for c in range(cols - 1)]
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b, cc, dd = idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)
            faces += [(a, cc, b), (b, cc, dd)]
    faces += [(idx(rows - 1, c), tip, idx(rows - 1, c + 1)) for c in range(cols - 1)]
    keypoints = {"tip": tip, "mid": idx(rows // 2, cols // 2)}
    return verts, np.array(faces), keypoints


def make_plant(
    n_leaves=5,
    height=0.25,
    stem_segments=10,
    petiole_length=0.05,
    petiole_segments=4,
    leaf_length=0.07,
    leaf_width=0.035,
    seed=0,
    physics: PhysicsParams | None = None,
) -> PlantAsset:
    rng = np.random.default_rng(seed)
    nodes, radius, edges = [], [], []

    # main stem: slightly bent vertical rod
    lean = rng.normal(0, 0.03, size=2)
    for i in range(stem_segments + 1):
        t = i / stem_segments
        nodes.append([lean[0] * t**2 * height, lean[1] * t**2 * height, t * height])
        radius.append(0.003 * (1 - 0.4 * t))
        if i:
            edges.append((i - 1, i))
    fixed = [True, True] + [False] * (len(nodes) - 2)  # clamped in the pot

    leaves = []
    golden = np.deg2rad(137.5)
    attach_on_stem = np.linspace(stem_segments * 0.4, stem_segments, n_leaves).round().astype(int)
    for k, sn in enumerate(attach_on_stem):
        az = k * golden + rng.normal(0, 0.1)
        out = np.array([np.cos(az), np.sin(az), 0.0])
        elev = np.deg2rad(45 + rng.normal(0, 5))
        prev = sn
        # petiole: arc that starts rising at `elev` and bends towards horizontal
        for j in range(1, petiole_segments + 1):
            t = j / petiole_segments
            ang = elev * (1 - 0.8 * t)
            dirn = np.cos(ang) * out + np.sin(ang) * UP
            p = np.array(nodes[prev]) + dirn * petiole_length / petiole_segments
            nodes.append(p.tolist())
            radius.append(0.0015)
            fixed.append(False)
            edges.append((prev, len(nodes) - 1))
            prev = len(nodes) - 1
        tip_dir = _unit(np.array(nodes[prev]) - np.array(nodes[edges[-1][0]]))
        leaf_dir = _unit(tip_dir + 0.5 * out - 0.4 * UP)
        V, F, kp = leaf_mesh(np.array(nodes[prev]), leaf_dir, leaf_length, leaf_width)
        leaves.append(Leaf(V, F, attach_node=prev, attach_vertex=0, keypoints=kp))

    stem = StemGraph(
        nodes=np.array(nodes), edges=np.array(edges), radius=np.array(radius), fixed=np.array(fixed)
    )
    return PlantAsset(stem, leaves, physics or PhysicsParams(), meta={"source": "procedural", "seed": seed})
