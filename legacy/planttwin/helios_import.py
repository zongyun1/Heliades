"""Helios (PyHelios plant architecture) -> PlantAsset.

Helios grows botanically structured plants (shoots -> phytomers -> petioles -> leaflets,
peduncles -> fruit) for ~30 species. It gives geometry and topology only; this module turns
that into a simulation-ready PlantAsset:

  * shoots, petioles, peduncles -> one rooted stem graph (rod centrelines + radii)
  * each leaflet base is inserted as a node on its petiole/rachis, so leaves weld to nodes
  * leaflet render meshes (~144 tris) -> coarse simulation meshes by clustering in texture UV
  * fruit -> pedicel segment ending in a node that carries the fruit mass

Requires `pip install pyhelios3d`.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np

from .asset import Fruit, Leaf, PhysicsParams, PlantAsset, StemGraph

STEM, PETIOLE, PEDUNCLE, PEDICEL = 0, 1, 2, 3


@contextmanager
def _quiet():
    """Helios prints progress bars to the C-level stdout; silence them."""
    sys.stdout.flush()
    fd, saved = sys.stdout.fileno(), os.dup(sys.stdout.fileno())
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), fd)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved, fd)
            os.close(saved)


def _v(p):
    return np.array([p.x, p.y, p.z], float) if hasattr(p, "x") else np.asarray(p, float)


class _Graph:
    def __init__(self):
        self.nodes, self.radius, self.organ, self.edges = [], [], [], []

    def add(self, pos, r, organ, parent=None):
        self.nodes.append(np.asarray(pos, float))
        self.radius.append(float(r))
        self.organ.append(organ)
        i = len(self.nodes) - 1
        if parent is not None:
            self.edges.append((parent, i))
        return i

    def chain(self, start, pts, radii, organ, min_len=5e-3):
        """Append a polyline hanging from node `start`; drops points closer than min_len.
        Returns the node index for every input point (dropped ones map to their predecessor)."""
        ids, prev = [], start
        for p, r in zip(pts, radii):
            if np.linalg.norm(p - self.nodes[prev]) < min_len:
                ids.append(prev)
                continue
            prev = self.add(p, r, organ, prev)
            ids.append(prev)
        return ids

    def parent_of(self, i):
        for a, b in self.edges:
            if b == i:
                return a
        return None

    def nearest(self, p, among):
        among = list(among)
        d = np.linalg.norm(np.array([self.nodes[i] for i in among]) - p, axis=1)
        return among[int(np.argmin(d))]


def _insert_points(poly, pts, snap=8e-3):
    """Insert points (projected onto polyline `poly`) into it, ordered by arclength.
    Points within `snap` (arclength) of an existing vertex are snapped to it instead, so no
    very short segments are created: a short, light segment carrying a heavy leaflet pair
    has tiny rotational inertia and makes the rod/attachment constraints fight (creep).
    Returns new polyline and, for each inserted point, its index in the new polyline."""
    seg = np.diff(poly, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    items = [(cum[k], poly[k], None) for k in range(len(poly))]
    for j, p in enumerate(pts):
        best = (np.inf, 0, 0.0)
        for k in range(len(seg)):
            t = np.clip((p - poly[k]) @ seg[k] / max(seglen[k] ** 2, 1e-18), 0, 1)
            d = np.linalg.norm(poly[k] + t * seg[k] - p)
            if d < best[0]:
                best = (d, k, t)
        _, k, t = best
        items.append((cum[k] + t * seglen[k], poly[k] + t * seg[k], j))
    items.sort(key=lambda it: (it[0], it[2] is not None))
    # snap inserted points that are close (in arclength) to an original vertex
    snapped = []
    for s, p, j in items:
        if j is not None:
            k = int(np.argmin(np.abs(cum - s)))
            if abs(cum[k] - s) < snap:
                s, p = cum[k], poly[k]
        snapped.append((s, p, j))
    items = sorted(snapped, key=lambda it: (it[0], it[2] is not None))
    out, where = [], {}
    for s, p, j in items:
        if out and abs(s - out[-1][0]) < snap and j is not None:   # merge with previous node
            where[j] = len(out) - 1
            continue
        if out and np.linalg.norm(p - out[-1][1]) < 1e-3:   # merge near-duplicates
            if j is not None:
                where[j] = len(out) - 1
            continue
        out.append((s, p))
        if j is not None:
            where[j] = len(out) - 1
    return np.array([p for _, p in out]), [where[j] for j in range(len(pts))]


def coarsen_leaf(V, F, UV, base, width_cells=4, max_cells=30):
    """Vertex-clustering decimation on a grid in the leaf's texture (u,v) space -- the leaf's
    own 2D parameterisation, robust to curled / arching blades (maize, wheat).
    Cell counts follow the 3D extent along each texture axis (~square cells, `width_cells`
    across the narrow direction). Returns (verts, faces, base_vertex_index)."""
    # 3D length per unit u and v, from mesh edges
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    dX = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)
    dU = np.abs(UV[E[:, 1]] - UV[E[:, 0]])
    scale = np.array([np.median(dX[dU[:, k] > 1e-6] / np.linalg.norm(dU[dU[:, k] > 1e-6], axis=1))
                      if (dU[:, k] > 1e-6).any() else 1.0 for k in range(2)])
    span = np.ptp(UV, axis=0) * scale                    # metres along u, v
    cell = max(span.min() / width_cells, 1e-4)
    n = np.clip(np.round(span / cell).astype(int), 1, max_cells)
    lo = UV.min(0)
    g = np.clip(((UV - lo) / (np.ptp(UV, 0) + 1e-12) * n).astype(int), 0, n - 1)
    key = g[:, 0] * n[1] + g[:, 1]
    uniq, lab = np.unique(key, return_inverse=True)
    newV = np.array([V[lab == k].mean(0) for k in range(len(uniq))])
    newF = lab[F]
    newF = newF[(newF[:, 0] != newF[:, 1]) & (newF[:, 1] != newF[:, 2]) & (newF[:, 0] != newF[:, 2])]
    if len(newF) == 0:
        raise ValueError("leaf collapsed during coarsening")
    newF = np.unique(np.sort(newF, 1), axis=0)
    used = np.unique(newF)
    remap = -np.ones(len(newV), int)
    remap[used] = np.arange(len(used))
    newV, newF = newV[used], remap[newF]
    cg = np.stack([uniq // n[1], uniq % n[1]], 1)[used]   # cluster grid coords
    # base: fan to the grid row (along the long axis) nearest the attachment point
    long_ax = int(np.argmax(span))
    near = int(np.argmin(np.linalg.norm(newV - base, axis=1)))
    first = np.flatnonzero(cg[:, long_ax] == cg[near, long_ax])
    first = first[np.argsort(cg[first, 1 - long_ax])]
    if len(first) < 2:
        first = np.argsort(np.linalg.norm(newV - base, axis=1))[:2]
    newV = np.vstack([newV, base])
    bi = len(newV) - 1
    newF = np.vstack([newF, [[bi, first[k], first[k + 1]] for k in range(len(first) - 1)]])
    return newV, newF, bi


def _welded_mesh(ctx, obj, with_uv=False):
    """Object primitives (triangles and 4-vertex patches) -> welded triangle mesh (+ UV)."""
    pts, uvs = [], []
    for u in ctx.getObjectPrimitiveUUIDs([obj]):
        P = [_v(p) for p in ctx.getPrimitiveVertices(u)]
        T = [(t.x, t.y) for t in ctx.getPrimitiveTextureUV(u)] if with_uv else [(0.0, 0.0)] * len(P)
        if len(T) != len(P):
            T = [(0.0, 0.0)] * len(P)
        for tri in ([0, 1, 2],) if len(P) == 3 else ([0, 1, 2], [0, 2, 3]):
            pts += [P[k] for k in tri]
            uvs += [T[k] for k in tri]
    flat = np.array(pts)
    _, first, inv = np.unique(np.round(flat / 1e-5).astype(np.int64), axis=0,
                              return_index=True, return_inverse=True)
    V, F = flat[first], inv.ravel().reshape(-1, 3)
    return (V, F, np.array(uvs)[first]) if with_uv else (V, F)


def build_helios_plant(species="tomato", age=75.0, seed=0, leaf_width_cells=4,
                       fruit_density=1000.0, physics: PhysicsParams | None = None):
    """Grow a plant in Helios and convert it. Returns (PlantAsset, render_meshes).
    render_meshes: dict organ -> list of (V, F) full-resolution meshes (for visualisation)."""
    from pyhelios import Context, DataTypes, PlantArchitecture

    ctx = Context()
    ctx.seedRandomGenerator(seed)
    pa = PlantArchitecture(ctx)
    with _quiet():
        pa.loadPlantModelFromLibrary(species)
        pid = pa.buildPlantInstanceFromLibrary(DataTypes.vec3(0, 0, 0), float(age))

    G = _Graph()
    shoot_nodes = {}                                    # shoot id -> node ids per internode vertex
    shoots = sorted(pa.getAllShootIDs(pid), key=lambda s: pa.getShootDepth(pid, s)
                    if hasattr(pa, "getShootDepth") else s)
    for s in shoots:
        V = np.array([_v(p) for p in pa.getShootInternodeVertices(pid, s)])
        R = np.array(pa.getShootInternodeRadii(pid, s))
        info = pa.getShoot(pid, s)
        if info["parent_shoot_id"] < 0:
            root = G.add(V[0], R[0], STEM)
            shoot_nodes[s] = [root] + G.chain(root, V[1:], R[1:], STEM)
        else:
            parent = G.nearest(V[0], shoot_nodes[info["parent_shoot_id"]])
            shoot_nodes[s] = [parent] + G.chain(parent, V[1:], R[1:], STEM)
    stem_nodes = [i for ids in shoot_nodes.values() for i in ids]

    leaves, render = [], {"leaf": [], "fruit": []}
    for s in shoots:
        for node in range(pa.getShootNodeCount(pid, s)):
            for pi, leaf_objs in enumerate(pa.getPhytomerLeafObjectIDs(pid, s, node)):
                P = np.array([_v(p) for p in pa.getPetioleVertices(pid, s, node, pi)])
                PR = np.array(pa.getPetioleRadii(pid, s, node, pi))
                bases = np.array([_v(pa.getLeafBasePosition(pid, s, node, pi, k))
                                  for k in range(len(leaf_objs))])
                poly, where = _insert_points(P, bases)
                # radius along the new polyline: nearest original petiole vertex
                rr = np.array([PR[np.argmin(np.linalg.norm(P - q, axis=1))] for q in poly])
                start = G.nearest(poly[0], stem_nodes)
                ids = G.chain(start, poly[1:], rr[1:], PETIOLE)
                ids = [start] + ids
                for k, obj in enumerate(leaf_objs):
                    Vf, Ff, UVf = _welded_mesh(ctx, obj, with_uv=True)
                    render["leaf"].append((Vf, Ff))
                    node_id = ids[where[k]]
                    if G.parent_of(node_id) is None:  # sessile leaf on the root node: no frame
                        continue
                    Vc, Fc, bi = coarsen_leaf(Vf, Ff, UVf, G.nodes[node_id], leaf_width_cells)
                    tip = int(np.argmax(np.linalg.norm(Vc - Vc[bi], axis=1)))
                    leaves.append(Leaf(Vc, Fc, attach_node=node_id, attach_vertex=bi,
                                       keypoints={"tip": tip}))

    fruits = []
    fruit_objs = pa.getPlantFruitObjectIDs(pid)
    ped_objs = pa.getPlantPeduncleObjectIDs(pid)
    ped_nodes = []
    for o in ped_objs:
        P = np.array([_v(p) for p in ctx.getTubeObjectNodes(o)])
        R = np.array(ctx.getTubeObjectNodeRadii(o))
        start = G.nearest(P[0], stem_nodes)
        ped_nodes += G.chain(start, P[1:], R[1:], PEDUNCLE)
    point_mass = {}
    for o in fruit_objs:
        Vf, Ff = _welded_mesh(ctx, o)
        render["fruit"].append((Vf, Ff))
        c = Vf.mean(0)
        radius = float(np.linalg.norm(Vf - c, axis=1).mean())
        mass = fruit_density * 4 / 3 * np.pi * radius**3
        anchor = G.nearest(c, ped_nodes or stem_nodes)
        # pedicel: from the truss to the fruit centre (fruit hangs as a lumped mass)
        n = G.add(c, 0.0012, PEDICEL, anchor)
        point_mass[n] = mass
        fruits.append(Fruit(node=n, radius=radius, mass=mass))

    N = len(G.nodes)
    pm = np.zeros(N)
    for k, m in point_mass.items():
        pm[k] = m
    fixed = np.zeros(N, bool)
    root_chain = shoot_nodes[shoots[0]]
    fixed[root_chain[:2]] = True                        # clamp the base in the soil/pot
    stem = StemGraph(np.array(G.nodes), np.array(G.edges, int), np.array(G.radius), fixed,
                     point_mass=pm, organ=np.array(G.organ))
    asset = PlantAsset(stem, leaves, physics or PhysicsParams(), fruits=fruits,
                       meta={"source": "helios", "species": species, "age_days": age, "seed": seed,
                             "helios_leaflets": len(render["leaf"])})
    return asset, render
