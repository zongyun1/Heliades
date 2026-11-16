"""B3: asset segment refinement (subdivide rod edges).

The tree solver uses one rigid segment per asset edge, so its discretisation error is
O(1/n^2) in the number of segments per organ (B1 tests 1-3). Helios petioles and
petiolules come with 1-2 edges per organ, which makes the inverse rest angles per
joint large (~1 rad) and the critical modulus discretisation dependent (B2 sec. 7).

``refine_asset`` subdivides every rod edge into m equal pieces, with

    m = max(ceil(l / max_length), ceil(min_per_organ / n_edges_of_its_organ_run))

Invariants (what makes this safe for the rest of heliades.sim):
  * the original N nodes keep their indices and positions; new interior nodes are
    appended after them (index >= N). So leaf ``attach_node``, ``Fruit.node``,
    ``point_mass``, ``fixed`` and any node index the caller holds stay valid, and the
    leaves and fruit stay attached at the same physical node;
  * new nodes lie on the straight edge (the polyline geometry is unchanged), with the
    radius interpolated linearly, organ copied from the edge's child node, and
    ``fixed`` only if both edge ends are fixed (a clamped base stays clamped);
  * point masses stay on their original node (no mass is moved).

Refinement of a straight edge adds joints whose rest relative rotation is the identity;
the total mass converges at O(1/m^2) for tapered edges (radius per piece = mean of its end
radii), exactly for constant radius.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from planttwin.asset import PlantAsset, StemGraph


@dataclass
class RefineMap:
    """Bookkeeping of a refinement. pieces[e] = number of pieces of original edge e;
    edge_of_new[e'] = original edge of refined edge e'; n_orig = N of the original asset."""
    pieces: np.ndarray
    edge_of_new: np.ndarray
    n_orig: int


def organ_runs(stem: StemGraph) -> np.ndarray:
    """Run id per edge: maximal chains of edges with the same organ between branch points
    (a node with != 1 child, or an organ change, ends a run). Returns (E,) int."""
    E = np.asarray(stem.edges, int).reshape(-1, 2)
    N = len(stem.nodes)
    org = np.zeros(N, int) if stem.organ is None else np.asarray(stem.organ, int)
    nchild = np.bincount(E[:, 0], minlength=N)
    edge_of_child = -np.ones(N, int)
    edge_of_child[E[:, 1]] = np.arange(len(E))
    run = -np.ones(len(E), int)
    # process edges root-to-tip (parents before children)
    depth = np.zeros(N, int)
    par = stem.parent()
    order = []
    for i in range(N):
        k, d = i, 0
        while par[k] >= 0:
            k, d = par[k], d + 1
        depth[i] = d
    order = np.argsort(depth[E[:, 1]], kind="stable")
    nrun = 0
    for e in order:
        a, b = E[e]
        pe = edge_of_child[a]
        if pe >= 0 and nchild[a] == 1 and org[a] == org[b] and run[pe] >= 0:
            run[e] = run[pe]
        else:
            run[e] = nrun
            nrun += 1
    return run


def refine_asset(asset: PlantAsset, max_length: float | None = None, min_per_organ: int = 1,
                 factor: int = 1, return_map: bool = False):
    """Copy of `asset` with rod edges subdivided.

    max_length:    target maximum segment length [m] (None: no length criterion)
    min_per_organ: minimum number of segments per organ run (chain of same-organ edges
                   between branch points, see ``organ_runs``); the pieces are spread evenly
                   over the run's edges
    factor:        additionally split every edge into at least `factor` pieces
    Original nodes keep their indices; new nodes are appended (module docstring)."""
    st = asset.stem
    X = np.asarray(st.nodes, float)
    E = np.asarray(st.edges, int).reshape(-1, 2)
    N = len(X)
    l = np.linalg.norm(X[E[:, 1]] - X[E[:, 0]], axis=1)
    m = np.full(len(E), max(1, int(factor)))
    if max_length is not None:
        m = np.maximum(m, np.ceil(l / float(max_length) - 1e-9).astype(int))
    if min_per_organ > 1:
        run = organ_runs(st)
        cnt = np.bincount(run)
        m = np.maximum(m, np.ceil(min_per_organ / cnt[run]).astype(int))
    m = np.maximum(m, 1)

    rad = np.asarray(st.radius, float)
    org = None if st.organ is None else np.asarray(st.organ, int)
    fixed = np.asarray(st.fixed, bool)
    pm = None if st.point_mass is None else np.asarray(st.point_mass, float)
    new_x, new_r, new_org, new_fix, edges, eon = [], [], [], [], [], []
    nxt = N
    for e, (a, b) in enumerate(E):
        k = int(m[e])
        if k == 1:
            edges.append((a, b))
            eon.append(e)
            continue
        t = np.arange(1, k) / k
        ids = nxt + np.arange(k - 1)
        nxt += k - 1
        new_x.append(X[a] + t[:, None] * (X[b] - X[a]))
        new_r.append(rad[a] + t * (rad[b] - rad[a]))
        if org is not None:
            new_org.append(np.full(k - 1, org[b]))
        new_fix.append(np.full(k - 1, fixed[a] and fixed[b]))
        chain = [a, *ids.tolist(), b]
        edges += list(zip(chain[:-1], chain[1:]))
        eon += [e] * k
    a2 = copy.deepcopy(asset)
    s2 = a2.stem
    if new_x:
        s2.nodes = np.vstack([X, *new_x])
        s2.radius = np.concatenate([rad, *new_r])
        s2.fixed = np.concatenate([fixed, *new_fix])
        if org is not None:
            s2.organ = np.concatenate([org, *new_org])
        if pm is not None:
            s2.point_mass = np.concatenate([pm, np.zeros(nxt - N)])
    s2.edges = np.array(edges, int).reshape(-1, 2)
    a2.meta = dict(a2.meta)
    a2.meta["refine"] = {"max_length": max_length, "min_per_organ": int(min_per_organ),
                         "factor": int(factor), "edges_before": int(len(E)), "edges_after": int(len(edges))}
    if return_map:
        return a2, RefineMap(m, np.array(eon, int), N)
    return a2


def refinement_summary(asset: PlantAsset) -> dict:
    """Edge counts and lengths per organ (for reports)."""
    st = asset.stem
    E = np.asarray(st.edges, int).reshape(-1, 2)
    l = np.linalg.norm(st.nodes[E[:, 1]] - st.nodes[E[:, 0]], axis=1)
    org = np.zeros(len(st.nodes), int) if st.organ is None else st.organ
    out = {"edges": int(len(E)), "nodes": int(len(st.nodes))}
    for o in np.unique(org[E[:, 1]]):
        k = org[E[:, 1]] == o
        out[f"organ{int(o)}"] = {"edges": int(k.sum()), "max_len": float(l[k].max()),
                                 "median_len": float(np.median(l[k]))}
    return out
