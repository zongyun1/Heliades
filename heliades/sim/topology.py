"""PlantAsset -> flattened, level-ordered articulated tree (parameter independent).

Bodies are the rod segments (stem edges) that are not clamped. Each body has one
3-DOF spherical joint at its proximal node, to the segment ending at that node (or to
the ground when that segment is clamped / the node is a root). Leaves and fruit are
*payloads*: rigid mass/inertia welded to the body that ends at their attachment node.

Layout (all numpy arrays, Warp-portable):
  bodies are renumbered in level order (depth 0 = jointed to ground), and inside a
  level sorted by parent, so level d is the contiguous slice lvl[d]:lvl[d+1] and the
  children of one parent are contiguous (segmented reductions, no atomics needed).

Mass properties are stored per *unit density* so that a batch of densities is a
linear combination: inertia = rho_stem * seg_part + rho_leaf * leaf_part + point_part.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotation import frame_from_tangent


@dataclass
class MassPart:
    """Mass moments about the body origin (proximal joint), in the body frame."""
    m: np.ndarray    # (nb,)
    h: np.ndarray    # (nb, 3) first moment m * c
    Io: np.ndarray   # (nb, 3, 3) second moment (inertia tensor about the origin)

    @staticmethod
    def zeros(nb):
        return MassPart(np.zeros(nb), np.zeros((nb, 3)), np.zeros((nb, 3, 3)))

    def add(self, b, m, c, Ic):
        """Add a rigid part with mass m, com c (body frame), inertia Ic about its com."""
        self.m[b] += m
        self.h[b] += m * c
        self.Io[b] += Ic + m * ((c @ c) * np.eye(3) - np.outer(c, c))


@dataclass
class LeafPayload:
    body: int               # host body (-1: on a clamped segment -> static)
    local: np.ndarray       # (V, 3) vertices in host body frame, relative to body origin
    faces: np.ndarray       # (F, 3)
    asset_index: int


class TreeTopology:
    def __init__(self, asset, include_leaves: bool = True):
        st = asset.stem
        X0 = np.asarray(st.nodes, float)
        E = np.asarray(st.edges, int).reshape(-1, 2)
        N, S = len(X0), len(E)
        fixed = np.asarray(st.fixed, bool)
        par_node = st.parent()
        self.N, self.X0 = N, X0
        self.fixed_nodes = fixed

        seg_of_child = -np.ones(N, int)
        seg_of_child[E[:, 1]] = np.arange(S)
        # clamped set must be closed under ancestors (a clamped base, free above)
        for i in np.flatnonzero(fixed):
            if par_node[i] >= 0 and not fixed[par_node[i]]:
                raise ValueError(f"fixed node {i} has a free parent; clamp must be a rooted subtree")
        roots = np.flatnonzero(par_node < 0)
        n_out = np.bincount(E[:, 0], minlength=N)
        for rt in roots:
            # B3: an unclamped root is a free-floating base (6-DOF root body); it must have
            # exactly one child edge so the floating base is a single rigid body
            if not fixed[rt] and n_out[rt] != 1:
                raise NotImplementedError("a free-floating root node must have exactly one child edge")
        static = fixed[E[:, 0]] & fixed[E[:, 1]]

        seg_l = np.linalg.norm(X0[E[:, 1]] - X0[E[:, 0]], axis=1)
        if (seg_l < 1e-9).any():
            raise ValueError("zero-length rod segment")
        rad = np.asarray(st.radius, float)
        seg_r = 0.5 * (rad[E[:, 0]] + rad[E[:, 1]])

        # rest frames by parallel transport, root to tip
        ndepth = np.zeros(N, int)
        for i in range(N):
            k = i
            while par_node[k] >= 0:
                k = par_node[k]
                ndepth[i] += 1
        R0 = np.zeros((S, 3, 3))
        for s in np.argsort(ndepth[E[:, 1]], kind="stable"):
            a, b = E[s]
            ps = seg_of_child[a]
            R0[s] = frame_from_tangent(X0[b] - X0[a], None if ps < 0 else R0[ps])

        # ---- bodies in level order
        free_segs = np.flatnonzero(~static)
        body_parent_seg = np.array([seg_of_child[E[s, 0]] for s in free_segs])
        body_parent_seg = np.where((body_parent_seg >= 0) & ~static[np.maximum(body_parent_seg, 0)],
                                   body_parent_seg, -1)
        seg2body = -np.ones(S, int)
        depth = {}
        order = []
        # BFS by depth
        level = [s for s, p in zip(free_segs, body_parent_seg) if p < 0]
        children = {}
        for s, p in zip(free_segs, body_parent_seg):
            if p >= 0:
                children.setdefault(int(p), []).append(int(s))
        lvl = [0]
        d = 0
        while level:
            for s in level:
                seg2body[s] = len(order)
                order.append(s)
                depth[s] = d
            lvl.append(len(order))
            nxt = []
            for s in level:                     # parents already in new order -> children grouped
                nxt += sorted(children.get(int(s), []))
            level = nxt
            d += 1
        order = np.array(order, int)
        nb = len(order)
        self.nb, self.nlev = nb, len(lvl) - 1
        self.lvl = np.array(lvl, int)
        self.seg = order                                    # body -> asset edge index
        self.seg2body = seg2body
        psg = np.array([seg_of_child[E[s, 0]] for s in order]) if nb else np.zeros(0, int)
        self.parent = np.where(psg >= 0, seg2body[np.maximum(psg, 0)], -1)  # -1 = ground
        self.node_a = E[order, 0]                           # proximal (joint) node
        self.node_b = E[order, 1]                           # distal node
        self.l = seg_l[order]
        self.r = seg_r[order]
        self.R0 = R0[order]                                 # rest world rotation
        Rp0 = np.where((self.parent >= 0)[:, None, None], self.R0[np.maximum(self.parent, 0)], np.eye(3))
        self.R0rel = np.einsum("nji,njk->nik", Rp0, self.R0)  # rest relative rotation
        self.o0_ground = X0[self.node_a]                    # origin for ground-jointed bodies
        # B3: floating-base bodies (level 0, proximal node not clamped): 6-DOF joint to ground
        self.floating = (self.parent < 0) & ~fixed[self.node_a] if nb else np.zeros(0, bool)
        self.float_idx = np.flatnonzero(self.floating)
        # per-level segmented reduction data (children of level d grouped by parent)
        self.red = []
        for dd in range(self.nlev):
            sl = slice(self.lvl[dd], self.lvl[dd + 1])
            p = self.parent[sl]
            if dd == 0:
                self.red.append(None)
                continue
            starts = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
            self.red.append((p[starts], starts))
        # B3: depth per body and ancestor table anc[b, t] = ancestor of b at depth t (b itself
        # at t = depth[b]; -1 below). Two bodies' lowest common ancestor is at depth
        # (number of equal leading entries) - 1 (used by the contact Delassus operator).
        self.depth = np.zeros(nb, int)
        for dd in range(self.nlev):
            self.depth[self.lvl[dd]:self.lvl[dd + 1]] = dd
        self.anc = -np.ones((nb, max(self.nlev, 1)), int)
        for dd in range(self.nlev):
            sl = slice(self.lvl[dd], self.lvl[dd + 1])
            if dd:
                self.anc[sl, :dd] = self.anc[self.parent[sl], :dd]
            self.anc[sl, dd] = np.arange(self.lvl[dd], self.lvl[dd + 1])
        # body that "owns" each node (node is its distal end); -1 for clamped nodes
        self.node_body = -np.ones(N, int)
        self.node_body[self.node_b] = np.arange(nb)

        # ---- joint compliance geometry: series of half segments (own + parent body)
        # compliance_bend = sum (l/2)/(E I) ; I = pi r^4/4, J = 2 I
        I_seg = np.pi * self.r ** 4 / 4
        cb = 0.5 * self.l / I_seg
        has_p = self.parent >= 0
        cb = cb + np.where(has_p, 0.5 * self.l[np.maximum(self.parent, 0)] /
                           I_seg[np.maximum(self.parent, 0)], 0.0)
        self.kgeom_bend = np.where(self.floating, 0.0, 1.0 / cb)   # k_bend = E * kgeom_bend
        self.kgeom_twist = np.where(self.floating, 0.0, 2.0 / cb)  # k_twist = G * kgeom_twist (J = 2 I)

        # ---- mass parts (unit densities)
        self.seg_part = MassPart.zeros(nb)
        for b in range(nb):
            l, r = self.l[b], self.r[b]
            m = np.pi * r * r * l
            it = m * (3 * r * r + l * l) / 12
            self.seg_part.add(b, m, np.array([0, 0, l / 2]), np.diag([it, it, m * r * r / 2]))
        self.point_part = MassPart.zeros(nb)
        pm = st.point_mass if st.point_mass is not None else np.zeros(N)
        fr_rad = {f.node: f.radius for f in getattr(asset, "fruits", [])}
        for n in np.flatnonzero(pm > 0):
            b = self.node_body[n]
            if b < 0:
                continue
            rf = fr_rad.get(int(n), 0.0)
            self.point_part.add(b, pm[n], np.array([0, 0, self.l[b]]),
                                0.4 * pm[n] * rf * rf * np.eye(3))
        self.leaf_part = MassPart.zeros(nb)
        self.leaves: list[LeafPayload] = []
        for k, leaf in enumerate(asset.leaves if include_leaves else []):
            b = self.node_body[leaf.attach_node]
            V = np.asarray(leaf.vertices, float)
            F = np.asarray(leaf.faces, int)
            if b < 0:
                self.leaves.append(LeafPayload(-1, V, F, k))
                continue
            loc = (V - X0[self.node_a[b]]) @ self.R0[b]       # R0^T (v - o)
            P = loc[F]
            A = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
            s = P.sum(1)
            C = (A[:, None, None] / 12.0) * (np.einsum("fki,fkj->fij", P, P) + np.einsum("fi,fj->fij", s, s))
            C = C.sum(0)
            self.leaf_part.m[b] += A.sum()
            self.leaf_part.h[b] += (A[:, None] * s / 3).sum(0)
            self.leaf_part.Io[b] += np.trace(C) * np.eye(3) - C
            self.leaves.append(LeafPayload(b, loc, F, k))
        self.leaf_area = sum(
            0.5 * np.linalg.norm(np.cross(L.local[L.faces][:, 1] - L.local[L.faces][:, 0],
                                          L.local[L.faces][:, 2] - L.local[L.faces][:, 0]), axis=1).sum()
            for L in self.leaves)
