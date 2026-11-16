"""Flattened multi-tree layout for the GPU solver (host side, numpy).

One *global body* index g covers every body of every environment. Bodies are ordered by

    (depth level d, rank k inside the level, environment e)

so that

* level d is one contiguous range [L[d], L[d+1]) over all environments: one kernel launch per
  level, launch dim = number of bodies at that depth summed over the batch, g = L[d] + tid;
* for a shared topology (B copies of one tree) this is the body-major layout g = i * B + e
  (i = TreeTopology body index): adjacent threads are the same body in adjacent
  environments, so every per-body load is coalesced;
* different topologies (different assets per environment) use the same code path: a level
  simply holds a different number of bodies per environment (no padding). Envs with fewer
  levels drop out of the deeper launches.

Children of a body are listed in a CSR (ch_start, ch_count, ch_list) in their topology order,
so the child-to-parent accumulation is a per-parent loop (segmented sum, no atomics).
Contact sample points (distal node + leaf vertices) are a per-topology CSR shared by all
environments that use the topology (no B-fold replication of leaf meshes).
"""
from __future__ import annotations

import numpy as np

from ..topology import TreeTopology


def contact_points(T: TreeTopology):
    """Contact sample points of a topology, grouped by body (same set as TreeSim.cp_*):
    the distal rod node of every body, then the leaf vertices of its leaves, in
    TreeSim's order within each body. Returns (start (nb,), count (nb,), local (P, 3),
    body (P,))."""
    cb = [np.arange(T.nb)]
    cl = [np.stack([np.zeros(T.nb), np.zeros(T.nb), T.l], 1)]
    for L in T.leaves:
        if L.body >= 0:
            cb.append(np.full(len(L.local), L.body))
            cl.append(L.local)
    body = np.concatenate(cb).astype(np.int64)
    loc = np.concatenate(cl)
    order = np.argsort(body, kind="stable")
    body, loc = body[order], loc[order]
    count = np.bincount(body, minlength=T.nb)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    return start, count, loc, body


class Layout:
    """Global body numbering for a batch of environments, each with its own TreeTopology
    (pass the *same* object B times for a shared topology)."""

    def __init__(self, topos: list[TreeTopology]):
        self.B = B = len(topos)
        self.topos = topos
        uniq = []
        tid = []
        for t in topos:
            for k, u in enumerate(uniq):
                if u is t:
                    tid.append(k)
                    break
            else:
                uniq.append(t)
                tid.append(len(uniq) - 1)
        self.unique = uniq
        self.topo_id = np.array(tid)
        self.shared = len(uniq) == 1
        self.nlev = max(t.nlev for t in topos)
        nbs = np.array([t.nb for t in topos])
        self.nb_env = nbs
        self.n = n = int(nbs.sum())

        # per-env local level index and rank of every body
        env_of, loc_of, lev_of, rank_of = [], [], [], []
        for e, t in enumerate(topos):
            lev = np.zeros(t.nb, np.int64)
            rank = np.zeros(t.nb, np.int64)
            for d in range(t.nlev):
                lev[t.lvl[d]:t.lvl[d + 1]] = d
                rank[t.lvl[d]:t.lvl[d + 1]] = np.arange(t.lvl[d + 1] - t.lvl[d])
            env_of.append(np.full(t.nb, e))
            loc_of.append(np.arange(t.nb))
            lev_of.append(lev)
            rank_of.append(rank)
        env_of = np.concatenate(env_of)
        loc_of = np.concatenate(loc_of)
        lev_of = np.concatenate(lev_of)
        rank_of = np.concatenate(rank_of)
        order = np.lexsort((env_of, rank_of, lev_of))          # sort by (lev, rank, env)
        self.env = env_of[order]
        self.local = loc_of[order]
        self.level = lev_of[order]
        self.L = np.searchsorted(self.level, np.arange(self.nlev + 1)).astype(np.int64)
        # g(e, i)
        offs = np.concatenate([[0], np.cumsum(nbs)[:-1]])
        flat_g = np.empty(n, np.int64)
        flat_g[order] = np.arange(n)
        self.env_offset = offs
        self.g_of = [flat_g[offs[e]:offs[e] + nbs[e]] for e in range(B)]   # local -> global
        if self.shared:
            nb = topos[0].nb
            assert np.array_equal(self.g_of[0], np.arange(nb) * B)       # body-major i * B + e
        # parents (global)
        par = np.empty(n, np.int64)
        for e, t in enumerate(topos):
            gp = np.where(t.parent >= 0, self.g_of[e][np.maximum(t.parent, 0)], -1)
            par[self.g_of[e]] = gp
        self.parent = par
        # children CSR in global order (children of a parent are increasing in g)
        has = np.flatnonzero(par >= 0)
        cl = has[np.argsort(par[has], kind="stable")]
        cnt = np.bincount(par[has], minlength=n) if len(has) else np.zeros(n, np.int64)
        self.ch_count = cnt
        self.ch_start = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        self.ch_list = cl.astype(np.int64)
        # max children (for docs/diagnostics)
        self.max_children = int(cnt.max()) if n else 0

        # per-body static geometry (global order)
        gl = lambda attr: np.concatenate([getattr(t, attr) for t in topos])
        loc_idx = np.concatenate([np.arange(t.nb) for t in topos])
        flat2g = np.concatenate(self.g_of)
        inv = np.empty(n, np.int64)
        inv[flat2g] = np.arange(n)                            # global -> flat (env-major) index
        self.flat_index = inv
        self.seg_l = gl("l")[inv]
        self.o0 = np.concatenate([t.o0_ground for t in topos])[inv]
        self.R0rel = np.concatenate([t.R0rel for t in topos])[inv]

        # contact sample points: per unique topology CSR, concatenated
        cps = [contact_points(t) for t in uniq]
        cp_off = np.concatenate([[0], np.cumsum([len(c[2]) for c in cps])[:-1]]).astype(np.int64)
        self.cp_local = np.concatenate([c[2] for c in cps]) if cps else np.zeros((0, 3))
        cs = np.empty(n, np.int64)
        cc = np.empty(n, np.int64)
        for e, t in enumerate(topos):
            k = self.topo_id[e]
            cs[self.g_of[e]] = cps[k][0] + cp_off[k]
            cc[self.g_of[e]] = cps[k][1]
        self.cp_start, self.cp_count = cs, cc
        self.cp_body_local = [c[3] for c in cps]
        # node index (into a concatenated env-major node array) of every body's distal node
        Ns = np.array([t.N for t in topos])
        self.N_env = Ns
        self.node_offset = np.concatenate([[0], np.cumsum(Ns)[:-1]]).astype(np.int64)
        gn = np.empty(n, np.int64)
        for e, t in enumerate(topos):
            gn[self.g_of[e]] = self.node_offset[e] + t.node_b
        self.gnode = gn

    # ------------------------------------------------------------------ gather / scatter
    def per_env_param(self, per_env, *, scalar=True):
        """Broadcast (B,) per-env values to global bodies."""
        return np.asarray(per_env)[self.env]

    def from_local(self, arrs):
        """List (per env) of (nb_e, ...) arrays, or one (B, nb, ...) array -> global (n, ...)."""
        if self.shared and not isinstance(arrs, list):
            a = np.asarray(arrs)
            return np.swapaxes(a, 0, 1).reshape((-1,) + a.shape[2:])
        out = None
        for e, a in enumerate(arrs):
            a = np.asarray(a)
            if out is None:
                out = np.empty((self.n,) + a.shape[1:], a.dtype)
            out[self.g_of[e]] = a
        return out

    def to_local(self, arr):
        """Global (n, ...) -> (B, nb, ...) for a shared topology, else a list per env."""
        arr = np.asarray(arr)
        if self.shared:
            nb = self.topos[0].nb
            return np.swapaxes(arr.reshape((nb, self.B) + arr.shape[1:]), 0, 1)
        return [arr[self.g_of[e]] for e in range(self.B)]

    def env_bodies(self, arr, e):
        return np.asarray(arr)[self.g_of[e]]
