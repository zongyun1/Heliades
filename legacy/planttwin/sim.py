"""Small-step XPBD for a PlantAsset, batched over parameter sets.

State (leading batch dim B = number of parameter sets simulated in lock-step)
  particles  x, v    (B, n, 3): stem nodes + leaf vertices (leaf base welded to stem node)
  segments   q, omega (B, S, 4|3): one orientation per stem edge (Cosserat rod)

Constraints
  RodShear      stem edge (p1, p2, q):  p2 - p1 = l * d3(q)        (stretch + shear)
  RodBendTwist  parent/child segments:  q_child = q_parent * Omega0 (bend E*I, twist G*J)
  FrameAttach   leaf base ring rigidly attached to the petiole-tip frame (two-way)
  Distance      leaf mesh edges                                     (leaf stretch)
  MidpointBend  near-collinear leaf vertex triples                  (leaf bending; works on
                any reconstructed leaf mesh, not just grids)
Colliders: kinematic spheres (finger / robot probe) and the ground plane.

Rod constraints follow the XPBD rigid-body formulation (Mueller et al. 2020) applied to
Cosserat segments (cf. Kugelstadt & Schoemer 2016). Joint compliance = l / (E I), so the
fitted stiffness is a physical modulus. Gauss-Seidel within a graph colour is vectorised;
numpy cost is per-op overhead, so batching B parameter sets is nearly free -- this is what
makes population-based Real2Sim fitting cheap. Port to Warp/Newton for real speed.

Masses (densities) are shared across the batch; stiffness and damping are per-batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .asset import PhysicsParams, PlantAsset

GRAVITY = np.array([0.0, 0.0, -9.81])
E3 = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- quaternions (w,x,y,z)
def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def qconj(q):
    return q * np.array([1.0, -1, -1, -1])


def qrot(q, v):
    """Rotate v by q (broadcasting). Uses v' = v + 2w(u x v) + 2u x (u x v)."""
    w, u = q[..., :1], q[..., 1:]
    uv = np.cross(u, v)
    return v + 2 * w * uv + 2 * np.cross(u, uv)


def qd3(q):
    """Third director d3 = R(q) e3."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)


def qnormalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def q_apply_rotvec(q, dtheta):
    """q <- normalize(q + 0.5 * (0, dtheta) * q): small world-frame rotation."""
    dq = np.concatenate([np.zeros(dtheta.shape[:-1] + (1,)), dtheta], -1)
    return qnormalize(q + 0.5 * qmul(dq, q))


def q_from_two(a, b):
    """Minimal rotation taking unit vector a to unit vector b."""
    c = np.cross(a, b)
    d = float(a @ b)
    if d < -0.999999:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        return np.concatenate([[0.0], axis / np.linalg.norm(axis)])
    return qnormalize(np.concatenate([[1.0 + d], c]))


# ---------------------------------------------------------------- colouring
def _color(idx: np.ndarray) -> list[np.ndarray]:
    """Greedy graph colouring: constraints in one colour share no entity."""
    used: list[set] = []
    assign = np.empty(len(idx), int)
    for r, tup in enumerate(idx):
        for c, u in enumerate(used):
            if not u.intersection(tup):
                break
        else:
            c = len(used)
            used.append(set())
        used[c].update(tup)
        assign[r] = c
    return [np.flatnonzero(assign == c) for c in range(len(used))]


def _safe_dir(C):
    c = np.linalg.norm(C, axis=-1)
    return c, C / np.maximum(c, 1e-12)[..., None]


# ---------------------------------------------------------------- particle constraints
class ParticleConstraint:
    """Scalar XPBD constraints on particles. Subclasses: eval -> (C (B,m), dC/dx (B,m,k,3))."""

    def __init__(self, idx, x):
        self.idx = np.asarray(idx, int).reshape(len(idx), -1)
        self.alpha = np.zeros(1)                      # (B,)
        self.rest = np.zeros(len(self.idx))
        self.rest = self.eval(x[None], slice(None))[0][0]
        self.colors = _color(self.idx)

    def project(self, S, h2):
        p, w, lam = S.p, S.w, S.lam[id(self)]
        at = (self.alpha / h2)[:, None]
        for sel in self.colors:
            I = self.idx[sel]
            C, G = self.eval(p, sel)
            wi = w[I]
            den = (wi * (G * G).sum(-1)).sum(-1) + at
            dl = -(C + at * lam[:, sel]) / den
            lam[:, sel] += dl
            p[:, I] += (wi * dl[..., None])[..., None] * G


class Distance(ParticleConstraint):
    def eval(self, p, sel):
        i, j = self.idx[sel, 0], self.idx[sel, 1]
        L, n = _safe_dir(p[:, j] - p[:, i])
        return L - self.rest[sel], np.stack([-n, n], 2)


class MidpointBend(ParticleConstraint):
    """C = |x_b - centroid(a,b,c)| - h0 (leaf mesh only: the mesh supplies the frame)."""

    def eval(self, p, sel):
        a, b, c = self.idx[sel].T
        L, n = _safe_dir(p[:, b] - (p[:, a] + p[:, b] + p[:, c]) / 3)
        C = np.where(L > 1e-12, L - self.rest[sel], 0.0)
        return C, np.stack([-n / 3, 2 * n / 3, -n / 3], 2)


# ---------------------------------------------------------------- rod constraints
class RodShear:
    """C = p2 - p1 - l d3(q)  (vector; projected along its own direction)."""

    def __init__(self, edges, x):
        self.idx = np.asarray(edges, int)            # (S, 2) particle ids; segment id = row
        self.seg = np.arange(len(self.idx))
        self.l = np.linalg.norm(x[self.idx[:, 1]] - x[self.idx[:, 0]], axis=1)
        self.alpha = np.zeros(1)
        ent = np.concatenate([self.idx, (self.seg + len(x))[:, None]], 1)
        self.colors = _color(ent)

    def project(self, S, h2):
        p, q, w, wq = S.p, S.q, S.w, S.wq
        at = (self.alpha / h2)[:, None]
        for sel in self.colors:
            i, j, s = self.idx[sel, 0], self.idx[sel, 1], self.seg[sel]
            l = self.l[sel][:, None]
            d3 = qd3(q[:, s])
            c, n = _safe_dir(p[:, j] - p[:, i] - l * d3)
            gth = l * np.cross(n, d3)
            den = w[i] + w[j] + wq[s] * (gth * gth).sum(-1) + at
            dl = np.where(den > 0, -c / np.maximum(den, 1e-30), 0.0)[..., None]
            p[:, i] -= w[i][:, None] * dl * n
            p[:, j] += w[j][:, None] * dl * n
            q[:, s] = q_apply_rotvec(q[:, s], wq[s][:, None] * dl * gth)


class RodBendTwist:
    """Relative rotation between parent and child segment kept at its rest value.
    Split into bend (perpendicular to the child tangent) and twist (along it)."""

    def __init__(self, pairs, q0):
        self.idx = np.asarray(pairs, int).reshape(-1, 2)  # (J, 2) segment ids
        self.omega0 = qmul(qconj(q0[self.idx[:, 0]]), q0[self.idx[:, 1]])
        self.alpha_bend = np.zeros((1, len(self.idx)))   # (B, J)
        self.alpha_twist = np.zeros((1, len(self.idx)))
        self.colors = _color(self.idx)

    def project(self, S, h2):
        q, wq = S.q, S.wq
        lb, lt = S.lam[id(self)]
        for sel in self.colors:
            a, b = self.idx[sel, 0], self.idx[sel, 1]
            e = qmul(q[:, b], qconj(qmul(q[:, a], self.omega0[sel])))
            e = e * np.where(e[..., :1] < 0, -1.0, 1.0)
            phi = 2.0 * e[..., 1:]                                # world rotation error
            t = qd3(q[:, b])
            tw = (phi * t).sum(-1, keepdims=True) * t
            for part, alpha, lam in ((phi - tw, self.alpha_bend, lb), (tw, self.alpha_twist, lt)):
                c, n = _safe_dir(part)
                at = alpha[:, sel] / h2
                den = wq[a] + wq[b] + at
                dl = np.where(den > 0, -(c + at * lam[:, sel]) / np.maximum(den, 1e-30), 0.0)
                lam[:, sel] += dl
                q[:, b] = q_apply_rotvec(q[:, b], (wq[b] * dl)[..., None] * n)
                q[:, a] = q_apply_rotvec(q[:, a], -(wq[a] * dl)[..., None] * n)


class FrameAttach:
    """Leaf vertex v rigidly attached to segment frame: p_v = p_tip + R(q) r_local."""

    def __init__(self, triples, x, q0, n_particles):
        self.idx = np.asarray(triples, int).reshape(-1, 3)    # (A, 3): v, tip, seg
        v, tip, s = self.idx.T
        self.r_local = qrot(qconj(q0[s]), x[v] - x[tip])
        ent = self.idx.copy()
        ent[:, 2] += n_particles
        self.colors = _color(ent)

    def project(self, S, h2):
        p, q, w, wq = S.p, S.q, S.w, S.wq
        for sel in self.colors:
            v, tip, s = self.idx[sel].T
            rw = qrot(q[:, s], self.r_local[sel])
            c, n = _safe_dir(p[:, v] - p[:, tip] - rw)
            gth = np.cross(n, rw)
            den = w[v] + w[tip] + wq[s] * (gth * gth).sum(-1)
            dl = np.where(den > 0, -c / np.maximum(den, 1e-30), 0.0)[..., None]
            p[:, v] += w[v][:, None] * dl * n
            p[:, tip] -= w[tip][:, None] * dl * n
            q[:, s] = q_apply_rotvec(q[:, s], wq[s][:, None] * dl * gth)


@dataclass
class Sphere:
    """Kinematic spherical collider. center(t) -> (3,) or (B,3); active(t) -> bool."""

    radius: float
    center: Callable[[float], np.ndarray]
    active: Callable[[float], bool] = lambda t: True


class _State:
    pass


# ---------------------------------------------------------------- model
class PlantModel:
    def __init__(self, asset: PlantAsset):
        self.asset = asset
        st = asset.stem
        N = len(st.nodes)
        xs, self.leaf_idx = [st.nodes], []
        nxt = N
        for leaf in asset.leaves:
            m = np.empty(len(leaf.vertices), dtype=int)
            others = [k for k in range(len(leaf.vertices)) if k != leaf.attach_vertex]
            m[others] = np.arange(nxt, nxt + len(others))
            m[leaf.attach_vertex] = leaf.attach_node
            nxt += len(others)
            self.leaf_idx.append(m)
            xs.append(leaf.vertices[others])
        x = np.concatenate(xs).astype(float)
        self.n = len(x)
        self.fixed = np.zeros(self.n, bool)
        self.fixed[:N] = st.fixed

        # --- rod: one segment per stem edge, frames by parallel transport from parents
        E = st.edges
        par = st.parent()
        depth = np.zeros(N, int)
        for b in range(N):
            k = b
            while par[k] >= 0:
                k, depth[b] = par[k], depth[b] + 1
        seg_of_child = -np.ones(N, int)
        seg_of_child[E[:, 1]] = np.arange(len(E))
        q0 = np.zeros((len(E), 4))
        for s in np.argsort(depth[E[:, 1]], kind="stable"):
            a, b = E[s]
            t = (x[b] - x[a]) / np.linalg.norm(x[b] - x[a])
            ps = seg_of_child[a]
            q0[s] = q_from_two(E3, t) if ps < 0 else qnormalize(qmul(q_from_two(qd3(q0[ps]), t), q0[ps]))
        self.q0 = q0
        self.seg_fixed = st.fixed[E[:, 0]] & st.fixed[E[:, 1]]
        pairs = [(seg_of_child[a], s) for s, (a, b) in enumerate(E) if seg_of_child[a] >= 0]

        # --- leaves
        dist, bend, attach = [], [], []
        for leaf, m in zip(asset.leaves, self.leaf_idx):
            F = m[leaf.faces]
            LE = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
            dist += LE.tolist()
            nbr: dict[int, set] = {}
            for u, v in LE:
                nbr.setdefault(u, set()).add(v); nbr.setdefault(v, set()).add(u)
            for b, ns in nbr.items():
                ns = sorted(ns)
                for ia in range(len(ns)):
                    for ic in range(ia + 1, len(ns)):
                        a, c = ns[ia], ns[ic]
                        u1, u2 = x[a] - x[b], x[c] - x[b]
                        if u1 @ u2 / (np.linalg.norm(u1) * np.linalg.norm(u2)) < -0.85:
                            bend.append((a, b, c))
            tip = leaf.attach_node
            attach += [(v, tip, seg_of_child[tip]) for v in sorted(nbr[tip])]

        self.shear = RodShear(E, x)
        self.bendtwist = RodBendTwist(pairs, q0)
        self.attach = FrameAttach(attach, x, q0, self.n)
        self.leaf_stretch = Distance(dist, x)
        self.leaf_bend = MidpointBend(bend, x)
        self.constraints = [self.shear, self.bendtwist, self.attach, self.leaf_stretch, self.leaf_bend]
        self.x_rest = x
        self.set_params(asset.physics)

    # ------------------------------------------------------------------------
    def set_params(self, phys: PhysicsParams | Sequence[PhysicsParams]):
        """One PhysicsParams, or a list of them (one per batch element). Densities must
        agree across the batch; the first one is used for masses."""
        plist = [phys] if isinstance(phys, PhysicsParams) else list(phys)
        self.phys_list, self.B = plist, len(plist)
        p0 = plist[0]
        st, x = self.asset.stem, self.x_rest
        E = st.edges
        r = 0.5 * (st.radius[E[:, 0]] + st.radius[E[:, 1]])
        l = self.shear.l
        mseg = p0.stem_density * np.pi * r * r * l
        m = np.zeros(self.n)
        np.add.at(m, E[:, 0], mseg / 2)
        np.add.at(m, E[:, 1], mseg / 2)
        if st.point_mass is not None:
            m[: len(st.point_mass)] += st.point_mass
        for leaf, idx in zip(self.asset.leaves, self.leaf_idx):
            P = x[idx[leaf.faces]]
            area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
            for k in range(3):
                m += np.bincount(idx[leaf.faces[:, k]], area * p0.leaf_areal_density / 3, minlength=self.n)
        self.mass = m
        self.w = np.where(self.fixed, 0.0, 1.0 / np.maximum(m, 1e-12))
        inertia = mseg * (l * l / 12 + r * r / 4)          # isotropic approximation
        self.wq = np.where(self.seg_fixed, 0.0, 1.0 / np.maximum(inertia, 1e-18))

        # joint compliance = l_joint / (E I),  twist: l_joint / (G J), J = 2 I
        a, b = self.bendtwist.idx.T
        rj, lj = 0.5 * (r[a] + r[b]), 0.5 * (l[a] + l[b])
        I = np.pi * rj**4 / 4
        Ey = np.array([p.stem_youngs_modulus for p in plist])[:, None]
        G = Ey / (2 * (1 + np.array([p.stem_poisson for p in plist])[:, None]))
        self.bendtwist.alpha_bend = lj / (Ey * I)
        self.bendtwist.alpha_twist = lj / (G * 2 * I)
        self.shear.alpha = np.array([p.stem_stretch_compliance for p in plist])
        self.leaf_stretch.alpha = np.array([p.leaf_stretch_compliance for p in plist])
        self.leaf_bend.alpha = np.array([p.leaf_bend_compliance for p in plist])
        self.damping = np.array([p.damping for p in plist])

    def leaf_keypoints(self, name="tip") -> np.ndarray:
        return np.array([m[l.keypoints[name]] for l, m in zip(self.asset.leaves, self.leaf_idx)])

    # ------------------------------------------------------------------------
    def simulate(
        self,
        duration: float,
        fps: int = 60,
        substeps: int = 64,
        colliders: tuple[Sphere, ...] = (),
        x0: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        damping: float | None = None,
        record: np.ndarray | None = None,
    ):
        """Simulate all B parameter sets. Returns (frames (F+1, B, K, 3), x_final, q_final),
        K = len(record) (default: all particles)."""
        B = self.B
        h = 1.0 / (fps * substeps)
        h2 = h * h
        dmp = self.damping if damping is None else np.full(B, damping)
        damp = np.exp(-dmp * h)[:, None, None]
        x = np.broadcast_to(self.x_rest if x0 is None else x0, (B, self.n, 3)).copy()
        q = np.broadcast_to(self.q0 if q_init is None else q_init, (B,) + self.q0.shape).copy()
        v = np.zeros_like(x)
        om = np.zeros(q.shape[:-1] + (3,))
        free = (~self.fixed)[None, :, None]
        rec = slice(None) if record is None else record
        S = _State()
        S.w, S.wq = self.w, self.wq
        frames = [x[:, rec].copy()]
        t = 0.0
        for _ in range(int(round(duration * fps))):
            for _ in range(substeps):
                t += h
                v = (v + GRAVITY * h * free) * damp
                om = om * damp
                S.p = x + v * h
                S.q = q_apply_rotvec(q, om * h)
                nj = len(self.bendtwist.idx)
                S.lam = {
                    id(self.leaf_stretch): np.zeros((B, len(self.leaf_stretch.idx))),
                    id(self.leaf_bend): np.zeros((B, len(self.leaf_bend.idx))),
                    id(self.bendtwist): (np.zeros((B, nj)), np.zeros((B, nj))),
                }
                for c in self.constraints:
                    c.project(S, h2)
                p = S.p
                for sph in colliders:
                    if sph.active(t):
                        c = np.broadcast_to(sph.center(t), (B, 3))[:, None, :]
                        L, n = _safe_dir(p - c)
                        hit = ((L < sph.radius) & ~self.fixed)[..., None]
                        p = np.where(hit, c + n * sph.radius, p)
                p[..., 2] = np.maximum(p[..., 2], 0.0)
                v = (p - x) / h
                dq = qmul(S.q, qconj(q))
                dq = dq * np.where(dq[..., :1] < 0, -1.0, 1.0)
                om = 2.0 * dq[..., 1:] / h
                x, q = p, S.q
            frames.append(x[:, rec].copy())
        return np.array(frames), x, q

    def settle(self, duration=1.0, damping=15.0, **kw):
        """Static equilibrium under gravity (heavily damped): what a photo sees.
        Returns (x_eq (B,n,3), q_eq (B,S,4))."""
        _, x, q = self.simulate(duration, damping=damping, record=np.array([0]), **kw)
        return x, q
