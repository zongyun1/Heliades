"""B3: unilateral contact with Coulomb friction on the tree solver (docs/sim/CONTACT.md).

Implicit, velocity-level, solved consistently with the per-step tree factorisation
(Baraff 1996 O(kn) Schur complement):

  1. TreeSim.step factors the implicit system  M~ a = rhs  once (articulated-body sweep)
     and computes the contact-free update  u* = u_n + h a*.
  2. Contacts are detected at q_n: plant features (rod capsules, leaf vertices, fruit
     spheres) against colliders (ground plane, kinematic spheres / capsules, static stakes).
     Each contact c has a frame D_c = [n; t1; t2] (n points from the collider into the plant),
     a gap phi_c and a point p_c on body b_c.
  3. Delassus operator  W = J M~^-1 J^T  (3k x 3k; velocity change per unit impulse) from the
     stored factorisation, in O(k depth + k^2): for every contact one up-sweep along its
     root path and one down-sweep (the test-force response), plus a pulled-back row ``rho``
     that carries a parent acceleration to the contact point through the off-path joints.
     W[c, c'] = Gw_c Vp_{c'}(e) + rho_c(e) a_{c'}(e), e = depth of the lowest common
     ancestor of b_c and b_c' (exact; checked against brute-force columns in the tests).
  4. Projected Gauss-Seidel on the 3k impulses only (B batched in lock-step; Warp kernel or
     numpy), Coulomb friction as an isotropic cone: per contact the normal projection, then
     the exact local tangential QP over the disk |lam_t| <= mu lam_n in the W_tt metric
     (maximal dissipation; non-associated, so no dilation). Normal target v_n >= -phi/h
     (speculative) or a capped Baumgarte recovery for penetration; warm-started from the
     previous step; per-contact impulse safeguard.
  5. Back-substitution: the contact impulses enter as body forces in one bias-only O(n)
     sweep of the same factorisation (TreeSim.response); u_{n+1} = u* + M~^-1 J^T lambda.

Penalty contact (B1-style implicit spring + damper, normal only) is kept as a baseline:
every collider takes ``mode="penalty"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tree import _cross, _mv

EYE6 = np.eye(6)


# ---------------------------------------------------------------- parameters, colliders
@dataclass
class ContactParams:
    iters: int = 200            # max PGS sweeps
    tol: float = 1e-6           # stop when a sweep changes no contact velocity by more than tol [m/s]
    erp: float = 0.5            # Baumgarte factor for penetration recovery (per step)
    max_recovery: float = 0.2   # max separation speed used to remove penetration [m/s]
    max_depth_recovery: float = 5e-3  # curved colliders: deeper penetrations recover slowly...
    deep_recovery: float = 0.02       # ...at this speed [m/s] (a half-space always recovers fully)
    impulse_cap: float = 100.0  # safeguard: lam_n <= cap (|v_free,n| + |target| + 1 mm/s) / W_nn
    margin: float = 2e-3        # speculative contact distance [m] (+ spec_factor * h * point speed)
    spec_factor: float = 3.0
    max_margin: float = 0.02    # cap of the speculative band [m] (fast whipping leaflets)
    max_contacts: int = 512     # per batch member; the smallest gaps are kept
    max_per_body: int = 0       # contact reduction (0 = off): at most this many per (feature group, collider)
    leaf_thickness: float = 5e-4  # contact thickness of leaf blades [m]
    warm_start: bool = True
    sor: float = 1.0            # PGS relaxation
    backend: str = "auto"       # PGS inner loop: "warp" (kernel, one thread per batch), "numpy", "auto"


class _Collider:
    contact_mode = "implicit"
    mu = 0.5

    def velocity_b(self, B):
        v = getattr(self, "velocity", None)
        return np.zeros((B, 3)) if v is None else np.broadcast_to(np.asarray(v, float), (B, 3))


@dataclass
class GroundPlane(_Collider):
    """Static half-space {x : n.x >= height} (n is normalised)."""
    height: float = 0.0
    normal: tuple = (0.0, 0.0, 1.0)
    mu: float = 0.5
    mode: str = "implicit"
    stiffness: float = 2e4      # penalty mode only
    damping: float = 20.0
    # feature filter: rod bodies whose organ is in `organs` (None: all), leaf vertices, fruit
    organs: tuple | None = None
    leaves: bool = True
    fruit: bool = True
    bodies: np.ndarray | None = None   # optional (nb,) bool mask of rod bodies (topology order)

    @property
    def contact_mode(self):
        return self.mode


@dataclass
class SphereObstacle(_Collider):
    """Kinematic sphere (finger tip). center (3,) or (B, 3); velocity of its translation."""
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.01
    velocity: np.ndarray | None = None
    mu: float = 0.5
    mode: str = "implicit"
    stiffness: float = 5e3
    damping: float = 2.0
    # feature filter: rod bodies whose organ is in `organs` (None: all), leaf vertices, fruit
    organs: tuple | None = None
    leaves: bool = True
    fruit: bool = True
    bodies: np.ndarray | None = None   # optional (nb,) bool mask of rod bodies (topology order)

    @property
    def contact_mode(self):
        return self.mode


@dataclass
class CapsuleObstacle(_Collider):
    """Kinematic capsule (finger / gripper jaw / stake): segment a-b swept by `radius`.
    a, b: (3,) or (B, 3); velocity: common translation velocity."""
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    radius: float = 0.01
    velocity: np.ndarray | None = None
    mu: float = 0.5
    mode: str = "implicit"
    stiffness: float = 5e3
    damping: float = 2.0
    # feature filter: rod bodies whose organ is in `organs` (None: all), leaf vertices, fruit
    organs: tuple | None = None
    leaves: bool = True
    fruit: bool = True
    bodies: np.ndarray | None = None   # optional (nb,) bool mask of rod bodies (topology order)

    @property
    def contact_mode(self):
        return self.mode


def stake(x, y, radius=0.006, height=1.2, z0=-0.05, mu=0.4, mode="implicit", organs=(0,), leaves=False,
          fruit=True, bodies=None):
    """Static vertical stake (capsule) at (x, y). By default it touches the main stem (organ 0)
    and fruit only: leaves and petioles of a real staked plant pass around the stake, and a
    vertical line through a bushy Helios canopy always intersects some of them initially."""
    return CapsuleObstacle(np.array([x, y, z0], float), np.array([x, y, z0 + height], float),
                           radius, None, mu, mode, organs=organs, leaves=leaves, fruit=fruit, bodies=bodies)


def main_axis(sim, node=None):
    """(nb,) bool mask of the bodies on the path root -> `node` (default: the highest stem node)."""
    T, st = sim.topo, sim.asset.stem
    if node is None:
        org = np.zeros(T.N, int) if st.organ is None else st.organ
        cand = np.flatnonzero(org == 0)
        node = cand[np.argmax(st.nodes[cand, 2])]
    m = np.zeros(T.nb, bool)
    b = T.node_body[node]
    while b >= 0:
        m[b], b = True, T.parent[b]
    return m


def _feature_masks(sim, cm, c):
    """(rod body mask (nb,), point mask (P,)) of the features collider c acts on."""
    T = sim.topo
    org = getattr(c, "organs", None)
    if org is None:
        rod = np.ones(T.nb, bool)
    else:
        o = sim.asset.stem.organ
        o = np.zeros(T.N, int) if o is None else np.asarray(o)
        rod = np.isin(o[T.node_b], np.asarray(org))
    if getattr(c, "bodies", None) is not None:
        rod = rod & np.asarray(c.bodies, bool)
    pt = np.where(cm.p_kind == 0, rod[cm.p_body], True)
    if not getattr(c, "leaves", True):
        pt &= cm.p_kind != 1
    if not getattr(c, "fruit", True):
        pt &= cm.p_kind != 2
    return rod, pt


# ---------------------------------------------------------------- plant contact features
class ContactModel:
    """Contact features of a TreeSim, in host-body coordinates (built once per sim):
    rod capsules (one per body: o -> tip, radius r), and points: distal rod nodes (radius r),
    proximal points of floating bases, leaf vertices (radius leaf_thickness / 2), fruit."""

    def __init__(self, sim, leaf_thickness):
        T = sim.topo
        nb = T.nb
        pb, pl, pr, pk = [np.arange(nb)], [np.stack([np.zeros(nb), np.zeros(nb), T.l], 1)], [T.r], [np.zeros(nb, int)]
        pg = [np.arange(nb)]                    # reduction group: rod -> body, each leaf / fruit its own
        fl = T.float_idx
        if len(fl):
            pb.append(fl), pl.append(np.zeros((len(fl), 3))), pr.append(T.r[fl]), pk.append(np.zeros(len(fl), int))
            pg.append(fl)
        g = nb
        for L in T.leaves:
            if L.body >= 0:
                pb.append(np.full(len(L.local), L.body)), pl.append(L.local)
                pr.append(np.full(len(L.local), 0.5 * leaf_thickness)), pk.append(np.ones(len(L.local), int))
                pg.append(np.full(len(L.local), g))
            g += 1
        for f in getattr(sim.asset, "fruits", []):
            b = T.node_body[f.node]
            if b >= 0:
                pb.append([b]), pl.append([[0.0, 0.0, T.l[b]]]), pr.append([f.radius]), pk.append([2])
                pg.append([g])
            g += 1
        self.p_group = np.concatenate(pg).astype(int)
        self.p_body = np.concatenate(pb).astype(int)
        self.p_local = np.concatenate(pl).astype(float)
        self.p_rad = np.concatenate(pr).astype(float)
        self.p_kind = np.concatenate(pk).astype(int)
        self.n_rod_nodes = nb
        self.nb = nb


def _contact_model(sim):
    prm = sim.contact_params or ContactParams()
    cm = getattr(sim, "_contact_model", None)
    if cm is None or cm.leaf_thickness != prm.leaf_thickness:
        cm = ContactModel(sim, prm.leaf_thickness)
        cm.leaf_thickness = prm.leaf_thickness
        sim._contact_model = cm
    return cm, prm


def _closest_seg_point(p, a, b):
    """Closest point on segments a-b to points p (broadcasting); returns (q, s)."""
    ab = b - a
    s = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-300), 0.0, 1.0)
    return a + s[..., None] * ab, s


def _closest_seg_seg(p1, q1, p2, q2):
    """Closest points between segments p1-q1 and p2-q2 (Ericson 5.1.9), broadcasting."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    den = a * e - b * b
    s = np.where(den > 1e-14 * a * e + 1e-300, np.clip((b * f - c * e) / np.where(den > 0, den, 1.0), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-300)
    s = np.where(t < 0, np.clip(-c / np.maximum(a, 1e-300), 0, 1), np.where(t > 1, np.clip((b - c) / np.maximum(a, 1e-300), 0, 1), s))
    t = np.clip(t, 0, 1)
    # degenerate segments (a sphere is a capsule with a == b)
    deg2 = e <= 1e-24
    deg1 = a <= 1e-24
    s = np.where(deg2, np.clip(-c / np.maximum(a, 1e-300), 0, 1), s)
    t = np.where(deg2, 0.0, t)
    t = np.where(deg1 & ~deg2, np.clip(f / np.maximum(e, 1e-300), 0, 1), t)
    s = np.where(deg1, 0.0, s)
    return p1 + s[..., None] * d1, p2 + t[..., None] * d2


def _tangents(n):
    a = np.where((np.abs(n[..., 2]) < 0.9)[..., None], np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    t1 = _cross(a, n)
    t1 /= np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-300)
    return t1, _cross(n, t1)


@dataclass
class _Cand:
    bi: np.ndarray      # batch index
    body: np.ndarray
    p: np.ndarray       # contact point on the plant surface (world)
    n: np.ndarray       # normal (collider -> plant)
    phi: np.ndarray     # gap (< 0: penetration)
    key: np.ndarray     # stable id (collider, feature) for warm starting
    vc: np.ndarray      # collider velocity at the point
    mu: np.ndarray
    col: np.ndarray     # collider index
    grp: np.ndarray     # reduction group (rod body / leaf / fruit)


def detect(sim, k, colliders, h, prm=None):
    """Contact candidates of all colliders at configuration k (TreeSim._Kin)."""
    cm, p0 = _contact_model(sim)
    prm = prm or p0
    B = k.R.shape[0]
    pw = k.o[:, cm.p_body] + _mv(k.R[:, cm.p_body], cm.p_local)            # (B, P, 3)
    vp = k.vO[:, cm.p_body] + _cross(k.w[:, cm.p_body], pw - k.o[:, cm.p_body])
    out = []
    nfeat = cm.nb + len(cm.p_body) + 1
    for ci, c in enumerate(colliders):
        vcol = c.velocity_b(B)
        if isinstance(c, GroundPlane):
            nrm = np.asarray(c.normal, float)
            nrm = nrm / np.linalg.norm(nrm)
            gap = pw @ nrm - c.height - cm.p_rad
            sel = gap < np.minimum(prm.margin + prm.spec_factor * h * np.linalg.norm(vp, axis=-1), prm.max_margin)
            sel &= _feature_masks(sim, cm, c)[1][None]
            bi, pi = np.nonzero(sel)
            n = np.broadcast_to(nrm, (len(bi), 3))
            p = pw[bi, pi] - cm.p_rad[pi, None] * n
            out.append(_Cand(bi, cm.p_body[pi], p, n.copy(), gap[bi, pi], ci * nfeat + cm.nb + pi,
                             vcol[bi], np.full(len(bi), c.mu), np.full(len(bi), ci), cm.p_group[pi]))
            continue
        if isinstance(c, SphereObstacle):
            ctr = np.broadcast_to(np.asarray(c.center, float), (B, 3))
            A_, B_ = ctr, ctr
        elif isinstance(c, CapsuleObstacle):
            A_ = np.broadcast_to(np.asarray(c.a, float), (B, 3))
            B_ = np.broadcast_to(np.asarray(c.b, float), (B, 3))
        else:
            raise TypeError(f"unknown collider {type(c)}")
        Rad = c.radius
        # (i) points that are not rod nodes (leaf vertices, fruit, floating-base ends): point-segment
        rodm, ptm = _feature_masks(sim, cm, c)
        pts = np.flatnonzero((cm.p_kind > 0) & ptm)
        q, _ = _closest_seg_point(pw[:, pts], A_[:, None], B_[:, None])
        dv = pw[:, pts] - q
        dist = np.linalg.norm(dv, axis=-1)
        gap = dist - Rad - cm.p_rad[pts]
        nn = np.where((dist > 1e-12)[..., None], dv / np.maximum(dist, 1e-12)[..., None], np.array([0.0, 0.0, 1.0]))
        sel = gap < np.minimum(prm.margin + prm.spec_factor * h * np.linalg.norm(vp[:, pts] - vcol[:, None], axis=-1),
                               prm.max_margin)
        bi, j = np.nonzero(sel)
        if len(bi):
            pi = pts[j]
            n = nn[bi, j]
            out.append(_Cand(bi, cm.p_body[pi], pw[bi, pi] - cm.p_rad[pi, None] * n, n, gap[bi, j],
                             ci * nfeat + cm.nb + pi, vcol[bi], np.full(len(bi), c.mu), np.full(len(bi), ci),
                             cm.p_group[pi]))
        # (ii) rod capsules: segment-segment
        x1, x2 = _closest_seg_seg(k.o, k.tip, A_[:, None], B_[:, None])
        dv = x1 - x2
        dist = np.linalg.norm(dv, axis=-1)
        gap = dist - Rad - sim.topo.r[None]
        # axis through the collider centre: fall back to the body's x axis (any normal to it)
        nn = np.where((dist > 1e-12)[..., None], dv / np.maximum(dist, 1e-12)[..., None], k.R[..., :, 0])
        vx = k.vO + _cross(k.w, x1 - k.o)
        sel = gap < np.minimum(prm.margin + prm.spec_factor * h * np.linalg.norm(vx - vcol[:, None], axis=-1),
                               prm.max_margin)
        sel &= rodm[None]
        bi, bb = np.nonzero(sel)
        if len(bi):
            n = nn[bi, bb]
            out.append(_Cand(bi, bb, x1[bi, bb] - sim.topo.r[bb, None] * n, n, gap[bi, bb], ci * nfeat + bb,
                             vcol[bi], np.full(len(bi), c.mu), np.full(len(bi), ci), bb))
    if not out:
        return None
    cat = lambda f: np.concatenate([getattr(o, f) for o in out])
    cand = _Cand(*(cat(f) for f in _Cand.__dataclass_fields__))
    cand = _reduce(cand, prm.max_per_body)
    return _cap(cand, prm.max_contacts)


def _cap(c: _Cand, kmax):
    """Keep at most kmax contacts per batch member (smallest gaps first)."""
    if not kmax or len(c.bi) == 0 or np.bincount(c.bi).max() <= kmax:
        return c
    o = np.lexsort((c.phi, c.bi))
    starts = np.r_[0, np.flatnonzero(np.diff(c.bi[o])) + 1]
    rank = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
    idx = np.sort(o[rank < kmax])
    return _Cand(*(getattr(c, f)[idx] for f in _Cand.__dataclass_fields__))


def _reduce(c: _Cand, m):
    """Keep at most m contacts per (batch, feature group, collider), feature group = one rod
    capsule / one leaf / one fruit: the deepest, then extremes along the contact tangents
    (a rigid patch is represented by its hull corners)."""
    if m is None or m <= 0 or len(c.bi) == 0:
        return c
    g = (c.bi.astype(np.int64) * (c.grp.max() + 1) + c.grp) * (c.col.max() + 1) + c.col
    cnt = np.bincount(np.unique(g, return_inverse=True)[1])
    if cnt.max() <= m:
        return c
    t1, t2 = _tangents(c.n)
    scores = [-c.phi, (c.p * t1).sum(-1), -(c.p * t1).sum(-1), (c.p * t2).sum(-1), -(c.p * t2).sum(-1),
              (c.p * (t1 + t2)).sum(-1), -(c.p * (t1 + t2)).sum(-1), (c.p * (t1 - t2)).sum(-1), -(c.p * (t1 - t2)).sum(-1)]
    keep = np.zeros(len(g), bool)
    for sc in scores[:max(1, m)]:
        o = np.lexsort((-sc, g))
        first = np.r_[True, g[o][1:] != g[o][:-1]]
        keep[o[first]] = True
    idx = np.flatnonzero(keep)
    return _Cand(*(getattr(c, f)[idx] for f in _Cand.__dataclass_fields__))


# ---------------------------------------------------------------- result
@dataclass
class ContactResult:
    valid: np.ndarray       # (B, K) bool
    body: np.ndarray        # (B, K)
    point: np.ndarray       # (B, K, 3)
    normal: np.ndarray      # (B, K, 3)
    phi: np.ndarray         # (B, K) gap at the start of the step
    lam: np.ndarray         # (B, K, 3) impulses in (n, t1, t2)
    force: np.ndarray       # (B, K, 3) world contact force on the plant [N] (= impulse / h)
    collider: np.ndarray    # (B, K)
    key: np.ndarray         # (B, K)
    iterations: int
    residual: float
    W: np.ndarray | None = None
    v_free: np.ndarray | None = None
    v_after: np.ndarray | None = None
    alpha: np.ndarray | None = None   # (B,) impulse scaling of the velocity safeguard (1 = none)
    timing: dict = field(default_factory=dict)

    @property
    def count(self):
        return self.valid.sum(1)

    def total_force(self, collider=None):
        """(B, 3) summed world force on the plant (optionally of one collider index)."""
        m = self.valid if collider is None else self.valid & (self.collider == collider)
        return (self.force * m[..., None]).sum(1)

    def normal_force(self):
        return np.where(self.valid, self.lam[..., 0], 0.0)


# ---------------------------------------------------------------- Delassus operator
def _pack(c: _Cand, B):
    cnt = np.bincount(c.bi, minlength=B)
    K = max(int(cnt.max()) if len(cnt) else 0, 1)
    o = np.argsort(c.bi, kind="stable")
    starts = np.r_[0, np.cumsum(cnt)[:-1]]
    slot = np.arange(len(o)) - starts[c.bi[o]]
    valid = np.zeros((B, K), bool)
    valid[c.bi[o], slot] = True

    def pk(x, fill=0):
        a = np.full((B, K) + x.shape[1:], fill, dtype=x.dtype)
        a[c.bi[o], slot] = x[o]
        return a
    return valid, pk


def delassus(sim, fac, body, p, D, valid):
    """W (B, 3K, 3K): velocity change of D_c . v(p_c) per unit impulse along D_c' at p_c',
    for the factorised implicit system of the current step. O(K depth + K^2)."""
    T = sim.topo
    B, K = body.shape
    M = B * K
    bix = np.repeat(np.arange(B), K)
    bod = body.reshape(M)
    val = valid.reshape(M)
    pp = p.reshape(M, 3)
    Dm = D.reshape(M, 3, 3)                                   # rows n, t1, t2
    dep = np.where(val, T.depth[bod], -1)
    smax = int(dep.max())
    # test force columns (bias = -force): pA = -[ (p - o_b) x d ; d ]
    rb = pp - fac.o[bix, bod]
    d_cols = np.swapaxes(Dm, -1, -2)                          # (M, 3, 3) columns d_i
    pA = np.zeros((M, 6, 3))
    pA[:, :3] = -np.swapaxes(_cross(rb[:, None, :], Dm), -1, -2)
    pA[:, 3:] = -d_cols
    UL = np.zeros((M, smax + 1, 3, 3))
    PA0 = np.zeros((M, 6, 3))
    for s in range(smax, -1, -1):
        idx = np.flatnonzero(dep >= s)
        j = T.anc[bod[idx], s]
        bb = bix[idx]
        R = fac.R[bb, j]
        ul = -np.swapaxes(R, -1, -2) @ pA[idx, :3]
        UL[idx, s] = ul
        if s == 0:
            PA0[idx] = pA[idx]
            break
        pa = pA[idx] + fac.UD[bb, j] @ ul
        pa[:, :3] += np.swapaxes(_cross(fac.r[bb, j][:, None, :], np.swapaxes(pa[:, 3:], -1, -2)), -1, -2)
        pA[idx] = pa
    # down sweep along each path: modified accelerations a and plain world-origin velocities Vp
    A = np.zeros((M, smax + 1, 6, 3))
    VP = np.zeros((M, smax + 1, 6, 3))
    fpos = -np.ones(T.nb, int)
    fpos[T.float_idx] = np.arange(len(T.float_idx))
    for s in range(smax + 1):
        idx = np.flatnonzero(dep >= s)
        j = T.anc[bod[idx], s]
        bb = bix[idx]
        R, oj = fac.R[bb, j], fac.o[bb, j]
        if s == 0:
            ud = fac.Dinv[bb, j] @ UL[idx, 0]
            a = fac.Sm[bb, j] @ ud
            wv = R @ ud
            vp = np.concatenate([wv, np.swapaxes(_cross(oj[:, None, :], np.swapaxes(wv, -1, -2)), -1, -2)], 1)
            fm = T.floating[j]
            if fm.any():
                af = -fac.IAinv_f[bb[fm], fpos[j[fm]]] @ PA0[idx[fm]]
                a[fm] = af
                vp[fm] = np.concatenate([af[:, :3], af[:, 3:] + np.swapaxes(
                    _cross(oj[fm][:, None, :], np.swapaxes(af[:, :3], -1, -2)), -1, -2)], 1)
        else:
            ap = fac.Xm[bb, j] @ A[idx, s - 1]
            ud = fac.Dinv[bb, j] @ (UL[idx, s] - fac.W[bb, j] @ ap)
            a = ap + fac.Sm[bb, j] @ ud
            wv = R @ ud
            vp = VP[idx, s - 1] + np.concatenate(
                [wv, np.swapaxes(_cross(oj[:, None, :], np.swapaxes(wv, -1, -2)), -1, -2)], 1)
        A[idx, s] = a
        VP[idx, s] = vp
    # contact rows about the world origin: Gw = [ (p x d)^T, d^T ]
    Gw = np.concatenate([_cross(pp[:, None, :], Dm), Dm], -1)  # (M, 3, 6)
    # pulled-back rows rho(e): off-path propagation from depth e to the contact body
    RHO = np.zeros((M, smax + 1, 3, 6))
    if smax >= 1:
        need = np.unique(np.concatenate([np.stack([bix[dep >= s], T.anc[bod[dep >= s], s]], 1)
                                         for s in range(1, smax + 1)]), axis=0)
        Psi = np.zeros((B, T.nb, 6, 6))
        Lam = np.zeros((B, T.nb, 6, 6))
        bb, j = need[:, 0], need[:, 1]
        DW = fac.Dinv[bb, j] @ fac.W[bb, j]                    # (n, 3, 6)
        Xm = fac.Xm[bb, j]
        Psi[bb, j] = (EYE6 - fac.Sm[bb, j] @ DW) @ Xm
        R, oj = fac.R[bb, j], fac.o[bb, j]
        Sh = np.concatenate([R, _cross(oj[:, None, :], np.swapaxes(R, -1, -2)).swapaxes(-1, -2)], 1)  # [R; [o]x R]
        Lam[bb, j] = Sh @ DW @ Xm
        for s in range(smax, 0, -1):
            idx = np.flatnonzero(dep >= s)
            j = T.anc[bod[idx], s]
            bb = bix[idx]
            RHO[idx, s - 1] = -Gw[idx] @ Lam[bb, j] + RHO[idx, s] @ Psi[bb, j]
    # LCA depth for every pair (per batch) and the k x k blocks
    ancC = np.where(val[:, None], T.anc[bod][:, :smax + 1], -1).reshape(B, K, smax + 1)
    cid = np.arange(M).reshape(B, K)
    Gw4 = Gw.reshape(B, K, 3, 6)
    W = np.zeros((B, K, 3, K, 3))
    # rows in chunks: memory O(B * chunk * K) instead of O(B * K^2) per gathered (6 x 3) block
    chunk = max(1, int(2 ** 22 // max(1, B * K * 18)))
    for r0 in range(0, K, chunk):
        r1 = min(K, r0 + chunk)
        eq = (ancC[:, r0:r1, None, :] == ancC[:, None, :, :]) & (ancC[:, r0:r1, None, :] >= 0)
        e = eq.sum(-1) - 1                                     # (B, rows, K) LCA depth
        ee = np.maximum(e, 0)
        ci = np.broadcast_to(cid[:, r0:r1, None], e.shape)
        cj = np.broadcast_to(cid[:, None, :], e.shape)
        Wb = np.einsum("bcij,bcdjk->bcdik", Gw4[:, r0:r1], VP[cj, ee]) \
            + np.einsum("bcdij,bcdjk->bcdik", RHO[ci, ee], A[cj, ee])
        W[:, r0:r1] = np.swapaxes(np.where((e >= 0)[..., None, None], Wb, 0.0), 2, 3)
    return W.reshape(B, 3 * K, 3 * K)


def point_velocity(sim, u, v_float, body, p):
    """(B, K, 3) world velocity of points p on bodies `body` for joint velocities u (plain
    kinematics at the current configuration)."""
    k = sim._kinematics(sim.Q, u, sim.p_float, v_float)
    bi = np.arange(body.shape[0])[:, None]
    return k.vO[bi, body] + _cross(k.w[bi, body], p - k.o[bi, body])


# ---------------------------------------------------------------- PGS
def _disk_qp(Wt, Wti, e, V, b, cap, newton=12):
    """Local tangential problem of one contact (others fixed): v_t = Wt lam + b,
    minimise 1/2 lam^T Wt lam + b^T lam over |lam| <= cap (Wt replaced by its symmetric part
    on the boundary). Interior: lam = -Wt^-1 b (stick, v_t = 0). Boundary: KKT gives
    v_t = -gamma lam, gamma >= 0, i.e. friction exactly opposes slip (maximal dissipation);
    lam = -(Ws + gamma I)^-1 b with |lam| = cap, gamma from the secular equation by Newton on
    1/|lam(gamma)| - 1/cap (More-Sorensen), using Ws = V diag(e) V^T. Batched over (B,)."""
    lam = -np.einsum("bij,bj->bi", Wti, b)
    nl = np.linalg.norm(lam, axis=-1)
    out = nl > cap
    if not out.any():
        return lam
    beta = np.einsum("bji,bj->bi", V, b)                     # V^T b
    capc = np.maximum(cap, 1e-300)
    # start from the isotropic estimate gamma0 = |b| / cap - e_max (>= 0)
    g = np.maximum(np.linalg.norm(b, axis=-1) / capc - e.max(-1), 0.0)
    for _ in range(newton):
        de = e + g[:, None]
        q = beta / de
        nq = np.sqrt((q * q).sum(-1)) + 1e-300
        dn = -(q * q / de).sum(-1) / nq                      # d|q|/dgamma
        phi = 1.0 / nq - 1.0 / capc
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            dphi = -dn / (nq * nq)
            stepg = np.where(np.abs(dphi) > 1e-300, phi / np.where(np.abs(dphi) > 1e-300, dphi, 1.0), 0.0)
        g = np.maximum(g - np.nan_to_num(stepg), 0.0)
    q = -beta / (e + g[:, None])
    lb = np.einsum("bij,bj->bi", V, q)
    nb = np.linalg.norm(lb, axis=-1)
    lb = lb * (cap / np.maximum(nb, 1e-300))[:, None]          # exact radius (round-off)
    lb = np.where((cap > 0)[:, None], lb, 0.0)
    return np.where(out[:, None], lb, lam)


def impulse_cap(W, v_free, target, valid, factor=100.0):
    """(B, K) safeguard bound on the normal impulses: factor x the impulse that changes the
    contact's own normal velocity by |v_free,n| + |target| + 1 mm/s. Generous for feasible
    problems; it keeps PGS bounded when the constraints are inconsistent or it fails to converge."""
    B, K = valid.shape
    Wnn = np.einsum("bii->bi", W)[:, 0::3]
    vn = np.abs(v_free.reshape(B, K, 3)[..., 0])
    cap = factor * (vn + np.abs(target) + 1e-3) / np.maximum(Wnn, 1e-300)
    return np.where(valid, cap, 0.0)


def pgs(W, v_free, target, mu, valid, lam0=None, iters=200, tol=1e-6, sor=1.0, backend="auto", lam_max=None):
    """Dispatch: the Warp kernel (contact_wp) or the numpy reference (pgs_numpy).
    lam_max (B, K): optional upper bound on the normal impulses (see impulse_cap)."""
    if lam_max is None:
        lam_max = np.full(valid.shape, np.inf)
    if backend in ("auto", "warp"):
        from . import contact_wp
        if contact_wp.available():
            return contact_wp.pgs_warp(W, v_free, target, mu, valid, lam0, iters, tol, sor, lam_max=lam_max)
        if backend == "warp":
            raise RuntimeError("warp is not available")
    return pgs_numpy(W, v_free, target, mu, valid, lam0, iters, tol, sor, lam_max)


def pgs_numpy(W, v_free, target, mu, valid, lam0=None, iters=200, tol=1e-6, sor=1.0, lam_max=None):
    """Projected Gauss-Seidel for the Coulomb contact problem, batched over B:
        v = v_free + W lam;  0 <= lam_n  _|_  v_n - target_n >= 0;
        |lam_t| <= mu lam_n, lam_t opposes v_t (stick: v_t = 0)   (isotropic cone)
    W (B, 3K, 3K), v_free (B, 3K), target (B, K), mu (B, K), valid (B, K).
    Stops when a full sweep changes no contact velocity by more than tol (|d lam| W_ii).
    Returns (lam (B, K, 3), iterations, last max velocity change [m/s])."""
    B, K = valid.shape
    lam = np.zeros((B, K, 3)) if lam0 is None else np.where(valid[..., None], lam0, 0.0).copy()
    lam = lam.reshape(B, 3 * K)
    v = v_free + np.einsum("bij,bj->bi", W, lam)
    diag = np.einsum("bii->bi", W).reshape(B, K, 3)
    Wnn = np.where(valid, diag[..., 0], 1.0)
    Wnn = np.where(Wnn > 1e-300, Wnn, 1.0)
    Wt = np.stack([np.stack([W[:, 3 * np.arange(K) + 1, 3 * np.arange(K) + 1], W[:, 3 * np.arange(K) + 1, 3 * np.arange(K) + 2]], -1),
                   np.stack([W[:, 3 * np.arange(K) + 2, 3 * np.arange(K) + 1], W[:, 3 * np.arange(K) + 2, 3 * np.arange(K) + 2]], -1)], -2)
    Wt = np.where(valid[..., None, None], Wt, np.eye(2))
    det = Wt[..., 0, 0] * Wt[..., 1, 1] - Wt[..., 0, 1] * Wt[..., 1, 0]
    det = np.where(np.abs(det) > 1e-300, det, 1.0)
    Wti = np.stack([np.stack([Wt[..., 1, 1], -Wt[..., 0, 1]], -1), np.stack([-Wt[..., 1, 0], Wt[..., 0, 0]], -1)], -2) / det[..., None, None]
    Ws = 0.5 * (Wt + np.swapaxes(Wt, -1, -2))
    Ws_e, Ws_V = np.linalg.eigh(Ws)                           # (B, K, 2), (B, K, 2, 2)
    Ws_e = np.maximum(Ws_e, 1e-300)
    WT = np.swapaxes(W, 1, 2).copy()                          # columns contiguous
    it, rel = 0, 0.0
    for it in range(1, iters + 1):
        dmax = np.zeros(B)
        for c in range(K):
            vm = valid[:, c]
            if not vm.any():
                continue
            i = 3 * c
            ln = lam[:, i]
            lnn = np.maximum(ln + sor * (target[:, c] - v[:, i]) / Wnn[:, c], 0.0)
            if lam_max is not None:
                lnn = np.minimum(lnn, lam_max[:, c])
            lnn = np.where(vm, lnn, 0.0)
            d = lnn - ln
            lam[:, i] = lnn
            v += WT[:, i] * d[:, None]
            lt = lam[:, i + 1:i + 3]
            cap = mu[:, c] * lnn
            ltn = _disk_qp(Wt[:, c], Wti[:, c], Ws_e[:, c], Ws_V[:, c],
                           v[:, i + 1:i + 3] - np.einsum("bij,bj->bi", Wt[:, c], lt), cap)
            ltn = np.where(vm[:, None], lt + sor * (ltn - lt), 0.0)
            dt = ltn - lt
            lam[:, i + 1:i + 3] = ltn
            v += np.einsum("bij,bj->bi", WT[:, i + 1:i + 3].swapaxes(1, 2), dt)
            dmax = np.maximum(dmax, np.maximum(np.abs(d) * Wnn[:, c],
                                               np.abs(dt).max(-1) * np.abs(Wt[:, c]).max((-1, -2))))
        rel = float(dmax.max())
        if rel <= tol:
            break
    return lam.reshape(B, K, 3), it, rel


# ---------------------------------------------------------------- step hook
def solve_contacts(sim, k, fac, h, udot_free, acc_free, colliders):
    """Called from TreeSim.step after the contact-free solve. Returns the change
    (d_udot, d_acc) caused by the contact impulses and stores sim.contact."""
    import time
    t0 = time.perf_counter()
    T, B = sim.topo, sim.B
    cm, prm = _contact_model(sim)
    cand = detect(sim, k, colliders, h, prm)
    zero = (np.zeros_like(udot_free), np.zeros_like(acc_free))
    if cand is None or len(cand.bi) == 0:
        sim.contact = None
        sim._contact_cache = None
        return zero
    valid, pk = _pack(cand, B)
    body, p, n, phi = pk(cand.body), pk(cand.p), pk(cand.n), pk(cand.phi, 1.0)
    key, vc, mu, col = pk(cand.key, -1), pk(cand.vc), pk(cand.mu), pk(cand.col, -1)
    t1, t2 = _tangents(np.where(valid[..., None], n, np.array([0.0, 0.0, 1.0])))
    D = np.stack([n, t1, t2], -2)                              # (B, K, 3, 3)
    t_det = time.perf_counter()
    W = delassus(sim, fac, body, p, D, valid)
    t_del = time.perf_counter()
    fl = T.float_idx
    u_free = sim.u + h * udot_free
    vf_free = sim.v_float + h * acc_free[:, fl, 3:] if len(fl) else sim.v_float
    vrel = point_velocity(sim, u_free, vf_free, body, p) - vc
    v_free = np.einsum("bkij,bkj->bki", D, vrel).reshape(B, -1)
    # normal targets: speculative (phi > 0) or capped Baumgarte recovery (phi < 0)
    target = np.where(phi >= 0, -phi / h, np.minimum(-prm.erp * phi / h, prm.max_recovery))
    # a body deep inside a sphere / capsule has contacts with opposing normals: separating all of
    # them at once can be infeasible, so deep ones only recover slowly (half-spaces are always
    # consistent: every normal is the same)
    planar = np.array([isinstance(c, GroundPlane) for c in colliders] + [True])[col]
    deep = (phi < -prm.max_depth_recovery) & ~planar
    target = np.where(deep, np.minimum(target, prm.deep_recovery), target)
    target = np.where(valid, target, 0.0)
    # warm start by key (impulse stored as a world vector, re-projected on the new frame)
    lam0 = None
    cache = sim._contact_cache
    if prm.warm_start and cache is not None:
        lam0 = np.zeros((B, valid.shape[1], 3))
        for b in range(B):
            kb, lw = cache
            kk, ll = kb[b], lw[b]
            m = kk >= 0
            if not m.any():
                continue
            kk, ll = kk[m], ll[m]
            o = np.argsort(kk)
            pos = np.searchsorted(kk[o], key[b])
            pos = np.minimum(pos, len(kk) - 1)
            hit = (kk[o][pos] == key[b]) & valid[b]
            lam0[b, hit] = np.einsum("kij,kj->ki", D[b, hit], ll[o][pos][hit])
    lam_max = impulse_cap(W, v_free, target, valid, prm.impulse_cap) if prm.impulse_cap else None
    lam, iters, rel = pgs(W, v_free, target, mu, valid, lam0, prm.iters, prm.tol, prm.sor, prm.backend, lam_max)
    lam, alpha = _limit_velocity(W, v_free, target, lam, valid)
    t_pgs = time.perf_counter()
    lam_w = np.einsum("bkji,bkj->bki", D, lam)                 # world impulse vectors
    sim._contact_cache = (np.where(valid, key, -1), lam_w)
    # back-substitution: impulses as body forces through the same factorisation
    f = lam_w / h
    bi = np.broadcast_to(np.arange(B)[:, None], body.shape)
    pA = np.zeros((B, T.nb, 6))
    m = valid
    rr = p - fac.o[bi, body]
    np.add.at(pA, (bi[m], body[m]), -np.concatenate([_cross(rr, f), f], -1)[m])
    d_udot, d_acc = sim.response(pA, fac)
    t_end = time.perf_counter()
    v_after = (v_free + np.einsum("bij,bj->bi", W, lam.reshape(B, -1))).reshape(B, -1, 3)
    sim.contact = ContactResult(valid, body, p, n, phi, lam, f * valid[..., None], col, key, iters, rel,
                                W=W, v_free=v_free.reshape(B, -1, 3), v_after=v_after,
                                alpha=alpha,
                                timing={"detect": t_det - t0, "delassus": t_del - t_det,
                                        "pgs": t_pgs - t_del, "backsub": t_end - t_pgs, "total": t_end - t0})
    return d_udot, d_acc


def _limit_velocity(W, v_free, target, lam, valid):
    """Step-level safeguard for non-converged / inconsistent problems: scale the impulses by
    alpha in (0, 1] so that no contact velocity exceeds max(|v_free,i|, Vmax), with
    Vmax = 2 (max |v_free| + max target) + 1 cm/s per batch member (v is linear in alpha).
    A converged, consistent solve has alpha = 1."""
    B, K = valid.shape
    lamf = lam.reshape(B, -1)
    a = v_free
    bv = np.einsum("bij,bj->bi", W, lamf)
    m3 = np.repeat(valid, 3, axis=1)
    vmax = 2.0 * (np.where(m3, np.abs(a), 0).max(1) + np.where(valid, target, 0).max(1)) + 0.01
    M = np.maximum(np.abs(a), vmax[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        al = np.where(np.abs(bv) > 1e-300, (M - a * np.sign(bv)) / np.abs(bv), np.inf)
    al = np.where(m3, al, np.inf).min(1)
    alpha = np.clip(al, 0.0, 1.0)
    return lam * alpha[:, None, None], alpha


def delassus_bruteforce(sim, fac, body, p, D, valid):
    """Reference W by 3K column solves (each one O(n) response); for tests."""
    T, B = sim.topo, sim.B
    K = body.shape[1]
    W = np.zeros((B, 3 * K, 3 * K))
    bi = np.arange(B)
    fl = T.float_idx
    for c in range(K):
        for i in range(3):
            pA = np.zeros((B, T.nb, 6))
            d = D[:, c, i]
            rr = p[:, c] - fac.o[bi, body[:, c]]
            pA[bi, body[:, c]] = -np.concatenate([_cross(rr, d), d], -1)
            ud, acc = sim.response(pA, fac)
            vf = acc[:, fl, 3:] if len(fl) else sim.v_float * 0
            v = point_velocity(sim, ud, vf, body, p)
            W[:, :, 3 * c + i] = np.einsum("bkij,bkj->bki", D, v).reshape(B, -1)
    m = np.repeat(valid, 3, axis=1)
    return W * m[:, :, None] * m[:, None, :]


# ---------------------------------------------------------------- penalty baseline
def apply_penalty(sim, k, colliders, h, Ieff, fext):
    """B1-style implicit penalty springs (normal only) for colliders with mode='penalty'."""
    for c in colliders:
        cand = detect(sim, k, [c], h, ContactParams(margin=0.0, spec_factor=0.0, max_per_body=0))
        if cand is None:
            continue
        m = cand.phi < 0
        if not m.any():
            continue
        bi, body, p, n, pen = cand.bi[m], cand.body[m], cand.p[m], cand.n[m], -cand.phi[m]
        rr = p - k.o[bi, body]
        g6 = np.concatenate([_cross(rr, n), n], -1)
        v6 = np.concatenate([k.w[bi, body], k.vO[bi, body]], -1)
        vn = (g6 * v6).sum(-1) - (cand.vc[m] * n).sum(-1)
        fmag = c.stiffness * pen - (h * c.stiffness + c.damping) * vn
        np.add.at(fext, (bi, body), fmag[:, None] * g6)
        np.add.at(Ieff, (bi, body), (h * h * c.stiffness + h * c.damping) * g6[:, :, None] * g6[:, None, :])
