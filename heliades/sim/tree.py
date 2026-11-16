"""Batched O(n) implicit articulated-rod solver for plant trees.

Formulation (see docs/sim/DESIGN.md):
  * generalized coordinates: one unit quaternion Q_i per spherical joint, the deviation
    from the rest relative rotation:  R_rel,i = R0rel,i R(Q_i)  (child frame, z = tangent);
  * generalized velocities: u_i = child-frame relative angular velocity, Q_dot = Q (0,u)/2;
  * potential: U = 2 v^T K v with v = vec(Q) = sin(theta/2) axis, K = diag(EI/l*, EI/l*, GJ/l*)
    (= 1/2 phi^T K phi + O(theta^4); smooth on SO(3), so joints never wind up);
  * linearly-implicit (backward) Euler in the spring and damping terms:
        (M + h C + h^2 K~) a = tau(Q_n) - c(q, v_n) - (h K~ + C) u_n + f_ext,
    tau = -2 Gamma K v, Gamma = w I - [v]x,  K~ = Gamma K Gamma^T (joint-local 3x3, SPD),
    solved exactly by the articulated-body algorithm with h C + h^2 K~ as "armature";
  * body-level linear forces (mass-proportional damping, penalty contact springs) are
    also implicit: their 6x6 Jacobians are added to the body inertia before the
    leaf-to-root sweep.

All arrays carry a leading batch dim B (parameter sets in lock-step). The three
sweeps loop over depth levels; inside a level every op is vectorised over (B, bodies).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from planttwin.asset import PhysicsParams, PlantAsset

from .rotation import qmul, quat_from_matrix, quat_from_rotvec, quat_to_matrix, rotvec_from_quat, skew
from .topology import TreeTopology

GRAVITY = np.array([0.0, 0.0, -9.81])


# ---------------------------------------------------------------- external actions
@dataclass
class PointForce:
    """World-frame force on a stem node. force: (3,) or (B, 3)."""
    node: int
    force: np.ndarray


@dataclass
class BodyForce:
    """B3: world-frame dead force at a point fixed on a body: `local` (3,) in the body frame,
    relative to the body origin (proximal joint). force: (3,) or (B, 3). Used for known
    support forces (contact-aware inverse statics) at arbitrary points of a rod capsule."""
    body: int
    local: np.ndarray
    force: np.ndarray


@dataclass
class SphereCollider:
    """Kinematic sphere pushing on rod nodes and leaf vertices with an implicit penalty
    spring (stiffness N/m, normal damping N s/m). center: (3,) or (B, 3)."""
    center: np.ndarray
    radius: float
    stiffness: float = 2e3
    damping: float = 2.0
    velocity: np.ndarray | None = None


@dataclass
class SimState:
    t: float
    Q: np.ndarray       # (B, nb, 4) joint deviation quaternions (from rest)
    u: np.ndarray       # (B, nb, 3) joint relative angular velocities (child frame)
    x: np.ndarray       # (B, N, 3) rod node positions (asset node order)
    R: np.ndarray       # (B, nb, 3, 3) body (segment) world rotations
    v_nodes: np.ndarray  # (B, N, 3) rod node velocities

    @property
    def quat(self):
        return quat_from_matrix(self.R)

    @property
    def phi(self):
        """(B, nb, 3) joint rotation vectors from rest (diagnostic)."""
        return rotvec_from_quat(self.Q)


@dataclass
class _Factor:
    """Per-step factorisation of the implicit system (kept for contact solves, B3)."""
    UD: np.ndarray      # (B, nb, 6, 3) IA S' D^-1
    W: np.ndarray       # (B, nb, 3, 6) S^T IA
    Dinv: np.ndarray    # (B, nb, 3, 3)
    Xm: np.ndarray      # (B, nb, 6, 6) modified motion map from the parent X'
    Sm: np.ndarray      # (B, nb, 6, 3) modified joint map S'
    R: np.ndarray       # (B, nb, 3, 3) body rotations (S = [R; 0])
    r: np.ndarray       # (B, nb, 3) o_i - o_parent
    o: np.ndarray       # (B, nb, 3) body origins
    IAinv_f: np.ndarray | None   # (B, nf, 6, 6) inverse articulated inertia of floating bases


@dataclass
class _Kin:
    R: np.ndarray
    o: np.ndarray
    tip: np.ndarray
    w: np.ndarray
    vO: np.ndarray
    r: np.ndarray       # o_i - o_parent
    c: np.ndarray       # (B, nb, 6) velocity-product acceleration terms


def _cross(a, b):
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def _mv(M, v):
    return np.einsum("...ij,...j->...i", M, v)


def _mtv(M, v):
    return np.einsum("...ji,...j->...i", M, v)


def _X(r):
    """Motion transform parent-origin -> child-origin (world orientation): [[I,0],[-r^,I]]."""
    sh = r.shape[:-1]
    X = np.zeros(sh + (6, 6))
    X[..., :3, :3] = np.eye(3)
    X[..., 3:, 3:] = np.eye(3)
    X[..., 3:, :3] = -skew(r)
    return X


def _gravity_stiffness_psd(g, mc):
    """PSD part of K = (g.mc) I - (g mc^T + mc g^T)/2 (closed-form eigen-split).
    Eigen: n = g x mc -> g.mc ; d- = g^ + c^ -> (g.mc - |g||mc|)/2 <= 0 ;
    d+ = g^ - c^ -> (g.mc + |g||mc|)/2 >= 0."""
    gn = np.linalg.norm(g)
    if gn == 0:
        return np.zeros(mc.shape + (3,))
    f = g / gn
    cm = np.linalg.norm(mc, axis=-1)
    e = mc / np.maximum(cm, 1e-300)[..., None]
    gc = (mc @ g)[..., None, None]
    K = gc * np.eye(3) - 0.5 * (f[:, None] * mc[..., None, :] + mc[..., :, None] * f[None, :]) * gn
    dm = f + e
    sm = np.linalg.norm(dm, axis=-1)
    dm = dm / np.maximum(sm, 1e-300)[..., None]
    lam_m = 0.5 * (gc[..., 0, 0] - gn * cm)
    nv = np.cross(f, e)
    sn = np.linalg.norm(nv, axis=-1)
    nv = nv / np.maximum(sn, 1e-300)[..., None]
    Kp = K - lam_m[..., None, None] * dm[..., :, None] * dm[..., None, :] \
        - np.minimum(gc, 0.0) * nv[..., :, None] * nv[..., None, :]
    # near-upright (g^ ~ -c^): every eigenvalue is <= ~0 -> no stabilising part
    Kp = np.where((sm < 1e-6)[..., None, None] | (cm < 1e-300)[..., None, None], 0.0, Kp)
    return Kp


class TreeSim:
    """Batched implicit reduced-coordinate rod-tree simulator.

    TreeSim(asset, params_list) with params_list a PhysicsParams or a list of B of them.
    Per batch: stem_youngs_modulus, stem_poisson, stem_density, leaf_areal_density, damping.
    joint_damping: Rayleigh stiffness-proportional damping beta [s] (shared).
    """

    def __init__(self, asset: PlantAsset, params_list: PhysicsParams | Sequence[PhysicsParams] | None = None,
                 *, gravity=GRAVITY, joint_damping: float = 0.0, include_leaves: bool = True,
                 implicit_gyro: bool = True, implicit_gravity: bool = True, colliders=None,
                 contact_params=None):
        self.asset = asset
        self.topo = T = TreeTopology(asset, include_leaves=include_leaves)
        # B3: persistent colliders (heliades.sim.contact), applied at every step in addition
        # to the ones passed in `external`; contact_params: contact.ContactParams
        self.colliders = list(colliders or [])
        self.contact_params = contact_params
        self.contact = None                   # last step's contact.ContactResult (B3)
        self._contact_cache = None
        self.gravity = np.asarray(gravity, float)
        self.joint_damping = float(joint_damping)
        self.implicit_gyro = bool(implicit_gyro)
        self.implicit_gravity = bool(implicit_gravity)
        # contact sample points: distal rod nodes + leaf vertices (host body, local coords)
        cb = [np.arange(T.nb)]
        cl = [np.stack([np.zeros(T.nb), np.zeros(T.nb), T.l], 1)]
        for L in T.leaves:
            if L.body >= 0:
                cb.append(np.full(len(L.local), L.body))
                cl.append(L.local)
        self.cp_body = np.concatenate(cb).astype(int)
        self.cp_local = np.concatenate(cl)
        self.R0rel = T.R0rel[None]            # (1 or B, nb, 3, 3) rest relative rotations
        self.set_params(params_list if params_list is not None else asset.physics)
        self.reset()

    def set_rest(self, R0rel=None):
        """Replace the joint rest relative rotations (B2 inverse statics). R0rel: (nb, 3, 3)
        shared or (B, nb, 3, 3) per batch element; None restores the asset's rest shape.
        Body frames, payload geometry and stiffness are unchanged; only the zero of each
        joint spring moves. Keeps Q (the deviation from rest), so call reset() as needed."""
        R = self.topo.R0rel if R0rel is None else np.asarray(R0rel, float)
        self.R0rel = R[None] if R.ndim == 3 else R
        self._kin_cache = None

    # ------------------------------------------------------------------ params
    def set_params(self, params):
        plist = [params] if isinstance(params, PhysicsParams) else list(params)
        self.params, self.B = plist, len(plist)
        T = self.topo
        col = lambda k: np.array([getattr(p, k) for p in plist], float)[:, None]
        E, nu = col("stem_youngs_modulus"), col("stem_poisson")
        rho, rhol = col("stem_density"), col("leaf_areal_density")
        G = E / (2 * (1 + nu))
        self.kb = E * T.kgeom_bend[None]                   # (B, nb)
        self.kt = G * T.kgeom_twist[None]
        self.K = np.stack([self.kb, self.kb, self.kt], -1)  # (B, nb, 3)
        sp, lp, pp = T.seg_part, T.leaf_part, T.point_part
        self.m = rho * sp.m + rhol * lp.m + pp.m            # (B, nb)
        self.h1 = rho[..., None] * sp.h + rhol[..., None] * lp.h + pp.h
        self.Io = rho[..., None, None] * sp.Io + rhol[..., None, None] * lp.Io + pp.Io
        self.alpha = col("damping")[:, 0]                   # mass-proportional damping [1/s]

    def reset(self, Q=None, u=None, p_float=None, v_float=None):
        """Q, u: joint state (default rest, at rest). p_float, v_float (B, nf, 3): origin
        position / linear velocity of floating-base bodies (default asset position, 0)."""
        T = self.topo
        if Q is None:
            Q = np.zeros((self.B, T.nb, 4))
            Q[..., 0] = 1.0
        self.Q = np.array(Q, float)
        self.u = np.zeros((self.B, T.nb, 3)) if u is None else np.array(u, float)
        nf = len(T.float_idx)
        self.p_float = np.broadcast_to(T.o0_ground[T.float_idx], (self.B, nf, 3)).copy() \
            if p_float is None else np.broadcast_to(np.asarray(p_float, float), (self.B, nf, 3)).copy()
        self.v_float = np.zeros((self.B, nf, 3)) if v_float is None else \
            np.broadcast_to(np.asarray(v_float, float), (self.B, nf, 3)).copy()
        self.t = 0.0
        self._kin_cache = None
        self._contact_cache = None

    # ------------------------------------------------------------------ kinematics
    def _kinematics(self, Q, u, p_float=None, v_float=None) -> _Kin:
        T, B = self.topo, Q.shape[0]
        fl = T.float_idx
        p_float = getattr(self, "p_float", None) if p_float is None else p_float
        v_float = getattr(self, "v_float", None) if v_float is None else v_float
        Rrel = self.R0rel @ quat_to_matrix(Q)
        R = np.empty_like(Rrel)
        o = np.empty((B, T.nb, 3))
        tip = np.empty_like(o)
        w = np.empty_like(o)
        vO = np.empty_like(o)
        r = np.zeros_like(o)
        c = np.zeros((B, T.nb, 6))
        for d in range(T.nlev):
            sl = slice(T.lvl[d], T.lvl[d + 1])
            if d == 0:
                R[:, sl] = Rrel[:, sl]
                o[:, sl] = T.o0_ground[sl]
                w[:, sl] = _mv(R[:, sl], u[:, sl])
                vO[:, sl] = 0.0
                if len(fl):
                    o[:, fl] = p_float
                    vO[:, fl] = v_float
            else:
                p = T.parent[sl]
                R[:, sl] = R[:, p] @ Rrel[:, sl]
                o[:, sl] = tip[:, p]
                rr = o[:, sl] - o[:, p]
                r[:, sl] = rr
                wp = w[:, p]
                wrel = _mv(R[:, sl], u[:, sl])
                w[:, sl] = wp + wrel
                vO[:, sl] = vO[:, p] + _cross(wp, rr)
                c[:, sl, :3] = _cross(wp, wrel)
                c[:, sl, 3:] = _cross(wp, _cross(wp, rr))
            tip[:, sl] = o[:, sl] + T.l[sl, None] * R[:, sl, :, 2]
        return _Kin(R, o, tip, w, vO, r, c)

    def _kin(self):
        if self._kin_cache is None:
            self._kin_cache = self._kinematics(self.Q, self.u)
        return self._kin_cache

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> SimState:
        k = self._kin()
        T = self.topo
        x = np.broadcast_to(T.X0, (self.B, T.N, 3)).copy()
        x[:, T.node_b] = k.tip
        v = np.zeros_like(x)
        v[:, T.node_b] = k.vO + _cross(k.w, k.tip - k.o)
        if len(T.float_idx):
            x[:, T.node_a[T.float_idx]] = k.o[:, T.float_idx]
            v[:, T.node_a[T.float_idx]] = k.vO[:, T.float_idx]
        return SimState(self.t, self.Q.copy(), self.u.copy(), x, k.R.copy(), v)

    def leaf_vertices(self) -> list[np.ndarray]:
        """World vertices of every asset leaf, (B, V, 3) each (asset leaf order)."""
        k = self._kin()
        out = []
        for L in self.topo.leaves:
            if L.body < 0:
                out.append(np.broadcast_to(L.local, (self.B,) + L.local.shape).copy())
            else:
                out.append(k.o[:, L.body, None] + np.einsum("bij,vj->bvi", k.R[:, L.body], L.local))
        return out

    def point_speeds(self):
        """(node speed max (B,), contact-point speed max incl. leaf vertices (B,))."""
        k = self._kin()
        vn = np.linalg.norm(k.vO + _cross(k.w, k.tip - k.o), axis=-1)
        b = self.cp_body
        rw = _mv(k.R[:, b], self.cp_local)
        vp = np.linalg.norm(k.vO[:, b] + _cross(k.w[:, b], rw), axis=-1)
        z = np.zeros(self.B)
        return (vn.max(1) if vn.size else z), (vp.max(1) if vp.size else z)

    def energy(self):
        """(kinetic, elastic, gravity potential), each (B,)."""
        k = self._kin()
        Ib = self._spatial_inertia(k)
        v = np.concatenate([k.w, k.vO], -1)
        ke = 0.5 * np.einsum("bni,bnij,bnj->b", v, Ib, v)
        el = 2.0 * (self.K * self.Q[..., 1:] ** 2).sum((1, 2))
        com = self.m[..., None] * k.o + _mv(k.R, self.h1)
        pe = -(com @ self.gravity).sum(1)
        return ke, el, pe

    # ------------------------------------------------------------------ dynamics
    def _spatial_inertia(self, k):
        IO = k.R @ self.Io @ np.swapaxes(k.R, -1, -2)
        mc = _mv(k.R, self.h1)
        Ib = np.zeros(IO.shape[:-2] + (6, 6))
        Ib[..., :3, :3] = IO
        S = skew(mc)
        Ib[..., :3, 3:] = S
        Ib[..., 3:, :3] = np.swapaxes(S, -1, -2)
        Ib[..., 3:, 3:] = self.m[..., None, None] * np.eye(3)
        return Ib

    def _external(self, k, external, h, Ieff, fext):
        """Accumulate user point forces / sphere penalty contacts (implicit) in place."""
        if external is None:
            return
        T, B = self.topo, self.B
        if isinstance(external, np.ndarray):
            external = [PointForce(n, external[..., n, :]) for n in range(T.N)
                        if np.any(external[..., n, :] != 0)]
        for e in external:
            if isinstance(e, PointForce):
                bdy = T.node_body[e.node]
                if bdy < 0:
                    continue
                f = np.broadcast_to(np.asarray(e.force, float), (B, 3))
                rr = k.tip[:, bdy] - k.o[:, bdy]
                fext[:, bdy, :3] += _cross(rr, f)
                fext[:, bdy, 3:] += f
            elif isinstance(e, BodyForce):
                bdy = int(e.body)
                f = np.broadcast_to(np.asarray(e.force, float), (B, 3))
                rr = _mv(k.R[:, bdy], np.asarray(e.local, float))
                fext[:, bdy, :3] += _cross(rr, f)
                fext[:, bdy, 3:] += f
            elif isinstance(e, SphereCollider):
                ctr = np.broadcast_to(np.asarray(e.center, float), (B, 3))
                vs = np.zeros((B, 3)) if e.velocity is None else np.broadcast_to(e.velocity, (B, 3))
                bb = self.cp_body
                rw = _mv(k.R[:, bb], self.cp_local)            # (B, P, 3)
                p = k.o[:, bb] + rw
                dvec = p - ctr[:, None]
                dist = np.linalg.norm(dvec, axis=-1)
                bi, pi = np.nonzero(dist < e.radius)
                if len(bi) == 0:
                    continue
                n = dvec[bi, pi] / np.maximum(dist[bi, pi], 1e-12)[:, None]
                pen = e.radius - dist[bi, pi]
                rr = rw[bi, pi]
                body = bb[pi]
                # point Jacobian P = [-r^, I]; n^T P = [ (r x n)^T, n^T ]
                g6 = np.concatenate([_cross(rr, n), n], -1)       # (C, 6)
                v6 = np.concatenate([k.w[bi, body], k.vO[bi, body]], -1)
                vn = (g6 * v6).sum(-1) - (vs[bi] * n).sum(-1)
                fmag = e.stiffness * pen - (h * e.stiffness + e.damping) * vn
                np.add.at(fext, (bi, body), fmag[:, None] * g6)
                np.add.at(Ieff, (bi, body), (h * h * e.stiffness + h * e.damping)
                          * g6[:, :, None] * g6[:, None, :])
            else:
                raise TypeError(f"unknown external action {type(e)}")

    def step(self, h: float, external=None, damping: float | np.ndarray | None = None):
        """Advance all B parameter sets by one implicit step of size h."""
        T, B = self.topo, self.B
        k = self._kin()
        external, implicit, penalty = self._split_external(external)
        alpha = self.alpha if damping is None else np.broadcast_to(np.asarray(damping, float), (B,))
        beta = self.joint_damping
        Ib = self._spatial_inertia(k)
        v = np.concatenate([k.w, k.vO], -1)                       # (B, nb, 6)
        IO, mc = Ib[..., :3, :3], _mv(k.R, self.h1)
        bias = np.concatenate([_cross(k.w, _mv(IO, k.w)), _cross(k.w, _cross(k.w, mc))], -1)
        g = self.gravity
        fext = np.concatenate([_cross(mc, np.broadcast_to(g, mc.shape)),
                               self.m[..., None] * g], -1)
        fext -= alpha[:, None, None] * _mv(Ib, v)                  # mass-proportional damping
        Ieff = (1.0 + h * alpha)[:, None, None, None] * Ib
        if self.implicit_gyro:
            # linearised-implicit body bias b(v_{n+1}) ~ b(v_n) + h (db/dv) a  (Catto 2015,
            # extended to the centrifugal term about the body origin); non-symmetric.
            w = k.w
            Sw = skew(w)
            Ieff[..., :3, :3] += h * (Sw @ IO - skew(_mv(IO, w)))
            wc = (w * mc).sum(-1)[..., None, None]
            Ieff[..., 3:, :3] += h * (w[..., :, None] * mc[..., None, :] + wc * np.eye(3)
                                      - 2.0 * mc[..., :, None] * w[..., None, :])
        if self.implicit_gravity:
            # geometric stiffness of gravity (body level, rotational block), PSD part only:
            # K = (g.mc) I - sym(g mc^T); its negative (buckling) part stays explicit.
            Kg = _gravity_stiffness_psd(g, mc)
            Ieff[..., :3, :3] += h * h * Kg
            fext[..., :3] -= h * _mv(Kg, k.w)
        self._external(k, external, h, Ieff, fext)
        if penalty:
            from .contact import apply_penalty
            apply_penalty(self, k, penalty, h, Ieff, fext)

        # joint springs (implicit): tau = -2 Gamma K v - (h+beta) K~ u, armature A
        Q = self.Q * np.where(self.Q[..., :1] < 0, -1.0, 1.0)
        qw, qv = Q[..., 0], Q[..., 1:]
        Gam = qw[..., None, None] * np.eye(3) - skew(qv)          # Gamma = w I - [v]x
        GK = Gam * self.K[..., None, :]                           # Gamma K
        Kt = GK @ np.swapaxes(Gam, -1, -2)                        # Gamma K Gamma^T
        tau = -2.0 * _mv(GK, qv) - (h + beta) * _mv(Kt, self.u)
        A = (h * h + h * beta) * Kt

        # ---- velocity-product (Coriolis/centripetal) propagation terms, linearised-implicit:
        #   c_i(v_{n+1}) ~ c_i + h Cp a_p + h Cu udot_i  =>  a_i = Xm a_p + Sm udot_i + c_i
        #   with Xm = X + h Cp (motion map from the parent), Sm = S + h Cu (joint map).
        # Forces still travel through the rigid X^T and the joint projects with the true S.
        Rb_all = k.R
        Sm = np.zeros((B, T.nb, 6, 3))
        Sm[..., :3, :] = Rb_all
        Xm = _X(k.r)
        if self.implicit_gyro:
            par = np.maximum(T.parent, 0)
            wp = np.where((T.parent >= 0)[None, :, None], k.w[:, par], 0.0)
            wrel = _mv(Rb_all, self.u)
            rr = k.r
            Sm[..., :3, :] += h * skew(wp) @ Rb_all
            Xm[..., :3, :3] += -h * skew(wrel)
            wr = (wp * rr).sum(-1)[..., None, None]
            Xm[..., 3:, :3] += h * (wp[..., :, None] * rr[..., None, :] + wr * np.eye(3)
                                    - 2.0 * rr[..., :, None] * wp[..., None, :])

        # ---- leaf -> root: articulated inertias (IA may be non-symmetric)
        IA = Ieff
        pA = bias - fext
        U = np.empty((B, T.nb, 6, 3))           # IA Sm
        W = np.empty((B, T.nb, 3, 6))           # S^T IA
        Dinv = np.empty((B, T.nb, 3, 3))
        UDall = np.zeros((B, T.nb, 6, 3))       # U D^-1 (kept for contact responses, B3)
        uu = np.empty((B, T.nb, 3))
        for d in range(T.nlev - 1, -1, -1):
            sl = slice(T.lvl[d], T.lvl[d + 1])
            RbT = np.swapaxes(Rb_all[:, sl], -1, -2)
            Ul = IA[:, sl] @ Sm[:, sl]
            Wl = RbT @ IA[:, sl, :3, :]
            D = Wl @ Sm[:, sl] + A[:, sl]
            Di = np.linalg.inv(D)
            ul = tau[:, sl] - _mv(RbT, pA[:, sl, :3])
            U[:, sl], W[:, sl], Dinv[:, sl], uu[:, sl] = Ul, Wl, Di, ul
            if d == 0:
                break
            UD = Ul @ Di
            UDall[:, sl] = UD
            Ia = IA[:, sl] - UD @ Wl
            pa = pA[:, sl] + _mv(Ia, k.c[:, sl]) + _mv(UD, ul)
            rr = k.r[:, sl]
            XtIaX = np.swapaxes(_X(rr), -1, -2) @ Ia @ Xm[:, sl]     # X^T Ia X'

            Xtpa = pa.copy()
            Xtpa[..., :3] += _cross(rr, pa[..., 3:])           # X^T f = [n + r x F; F]
            par, starts = T.red[d]
            IA[:, par] += np.add.reduceat(XtIaX, starts, axis=1)
            pA[:, par] += np.add.reduceat(Xtpa, starts, axis=1)

        # ---- floating bases (B3): 6-DOF root joint, IA a = -pA at the root
        fl = T.float_idx
        IAinv_f = np.linalg.inv(IA[:, fl]) if len(fl) else None
        pA_f = pA[:, fl] if len(fl) else None

        # ---- root -> leaf: accelerations
        acc = np.empty((B, T.nb, 6))
        udot = np.empty((B, T.nb, 3))
        for d in range(T.nlev):
            sl = slice(T.lvl[d], T.lvl[d + 1])
            if d == 0:
                ap = k.c[:, sl]
            else:
                ap = k.c[:, sl] + _mv(Xm[:, sl], acc[:, T.parent[sl]])
            ud = _mv(Dinv[:, sl], uu[:, sl] - _mv(W[:, sl], ap))
            udot[:, sl] = ud
            acc[:, sl] = ap + _mv(Sm[:, sl], ud)
            if d == 0 and len(fl):
                af = -_mv(IAinv_f, pA_f)
                acc[:, fl] = af
                udot[:, fl] = _mtv(Rb_all[:, fl], af[..., :3])      # omega_dot = R u_dot

        self._factor = _Factor(UDall, W, Dinv, Xm, Sm, Rb_all, k.r, k.o, IAinv_f)
        if implicit:
            from .contact import solve_contacts
            d_udot, d_acc = solve_contacts(self, k, self._factor, h, udot, acc, implicit)
            udot = udot + d_udot
            acc = acc + d_acc
        else:
            self.contact = None

        self.u = self.u + h * udot
        if len(fl):
            self.v_float = self.v_float + h * acc[:, fl, 3:]
            self.p_float = self.p_float + h * self.v_float
        Qn = qmul(self.Q, quat_from_rotvec(h * self.u))
        self.Q = Qn / np.linalg.norm(Qn, axis=-1, keepdims=True)
        self.t += h
        self._kin_cache = None
        return udot

    # ------------------------------------------------------------------ B3 helpers
    def _split_external(self, external):
        """-> (B1 actions [PointForce, SphereCollider], implicit colliders, penalty colliders).
        Colliders are heliades.sim.contact objects (attribute `contact_mode`); the
        persistent `self.colliders` are always included."""
        T = self.topo
        if isinstance(external, np.ndarray):
            external = [PointForce(n, external[..., n, :]) for n in range(T.N)
                        if np.any(external[..., n, :] != 0)]
        acts, imp, pen = [], [], []
        for e in list(external or []) + self.colliders:
            mode = getattr(e, "contact_mode", None)
            if mode is None:
                acts.append(e)
            elif mode == "implicit":
                imp.append(e)
            elif mode == "penalty":
                pen.append(e)
            else:
                raise ValueError(f"unknown contact_mode {mode!r}")
        return acts, imp, pen

    def response(self, pA_extra, fac=None):
        """Linear response of the last step's factorised system to extra body *bias*
        pA_extra (B, nb, 6) (= minus a spatial force [moment about o_b; force], world).
        Returns (udot (B, nb, 3), acc (B, nb, 6)): the change of the joint accelerations
        and of the (modified) body accelerations. Velocity-product terms excluded; O(n)."""
        T, B = self.topo, self.B
        f = self._factor if fac is None else fac
        pA = np.array(pA_extra, float)
        ul = np.empty((B, T.nb, 3))
        for d in range(T.nlev - 1, -1, -1):
            sl = slice(T.lvl[d], T.lvl[d + 1])
            ul[:, sl] = -_mtv(f.R[:, sl], pA[:, sl, :3])
            if d == 0:
                break
            pa = pA[:, sl] + _mv(f.UD[:, sl], ul[:, sl])
            pa[..., :3] += _cross(f.r[:, sl], pa[..., 3:])
            par, starts = T.red[d]
            pA[:, par] += np.add.reduceat(pa, starts, axis=1)
        fl = T.float_idx
        acc = np.empty((B, T.nb, 6))
        udot = np.empty((B, T.nb, 3))
        for d in range(T.nlev):
            sl = slice(T.lvl[d], T.lvl[d + 1])
            if d == 0:
                ud = _mv(f.Dinv[:, sl], ul[:, sl])
                acc[:, sl] = _mv(f.Sm[:, sl], ud)
            else:
                ap = _mv(f.Xm[:, sl], acc[:, T.parent[sl]])
                ud = _mv(f.Dinv[:, sl], ul[:, sl] - _mv(f.W[:, sl], ap))
                acc[:, sl] = ap + _mv(f.Sm[:, sl], ud)
            udot[:, sl] = ud
            if d == 0 and len(fl):
                af = -_mv(f.IAinv_f, pA[:, fl])
                acc[:, fl] = af
                udot[:, fl] = _mtv(f.R[:, fl], af[..., :3])
        return udot, acc

    def run(self, duration: float, h: float, external=None, damping=None, record=None, every=1):
        """Step for `duration`; returns array of recorded node positions (F, B, N, 3) or
        record(sim) outputs, sampled every `every` steps (including t=0)."""
        nsteps = int(round(duration / h))
        rec = (lambda s: s.state.x) if record is None else record
        out = [rec(self)]
        for i in range(nsteps):
            ext = external(self.t) if callable(external) else external
            self.step(h, ext, damping)
            if (i + 1) % every == 0:
                out.append(rec(self))
        return np.array(out)

    def settle(self, h: float = 1 / 60, max_time: float = 3.0, tol: float = 1e-3,
               damping=None, external=None, include_leaves: bool = False, window: float = 0.25):
        """Integrate until every batch element's max node speed has stayed < tol [m/s]
        for `window` seconds (with include_leaves, leaf-vertex speeds count too).
        Returns an info dict; t_converged is the start of the quiet window."""
        nmax = int(round(max_time / h))
        nwin = max(1, int(round(window / h)))
        hist = []
        quiet = np.zeros(self.B, int)
        conv_t = np.full(self.B, np.nan)
        for i in range(nmax):
            self.step(h, external, damping)
            vn, vp = self.point_speeds()
            sp = np.maximum(vn, vp) if include_leaves else vn
            hist.append(sp)
            quiet = np.where(sp < tol, quiet + 1, 0)
            newly = (quiet >= nwin) & np.isnan(conv_t)
            conv_t[newly] = self.t - (nwin - 1) * h
            if not np.isnan(conv_t).any():
                break
        return {"time": self.t, "steps": i + 1, "converged": ~np.isnan(conv_t),
                "t_converged": conv_t, "max_speed": hist[-1], "speed_history": np.array(hist)}

    def solve_static(self, h: float = 0.2, iters: int = 200, tol: float = 1e-6, external=None):
        """Pseudo-time static solve: one implicit step with zero damping, then velocities
        reset to zero ("modified Newton" with Hessian ~ M/h^2 + K~ + Kg_psd). Converges
        fast for well-supported structures, slowly near buckling. The fixed point is the
        same equilibrium the dynamic settle reaches. Returns info dict
        (iterations, converged (B,), last max node increment (B,))."""
        self.u[:] = 0.0
        self._kin_cache = None
        x_prev = self.state.x
        hist = []
        for it in range(iters):
            self.step(h, external, damping=0.0)
            self.u[:] = 0.0
            self._kin_cache = None
            x = self.state.x
            dx = np.abs(x - x_prev).max((1, 2))
            hist.append(dx)
            x_prev = x
            if (dx < tol).all():
                break
        return {"iterations": it + 1, "converged": hist[-1] < tol, "increment": hist[-1],
                "history": np.array(hist)}
