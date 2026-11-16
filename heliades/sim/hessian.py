"""Joint-space statics of the B1 tree model: subtree loads, potential gradient, and the
exact tangent stiffness (Hessian) with an O(n) tree factorisation (B2, docs/sim/INVERSE.md).

Notation (B1): body j has world rotation R_j, origin o_j (its proximal joint) and a
child-frame perturbation delta_j (Q_j <- Q_j exp(delta_j)). The potential is

    V(Q) = sum_j 2 v_j^T K_j v_j  -  sum_b F_b . p_b          (dead loads F_b: gravity, PointForce)

* subtree load matrix about o_j:  L_j = sum_{loads in sub(j)} (p - o_j) F^T   (3x3)
  moment N_j = vee(L_j^T - L_j) = sum (p - o_j) x F;   A_j = L_j - tr(L_j) I
* gradient (child frame):  g_j = 2 Gamma_j K_j v_j - R_j^T N_j          (= -tau of B1)
* derivative of the gradient, world variables e_j = R_j delta_j:
      dg_j/de_j       = R_j H_el,j R_j^T - A_j^T                       (diagonal)
      dg_j/de_k       = -A_j^T     for k a strict ancestor of j
      dg_k/de_j       = -A_j       (the transpose: the coupling is symmetric)
  H_el = Gamma K Gamma^T + [K v]x Gamma^T - (K v) v^T.
  Only the diagonal blocks are non-symmetric away from equilibrium (by terms ~ the
  joint's own gradient); at an equilibrium the whole matrix is symmetric.

Because the coupling of j to all its ancestors goes through one 3-vector, the total
rotation of its parent Omega_p = sum_{k anc} e_k, the matrix has the same "ancestor"
structure as an articulated-body mass matrix and admits an exact leaf-to-root block
elimination with 3x3 pivots (``eliminate``). By Sylvester/Haynsworth, the inertia of H is
the sum of the inertias of the pivots: an exact O(n) stability (Sturm) count, usable with
a shift (H - sigma M) for eigenvalue bisection and with M = K_el for the buckling pencil.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotation import qmul, quat_from_rotvec, quat_to_matrix, skew
from .tree import PointForce, _cross, _mv

EYE3 = np.eye(3)


def _T(M):
    return np.swapaxes(M, -1, -2)


def canon(Q):
    """Quaternion with w >= 0 (same rotation; B1's energy is even in Q)."""
    return Q * np.where(Q[..., :1] < 0, -1.0, 1.0)


# ---------------------------------------------------------------- kinematics, loads
def forward_kinematics(topo, Rrel):
    """Body world rotations / origins / tips from joint relative rotations (B, nb, 3, 3)."""
    T = topo
    B = Rrel.shape[0]
    R = np.empty_like(Rrel)
    o = np.empty((B, T.nb, 3))
    tip = np.empty_like(o)
    for d in range(T.nlev):
        sl = slice(T.lvl[d], T.lvl[d + 1])
        if d == 0:
            R[:, sl] = Rrel[:, sl]
            o[:, sl] = T.o0_ground[sl]
        else:
            p = T.parent[sl]
            R[:, sl] = R[:, p] @ Rrel[:, sl]
            o[:, sl] = tip[:, p]
        tip[:, sl] = o[:, sl] + T.l[sl, None] * R[:, sl, :, 2]
    return R, o, tip


def subtree_sum(topo, X):
    """In-place leaf-to-root subtree sums of a per-body array X (B, nb, ...)."""
    T = topo
    for d in range(T.nlev - 1, 0, -1):
        par, starts = T.red[d]
        X[:, par] += np.add.reduceat(X[:, T.lvl[d]:T.lvl[d + 1]], starts, axis=1)
    return X


@dataclass
class Loads:
    F: np.ndarray    # (B, nb, 3)    subtree resultant
    L: np.ndarray    # (B, nb, 3, 3) subtree load matrix about o_j: sum (p - o_j) F^T
    N: np.ndarray    # (B, nb, 3)    subtree moment about o_j
    A: np.ndarray    # (B, nb, 3, 3) L - tr(L) I  (rotational load stiffness)


def subtree_loads(sim, R, o, tip, external=None) -> Loads:
    """Dead loads (gravity of segments, leaves and fruit payloads; optional PointForce list)
    summed over every subtree in one leaf-to-root sweep. O(n), batched."""
    T = sim.topo
    B = R.shape[0]
    g = np.asarray(sim.gravity, float)
    m = np.broadcast_to(sim.m, (B, T.nb))
    mc = _mv(R, np.broadcast_to(sim.h1, (B, T.nb, 3)))          # world first moment about o_b
    F = m[..., None] * g
    Lb = mc[..., :, None] * g[None, None, None, :]
    for e in external or []:
        if not isinstance(e, PointForce):
            raise TypeError("inverse statics supports dead PointForce loads only")
        b = T.node_body[e.node]
        if b < 0:
            continue
        f = np.broadcast_to(np.asarray(e.force, float), (B, 3))
        F[:, b] += f
        Lb[:, b] += (tip[:, b] - o[:, b])[..., :, None] * f[..., None, :]
    W = Lb + o[..., :, None] * F[..., None, :]                   # world: sum p F^T
    F = subtree_sum(T, F.copy())
    W = subtree_sum(T, W)
    L = W - o[..., :, None] * F[..., None, :]
    N = np.stack([L[..., 1, 2] - L[..., 2, 1], L[..., 2, 0] - L[..., 0, 2],
                  L[..., 0, 1] - L[..., 1, 0]], -1)
    A = L - np.trace(L, axis1=-2, axis2=-1)[..., None, None] * EYE3
    return Loads(F, L, N, A)


# ---------------------------------------------------------------- elastic joint terms
def elastic_terms(K, Q):
    """Gradient 2 Gamma K v and Hessian blocks H_el (B, nb, 3, 3) of U = 2 v^T K v under
    right perturbation Q exp(delta)."""
    Q = canon(Q)
    w, v = Q[..., 0], Q[..., 1:]
    Gam = w[..., None, None] * EYE3 - skew(v)
    Kv = K * v
    g = 2.0 * _mv(Gam, Kv)
    GK = Gam * K[..., None, :]
    H = GK @ _T(Gam) + skew(Kv) @ _T(Gam) - Kv[..., :, None] * v[..., None, :]
    return g, H


# ---------------------------------------------------------------- statics of the forward model
@dataclass
class Statics:
    R: np.ndarray
    o: np.ndarray
    tip: np.ndarray
    loads: Loads
    g: np.ndarray          # (B, nb, 3) potential gradient (child frame); 0 at equilibrium
    Hel: np.ndarray        # (B, nb, 3, 3)
    Dw: np.ndarray         # (B, nb, 3, 3) world diagonal blocks of dg/de
    Kw: np.ndarray         # (B, nb, 3, 3) world joint stiffness R K R^T (the K_el metric)


def statics(sim, Q=None, external=None, Rrel_rest=None) -> Statics:
    """Gradient and tangent stiffness of the forward model at deviation Q (default sim.Q)."""
    Q = sim.Q if Q is None else Q
    R0 = sim.R0rel if Rrel_rest is None else Rrel_rest
    Rrel = R0 @ quat_to_matrix(Q)
    R, o, tip = forward_kinematics(sim.topo, np.broadcast_to(Rrel, Q.shape[:2] + (3, 3)).copy())
    ld = subtree_loads(sim, R, o, tip, external)
    K = np.broadcast_to(sim.K, Q.shape[:2] + (3,))
    gel, Hel = elastic_terms(K, Q)
    g = gel - _mv(_T(R), ld.N)
    Dw = R @ Hel @ _T(R) - _T(ld.A)
    Kw = (R * K[..., None, :]) @ _T(R)
    return Statics(R, o, tip, ld, g, Hel, Dw, Kw)


def potential(sim, Q=None, external=None):
    """V = elastic + dead-load potential, (B,)."""
    Q = sim.Q if Q is None else Q
    Rrel = sim.R0rel @ quat_to_matrix(Q)
    R, o, tip = forward_kinematics(sim.topo, np.broadcast_to(Rrel, Q.shape[:2] + (3, 3)).copy())
    B = Q.shape[0]
    el = 2.0 * (np.broadcast_to(sim.K, Q.shape[:2] + (3,)) * canon(Q)[..., 1:] ** 2).sum((1, 2))
    com = np.broadcast_to(sim.m, (B, sim.topo.nb))[..., None] * o + _mv(R, np.broadcast_to(sim.h1, (B, sim.topo.nb, 3)))
    pe = -(com @ np.asarray(sim.gravity, float)).sum(1)
    for e in external or []:
        b = sim.topo.node_body[e.node]
        if b >= 0:
            pe = pe - (np.broadcast_to(np.asarray(e.force, float), (B, 3)) * tip[:, b]).sum(-1)
    return el + pe


# ---------------------------------------------------------------- O(n) tree factorisation
@dataclass
class TreeFactor:
    P: np.ndarray        # (B, nb, 3, 3) pivots
    Pinv: np.ndarray
    C: np.ndarray        # A^T + Phibar   (coupling to Omega_parent)
    E: np.ndarray        # A + Phibar
    eig: np.ndarray | None   # (B, nb, 3) pivot eigenvalues (symmetric mode)

    @property
    def n_negative(self):
        return (self.eig < 0).sum((1, 2))

    def joint_negative(self):
        return (self.eig < 0).sum(-1)


def eliminate(topo, Dw, A, symmetric=True, shift=None, metric=None) -> TreeFactor:
    """Leaf-to-root block elimination of the ancestor-structured matrix
    M = blkdiag(Dw) + coupling(-A), in world variables. With shift sigma (B,) and metric
    (B, nb, 3, 3) (default identity) it factors M - sigma * metric.
    symmetric=True symmetrises the diagonal blocks (exact at an equilibrium) and returns
    pivot eigenvalues: inertia(M) = sum inertia(P_j)  (Haynsworth)."""
    T = topo
    B = Dw.shape[0]
    D = Dw
    if shift is not None:
        Mt = EYE3 if metric is None else metric
        D = Dw - np.asarray(shift, float).reshape(-1, 1, 1, 1) * Mt
    if symmetric:
        D = 0.5 * (D + _T(D))
    At = _T(A)
    Phib = np.zeros((B, T.nb, 3, 3))
    P = np.empty_like(Phib)
    Pinv = np.empty_like(Phib)
    C = np.empty_like(Phib)
    E = np.empty_like(Phib)
    eig = np.empty((B, T.nb, 3)) if symmetric else None
    for d in range(T.nlev - 1, -1, -1):
        sl = slice(T.lvl[d], T.lvl[d + 1])
        Pb = Phib[:, sl]
        Pl = D[:, sl] - Pb
        if symmetric:
            Pl = 0.5 * (Pl + _T(Pl))
            lam, V = np.linalg.eigh(Pl)
            # an exactly singular pivot (shift on an eigenvalue): perturb it to +tiny, the
            # usual Sturm-sequence convention (counts eigenvalues strictly below the shift)
            tiny = 1e-14 * np.abs(lam).max(-1, keepdims=True) + 1e-300
            lam = np.where(np.abs(lam) < tiny, tiny, lam)
            eig[:, sl] = lam
            Pi = (V / lam[..., None, :]) @ _T(V)
        else:
            Pi = np.linalg.inv(Pl)
        Cl = At[:, sl] + Pb
        El = A[:, sl] + Pb
        P[:, sl], Pinv[:, sl], C[:, sl], E[:, sl] = Pl, Pi, Cl, El
        if d == 0:
            break
        Phi = Pb + El @ Pi @ Cl
        par, starts = T.red[d]
        Phib[:, par] += np.add.reduceat(Phi, starts, axis=1)
    return TreeFactor(P, Pinv, C, E, eig)


def tree_solve(topo, fac: TreeFactor, rhs):
    """Solve M x = rhs (world variables, (B, nb, 3)) with a TreeFactor. O(n)."""
    T = topo
    B = rhs.shape[0]
    phib = np.zeros((B, T.nb, 3))
    r = np.empty_like(phib)
    for d in range(T.nlev - 1, -1, -1):
        sl = slice(T.lvl[d], T.lvl[d + 1])
        rl = rhs[:, sl] + phib[:, sl]
        r[:, sl] = rl
        if d == 0:
            break
        phi = phib[:, sl] + _mv(fac.E[:, sl] @ fac.Pinv[:, sl], rl)
        par, starts = T.red[d]
        phib[:, par] += np.add.reduceat(phi, starts, axis=1)
    x = np.empty_like(phib)
    Om = np.zeros_like(phib)
    for d in range(T.nlev):
        sl = slice(T.lvl[d], T.lvl[d + 1])
        Op = 0.0 if d == 0 else Om[:, T.parent[sl]]
        xl = _mv(fac.Pinv[:, sl], r[:, sl] + (_mv(fac.C[:, sl], Op) if d else 0.0))
        x[:, sl] = xl
        Om[:, sl] = Op + xl
    return x


def tree_matvec(topo, Dw, A, x):
    """y = M x for the ancestor-structured matrix (world variables), O(n). For tests."""
    T = topo
    Om = np.zeros_like(x)
    for d in range(T.nlev):
        sl = slice(T.lvl[d], T.lvl[d + 1])
        Om[:, sl] = x[:, sl] + (Om[:, T.parent[sl]] if d else 0.0)
    Omp = Om - x
    s = subtree_sum(topo, _mv(A, x)) - _mv(A, x)                 # strict-descendant sums
    return _mv(Dw, x) - _mv(_T(A), Omp) - s


def dense_hessian(topo, st: Statics, symmetric=True):
    """Dense joint-space matrix dg/ddelta (B, 3nb, 3nb), child-frame coordinates. O(n^2)."""
    T = topo
    B, nb = st.g.shape[:2]
    R, A = st.R, st.loads.A
    H = np.zeros((B, nb, 3, nb, 3))
    D = st.Dw
    if symmetric:
        D = 0.5 * (D + _T(D))
    for j in range(nb):
        H[:, j, :, j, :] = _T(R[:, j]) @ D[:, j] @ R[:, j]
        k = T.parent[j]
        while k >= 0:
            blk = -_T(R[:, j]) @ _T(A[:, j]) @ R[:, k]
            H[:, j, :, k, :] = blk
            H[:, k, :, j, :] = _T(blk)
            k = T.parent[k]
    return H.reshape(B, 3 * nb, 3 * nb)


def dense_metric(st: Statics):
    """Dense block-diagonal K_el (child frame), (B, 3nb, 3nb)."""
    B, nb = st.g.shape[:2]
    K = np.zeros((B, nb, 3, nb, 3))
    Kc = _T(st.R) @ st.Kw @ st.R
    for j in range(nb):
        K[:, j, :, j, :] = Kc[:, j]
    return K.reshape(B, 3 * nb, 3 * nb)


# ---------------------------------------------------------------- Sturm bisection
def count_below(topo, st: Statics, sigma, metric="euclid"):
    """Number of eigenvalues of H (metric='euclid') or of the pencil (H, K_el)
    (metric='K') strictly below sigma (B,). Exact (Sylvester), O(n)."""
    M = None if metric == "euclid" else st.Kw
    fac = eliminate(topo, st.Dw, st.loads.A, symmetric=True, shift=sigma, metric=M)
    return fac.n_negative


def min_eigenvalue(topo, st: Statics, metric="euclid", rtol=1e-10, max_iter=200):
    """Smallest eigenvalue of H (or generalized eigenvalue of (H, K_el)) by Sturm-count
    bisection, batched (B,). O(n log(1/rtol))."""
    B = st.g.shape[0]
    if metric == "K":
        lo, hi = np.full(B, -1.0), np.full(B, 1.0)
    else:
        s = np.abs(st.Dw).sum((-1, -2)).max(1) + np.abs(st.loads.A).sum((-1, -2)).max(1) * topo.nlev
        lo, hi = -s, s
    # bracket: count(lo) == 0, count(hi) >= 1
    for _ in range(200):
        c = count_below(topo, st, lo, metric)
        if (c == 0).all():
            break
        lo = np.where(c > 0, 2 * lo - hi, lo)
    for _ in range(200):
        c = count_below(topo, st, hi, metric)
        if (c >= 1).all():
            break
        hi = np.where(c < 1, hi + 2 * (hi - lo), hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        c = count_below(topo, st, mid, metric)
        lo = np.where(c == 0, mid, lo)
        hi = np.where(c >= 1, mid, hi)
        if (hi - lo <= rtol * np.maximum(np.abs(hi), 1e-300)).all():
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- forward Newton statics
def newton_static(sim, iters=50, tol=1e-12, external=None, verbose=False):
    """Exact Newton on the forward equilibrium g(Q) = 0 using the O(n) tree solve of dg/dδ
    (with energy backtracking). Polishes TreeSim.solve_static to round-off. Returns info."""
    T = sim.topo
    hist = []
    for it in range(iters):
        st = statics(sim, external=external)
        gn = np.abs(st.g).max((1, 2))
        scale = np.abs(st.loads.N).max((1, 2)) + 1e-30
        hist.append(gn / scale)
        if (gn <= tol * scale).all():
            break
        fac = eliminate(T, st.Dw, st.loads.A, symmetric=False)
        ge = _mv(st.R, st.g)
        x = tree_solve(T, fac, -ge)
        dlt = _mv(_T(st.R), x)
        V0 = potential(sim, external=external)
        a = np.ones(sim.B)
        for _ in range(30):
            Qn = qmul(sim.Q, quat_from_rotvec(a[:, None, None] * dlt))
            Qn /= np.linalg.norm(Qn, axis=-1, keepdims=True)
            V1 = potential(sim, Qn, external)
            bad = V1 > V0 + 1e-12 * np.abs(V0) + 1e-300
            # accept full steps once close (V differences fall below round-off)
            bad &= gn > 1e-8 * scale
            if not bad.any():
                break
            a = np.where(bad, 0.5 * a, a)
        sim.Q = Qn
        sim._kin_cache = None
        if verbose:
            print(it, gn / scale, a)
    sim.u[:] = 0.0
    sim._kin_cache = None
    return {"iterations": it + 1, "residual": hist[-1], "converged": hist[-1] <= tol, "history": np.array(hist)}
