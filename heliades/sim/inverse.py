"""B2: exact discrete inverse statics for the B1 tree model (docs/sim/INVERSE.md).

Given a *target* (loaded) configuration, e.g. the Helios / photo shape, find joint rest
rotations such that the target is an exact static equilibrium of the discrete forward
model ``TreeSim``, and check that this equilibrium is stable.

  1. kinematics of the target (one root-to-leaf sweep) -> R_j, o_j
  2. dead loads of every subtree (one leaf-to-root segmented sum) -> moment N_j about o_j
  3. per joint (independent, vectorised): solve  2 Gamma(Q) K v(Q) = m_j := R_j^T N_j  for
     the unit deviation quaternion Q_j on the small-angle branch (Newton, 3x3)
  4. rest relative rotation R0rel_j = R_rel,target,j R(Q_j)^T
  5. stability: Sturm count of the exact tangent stiffness at the target (hessian.py), O(n)

All steps are O(n) per batch member and vectorised over the batch B of parameter sets.
Leaves and fruit are rigid payloads (their weight enters the subtree loads).

Usage:
    sim = TreeSim(asset, [phys(E1), phys(E2), ...])
    res = inverse_statics(sim)            # target = asset geometry (the loaded shape)
    apply_rest(sim, res)                  # sim now starts at the target, in equilibrium
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from .hessian import (Statics, _T, canon, dense_hessian, dense_metric, eliminate, elastic_terms,
                      forward_kinematics, min_eigenvalue, subtree_loads)
from .rotation import quat_to_matrix, skew
from .tree import _cross, _mv

EYE3 = np.eye(3)


# ---------------------------------------------------------------- local joint solve
def _joint_newton(m, K, v, iters=40, tol=1e-15):
    """Damped Newton for 2 (w K v - v x K v) = m on flat (n, 3) arrays, active-set."""
    v = v.copy()
    scale = np.abs(m).sum(-1) + 1e-300

    def resid(v, m, K):
        w = np.sqrt(np.maximum(1.0 - (v * v).sum(-1), 0.0))
        Kv = K * v
        return 2.0 * (w[:, None] * Kv - _cross(v, Kv)) - m, w, Kv

    act = np.arange(len(m))
    for _ in range(iters):
        r, w, Kv = resid(v[act], m[act], K[act])
        rn = np.abs(r).sum(-1)
        keep = rn > tol * scale[act]
        act, r, w, Kv, rn = act[keep], r[keep], w[keep], Kv[keep], rn[keep]
        if len(act) == 0:
            break
        va, Ka, ma = v[act], K[act], m[act]
        Kd = Ka[:, None, :] * EYE3
        J = 2.0 * (w[:, None, None] * Kd - Kv[:, :, None] * va[:, None, :] / np.maximum(w, 1e-12)[:, None, None]
                   - skew(va) @ Kd + skew(Kv))
        dv = -np.linalg.solve(J, r[..., None])[..., 0]
        a = np.ones(len(act))
        vn = va
        for _ in range(40):                       # keep |v| < 1 (w > 0) and decrease |r|
            vt = va + a[:, None] * dv
            inside = (vt * vt).sum(-1) < 1.0 - 1e-12
            rr, _, _ = resid(np.where(inside[:, None], vt, va), ma, Ka)
            good = inside & (np.abs(rr).sum(-1) < rn * (1 - 1e-4 * a) + 1e-300)
            vn = np.where(good[:, None], vt, vn)
            if good.all():
                break
            a = np.where(good, a, 0.5 * a)
        stuck = ~good
        v[act] = vn
        act = act[~stuck]
    r, w, _ = resid(v, m, K)
    return v, w, np.abs(r).sum(-1) <= 1e-9 * scale + 1e-300


def solve_joint_deviation(m, K, n_cont=16):
    """Unit quaternions Q (..., 4), w >= 0, with 2 Gamma(Q) K v = m, on the branch connected
    to Q = 1 (the smallest-rotation, locally stable branch). m, K: (..., 3). Returns (Q, ok);
    ok is False where that branch does not reach m: the joint torque exceeds the quaternion
    spring's capacity (isotropic K: |m| > k, max of k sin(theta)). Newton from the linear
    guess; joints that land on the wrong branch or fail are redone by continuation m -> t m."""
    shape = m.shape
    m = np.asarray(m, float).reshape(-1, 3)
    K = np.broadcast_to(K, shape).reshape(-1, 3)
    v0 = 0.5 * m / K
    nv = np.linalg.norm(v0, axis=-1, keepdims=True)
    v0 = np.where(nv > 0.5, v0 * 0.5 / np.maximum(nv, 1e-300), v0)
    v, w, conv = _joint_newton(m, K, v0)

    def branch_ok(v, w, K, conv):
        Q = np.concatenate([w[:, None], v], -1)
        # on the branch through Q = 1, H_el = (1/2)(dm/dv) Gamma^T has det > 0; it vanishes
        # at the fold (capacity)
        _, Hel = elastic_terms(K, Q)
        return conv & (np.linalg.det(Hel) > 0) & (w > 1e-6)

    ok = branch_ok(v, w, K, conv)
    bad = np.flatnonzero(~ok)
    if len(bad):
        mb, Kb = m[bad], K[bad]
        vb = np.zeros((len(bad), 3))
        okb = np.ones(len(bad), bool)
        for t in np.linspace(1.0 / n_cont, 1.0, n_cont):
            vb, wb, cb = _joint_newton(t * mb, Kb, vb)
            okb &= branch_ok(vb, wb, Kb, cb)
        v[bad], w[bad], ok[bad] = vb, wb, okb
    Q = np.concatenate([w[:, None], v], -1)
    return Q.reshape(shape[:-1] + (4,)), ok.reshape(shape[:-1])


# ---------------------------------------------------------------- result
@dataclass
class InverseResult:
    R0rel: np.ndarray        # (B, nb, 3, 3) rest relative rotations (feed TreeSim.set_rest)
    Q: np.ndarray            # (B, nb, 4) deviation from rest at the target (TreeSim.reset(Q))
    torque: np.ndarray       # (B, nb, 3) joint moment m_j = R_j^T N_j (child frame) [N m]
    rest_angle: np.ndarray   # (B, nb) |rest - target| per joint [rad]
    joint_ok: np.ndarray     # (B, nb) local solve feasible
    feasible: np.ndarray     # (B,)
    n_negative: np.ndarray   # (B,) number of negative eigenvalues of the tangent stiffness
    joint_negative: np.ndarray  # (B, nb) negative pivots per joint (localises instability)
    stable: np.ndarray       # (B,) feasible and n_negative == 0
    status: list             # per batch: "stable" | "unstable" | "infeasible"
    statics: Statics | None = None
    timing: dict = field(default_factory=dict)
    # optional diagnostics (min_eig=True)
    min_eig: np.ndarray | None = None       # (B,) smallest eigenvalue of H [N m / rad]
    margin: np.ndarray | None = None        # (B,) smallest generalized eigenvalue of (H, K_el)
    E_cr_linear: np.ndarray | None = None   # (B,) E (1 - margin): linearised buckling modulus


def target_from_asset(sim):
    """Target joint relative rotations of the asset geometry itself (Q = 1, asset rest)."""
    return np.broadcast_to(sim.topo.R0rel, (sim.B,) + sim.topo.R0rel.shape).copy()


def target_from_state(sim):
    """Target = the simulator's current configuration (R0rel_current R(Q))."""
    return np.broadcast_to(sim.R0rel @ quat_to_matrix(sim.Q), (sim.B, sim.topo.nb, 3, 3)).copy()


def inverse_statics(sim, target_Rrel=None, external=None, stability=True, min_eig=False,
                    min_eig_method="sturm") -> InverseResult:
    """Rest shape making `target_Rrel` (B or 1, nb, 3, 3) an exact equilibrium of `sim`'s
    discrete model (gravity + dead PointForce loads; stiffness sim.K per batch member).
    Default target: the asset geometry. Does not modify sim."""
    T, B = sim.topo, sim.B
    t0 = time.perf_counter()
    Rt = target_from_asset(sim) if target_Rrel is None else np.broadcast_to(
        np.asarray(target_Rrel, float), (B, T.nb, 3, 3)).copy()
    R, o, tip = forward_kinematics(T, Rt)
    ld = subtree_loads(sim, R, o, tip, external)
    m = _mv(_T(R), ld.N)
    t1 = time.perf_counter()
    K = np.broadcast_to(sim.K, (B, T.nb, 3))
    Q, jok = solve_joint_deviation(m, K)
    R0rel = Rt @ _T(quat_to_matrix(Q))
    t2 = time.perf_counter()
    feasible = jok.all(1)
    vv = np.linalg.norm(Q[..., 1:], axis=-1)
    angle = 2 * np.arctan2(vv, Q[..., 0])
    res = InverseResult(R0rel, Q, m, angle, jok, feasible, np.zeros(B, int), np.zeros((B, T.nb), int),
                        feasible.copy(), [])
    if stability:
        gel, Hel = elastic_terms(K, Q)
        g = gel - m                                   # = 0 to round-off by construction
        Dw = R @ Hel @ _T(R) - _T(ld.A)
        Kw = (R * K[..., None, :]) @ _T(R)
        st = Statics(R, o, tip, ld, g, Hel, Dw, Kw)
        fac = eliminate(T, Dw, ld.A, symmetric=True)
        res.statics = st
        res.n_negative = fac.n_negative
        res.joint_negative = fac.joint_negative()
        res.stable = feasible & (res.n_negative == 0)
        if min_eig:
            if min_eig_method == "dense":
                import scipy.linalg as sl
                H, Km = dense_hessian(T, st), dense_metric(st)
                res.min_eig = np.array([np.linalg.eigvalsh(H[b])[0] for b in range(B)])
                res.margin = np.array([sl.eigh(H[b], Km[b], eigvals_only=True, subset_by_index=[0, 0])[0]
                                       for b in range(B)])
            else:
                res.min_eig = min_eigenvalue(T, st, "euclid")
                res.margin = min_eigenvalue(T, st, "K")
            E = np.array([p.stem_youngs_modulus for p in sim.params])
            res.E_cr_linear = E * (1.0 - res.margin)
    t3 = time.perf_counter()
    res.status = ["infeasible" if not f else ("stable" if s else "unstable")
                  for f, s in zip(res.feasible, res.stable)]
    res.timing = {"loads": t1 - t0, "local": t2 - t1, "stability": t3 - t2, "total": t3 - t0}
    return res


def apply_rest(sim, res: InverseResult):
    """Load the inverse rest shape into sim and place it at the target, at rest."""
    sim.set_rest(res.R0rel)
    sim.reset(Q=res.Q)
    return sim


def rest_shape_nodes(sim, res: InverseResult):
    """(B, N, 3) rod node positions of the unloaded rest shape (Q = 1)."""
    T = sim.topo
    _, _, tip = forward_kinematics(T, res.R0rel.copy())
    x = np.broadcast_to(T.X0, (res.R0rel.shape[0], T.N, 3)).copy()
    x[:, T.node_b] = tip
    return x


# ---------------------------------------------------------------- critical modulus
def _with_moduli(sim, Es, base=None, stiffness_scale=None):
    s = copy.copy(sim)
    p0 = sim.params[0] if base is None else base
    s.set_params([p0.copy(stem_youngs_modulus=float(E)) for E in Es])
    if stiffness_scale is not None:
        s.K = s.K * np.asarray(stiffness_scale, float).reshape(1, -1, 1)
    s.R0rel = sim.topo.R0rel[None]
    return s


def critical_modulus(sim, target_Rrel=None, external=None, E_lo=1e5, E_hi=1e12, rtol=1e-4,
                     n_probe=16, base=None, stiffness_scale=None):
    """Exact critical modulus of the inverse problem: the smallest E for which the inverse
    rest shape exists and holds the target *stably* (stiffness sim.K ∝ E, other params from
    `base` or sim.params[0]). Batched multisection over E (n_probe moduli per round, each
    one O(n) inverse + Sturm count). stiffness_scale (nb,) optionally multiplies every
    joint's stiffness (e.g. 1e3 on petioles to isolate the main-stem mode).
    Returns dict(E_cr, lo, hi, rounds, E_cr_linear_at_hi)."""
    lo, hi = float(E_lo), float(E_hi)
    rounds = 0
    tgt = None if target_Rrel is None else np.asarray(target_Rrel)[:1]
    if tgt is not None and tgt.ndim == 3:
        tgt = tgt[None]
    while hi / lo - 1 > rtol:
        Es = np.geomspace(lo, hi, n_probe)
        s = _with_moduli(sim, Es, base, stiffness_scale)
        r = inverse_statics(s, tgt, external)
        ok = r.stable
        if not ok[-1]:
            raise ValueError(f"target not stable even at E = {hi:g}")
        if ok[0]:
            if lo <= E_lo * (1 + 1e-12):
                return {"E_cr": lo, "lo": lo, "hi": lo, "rounds": rounds, "note": "stable at E_lo"}
            lo = lo / 1e3
            continue
        i = int(np.flatnonzero(~ok)[-1])
        lo, hi = Es[i], Es[i + 1]
        rounds += 1
    # linearised estimate at the critical point (generalized eigenproblem, for comparison)
    s = _with_moduli(sim, [hi], base, stiffness_scale)
    r = inverse_statics(s, tgt, external, min_eig=True)
    return {"E_cr": float(np.sqrt(lo * hi)), "lo": lo, "hi": hi, "rounds": rounds,
            "E_cr_linear_at_hi": float(r.E_cr_linear[0])}


def linear_critical_modulus(sim, target_Rrel=None, external=None, method="sturm"):
    """Linearised critical modulus for each batch member: E_cr = E (1 - mu_min) with mu_min
    the smallest generalized eigenvalue of (H(E), K_el(E)) at the inverse rest; exact when
    the elastic Hessian is E-linear (straight unloaded joints, e.g. a vertical column)."""
    r = inverse_statics(sim, target_Rrel, external, min_eig=True, min_eig_method=method)
    return r.E_cr_linear, r


# ---------------------------------------------------------------- best-feasible fallback
def subtree_masks(topo):
    """(nb, nb) bool: S[j, k] = k in subtree(j)."""
    nb = topo.nb
    S = np.eye(nb, dtype=bool)
    for k in range(nb):
        j = topo.parent[k]
        while j >= 0:
            S[j, k] = True
            j = topo.parent[j]
    return S


def stiffen_to_stable(sim, target_Rrel=None, external=None, margin=0.02, f_max=1e4, rtol=1e-3,
                      max_rounds=500):
    """Best-feasible fallback for batch member 0: per-joint stiffness multipliers s_j >= 1 that
    make the inverse rest shape hold the target stably, stiffening only *minimal unstable
    subtrees*. Leaf-to-root elimination localises instability: a negative pivot at joint j
    whose children carry none means subtree(j), clamped at its base, is unstable while all
    its child subtrees are stable. Repeat: pick the deepest such joint, find the smallest
    uniform factor f on subtree(j) (Sturm-count bisection, O(n) per probe) that removes the
    negative pivots inside subtree(j), apply f (1 + margin). Returns dict(s, rounds, result)
    where s (nb,) multiplies sim.K (E_required,j = s_j E)."""
    T = sim.topo
    S = subtree_masks(T)
    depth = np.zeros(T.nb, int)
    for d in range(T.nlev):
        depth[T.lvl[d]:T.lvl[d + 1]] = d
    base = copy.copy(sim)
    base.set_params([sim.params[0]])
    K0 = base.K.copy()
    s = np.ones(T.nb)
    tgt = None if target_Rrel is None else np.asarray(target_Rrel)[:1]

    def run(scales):
        b = copy.copy(base)
        b.K = K0 * scales[:, :, None]
        b.B = scales.shape[0]
        b.params = [sim.params[0]] * b.B
        b.m, b.h1 = base.m, base.h1
        return inverse_statics(b, tgt, external)

    rounds = 0
    for rounds in range(max_rounds):
        r = run(s[None])
        neg = (r.joint_negative[0] > 0) | ~r.joint_ok[0]
        if not neg.any():
            return {"s": s, "rounds": rounds, "result": r}
        cand = np.flatnonzero(neg)
        j = cand[np.argmax(depth[cand])]
        mask = S[j]
        lo, hi = 1.0, 2.0
        while True:
            rr = run((s * np.where(mask, hi, 1.0))[None])
            if not ((rr.joint_negative[0][mask] > 0) | ~rr.joint_ok[0][mask]).any():
                break
            lo, hi = hi, hi * 4
            if hi > f_max:
                raise ValueError(f"subtree {j} not stabilised by factor {f_max}")
        while hi / lo - 1 > rtol:
            fs = np.geomspace(lo, hi, 8)
            rr = run(s[None] * np.where(mask[None], fs[:, None], 1.0))
            okk = ~(((rr.joint_negative > 0) | ~rr.joint_ok)[:, mask]).any(1)
            i = int(np.flatnonzero(okk)[0])
            lo, hi = (fs[i - 1] if i else lo), fs[i]
        s = s * np.where(mask, hi * (1 + margin), 1.0)
    raise RuntimeError("stiffen_to_stable did not converge")
