"""B3 stretch: contact-aware inverse statics (docs/sim/CONTACT.md sec. 6).

B2 inverts a clamped tree under *dead* loads, which is statically determinate. A plant that
rests on a stake or on the ground in its target shape is not: the support forces are extra
unknowns (Derouet-Jourdan 2013's indeterminacy). We pick them, then reuse B2 unchanged:

  1. candidate supports: plant features within `tol` of a collider surface in the target
     (rod capsules, fruit), each with a contact frame (n, t1, t2) and friction coefficient;
  2. admissible forces: non-negative combinations of the edges of an inscribed friction
     pyramid, f_c = sum_e beta_ce (n + mu (cos a_e t1 + sin a_e t2)), beta >= 0 (frictionless:
     f_c = beta_c n). Inscribed, so every admissible force is inside the Coulomb cone;
  3. joint moments are linear in beta: m = m_dead + G beta (G from the target kinematics);
     choose beta minimising the rest-shape prestress  sum_j m_j^T K_j^-1 m_j  (the elastic
     energy of the rest deviation to first order): a non-negative least-squares problem;
  4. B2 inverse statics with gravity + the chosen forces as dead BodyForce loads -> exact rest
     rotations; the target is then an exact equilibrium with those contact reactions;
  5. stability under the support: the dense Hessian (incl. the dead support loads' load
     stiffness) projected on the null space of the active contact constraints (normal rows;
     with friction also the tangential rows of sticking contacts). Surface curvature of the
     collider is neglected (flat-contact approximation).

Batch member 0 only (B = 1 workflow); dense stability check (O(n^3)), for checks and demos.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .contact import ContactParams, _tangents, detect
from .hessian import _T, dense_hessian, forward_kinematics, subtree_loads
from .inverse import InverseResult, inverse_statics, target_from_asset
from .tree import BodyForce, _cross, _mv


@dataclass
class SupportedInverse:
    result: InverseResult        # B2 result with the support forces as dead loads
    forces: list                 # BodyForce loads (the support reactions at the target)
    body: np.ndarray             # (k,) candidate supports
    point: np.ndarray            # (k, 3) world contact points
    frame: np.ndarray            # (k, 3, 3) rows n, t1, t2
    force: np.ndarray            # (k, 3) world support force per candidate
    active: np.ndarray           # (k,) carries load
    prestress: float             # sum m^T K^-1 m with supports
    prestress_free: float        # the same without supports (plain B2)
    n_negative_free: int         # B2 Sturm count (supports as dead loads only)
    n_negative_supported: int    # projected on the contact constraints
    status: str                  # "stable" | "unstable" | "infeasible" (under the support)


def _target_kin(sim, Rt):
    R, o, tip = forward_kinematics(sim.topo, Rt)
    z = np.zeros_like(o)
    return SimpleNamespace(R=R, o=o, tip=tip, w=z, vO=z), R, o, tip


def _moment_columns(sim, R, o, body, p, dirs):
    """G (3 nb, ncol): child-frame joint moments of unit forces dirs (ncol, 3) at points p
    (ncol, 3) on bodies `body` (ncol,): m_j = R_j^T (p - o_j) x d for j an ancestor-or-self."""
    T = sim.topo
    G = np.zeros((T.nb, 3, len(body)))
    for c, (b, pc, d) in enumerate(zip(body, p, dirs)):
        j = T.anc[b, :T.depth[b] + 1]
        G[j, :, c] = np.einsum("jki,jk->ji", R[0, j], _cross(pc - o[0, j], d))
    return G.reshape(3 * T.nb, -1)


def supported_inverse(sim, colliders, target_Rrel=None, tol=1e-3, friction=True, n_edges=8,
                      constrain_tangential=None, min_force=1e-9):
    """Rest shape + support forces making the target an equilibrium supported by `colliders`
    (GroundPlane / SphereObstacle / CapsuleObstacle, e.g. contact.stake). See module doc."""
    from scipy.optimize import nnls

    T = sim.topo
    if sim.B != 1:
        raise ValueError("supported_inverse works on batch member 0 of a B = 1 sim")
    Rt = target_from_asset(sim) if target_Rrel is None else np.broadcast_to(
        np.asarray(target_Rrel, float), (1, T.nb, 3, 3)).copy()
    k, R, o, tip = _target_kin(sim, Rt)
    cand = detect(sim, k, colliders, 0.0, ContactParams(margin=tol, spec_factor=0.0, max_per_body=1))
    if cand is None or len(cand.bi) == 0:
        raise ValueError("no support candidates within tol of the colliders")
    n = cand.n
    t1, t2 = _tangents(n)
    frame = np.stack([n, t1, t2], 1)
    kc = len(cand.body)
    if friction:
        ang = 2 * np.pi * np.arange(n_edges) / n_edges
        dirs = (n[:, None] + cand.mu[:, None, None] * (np.cos(ang)[None, :, None] * t1[:, None]
                                                       + np.sin(ang)[None, :, None] * t2[:, None]))
        ne = n_edges
    else:
        dirs, ne = n[:, None], 1
    dirs = dirs.reshape(-1, 3)
    cb = np.repeat(cand.body, ne)
    cp = np.repeat(cand.p, ne, axis=0)
    G = _moment_columns(sim, R, o, cb, cp, dirs)
    ld = subtree_loads(sim, R, o, tip, None)
    m0 = _mv(_T(R), ld.N)[0].reshape(-1)
    Kinv = 1.0 / np.broadcast_to(sim.K, (1, T.nb, 3))[0].reshape(-1)
    w = np.sqrt(Kinv)
    beta, _ = nnls(w[:, None] * G, -w * m0, maxiter=50 * G.shape[1])
    fvec = (beta.reshape(kc, ne)[..., None] * dirs.reshape(kc, ne, 3)).sum(1)
    active = np.linalg.norm(fvec, axis=1) > min_force
    m = m0 + G @ beta
    pre, pre0 = float((Kinv * m * m).sum()), float((Kinv * m0 * m0).sum())
    forces = []
    for c in np.flatnonzero(active):
        b = int(cand.body[c])
        local = R[0, b].T @ (cand.p[c] - o[0, b])
        forces.append(BodyForce(b, local, fvec[c]))
    res = inverse_statics(sim, Rt, forces)
    # stability under the support: H projected on the null space of the contact rows
    H = dense_hessian(T, res.statics)[0]
    rows = []
    tang = friction if constrain_tangential is None else constrain_tangential
    for c in np.flatnonzero(active):
        use = [0, 1, 2] if tang else [0]
        b = int(cand.body[c])
        j = T.anc[b, :T.depth[b] + 1]
        for r in use:
            d = frame[c, r]
            row = np.zeros((T.nb, 3))
            # d . dp/d(delta_j) = R_j^T ((p - o_j) x d)
            row[j] = np.einsum("jki,jk->ji", R[0, j], _cross(cand.p[c] - o[0, j], d))
            rows.append(row.reshape(-1))
    if rows:
        J = np.array(rows)
        _, sv, Vt = np.linalg.svd(J)
        rank = int((sv > 1e-12 * sv.max()).sum())
        Z = Vt[rank:].T
        Hc = Z.T @ (0.5 * (H + H.T)) @ Z
    else:
        Hc = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(Hc)
    nneg = int((ev < -1e-12 * np.abs(ev).max()).sum())
    feasible = bool(res.feasible[0])
    status = "infeasible" if not feasible else ("stable" if nneg == 0 else "unstable")
    return SupportedInverse(res, forces, cand.body, cand.p, frame, fvec, active, pre, pre0,
                            int(res.n_negative[0]), nneg, status)
