"""Contact slot of the GPU step (B4 -> B3 interface). See docs/sim/GPU.md section 6.

B3 (contact + ground, Baraff-style O(kn) Schur complement on the tree factorisation) plugs in
here. A step on the GPU is

    up sweep (factorise)  ->  [ContactSolver.solve(sim, h)]  ->  down sweep (integrate)

and `solve` runs *inside* the captured CUDA graph, so it must only launch kernels on fixed,
preallocated device buffers (no host syncs, no allocation). It sees the complete
factorisation of this step's implicit system (M + hD + h^2 K~ + implicit terms, in
articulated-body form) with the kinematics still at q_n, and it has three primitives:

  sim.slot_free_motion(acc0, acc1, udot)
      contact-free joint accelerations udot and body accelerations [acc0; acc1]
      (classical, about the body origin o, world frame) of this step: one O(n) sweep.
  sim.factor_response_device(df0, df1, dP0, dP1, du, da0, da1, dud)
      linear response (d udot, d acc) to extra spatial body forces [df0; df1] (moment about o;
      force), world frame: two O(n) sweeps on the stored factor, no refactorisation.
      k of these give the k x k Delassus operator  G A^-1 G^T  column by column (O(kn)),
      the operator a PGS / LCP on k unknowns needs.
  sim.slot_apply(df0, df1, dP0, dP1, du)
      fold the solved contact forces into this step (u~ += d u~); the regular down sweep then
      integrates the contact-coupled motion exactly (linearity of the factor in the forces).

Units: velocity-level contact (impulses p) enter as forces f = p / h, since the implicit step
solves for accelerations a with u_{n+1} = u_n + h a.

Expected B3 pipeline per step (k contacts per env, all envs in lock-step):
  1. detect   : kernel over contact candidates -> (env, body, local point, normal, depth),
                compacted to k_max per env (device counters; fixed capacity, graph-safe)
  2. free     : slot_free_motion -> contact-point normal velocities v_free = G (u_n + h a_free)
  3. Delassus : k_max response solves batched as k_max extra "right-hand sides" (or the
                O(k * depth) path-only variant: only the ancestors of a contact body change in the
                up sweep), giving W_e = G A^-1 G^T (k x k per env)
  4. PGS      : projected Gauss-Seidel on 0 <= lambda  _|_  W lambda + v_free + b >= 0 per env
                (one thread block per env; k is small)
  5. apply    : slot_apply(G^T lambda / h)
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import warp as wp


class ContactSolver(Protocol):
    """Interface for B3's contact solve on the GPU (set `sim.contact = solver`)."""

    def solve(self, sim, h: float) -> None:
        """Called inside the step graph between the factorisation and the integration sweep.
        Must only launch kernels on preallocated device buffers."""
        ...


class ConstantBodyForces:
    """Minimal ContactSolver: applies fixed spatial body forces through the contact slot
    (slot_apply). Used to test the slot against TreeSim PointForce steps; also a template
    for B3 (allocate scratch once, launch only inside `solve`)."""

    def __init__(self, sim, df):
        k, L, dev = sim.k, sim.lay, sim.device
        dfg = L.from_local(np.asarray(df, float))
        self.df0 = wp.array(dfg[:, :3].astype(sim.dt), dtype=k.V3, device=dev)
        self.df1 = wp.array(dfg[:, 3:].astype(sim.dt), dtype=k.V3, device=dev)
        z = lambda: wp.zeros(L.n, dtype=k.V3, device=dev)
        self.dP0, self.dP1, self.du = z(), z(), z()

    def solve(self, sim, h):
        sim.slot_apply(self.df0, self.df1, self.dP0, self.dP1, self.du)


class ContactSlotSolver:
    """B3 contact (implicit, velocity-level, Coulomb cone; heliades.sim.contact) through the B4
    slot. The same staged pipeline as the CPU TreeSim step (contact.solve_contacts):

      1. detect   : contact.prepare_contacts on a host mirror of the kinematics at q_n
      2. free     : sim.slot_free_motion -> u* = u_n + h udot_free -> v_free = G u*
      3. operator : 3 K_max columns of sim.factor_response_device (each batched over all envs:
                    unit force along n / t1 / t2 at every env's c-th contact) -> W = G A^-1 G^T,
                    i.e. Baraff's O(kn) column construction (the CPU uses the equivalent
                    O(k depth + k^2) path formula; both are exact)
      4. PGS      : contact.solve_impulses (same targets, warm start, safeguards, Warp PGS)
      5. apply    : sim.slot_apply(contact forces = impulses / h)

    This first wiring is host-synchronous (detection, operator assembly and PGS read device
    buffers each step), so it requires TreeSimGPU(..., use_graph=False). What a graph-resident
    version needs: docs/sim/CONTACT.md sec. 11. Shared-topology batches only."""

    def __init__(self, sim, colliders, params=None):
        from ..contact import ContactParams
        from ..tree import TreeSim
        if sim.hetero:
            raise NotImplementedError("ContactSlotSolver: shared-topology batches only")
        if sim.use_graph:
            raise ValueError("ContactSlotSolver is host-synchronous: create TreeSimGPU(..., use_graph=False)")
        self.colliders = list(colliders)
        self.params = params or ContactParams()
        # host mirror: geometry, contact features and kinematics of the same tree (float64)
        self.mirror = TreeSim(sim.asset, [sim.asset.physics] * sim.B, contact_params=self.params)
        k, L, dev = sim.k, sim.lay, sim.device
        z = lambda: wp.zeros(L.n, dtype=k.V3, device=dev)
        self._acc0, self._acc1, self._ud = z(), z(), z()
        self._df0, self._df1, self._dP0, self._dP1, self._du = z(), z(), z(), z(), z()
        self._da0, self._da1, self._dud = z(), z(), z()
        self.cache = None
        self.result = None          # last step: dict(stage, lam, force, iterations, residual, W)

    def _vel(self, u, body, p):
        from ..contact import point_velocity
        m = self.mirror
        return point_velocity(m, u, m.v_float, body, p)

    def _put(self, sim, F):
        """(B, nb, 6) host spatial forces [moment about o; force] -> device df0 / df1."""
        g = sim.lay.from_local(F)
        self._df0.assign(g[:, :3].astype(sim.dt))
        self._df1.assign(g[:, 3:].astype(sim.dt))

    def solve(self, sim, h):
        from ..contact import _contact_model, contact_body_forces, free_velocities, prepare_contacts, solve_impulses
        from ..tree import _cross
        L, m = sim.lay, self.mirror
        m.set_rest(sim.R0rel)
        m.Q, m.u = sim.Q, sim.u
        m._kin_cache = None
        k = m._kin()
        _, prm = _contact_model(m)
        st = prepare_contacts(m, k, self.colliders, h, prm)                     # 1. detect
        if st is None:
            self.cache, self.result = None, None
            return
        sim.slot_free_motion(self._acc0, self._acc1, self._ud)                    # 2. free motion
        u_free = m.u + h * L.to_local(self._ud.numpy().astype(float))
        v_free = free_velocities(st, lambda b, p: self._vel(u_free, b, p))
        B, K = st.valid.shape
        nb = m.topo.nb
        W = np.zeros((B, 3 * K, 3 * K))                                           # 3. operator
        bi = np.arange(B)
        rr = st.p - k.o[bi[:, None], st.body]
        for c in range(K):
            v = st.valid[:, c]
            if not v.any():
                continue
            for i in range(3):
                d = st.D[:, c, i]
                F = np.zeros((B, nb, 6))
                F[bi[v], st.body[v, c]] = np.concatenate([_cross(rr[:, c], d), d], -1)[v]
                self._put(sim, F)
                sim.factor_response_device(self._df0, self._df1, self._dP0, self._dP1, self._du,
                                           self._da0, self._da1, self._dud)
                dud = L.to_local(self._dud.numpy().astype(float))
                dv = np.einsum("bkij,bkj->bki", st.D, self._vel(dud, st.body, st.p)).reshape(B, -1)
                W[:, :, 3 * c + i] = dv * v[:, None]
        msk = np.repeat(st.valid, 3, axis=1)
        W *= msk[:, :, None]
        lam, lam_w, iters, rel, alpha, self.cache = solve_impulses(              # 4. PGS
            st, W, v_free, h, prm, self.colliders, self.cache)
        F = contact_body_forces(st, lam_w, h, k.o, nb)                           # 5. apply
        self._put(sim, F)
        sim.slot_apply(self._df0, self._df1, self._dP0, self._dP1, self._du)
        self.result = {"stage": st, "lam": lam, "force": lam_w / h * st.valid[..., None], "iterations": iters,
                       "residual": rel, "alpha": alpha, "W": W, "v_free": v_free}

    def total_force(self, collider=None):
        """(B, 3) summed world contact force on the plant of the last step (None: no contact)."""
        if self.result is None:
            return None
        st = self.result["stage"]
        msk = st.valid if collider is None else st.valid & (st.col == collider)
        return (self.result["force"] * msk[..., None]).sum(1)
