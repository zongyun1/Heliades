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
