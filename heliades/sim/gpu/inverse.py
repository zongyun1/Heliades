"""B2 inverse statics + Sturm stability count on the GPU, batched over B (docs/sim/GPU.md §4).

Same algorithm and result type as heliades.sim.inverse.inverse_statics (docs/sim/INVERSE.md):

  root -> leaf   inv_fk_level  per level: target kinematics R_j, o_j
  leaf -> root   inv_up_level  per level, fused: subtree loads (segmented sum), joint moment
                 m_j = R_j^T N_j, local 3x3 Newton for the deviation quaternion, rest rotation,
                 exact tangent-stiffness blocks, and the symmetric block elimination with 3x3
                 pivots (pivot eigenvalues -> Sturm count)

so one inverse with its stability check is 2 * nlev launches for any B. Shifted Sturm counts
(elim_level) give lambda_min and the buckling margin by bisection, all envs in lock-step.
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

from ..inverse import InverseResult
from ..tree import PointForce


class _InvBuffers:
    def __init__(self, sim):
        k, n, dev = sim.k, sim.lay.n, sim.device
        z = lambda dt: wp.zeros(n, dtype=dt, device=dev)
        self.R, self.o, self.tip = z(k.M33), z(k.V3), z(k.V3)
        self.Fs, self.Ws = z(k.V3), z(k.M33)
        self.Q, self.ok, self.torque, self.R0 = z(k.V4), z(wp.int32), z(k.V3), z(k.M33)
        self.Dw, self.A, self.Kw, self.Phi, self.eig = z(k.M33), z(k.M33), z(k.M33), z(k.M33), z(k.V3)
        self.jneg = z(wp.int32)
        self.nneg = wp.zeros(sim.B, dtype=wp.int32, device=dev)
        self.nbad = wp.zeros(sim.B, dtype=wp.int32, device=dev)
        self.Rt = z(k.M33)


def _sync_time():
    wp.synchronize()
    return time.perf_counter()


def inverse_statics_gpu(sim, target_Rrel=None, external=None, stability: bool = True,
                        min_eig: bool = False, rtol: float | None = None, full: bool = True) -> InverseResult:
    """Rest shape making `target_Rrel` ((B or 1, nb, 3, 3); default: the asset geometry) an
    exact equilibrium of `sim`'s discrete model (gravity + dead PointForce loads), with the
    O(n) Sturm count of the tangent stiffness. `sim` is a TreeSimGPU (shared topology);
    it is not modified. Returns heliades.sim.inverse.InverseResult (host arrays).
    full=False reads back only the per-env summary (status, feasible, n_negative, stable;
    per-joint fields are None): the fast path for screening thousands of (asset, E) pairs.
    Device buffers are cached on `sim` and reused by the next call (res.device_buffers)."""
    k, L, dev, B = sim.k, sim.lay, sim.device, sim.B
    if not L.shared:
        raise NotImplementedError("inverse_statics_gpu: shared topology only (kernels are generic; "
                                  "result packing per env is not implemented)")
    T = sim.topo
    t0 = _sync_time()
    buf = getattr(sim, "_inv_buf", None)
    if buf is None:
        buf = sim._inv_buf = _InvBuffers(sim)
    buf.nneg.zero_()
    buf.nbad.zero_()
    if target_Rrel is None:
        Rt_g = L.R0rel
    else:
        Rt = np.broadcast_to(np.asarray(target_Rrel, float), (B, T.nb, 3, 3))
        Rt_g = L.from_local(Rt)
    buf.Rt.assign(np.asarray(Rt_g, sim.dt))
    # dead point loads
    for e in external or []:
        if not isinstance(e, PointForce):
            raise TypeError("inverse statics supports dead PointForce loads only")
    ek = sim._set_external(list(external) if external else None)
    use_pf = 0
    if ek[0] == "sparse":
        sim.d_body_force.zero_()
        wp.launch(k.scatter_forces, dim=ek[1], device=dev, inputs=[sim.d_pf_idx, sim.d_pf_f, sim.d_body_force])
        use_pf = 1
    grav = k.V3(*[float(x) for x in sim.gravity])
    for d in range(L.nlev):
        wp.launch(k.inv_fk_level, dim=int(L.L[d + 1] - L.L[d]), device=dev,
                  inputs=[int(L.L[d]), sim.d_parent, buf.Rt, sim.d_o0, sim.d_seg_l, buf.R, buf.o, buf.tip])
    for d in range(L.nlev - 1, -1, -1):
        wp.launch(k.inv_up_level, dim=int(L.L[d + 1] - L.L[d]), device=dev,
                  inputs=[int(L.L[d]), sim.d_parent, sim.d_env, sim.d_ch_start, sim.d_ch_count, sim.d_ch_list,
                          buf.R, buf.o, buf.tip, sim.d_m, sim.d_h1, sim.d_K, use_pf, sim.d_body_force, grav,
                          buf.Rt, int(stability), buf.Fs, buf.Ws, buf.Q, buf.ok, buf.torque, buf.R0,
                          buf.Dw, buf.A, buf.Kw, buf.Phi, buf.eig, buf.jneg, buf.nneg, buf.nbad])
    t1 = _sync_time()
    loc = lambda a: L.to_local(a.numpy().astype(float))
    if full:
        Q = loc(buf.Q)
        jok = L.to_local(buf.ok.numpy()) > 0
        feasible = jok.all(1)
        vv = np.linalg.norm(Q[..., 1:], axis=-1)
        angle = 2 * np.arctan2(vv, Q[..., 0])
        res = InverseResult(loc(buf.R0), Q, loc(buf.torque), angle, jok, feasible, np.zeros(B, int),
                            np.zeros((B, T.nb), int), feasible.copy(), [])
    else:
        feasible = buf.nbad.numpy() == 0
        res = InverseResult(None, None, None, None, None, feasible, np.zeros(B, int), None, feasible.copy(), [])
    if stability:
        res.n_negative = buf.nneg.numpy().astype(int)
        if full:
            res.joint_negative = L.to_local(buf.jneg.numpy()).astype(int)
        res.stable = feasible & (res.n_negative == 0)
        if min_eig:
            res.min_eig = min_eigenvalue_gpu(sim, buf, "euclid", rtol=rtol)
            res.margin = min_eigenvalue_gpu(sim, buf, "K", rtol=rtol)
            E = np.array([p.stem_youngs_modulus for p in sim.params])
            res.E_cr_linear = E * (1.0 - res.margin)
    t2 = _sync_time()
    res.status = ["infeasible" if not f else ("stable" if s else "unstable")
                  for f, s in zip(res.feasible, res.stable)]
    res.timing = {"gpu_inverse_and_count": t1 - t0, "readback_and_min_eig": t2 - t1, "total": t2 - t0}
    res.device_buffers = buf
    return res


def count_below_gpu(sim, buf: _InvBuffers, sigma, metric: str = "euclid", out=None):
    """Number of eigenvalues of H (metric='euclid') or of the pencil (H, K_el) (metric='K')
    strictly below sigma (B,) for every env, from the buffers of inverse_statics_gpu.
    One shifted O(n) elimination (nlev launches)."""
    k, L, dev = sim.k, sim.lay, sim.device
    shift = sigma if isinstance(sigma, wp.array) else wp.array(np.asarray(sigma, sim.dt).reshape(-1), dtype=k.F,
                                                               device=dev)
    cnt = out if out is not None else wp.zeros(sim.B, dtype=wp.int32, device=dev)
    cnt.zero_()
    _elim(sim, buf, shift, 1 if metric == "K" else 0, cnt)
    return cnt


def _elim(sim, buf, shift, use_K, cnt):
    k, L, dev = sim.k, sim.lay, sim.device
    for d in range(L.nlev - 1, -1, -1):
        wp.launch(k.elim_level, dim=int(L.L[d + 1] - L.L[d]), device=dev,
                  inputs=[int(L.L[d]), sim.d_env, sim.d_ch_start, sim.d_ch_count, sim.d_ch_list,
                          buf.Dw, buf.A, buf.Kw, shift, use_K, buf.Phi, buf.eig, cnt])


def min_eigenvalue_gpu(sim, buf: _InvBuffers, metric: str = "euclid", rtol: float | None = None,
                       max_iter: int = 200, check_every: int = 8):
    """Smallest eigenvalue of H (or generalized eigenvalue of (H, K_el)) by Sturm bisection,
    all envs in lock-step on the device (hessian.min_eigenvalue). Returns (B,)."""
    k, L, dev, B = sim.k, sim.lay, sim.device, sim.B
    if rtol is None:
        rtol = 1e-10 if sim.precision == "float64" else 1e-6
    use_K = 1 if metric == "K" else 0
    if use_K:
        lo, hi = np.full(B, -1.0), np.full(B, 1.0)
    else:
        sD = wp.zeros(B, dtype=k.F, device=dev)
        sA = wp.zeros(B, dtype=k.F, device=dev)
        wp.launch(k.gersh, dim=L.n, device=dev, inputs=[sim.d_env, buf.Dw, buf.A, sD, sA])
        s = sD.numpy().astype(float) + sA.numpy().astype(float) * L.nlev
        lo, hi = -s, s.copy()
    cnt = wp.zeros(B, dtype=wp.int32, device=dev)
    d_shift = wp.zeros(B, dtype=k.F, device=dev)

    def count(x):
        d_shift.assign(np.asarray(x, sim.dt))
        cnt.zero_()
        _elim(sim, buf, d_shift, use_K, cnt)
        return cnt.numpy()

    for _ in range(200):                       # bracket: count(lo) == 0, count(hi) >= 1
        c = count(lo)
        if (c == 0).all():
            break
        lo = np.where(c > 0, 2 * lo - hi, lo)
    for _ in range(200):
        c = count(hi)
        if (c >= 1).all():
            break
        hi = np.where(c < 1, hi + 2 * (hi - lo), hi)
    d_lo = wp.array(lo.astype(sim.dt), dtype=k.F, device=dev)
    d_hi = wp.array(hi.astype(sim.dt), dtype=k.F, device=dev)
    d_mid = wp.zeros(B, dtype=k.F, device=dev)

    def chunk():
        for _ in range(check_every):
            wp.launch(k.bis_mid, dim=B, device=dev, inputs=[d_lo, d_hi, d_mid, cnt])
            _elim(sim, buf, d_mid, use_K, cnt)
            wp.launch(k.bis_update, dim=B, device=dev, inputs=[d_lo, d_hi, d_mid, cnt])

    graph = None
    if sim.use_graph:
        with wp.ScopedCapture(device=dev) as cap:
            chunk()
        graph = cap.graph
    it = 0
    lo_h, hi_h = lo, hi
    while it < max_iter:
        if graph is not None:
            wp.capture_launch(graph)
        else:
            chunk()
        it += check_every
        lo_h, hi_h = d_lo.numpy().astype(float), d_hi.numpy().astype(float)
        if (hi_h - lo_h <= rtol * np.maximum(np.abs(hi_h), 1e-300)).all():
            break
    return 0.5 * (lo_h + hi_h)


def critical_modulus_gpu(sim, target_Rrel=None, external=None, E_lo=1e5, E_hi=1e12, rtol=1e-4,
                         n_probe=16, base=None, stiffness_scale=None):
    """heliades.sim.inverse.critical_modulus with every probe round as one GPU batch
    (n_probe moduli at once). Returns dict(E_cr, lo, hi, rounds, E_cr_linear_at_hi)."""
    from .sim import TreeSimGPU
    p0 = sim.params[0] if base is None else base
    tgt = None if target_Rrel is None else np.asarray(target_Rrel)[:1]
    if tgt is not None and tgt.ndim == 3:
        tgt = tgt[None]

    def probe_sim(Es):
        s = TreeSimGPU(sim.asset, [p0.copy(stem_youngs_modulus=float(E)) for E in Es], gravity=sim.gravity,
                       joint_damping=sim.joint_damping, precision=sim.precision, device=str(sim.device),
                       use_graph=False, topology=sim.topo)
        if stiffness_scale is not None:
            s.K = s.K * np.asarray(stiffness_scale, float).reshape(1, -1, 1)
        return s

    lo, hi = float(E_lo), float(E_hi)
    rounds = 0
    while hi / lo - 1 > rtol:
        Es = np.geomspace(lo, hi, n_probe)
        r = inverse_statics_gpu(probe_sim(Es), tgt, external, full=False)
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
    r = inverse_statics_gpu(probe_sim([hi]), tgt, external, min_eig=True)
    return {"E_cr": float(np.sqrt(lo * hi)), "lo": lo, "hi": hi, "rounds": rounds,
            "E_cr_linear_at_hi": float(r.E_cr_linear[0])}
