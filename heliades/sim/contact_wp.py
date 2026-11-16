"""Warp kernel for the contact PGS (B3): the same algorithm as contact.pgs (numpy reference),
one thread per batch element. Gauss-Seidel is sequential over the contacts, so numpy pays
Python overhead per contact and per sweep; this kernel removes it. Float64 throughout.

The kernel takes W^T (rows = columns of W, contiguous) and the initial v = v_free + W lam0.
Per contact: normal (scalar projection) then the exact local disk QP for the tangential
impulse in the W_tt metric (secular equation by Newton), as in contact._disk_qp.
"""
from __future__ import annotations

import numpy as np

try:
    import warp as wp
    _OK = True
except Exception:  # pragma: no cover
    wp = None
    _OK = False

_initialised = False


def available():
    return _OK


if _OK:
    @wp.kernel
    def _pgs_kernel(WT: wp.array3d(dtype=wp.float64), v: wp.array2d(dtype=wp.float64),
                    lam: wp.array2d(dtype=wp.float64), target: wp.array2d(dtype=wp.float64),
                    mu: wp.array2d(dtype=wp.float64), valid: wp.array2d(dtype=wp.int32),
                    lam_max: wp.array2d(dtype=wp.float64),
                    iters: int, tol: wp.float64, sor: wp.float64, n_newton: int,
                    out_it: wp.array(dtype=wp.int32), out_res: wp.array(dtype=wp.float64)):
        b = wp.tid()
        K = valid.shape[1]
        n = 3 * K
        zero = wp.float64(0.0)
        one = wp.float64(1.0)
        tiny = wp.float64(1e-300)
        it_done = int(0)
        res = zero
        for it in range(iters):
            dmax = zero
            for c in range(K):
                if valid[b, c] == 0:
                    continue
                i = 3 * c
                wnn = WT[b, i, i]
                if wnn <= tiny:
                    wnn = one
                ln = lam[b, i]
                lnn = wp.min(wp.max(ln + sor * (target[b, c] - v[b, i]) / wnn, zero), lam_max[b, c])
                d = lnn - ln
                if d != zero:
                    lam[b, i] = lnn
                    for r in range(n):
                        v[b, r] = v[b, r] + WT[b, i, r] * d
                # tangential: v_t = A lam_t + bb (A = W_tt, row-major from W^T)
                a11 = WT[b, i + 1, i + 1]
                a12 = WT[b, i + 2, i + 1]
                a21 = WT[b, i + 1, i + 2]
                a22 = WT[b, i + 2, i + 2]
                l1 = lam[b, i + 1]
                l2 = lam[b, i + 2]
                b1 = v[b, i + 1] - (a11 * l1 + a12 * l2)
                b2 = v[b, i + 2] - (a21 * l1 + a22 * l2)
                cap = mu[b, c] * lnn
                det = a11 * a22 - a12 * a21
                if wp.abs(det) <= tiny:
                    det = one
                x1 = -(a22 * b1 - a12 * b2) / det
                x2 = -(-a21 * b1 + a11 * b2) / det
                if x1 * x1 + x2 * x2 > cap * cap:
                    if cap <= zero:
                        x1 = zero
                        x2 = zero
                    else:
                        s11 = a11
                        s22 = a22
                        s12 = wp.float64(0.5) * (a12 + a21)
                        half = wp.float64(0.5) * (s11 + s22)
                        rad = wp.sqrt(wp.float64(0.25) * (s11 - s22) * (s11 - s22) + s12 * s12)
                        emax = half + rad
                        g = wp.max(wp.sqrt(b1 * b1 + b2 * b2) / cap - emax, zero)
                        for _k in range(n_newton):
                            m11 = s11 + g
                            m22 = s22 + g
                            dt = m11 * m22 - s12 * s12
                            if wp.abs(dt) <= tiny:
                                dt = tiny
                            q1 = -(m22 * b1 - s12 * b2) / dt
                            q2 = -(-s12 * b1 + m11 * b2) / dt
                            # dq/dg = -(S + g I)^-1 q
                            e1 = -(m22 * q1 - s12 * q2) / dt
                            e2 = -(-s12 * q1 + m11 * q2) / dt
                            nq = wp.sqrt(q1 * q1 + q2 * q2) + tiny
                            dn = (q1 * e1 + q2 * e2) / nq
                            phi = one / nq - one / cap
                            dphi = -dn / (nq * nq)
                            if wp.abs(dphi) <= tiny:
                                break
                            g = wp.max(g - phi / dphi, zero)
                        m11 = s11 + g
                        m22 = s22 + g
                        dt = m11 * m22 - s12 * s12
                        if wp.abs(dt) <= tiny:
                            dt = tiny
                        x1 = -(m22 * b1 - s12 * b2) / dt
                        x2 = -(-s12 * b1 + m11 * b2) / dt
                        nx = wp.sqrt(x1 * x1 + x2 * x2)
                        if nx > tiny:
                            x1 = x1 * cap / nx
                            x2 = x2 * cap / nx
                x1 = l1 + sor * (x1 - l1)
                x2 = l2 + sor * (x2 - l2)
                d1 = x1 - l1
                d2 = x2 - l2
                if d1 != zero or d2 != zero:
                    lam[b, i + 1] = x1
                    lam[b, i + 2] = x2
                    for r in range(n):
                        v[b, r] = v[b, r] + WT[b, i + 1, r] * d1 + WT[b, i + 2, r] * d2
                am = wp.max(wp.max(wp.abs(a11), wp.abs(a22)), wp.max(wp.abs(a12), wp.abs(a21)))
                dmax = wp.max(dmax, wp.max(wp.abs(d) * wnn, wp.max(wp.abs(d1), wp.abs(d2)) * am))
            it_done = it + 1
            res = dmax
            if dmax <= tol:
                break
        out_it[b] = it_done
        out_res[b] = res


def pgs_warp(W, v_free, target, mu, valid, lam0=None, iters=200, tol=1e-6, sor=1.0, n_newton=12, lam_max=None):
    """Same contract as contact.pgs (numpy). Returns (lam (B, K, 3), max iterations, residual)."""
    global _initialised
    if not _initialised:
        wp.init()
        _initialised = True
    B, K = valid.shape
    lam = np.zeros((B, 3 * K)) if lam0 is None else np.where(valid[..., None], lam0, 0.0).reshape(B, 3 * K)
    v = v_free + np.einsum("bij,bj->bi", W, lam)
    dev = "cpu"
    WT = wp.array(np.ascontiguousarray(np.swapaxes(W, 1, 2)), dtype=wp.float64, device=dev)
    vw = wp.array(np.ascontiguousarray(v), dtype=wp.float64, device=dev)
    lw = wp.array(np.ascontiguousarray(lam), dtype=wp.float64, device=dev)
    tw = wp.array(np.ascontiguousarray(target, dtype=np.float64), dtype=wp.float64, device=dev)
    mw = wp.array(np.ascontiguousarray(mu, dtype=np.float64), dtype=wp.float64, device=dev)
    vaw = wp.array(np.ascontiguousarray(valid.astype(np.int32)), dtype=wp.int32, device=dev)
    lm = np.full((B, K), 1e300) if lam_max is None else np.minimum(np.asarray(lam_max, float), 1e300)
    lmw = wp.array(np.ascontiguousarray(lm), dtype=wp.float64, device=dev)
    oi = wp.zeros(B, dtype=wp.int32, device=dev)
    orr = wp.zeros(B, dtype=wp.float64, device=dev)
    wp.launch(_pgs_kernel, dim=B, inputs=[WT, vw, lw, tw, mw, vaw, lmw, int(iters), float(tol), float(sor),
                                          int(n_newton), oi, orr], device=dev)
    return lw.numpy().reshape(B, K, 3), int(oi.numpy().max()), float(orr.numpy().max())
