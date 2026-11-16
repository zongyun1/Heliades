"""Warp kernels for the GPU tree solver (B4). One kernel set per floating-point precision.

Everything is written over a flattened *global body* index g (see layout.py): bodies of all
environments are numbered level by level, so depth level d is the contiguous range
[L_d, L_{d+1}). Per-level kernels are launched with dim = L_{d+1} - L_d and g = L_d + tid.
Parents live in level d-1 and children in level d+1; children of one body are listed in a
CSR (ch_start, ch_count, ch_list). No atomics are used in the solver sweeps.

Math is a 1:1 port of heliades/sim/tree.py (TreeSim.step) and heliades/sim/hessian.py /
inverse.py (B2), written in 3x3 blocks:

  spatial inertia  [[A00, A01], [A10, A11]]   spatial vectors [ang; lin]
  joint map        S' = [Smt; 0]              (Smt = R + h [w_p]x R)
  motion map       X' = [[Xa, 0], [Xc, I]]    (Xa = I - h [w_rel]x, Xc = -[r]x + h(...))
  force map        X^T = [[I, [r]x], [0, I]]

`get_kernels(precision)` returns a namespace with the kernels and the vector/matrix types.
"""

import functools
from types import SimpleNamespace

import warp as wp

wp.config.quiet = True


@functools.lru_cache(maxsize=None)
def get_kernels(precision: str = "float32") -> SimpleNamespace:
    if precision not in ("float32", "float64"):
        raise ValueError(f"precision must be 'float32' or 'float64', got {precision!r}")
    f64 = precision == "float64"
    F = wp.float64 if f64 else wp.float32
    V3 = wp.vec3d if f64 else wp.vec3f
    V4 = wp.vec4d if f64 else wp.vec4f
    M33 = wp.mat33d if f64 else wp.mat33f
    TINY = 1e-300 if f64 else 1e-30
    # local joint Newton (B2): residual tolerance, convergence check, |v| < 1 guard
    TOLN = 1e-15 if f64 else 4e-7
    CTOL = 1e-9 if f64 else 2e-5
    INS = 1e-12 if f64 else 1e-6
    BIG = 1e300 if f64 else 1e30

    # ------------------------------------------------------------------ helpers
    @wp.func
    def eye3(x: F) -> M33:
        return M33(x, F(0.0), F(0.0), F(0.0), x, F(0.0), F(0.0), F(0.0), x)

    @wp.func
    def qmul(a: V4, b: V4) -> V4:
        return V4(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])

    @wp.func
    def quat_from_rotvec(phi: V3) -> V4:
        th2 = wp.dot(phi, phi)
        th = wp.sqrt(th2)
        s = F(0.0)
        if th < F(1e-4):
            s = F(0.5) - th2 / F(48.0)
        else:
            s = wp.sin(F(0.5) * th) / th
        return V4(wp.cos(F(0.5) * th), s * phi[0], s * phi[1], s * phi[2])

    @wp.func
    def quat_to_matrix(q: V4) -> M33:
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        one = F(1.0)
        two = F(2.0)
        return M33(one - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y),
                   two * (x * y + w * z), one - two * (x * x + z * z), two * (y * z - w * x),
                   two * (x * z - w * y), two * (y * z + w * x), one - two * (x * x + y * y))

    @wp.func
    def col2(R: M33) -> V3:
        return V3(R[0, 2], R[1, 2], R[2, 2])

    @wp.func
    def grav_psd(g: V3, mc: V3) -> M33:
        """PSD part of (g.mc) I - sym(g mc^T) |g| (closed-form eigen split; tree.py)."""
        Z = eye3(F(0.0))
        gn = wp.length(g)
        if gn == F(0.0):
            return Z
        f = g / gn
        cm = wp.length(mc)
        e = mc / wp.max(cm, F(TINY))
        gc = wp.dot(mc, g)
        K = eye3(gc) - (F(0.5) * gn) * (wp.outer(f, mc) + wp.outer(mc, f))
        dm = f + e
        sm = wp.length(dm)
        dm = dm / wp.max(sm, F(TINY))
        lam_m = F(0.5) * (gc - gn * cm)
        nv = wp.cross(f, e)
        sn = wp.length(nv)
        nv = nv / wp.max(sn, F(TINY))
        Kp = K - lam_m * wp.outer(dm, dm) - wp.min(gc, F(0.0)) * wp.outer(nv, nv)
        if sm < F(1e-6) or cm < F(TINY):
            return Z
        return Kp

    @wp.func
    def fk_body(g: int, p: int, Q: V4, u: V3,
                R0rel: wp.array(dtype=M33), o0: wp.array(dtype=V3), seg_l: wp.array(dtype=F),
                Ra: wp.array(dtype=M33), oa: wp.array(dtype=V3), tipa: wp.array(dtype=V3),
                wa: wp.array(dtype=V3), vOa: wp.array(dtype=V3)):
        Rrel = R0rel[g] * quat_to_matrix(Q)
        R = Rrel
        o = o0[g]
        w = Rrel * u
        vO = V3(F(0.0), F(0.0), F(0.0))
        if p >= 0:
            R = Ra[p] * Rrel
            o = tipa[p]
            wpar = wa[p]
            w = wpar + R * u
            vO = vOa[p] + wp.cross(wpar, o - oa[p])
        Ra[g] = R
        oa[g] = o
        wa[g] = w
        vOa[g] = vO
        tipa[g] = o + seg_l[g] * col2(R)

    # ------------------------------------------------------------------ forward kinematics
    @wp.kernel(module="unique")
    def fk_level(L0: int, parent: wp.array(dtype=int), Qa: wp.array(dtype=V4), ua: wp.array(dtype=V3),
                 R0rel: wp.array(dtype=M33), o0: wp.array(dtype=V3), seg_l: wp.array(dtype=F),
                 Ra: wp.array(dtype=M33), oa: wp.array(dtype=V3), tipa: wp.array(dtype=V3),
                 wa: wp.array(dtype=V3), vOa: wp.array(dtype=V3)):
        g = L0 + wp.tid()
        fk_body(g, parent[g], Qa[g], ua[g], R0rel, o0, seg_l, Ra, oa, tipa, wa, vOa)

    # ------------------------------------------------------------------ external forces
    @wp.kernel(module="unique")
    def scatter_forces(idx: wp.array(dtype=int), f: wp.array(dtype=V3), out: wp.array(dtype=V3)):
        k = wp.tid()
        out[idx[k]] = f[k]

    @wp.kernel(module="unique")
    def node_forces_to_bodies(gnode: wp.array(dtype=int), fnode: wp.array(dtype=V3), out: wp.array(dtype=V3)):
        g = wp.tid()
        out[g] = fnode[gnode[g]]

    # ------------------------------------------------------------------ leaf -> root sweep
    @wp.kernel(module="unique")
    def aba_up(L0: int, done: wp.array(dtype=int),
               parent: wp.array(dtype=int), env: wp.array(dtype=int),
               ch_start: wp.array(dtype=int), ch_count: wp.array(dtype=int), ch_list: wp.array(dtype=int),
               Qa: wp.array(dtype=V4), ua: wp.array(dtype=V3),
               Ra: wp.array(dtype=M33), oa: wp.array(dtype=V3), tipa: wp.array(dtype=V3),
               wa: wp.array(dtype=V3), vOa: wp.array(dtype=V3),
               m_a: wp.array(dtype=F), h1_a: wp.array(dtype=V3), Io_a: wp.array(dtype=M33),
               K_a: wp.array(dtype=V3), alpha_a: wp.array(dtype=F),
               use_pf: int, body_force: wp.array(dtype=V3),
               n_sph: int, cp_start: wp.array(dtype=int), cp_count: wp.array(dtype=int),
               cp_local: wp.array(dtype=V3),
               sph_c: wp.array(dtype=V3), sph_v: wp.array(dtype=V3), sph_p: wp.array(dtype=V3),
               grav: V3, h: F, beta: F, gyro: int, gimp: int,
               upI00: wp.array(dtype=M33), upI01: wp.array(dtype=M33),
               upI10: wp.array(dtype=M33), upI11: wp.array(dtype=M33),
               upP0: wp.array(dtype=V3), upP1: wp.array(dtype=V3),
               Xa_a: wp.array(dtype=M33), Xc_a: wp.array(dtype=M33), Smt_a: wp.array(dtype=M33),
               W0_a: wp.array(dtype=M33), W1_a: wp.array(dtype=M33), Dinv_a: wp.array(dtype=M33),
               UD0_a: wp.array(dtype=M33), UD1_a: wp.array(dtype=M33),
               uu_a: wp.array(dtype=V3), c0_a: wp.array(dtype=V3), c1_a: wp.array(dtype=V3),
               r_a: wp.array(dtype=V3)):
        g = L0 + wp.tid()
        if done[0] != 0:
            return
        p = parent[g]
        e = env[g]
        R = Ra[g]
        RT = wp.transpose(R)
        w = wa[g]
        vO = vOa[g]
        o = oa[g]
        u = ua[g]
        alpha = alpha_a[e]
        I3 = eye3(F(1.0))

        # ---- body inertia, bias, gravity, damping (pass 1b)
        IO = R * Io_a[g] * RT
        mc = R * h1_a[g]
        m = m_a[g]
        Smc = wp.skew(mc)
        IOw = IO * w
        Ibv0 = IOw + wp.cross(mc, vO)
        Ibv1 = wp.cross(w, mc) + m * vO
        b0 = wp.cross(w, IOw)
        b1 = wp.cross(w, wp.cross(w, mc))
        f0 = wp.cross(mc, grav) - alpha * Ibv0
        f1 = m * grav - alpha * Ibv1
        s = F(1.0) + h * alpha
        I00 = s * IO
        I01 = s * Smc
        I10 = (-s) * Smc
        I11 = eye3(s * m)
        if gyro != 0:
            I00 = I00 + h * (wp.skew(w) * IO - wp.skew(IOw))
            I10 = I10 + h * (wp.outer(w, mc) + eye3(wp.dot(w, mc)) - F(2.0) * wp.outer(mc, w))
        if gimp != 0:
            Kg = grav_psd(grav, mc)
            I00 = I00 + (h * h) * Kg
            f0 = f0 - h * (Kg * w)
        # ---- external point forces (at the distal node)
        if use_pf != 0:
            fp = body_force[g]
            f0 = f0 + wp.cross(tipa[g] - o, fp)
            f1 = f1 + fp
        # ---- kinematic spheres: implicit penalty spring + damper per contact point
        if n_sph > 0:
            c0n = cp_start[g]
            cn = cp_count[g]
            for si in range(n_sph):
                sidx = e * n_sph + si
                ctr = sph_c[sidx]
                vs = sph_v[sidx]
                prm = sph_p[sidx]
                rad = prm[0]
                ks = prm[1]
                cs = prm[2]
                cc = h * h * ks + h * cs
                for k in range(cn):
                    rw = R * cp_local[c0n + k]
                    dvec = o + rw - ctr
                    dist = wp.length(dvec)
                    if dist < rad:
                        nrm = dvec / wp.max(dist, F(1e-12))
                        pen = rad - dist
                        ga = wp.cross(rw, nrm)
                        vn = wp.dot(ga, w) + wp.dot(nrm, vO) - wp.dot(vs, nrm)
                        fm = ks * pen - (h * ks + cs) * vn
                        f0 = f0 + fm * ga
                        f1 = f1 + fm * nrm
                        I00 = I00 + cc * wp.outer(ga, ga)
                        I01 = I01 + cc * wp.outer(ga, nrm)
                        I10 = I10 + cc * wp.outer(nrm, ga)
                        I11 = I11 + cc * wp.outer(nrm, nrm)

        # ---- joint spring (implicit): tau = -2 Gamma K v - (h + beta) K~ u, armature
        Q = Qa[g]
        if Q[0] < F(0.0):
            Q = -Q
        qv = V3(Q[1], Q[2], Q[3])
        Gam = eye3(Q[0]) - wp.skew(qv)
        GK = Gam * wp.diag(K_a[g])
        Kt = GK * wp.transpose(Gam)
        tau = F(-2.0) * (GK * qv) - (h + beta) * (Kt * u)
        Aarm = (h * h + h * beta) * Kt

        # ---- linearised velocity-product propagation terms
        wpar = V3(F(0.0), F(0.0), F(0.0))
        r = V3(F(0.0), F(0.0), F(0.0))
        if p >= 0:
            wpar = wa[p]
            r = o - oa[p]
        wrel = R * u
        Smt = R
        Xa = I3
        Xc = -wp.skew(r)
        if gyro != 0:
            Smt = Smt + h * (wp.skew(wpar) * R)
            Xa = Xa - h * wp.skew(wrel)
            Xc = Xc + h * (wp.outer(wpar, r) + eye3(wp.dot(wpar, r)) - F(2.0) * wp.outer(r, wpar))
        c0 = wp.cross(wpar, wrel)
        c1 = wp.cross(wpar, wp.cross(wpar, r))

        # ---- pull the children's articulated contributions (segmented sum, no atomics)
        A00 = I00
        A01 = I01
        A10 = I10
        A11 = I11
        p0 = b0 - f0
        p1 = b1 - f1
        nch = ch_count[g]
        if nch > 0:
            cs0 = ch_start[g]
            c = ch_list[cs0]
            S00 = upI00[c]
            S01 = upI01[c]
            S10 = upI10[c]
            S11 = upI11[c]
            Sp0 = upP0[c]
            Sp1 = upP1[c]
            for k in range(1, nch):
                c = ch_list[cs0 + k]
                S00 = S00 + upI00[c]
                S01 = S01 + upI01[c]
                S10 = S10 + upI10[c]
                S11 = S11 + upI11[c]
                Sp0 = Sp0 + upP0[c]
                Sp1 = Sp1 + upP1[c]
            A00 = A00 + S00
            A01 = A01 + S01
            A10 = A10 + S10
            A11 = A11 + S11
            p0 = p0 + Sp0
            p1 = p1 + Sp1

        # ---- articulated-body projection through the joint
        U0 = A00 * Smt
        U1 = A10 * Smt
        W0 = RT * A00
        W1 = RT * A01
        D = W0 * Smt + Aarm
        Di = wp.inverse(D)
        ul = tau - RT * p0
        UD0 = U0 * Di
        UD1 = U1 * Di
        Xa_a[g] = Xa
        Xc_a[g] = Xc
        Smt_a[g] = Smt
        W0_a[g] = W0
        W1_a[g] = W1
        Dinv_a[g] = Di
        UD0_a[g] = UD0
        UD1_a[g] = UD1
        uu_a[g] = ul
        c0_a[g] = c0
        c1_a[g] = c1
        r_a[g] = r
        if p >= 0:
            J00 = A00 - UD0 * W0
            J01 = A01 - UD0 * W1
            J10 = A10 - UD1 * W0
            J11 = A11 - UD1 * W1
            pa0 = p0 + J00 * c0 + J01 * c1 + UD0 * ul
            pa1 = p1 + J10 * c0 + J11 * c1 + UD1 * ul
            B00 = J00 * Xa + J01 * Xc
            B10 = J10 * Xa + J11 * Xc
            Sr = wp.skew(r)
            upI00[g] = B00 + Sr * B10
            upI01[g] = J01 + Sr * J11
            upI10[g] = B10
            upI11[g] = J11
            upP0[g] = pa0 + wp.cross(r, pa1)
            upP1[g] = pa1

    # ------------------------------------------------------------------ root -> leaf sweep
    @wp.kernel(module="unique")
    def aba_down(L0: int, done: wp.array(dtype=int), parent: wp.array(dtype=int),
                 Qa: wp.array(dtype=V4), ua: wp.array(dtype=V3),
                 R0rel: wp.array(dtype=M33), o0: wp.array(dtype=V3), seg_l: wp.array(dtype=F),
                 Ra: wp.array(dtype=M33), oa: wp.array(dtype=V3), tipa: wp.array(dtype=V3),
                 wa: wp.array(dtype=V3), vOa: wp.array(dtype=V3),
                 Xa_a: wp.array(dtype=M33), Xc_a: wp.array(dtype=M33), Smt_a: wp.array(dtype=M33),
                 W0_a: wp.array(dtype=M33), W1_a: wp.array(dtype=M33), Dinv_a: wp.array(dtype=M33),
                 uu_a: wp.array(dtype=V3), c0_a: wp.array(dtype=V3), c1_a: wp.array(dtype=V3),
                 acc0: wp.array(dtype=V3), acc1: wp.array(dtype=V3), udot: wp.array(dtype=V3),
                 h: F, zero_u: int):
        g = L0 + wp.tid()
        if done[0] != 0:
            return
        p = parent[g]
        ap0 = V3(F(0.0), F(0.0), F(0.0))
        ap1 = V3(F(0.0), F(0.0), F(0.0))
        if p >= 0:
            ap0 = acc0[p]
            ap1 = acc1[p]
        a0 = Xa_a[g] * ap0 + c0_a[g]
        a1 = Xc_a[g] * ap0 + ap1 + c1_a[g]
        ud = Dinv_a[g] * (uu_a[g] - W0_a[g] * a0 - W1_a[g] * a1)
        acc0[g] = a0 + Smt_a[g] * ud
        acc1[g] = a1
        udot[g] = ud
        un = ua[g] + h * ud
        Qn = qmul(Qa[g], quat_from_rotvec(h * un))
        Qn = Qn / wp.length(Qn)
        if zero_u != 0:
            un = V3(F(0.0), F(0.0), F(0.0))
        Qa[g] = Qn
        ua[g] = un
        # forward kinematics of the new state (parents were updated by the previous launch)
        fk_body(g, p, Qn, un, R0rel, o0, seg_l, Ra, oa, tipa, wa, vOa)

    # ------------------------------------------------------------------ factor reuse (contact slot)
    @wp.kernel(module="unique")
    def resp_up(L0: int, parent: wp.array(dtype=int),
                ch_start: wp.array(dtype=int), ch_count: wp.array(dtype=int), ch_list: wp.array(dtype=int),
                Ra: wp.array(dtype=M33), r_a: wp.array(dtype=V3),
                UD0_a: wp.array(dtype=M33), UD1_a: wp.array(dtype=M33),
                df0: wp.array(dtype=V3), df1: wp.array(dtype=V3),
                dP0: wp.array(dtype=V3), dP1: wp.array(dtype=V3), du: wp.array(dtype=V3)):
        """Leaf-to-root bias propagation of extra body forces df (spatial, about o, world) on
        the *stored* factorisation: dpA = -df + sum_children X^T dpa, du = -R^T dpA0,
        dpa = dpA + UD du."""
        g = L0 + wp.tid()
        p0 = -df0[g]
        p1 = -df1[g]
        cs0 = ch_start[g]
        for k in range(ch_count[g]):
            c = ch_list[cs0 + k]
            p0 = p0 + dP0[c]
            p1 = p1 + dP1[c]
        d = -(wp.transpose(Ra[g]) * p0)
        du[g] = d
        p = parent[g]
        if p >= 0:
            pa0 = p0 + UD0_a[g] * d
            pa1 = p1 + UD1_a[g] * d
            r = r_a[g]
            dP0[g] = pa0 + wp.cross(r, pa1)
            dP1[g] = pa1

    @wp.kernel(module="unique")
    def resp_down(L0: int, parent: wp.array(dtype=int),
                  Xa_a: wp.array(dtype=M33), Xc_a: wp.array(dtype=M33), Smt_a: wp.array(dtype=M33),
                  W0_a: wp.array(dtype=M33), W1_a: wp.array(dtype=M33), Dinv_a: wp.array(dtype=M33),
                  du: wp.array(dtype=V3), da0: wp.array(dtype=V3), da1: wp.array(dtype=V3),
                  dud: wp.array(dtype=V3), with_bias: int, c0_a: wp.array(dtype=V3), c1_a: wp.array(dtype=V3)):
        """Root-to-leaf solve on the stored factor. with_bias=1 (du = uu) gives the free-motion
        accelerations of the step without integrating; with_bias=0 the linear response."""
        g = L0 + wp.tid()
        p = parent[g]
        ap0 = V3(F(0.0), F(0.0), F(0.0))
        ap1 = V3(F(0.0), F(0.0), F(0.0))
        if p >= 0:
            ap0 = da0[p]
            ap1 = da1[p]
        a0 = Xa_a[g] * ap0
        a1 = Xc_a[g] * ap0 + ap1
        if with_bias != 0:
            a0 = a0 + c0_a[g]
            a1 = a1 + c1_a[g]
        ud = Dinv_a[g] * (du[g] - W0_a[g] * a0 - W1_a[g] * a1)
        da0[g] = a0 + Smt_a[g] * ud
        da1[g] = a1
        dud[g] = ud

    @wp.kernel(module="unique")
    def add_into(dst: wp.array(dtype=V3), src: wp.array(dtype=V3)):
        g = wp.tid()
        dst[g] = dst[g] + src[g]

    # ------------------------------------------------------------------ diagnostics
    @wp.kernel(module="unique")
    def speeds(env: wp.array(dtype=int), Ra: wp.array(dtype=M33), oa: wp.array(dtype=V3),
               tipa: wp.array(dtype=V3), wa: wp.array(dtype=V3), vOa: wp.array(dtype=V3),
               cp_start: wp.array(dtype=int), cp_count: wp.array(dtype=int), cp_local: wp.array(dtype=V3),
               vn_max: wp.array(dtype=F), vp_max: wp.array(dtype=F)):
        g = wp.tid()
        e = env[g]
        R = Ra[g]
        w = wa[g]
        vO = vOa[g]
        vn = wp.length(vO + wp.cross(w, tipa[g] - oa[g]))
        if not (vn < F(BIG)):
            vn = F(BIG)                   # NaN / inf never counts as quiet
        wp.atomic_max(vn_max, e, vn)
        vp = vn
        c0 = cp_start[g]
        for k in range(cp_count[g]):
            vk = wp.length(vO + wp.cross(w, R * cp_local[c0 + k]))
            if not (vk < F(BIG)):
                vk = F(BIG)
            vp = wp.max(vp, vk)
        wp.atomic_max(vp_max, e, vp)

    @wp.kernel(module="unique")
    def settle_update(done: wp.array(dtype=int), istep: wp.array(dtype=int),
                      vn: wp.array(dtype=F), vp: wp.array(dtype=F), include_leaves: int, tol: F,
                      nwin: int, quiet: wp.array(dtype=int), conv_step: wp.array(dtype=int),
                      hist: wp.array(dtype=F), n_open: wp.array(dtype=int)):
        e = wp.tid()
        if done[0] != 0:
            return
        sp = vn[e]
        if include_leaves != 0:
            sp = wp.max(sp, vp[e])
        i = istep[0]
        nb = vn.shape[0]
        hist[i * nb + e] = sp
        q = int(0)
        if sp < tol:
            q = quiet[e] + 1
        quiet[e] = q
        if q >= nwin and conv_step[e] < 0:
            conv_step[e] = i
        if conv_step[e] < 0:
            wp.atomic_add(n_open, 0, 1)

    @wp.kernel(module="unique")
    def increment(done: wp.array(dtype=int), env: wp.array(dtype=int), tipa: wp.array(dtype=V3),
                  prev: wp.array(dtype=V3), dx: wp.array(dtype=F)):
        g = wp.tid()
        if done[0] != 0:
            return
        d = tipa[g] - prev[g]
        m = wp.max(wp.abs(d[0]), wp.max(wp.abs(d[1]), wp.abs(d[2])))
        if not (m < F(BIG)):
            m = F(BIG)
        wp.atomic_max(dx, env[g], m)
        prev[g] = tipa[g]

    @wp.kernel(module="unique")
    def static_update(done: wp.array(dtype=int), istep: wp.array(dtype=int), dx: wp.array(dtype=F),
                      tol: F, hist: wp.array(dtype=F), n_open: wp.array(dtype=int)):
        e = wp.tid()
        if done[0] != 0:
            return
        nb = dx.shape[0]
        hist[istep[0] * nb + e] = dx[e]
        if not (dx[e] < tol):
            wp.atomic_add(n_open, 0, 1)

    @wp.kernel(module="unique")
    def finalize(done: wp.array(dtype=int), istep: wp.array(dtype=int), n_open: wp.array(dtype=int),
                 nmax: int):
        if done[0] != 0:
            return
        istep[0] = istep[0] + 1
        if n_open[0] == 0 or istep[0] >= nmax:
            done[0] = 1
        n_open[0] = 0

    # ================================================================== B2: inverse statics
    @wp.kernel(module="unique")
    def inv_fk_level(L0: int, parent: wp.array(dtype=int), Rt: wp.array(dtype=M33), o0: wp.array(dtype=V3),
                     seg_l: wp.array(dtype=F), Ra: wp.array(dtype=M33), oa: wp.array(dtype=V3),
                     tipa: wp.array(dtype=V3)):
        g = L0 + wp.tid()
        p = parent[g]
        R = Rt[g]
        o = o0[g]
        if p >= 0:
            R = Ra[p] * Rt[g]
            o = tipa[p]
        Ra[g] = R
        oa[g] = o
        tipa[g] = o + seg_l[g] * col2(R)

    @wp.func
    def l1(v: V3) -> F:
        return wp.abs(v[0]) + wp.abs(v[1]) + wp.abs(v[2])

    @wp.func
    def wfromv(v: V3) -> F:
        return wp.sqrt(wp.max(F(1.0) - wp.dot(v, v), F(0.0)))

    @wp.func
    def jresid(v: V3, m: V3, K: V3) -> V3:
        Kv = wp.cw_mul(K, v)
        return F(2.0) * (wfromv(v) * Kv - wp.cross(v, Kv)) - m

    @wp.func
    def elastic_H(K: V3, Q: V4) -> M33:
        """H_el = Gamma K Gamma^T + [K v]x Gamma^T - (K v) v^T (hessian.elastic_terms)."""
        q = Q
        if q[0] < F(0.0):
            q = -q
        v = V3(q[1], q[2], q[3])
        Gam = eye3(q[0]) - wp.skew(v)
        Kv = wp.cw_mul(K, v)
        GT = wp.transpose(Gam)
        return Gam * wp.diag(K) * GT + wp.skew(Kv) * GT - wp.outer(Kv, v)

    @wp.func
    def joint_newton(m: V3, K: V3, v0: V3) -> V4:
        """Damped Newton for 2 (w K v - v x K v) = m (inverse._joint_newton).
        Returns (conv, v0, v1, v2)."""
        v = v0
        scale = l1(m) + F(TINY)
        Kd = wp.diag(K)
        it = int(0)
        while it < 40:
            r = jresid(v, m, K)
            rn = l1(r)
            if rn <= F(TOLN) * scale:
                break
            w = wfromv(v)
            Kv = wp.cw_mul(K, v)
            J = F(2.0) * (w * Kd - wp.outer(Kv, v) / wp.max(w, F(1e-12)) - wp.skew(v) * Kd + wp.skew(Kv))
            dv = -(wp.inverse(J) * r)
            a = F(1.0)
            good = int(0)
            vn = v
            ls = int(0)
            while ls < 40:
                vt = v + a * dv
                if wp.dot(vt, vt) < F(1.0) - F(INS):
                    rr = jresid(vt, m, K)
                    if l1(rr) < rn * (F(1.0) - F(1e-4) * a) + F(TINY):
                        vn = vt
                        good = 1
                        break
                a = a * F(0.5)
                ls = ls + 1
            v = vn
            if good == 0:
                break
            it = it + 1
        conv = F(0.0)
        if l1(jresid(v, m, K)) <= F(CTOL) * scale + F(TINY):
            conv = F(1.0)
        return V4(conv, v[0], v[1], v[2])

    @wp.func
    def branch_ok(res: V4, K: V3) -> int:
        v = V3(res[1], res[2], res[3])
        w = wfromv(v)
        ok = int(0)
        if res[0] > F(0.5) and w > F(1e-6):
            if wp.determinant(elastic_H(K, V4(w, v[0], v[1], v[2]))) > F(0.0):
                ok = 1
        return ok

    @wp.func
    def solve_joint(m: V3, K: V3) -> V4:
        """Unit quaternion (w, v) on the branch through Q = 1 with 2 Gamma K v = m; w is set
        negative (-1) as a flag when infeasible (inverse.solve_joint_deviation)."""
        v0 = F(0.5) * wp.cw_div(m, K)
        nv = wp.length(v0)
        if nv > F(0.5):
            v0 = v0 * (F(0.5) / wp.max(nv, F(TINY)))
        res = joint_newton(m, K, v0)
        ok = branch_ok(res, K)
        v = V3(res[1], res[2], res[3])
        if ok == 0:
            vb = V3(F(0.0), F(0.0), F(0.0))
            okb = int(1)
            kk = int(0)
            while kk < 16:
                t = F(kk + 1) / F(16.0)
                rb = joint_newton(t * m, K, vb)
                vb = V3(rb[1], rb[2], rb[3])
                okb = okb * branch_ok(rb, K)
                kk = kk + 1
            v = vb
            ok = okb
        w = wfromv(v)
        if ok == 0:
            w = -w - F(1.0)          # flag (decoded on store)
        return V4(w, v[0], v[1], v[2])

    @wp.func
    def pivot(Pl: M33, eigv: wp.array(dtype=V3), g: int) -> M33:
        """Symmetric pivot -> inverse (Sturm convention for singular), eigenvalues stored."""
        P = F(0.5) * (Pl + wp.transpose(Pl))
        Vm = eye3(F(0.0))
        lam = V3(F(0.0), F(0.0), F(0.0))
        Vm, lam = wp.eig3(P)
        amax = wp.max(wp.abs(lam[0]), wp.max(wp.abs(lam[1]), wp.abs(lam[2])))
        tiny = F(1e-14) * amax + F(TINY)
        l0 = lam[0]
        l1_ = lam[1]
        l2 = lam[2]
        if wp.abs(l0) < tiny:
            l0 = tiny
        if wp.abs(l1_) < tiny:
            l1_ = tiny
        if wp.abs(l2) < tiny:
            l2 = tiny
        eigv[g] = V3(l0, l1_, l2)
        return Vm * wp.diag(V3(F(1.0) / l0, F(1.0) / l1_, F(1.0) / l2)) * wp.transpose(Vm)

    @wp.kernel(module="unique")
    def inv_up_level(L0: int, parent: wp.array(dtype=int), env: wp.array(dtype=int),
                     ch_start: wp.array(dtype=int), ch_count: wp.array(dtype=int), ch_list: wp.array(dtype=int),
                     Ra: wp.array(dtype=M33), oa: wp.array(dtype=V3), tipa: wp.array(dtype=V3),
                     m_a: wp.array(dtype=F), h1_a: wp.array(dtype=V3), K_a: wp.array(dtype=V3),
                     use_pf: int, body_force: wp.array(dtype=V3), grav: V3, Rt: wp.array(dtype=M33),
                     do_stab: int,
                     Fs: wp.array(dtype=V3), Ws: wp.array(dtype=M33),
                     Qout: wp.array(dtype=V4), okout: wp.array(dtype=int), torque: wp.array(dtype=V3),
                     R0out: wp.array(dtype=M33), Dw_a: wp.array(dtype=M33), A_a: wp.array(dtype=M33),
                     Kw_a: wp.array(dtype=M33), Phi_a: wp.array(dtype=M33), eigv: wp.array(dtype=V3),
                     jneg: wp.array(dtype=int), nneg: wp.array(dtype=int), nbad: wp.array(dtype=int)):
        g = L0 + wp.tid()
        e = env[g]
        R = Ra[g]
        RT = wp.transpose(R)
        o = oa[g]
        mc = R * h1_a[g]
        Fb = m_a[g] * grav
        Lb = wp.outer(mc, grav)
        if use_pf != 0:
            fp = body_force[g]
            Fb = Fb + fp
            Lb = Lb + wp.outer(tipa[g] - o, fp)
        Wb = Lb + wp.outer(o, Fb)
        nch = ch_count[g]
        cs0 = ch_start[g]
        Phib = eye3(F(0.0))
        if nch > 0:
            c = ch_list[cs0]
            SF = Fs[c]
            SW = Ws[c]
            SP = Phi_a[c]
            for k in range(1, nch):
                c = ch_list[cs0 + k]
                SF = SF + Fs[c]
                SW = SW + Ws[c]
                SP = SP + Phi_a[c]
            Fb = Fb + SF
            Wb = Wb + SW
            Phib = SP
        Fs[g] = Fb
        Ws[g] = Wb
        L = Wb - wp.outer(o, Fb)
        N = V3(L[1, 2] - L[2, 1], L[2, 0] - L[0, 2], L[0, 1] - L[1, 0])
        A = L - eye3(L[0, 0] + L[1, 1] + L[2, 2])
        mj = RT * N
        K = K_a[g]
        res = solve_joint(mj, K)
        ok = int(1)
        w = res[0]
        if w < F(0.0):
            ok = 0
            w = -w - F(1.0)
            wp.atomic_add(nbad, e, 1)
        Q = V4(w, res[1], res[2], res[3])
        Qout[g] = Q
        okout[g] = ok
        torque[g] = mj
        R0out[g] = Rt[g] * wp.transpose(quat_to_matrix(Q))
        Dw = R * elastic_H(K, Q) * RT - wp.transpose(A)
        Dw_a[g] = Dw
        A_a[g] = A
        Kw_a[g] = R * wp.diag(K) * RT
        if do_stab != 0:
            Pinv = pivot(Dw - Phib, eigv, g)
            lam = eigv[g]
            cnt = int(0)
            if lam[0] < F(0.0):
                cnt = cnt + 1
            if lam[1] < F(0.0):
                cnt = cnt + 1
            if lam[2] < F(0.0):
                cnt = cnt + 1
            jneg[g] = cnt
            if cnt > 0:
                wp.atomic_add(nneg, e, cnt)
            Phi_a[g] = Phib + (A + Phib) * Pinv * (wp.transpose(A) + Phib)

    @wp.kernel(module="unique")
    def elim_level(L0: int, env: wp.array(dtype=int),
                   ch_start: wp.array(dtype=int), ch_count: wp.array(dtype=int), ch_list: wp.array(dtype=int),
                   Dw_a: wp.array(dtype=M33), A_a: wp.array(dtype=M33), Kw_a: wp.array(dtype=M33),
                   shift: wp.array(dtype=F), use_K: int, Phi_a: wp.array(dtype=M33), eigv: wp.array(dtype=V3),
                   count: wp.array(dtype=int)):
        """Shifted symmetric elimination of H - sigma M (M = I or K_el): Sturm count per env."""
        g = L0 + wp.tid()
        e = env[g]
        sig = shift[e]
        Mt = eye3(F(1.0))
        if use_K != 0:
            Mt = Kw_a[g]
        D = Dw_a[g] - sig * Mt
        D = F(0.5) * (D + wp.transpose(D))
        Phib = eye3(F(0.0))
        cs0 = ch_start[g]
        nch = ch_count[g]
        if nch > 0:
            SP = Phi_a[ch_list[cs0]]
            for k in range(1, nch):
                SP = SP + Phi_a[ch_list[cs0 + k]]
            Phib = SP
        Pinv = pivot(D - Phib, eigv, g)
        lam = eigv[g]
        cnt = int(0)
        if lam[0] < F(0.0):
            cnt = cnt + 1
        if lam[1] < F(0.0):
            cnt = cnt + 1
        if lam[2] < F(0.0):
            cnt = cnt + 1
        if cnt > 0:
            wp.atomic_add(count, e, cnt)
        A = A_a[g]
        Phi_a[g] = Phib + (A + Phib) * Pinv * (wp.transpose(A) + Phib)

    @wp.kernel(module="unique")
    def gersh(env: wp.array(dtype=int), Dw_a: wp.array(dtype=M33), A_a: wp.array(dtype=M33),
              sD: wp.array(dtype=F), sA: wp.array(dtype=F)):
        g = wp.tid()
        e = env[g]
        D = Dw_a[g]
        A = A_a[g]
        a = F(0.0)
        b = F(0.0)
        for i in range(3):
            for j in range(3):
                a = a + wp.abs(D[i, j])
                b = b + wp.abs(A[i, j])
        wp.atomic_max(sD, e, a)
        wp.atomic_max(sA, e, b)

    @wp.kernel(module="unique")
    def bis_mid(lo: wp.array(dtype=F), hi: wp.array(dtype=F), shift: wp.array(dtype=F),
                count: wp.array(dtype=int)):
        e = wp.tid()
        shift[e] = F(0.5) * (lo[e] + hi[e])
        count[e] = 0

    @wp.kernel(module="unique")
    def bis_update(lo: wp.array(dtype=F), hi: wp.array(dtype=F), shift: wp.array(dtype=F),
                   count: wp.array(dtype=int)):
        e = wp.tid()
        if count[e] == 0:
            lo[e] = shift[e]
        else:
            hi[e] = shift[e]

    return SimpleNamespace(**{k: v for k, v in locals().items()
                              if type(v).__name__ == "Kernel" or k in ("F", "V3", "V4", "M33")},
                           precision=precision, np_dtype=("float64" if f64 else "float32"))
