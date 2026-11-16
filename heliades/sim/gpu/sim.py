"""TreeSimGPU: the B1 tree solver on the GPU (NVIDIA Warp), batched over B environments.

Same model, API and semantics as heliades.sim.TreeSim (docs/sim/DESIGN.md), see
docs/sim/GPU.md for the port. Per step:

    [scatter external point forces]                       1 launch (if any)
    leaf -> root   aba_up  per level  (bulk terms fused)  nlev launches
    [contact slot: B3's k x k contact solve on the factor] (hook, see contact.py)
    root -> leaf   aba_down per level (update + FK fused) nlev launches

All state lives on the device; a step has no host synchronisation and is replayed as one
CUDA graph (wp.ScopedCapture) per (h, damping source, external layout) key.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import warp as wp

from planttwin.asset import PhysicsParams, PlantAsset

from ..rotation import quat_from_matrix
from ..topology import TreeTopology
from ..tree import GRAVITY, PointForce, SimState, SphereCollider
from .kernels import get_kernels
from .layout import Layout


def _plist(params):
    return [params] if isinstance(params, PhysicsParams) else list(params)


class TreeSimGPU:
    """Batched implicit reduced-coordinate rod-tree simulator on the GPU.

    TreeSimGPU(asset, params_list, precision="float32" | "float64", ...) mirrors
    TreeSim(asset, params_list, ...). `asset` may also be a list of B assets (one per
    environment, different topologies): the heterogeneous multi-tree batch (docs/sim/GPU.md
    section 5); then use `env_state(e)` instead of `state`.
    """

    def __init__(self, asset: PlantAsset | Sequence[PlantAsset],
                 params_list: PhysicsParams | Sequence[PhysicsParams] | None = None, *,
                 gravity=GRAVITY, joint_damping: float = 0.0, include_leaves: bool = True,
                 implicit_gyro: bool = True, implicit_gravity: bool = True,
                 precision: str = "float32", device: str | None = None, use_graph: bool = True,
                 topology: TreeTopology | Sequence[TreeTopology] | None = None):
        wp.init()
        self.device = wp.get_device(device or "cuda:0")
        if not self.device.is_cuda:
            raise RuntimeError("TreeSimGPU needs a CUDA device")
        self.k = get_kernels(precision)
        self.precision = precision
        self.dt = np.float64 if precision == "float64" else np.float32
        hetero = isinstance(asset, (list, tuple))
        if hetero:
            assets = list(asset)
            plist = _plist(params_list) if params_list is not None else [a.physics for a in assets]
            if len(plist) != len(assets):
                raise ValueError("heterogeneous batch: one PhysicsParams per asset")
            if topology is None:
                cache = {}
                topos = []
                for a in assets:
                    if id(a) not in cache:
                        cache[id(a)] = TreeTopology(a, include_leaves=include_leaves)
                    topos.append(cache[id(a)])
            else:
                topos = list(topology)
            self.asset, self.assets = assets[0], assets
        else:
            plist = _plist(params_list if params_list is not None else asset.physics)
            T = topology if topology is not None else TreeTopology(asset, include_leaves=include_leaves)
            topos = [T] * len(plist)
            self.asset, self.assets = asset, [asset] * len(plist)
        self.hetero = hetero
        self.lay = Layout(topos)
        self.topo = topos[0]
        self.topos = topos
        self.B = len(plist)
        self.gravity = np.asarray(gravity, float)
        self.joint_damping = float(joint_damping)
        self.implicit_gyro = bool(implicit_gyro)
        self.implicit_gravity = bool(implicit_gravity)
        self.use_graph = bool(use_graph)
        self.contact = None          # B3 contact solver hook (contact.py ContactSolver)
        self._graphs = {}
        self._alloc()
        self.set_params(plist)
        self.set_rest(None)
        self.reset()

    # ------------------------------------------------------------------ device buffers
    def _arr(self, data=None, shape=None, dtype=None):
        dtype = dtype or self.k.F
        if data is not None:
            if dtype in (int, wp.int32):
                return wp.array(np.asarray(data, np.int32), dtype=wp.int32, device=self.device)
            return wp.array(np.asarray(data, self.dt), dtype=dtype, device=self.device)
        return wp.zeros(shape, dtype=dtype, device=self.device)

    def _alloc(self):
        k, L, n = self.k, self.lay, self.lay.n
        V3, V4, M33, F = k.V3, k.V4, k.M33, k.F
        I = wp.int32
        self.d_parent = self._arr(L.parent, dtype=I)
        self.d_env = self._arr(L.env, dtype=I)
        self.d_ch_start = self._arr(L.ch_start, dtype=I)
        self.d_ch_count = self._arr(L.ch_count, dtype=I)
        self.d_ch_list = self._arr(L.ch_list if len(L.ch_list) else np.zeros(1), dtype=I)
        self.d_seg_l = self._arr(L.seg_l, dtype=F)
        self.d_o0 = self._arr(L.o0, dtype=V3)
        self.d_cp_start = self._arr(L.cp_start, dtype=I)
        self.d_cp_count = self._arr(L.cp_count, dtype=I)
        self.d_cp_local = self._arr(L.cp_local if len(L.cp_local) else np.zeros((1, 3)), dtype=V3)
        self.d_gnode = self._arr(L.gnode, dtype=I)
        z = lambda dt: wp.zeros(n, dtype=dt, device=self.device)
        # state + kinematics
        self.d_Q, self.d_u = z(V4), z(V3)
        self.d_R0rel = z(M33)
        self.d_R, self.d_o, self.d_tip, self.d_w, self.d_vO = z(M33), z(V3), z(V3), z(V3), z(V3)
        # parameters
        self.d_m, self.d_h1, self.d_Io, self.d_K = z(F), z(V3), z(M33), z(V3)
        self.d_alpha = wp.zeros(self.B, dtype=F, device=self.device)
        self.d_alpha_override = wp.zeros(self.B, dtype=F, device=self.device)
        self.d_alpha_zero = wp.zeros(self.B, dtype=F, device=self.device)
        self._alpha_override_val = None
        # factorisation (up sweep) + down sweep
        self.d_upI = [z(M33) for _ in range(4)]
        self.d_upP = [z(V3), z(V3)]
        self.d_Xa, self.d_Xc, self.d_Smt = z(M33), z(M33), z(M33)
        self.d_W0, self.d_W1, self.d_Dinv = z(M33), z(M33), z(M33)
        self.d_UD0, self.d_UD1 = z(M33), z(M33)
        self.d_uu, self.d_c0, self.d_c1, self.d_r = z(V3), z(V3), z(V3), z(V3)
        self.d_acc0, self.d_acc1, self.d_udot = z(V3), z(V3), z(V3)
        # externals
        self.d_body_force = z(V3)
        self._pf_cache = None
        self.d_pf_idx = wp.zeros(1, dtype=I, device=self.device)
        self.d_pf_f = wp.zeros(1, dtype=V3, device=self.device)
        self._sph_cache = None
        self._nsph = 0
        self.d_sph_c = wp.zeros(1, dtype=V3, device=self.device)
        self.d_sph_v = wp.zeros(1, dtype=V3, device=self.device)
        self.d_sph_p = wp.zeros(1, dtype=V3, device=self.device)
        # flags / diagnostics
        self.d_done0 = wp.zeros(1, dtype=I, device=self.device)
        self.d_vn = wp.zeros(self.B, dtype=F, device=self.device)
        self.d_vp = wp.zeros(self.B, dtype=F, device=self.device)

    def memory_bytes(self) -> int:
        """Device bytes held by this simulator (all buffers)."""
        tot = 0
        for v in self.__dict__.values():
            arrs = v if isinstance(v, list) else [v]
            for a in arrs:
                if isinstance(a, wp.array):
                    tot += a.capacity
        return tot

    # ------------------------------------------------------------------ params / rest / reset
    def set_params(self, params):
        plist = _plist(params)
        if len(plist) != self.B:
            raise ValueError(f"set_params: need {self.B} parameter sets (layout is fixed)")
        self.params = plist
        L = self.lay
        col = lambda key: np.array([getattr(p, key) for p in plist], float)
        E, nu = col("stem_youngs_modulus"), col("stem_poisson")
        rho, rhol = col("stem_density"), col("leaf_areal_density")
        G = E / (2 * (1 + nu))
        K, m, h1, Io = [], [], [], []
        for e, T in enumerate(self.topos):
            kb = E[e] * T.kgeom_bend
            kt = G[e] * T.kgeom_twist
            K.append(np.stack([kb, kb, kt], -1))
            sp, lp, pp = T.seg_part, T.leaf_part, T.point_part
            m.append(rho[e] * sp.m + rhol[e] * lp.m + pp.m)
            h1.append(rho[e] * sp.h + rhol[e] * lp.h + pp.h)
            Io.append(rho[e] * sp.Io + rhol[e] * lp.Io + pp.Io)
        self._K_g = L.from_local(K)
        self._m_g = L.from_local(m)
        self._h1_g = L.from_local(h1)
        self._Io_g = L.from_local(Io)
        self.alpha = col("damping")
        self.d_K.assign(self._K_g.astype(self.dt))
        self.d_m.assign(self._m_g.astype(self.dt))
        self.d_h1.assign(self._h1_g.astype(self.dt))
        self.d_Io.assign(self._Io_g.astype(self.dt))
        self.d_alpha.assign(self.alpha.astype(self.dt))

    # host views mirroring TreeSim attributes (shared topology: (B, nb, ...))
    @property
    def K(self):
        return self.lay.to_local(self._K_g)

    @K.setter
    def K(self, val):
        val = np.asarray(val, float)
        if self.lay.shared:
            val = np.broadcast_to(val, (self.B, self.topo.nb, 3))
        self._K_g = self.lay.from_local(val if self.lay.shared else list(val))
        self.d_K.assign(self._K_g.astype(self.dt))

    @property
    def m(self):
        return self.lay.to_local(self._m_g)

    @property
    def h1(self):
        return self.lay.to_local(self._h1_g)

    @property
    def Io(self):
        return self.lay.to_local(self._Io_g)

    def set_rest(self, R0rel=None):
        """Joint rest relative rotations (B2 inverse statics): (nb, 3, 3) shared, (B, nb, 3, 3)
        per env, or None for the asset's rest shape. Keeps Q (deviation from rest)."""
        L = self.lay
        if R0rel is None:
            g = L.R0rel
        else:
            R = np.asarray(R0rel, float)
            if not L.shared:
                g = L.from_local(list(R))
            else:
                R = np.broadcast_to(R, (self.B,) + R.shape[-3:]) if R.ndim == 3 else R
                g = L.from_local(R)
        self._R0rel_g = np.array(g, float)
        self.d_R0rel.assign(self._R0rel_g.astype(self.dt))
        self._fk()

    @property
    def R0rel(self):
        return self.lay.to_local(self._R0rel_g)

    def reset(self, Q=None, u=None):
        L = self.lay
        if Q is None:
            Qg = np.zeros((L.n, 4))
            Qg[:, 0] = 1.0
        else:
            Qg = self._to_global(Q, 4)
        ug = np.zeros((L.n, 3)) if u is None else self._to_global(u, 3)
        self.d_Q.assign(Qg.astype(self.dt))
        self.d_u.assign(ug.astype(self.dt))
        self.t = 0.0
        self._fk()

    def _to_global(self, X, last):
        X = np.asarray(X, float)
        if self.lay.shared:
            return self.lay.from_local(np.broadcast_to(X, (self.B, self.topo.nb, last)))
        return self.lay.from_local(list(X))

    @property
    def Q(self):
        return self.lay.to_local(self.d_Q.numpy().astype(float))

    @Q.setter
    def Q(self, val):
        self.d_Q.assign(self._to_global(val, 4).astype(self.dt))
        self._fk()

    @property
    def u(self):
        return self.lay.to_local(self.d_u.numpy().astype(float))

    @u.setter
    def u(self, val):
        self.d_u.assign(self._to_global(val, 3).astype(self.dt))
        self._fk()

    def _fk(self):
        k, L = self.k, self.lay
        for d in range(L.nlev):
            wp.launch(k.fk_level, dim=int(L.L[d + 1] - L.L[d]), device=self.device,
                      inputs=[int(L.L[d]), self.d_parent, self.d_Q, self.d_u, self.d_R0rel, self.d_o0,
                              self.d_seg_l, self.d_R, self.d_o, self.d_tip, self.d_w, self.d_vO])

    # ------------------------------------------------------------------ externals
    def _set_external(self, external):
        """Returns the external part of the graph key; uploads changed data (rare)."""
        L, B = self.lay, self.B
        if external is None:
            return ("none", 0, ("sph", 0))
        if isinstance(external, wp.array):
            # device node forces, env-major (sum_e N_e,) vec3: zero-copy path for RL
            return ("device", external.ptr, external, ("sph", 0))
        pf, sph = [], []                    # pf: (env or None for all, node, force (B,3) or (3,))
        if isinstance(external, np.ndarray):
            ext = np.asarray(external, float)
            if L.shared:
                ext = np.broadcast_to(ext, (B, self.topo.N, 3))
                for nd in np.flatnonzero(np.any(ext != 0, axis=(0, 2))):
                    pf.append((None, int(nd), ext[:, nd]))
            else:
                for e in range(B):
                    for nd in np.flatnonzero(np.any(ext[e] != 0, axis=-1)):
                        pf.append((e, int(nd), ext[e][nd]))
        else:
            for a in external:
                if isinstance(a, PointForce):
                    f = np.broadcast_to(np.asarray(a.force, float), (B, 3))
                    if L.shared:
                        pf.append((None, int(a.node), f))
                    else:
                        for e in range(B):
                            pf.append((e, int(a.node), f[e]))
                elif isinstance(a, SphereCollider):
                    sph.append(a)
                else:
                    raise TypeError(f"unknown external action {type(a)}")
        # ---- point forces -> unique bodies (summed in list order, like TreeSim)
        acc = {}
        for e, node, f in pf:
            b = self.topos[0 if e is None else e].node_body[node]
            if b < 0:
                continue
            key = (e, int(b))
            acc[key] = acc[key] + f if key in acc else np.array(f, float)
        idx, frc = [], []
        for (e, b), f in acc.items():
            if e is None:                    # shared topology: g = b * B + env
                idx.append(b * B + np.arange(B))
                frc.append(f.reshape(B, 3))
            else:
                idx.append(np.array([L.g_of[e][b]]))
                frc.append(f.reshape(1, 3))
        key_pf = ("none", 0)
        if idx:
            ia = np.concatenate(idx).astype(np.int32)
            fa = np.concatenate(frc)
            c = self._pf_cache
            if c is None or len(c[0]) != len(ia) or not (np.array_equal(c[0], ia) and np.array_equal(c[1], fa)):
                if self.d_pf_idx.shape[0] < len(ia):
                    cap = max(len(ia), 2 * self.d_pf_idx.shape[0])
                    self.d_pf_idx = wp.zeros(cap, dtype=wp.int32, device=self.device)
                    self.d_pf_f = wp.zeros(cap, dtype=self.k.V3, device=self.device)
                    self._graphs.clear()
                wp.copy(self.d_pf_idx, wp.array(ia, dtype=wp.int32, device=self.device), count=len(ia))
                wp.copy(self.d_pf_f, wp.array(fa.astype(self.dt), dtype=self.k.V3, device=self.device), count=len(ia))
                self._pf_cache = (ia, fa)
            key_pf = ("sparse", len(ia))
        # ---- spheres -> (B, nS) arrays
        ns = len(sph)
        if ns:
            ctr = np.stack([np.broadcast_to(np.asarray(s.center, float), (B, 3)) for s in sph], 1)
            vel = np.stack([np.zeros((B, 3)) if s.velocity is None else
                            np.broadcast_to(np.asarray(s.velocity, float), (B, 3)) for s in sph], 1)
            prm = np.broadcast_to(np.array([[s.radius, s.stiffness, s.damping] for s in sph], float), (B, ns, 3))
            data = (ctr.reshape(-1, 3), vel.reshape(-1, 3), np.ascontiguousarray(prm).reshape(-1, 3))
            c = self._sph_cache
            if c is None or c[0].shape != data[0].shape or not all(np.array_equal(a, b) for a, b in zip(c, data)):
                if self.d_sph_c.shape[0] < B * ns:
                    self.d_sph_c = wp.zeros(B * ns, dtype=self.k.V3, device=self.device)
                    self.d_sph_v = wp.zeros(B * ns, dtype=self.k.V3, device=self.device)
                    self.d_sph_p = wp.zeros(B * ns, dtype=self.k.V3, device=self.device)
                    self._graphs.clear()
                for dst, src in zip((self.d_sph_c, self.d_sph_v, self.d_sph_p), data):
                    wp.copy(dst, wp.array(src.astype(self.dt), dtype=self.k.V3, device=self.device), count=len(src))
                self._sph_cache = data
        return key_pf + (("sph", ns),)

    def _alpha_buffer(self, damping):
        if damping is None:
            return self.d_alpha, "param"
        a = np.broadcast_to(np.asarray(damping, float), (self.B,))
        if self._alpha_override_val is None or not np.array_equal(self._alpha_override_val, a):
            self.d_alpha_override.assign(a.astype(self.dt))
            self._alpha_override_val = a.copy()
        return self.d_alpha_override, "override"

    # ------------------------------------------------------------------ launch sequence
    def _seq_up(self, h, alpha, ext_key, done):
        """External force scatter + leaf-to-root sweep (bulk terms fused): the factorisation."""
        k, L, dev = self.k, self.lay, self.device
        pf_mode = ext_key[0]
        if pf_mode == "sparse":
            self.d_body_force.zero_()
            wp.launch(k.scatter_forces, dim=ext_key[1], device=dev,
                      inputs=[self.d_pf_idx, self.d_pf_f, self.d_body_force])
        elif pf_mode == "device":
            wp.launch(k.node_forces_to_bodies, dim=L.n, device=dev,
                      inputs=[self.d_gnode, ext_key[2], self.d_body_force])
        use_pf = 0 if pf_mode == "none" else 1
        nsph = ext_key[-1][1]
        grav = k.V3(*[float(x) for x in self.gravity])
        up_tail = [grav, float(h), self.joint_damping, int(self.implicit_gyro), int(self.implicit_gravity),
                   *self.d_upI, *self.d_upP, self.d_Xa, self.d_Xc, self.d_Smt, self.d_W0, self.d_W1,
                   self.d_Dinv, self.d_UD0, self.d_UD1, self.d_uu, self.d_c0, self.d_c1, self.d_r]
        for d in range(L.nlev - 1, -1, -1):
            wp.launch(k.aba_up, dim=int(L.L[d + 1] - L.L[d]), device=dev,
                      inputs=[int(L.L[d]), done, self.d_parent, self.d_env, self.d_ch_start, self.d_ch_count,
                              self.d_ch_list, self.d_Q, self.d_u, self.d_R, self.d_o, self.d_tip, self.d_w,
                              self.d_vO, self.d_m, self.d_h1, self.d_Io, self.d_K, alpha,
                              use_pf, self.d_body_force, nsph, self.d_cp_start, self.d_cp_count, self.d_cp_local,
                              self.d_sph_c, self.d_sph_v, self.d_sph_p] + up_tail)

    def _seq_down(self, h, done, zero_u=0):
        """Root-to-leaf sweep with the state update and forward kinematics fused."""
        k, L, dev = self.k, self.lay, self.device
        for d in range(L.nlev):
            wp.launch(k.aba_down, dim=int(L.L[d + 1] - L.L[d]), device=dev,
                      inputs=[int(L.L[d]), done, self.d_parent, self.d_Q, self.d_u, self.d_R0rel, self.d_o0,
                              self.d_seg_l, self.d_R, self.d_o, self.d_tip, self.d_w, self.d_vO,
                              self.d_Xa, self.d_Xc, self.d_Smt, self.d_W0, self.d_W1, self.d_Dinv,
                              self.d_uu, self.d_c0, self.d_c1, self.d_acc0, self.d_acc1, self.d_udot,
                              float(h), int(zero_u)])

    def _seq_step(self, h, alpha, ext_key, done, zero_u=0):
        self._seq_up(h, alpha, ext_key, done)
        if self.contact is not None:
            # contact slot: factor is complete, kinematics still at q_n (see contact.py)
            self.contact.solve(self, h)
        self._seq_down(h, done, zero_u)

    def _run(self, key, seq):
        """Launch `seq` as a cached CUDA graph (captured on first use) or directly."""
        if not self.use_graph:
            seq()
            return
        gr = self._graphs.get(key)
        if gr is None:
            with wp.ScopedCapture(device=self.device) as cap:
                seq()
            gr = self._graphs[key] = cap.graph
        wp.capture_launch(gr)

    # ------------------------------------------------------------------ dynamics
    def step(self, h: float, external=None, damping=None, return_udot: bool = False):
        """Advance all B environments by one implicit step of size h (no host sync)."""
        ext_key = self._set_external(external)
        alpha, akey = self._alpha_buffer(damping)
        key = ("step", float(h), akey, ext_key[:2], ext_key[-1], id(self.contact))
        self._run(key, lambda: self._seq_step(h, alpha, ext_key, self.d_done0))
        self.t += h
        if return_udot:
            return self.lay.to_local(self.d_udot.numpy().astype(float))
        return None

    def run(self, duration: float, h: float, external=None, damping=None, record=None, every=1):
        nsteps = int(round(duration / h))
        rec = (lambda s: s.state.x) if record is None else record
        out = [rec(self)]
        for i in range(nsteps):
            ext = external(self.t) if callable(external) else external
            self.step(h, ext, damping)
            if (i + 1) % every == 0:
                out.append(rec(self))
        return np.array(out)

    def _flags(self, nmax):
        I, dev = wp.int32, self.device
        return (wp.zeros(1, dtype=I, device=dev), wp.zeros(1, dtype=I, device=dev),
                wp.zeros(1, dtype=I, device=dev), wp.zeros(nmax * self.B, dtype=self.k.F, device=dev))

    def _drive(self, key, seq, nmax, done, istep, check_every):
        launched = 0
        self._graphs.pop(key, None)
        while launched < nmax:
            kk = min(check_every, nmax - launched)
            for _ in range(kk):
                self._run(key, seq)
            launched += kk
            if int(done.numpy()[0]):
                break
        self._graphs.pop(key, None)
        return int(istep.numpy()[0])

    def settle(self, h: float = 1 / 60, max_time: float = 3.0, tol: float = 1e-3, damping=None,
               external=None, include_leaves: bool = False, window: float = 0.25, check_every: int = 8):
        """TreeSim.settle on the device: the quiet-window test runs in a kernel after every
        step; the host polls a done flag every `check_every` steps (steps after convergence
        are no-ops, so the result equals a per-step check)."""
        k, dev, B = self.k, self.device, self.B
        nmax = int(round(max_time / h))
        nwin = max(1, int(round(window / h)))
        ext_key = self._set_external(external)
        alpha, _ = self._alpha_buffer(damping)
        done, istep, n_open, hist = self._flags(nmax)
        quiet = wp.zeros(B, dtype=wp.int32, device=dev)
        conv = wp.array(np.full(B, -1, np.int32), dtype=wp.int32, device=dev)

        def seq():
            self._seq_step(h, alpha, ext_key, done)
            self.d_vn.zero_()
            self.d_vp.zero_()
            wp.launch(k.speeds, dim=self.lay.n, device=dev,
                      inputs=[self.d_env, self.d_R, self.d_o, self.d_tip, self.d_w, self.d_vO,
                              self.d_cp_start, self.d_cp_count, self.d_cp_local, self.d_vn, self.d_vp])
            wp.launch(k.settle_update, dim=B, device=dev,
                      inputs=[done, istep, self.d_vn, self.d_vp, int(include_leaves), float(tol), nwin,
                              quiet, conv, hist, n_open])
            wp.launch(k.finalize, dim=1, device=dev, inputs=[done, istep, n_open, nmax])

        steps = self._drive(("settle", id(hist)), seq, nmax, done, istep, check_every)
        ts = []
        for _ in range(steps):
            self.t += h
            ts.append(self.t)
        cs = conv.numpy()
        conv_t = np.array([ts[c] - (nwin - 1) * h if c >= 0 else np.nan for c in cs])
        H = hist.numpy().astype(float).reshape(nmax, B)[:steps]
        return {"time": self.t, "steps": steps, "converged": ~np.isnan(conv_t), "t_converged": conv_t,
                "max_speed": H[-1], "speed_history": H}

    def solve_static(self, h: float = 0.2, iters: int = 200, tol: float = 1e-6, external=None,
                     check_every: int = 8):
        """TreeSim.solve_static (pseudo-time modified Newton) on the device."""
        k, dev, B = self.k, self.device, self.B
        self.d_u.zero_()
        self._fk()
        prev = wp.clone(self.d_tip)
        ext_key = self._set_external(external)
        done, istep, n_open, hist = self._flags(iters)
        dx = wp.zeros(B, dtype=k.F, device=dev)

        def seq():
            self._seq_step(h, self.d_alpha_zero, ext_key, done, zero_u=1)
            dx.zero_()
            wp.launch(k.increment, dim=self.lay.n, device=dev, inputs=[done, self.d_env, self.d_tip, prev, dx])
            wp.launch(k.static_update, dim=B, device=dev, inputs=[done, istep, dx, float(tol), hist, n_open])
            wp.launch(k.finalize, dim=1, device=dev, inputs=[done, istep, n_open, iters])

        steps = self._drive(("static", id(hist)), seq, iters, done, istep, check_every)
        for _ in range(steps):
            self.t += h
        H = hist.numpy().astype(float).reshape(iters, B)[:steps]
        return {"iterations": steps, "converged": H[-1] < tol, "increment": H[-1], "history": H}

    # ------------------------------------------------------------------ factor reuse (contact slot)
    def factorize(self, h: float, external=None, damping=None):
        """Run only the leaf-to-root sweep at the current state (the factorisation of the
        implicit step), leaving the state untouched. Inside a step the contact slot sees the
        same factor."""
        ext_key = self._set_external(external)
        alpha, _ = self._alpha_buffer(damping)
        self._seq_up(h, alpha, ext_key, self.d_done0)

    def factor_response(self, df):
        """Change of the joint accelerations udot caused by extra spatial body forces df
        ([moment about the body origin; force], world frame) on the current factorisation
        (after `factorize`, or inside the contact slot). Two O(n) sweeps on the stored factor,
        with no refactorisation: the back-substitution primitive of a Baraff/Delassus contact
        solve (contact.py). df: (B, nb, 6) host array. Returns (B, nb, 3)."""
        k, L, dev = self.k, self.lay, self.device
        dfg = L.from_local(np.asarray(df, float)) if L.shared else L.from_local(list(df))
        z = lambda: wp.zeros(L.n, dtype=k.V3, device=dev)
        df0 = wp.array(dfg[:, :3].astype(self.dt), dtype=k.V3, device=dev)
        df1 = wp.array(dfg[:, 3:].astype(self.dt), dtype=k.V3, device=dev)
        dP0, dP1, du, da0, da1, dud = z(), z(), z(), z(), z(), z()
        self.factor_response_device(df0, df1, dP0, dP1, du, da0, da1, dud)
        return L.to_local(dud.numpy().astype(float))

    def factor_response_device(self, df0, df1, dP0, dP1, du, da0, da1, dud):
        """Device version (graph-capturable): df0/df1 in, dud (= delta udot) and da0/da1
        (= delta body accelerations, classical, about o) out; dP0/dP1/du are scratch
        (du = the induced change of the joint bias u~)."""
        self.slot_bias_change(df0, df1, dP0, dP1, du)
        self._resp_down(du, da0, da1, dud, 0)

    def slot_bias_change(self, df0, df1, dP0, dP1, du):
        """Leaf-to-root propagation of extra body forces on the stored factor: writes du, the
        change of every joint's bias u~ (no refactorisation; IA and D^-1 are force-independent)."""
        k, L, dev = self.k, self.lay, self.device
        for d in range(L.nlev - 1, -1, -1):
            wp.launch(k.resp_up, dim=int(L.L[d + 1] - L.L[d]), device=dev,
                      inputs=[int(L.L[d]), self.d_parent, self.d_ch_start, self.d_ch_count, self.d_ch_list,
                              self.d_R, self.d_r, self.d_UD0, self.d_UD1, df0, df1, dP0, dP1, du])

    def _resp_down(self, du, da0, da1, dud, with_bias):
        k, L, dev = self.k, self.lay, self.device
        for d in range(L.nlev):
            wp.launch(k.resp_down, dim=int(L.L[d + 1] - L.L[d]), device=dev,
                      inputs=[int(L.L[d]), self.d_parent, self.d_Xa, self.d_Xc, self.d_Smt, self.d_W0,
                              self.d_W1, self.d_Dinv, du, da0, da1, dud, int(with_bias), self.d_c0, self.d_c1])

    def slot_free_motion(self, acc0, acc1, udot):
        """Free-motion (contact-free) accelerations of the current step from the stored factor,
        without integrating: body accelerations acc0/acc1 ([ang; lin], about o) and udot."""
        self._resp_down(self.d_uu, acc0, acc1, udot, 1)

    def slot_apply(self, df0, df1, dP0, dP1, du):
        """Fold extra body forces (e.g. solved contact forces) into the step: u~ += delta u~.
        The step's own root-to-leaf sweep then integrates the contact-coupled motion exactly."""
        self.slot_bias_change(df0, df1, dP0, dP1, du)
        wp.launch(self.k.add_into, dim=self.lay.n, device=self.device, inputs=[self.d_uu, du])

    # ------------------------------------------------------------------ host views
    def _host_kin(self):
        L = self.lay
        f = lambda a: L.to_local(a.numpy().astype(float))
        return f(self.d_R), f(self.d_o), f(self.d_tip), f(self.d_w), f(self.d_vO)

    @property
    def state(self) -> SimState:
        if self.hetero:
            raise RuntimeError("heterogeneous batch: use env_state(e)")
        R, o, tip, w, vO = self._host_kin()
        T = self.topo
        x = np.broadcast_to(T.X0, (self.B, T.N, 3)).copy()
        x[:, T.node_b] = tip
        v = np.zeros_like(x)
        v[:, T.node_b] = vO + np.cross(w, tip - o)
        return SimState(self.t, self.Q, self.u, x, R, v)

    def env_state(self, e: int) -> SimState:
        """State of environment e (leading batch dim 1); works for heterogeneous batches."""
        L = self.lay
        T = self.topos[e]
        sel = lambda a: a.numpy().astype(float)[L.g_of[e]][None]
        R, o, tip, w, vO = (sel(a) for a in (self.d_R, self.d_o, self.d_tip, self.d_w, self.d_vO))
        x = T.X0[None].copy()
        x[:, T.node_b] = tip
        v = np.zeros_like(x)
        v[:, T.node_b] = vO + np.cross(w, tip - o)
        return SimState(self.t, sel(self.d_Q), sel(self.d_u), x, R, v)

    def leaf_vertices(self):
        R, o, _, _, _ = self._host_kin()
        out = []
        for Lf in self.topo.leaves:
            if Lf.body < 0:
                out.append(np.broadcast_to(Lf.local, (self.B,) + Lf.local.shape).copy())
            else:
                out.append(o[:, Lf.body, None] + np.einsum("bij,vj->bvi", R[:, Lf.body], Lf.local))
        return out

    def point_speeds(self):
        k, dev = self.k, self.device
        self.d_vn.zero_()
        self.d_vp.zero_()
        wp.launch(k.speeds, dim=self.lay.n, device=dev,
                  inputs=[self.d_env, self.d_R, self.d_o, self.d_tip, self.d_w, self.d_vO,
                          self.d_cp_start, self.d_cp_count, self.d_cp_local, self.d_vn, self.d_vp])
        return self.d_vn.numpy().astype(float), self.d_vp.numpy().astype(float)

    def energy(self):
        """(kinetic, elastic, gravity potential), each (B,), evaluated on the host in float64."""
        R, o, _, w, vO = self._host_kin()
        IO = R @ self.Io @ np.swapaxes(R, -1, -2)
        mc = np.einsum("bnij,bnj->bni", R, self.h1)
        m = self.m
        ke = 0.5 * (np.einsum("bni,bnij,bnj->b", w, IO, w) + 2 * np.einsum("bni,bni->b", w, np.cross(mc, vO))
                    + (m * (vO * vO).sum(-1)).sum(1))
        Q = self.Q
        el = 2.0 * (self.K * Q[..., 1:] ** 2).sum((1, 2))
        com = m[..., None] * o + mc
        pe = -(com @ self.gravity).sum(1)
        return ke, el, pe
