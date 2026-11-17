"""Acceptance tests for heliades.sim.gpu (B4: the Warp port of B1 + B2). Skipped without CUDA.

See docs/sim/GPU.md and migration/notes/B4.md. float64 builds are checked against the CPU
float64 reference to <= 1e-9 relative; float32 errors are reported (printed) and bounded.
"""
import time

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from heliades.sim import PointForce, SphereCollider, TreeSim  # noqa: E402
from heliades.sim.analytic import (cantilever_asset, elastica, first_frequency, section,  # noqa: E402
                                   tip_load_linear)
from heliades.sim.inverse import apply_rest, critical_modulus, inverse_statics  # noqa: E402


def _cuda():
    try:
        wp.config.quiet = True
        wp.init()
        return len(wp.get_cuda_devices()) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _cuda(), reason="no CUDA device")

if _cuda():
    from heliades.sim.gpu import TreeSimGPU, critical_modulus_gpu, inverse_statics_gpu
    from heliades.sim.gpu.contact import ConstantBodyForces

ZERO_G = (0.0, 0.0, 0.0)
CPU_BASELINE = 560.0          # env-steps/s, B1 notes (numpy, tomato, B >= 16)


@pytest.fixture(scope="module")
def tomato():
    pytest.importorskip("pyhelios")
    from planttwin.helios_import import build_helios_plant
    asset, _ = build_helios_plant("tomato", 75, seed=1)
    return asset


def _rel(xg, xc, x0):
    """max |x_gpu - x_cpu| relative to the max displacement of the CPU run."""
    return np.abs(xg - xc).max() / max(np.abs(xc - x0).max(), 1e-300)


# ---------------------------------------------------------------- parity with the CPU solver
def test_parity_tomato_one_step_and_3s_rollout(tomato):
    """Same topology and parameters: float64 GPU <= 1e-9 relative to the CPU float64 solver
    over one step and a 3 s rollout (E = 1e9 and 5e8 in one batch, h = 1/60, alpha = 5/s)."""
    P = [tomato.physics.copy(stem_youngs_modulus=E) for E in (1e9, 5e8)]
    cpu = TreeSim(tomato, P)
    gpu = {p: TreeSimGPU(tomato, P, precision=p) for p in ("float64", "float32")}
    x0 = cpu.state.x
    report = {}
    for i in range(180):
        cpu.step(1 / 60, damping=5.0)
        for g in gpu.values():
            g.step(1 / 60, damping=5.0)
        if i + 1 in (1, 180):
            xc = cpu.state.x
            for p, g in gpu.items():
                report[(p, i + 1)] = (_rel(g.state.x, xc, x0), np.abs(g.state.x - xc).max(),
                                      np.abs(g.u - cpu.u).max() / np.abs(cpu.u).max())
    for k, v in report.items():
        print(f"{k[0]} after {k[1]:3d} steps: rel(disp) {v[0]:.2e}  abs {v[1]:.2e} m  rel(u) {v[2]:.2e}")
    for n in (1, 180):
        assert report[("float64", n)][0] <= 1e-9 and report[("float64", n)][2] <= 1e-9
        assert report[("float32", n)][1] < 1e-5                  # fp32: ~2e-7 m, not growing


def test_parity_externals_cantilever_and_tomato(tomato):
    """Point forces, a moving kinematic sphere (rod nodes + leaf vertices) and batch-varying
    parameters, float64 vs CPU."""
    P = [tomato.physics.copy(stem_youngs_modulus=1e9), tomato.physics.copy(stem_youngs_modulus=5e8, damping=2.0)]
    top = tomato.stem.nodes[np.argmax(tomato.stem.nodes[:, 2])]
    sph = SphereCollider(center=np.array([top + [0.03, 0, 0.02], top + [0, 0.03, 0.02]]), radius=0.04,
                         stiffness=500.0, damping=1.0, velocity=[0.0, 0.0, -0.05])
    ext = [sph, PointForce(300, [0.05, 0.0, 0.02])]
    cpu, gpu = TreeSim(tomato, P), TreeSimGPU(tomato, P, precision="float64")
    x0 = cpu.state.x
    for _ in range(60):
        cpu.step(1 / 60, ext)
        gpu.step(1 / 60, ext)
    assert _rel(gpu.state.x, cpu.state.x, x0) <= 1e-9
    # dense (B, N, 3) node-force array + joint damping + sphere on a cantilever
    a = cantilever_asset(20, 0.5, 0.003)
    Pc = [a.physics.copy(damping=5.0), a.physics.copy(stem_youngs_modulus=3e8, damping=1.0)]
    F = np.zeros((2, 21, 3))
    F[:, 20] = [0.0, 0.01, -0.02]
    F[1, 10] = [0.01, 0.0, 0.0]
    sc = SphereCollider(center=[0.5, 0.0, 1.016], radius=0.02, stiffness=5e3)
    cpu = TreeSim(a, Pc, joint_damping=0.01)
    gpu = TreeSimGPU(a, Pc, joint_damping=0.01, precision="float64")
    x0 = cpu.state.x
    for i in range(120):
        e = F if i % 2 else [sc]
        cpu.step(1 / 60, e)
        gpu.step(1 / 60, e)
    assert _rel(gpu.state.x, cpu.state.x, x0) <= 1e-9
    # zero-copy device path (RL): node forces as a device vec3 array, env-major (B * N,)
    g2 = TreeSimGPU(a, Pc, joint_damping=0.01, precision="float64")
    g3 = TreeSimGPU(a, Pc, joint_damping=0.01, precision="float64")
    dF = wp.array(F.reshape(-1, 3), dtype=wp.vec3d, device="cuda:0")
    for _ in range(30):
        g2.step(1 / 60, dF)
        g3.step(1 / 60, F)
    assert np.abs(g2.state.x - g3.state.x).max() == 0.0


def test_settle_and_static_match_cpu(tomato):
    """Device-side settle (quiet window) and pseudo-time static solve reproduce TreeSim."""
    P = [tomato.physics.copy(stem_youngs_modulus=E) for E in (5e8, 1e9)]
    c = TreeSim(tomato, P)
    ic = c.settle(h=1 / 30, max_time=3.0, tol=1e-3, damping=5.0)
    g = TreeSimGPU(tomato, P, precision="float64")
    ig = g.settle(h=1 / 30, max_time=3.0, tol=1e-3, damping=5.0)
    assert ig["steps"] == ic["steps"]
    np.testing.assert_allclose(ig["t_converged"], ic["t_converged"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ig["speed_history"], ic["speed_history"], rtol=1e-8, atol=1e-12)
    assert np.abs(g.state.x - c.state.x).max() < 1e-12
    c, g = TreeSim(tomato, P[1:]), TreeSimGPU(tomato, P[1:], precision="float64")
    sc, sg = c.solve_static(h=0.2, iters=300, tol=1e-7), g.solve_static(h=0.2, iters=300, tol=1e-7)
    assert sg["iterations"] == sc["iterations"] and sg["converged"].all()
    assert np.abs(g.state.x - c.state.x).max() < 1e-12


# ---------------------------------------------------------------- B1 analytic tests, float32
def test_fp32_cantilever_tip_load_within_1pct():
    n, L, r, E = 20, 1.0, 0.005, 1e9
    _, I = section(r)
    P = 1e-3 * 3 * E * I / L ** 3
    ref = tip_load_linear(P, L, E, I)
    a = cantilever_asset(n, L, r, E=E)
    d = []
    for h in (1 / 60, 1 / 240, 1 / 1000):
        s = TreeSimGPU(a, a.physics.copy(damping=15.0), gravity=ZERO_G, precision="float32")
        assert s.settle(h=h, max_time=5.0, tol=1e-7, external=[PointForce(n, [0, 0, -P])])["converged"].all()
        d.append(float(1.0 - s.state.x[0, -1, 2]))
    err = np.array(d) / ref - 1
    print("fp32 tip-load error", err, "model 1/(2n^2) =", 1 / (2 * n * n))
    assert (np.abs(err) < 0.01).all()
    assert (np.abs(err - 1 / (2 * n * n)) < 1e-4).all()       # = the discrete model's own error
    assert np.ptp(d) / np.mean(d) < 1e-3                       # h-independent equilibrium


def test_fp32_first_frequency_within_3pct():
    n, L, r, E, rho = 20, 0.5, 0.003, 1e9, 1000.0
    A, I = section(r)
    f_ref = first_frequency(E, I, rho * A, L)
    a = cantilever_asset(n, L, r, E=E, rho=rho)
    s = TreeSimGPU(a, a.physics.copy(damping=0.0), gravity=ZERO_G, precision="float32")
    P = 1e-3 * 3 * E * I / L ** 3
    s.settle(h=1 / 240, max_time=5.0, tol=1e-7, external=[PointForce(n, [0, 0, -P])], damping=30.0)
    h = 1 / 1000
    z = s.run(2.0, h, record=lambda sim: sim.state.x[0, -1, 2].copy())
    z = z - z[len(z) // 2:].mean()
    i = np.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))
    tc = (i - z[i] / (z[i + 1] - z[i])) * h
    f = (len(tc) - 1) / (tc[-1] - tc[0])
    print(f"fp32 frequency error {f / f_ref - 1:+.4%}")
    assert abs(f / f_ref - 1) < 0.03


def test_fp32_mass_ratio_1000_stable_at_60hz():
    n, L, r, E = 30, 0.3, 0.001, 2e11
    A, I = section(r)
    M = 1000.0 * 1000.0 * A * L
    a = cantilever_asset(n, L, r, E=E, tip_mass=M, tip_radius=0.01)
    _, xr, zr, _ = elastica(L, E * I, P=M * 9.81, w=1000.0 * A * 9.81)
    out = {}
    for p in ("float32", "float64"):
        s = TreeSimGPU(a, a.physics.copy(damping=5.0), precision=p)
        info = s.settle(h=1 / 60, max_time=20.0, tol=1e-6)
        assert info["converged"].all()
        out[p] = s.state.x[0, -1]
    tip = out["float32"]
    print("fp32 mass ratio: tip z err", (1.0 - tip[2]) / (-zr[-1]) - 1, "x err", tip[0] / xr[-1] - 1,
          "|fp32 - fp64|", np.abs(out["float32"] - out["float64"]).max())
    assert np.isfinite(tip).all()
    assert abs((1.0 - tip[2]) / (-zr[-1]) - 1) < 0.01 and abs(tip[0] / xr[-1] - 1) < 0.01
    assert np.abs(out["float32"] - out["float64"]).max() < 1e-5


def test_fp32_energy_never_increases(tomato):
    s = TreeSimGPU(tomato, [tomato.physics.copy(damping=0.0, stem_youngs_modulus=3e8)], gravity=ZERO_G,
                   precision="float32")
    s.u = np.random.default_rng(0).normal(size=s.u.shape) * 2.0
    E0 = sum(s.energy())[0]
    Es = s.run(10.0, 1 / 60, record=lambda sim: sum(sim.energy())[0])
    assert np.isfinite(Es).all() and Es.max() <= E0 * (1 + 1e-6)
    assert np.all(np.diff(Es) <= 1e-6 * E0)


# ---------------------------------------------------------------- tomato: settle + inverse hold
def test_tomato_settles_on_gpu_fp32(tomato):
    s = TreeSimGPU(tomato, [tomato.physics.copy(stem_youngs_modulus=1e9)], precision="float32")
    x0 = s.state.x[0]
    info = s.settle(h=1 / 30, max_time=3.0, tol=1e-3, damping=5.0)
    assert info["converged"].all(), info["max_speed"]
    x = s.state.x
    stem = tomato.stem.organ == 0
    sag = (x0[stem, 2].max() - x[0, stem, 2].max()) / x0[:, 2].max()
    print(f"fp32 settle: t = {info['t_converged'][0]:.3f} s, sag {sag:.4%}")
    assert 0 < sag < 0.03 and np.abs(x - x0).max() < 0.5
    xe = x.copy()
    s.run(1.0, 1 / 30, damping=5.0, record=lambda sim: 0)
    assert np.abs(s.state.x - xe).max() < 2e-3


def _points(sim):
    return np.concatenate([sim.state.x] + sim.leaf_vertices(), 1)


def _hold(sim, duration=3.0, h=1 / 60, perturb=0.0, damping=5.0, seed=0):
    p0 = _points(sim)
    if perturb:
        sim.u = np.random.default_rng(seed).normal(size=sim.u.shape) * perturb
    sim.run(duration, h, damping=damping, record=lambda s: 0)
    drift = np.linalg.norm(_points(sim) - p0, axis=-1).max(1)
    vn, vp = sim.point_speeds()
    return drift, np.maximum(vn, vp)


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_inverse_statics_holds_tomato_on_gpu(tomato, precision):
    """B2's hold test, entirely on the GPU: inverse + Sturm count, then a 3 s hold."""
    Es = [3e8, 5e8, 1e9]
    sim = TreeSimGPU(tomato, [tomato.physics.copy(stem_youngs_modulus=E) for E in Es], precision=precision)
    res = inverse_statics_gpu(sim)
    print(precision, "status", res.status, "negative", res.n_negative, "over capacity", (~res.joint_ok).sum(1))
    assert res.status == ["infeasible", "unstable", "stable"]
    assert list(res.n_negative) == [65, 11, 0] and list((~res.joint_ok).sum(1)) == [10, 0, 0]
    fwd = TreeSimGPU(tomato, [tomato.physics.copy(stem_youngs_modulus=1e9)], precision=precision)
    dfwd, _ = _hold(fwd)
    apply_rest(sim, res)
    drift, speed = _hold(sim)
    print(precision, "forward-only drift", dfwd, "inverse hold drift", drift, "speed", speed)
    assert dfwd[0] > 0.05
    assert drift[2] < 1e-3 and speed[2] < 1e-3                # E = 1e9 holds
    # exact but unstable equilibrium: fp64 holds it to ~1e-12 m; in fp32 round-off (~1e-7)
    # seeds the unstable modes and they grow to ~0.7 mm in 3 s
    assert drift[1] < (1e-9 if precision == "float64" else 1e-2)
    assert drift[0] > 0.05                                     # infeasible: collapses
    sim2 = TreeSimGPU(tomato, [tomato.physics.copy(stem_youngs_modulus=E) for E in Es[1:]], precision=precision)
    apply_rest(sim2, inverse_statics_gpu(sim2))
    drift_p, speed_p = _hold(sim2, perturb=1e-3)
    assert drift_p[0] > 1e-2                                   # unstable: kicked, it leaves
    assert drift_p[1] < 1e-3 and speed_p[1] < 1e-3             # stable: kicked, it returns


def test_inverse_gpu_matches_cpu(tomato):
    """Rest rotations, deviations, statuses, negative-mode counts and margins vs B2 (CPU)."""
    P = [tomato.physics.copy(stem_youngs_modulus=4e8, stem_density=900.0),
         tomato.physics.copy(stem_youngs_modulus=1e9, leaf_areal_density=0.3),
         tomato.physics.copy(stem_youngs_modulus=6e8, stem_poisson=0.3)]
    rc = inverse_statics(TreeSim(tomato, P), min_eig=True)
    for p, tol in (("float64", 1e-12), ("float32", 5e-6)):
        rg = inverse_statics_gpu(TreeSimGPU(tomato, P, precision=p), min_eig=True)
        ok = rc.joint_ok & rg.joint_ok
        eR = max(np.abs(rg.R0rel[b] - rc.R0rel[b])[ok[b]].max() for b in range(3))
        eQ = max(np.abs(rg.Q[b] - rc.Q[b])[ok[b]].max() for b in range(3))
        print(f"{p}: R0rel err {eR:.1e}, Q err {eQ:.1e}, margin {rg.margin} vs {rc.margin}")
        assert eR < tol and eQ < tol
        assert rg.status == rc.status and np.array_equal(rg.n_negative, rc.n_negative)
        assert np.array_equal(rg.joint_ok, rc.joint_ok)
        np.testing.assert_allclose(rg.margin, rc.margin, rtol=1e-8 if p == "float64" else 1e-4)
        np.testing.assert_allclose(rg.min_eig, rc.min_eig, rtol=1e-8 if p == "float64" else 1e-4)


def test_critical_modulus_gpu(tomato):
    g = TreeSimGPU(tomato, [tomato.physics.copy(stem_youngs_modulus=1e9)], precision="float32")
    c = critical_modulus_gpu(g, rtol=1e-4)
    print("GPU fp32 E_cr", c["E_cr"])
    assert abs(c["E_cr"] / 7.316e8 - 1) < 1e-3                 # B2 (CPU): 7.316e8
    assert abs(c["E_cr_linear_at_hi"] / c["E_cr"] - 1) < 0.01


def test_inverse_batch_screening(tomato):
    """1024 moduli at once (summary readback): the stable set is an interval [E_cr, inf)."""
    Es = np.geomspace(3e8, 3e9, 1024)
    g = TreeSimGPU(tomato, [tomato.physics.copy(stem_youngs_modulus=E) for E in Es], precision="float32")
    t = time.perf_counter()
    r = inverse_statics_gpu(g, full=False)
    dt = time.perf_counter() - t
    first = int(np.argmax(r.stable))
    print(f"1024 inverse + Sturm counts in {dt * 1e3:.1f} ms; first stable E = {Es[first]:.4g}")
    assert r.stable[first:].all() and not r.stable[:first].any()
    assert Es[first - 1] < 7.316e8 <= Es[first] * (1 + 1e-3)


# ---------------------------------------------------------------- batch consistency
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_batch_env_equals_single_run(tomato, precision):
    P = [tomato.physics.copy(stem_youngs_modulus=3e8, stem_density=900.0, damping=1.0),
         tomato.physics.copy(stem_youngs_modulus=1e9, leaf_areal_density=0.3, damping=3.0),
         tomato.physics.copy(stem_youngs_modulus=6e8, stem_poisson=0.3, damping=8.0)]
    sb = TreeSimGPU(tomato, P, precision=precision)
    xb = sb.run(0.5, 1 / 60, record=lambda sim: sim.state.x)
    for b, p in enumerate(P):
        x1 = TreeSimGPU(tomato, [p], precision=precision).run(0.5, 1 / 60, record=lambda sim: sim.state.x)
        np.testing.assert_allclose(xb[:, b], x1[:, 0], rtol=0, atol=1e-11 if precision == "float64" else 1e-6)


def test_heterogeneous_topologies_in_one_batch(tomato):
    """Stretch goal: different plant topologies batched as one flattened multi-tree."""
    cant = cantilever_asset(20, 0.5, 0.003)
    small = cantilever_asset(7, 0.2, 0.002, direction=(0.0, 1.0, 0.5))
    assets = [tomato, cant, tomato, small]
    P = [tomato.physics.copy(stem_youngs_modulus=1e9), cant.physics.copy(damping=1.0),
         tomato.physics.copy(stem_youngs_modulus=6e8, stem_density=900.0), small.physics.copy(damping=0.5)]
    g = TreeSimGPU(assets, P, precision="float64")
    for _ in range(30):
        g.step(1 / 60)
    for e in range(4):
        c = TreeSim(assets[e], [P[e]])
        c.run(0.5, 1 / 60, record=lambda sim: 0)
        assert np.abs(g.env_state(e).x - c.state.x).max() < 1e-12


# ---------------------------------------------------------------- contact slot
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_contact_slot_factor_reuse(tomato, precision):
    """B3's slot: forces folded into the stored factor (slot_apply) and the factor response
    reproduce a step with the same force applied directly."""
    P = [tomato.physics.copy(stem_youngs_modulus=1e9)]
    g = TreeSimGPU(tomato, P, precision=precision)
    g.run(0.3, 1 / 60, record=lambda sim: 0)
    Q, u = g.Q, g.u
    T = g.topo
    node, f = 300, np.array([0.05, -0.02, 0.03])
    b = T.node_body[node]
    _, o, tip, _, _ = g._host_kin()
    df = np.zeros((1, T.nb, 6))
    df[0, b, :3] = np.cross(tip[0, b] - o[0, b], f)
    df[0, b, 3:] = f
    g.factorize(1 / 60)
    dud = g.factor_response(df)
    u0 = g.step(1 / 60, return_udot=True)
    x_free = g.state.x
    g.reset(Q, u)
    u1 = g.step(1 / 60, [PointForce(node, f)], return_udot=True)
    x_pf = g.state.x
    g.reset(Q, u)
    g.contact = ConstantBodyForces(g, df)
    u2 = g.step(1 / 60, return_udot=True)
    tol = 1e-12 if precision == "float64" else 1e-5
    scale = np.abs(dud).max()
    assert np.abs((u1 - u0) - dud).max() < tol * scale * 10
    assert np.abs(u2 - u1).max() < tol * max(scale, np.abs(u1).max()) * 10
    assert np.abs(g.state.x - x_pf).max() < (1e-14 if precision == "float64" else 1e-6)
    assert np.abs(x_pf - x_free).max() > 1e-6


# ---------------------------------------------------------------- throughput
def test_throughput_and_memory(tomato):
    """float32 tomato at B = 256 must beat the CPU numpy baseline (~560 env-steps/s) by >= 20x
    (measured ~2.6e5 env-steps/s; full table: scripts/gpu_benchmark.py)."""
    B = 256
    g = TreeSimGPU(tomato, [tomato.physics.copy(stem_youngs_modulus=1e9)] * B, precision="float32")
    for _ in range(5):
        g.step(1 / 60)
    wp.synchronize()
    n = 100
    t = time.perf_counter()
    for _ in range(n):
        g.step(1 / 60)
    wp.synchronize()
    rate = B * n / (time.perf_counter() - t)
    mem = g.memory_bytes() / B
    print(f"B = {B}: {rate:.3g} env-steps/s ({rate / CPU_BASELINE:.0f}x CPU), {mem / 1e3:.0f} kB/env")
    assert rate > 20 * CPU_BASELINE
    assert mem < 1e6
