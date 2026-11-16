"""Acceptance tests for heliades.sim B2: exact discrete inverse statics + stability.

See docs/sim/INVERSE.md and migration/notes/B2.md. Tomato cases need pyhelios.
"""
import numpy as np
import pytest

from heliades.sim import TreeSim
from heliades.sim.analytic import cantilever_asset, section
from heliades.sim.hessian import (dense_hessian, eliminate, min_eigenvalue, newton_static, potential,
                                  statics, tree_matvec, tree_solve)
from heliades.sim.inverse import (apply_rest, critical_modulus, inverse_statics, rest_shape_nodes,
                                  solve_joint_deviation, stiffen_to_stable, target_from_state)
from heliades.sim.rotation import qmul, quat_from_rotvec
from heliades.sim.tree import PointForce

GREENHILL = 7.837


@pytest.fixture(scope="module")
def tomato():
    pytest.importorskip("pyhelios")
    from planttwin.helios_import import build_helios_plant
    asset, _ = build_helios_plant("tomato", 75, seed=1)
    return asset


def _sims(asset, Es, **kw):
    return TreeSim(asset, [asset.physics.copy(stem_youngs_modulus=E, **kw) for E in Es])


def _settle_exact(sim, external=None):
    sim.solve_static(h=0.2, iters=400, tol=1e-7, external=external)
    info = newton_static(sim, external=external)
    assert info["converged"].all(), info["residual"]
    return info


def _round_trip(asset, Es, external=None):
    """rest (asset) -> forward static equilibrium -> inverse -> recovered rest."""
    sim = _sims(asset, Es)
    _settle_exact(sim, external)
    loaded = sim.state.x.copy()
    res = inverse_statics(sim, target_from_state(sim), external)
    err_R = np.abs(res.R0rel - sim.topo.R0rel[None]).max((1, 2, 3))
    err_x = np.abs(rest_shape_nodes(sim, res) - asset.stem.nodes[None]).max((1, 2))
    sag = np.abs(loaded - asset.stem.nodes[None]).max((1, 2))
    return res, err_R, err_x, sag


# ---------------------------------------------------------------- building blocks
def test_tangent_stiffness_matches_finite_differences(tomato):
    """Gradient, Hessian (incl. gravity geometric stiffness), O(n) matvec/solve and the Sturm
    count against finite differences / dense linear algebra, at a non-equilibrium state."""
    sim = _sims(tomato, [5e8])
    T = sim.topo
    rng = np.random.default_rng(0)
    sim.Q = qmul(sim.Q, quat_from_rotvec(rng.normal(size=(1, T.nb, 3)) * 0.1))
    ext = [PointForce(100, [0.1, 0.2, -0.3])]
    st = statics(sim, external=ext)
    H = dense_hessian(T, st, symmetric=False)[0]
    eps = 1e-6
    for j in (0, 7, 150, T.nb - 1):
        for a in range(3):
            d = np.zeros((1, T.nb, 3))
            d[0, j, a] = eps
            Qp, Qm = qmul(sim.Q, quat_from_rotvec(d)), qmul(sim.Q, quat_from_rotvec(-d))
            dV = (potential(sim, Qp, ext) - potential(sim, Qm, ext))[0] / (2 * eps)
            assert abs(dV - st.g[0, j, a]) < 1e-7 * (1 + abs(st.g[0, j, a]))
            col = ((statics(sim, Qp, ext).g - statics(sim, Qm, ext).g) / (2 * eps)).reshape(-1)
            assert np.abs(col - H[:, 3 * j + a]).max() < 1e-7 * (1 + np.abs(col).max())
    x = rng.normal(size=(1, T.nb, 3))
    y = tree_matvec(T, st.Dw, st.loads.A, x)
    yd = np.einsum("nij,nj->ni", st.R[0], (H @ np.einsum("nji,nj->ni", st.R[0], x[0]).reshape(-1)).reshape(-1, 3))
    np.testing.assert_allclose(y[0], yd, atol=1e-10 * np.abs(y).max())
    fac = eliminate(T, st.Dw, st.loads.A, symmetric=False)
    np.testing.assert_allclose(tree_solve(T, fac, y), x, atol=1e-10)
    ev = np.linalg.eigvalsh(dense_hessian(T, st)[0])
    assert eliminate(T, st.Dw, st.loads.A).n_negative[0] == (ev < 0).sum() > 0
    assert abs(min_eigenvalue(T, st)[0] / ev[0] - 1) < 1e-8


def test_local_joint_solve_and_capacity():
    """2 Gamma K v = m on the stable branch; beyond the spring's capacity (|m| > k for
    isotropic K: torque = k sin(theta)) the joint is flagged."""
    k = np.array([2.0, 2.0, 1.5])
    rng = np.random.default_rng(1)
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    m = d * rng.uniform(0, 1.3, size=(200, 1))
    Q, ok = solve_joint_deviation(m, np.broadcast_to(k, m.shape))
    w, v = Q[..., 0], Q[..., 1:]
    Kv = k * v
    r = 2 * (w[:, None] * Kv - np.cross(v, Kv)) - m
    assert np.abs(r[ok]).max() < 1e-12
    iso = np.array([2.0, 2.0, 2.0])
    mi = d * np.array([0.5, 1.9, 2.1, 3.0]).repeat(50)[:, None]
    Qi, oki = solve_joint_deviation(mi, np.broadcast_to(iso, mi.shape))
    mag = np.linalg.norm(mi, axis=1)
    assert oki[mag < 2.0].all() and not oki[mag > 2.0].any()
    th = 2 * np.arctan2(np.linalg.norm(Qi[oki, 1:], axis=1), Qi[oki, 0])
    np.testing.assert_allclose(2.0 * np.sin(th), mag[oki], rtol=1e-10)   # k sin(theta) = |m|


# ---------------------------------------------------------------- 1. round trip
@pytest.mark.parametrize("case", ["horizontal_tip_mass", "inclined_up", "large_deflection", "point_load"])
def test_round_trip_cantilever(case):
    ext = None
    if case == "horizontal_tip_mass":
        a = cantilever_asset(20, 0.5, 0.003, tip_mass=0.02, tip_radius=0.01)
        Es = [3e8, 1e9, 5e9]
    elif case == "inclined_up":
        a = cantilever_asset(25, 0.4, 0.002, direction=(1.0, 0.3, 1.0))
        Es = [2e8, 1e9]
    elif case == "large_deflection":
        a = cantilever_asset(30, 1.0, 0.004)
        Es = [2e7, 1e8]
    else:
        a = cantilever_asset(15, 0.3, 0.002, direction=(0.0, 1.0, 0.2))
        ext = [PointForce(15, [0.05, -0.02, -0.1]), PointForce(7, [0.0, 0.03, 0.0])]
        Es = [1e9]
    res, err_R, err_x, sag = _round_trip(a, Es, ext)
    print(case, "sag", sag, "rest err R", err_R, "x", err_x)
    assert (sag > 1e-3).all()                              # really loaded
    assert (err_R < 1e-8).all() and (err_x < 1e-8).all()
    assert res.stable.all()


@pytest.mark.parametrize("E", [5e8, 1e9])
def test_round_trip_tomato(tomato, E):
    res, err_R, err_x, sag = _round_trip(tomato, [E])
    print(f"tomato E={E:g}: loaded max disp {sag[0] * 1e3:.1f} mm, rest err R {err_R[0]:.1e}, x {err_x[0]:.1e} m")
    assert sag[0] > 0.1                                    # leaflets droop by > 10 cm
    assert err_R[0] < 1e-8 and err_x[0] < 1e-8
    assert res.status == ["stable"]                        # a forward-settled state is stable


# ---------------------------------------------------------------- 2. hold the target
def _points(sim):
    """Rod nodes + leaf vertices, (B, P, 3)."""
    return np.concatenate([sim.state.x] + sim.leaf_vertices(), 1)


def _hold(sim, duration=3.0, h=1 / 60, perturb=0.0, damping=5.0, seed=0):
    p0 = _points(sim)
    if perturb:
        sim.u = np.random.default_rng(seed).normal(size=sim.u.shape) * perturb
        sim._kin_cache = None
    sim.run(duration, h, damping=damping, record=lambda s: 0)
    drift = np.linalg.norm(_points(sim) - p0, axis=-1).max(1)
    vn, vp = sim.point_speeds()
    return drift, np.maximum(vn, vp)


def test_hold_target_tomato(tomato):
    Es = [3e8, 5e8, 1e9]
    sim = _sims(tomato, Es)
    res = inverse_statics(sim)
    print("status", res.status, "negative modes", res.n_negative, "over-capacity joints", (~res.joint_ok).sum(1))
    assert res.status == ["infeasible", "unstable", "stable"]
    # forward-only (asset taken as rest) sags; the inverse rest holds the target
    fwd = _sims(tomato, [1e9])
    dfwd, _ = _hold(fwd)
    apply_rest(sim, res)
    st = statics(sim)
    assert np.abs(st.g[1:]).max() < 1e-12                  # exact discrete equilibrium (feasible ones)
    drift, speed = _hold(sim)
    print("forward-only drift (1e9)", dfwd, "inverse hold drift", drift, "speed", speed)
    assert dfwd[0] > 0.05
    assert drift[2] < 1e-3 and speed[2] < 1e-3            # E = 1e9: holds
    assert drift[1] < 1e-3                                 # E = 5e8: exact but unstable equilibrium...
    assert drift[0] > 0.05                                 # E = 3e8: no rest shape holds it -> collapses
    # ...a 1e-3 rad/s kick separates stable from unstable
    sim2 = _sims(tomato, Es[1:])
    apply_rest(sim2, inverse_statics(sim2))
    drift_p, speed_p = _hold(sim2, perturb=1e-3)
    print("perturbed drift", drift_p, "speed", speed_p)
    assert drift_p[0] > 1e-2                               # unstable: grows to > 1 cm
    assert drift_p[1] < 1e-3 and speed_p[1] < 1e-3         # stable: decays


def test_best_feasible_stiffening_holds(tomato):
    """Fallback: stiffen minimal unstable subtrees until the inverse rest is stable at E = 5e8."""
    sim = _sims(tomato, [5e8])
    fb = stiffen_to_stable(sim)
    s = fb["s"]
    r = fb["result"]
    print(f"rounds {fb['rounds']}, joints x>1.05: {(s > 1.05).sum()}/{len(s)}, max {s.max():.2f}")
    assert r.status == ["stable"] and s.min() >= 1.0 and s.max() < 2.0
    sim.K = sim.K * s[None, :, None]
    apply_rest(sim, r)
    drift, speed = _hold(sim, perturb=1e-3)
    assert drift[0] < 1e-3 and speed[0] < 1e-3


# ---------------------------------------------------------------- 3. critical modulus
def test_greenhill_column():
    """Vertical clamped column under self-weight: rho A g L^3 / (E_cr I) = 7.837."""
    L, r, rho = 1.0, 0.005, 1000.0
    A, I = section(r)
    Eg = rho * A * 9.81 * L ** 3 / (GREENHILL * I)
    a = cantilever_asset(40, L, r, rho=rho, z=0.0, direction=(0, 0, 1))
    sim = _sims(a, [0.9 * Eg, 0.99 * Eg, 1.01 * Eg, 1.1 * Eg])
    c = critical_modulus(sim, rtol=1e-6)
    res = inverse_statics(sim, min_eig=True)
    print("E_cr / Greenhill - 1 =", c["E_cr"] / Eg - 1, "linear:", res.E_cr_linear / Eg - 1)
    assert abs(c["E_cr"] / Eg - 1) < 0.02
    np.testing.assert_allclose(res.E_cr_linear, c["E_cr"], rtol=1e-4)  # generalized eigenproblem
    assert res.status == ["unstable", "unstable", "stable", "stable"]
    assert res.n_negative[0] == 2                          # the two lateral (x, y) buckling modes
    # equivalently the critical height at fixed E
    Lcr = (GREENHILL * c["E_cr"] * I / (rho * A * 9.81)) ** (1 / 3)
    assert abs(Lcr / L - 1) < 0.01


def test_column_classification_matches_dynamics():
    """Below E_cr the straight column (an exact equilibrium) falls when kicked; above it holds."""
    L, r, rho = 1.0, 0.005, 1000.0
    A, I = section(r)
    Eg = rho * A * 9.81 * L ** 3 / (GREENHILL * I)
    a = cantilever_asset(20, L, r, rho=rho, z=0.0, direction=(0, 0, 1))
    sim = _sims(a, [0.5 * Eg, 1.25 * Eg])
    res = inverse_statics(sim)
    assert res.status == ["unstable", "stable"]
    apply_rest(sim, res)
    drift, _ = _hold(sim, duration=3.0, perturb=1e-3, damping=1.0)
    assert drift[0] > 0.1 and drift[1] < 1e-3


def test_tomato_critical_modulus(tomato):
    """Exact critical modulus of holding the Helios shape, vs the linearised generalized
    eigenproblem and the main-axis-only mode (comparable to B1's forward estimate ~2e8)."""
    sim = _sims(tomato, [1e9])
    T = sim.topo
    c = critical_modulus(sim, rtol=1e-4)
    stem = tomato.stem.organ[T.node_b] == 0
    nodes = np.flatnonzero(tomato.stem.organ == 0)
    b = T.node_body[nodes[np.argmax(tomato.stem.nodes[nodes, 2])]]
    main = np.zeros(T.nb, bool)
    while b >= 0:
        main[b], b = True, T.parent[b]
    c_stem = critical_modulus(sim, rtol=1e-3, stiffness_scale=np.where(stem, 1.0, 1e3))
    c_main = critical_modulus(sim, rtol=1e-3, stiffness_scale=np.where(main, 1.0, 1e3))
    print(f"E_cr all {c['E_cr']:.4g} (linear {c['E_cr_linear_at_hi']:.4g}), stems {c_stem['E_cr']:.4g}, "
          f"main axis {c_main['E_cr']:.4g}")
    assert 5e8 < c["E_cr"] < 1e9
    assert abs(c["E_cr_linear_at_hi"] / c["E_cr"] - 1) < 0.01
    assert c_main["E_cr"] < c_stem["E_cr"] < c["E_cr"]
    assert 1.5e8 < c_main["E_cr"] < 3.5e8                  # B1: ~2e8 from sag amplification


# ---------------------------------------------------------------- 4. batch
def test_batch_equals_single(tomato):
    P = [tomato.physics.copy(stem_youngs_modulus=4e8, stem_density=900.0),
         tomato.physics.copy(stem_youngs_modulus=1e9, leaf_areal_density=0.3),
         tomato.physics.copy(stem_youngs_modulus=6e8, stem_poisson=0.3)]
    rb = inverse_statics(TreeSim(tomato, P), min_eig=True)
    for b, p in enumerate(P):
        r1 = inverse_statics(TreeSim(tomato, [p]), min_eig=True)
        np.testing.assert_allclose(rb.R0rel[b], r1.R0rel[0], rtol=0, atol=1e-13)
        np.testing.assert_allclose(rb.Q[b], r1.Q[0], rtol=0, atol=1e-13)
        assert rb.status[b] == r1.status[0] and rb.n_negative[b] == r1.n_negative[0]
        np.testing.assert_allclose(rb.margin[b], r1.margin[0], rtol=1e-9)
