"""Acceptance tests for heliades.sim (B1: forward tree direct solver).

Analytic cantilever cases run in a few seconds each. The Helios tomato cases need pyhelios
(skipped otherwise). The old-XPBD contrast takes minutes and runs only with HELIADES_SLOW=1.
"""
import os

import numpy as np
import pytest

from heliades.sim import PointForce, SphereCollider, TreeSim
from heliades.sim.analytic import (cantilever_asset, elastica, first_frequency, section,
                                   self_weight_linear, tip_load_linear, with_hanging_fruit)

ZERO_G = (0.0, 0.0, 0.0)
SLOW = os.environ.get("HELIADES_SLOW", "0") == "1"


def _tip_deflection(sim, z0=1.0):
    return z0 - sim.state.x[:, -1, 2]


# ---------------------------------------------------------------- 1. tip load, small deflection
def test_tip_load_matches_beam_theory_and_is_step_independent():
    n, L, r, E = 20, 1.0, 0.005, 1e9
    _, I = section(r)
    P = 1e-3 * 3 * E * I / L ** 3                          # delta/L = 1e-3 (linear regime)
    ref = tip_load_linear(P, L, E, I)
    a = cantilever_asset(n, L, r, E=E)
    d = {}
    for h in (1 / 60, 1 / 240, 1 / 1000):
        s = TreeSim(a, a.physics.copy(damping=15.0), gravity=ZERO_G)
        info = s.settle(h=h, max_time=5.0, tol=1e-9, external=[PointForce(n, [0, 0, -P])])
        assert info["converged"].all()
        d[h] = float(_tip_deflection(s)[0])
    for h, v in d.items():
        assert abs(v / ref - 1) < 0.01, (h, v, ref)
        # rigid-segment discretisation error is exactly 1/(2 n^2) for this model
        assert abs(v / ref - 1 - 1 / (2 * n * n)) < 1e-4
    vals = np.array(list(d.values()))
    assert (vals.max() - vals.min()) / vals.mean() < 1e-3   # h-independent equilibrium


# ---------------------------------------------------------------- 2. self weight
def test_self_weight_converges_in_segment_count():
    L, r, E = 0.3, 0.005, 1e9
    A, I = section(r)
    ref = self_weight_linear(1000.0 * A * 9.81, L, E, I)
    errs = []
    for n in (5, 10, 20, 40):
        a = cantilever_asset(n, L, r, E=E)
        s = TreeSim(a, a.physics.copy(damping=60.0))
        assert s.settle(h=1 / 240, max_time=5.0, tol=1e-8)["converged"].all()
        errs.append(float(_tip_deflection(s)[0] / ref - 1))
    errs = np.array(errs)
    assert np.all(np.diff(np.abs(errs)) < 0)                # monotone convergence
    np.testing.assert_allclose(errs, 1 / np.array([5, 10, 20, 40]) ** 2, rtol=0.05)  # O(1/n^2)
    assert abs(errs[2]) < 0.01                              # n = 20 within 1%


# ---------------------------------------------------------------- 3. large deflection
def test_large_deflection_matches_elastica():
    n, L, r, E = 40, 1.0, 0.005, 1e9
    _, I = section(r)
    EI = E * I
    P = 3.0 * EI / L ** 2                                   # tip drops ~60% of L
    _, xr, zr, _ = elastica(L, EI, P=P)
    a = cantilever_asset(n, L, r, E=E, rho=1e-6)            # weightless rod
    s = TreeSim(a, a.physics.copy(damping=30.0), gravity=ZERO_G, joint_damping=0.01)
    assert s.settle(h=1 / 240, max_time=10.0, tol=1e-8, external=[PointForce(n, [0, 0, -P])])["converged"].all()
    tip = s.state.x[0, -1]
    assert abs((1.0 - tip[2]) / (-zr[-1]) - 1) < 0.02
    assert abs(tip[0] / xr[-1] - 1) < 0.02
    assert -zr[-1] > 0.5 * L                                # really a large deflection


# ---------------------------------------------------------------- 4. dynamics
def test_first_natural_frequency():
    n, L, r, E, rho = 20, 0.5, 0.003, 1e9, 1000.0
    A, I = section(r)
    f_ref = first_frequency(E, I, rho * A, L)
    a = cantilever_asset(n, L, r, E=E, rho=rho)
    s = TreeSim(a, a.physics.copy(damping=0.0), gravity=ZERO_G)
    P = 1e-3 * 3 * E * I / L ** 3
    s.settle(h=1 / 240, max_time=5.0, tol=1e-9, external=[PointForce(n, [0, 0, -P])], damping=30.0)
    h = 1 / 1000
    z = s.run(2.0, h, record=lambda sim: sim.state.x[0, -1, 2].copy())   # release -> ring-down
    z = z - z[len(z) // 2:].mean()
    i = np.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))
    tc = (i - z[i] / (z[i + 1] - z[i])) * h
    f = (len(tc) - 1) / (tc[-1] - tc[0])
    assert abs(f / f_ref - 1) < 0.03, (f, f_ref)


# ---------------------------------------------------------------- 5. mass ratio
def test_mass_ratio_1000_stable_at_60hz():
    n, L, r, E = 30, 0.3, 0.001, 2e11
    A, I = section(r)
    m_chain = 1000.0 * A * L
    M = 1000.0 * m_chain
    a = cantilever_asset(n, L, r, E=E, tip_mass=M, tip_radius=0.01)
    _, xr, zr, _ = elastica(L, E * I, P=M * 9.81, w=1000.0 * A * 9.81)
    s = TreeSim(a, a.physics.copy(damping=5.0))
    info = s.settle(h=1 / 60, max_time=20.0, tol=1e-6)
    assert info["converged"].all()
    tip = s.state.x[0, -1]
    assert np.isfinite(tip).all()
    assert abs((1.0 - tip[2]) / (-zr[-1]) - 1) < 0.01
    assert abs(tip[0] / xr[-1] - 1) < 0.01


# ---------------------------------------------------------------- contact
def test_sphere_collider_pushes_and_balances_beam():
    """A kinematic sphere pressed onto the tip: in equilibrium the penalty force on the rod
    equals the beam's reaction k_beam * deflection (implicit penalty, stiff, stable at 60 Hz)."""
    n, L, r, E = 20, 0.5, 0.003, 1e9
    _, I = section(r)
    a = cantilever_asset(n, L, r, E=E)
    s = TreeSim(a, a.physics.copy(damping=20.0), gravity=ZERO_G)
    R = 0.02
    push = 0.004                                             # sphere centre 4 mm below contact
    sph = SphereCollider(center=[L, 0.0, 1.0 + R - push], radius=R, stiffness=5e3)
    assert s.settle(h=1 / 60, max_time=5.0, tol=1e-7, external=[sph])["converged"].all()
    d = float(_tip_deflection(s)[0])
    k_beam = 3 * E * I / L ** 3 / (1 + 1 / (2 * n * n))
    pen = push - d
    assert 0 < d < push
    assert abs(sph.stiffness * pen / (k_beam * d) - 1) < 0.02


# ---------------------------------------------------------------- tomato
@pytest.fixture(scope="module")
def tomato():
    pytest.importorskip("pyhelios")
    from planttwin.helios_import import build_helios_plant
    asset, _ = build_helios_plant("tomato", 75, seed=0)
    return asset


def _sag(asset, x0, x):
    stem = asset.stem.organ == 0
    H = x0[:, 2].max()
    return (x0[stem, 2].max() - x[:, stem, 2].max(1)) / H


def test_tomato_settles_without_collapse(tomato):
    Es = [5e8, 1e9]
    s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=E) for E in Es])
    x0 = s.state.x[0]
    info = s.settle(h=1 / 30, max_time=3.0, tol=1e-3, damping=5.0)
    assert info["converged"].all(), info["max_speed"]      # < 1 mm/s within 3 s
    x = s.state.x
    assert np.isfinite(x).all()
    sag = _sag(tomato, x0, x)
    disp = np.linalg.norm(x - x0, axis=-1).max(1)
    assert np.all(sag > 0) and sag[0] > sag[1]            # E-dependent, stiffer sags less
    assert sag[1] < 0.03 and sag[0] < 0.06                # bounded (no collapse)
    assert np.all(disp < 0.5)
    # the settled state is a true equilibrium: continuing does not creep
    xe = x.copy()
    s.run(1.0, 1 / 30, damping=5.0, record=lambda sim: 0)
    assert np.abs(s.state.x - xe).max() < 2e-3


def test_tomato_with_heavy_fruit_settles(tomato):
    a = with_hanging_fruit(tomato, n_fruit=6, mass=0.011)
    s = TreeSim(a, [a.physics.copy(stem_youngs_modulus=1e9)])
    info = s.settle(h=1 / 30, max_time=3.0, tol=1e-3, damping=5.0)
    assert info["converged"].all()
    x0 = a.stem.nodes
    assert _sag(a, x0, s.state.x)[0] < 0.03


def test_tomato_static_solve_matches_dynamic_settle(tomato):
    P = [tomato.physics.copy(stem_youngs_modulus=1e9)]
    s1 = TreeSim(tomato, P)
    s1.settle(h=1 / 30, max_time=6.0, tol=1e-5, damping=5.0)
    s2 = TreeSim(tomato, P)
    info = s2.solve_static(h=0.2, iters=300, tol=1e-7)
    assert info["converged"].all()
    assert np.abs(s1.state.x - s2.state.x).max() < 1e-4


def test_tomato_sag_is_E_dependent_and_bounded(tomato):
    """Static equilibria over E = 2e8..1e9: monotone in E, finite, and stable (a perturbed
    dynamic run stays put). Amplification 1/(1 - E_cr/E) shows E_cr ~ 2e8 Pa."""
    Es = [2e8, 3e8, 5e8, 1e9]
    s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=E) for E in Es])
    x0 = s.state.x[0]
    assert s.solve_static(h=0.2, iters=400, tol=1e-6)["converged"].all()
    xs = s.state.x
    sag = _sag(tomato, x0, xs)
    assert np.all(np.diff(sag) < 0) and sag[-1] > 0 and sag[0] < 0.3
    s.u = np.random.default_rng(0).normal(size=s.u.shape) * 1e-3
    s._kin_cache = None
    s.run(1.0, 1 / 30, damping=5.0, record=lambda sim: 0)
    assert np.abs(s.state.x - xs).max() < 1e-3


def test_tomato_soft_stem_buckles_step_independently(tomato):
    """E = 1e8 Pa is below the self-weight buckling limit of this Helios tomato (Greenhill
    estimate with the base radius gives L_crit ~ plant height; sag amplification gives
    E_cr ~ 2e8): the stem folds over to a post-buckled equilibrium (~45% sag). This is a
    property of the model, not of the integrator: h = 1/60 and 1/240 give the same fall."""
    stem = tomato.stem.organ == 0
    tops = []
    for h in (1 / 60, 1 / 240):
        s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=1e8)])
        s.run(2.0, h, damping=5.0, record=lambda sim: 0)
        tops.append(s.state.x[0, stem, 2].max())
    H = tomato.stem.nodes[:, 2].max()
    assert tops[0] < 0.75 * H                                # it falls
    assert abs(tops[0] - tops[1]) < 0.01 * H                 # same answer at 4x smaller step


# ---------------------------------------------------------------- 7. batch consistency
def test_batch_equals_individual_runs(tomato):
    P = [tomato.physics.copy(stem_youngs_modulus=3e8, stem_density=900.0, damping=1.0),
         tomato.physics.copy(stem_youngs_modulus=1e9, leaf_areal_density=0.3, damping=3.0),
         tomato.physics.copy(stem_youngs_modulus=6e8, stem_poisson=0.3, damping=8.0)]
    sb = TreeSim(tomato, P)
    xb = sb.run(0.5, 1 / 60, record=lambda sim: sim.state.x)
    for b, p in enumerate(P):
        s1 = TreeSim(tomato, [p])
        x1 = s1.run(0.5, 1 / 60, record=lambda sim: sim.state.x)
        np.testing.assert_allclose(xb[:, b], x1[:, 0], rtol=0, atol=1e-11)


# ---------------------------------------------------------------- 8. energy
@pytest.mark.parametrize("case", ["cantilever", "tomato"])
def test_energy_does_not_blow_up(case, request):
    if case == "tomato":
        asset = request.getfixturevalue("tomato")
        phys = asset.physics.copy(damping=0.0, stem_youngs_modulus=3e8)
    else:
        asset = cantilever_asset(20, 0.5, 0.003, E=1e9)
        phys = asset.physics.copy(damping=0.0)
    s = TreeSim(asset, [phys], gravity=ZERO_G)
    rng = np.random.default_rng(0)
    s.u = rng.normal(size=s.u.shape) * 2.0
    s._kin_cache = None
    E0 = sum(s.energy())[0]
    Es = s.run(10.0, 1 / 60, record=lambda sim: sum(sim.energy())[0])
    assert np.isfinite(Es).all()
    assert Es.max() <= E0 * (1 + 1e-6)                       # never gains energy
    # implicit Euler dissipates (numerical damping ratio ~ omega h / 2); quantified in
    # migration/notes/B1.md. Here we only require it to be monotone-bounded, never growing.
    print(f"{case}: energy after 10 s = {Es[-1] / E0:.3e} E0")
    assert np.all(np.diff(Es) <= 1e-9 * E0)


# ---------------------------------------------------------------- XPBD contrast (slow)
@pytest.mark.skipif(not SLOW, reason="minutes; set HELIADES_SLOW=1")
def test_old_xpbd_does_not_settle_tomato(tomato):
    from planttwin.sim import PlantModel
    m = PlantModel(tomato)
    m.set_params([tomato.physics.copy(stem_youngs_modulus=1e9, damping=5.0)])
    N = len(tomato.stem.nodes)
    X, _, _ = m.simulate(1.5, fps=30, substeps=128, record=np.arange(N))
    v = np.linalg.norm(X[-1] - X[-2], axis=-1).max() * 30
    assert v > 1e-2                                          # still moving > 1 cm/s
