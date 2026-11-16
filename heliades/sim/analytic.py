"""Analytic references and test assets: Euler-Bernoulli cantilevers and the planar
elastica (tip load + self-weight) solved as a BVP."""
from __future__ import annotations

import numpy as np

from planttwin.asset import PhysicsParams, PlantAsset, StemGraph


def cantilever_asset(n: int, L: float, r: float, *, E=1e9, rho=1000.0, nu=0.4,
                     z: float = 1.0, tip_mass: float = 0.0, tip_radius: float = 0.0,
                     damping: float = 0.0, direction=(1.0, 0.0, 0.0)) -> PlantAsset:
    """Straight rod of n segments, clamped at node 0, along `direction` at height z."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    nodes = np.array([0.0, 0.0, z]) + np.linspace(0, L, n + 1)[:, None] * d
    edges = np.stack([np.arange(n), np.arange(1, n + 1)], 1)
    fixed = np.zeros(n + 1, bool)
    fixed[0] = True
    pm = np.zeros(n + 1)
    pm[-1] = tip_mass
    from planttwin.asset import Fruit
    fruits = [Fruit(node=n, radius=tip_radius, mass=tip_mass)] if tip_mass > 0 else []
    stem = StemGraph(nodes, edges, np.full(n + 1, r), fixed, point_mass=pm,
                     organ=np.zeros(n + 1, int))
    phys = PhysicsParams(stem_density=rho, stem_youngs_modulus=E, stem_poisson=nu, damping=damping)
    return PlantAsset(stem, [], phys, fruits=fruits, meta={"kind": "cantilever", "n": n, "L": L, "r": r})


def section(r):
    A = np.pi * r * r
    I = np.pi * r ** 4 / 4
    return A, I


def tip_load_linear(P, L, E, I):
    return P * L ** 3 / (3 * E * I)


def self_weight_linear(w, L, E, I):
    return w * L ** 4 / (8 * E * I)


def first_frequency(E, I, rhoA, L):
    """Euler-Bernoulli clamped-free first natural frequency [Hz]."""
    return 1.875104068711961 ** 2 / (2 * np.pi) * np.sqrt(E * I / (rhoA * L ** 4))


def elastica(L, EI, P=0.0, w=0.0, n_mesh=400):
    """Planar inextensible elastica, clamped horizontal at s=0, tip load P and uniform
    weight w per length (both downward). EI theta'' = -(P + w (L - s)) cos(theta),
    theta(0) = 0, theta'(L) = 0, x' = cos, z' = -sin. Returns (s, x, z, theta)."""
    from scipy.integrate import solve_bvp

    def f(s, y):
        th, k, x, z = y
        return np.vstack([k, -(P + w * (L - s)) * np.cos(th) / EI, np.cos(th), -np.sin(th)])

    def bc(ya, yb):
        return np.array([ya[0], yb[1], ya[2], ya[3]])

    s = np.linspace(0, L, n_mesh)
    # initial guess: linear-beam shape
    th0 = (P * (L * s - s ** 2 / 2) + w * (L ** 3 - (L - s) ** 3) / 6) / EI
    th0 = np.clip(th0, 0, 1.5)
    y0 = np.vstack([th0, np.gradient(th0, s), np.cumsum(np.cos(th0)) * (s[1] - s[0]),
                    -np.cumsum(np.sin(th0)) * (s[1] - s[0])])
    sol = solve_bvp(f, bc, s, y0, tol=1e-10, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    y = sol.sol(s)
    return s, y[2], y[3], y[0]


def with_hanging_fruit(asset: PlantAsset, n_fruit: int = 6, mass: float = 0.011, radius: float = 0.014,
                       pedicel_len: float = 0.02, pedicel_radius: float = 0.0012, seed: int = 0,
                       organ_pick: int = 0) -> PlantAsset:
    """Copy of `asset` with `n_fruit` fruits (point mass + sphere inertia) hanging straight
    down on short pedicels from random free nodes of organ `organ_pick` (upper 70% of the
    plant). Reproduces the "11 g fruit on near-massless pedicels" stress case."""
    from copy import deepcopy

    from planttwin.asset import Fruit
    a = deepcopy(asset)
    st = a.stem
    rng = np.random.default_rng(seed)
    org = st.organ if st.organ is not None else np.zeros(len(st.nodes), int)
    cand = np.flatnonzero((org == organ_pick) & ~st.fixed & (st.nodes[:, 2] > 0.3 * st.nodes[:, 2].max()))
    picks = rng.choice(cand, size=min(n_fruit, len(cand)), replace=False)
    N = len(st.nodes)
    new = st.nodes[picks] - np.array([0, 0, pedicel_len])
    st.nodes = np.vstack([st.nodes, new])
    st.edges = np.vstack([st.edges, np.stack([picks, N + np.arange(len(picks))], 1)])
    st.radius = np.concatenate([st.radius, np.full(len(picks), pedicel_radius)])
    st.fixed = np.concatenate([st.fixed, np.zeros(len(picks), bool)])
    pm = st.point_mass if st.point_mass is not None else np.zeros(N)
    st.point_mass = np.concatenate([pm, np.full(len(picks), mass)])
    st.organ = np.concatenate([org, np.full(len(picks), 3)])
    a.fruits = list(a.fruits) + [Fruit(node=int(N + i), radius=radius, mass=mass) for i in range(len(picks))]
    return a
