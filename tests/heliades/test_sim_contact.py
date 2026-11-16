"""Acceptance tests for heliades.sim B3: contact + ground on the tree solver, segment
refinement, contact-aware inverse statics. See docs/sim/CONTACT.md and migration/notes/B3.md.
Tomato cases need pyhelios."""
import numpy as np
import pytest

from heliades.sim import BodyForce, TreeSim
from heliades.sim.analytic import cantilever_asset, section
from heliades.sim.contact import (CapsuleObstacle, ContactParams, GroundPlane, SphereObstacle, _tangents,
                                  delassus_bruteforce, pgs)
from heliades.sim.hessian import newton_static
from heliades.sim.inverse import apply_rest, critical_modulus, inverse_statics, target_from_state
from heliades.sim.inverse_contact import supported_inverse
from heliades.sim.refine import organ_runs, refine_asset
from heliades.sim.rotation import quat_from_rotvec
from heliades.sim.scenarios import (block_asset, finger_for_leaf, floating_chain, min_gap_ground, points,
                                    stake_in_front)

ZERO_G = (0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def tomato():
    pytest.importorskip("pyhelios")
    from planttwin.helios_import import build_helios_plant
    asset, _ = build_helios_plant("tomato", 75, seed=1)
    return asset


# ---------------------------------------------------------------- 1. segment refinement
def test_refine_keeps_nodes_leaves_and_geometry(tomato):
    r, mp = refine_asset(tomato, max_length=0.01, min_per_organ=4, return_map=True)
    N = len(tomato.stem.nodes)
    np.testing.assert_array_equal(r.stem.nodes[:N], tomato.stem.nodes)        # original nodes kept
    assert [L.attach_node for L in r.leaves] == [L.attach_node for L in tomato.leaves]
    E = r.stem.edges
    l = np.linalg.norm(r.stem.nodes[E[:, 1]] - r.stem.nodes[E[:, 0]], axis=1)
    assert l.max() <= 0.01 + 1e-12
    runs = organ_runs(r.stem)
    assert np.bincount(runs).min() >= 4                                         # >= 4 segments per organ run
    # every refined edge lies on its original edge: same polyline, same total length
    E0 = tomato.stem.edges
    L0 = np.linalg.norm(tomato.stem.nodes[E0[:, 1]] - tomato.stem.nodes[E0[:, 0]], axis=1)
    np.testing.assert_allclose(np.bincount(mp.edge_of_new, l, minlength=len(E0)), L0, rtol=1e-12)
    # leaves ride on the body that ends at their (unchanged) attachment node; same leaf mass
    s0, s1 = TreeSim(tomato), TreeSim(r)
    for L in s1.topo.leaves[:50]:
        assert s1.topo.node_b[L.body] == r.leaves[L.asset_index].attach_node
    np.testing.assert_allclose(s1.topo.leaf_part.m.sum(), s0.topo.leaf_part.m.sum(), rtol=1e-12)
    np.testing.assert_allclose(s1.m.sum(), s0.m.sum(), rtol=0.02)               # tapered rods: O(1/m^2)
    # the refined plant in its asset shape is the same plant: same leaf vertex positions
    for V0, V1 in zip(s0.leaf_vertices()[:20], s1.leaf_vertices()[:20]):
        np.testing.assert_allclose(V0, V1, atol=1e-12)


def _ycross(Y):
    """Clamped vertical trunk with one (chain) or two (Y junction) inclined branches + tip masses."""
    from planttwin.asset import Fruit, PhysicsParams, PlantAsset, StemGraph
    nodes, edges = [[0, 0, 0.0]], []
    for i in range(1, 5):
        nodes.append([0, 0, 0.1 * i])
        edges.append((i - 1, i))
    tips = []
    for d in ([(0.5, 0, 1), (-0.5, 0.2, 1)] if Y else [(0.5, 0, 1)]):
        d = np.array(d) / np.linalg.norm(d)
        prev = 4
        for j in range(1, 4):
            nodes.append((np.array(nodes[4]) + 0.1 * j * d).tolist())
            edges.append((prev, len(nodes) - 1))
            prev = len(nodes) - 1
        tips.append(prev)
    N = len(nodes)
    fixed = np.zeros(N, bool)
    fixed[0] = True
    pm = np.zeros(N)
    pm[tips] = 0.01
    st = StemGraph(np.array(nodes), np.array(edges), np.full(N, 0.003), fixed, point_mass=pm, organ=np.zeros(N, int))
    return PlantAsset(st, [], PhysicsParams(), fruits=[Fruit(t, 0.005, 0.01) for t in tips])


def test_refinement_convergence_order_of_critical_modulus():
    """Kinked chain: E_cr converges at O(1/m^2); a junction adds an O(1/m) term (B1's joint
    compliance counts the parent's half segment once per child), which refinement removes."""
    orders = {}
    for Y in (False, True):
        a = _ycross(Y)
        Es = []
        for f in (2, 4, 8, 16):
            r = refine_asset(a, factor=f)
            Es.append(critical_modulus(TreeSim(r, [r.physics.copy(stem_youngs_modulus=1e9)]), rtol=1e-7)["E_cr"])
        d = np.abs(np.diff(Es))
        orders[Y] = np.log2(d[:-1] / d[1:])
    print("observed orders: chain", orders[False], "Y", orders[True])
    assert np.all(np.abs(orders[False] - 2) < 0.1)
    assert np.all(np.abs(orders[True][-1] - 1) < 0.2)


def test_tomato_critical_modulus_converges_with_refinement(tomato):
    Es = {}
    for f in (1, 2, 4):
        r = refine_asset(tomato, factor=f)
        Es[f] = critical_modulus(TreeSim(r, [r.physics.copy(stem_youngs_modulus=1e9)]), rtol=1e-4)["E_cr"]
    print("tomato E_cr vs refinement:", Es)
    assert abs(Es[1] / 7.316e8 - 1) < 1e-3                                      # B2 value, unrefined
    assert Es[1] < Es[2] < Es[4] < 1e9                                          # monotone, bounded
    assert (Es[4] - Es[2]) < 0.6 * (Es[2] - Es[1])                              # converging (~O(1/m))


# ---------------------------------------------------------------- 2. Delassus operator, PGS backends
@pytest.mark.parametrize("floating", [False, True])
def test_delassus_matches_bruteforce_columns(floating):
    """W = J M~^-1 J^T from the O(k depth + k^2) path formula equals 3k brute-force O(n)
    responses of the stored factorisation (incl. the linearised-implicit gyro terms)."""
    a = cantilever_asset(20, 0.5, 0.003, E=1e9)
    if floating:
        a.stem.fixed[:] = False
    rng = np.random.default_rng(1)
    s = TreeSim(a, [a.physics.copy(damping=1.0), a.physics.copy(damping=2.0, stem_youngs_modulus=3e8)])
    s.reset(Q=quat_from_rotvec(rng.normal(size=(2, s.topo.nb, 3)) * 0.15), u=rng.normal(size=s.u.shape) * 0.5,
            v_float=[0.1, 0, -0.2] if floating else None)
    st0 = (s.Q.copy(), s.u.copy(), s.p_float.copy(), s.v_float.copy())
    x = s.state.x
    cols = [GroundPlane(float(np.percentile(x[..., 2], 30)), mu=0.3),
            SphereObstacle(center=x[0, 12] + [0, 0, 0.004], radius=0.01, velocity=[0, 0, -0.1])]
    s.step(1 / 60, external=cols)
    c = s.contact
    assert (c.count >= 5).all()
    s.Q, s.u, s.p_float, s.v_float = st0
    s._kin_cache = None
    D = np.stack([c.normal, *_tangents(np.where(c.valid[..., None], c.normal, [0, 0, 1.0]))], -2)
    Wb = delassus_bruteforce(s, s._factor, c.body, c.point, D, c.valid)
    assert np.abs(Wb - c.W).max() < 1e-12 * np.abs(Wb).max()


def test_warp_pgs_matches_numpy():
    rng = np.random.default_rng(3)
    B, K = 3, 7
    J = rng.normal(size=(B, 3 * K, 12))
    W = J @ np.swapaxes(J, 1, 2) + 1e-3 * np.eye(3 * K)
    W[:, 0, 4] += 0.05                                                          # slightly non-symmetric
    vf = rng.normal(size=(B, 3 * K))
    tgt = rng.normal(size=(B, K)) * 0.1
    mu = rng.uniform(0.2, 0.8, size=(B, K))
    valid = rng.uniform(size=(B, K)) > 0.2
    l1, _, _ = pgs(W, vf, tgt, mu, valid, None, 300, 1e-14, 1.0, "numpy")
    l2, _, _ = pgs(W, vf, tgt, mu, valid, None, 300, 1e-14, 1.0, "warp")
    np.testing.assert_allclose(l1, l2, atol=1e-12 * np.abs(l1).max())
    v = (vf + np.einsum("bij,bj->bi", W, l1.reshape(B, -1))).reshape(B, K, 3)
    vn = v[..., 0]
    assert (vn[valid] >= tgt[valid] - 1e-8).all() and (l1[..., 0] >= 0).all()          # unilateral
    lt = np.linalg.norm(l1[..., 1:], axis=-1)
    assert (lt <= mu * l1[..., 0] + 1e-12).all()                                        # inside the cone
    slip = valid & (lt > mu * l1[..., 0] * (1 - 1e-9)) & (l1[..., 0] > 1e-9)
    cosang = (l1[..., 1:] * v[..., 1:]).sum(-1) / np.maximum(lt * np.linalg.norm(v[..., 1:], axis=-1), 1e-300)
    assert (cosang[slip] < -0.999).all()                                                # opposes slip
    assert (np.linalg.norm(v[..., 1:], axis=-1)[valid & ~slip & (l1[..., 0] > 1e-9)] < 1e-8).all()  # sticks


# ---------------------------------------------------------------- 3. acceptance: ground, sphere, friction
def test_stiff_chain_dropped_on_ground_rests_with_correct_reaction():
    """Free (floating-base) stiff chain dropped 4.5 cm onto the ground at h = 1/120: rests with
    no penetration > 0.1 mm and a support reaction equal to its weight."""
    a = floating_chain()
    s = TreeSim(a, a.physics.copy(damping=0.5), colliders=[GroundPlane(0.0, mu=0.5)])
    worst = 0.0
    for _ in range(240):
        s.step(1 / 120)
        worst = min(worst, float(min_gap_ground(s)[0]))
    W = s.m.sum() * 9.81
    F = s.contact.total_force()[0]
    vn, _ = s.point_speeds()
    print(f"chain: worst gap during drop {worst * 1e3:.4f} mm, rest gap {min_gap_ground(s)[0]:.2e} m, "
          f"reaction / weight - 1 = {F[2] / W - 1:.2e}")
    assert min_gap_ground(s)[0] > -1e-4 and worst > -1e-4
    assert abs(F[2] / W - 1) < 1e-3 and np.abs(F[:2]).max() < 1e-3 * W
    assert vn[0] < 1e-4


@pytest.mark.parametrize("mode", ["implicit", "penalty"])
def test_cantilever_pressed_by_kinematic_sphere(mode):
    """Rigid sphere pressed 4 mm onto a cantilever tip: implicit (hard) contact gives deflection
    = push and a contact force = analytic beam reaction (3EI/L^3 per discrete model) within 2%;
    the penalty baseline balances too but penetrates (force = k * penetration)."""
    n, L, r, E = 20, 0.5, 0.003, 1e9
    _, I = section(r)
    a = cantilever_asset(n, L, r, E=E)
    s = TreeSim(a, a.physics.copy(damping=20.0), gravity=ZERO_G)
    Rs, push = 0.02, 0.004
    sph = SphereObstacle(center=np.array([L, 0, 1.0 + Rs + r - push]), radius=Rs, mu=0.0, mode=mode, stiffness=5e3)
    assert s.settle(h=1 / 60, max_time=5.0, tol=1e-7, external=[sph])["converged"].all()
    d = float(1.0 - s.state.x[0, -1, 2])
    k_beam = 3 * E * I / L ** 3 / (1 + 1 / (2 * n * n))
    if mode == "implicit":
        F = -s.contact.total_force()[0][2]
        print(f"implicit: deflection/push = {d / push:.5f}, F / (k_beam d) - 1 = {F / (k_beam * d) - 1:.2e}")
        assert abs(d / push - 1) < 0.01
        assert abs(F / (k_beam * d) - 1) < 0.02
        assert abs(F / (k_beam * push) - 1) < 0.02                                    # vs the imposed push
    else:
        from heliades.sim.contact import detect
        c = detect(s, s._kin(), [sph], 0.0, ContactParams(margin=0.0, spec_factor=0.0, max_per_body=1))
        pen = float(-c.phi.min())
        print(f"penalty: penetration {pen * 1e6:.2f} um, k pen / (k_beam d) - 1 = {sph.stiffness * pen / (k_beam * d) - 1:.2e}")
        assert pen > 0 and abs(sph.stiffness * pen / (k_beam * d) - 1) < 0.02


@pytest.mark.parametrize("ratio", [0.98, 1.02])
def test_block_on_incline_sticks_or_slips_at_mu(ratio):
    """Block (rigid plate, 4 corner contacts) on an incline with tan(phi) = ratio * mu: sticks
    below mu, slips above with a = g (sin phi - mu cos phi). Isotropic Coulomb cone."""
    mu = 0.5
    phi = np.arctan(mu * ratio)
    g = 9.81 * np.array([np.sin(phi), 0, -np.cos(phi)])
    a = block_asset()
    s = TreeSim(a, a.physics, gravity=g, colliders=[GroundPlane(0.0, mu=mu)])
    xs = np.array(s.run(2.0, 1 / 60, record=lambda q: q.state.x[0, 0].copy()))
    t = np.arange(len(xs)) / 60
    disp = xs[-1, 0] - xs[0, 0]
    c = s.contact
    F = c.total_force()[0]
    print(f"ratio {ratio}: displacement {disp:.3e} m, |Ft|/Fn = {np.linalg.norm(F[:2]) / F[2]:.5f}")
    assert c.count[0] == 4 and min_gap_ground(s)[0] > -1e-6
    if ratio < 1:
        assert abs(disp) < 1e-6
        assert abs(np.linalg.norm(F[:2]) / F[2] - mu * ratio) < 1e-6
    else:
        acc = 2 * np.polyfit(t[30:], xs[30:, 0], 2)[0]
        an = 9.81 * (np.sin(phi) - mu * np.cos(phi))
        print(f"   slip acceleration {acc:.6f} vs analytic {an:.6f}")
        assert abs(acc / an - 1) < 0.02
        assert abs(np.linalg.norm(F[:2]) / F[2] - mu) < 1e-6


# ---------------------------------------------------------------- 4. tomato
def test_tomato_below_critical_falls_and_rests_on_ground(tomato):
    """E = 1e8 (< E_cr): the plant folds over onto the ground at h = 1/60 and comes to rest,
    no explosion and no penetration > 0.1 mm once landed."""
    s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=1e8)], colliders=[GroundPlane(0.0, mu=0.5)])
    H = tomato.stem.nodes[:, 2].max()
    gaps, speeds = [], []
    for i in range(720):                                                        # 12 s
        s.step(1 / 60, damping=5.0)
        if i % 10 == 9:
            gaps.append(float(min_gap_ground(s)[0]))
            speeds.append(float(s.point_speeds()[1][0]))
    x = points(s)[0]
    print(f"tomato on ground: worst gap {min(gaps) * 1e3:.2f} mm (impact), last 4 s {min(gaps[-24:]) * 1e3:.4f} mm, "
          f"final max speed {speeds[-1] * 1e3:.1f} mm/s, stem top z {s.state.x[0, tomato.stem.organ == 0, 2].max():.3f} m, "
          f"contacts {s.contact.count[0]}")
    assert np.isfinite(x).all() and np.abs(x).max() < 2 * H
    assert min(gaps[-24:]) > -1e-4                                              # at rest: no penetration
    assert min(gaps) > -5e-3                                                    # impact transient bounded
    assert speeds[-1] < 0.02 and speeds[-1] < 0.1 * max(speeds)                 # comes to rest
    assert s.state.x[0, tomato.stem.organ == 0, 2].max() < 0.7 * H              # it really fell


def test_finger_pushes_leaf_and_plant_recovers_target(tomato):
    """E = 1e9 plant at its inverse-statics rest (stable); a capsule finger sweeps a leaf aside
    by > 3 cm, holds, retracts; the plant returns to the target (< 1 mm)."""
    s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=1e9)], colliders=[GroundPlane(0.0, mu=0.5)])
    res = inverse_statics(s)
    assert res.status == ["stable"]
    apply_rest(s, res)
    p0 = points(s)[0]
    finger, li = finger_for_leaf(tomato)
    dmax, Fmax = 0.0, 0.0
    for _ in range(int(4.0 * 60)):
        s.step(1 / 60, external=[finger(s.t)], damping=5.0)
        if s.contact is not None and s.t < finger.released:
            Fmax = max(Fmax, float(np.linalg.norm(s.contact.total_force(collider=0)[0])))   # external first
        dmax = max(dmax, float(np.linalg.norm(points(s)[0] - p0, axis=-1).max()))
    drift = float(np.linalg.norm(points(s)[0] - p0, axis=-1).max())
    print(f"finger: leaf {li}, max displacement {dmax * 1e3:.1f} mm, max finger force {Fmax * 1e3:.1f} mN, "
          f"drift after release {drift * 1e3:.4f} mm")
    assert dmax > 0.03 and Fmax > 1e-3
    assert drift < 1e-3


def test_stake_holds_soft_plant_up(tomato):
    """E = 1e8 plant leaning on a rough stake (mu = 1) stays upright; on a smooth one (mu = 0.4)
    the stem slides off sideways; without a stake it folds."""
    probe = TreeSim(tomato, [tomato.physics])
    tops = {}
    for name, cols in (("none", []), ("rough", [stake_in_front(probe, mu=1.0)]), ("smooth", [stake_in_front(probe, mu=0.4)])):
        s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=1e8)], colliders=[GroundPlane(0.0, mu=0.5)] + cols)
        s.run(4.0, 1 / 60, damping=5.0, record=lambda q: 0)
        tops[name] = float(s.state.x[0, tomato.stem.organ == 0, 2].max())
        if name == "rough":
            Fs = s.contact.total_force(collider=1)[0]
            assert np.linalg.norm(Fs) > 1e-2
            assert min_gap_ground(s)[0] > -1e-4
    print("stem top z:", tops)
    H = tomato.stem.nodes[:, 2].max()
    assert tops["rough"] > 0.9 * H and tops["none"] < 0.65 * H and tops["smooth"] < 0.65 * H


# ---------------------------------------------------------------- 5. batch
def test_batch_equals_single_with_contacts(tomato):
    """B = 3 parameter sets (E, rho, rho_leaf, nu differ), each at its inverse rest, finger push
    + ground, vs 3 single runs: same contacts, same trajectories (round-off)."""
    P = [tomato.physics.copy(stem_youngs_modulus=1e9, stem_density=900.0),
         tomato.physics.copy(stem_youngs_modulus=8e8, leaf_areal_density=0.3),
         tomato.physics.copy(stem_youngs_modulus=1.2e9, stem_poisson=0.3)]
    finger, _ = finger_for_leaf(tomato)

    def run(p):
        s = TreeSim(tomato, p, colliders=[GroundPlane(0.0, mu=0.5)])
        apply_rest(s, inverse_statics(s))
        return s.run(0.7, 1 / 60, external=lambda t: [finger(t)], damping=5.0, record=lambda q: q.state.x), s
    xb, sb = run(P)
    for b, p in enumerate(P):
        x1, s1 = run([p])
        assert s1.contact.count[0] == sb.contact.count[b] > 0
        np.testing.assert_allclose(xb[:, b], x1[:, 0], rtol=0, atol=1e-12)


def test_collider_spawned_inside_canopy_does_not_explode(tomato):
    """Robustness: a sphere placed 4 cm deep inside the canopy (inconsistent penetration
    constraints: leaf vertices on all sides of it) at B = 3. Deep penetrations are only kept
    from growing and the impulses are capped, so the step stays bounded."""
    P = [tomato.physics.copy(stem_youngs_modulus=E) for E in (1e8, 2e8, 1.5e8)]
    sph = SphereObstacle(center=[0.15, 0.0, 0.5], radius=0.05)
    s = TreeSim(tomato, P, colliders=[GroundPlane(0.0, mu=0.5), sph])
    s.run(1.0, 1 / 60, damping=5.0, record=lambda q: 0)
    vn, vp = s.point_speeds()
    print("max speeds after 1 s", vp)
    assert np.isfinite(s.state.x).all() and (vp < 5.0).all()


def test_batched_contact_runs_at_B16(tomato):
    s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=1e8)] * 16, colliders=[GroundPlane(0.0, mu=0.5)])
    s.run(1.0, 1 / 60, damping=5.0, record=lambda q: 0)
    assert (s.contact.count > 0).all() and np.isfinite(s.state.x).all()
    assert np.abs(s.state.x - s.state.x[:1]).max() < 1e-9                       # identical members


# ---------------------------------------------------------------- 6. contact-aware inverse statics
def test_supported_inverse_column_below_critical():
    """Tilted column at 0.5 E_cr (Greenhill), tip leaning on a stake. B2 alone: unstable
    (falls when kicked). Supported inverse: support force + rest shape, stable under the
    contact constraint, and the forward sim with the real stake holds it."""
    L, r, rho, n = 1.0, 0.005, 1000.0, 20
    A, I = section(r)
    Eg = rho * A * 9.81 * L ** 3 / (7.837 * I)
    tilt = np.deg2rad(3.0)
    a = cantilever_asset(n, L, r, rho=rho, z=0.0, direction=(np.sin(tilt), 0, np.cos(tilt)))
    tip = a.stem.nodes[-1]
    Rs = 0.01
    stk = CapsuleObstacle(np.array([tip[0] + r + Rs, 0, 0.5]), np.array([tip[0] + r + Rs, 0, 1.2]), Rs, None, 0.5)
    sim = TreeSim(a, [a.physics.copy(stem_youngs_modulus=0.5 * Eg)])
    b2 = inverse_statics(sim)
    sup = supported_inverse(sim, [stk], tol=1e-4)
    print("B2:", b2.status, b2.n_negative, "supported:", sup.status, "force", sup.force[sup.active],
          "prestress ratio", sup.prestress / sup.prestress_free)
    assert b2.status == ["unstable"] and sup.status == "stable"
    assert sup.n_negative_free == 2 and sup.n_negative_supported == 0
    assert sup.active.sum() == 1 and sup.force[sup.active][0][0] < 0           # stake pushes back
    drift = {}
    for name, res, cols in (("free", b2, []), ("staked", sup.result, [stk])):
        s = TreeSim(a, [a.physics.copy(stem_youngs_modulus=0.5 * Eg)], colliders=cols)
        apply_rest(s, res)
        x0 = s.state.x.copy()
        s.u = np.random.default_rng(0).normal(size=s.u.shape) * 1e-3
        s._kin_cache = None
        s.run(3.0, 1 / 60, damping=1.0, record=lambda q: 0)
        drift[name] = float(np.abs(s.state.x - x0).max())
        if name == "staked":
            Fn = -s.contact.total_force()[0][0]
            assert abs(Fn / -sup.force[sup.active][0][0] - 1) < 0.02            # normal reaction as designed
    print("kicked drift", drift)
    assert drift["free"] > 0.05 and drift["staked"] < 1e-4


def test_inverse_with_known_support_forces_tomato(tomato):
    """B2 inverse with known external support forces: E = 1e9 tomato pressed by a stake into its
    main stem; the measured stake reactions as dead BodyForce loads -> the inverse recovers the
    asset rest shape exactly (round trip)."""
    from heliades.sim.contact import main_axis, stake
    probe = TreeSim(tomato, [tomato.physics])
    T, ax = probe.topo, main_axis(probe)
    bm = np.flatnonzero(ax)[np.argmin(np.abs(T.X0[T.node_b[ax], 2] - 0.45))]
    pos = T.X0[T.node_b[bm], :2] + np.array([0.95593315, -0.29358444]) * (T.r[bm] + 0.006 - 0.004)
    stk = stake(pos[0], pos[1], 0.006, 1.0, mu=0.5, bodies=ax)
    s = TreeSim(tomato, [tomato.physics.copy(stem_youngs_modulus=1e9)], colliders=[stk])
    s.settle(h=1 / 60, max_time=4.0, tol=1e-5, damping=5.0)
    c, k = s.contact, s._kin()
    forces = [BodyForce(int(c.body[0, j]), k.R[0, c.body[0, j]].T @ (c.point[0, j] - k.o[0, c.body[0, j]]),
                        c.force[0, j].copy()) for j in np.flatnonzero(c.valid[0] & (c.lam[0, :, 0] > 0))]
    Ftot = sum(f.force for f in forces)
    assert np.linalg.norm(Ftot) > 0.1
    s.colliders = []
    assert newton_static(s, external=forces)["converged"].all()
    res = inverse_statics(s, target_from_state(s), forces)
    err = np.abs(res.R0rel - T.R0rel[None]).max()
    print(f"known support force {Ftot} N, rest recovered to {err:.1e}")
    assert err < 1e-8 and res.status == ["stable"]


# ---------------------------------------------------------------- 7. GPU contact slot (B4)
def _cuda():
    try:
        import warp as wp
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda(), reason="needs a CUDA device")
def test_gpu_contact_slot_matches_cpu(tomato):
    """The same staged contact pipeline through B4's GPU slot (free motion, 3K factor-response
    columns, PGS, slot_apply) reproduces the CPU contact: finger push (E = 1e9, inverse rest)
    and the plant lying on the ground (E = 1e8). fp64: round-off; fp32: fp32 tolerance."""
    from heliades.sim.gpu import TreeSimGPU
    from heliades.sim.gpu.contact import ContactSlotSolver
    H = 1 / 60
    P = [tomato.physics.copy(stem_youngs_modulus=1e9)]
    finger, _ = finger_for_leaf(tomato)
    res = inverse_statics(TreeSim(tomato, P))
    for prec, tol in (("float64", 1e-10), ("float32", 1e-4)):
        c = TreeSim(tomato, P, colliders=[GroundPlane(0.0, mu=0.5)])
        apply_rest(c, res)
        g = TreeSimGPU(tomato, P, precision=prec, use_graph=False)
        g.set_rest(res.R0rel)
        g.reset(Q=res.Q)
        sol = ContactSlotSolver(g, [])
        g.contact = sol
        dmax, touched = 0.0, False
        for _ in range(54):
            fz = finger(c.t)
            c.step(H, external=[fz], damping=5.0)
            sol.colliders = [fz, GroundPlane(0.0, mu=0.5)]
            g.step(H, damping=5.0)
            touched |= sol.result is not None
            dmax = max(dmax, float(np.abs(g.state.x - c.state.x).max()))
        print(f"finger {prec}: max |x_gpu - x_cpu| = {dmax:.2e} m")
        assert touched and dmax < tol
    P = [tomato.physics.copy(stem_youngs_modulus=1e8)]
    base = TreeSim(tomato, P, colliders=[GroundPlane(0.0, mu=0.5)])
    base.run(3.0, H, damping=5.0, record=lambda q: 0)
    c = TreeSim(tomato, P, colliders=[GroundPlane(0.0, mu=0.5)])
    c.reset(Q=base.Q, u=base.u)
    g = TreeSimGPU(tomato, P, precision="float32", use_graph=False)
    g.reset(Q=base.Q, u=base.u)
    g.contact = sol = ContactSlotSolver(g, [GroundPlane(0.0, mu=0.5)])
    dmax = 0.0
    for _ in range(15):
        c.step(H, damping=5.0)
        g.step(H, damping=5.0)
        dmax = max(dmax, float(np.abs(g.state.x - c.state.x).max()))
    Fc, Fg = c.contact.total_force()[0], sol.total_force()[0]
    print(f"ground fp32: max |dx| = {dmax:.2e} m, reaction cpu {Fc} gpu {Fg}")
    assert sol.result["stage"].valid.sum() == c.contact.count[0] > 20
    assert dmax < 1e-4 and np.abs(Fg - Fc).max() < 1e-3 * np.abs(Fc).max()
