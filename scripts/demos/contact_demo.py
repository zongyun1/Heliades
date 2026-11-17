"""B3 demo: contact and ground on the tree solver (heliades.sim.contact).

  outputs/sim/tomato_ground_rest.mp4   E = 1e8 tomato (below its critical E) folding over, 8 s:
      (a) no contact (B1): falls through the ground
      (b) penalty ground (implicit spring 2e4 N/m + damper, B1-style baseline)
      (c) implicit contact (Baraff Schur complement + PGS, Coulomb cone mu = 0.5)
  outputs/sim/tomato_finger_push.mp4   E = 1e9 tomato at its inverse-statics rest (stable); a
      capsule finger sweeps a leaflet aside, holds, retracts; the plant returns to the target.
      (a) whole plant, implicit  (b) close-up, implicit  (c) close-up, penalty finger (5e3 N/m)
  outputs/sim/contact_summary.json     numbers for migration/notes/B3.md (comparisons,
      performance at B = 1 / 16 / 64, refinement convergence of the critical modulus)

Run:  .venv/bin/python scripts/demos/contact_demo.py [--no-video]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))  # planttwin
from planttwin import render as R  # noqa: E402

OUT = ROOT / "outputs" / "sim"
FPS, H = 30, 1 / 60


# ---------------------------------------------------------------- drawing
def draw(ax, asset, x, tris, capsules=(), ghost=None):
    st = asset.stem
    E = st.edges
    organ = st.organ if st.organ is not None else np.zeros(len(st.nodes), int)
    widths = np.clip(st.radius[E[:, 1]] * 1e3 * 0.8, 0.5, 4)
    if ghost is not None:
        k = organ[E[:, 1]] == 0
        R.draw_polylines(ax, ghost[E][k], widths[k] * 0.6, "#9a9a9a")
    for o in np.unique(organ[E[:, 1]]):
        k = organ[E[:, 1]] == o
        R.draw_polylines(ax, x[E][k], widths[k], R.ORGAN_COLOR[int(o)])
    if len(tris):
        R.draw_tris(ax, tris, R.LEAF, R.LEAF_EDGE, 0.2, 0.95)
    ax.computed_zorder = not capsules
    for (a, b, r, col) in capsules:
        u, v = np.mgrid[0:2 * np.pi:14j, 0:1:2j]
        ax_ = (b - a) / np.linalg.norm(b - a)
        e1 = np.cross(ax_, [1.0, 0, 0] if abs(ax_[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax_, e1)
        P = a[:, None, None] + (b - a)[:, None, None] * v + r * (np.cos(u) * e1[:, None, None] + np.sin(u) * e2[:, None, None])
        ax.plot_surface(P[0], P[1], P[2], color=col, alpha=0.9, linewidth=0, zorder=10)


def tris_of(sim, b):
    return np.concatenate([V[b][L.faces] for V, L in zip(sim.leaf_vertices(), sim.topo.leaves)])


# ---------------------------------------------------------------- videos
def ground_video(asset, summary, video=True):
    from heliades.sim import TreeSim
    from heliades.sim.contact import GroundPlane
    from heliades.sim.scenarios import min_gap_ground
    phys = [asset.physics.copy(stem_youngs_modulus=1e8)]
    sims = [TreeSim(asset, phys), TreeSim(asset, phys, colliders=[GroundPlane(0.0, mu=0.5, mode="penalty")]),
            TreeSim(asset, phys, colliders=[GroundPlane(0.0, mu=0.5)])]
    names = ["(a) no contact (B1)", "(b) penalty ground 2e4 N/m", "(c) implicit contact + Coulomb (B3)"]
    if video:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14.4, 5.6), dpi=100)
        axes = [fig.add_axes([i / 3, 0, 1 / 3, 0.88], projection="3d") for i in range(3)]
        vw = R.VideoWriter(OUT / "tomato_ground_rest.mp4", fig, FPS)
    every = int(round(1 / (FPS * H)))
    rows, wall = [], np.zeros(3)
    duration = 8.0
    dead = [None] * 3                              # time of divergence (penalty at h = 1/60)
    for f in range(int(duration * FPS) + 1):
        if f:
            for i, s in enumerate(sims):
                if dead[i] is not None:
                    continue
                t0 = time.perf_counter()
                st = (s.Q.copy(), s.u.copy(), s.t)
                try:
                    for _ in range(every):
                        s.step(H, damping=5.0)
                    if not (np.isfinite(s.u).all() and np.abs(s.u).max() < 1e4):
                        raise FloatingPointError
                except (np.linalg.LinAlgError, FloatingPointError):
                    dead[i] = s.t
                    s.Q, s.u, s.t = st
                    s._kin_cache = None
                    s.contact = None
                wall[i] += time.perf_counter() - t0
        gaps = [float(min_gap_ground(s)[0]) for s in sims]
        rows.append([f / FPS, *gaps])
        if video:
            for i, (ax, s) in enumerate(zip(axes, sims)):
                R.setup_axes(ax, (0.15, 0), 0.5, zmin=-0.15, azim=-60)
                R.draw_ground(ax, (0.15, 0), 0.5)
                draw(ax, asset, s.state.x[0], tris_of(s, 0))
                k = s.contact.count[0] if s.contact is not None else 0
                ax.text2D(0.03, 0.93, names[i], transform=ax.transAxes, fontsize=10, va="top")
                ax.text2D(0.03, 0.85, f"deepest point {min(gaps[i], 0) * 1e3:7.2f} mm   contacts {k}",
                          transform=ax.transAxes, fontsize=9)
                if dead[i] is not None:
                    ax.text2D(0.03, 0.78, f"DIVERGED at t = {dead[i]:.2f} s (frozen)", transform=ax.transAxes,
                              fontsize=10, color="#c0392b")
            fig.suptitle(f"Helios tomato 75 d, E = 1e8 Pa (below E_cr), h = 1/60, damping 5/s   t = {f / FPS:4.2f} s",
                         fontsize=11)
            vw.frame()
    if video:
        vw.close()
    rows = np.array(rows)
    late = rows[:, 0] > duration - 2.0
    summary["ground_rest"] = {
        "names": names, "worst_gap_mm": (rows[:, 1:].min(0) * 1e3).tolist(),
        "gap_last_2s_mm": (rows[late, 1:].min(0) * 1e3).tolist(),
        "wall_s_for_8s": wall.tolist(), "diverged_at_s": dead,
        "final_max_speed_mm_s": [float(s.point_speeds()[1][0] * 1e3) for s in sims],
        "stem_top_z": [float(s.state.x[0, asset.stem.organ == 0, 2].max()) for s in sims]}
    return sims[2]


def finger_video(asset, summary, video=True):
    from heliades.sim import TreeSim
    from heliades.sim.contact import GroundPlane
    from heliades.sim.inverse import apply_rest, inverse_statics
    from heliades.sim.scenarios import finger_for_leaf, points
    phys = [asset.physics.copy(stem_youngs_modulus=1e9)]
    fi, li = finger_for_leaf(asset)
    fp, _ = finger_for_leaf(asset, mode="penalty")
    sims = [TreeSim(asset, phys, colliders=[GroundPlane(0.0, mu=0.5)]) for _ in range(2)]
    res = inverse_statics(sims[0])
    for s in sims:
        apply_rest(s, res)
    fingers = [fi, fp]
    p0 = points(sims[0])[0]
    target = asset.stem.nodes
    c0 = asset.leaves[li].vertices.mean(0)
    if video:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14.4, 5.6), dpi=100)
        axes = [fig.add_axes([i / 3, 0, 1 / 3, 0.88], projection="3d") for i in range(3)]
        vw = R.VideoWriter(OUT / "tomato_finger_push.mp4", fig, FPS)
    every = int(round(1 / (FPS * H)))
    duration = 4.0
    rec = {"t": [], "drift": [[], []], "force": [[], []], "pen": [[], []]}
    dead = [None, None]
    for f in range(int(duration * FPS) + 1):
        if f:
            for i, s in enumerate(sims):
                if dead[i] is not None:
                    s.t += every * H
                    continue
                st = (s.Q.copy(), s.u.copy(), s.t)
                try:
                    for _ in range(every):
                        s.step(H, external=[fingers[i](s.t)], damping=5.0)
                    if not (np.isfinite(s.u).all() and np.abs(s.u).max() < 1e4):
                        raise FloatingPointError
                except (np.linalg.LinAlgError, FloatingPointError):
                    dead[i] = s.t
                    s.Q, s.u, s.t = st[0], st[1], st[2] + every * H
                    s._kin_cache = None
        t = sims[0].t
        rec["t"].append(t)
        caps = []
        for i, s in enumerate(sims):
            cap = fingers[i](t)
            P = points(s)[0]
            # penetration of plant points into the finger capsule
            ab = cap.b - cap.a
            tt = np.clip(((P - cap.a) @ ab) / (ab @ ab), 0, 1)
            dist = np.linalg.norm(P - (cap.a + tt[:, None] * ab), axis=1) - cap.radius
            rec["pen"][i].append(float(max(-dist.min(), 0.0)))
            rec["drift"][i].append(float(np.linalg.norm(P - p0, axis=-1).max()))
            F = 0.0
            if s.contact is not None and i == 0:
                F = float(np.linalg.norm(s.contact.total_force(collider=0)[0]))
            rec["force"][i].append(F)
            caps.append((cap.a, cap.b, cap.radius, "#c0392b"))
        if video:
            views = [(sims[0], caps[0], (0.1, 0), 0.5, 0.0, 0), (sims[0], caps[0], c0[:2], 0.12, c0[2] - 0.12, 0),
                     (sims[1], caps[1], c0[:2], 0.12, c0[2] - 0.12, 1)]
            titles = ["(a) implicit contact, whole plant", "(b) implicit contact, close-up",
                      "(c) penalty finger 5e3 N/m, close-up"]
            for ax, (s, cap, ctr, half, zmin, i), ttl in zip(axes, views, titles):
                R.setup_axes(ax, ctr, half, zmin=zmin, azim=-60)
                if half > 0.3:
                    R.draw_ground(ax, ctr, half)
                draw(ax, asset, s.state.x[0], tris_of(s, 0), capsules=[cap] if t < fingers[i].released else [],
                     ghost=target)
                ax.text2D(0.03, 0.93, ttl, transform=ax.transAxes, fontsize=10, va="top")
                ax.text2D(0.03, 0.85, f"drift from target {rec['drift'][i][-1] * 1e3:7.2f} mm   "
                          f"finger penetration {rec['pen'][i][-1] * 1e3:5.2f} mm", transform=ax.transAxes, fontsize=9)
                if dead[i] is not None:
                    ax.text2D(0.03, 0.78, f"DIVERGED at t = {dead[i]:.2f} s (frozen)", transform=ax.transAxes,
                              fontsize=10, color="#c0392b")
            fig.suptitle(f"E = 1e9 Pa tomato at its inverse-statics rest; finger push and release   t = {t:4.2f} s",
                         fontsize=11)
            vw.frame()
    if video:
        vw.close()
    summary["finger_push"] = {
        "leaf": li, "max_drift_mm": [max(d) * 1e3 for d in rec["drift"]],
        "final_drift_mm": [d[-1] * 1e3 for d in rec["drift"]],
        "max_finger_penetration_mm": [max(p) * 1e3 for p in rec["pen"]],
        "max_force_implicit_mN": max(rec["force"][0]) * 1e3, "names": ["implicit", "penalty 5e3 N/m"],
        "diverged_at_s": dead}


# ---------------------------------------------------------------- numbers
def compare_chain(summary):
    from heliades.sim import TreeSim
    from heliades.sim.contact import GroundPlane
    from heliades.sim.scenarios import floating_chain, min_gap_ground
    out = {}
    a = floating_chain()
    for mode in ("implicit", "penalty"):
        for h in (1 / 60, 1 / 240):
            s = TreeSim(a, a.physics.copy(damping=0.5), colliders=[GroundPlane(0.0, mu=0.5, mode=mode)])
            worst = 0.0
            t0 = time.perf_counter()
            for _ in range(int(2.0 / h)):
                s.step(h)
                worst = min(worst, float(min_gap_ground(s)[0]))
            wall = time.perf_counter() - t0
            W = s.m.sum() * 9.81
            if mode == "implicit":
                Fz = float(s.contact.total_force()[0][2])
            else:
                Fz = float("nan")
            out[f"{mode},h=1/{round(1 / h)}"] = {"rest_gap_mm": float(min_gap_ground(s)[0] * 1e3),
                                                 "worst_gap_mm": worst * 1e3,
                                                 "reaction_over_weight": Fz / W,
                                                 "max_speed_mm_s": float(s.point_speeds()[0][0] * 1e3),
                                                 "wall_s": wall}
    summary["stiff_chain_on_ground"] = out


def compare_stake(asset, summary):
    from heliades.sim import TreeSim
    from heliades.sim.contact import GroundPlane, detect, ContactParams
    from heliades.sim.scenarios import stake_in_front
    probe = TreeSim(asset, [asset.physics])
    out = {}
    for name, st in (("none", None), ("implicit mu=1", stake_in_front(probe, mu=1.0)),
                     ("implicit mu=0.4", stake_in_front(probe, mu=0.4)),
                     ("penalty 5e3 N/m", stake_in_front(probe, mu=1.0, mode="penalty"))):
        cols = [GroundPlane(0.0, mu=0.5)] + ([st] if st is not None else [])
        s = TreeSim(asset, [asset.physics.copy(stem_youngs_modulus=1e8)], colliders=cols)
        t0 = time.perf_counter()
        diverged = None
        try:
            s.run(5.0, H, damping=5.0, record=lambda q: 0)
            if not np.isfinite(s.u).all():
                raise FloatingPointError
        except (np.linalg.LinAlgError, FloatingPointError):
            diverged = s.t
            out[name] = {"diverged_at_s": diverged}
            continue
        pen = 0.0
        if st is not None:
            c = detect(s, s._kin(), [st], 0.0, ContactParams(margin=0.0, spec_factor=0.0, max_per_body=1))
            pen = float(max(-c.phi.min(), 0.0)) if c is not None else 0.0
        F = None
        if st is not None and s.contact is not None and st.mode == "implicit":
            F = s.contact.total_force(collider=1)[0].tolist()
        out[name] = {"stem_top_z": float(s.state.x[0, asset.stem.organ == 0, 2].max()),
                     "stake_penetration_mm": pen * 1e3, "stake_force_N": F,
                     "max_speed_mm_s": float(s.point_speeds()[1][0] * 1e3), "wall_s": time.perf_counter() - t0}
    summary["stake_lean"] = out


def performance(rest_sim, summary):
    from heliades.sim import TreeSim
    from heliades.sim.contact import ContactParams, GroundPlane
    a = rest_sim.asset
    out = {}
    for backend in ("warp", "numpy"):
        for B in ((1, 16, 64) if backend == "warp" else (1,)):
            for contact in ((True, False) if backend == "warp" else (True,)):
                s = TreeSim(a, [a.physics.copy(stem_youngs_modulus=1e8)] * B,
                            colliders=[GroundPlane(0.0, mu=0.5)] if contact else [],
                            contact_params=ContactParams(backend=backend))
                s.reset(Q=np.broadcast_to(rest_sim.Q, (B,) + rest_sim.Q.shape[1:]),
                        u=np.broadcast_to(rest_sim.u, (B,) + rest_sim.u.shape[1:]))
                s.step(H, damping=5.0)
                n = 30 if B < 64 else 10
                tm = {"detect": 0, "delassus": 0, "pgs": 0, "backsub": 0}
                its = []
                t0 = time.perf_counter()
                for _ in range(n):
                    s.step(H, damping=5.0)
                    if s.contact is not None:
                        for kk in tm:
                            tm[kk] += s.contact.timing[kk] / n
                        its.append(s.contact.iterations)
                dt = (time.perf_counter() - t0) / n
                key = f"{backend},B={B},{'contact' if contact else 'no contact'}"
                out[key] = {"ms_per_step": dt * 1e3, "steps_per_s": 1 / dt, "env_steps_per_s": B / dt,
                            "contacts_per_env": int(s.contact.count.mean()) if s.contact is not None else 0,
                            "pgs_sweeps_mean": float(np.mean(its)) if its else 0,
                            "breakdown_ms": {k: v * 1e3 for k, v in tm.items()}}
                print(key, json.dumps(out[key]), flush=True)
    summary["performance"] = out


def refinement(asset, summary):
    from heliades.sim import TreeSim
    from heliades.sim.inverse import critical_modulus, inverse_statics
    from heliades.sim.refine import refine_asset
    rows = []
    for f in (1, 2, 3, 4, 6, 8):
        r = refine_asset(asset, factor=f)
        s = TreeSim(r, [r.physics.copy(stem_youngs_modulus=1e9)])
        t0 = time.perf_counter()
        c = critical_modulus(s, rtol=1e-4)
        res = inverse_statics(s)
        rows.append({"factor": f, "bodies": s.topo.nb, "levels": s.topo.nlev, "E_cr": c["E_cr"],
                     "max_rest_angle_rad_at_1e9": float(res.rest_angle.max()), "wall_s": time.perf_counter() - t0})
        print(rows[-1], flush=True)
    E = np.array([r["E_cr"] for r in rows])
    m = np.array([r["factor"] for r in rows], float)
    # Richardson with p = 1 on the two finest
    Einf = (m[-1] * E[-1] - m[-2] * E[-2]) / (m[-1] - m[-2])
    p_obs = np.log((E[1] - E[0]) / (E[3] - E[1])) / np.log(2)
    summary["refinement"] = {"rows": rows, "E_cr_extrapolated_p1": float(Einf), "observed_order_1_2_4": float(p_obs)}


def main():
    video = "--no-video" not in sys.argv
    import matplotlib
    matplotlib.use("Agg")
    from planttwin.helios_import import build_helios_plant
    asset, _ = build_helios_plant("tomato", 75, seed=1)
    summary = {}
    t0 = time.time()
    rest = ground_video(asset, summary, video)
    print("ground video", time.time() - t0, json.dumps(summary["ground_rest"]), flush=True)
    finger_video(asset, summary, video)
    print("finger video", time.time() - t0, json.dumps(summary["finger_push"]), flush=True)
    compare_chain(summary)
    print(json.dumps(summary["stiff_chain_on_ground"], indent=1), flush=True)
    compare_stake(asset, summary)
    print(json.dumps(summary["stake_lean"], indent=1), flush=True)
    performance(rest, summary)
    refinement(asset, summary)
    (OUT / "contact_summary.json").write_text(json.dumps(summary, indent=1))
    print("done", time.time() - t0)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
