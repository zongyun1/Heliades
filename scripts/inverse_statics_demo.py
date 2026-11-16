"""B2 demo: inverse statics on the Helios tomato (heliades.sim.inverse).

  outputs/sim/tomato_inverse_hold.mp4   three panels, 3 s at h = 1/60, damping 5/s:
      (a) forward only, E = 1e9: the Helios (already loaded) shape used as rest -> sags again
      (b) inverse statics, E = 1e9: rest shape from B2 -> holds the Helios shape (stable)
      (c) inverse statics, E = 5e8: exact equilibrium but flagged unstable; a 1e-3 rad/s
          kick makes it leave the target (below E_cr ~ 7.3e8 for this target)
      grey: the target (Helios) stem for reference
  outputs/sim/inverse_summary.json      numbers for migration/notes/B2.md

Run:  .venv/bin/python scripts/inverse_statics_demo.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planttwin import render as R  # noqa: E402

OUT = ROOT / "outputs" / "sim"
FPS, DURATION, H, DAMP = 30, 3.0, 1 / 60, 5.0


def draw(ax, asset, x, tris, ghost=None):
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


def tris_of(sim, b):
    return np.concatenate([V[b][L.faces] for V, L in zip(sim.leaf_vertices(), sim.topo.leaves)])


def points(sim):
    return np.concatenate([sim.state.x] + sim.leaf_vertices(), 1)


def timing(asset):
    from heliades.sim import TreeSim
    from heliades.sim.inverse import inverse_statics
    out = {}
    for B in (1, 64):
        for E in (5e8, 1e9):
            sim = TreeSim(asset, [asset.physics.copy(stem_youngs_modulus=E)] * B)
            inverse_statics(sim)
            ts, tn = [], []
            for _ in range(5):
                t = time.perf_counter()
                inverse_statics(sim)
                ts.append(time.perf_counter() - t)
                t = time.perf_counter()
                inverse_statics(sim, stability=False)
                tn.append(time.perf_counter() - t)
            out[f"B={B},E={E:g}"] = {"with_stability_s": min(ts), "inverse_only_s": min(tn)}
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from heliades.sim import TreeSim
    from heliades.sim.inverse import apply_rest, critical_modulus, inverse_statics
    from planttwin.helios_import import build_helios_plant

    asset, _ = build_helios_plant("tomato", 75, seed=1)
    phys = lambda E: asset.physics.copy(stem_youngs_modulus=E)
    fwd = TreeSim(asset, [phys(1e9)])
    inv = TreeSim(asset, [phys(1e9), phys(5e8)])
    t0 = time.perf_counter()
    res = inverse_statics(inv, min_eig=True)
    t_inv = time.perf_counter() - t0
    apply_rest(inv, res)
    inv.u[1] = np.random.default_rng(0).normal(size=inv.u[1].shape) * 1e-3   # kick the 5e8 one
    inv._kin_cache = None
    p0 = points(fwd)[0]
    target = asset.stem.nodes
    stem = asset.stem.organ == 0
    Hgt = target[:, 2].max()

    fig = plt.figure(figsize=(14.4, 5.6), dpi=100)
    axes = [fig.add_axes([i / 3, 0, 1 / 3, 0.88], projection="3d") for i in range(3)]
    vw = R.VideoWriter(OUT / "tomato_inverse_hold.mp4", fig, FPS)
    every = int(round(1 / (FPS * H)))
    rows = []
    titles = ["(a) forward only, E = 1e9 Pa\nHelios shape used as rest: gravity counted twice",
              "(b) inverse statics, E = 1e9 Pa\nB2 rest shape: holds the Helios shape (stable)",
              f"(c) inverse statics, E = 5e8 Pa\nexact equilibrium, flagged UNSTABLE "
              f"({res.n_negative[1]} neg. modes), kicked"]
    for f in range(int(DURATION * FPS) + 1):
        if f:
            for _ in range(every):
                fwd.step(H, damping=DAMP)
                inv.step(H, damping=DAMP)
        pf, pi = points(fwd)[0], points(inv)
        drift = [np.linalg.norm(pf - p0, axis=-1).max(), *np.linalg.norm(pi - p0, axis=-1).max(1)]
        xs = [fwd.state.x[0], inv.state.x[0], inv.state.x[1]]
        sag = [(target[stem, 2].max() - x[stem, 2].max()) / Hgt * 100 for x in xs]
        rows.append([f / FPS, *drift, *sag])
        tr = [tris_of(fwd, 0), tris_of(inv, 0), tris_of(inv, 1)]
        for i, ax in enumerate(axes):
            R.setup_axes(ax, (0, 0), 0.45, azim=-70)
            R.draw_ground(ax, (0, 0), 0.45)
            draw(ax, asset, xs[i], tr[i], ghost=target)
            ax.text2D(0.03, 0.93, titles[i], transform=ax.transAxes, fontsize=10, va="top")
            ax.text2D(0.03, 0.80, f"max drift from target {drift[i] * 1e3:8.2f} mm   stem-top sag {sag[i]:5.2f}%",
                      transform=ax.transAxes, fontsize=9)
        fig.suptitle(f"Helios tomato 75 d, heliades.sim B2 inverse statics (exact for the discrete model, O(n); "
                     f"grey = target stem)   t = {f / FPS:4.2f} s", fontsize=11)
        vw.frame()
    vw.close()

    crit = critical_modulus(TreeSim(asset, [phys(1e9)]), rtol=1e-4)
    summary = {
        "status_E1e9_E5e8": res.status, "n_negative": res.n_negative.tolist(),
        "margin_mu_min": res.margin.tolist(), "E_cr_linear": res.E_cr_linear.tolist(),
        "E_cr_exact": crit["E_cr"],
        "rows(t, drift_fwd, drift_inv1e9, drift_inv5e8, sag_fwd%, sag_inv1e9%, sag_inv5e8%)": rows,
        "final_drift_m": rows[-1][1:4],
        "timing": timing(asset),
    }
    (OUT / "inverse_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("rows")}, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
