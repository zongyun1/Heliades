"""Render the Helios tomato settling in the B1 tree direct solver (heliades.sim.TreeSim).

  outputs/sim/tomato_settle_tree.mp4   three stem moduli stepped as one batch (B = 3):
                                       E = 1e8 (buckles), 3e8, 1e9 Pa
  outputs/sim/tomato_tree_vs_xpbd.mp4  side by side at E = 1e9 Pa, same damping:
                                       TreeSim h = 1/60  vs  planttwin XPBD 128 substeps/frame
  outputs/sim/summary.json             sag / settle numbers for the notes

Run:  .venv/bin/python scripts/sim_tree_demo.py [tree] [xpbd]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))  # planttwin
from planttwin import render as R  # noqa: E402

OUT = ROOT / "outputs" / "sim"
FPS = 30
DURATION = 3.0
H = 1 / 60
DAMP = 5.0


def _fig(ncols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.8 * ncols, 5.4), dpi=100)
    axes = [fig.add_axes([i / ncols, 0, 1 / ncols, 0.92], projection="3d") for i in range(ncols)]
    return fig, axes


def draw_tree_state(ax, asset, x, leaf_tris):
    """x: (N, 3) rod nodes; leaf_tris: (T, 3, 3)."""
    st = asset.stem
    E = st.edges
    organ = st.organ if st.organ is not None else np.zeros(len(st.nodes), int)
    widths = np.clip(st.radius[E[:, 1]] * 1e3 * 0.8, 0.5, 4)
    for o in np.unique(organ[E[:, 1]]):
        k = organ[E[:, 1]] == o
        R.draw_polylines(ax, x[E][k], widths[k], R.ORGAN_COLOR[int(o)])
    if len(leaf_tris):
        R.draw_tris(ax, leaf_tris, R.LEAF, R.LEAF_EDGE, 0.2, 0.95)
    for f in asset.fruits:
        ax.scatter(*x[f.node], s=(f.radius * 1e3) ** 2 * 0.6, color=R.FRUIT, depthshade=False)


def _leaf_tris(sim, b):
    return np.concatenate([V[b][L.faces] for V, L in zip(sim.leaf_vertices(), sim.topo.leaves)])


def tree():
    from heliades.sim import TreeSim
    from planttwin.helios_import import build_helios_plant

    asset, _ = build_helios_plant("tomato", 75, seed=0)
    Es = [1e8, 3e8, 1e9]
    sim = TreeSim(asset, [asset.physics.copy(stem_youngs_modulus=E) for E in Es])
    x0 = sim.state.x[0]
    Hgt = x0[:, 2].max()
    stem = asset.stem.organ == 0
    fig, axes = _fig(3)
    vw = R.VideoWriter(OUT / "tomato_settle_tree.mp4", fig, FPS)
    every = int(round(1 / (FPS * H)))
    speeds, t_sim = [], 0.0
    for f in range(int(DURATION * FPS) + 1):
        if f:
            t0 = time.perf_counter()
            for _ in range(every):
                sim.step(H, damping=DAMP)
            t_sim += time.perf_counter() - t0
        st = sim.state
        vmax = np.linalg.norm(st.v_nodes, axis=-1).max(1)
        speeds.append(vmax)
        for b, ax in enumerate(axes):
            R.setup_axes(ax, (0, 0), 0.45, azim=-70)
            R.draw_ground(ax, (0, 0), 0.45)
            draw_tree_state(ax, asset, st.x[b], _leaf_tris(sim, b))
            sag = (x0[stem, 2].max() - st.x[b, stem, 2].max()) / Hgt * 100
            ax.text2D(0.03, 0.97, f"E = {Es[b]:.0e} Pa", transform=ax.transAxes, fontsize=12)
            ax.text2D(0.03, 0.92, f"stem-top sag {sag:5.1f}%   max |v| {vmax[b] * 1e3:7.1f} mm/s",
                      transform=ax.transAxes, fontsize=9)
        fig.suptitle(f"Helios tomato 75 d, heliades.sim.TreeSim (O(n) implicit tree solver, "
                     f"h = 1/60, B = 3 in lock-step)   t = {sim.t:4.2f} s", fontsize=11)
        vw.frame()
    vw.close()
    st = sim.state
    summary = {
        "E": Es, "h": H, "damping": DAMP, "duration": DURATION,
        "stem_top_sag_pct": ((x0[stem, 2].max() - st.x[:, stem, 2].max(1)) / Hgt * 100).tolist(),
        "max_disp_m": np.linalg.norm(st.x - x0, axis=-1).max(1).tolist(),
        "final_max_node_speed": speeds[-1].tolist(),
        "sim_wall_s": t_sim, "steps": int(DURATION / H),
    }
    return summary


def xpbd():
    from heliades.sim import TreeSim
    from planttwin.helios_import import build_helios_plant
    from planttwin.sim import PlantModel

    asset, _ = build_helios_plant("tomato", 75, seed=0)
    phys = asset.physics.copy(stem_youngs_modulus=1e9, damping=DAMP)
    t0 = time.perf_counter()
    model = PlantModel(asset)
    model.set_params(phys)
    X, _, _ = model.simulate(DURATION, fps=FPS, substeps=128)
    t_xpbd = time.perf_counter() - t0
    sim = TreeSim(asset, [phys])
    N = len(asset.stem.nodes)
    stem = asset.stem.organ == 0
    x0 = asset.stem.nodes
    Hgt = x0[:, 2].max()
    fig, axes = _fig(2)
    vw = R.VideoWriter(OUT / "tomato_tree_vs_xpbd.mp4", fig, FPS)
    every = int(round(1 / (FPS * H)))
    t_tree = 0.0
    rows = []
    for f in range(len(X)):
        if f:
            t1 = time.perf_counter()
            for _ in range(every):
                sim.step(H)
            t_tree += time.perf_counter() - t1
        st = sim.state
        xs = X[f, 0]
        v_t = np.linalg.norm(st.v_nodes[0], axis=-1).max()
        v_x = np.linalg.norm(X[f, 0, :N] - X[max(f - 1, 0), 0, :N], axis=-1).max() * FPS
        sag_t = (x0[stem, 2].max() - st.x[0, stem, 2].max()) / Hgt * 100
        sag_x = (x0[stem, 2].max() - xs[:N][stem, 2].max()) / Hgt * 100
        rows.append((f / FPS, sag_t, v_t, sag_x, v_x))
        for ax in axes:
            R.setup_axes(ax, (0, 0), 0.45, azim=-70)
            R.draw_ground(ax, (0, 0), 0.45)
        draw_tree_state(axes[0], asset, st.x[0], _leaf_tris(sim, 0))
        R.draw_asset_state(axes[1], model, xs)
        axes[0].text2D(0.03, 0.97, "TreeSim (B1), h = 1/60", transform=axes[0].transAxes, fontsize=12)
        axes[1].text2D(0.03, 0.97, "planttwin XPBD, 128 substeps/frame", transform=axes[1].transAxes,
                       fontsize=12)
        axes[0].text2D(0.03, 0.92, f"stem-top sag {sag_t:5.1f}%  max |v| {v_t * 1e3:7.1f} mm/s",
                       transform=axes[0].transAxes, fontsize=9)
        axes[1].text2D(0.03, 0.92, f"stem-top sag {sag_x:5.1f}%  max |v| {v_x * 1e3:7.1f} mm/s",
                       transform=axes[1].transAxes, fontsize=9)
        fig.suptitle(f"Helios tomato 75 d, E = 1e9 Pa, damping {DAMP}/s   t = {f / FPS:4.2f} s", fontsize=11)
        vw.frame()
    vw.close()
    return {"E": 1e9, "rows(t, sag_tree%, vmax_tree, sag_xpbd%, vmax_xpbd)": rows,
            "wall_tree_s": t_tree, "wall_xpbd_s": t_xpbd}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["tree", "xpbd"]
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name in which:
        t = time.time()
        summary[name] = globals()[name]()
        print(f"{name}: done in {time.time() - t:.0f}s", flush=True)
        path.write_text(json.dumps(summary, indent=1))
    print("wrote", sorted(p.name for p in OUT.glob("*")))
