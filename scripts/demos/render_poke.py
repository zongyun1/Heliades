"""Render the plant asset and a poke: rest shape, gravity equilibrium, pressed, ringing.

Run:  python scripts/demos/render_poke.py   ->  outputs/poke_frames.png
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "legacy"))  # planttwin
from planttwin.procedural import make_plant  # noqa: E402
from planttwin.real2sim import PokeExperiment  # noqa: E402
from planttwin.sim import PlantModel  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "outputs"


def draw(ax, model, x, sphere=None, title=""):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for a, b in model.asset.stem.edges:
        ax.plot(*x[[a, b]].T, color="#6b4f2a", lw=2.2)
    for leaf, idx in zip(model.asset.leaves, model.leaf_idx):
        tri = x[idx[leaf.faces]]
        ax.add_collection3d(Poly3DCollection(tri, facecolor="#5aa65a", edgecolor="#2f6b2f",
                                             linewidths=0.3, alpha=0.9))
    if sphere is not None:
        c, r = sphere
        u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:8j]
        ax.plot_surface(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v),
                        c[2] + r * np.cos(v), color="#d9534f", alpha=0.8)
    ax.set_xlim(-0.15, 0.15); ax.set_ylim(-0.15, 0.15); ax.set_zlim(0, 0.3)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(12, -80); ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    asset = make_plant(seed=0)
    exp = PokeExperiment()
    model = PlantModel(asset)
    xe, qe = model.settle(exp.settle_time, substeps=exp.substeps)
    press = model.leaf_idx[exp.leaf][asset.leaves[exp.leaf].keypoints["mid"]]
    (sph,) = exp.colliders(xe[:, press])
    X, _, _ = model.simulate(1.0, fps=exp.fps, substeps=exp.substeps, colliders=(sph,), x0=xe, q_init=qe)
    f_press = int(exp.t_press * exp.fps)
    shots = [
        (model.x_rest, None, "asset rest shape"),
        (xe[0], None, "gravity equilibrium (the 'photo')"),
        (X[f_press, 0], (sph.center(exp.t_press)[0], sph.radius), f"pressed, t={exp.t_press:.2f}s"),
        (X[f_press + 9, 0], None, f"released, t={(f_press + 9) / exp.fps:.2f}s"),
    ]
    fig = plt.figure(figsize=(12, 3.6))
    for i, (x, s, title) in enumerate(shots):
        draw(fig.add_subplot(1, 4, i + 1, projection="3d"), model, x, s, title)
    fig.tight_layout()
    fig.savefig(OUT / "poke_frames.png", dpi=130)
    print("wrote", OUT / "poke_frames.png")


if __name__ == "__main__":
    main()
