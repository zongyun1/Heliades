"""Grow several crops in Helios, convert each to a PlantAsset, save JSON + a gallery figure.

Top row: Helios render meshes. Bottom row: the simulation asset (rod centrelines coloured
by organ, coarse leaf sim meshes, fruit spheres).

Run:  .venv/bin/python scripts/helios_gallery.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planttwin.helios_import import build_helios_plant  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "helios"
CROPS = [("tomato", 75), ("cherrytomato", 75), ("capsicum", 80), ("strawberry", 60),
         ("bean", 45), ("soybean", 50), ("grapevine_VSP", 120), ("maize", 50)]
ORGAN_COLOR = {0: "#6b4f2a", 1: "#7a8f3a", 2: "#3f7f3f", 3: "#3f7f3f"}


def setup(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(max(c[2] - r, 0), c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(10, -70); ax.set_axis_off()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(3.2 * len(CROPS), 7))
    summary = {}
    for col, (sp, age) in enumerate(CROPS):
        t0 = time.time()
        try:
            asset, render = build_helios_plant(sp, age, seed=0)
        except Exception as e:  # report and keep going
            summary[sp] = {"error": repr(e)}
            print(sp, "FAILED:", e)
            continue
        asset.save(OUT / f"{sp}.json")
        st = asset.stem
        L = np.linalg.norm(st.nodes[st.edges[:, 1]] - st.nodes[st.edges[:, 0]], axis=1)
        summary[sp] = {
            "age_days": age, "height_m": float(st.nodes[:, 2].max()), "stem_nodes": len(st.nodes),
            "organ_counts": np.bincount(st.organ, minlength=4).tolist(),
            "leaflets_helios": asset.meta["helios_leaflets"], "leaves_in_asset": len(asset.leaves),
            "fruits": len(asset.fruits), "fruit_mass_g": round(sum(f.mass for f in asset.fruits) * 1e3, 1),
            "sim_particles": int(len(st.nodes) + sum(len(l.vertices) - 1 for l in asset.leaves)),
            "min_segment_mm": float(L.min() * 1e3), "seconds": round(time.time() - t0, 2),
        }
        print(sp, summary[sp])

        allpts = np.vstack([st.nodes] + [V for V, _ in render["leaf"]])
        ax = fig.add_subplot(2, len(CROPS), col + 1, projection="3d")
        tris = [V[F] for V, F in render["leaf"]]
        if tris:
            ax.add_collection3d(Poly3DCollection(np.concatenate(tris), facecolor="#5aa65a", linewidths=0, alpha=0.95))
        ftris = [V[F] for V, F in render["fruit"]]
        if ftris:
            ax.add_collection3d(Poly3DCollection(np.concatenate(ftris), facecolor="#d9534f", linewidths=0))
        for a, b in st.edges:
            if st.organ[b] == 0:
                ax.plot(*st.nodes[[a, b]].T, color=ORGAN_COLOR[0], lw=1.5)
        setup(ax, allpts)
        ax.set_title(f"{sp} ({age:g} d)\nHelios render mesh", fontsize=9)

        ax = fig.add_subplot(2, len(CROPS), len(CROPS) + col + 1, projection="3d")
        for a, b in st.edges:
            ax.plot(*st.nodes[[a, b]].T, color=ORGAN_COLOR[int(st.organ[b])], lw=1.2 if st.organ[b] else 2)
        polys = [l.vertices[l.faces] for l in asset.leaves]
        if polys:
            ax.add_collection3d(Poly3DCollection(np.concatenate(polys), facecolor="#8fd18f",
                                                 edgecolor="#2f6b2f", linewidths=0.2, alpha=0.9))
        for f in asset.fruits:
            ax.scatter(*st.nodes[f.node], s=(f.radius * 1e3) ** 2 / 3, color="#d9534f")
        setup(ax, allpts)
        ax.set_title(f"sim asset: {summary[sp]['sim_particles']} particles", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "gallery.png", dpi=110)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", OUT / "gallery.png")


if __name__ == "__main__":
    main()
