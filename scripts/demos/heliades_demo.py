"""Build a toy plant from Heliades compound objects (tube stem + petioles, tile leaves), export USD,
render a PNG, and confirm the same calls on Helios give an identical scene.

  .venv/bin/python scripts/demos/heliades_demo.py   ->  outputs/heliades/demo_plant.usda, demo_plant.png
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from heliades.parity.backends import ForgeBackend, HeliosBackend  # noqa: E402
from heliades.parity.compare import compare_snapshots  # noqa: E402

OUT = ROOT / "outputs" / "heliades"


def build(B, seed=0):
    r = np.random.default_rng(seed)
    t = np.linspace(0, 1, 12)
    stem = np.c_[0.03 * np.sin(3 * t), 0.02 * np.sin(2 * t + 1), 0.5 * t]
    B.add_tube(10, stem, np.linspace(0.006, 0.002, len(t)), [(0.35, 0.45, 0.15)] * len(t))
    for k in range(1, 10):
        base = stem[k]
        az = k * 2.39996                       # golden-angle phyllotaxis
        out = np.array([math.cos(az), math.sin(az), 0.0])
        s = np.linspace(0, 1, 6)[:, None]
        L = 0.12 * (1 - 0.06 * k)
        pet = base + L * (s * out + 0.5 * np.sin(np.pi * s * 0.7) * [0, 0, 1] - 0.3 * s ** 2 * [0, 0, 1])
        B.add_tube(6, pet, np.full(6, 0.0015), [(0.4, 0.55, 0.2)] * 6)
        tip = pet[-1]
        size = (L * 0.9, L * 0.45)
        center = tip + 0.5 * size[0] * out
        B.add_tile(center, size, (r.uniform(-0.4, -0.1), -az), (4, 2), (0.25, 0.6, 0.2))
    return B.snapshot()


def render(snap, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6, 6), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    polys = [snap.vertices[i, : snap.nverts[i]] for i in range(len(snap))]
    ax.add_collection3d(Poly3DCollection(polys, facecolor=snap.color[:, :3], edgecolor="none"))
    ax.set_xlim(-0.25, 0.25), ax.set_ylim(-0.25, 0.25), ax.set_zlim(0, 0.5)
    ax.set_box_aspect((1, 1, 1)), ax.view_init(15, -60), ax.set_axis_off()
    fig.savefig(path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    snap = build(ForgeBackend())
    from heliades.io.usd import write_usd
    write_usd(snap, OUT / "demo_plant.usda")
    render(snap, OUT / "demo_plant.png")
    rep = compare_snapshots(snap, build(HeliosBackend()))
    print(f"{len(snap)} primitives, {len(snap.objects)} objects -> {OUT}/demo_plant.usda, demo_plant.png")
    print("parity vs Helios:", rep)
