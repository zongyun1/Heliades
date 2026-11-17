"""Render videos to outputs/videos/.

  growth   Helios tomato growing day 10 -> 80 (Helios render meshes, orbiting camera)
  poke     procedural plant: finger presses a leaf, releases, plant rings down (stable case)
  settle   Helios tomato dropped into the current XPBD sim (shows the collapse problem)

Run:  .venv/bin/python scripts/make_videos.py [growth poke settle]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))  # planttwin
from planttwin import render as R  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "videos"


def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    return fig, ax


def growth(species="tomato", start=10, end=80, fps=24, frames_per_day=2):
    from pyhelios import Context, DataTypes, PlantArchitecture

    from planttwin.helios_import import _quiet, _v, _welded_mesh

    ctx = Context()
    ctx.seedRandomGenerator(0)
    pa = PlantArchitecture(ctx)
    with _quiet():
        pa.loadPlantModelFromLibrary(species)
        pid = pa.buildPlantInstanceFromLibrary(DataTypes.vec3(0, 0, 0), float(start))
    fig, ax = _fig()
    vw = R.VideoWriter(OUT / f"growth_{species}.mp4", fig, fps)
    n = (end - start) * frames_per_day
    for i in range(n + 1):
        if i:
            with _quiet():
                pa.advanceTime(1.0 / frames_per_day, plant_id=pid)
        age = start + i / frames_per_day
        R.setup_axes(ax, (0, 0), 0.5, azim=-70 + 0.6 * i)
        R.draw_ground(ax, (0, 0), 0.5)
        segs, widths = [], []
        for objs, scale in ((pa.getPlantInternodeObjectIDs(pid), 1.0), (pa.getPlantPetioleObjectIDs(pid), 1.0),
                            (pa.getPlantPeduncleObjectIDs(pid), 1.0)):
            for o in objs:
                P = np.array([_v(p) for p in ctx.getTubeObjectNodes(o)])
                rr = np.array(ctx.getTubeObjectNodeRadii(o))
                segs += [P[k:k + 2] for k in range(len(P) - 1)]
                widths += list(np.clip(rr[1:] * 1e3 * scale, 0.4, 5))
        R.draw_polylines(ax, np.array(segs), widths, R.STEM)
        leaf = [_welded_mesh(ctx, o) for o in pa.getPlantLeafObjectIDs(pid)]
        if leaf:
            R.draw_tris(ax, np.concatenate([V[F] for V, F in leaf]), R.LEAF)
        fl = [_welded_mesh(ctx, o) for o in pa.getPlantFlowerObjectIDs(pid)]
        if fl:
            R.draw_tris(ax, np.concatenate([V[F] for V, F in fl]), "#f4d03f")
        fr = [_welded_mesh(ctx, o) for o in pa.getPlantFruitObjectIDs(pid)]
        if fr:
            R.draw_tris(ax, np.concatenate([V[F] for V, F in fr]), R.FRUIT)
        ax.text2D(0.04, 0.95, f"Helios {species}   day {age:4.1f}", transform=ax.transAxes, fontsize=13)
        vw.frame()
        if i % 20 == 0:
            print(f"  growth frame {i}/{n}", flush=True)
    vw.close()


def poke(fps=30):
    from planttwin.procedural import make_plant
    from planttwin.real2sim import PokeExperiment
    from planttwin.sim import PlantModel

    asset = make_plant(seed=0)
    exp = PokeExperiment(fps=fps, duration=3.0)
    model = PlantModel(asset)
    xe, qe = model.settle(exp.settle_time, substeps=exp.substeps * 2)
    press = model.leaf_idx[exp.leaf][asset.leaves[exp.leaf].keypoints["mid"]]
    (sph,) = exp.colliders(xe[:, press])
    X, _, _ = model.simulate(exp.duration, fps=fps, substeps=exp.substeps * 2, colliders=(sph,),
                             x0=xe, q_init=qe)
    fig, ax = _fig()
    vw = R.VideoWriter(OUT / "poke_procedural.mp4", fig, fps)
    for f in range(len(X)):
        t = f / fps
        R.setup_axes(ax, (0, 0), 0.16, azim=-80 + 0.15 * f)
        R.draw_ground(ax, (0, 0), 0.16)
        s = (sph.center(t)[0], sph.radius) if t < exp.t_release else None
        R.draw_asset_state(ax, model, X[f, 0], s)
        ax.text2D(0.04, 0.95, f"poke: press, release at {exp.t_release:.2f}s   t={t:4.2f}s",
                  transform=ax.transAxes, fontsize=12)
        vw.frame()
    vw.close()


def settle(fps=30, duration=2.0, substeps=128):
    from planttwin.helios_import import build_helios_plant
    from planttwin.sim import PlantModel

    asset, _ = build_helios_plant("tomato", 75, seed=0)
    model = PlantModel(asset)
    model.set_params(asset.physics.copy(stem_youngs_modulus=3e8, damping=3.0))
    X, _, _ = model.simulate(duration, fps=fps, substeps=substeps)
    fig, ax = _fig()
    vw = R.VideoWriter(OUT / "settle_helios_tomato.mp4", fig, fps)
    for f in range(len(X)):
        R.setup_axes(ax, (0, 0), 0.45, azim=-70)
        R.draw_ground(ax, (0, 0), 0.45)
        R.draw_asset_state(ax, model, X[f, 0])
        ax.text2D(0.04, 0.95, f"Helios tomato in current XPBD sim (E=3e8 Pa)   t={f / fps:4.2f}s",
                  transform=ax.transAxes, fontsize=11)
        ax.text2D(0.04, 0.91, "known issue: iterative solver can't hold the stiff rod tree",
                  transform=ax.transAxes, fontsize=10, color="#c0392b")
        vw.frame()
    vw.close()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["growth", "poke", "settle"]
    for name in which:
        t = time.time()
        globals()[name]()
        print(f"{name}: done in {time.time() - t:.0f}s", flush=True)
    print("wrote", sorted(p.name for p in OUT.glob("*.mp4")))
