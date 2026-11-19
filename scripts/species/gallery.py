"""Render species-file plants at a few ages to PNG, for a human to eyeball (S0).

  .venv/bin/python scripts/species/gallery.py                    # every species file, default ages
  .venv/bin/python scripts/species/gallery.py eggplant barley --ages 30 60 90 --seed 2
  .venv/bin/python scripts/species/gallery.py eggplant --compare tomato   # a library model in the next row

Writes ``outputs/species/<species>.png`` (one panel per age) and ``outputs/species/gallery.png`` (all rows). Plants are
built in fast mode (``heliades.plantarchitecture.batch.build_plant``) and drawn with matplotlib: every visible primitive
as a polygon, textured primitives in the texel colour at their UV centroid (dropped where that texel is transparent,
which recovers the blade outline), the rest in their own colour.
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "species"

DEFAULT_AGES = {"eggplant": (30, 60, 90, 120), "barley": (20, 40, 60, 90), "cucumber_from_tomato": (20, 40, 60, 90)}


@lru_cache(maxsize=None)
def _texture(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0


def primitive_colors(snap):
    """(n, 3) colours and (n,) visible mask."""
    col = np.asarray(snap.color, np.float32)[:, :3].copy()
    keep = ~np.asarray(snap.hidden, bool)
    for i, tex in enumerate(snap.texture):
        if not tex or not keep[i] or snap.nuv[i] == 0:
            continue
        img = _texture(tex)
        uv = snap.uv[i, :snap.nuv[i]].mean(0)
        h, w = img.shape[:2]
        x = min(w - 1, max(0, int(uv[0] * w)))
        y = min(h - 1, max(0, int((1 - uv[1]) * h)))
        t = img[y, x]
        if t[3] < 250 / 255:
            keep[i] = False
        col[i] = t[:3]
    return col, keep


def draw(ax, snap, title):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    col, keep = primitive_colors(snap)
    idx = np.nonzero(keep)[0]
    polys = [snap.vertices[i, :snap.nverts[i]] for i in idx]
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolor=np.clip(col[idx], 0, 1), edgecolor="none"))
        P = np.concatenate(polys)
        lo, hi = P.min(0), P.max(0)
        c, r = (lo + hi) / 2, max((hi - lo).max() / 2 * 1.05, 0.05)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(max(lo[2], c[2] - r), max(lo[2], c[2] - r) + 2 * r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(8, -60)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def build(model, age, seed):
    from heliades.plantarchitecture.batch import build_plant
    sc, pa, pid = build_plant(model, float(age), seed=seed, mode="fast")
    h = float(pa.getPlantHeight(pid))
    return sc.snapshot(), h, pa, pid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="*")
    ap.add_argument("--ages", nargs="*", type=float)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", nargs="*", default=[], help="library models to draw in extra rows")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from heliades.species import available_species
    names = a.species or [n for n in available_species() if n != "cucumber_from_tomato"]
    rows = names + a.compare
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ages_of = {n: tuple(a.ages) if a.ages else DEFAULT_AGES.get(n, (20, 40, 60, 90)) for n in rows}
    ncol = max(len(v) for v in ages_of.values())
    big = plt.figure(figsize=(3.4 * ncol, 3.8 * len(rows)), dpi=110)
    written = []
    for r, name in enumerate(rows):
        fig = plt.figure(figsize=(3.4 * len(ages_of[name]), 3.8), dpi=130)
        for c, age in enumerate(ages_of[name]):
            snap, h, pa, pid = build(name, age, a.seed)
            title = (f"{name} {age:g} d\nh {h:.2f} m, {pa.getPlantLeafCount(pid)} leaves, "
                     f"{len(pa.getPlantFruitObjectIDs(pid))} fruit")
            draw(fig.add_subplot(1, len(ages_of[name]), c + 1, projection="3d"), snap, title)
            draw(big.add_subplot(len(rows), ncol, r * ncol + c + 1, projection="3d"), snap, title)
            print(title.replace("\n", ": "))
        fig.tight_layout()
        p = out / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        written.append(p)
    big.tight_layout()
    p = out / "gallery.png"
    big.savefig(p)
    written.append(p)
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
