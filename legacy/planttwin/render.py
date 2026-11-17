"""Minimal offline rendering: matplotlib 3D frames piped into ffmpeg (H.264 mp4)."""
from __future__ import annotations

import subprocess

import numpy as np

LEAF, LEAF_EDGE, STEM, FRUIT = "#5aa65a", "#2f6b2f", "#6b4f2a", "#d9534f"
ORGAN_COLOR = {0: "#6b4f2a", 1: "#7a8f3a", 2: "#3f7f3f", 3: "#3f7f3f"}


class VideoWriter:
    def __init__(self, path, fig, fps=30):
        self.fig = fig
        w, h = fig.canvas.get_width_height()
        w2, h2 = w - w % 2, h - h % 2
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-vf", f"crop={w2}:{h2}:0:0",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
            stdin=subprocess.PIPE)

    def frame(self):
        self.fig.canvas.draw()
        self.proc.stdin.write(np.asarray(self.fig.canvas.buffer_rgba()).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def setup_axes(ax, center, half, zmin=0.0, elev=12, azim=-70):
    ax.cla()
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(zmin, zmin + 2 * half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev, azim)
    ax.set_axis_off()


def draw_ground(ax, center, half, z=0.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c = np.array(center[:2])
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half + c
    ax.add_collection3d(Poly3DCollection([np.c_[sq, np.full(4, z)]], facecolor="#e9e2d0",
                                         edgecolor="none", alpha=0.6))


def draw_tris(ax, tris, color, edge="none", lw=0.0, alpha=1.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    if len(tris):
        ax.add_collection3d(Poly3DCollection(tris, facecolor=color, edgecolor=edge,
                                             linewidths=lw, alpha=alpha))


def draw_polylines(ax, segs, widths, color):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    if len(segs):
        ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=widths))


def draw_asset_state(ax, model, x, sphere=None):
    """Draw a PlantModel configuration x (n,3): rods by organ, sim leaf meshes, fruit."""
    st = model.asset.stem
    E = st.edges
    organ = st.organ if st.organ is not None else np.zeros(len(st.nodes), int)
    segs = x[E]
    widths = np.clip(st.radius[E[:, 1]] * 1e3 * 0.8, 0.5, 4)
    for o in np.unique(organ[E[:, 1]]):
        k = organ[E[:, 1]] == o
        draw_polylines(ax, segs[k], widths[k], ORGAN_COLOR[int(o)])
    tris = [x[idx[l.faces]] for l, idx in zip(model.asset.leaves, model.leaf_idx)]
    if tris:
        draw_tris(ax, np.concatenate(tris), LEAF, LEAF_EDGE, 0.2, 0.95)
    for f in model.asset.fruits:
        ax.scatter(*x[f.node], s=(f.radius * 1e3) ** 2 * 0.6, color=FRUIT, depthshade=False)
    if sphere is not None:
        # matplotlib's per-collection depth sort hides the probe behind the leaf it presses;
        # draw in call order instead, probe last.
        ax.computed_zorder = False
        c, r = sphere
        u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:8j]
        ax.plot_surface(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v),
                        c[2] + r * np.cos(v), color="#c0392b", alpha=0.9, linewidth=0, zorder=10)
    else:
        ax.computed_zorder = True
