"""B3 contact scenarios shared by tests/heliades/test_sim_contact.py and scripts/contact_demo.py."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heliades.asset import Leaf, PhysicsParams, PlantAsset, StemGraph

from .analytic import cantilever_asset
from .contact import CapsuleObstacle, main_axis, stake


def floating_chain(n=10, L=0.3, r=0.005, E=1e10, z=0.05, direction=(1.0, 0.2, -0.15), damping=0.5):
    """Stiff ("rigid-ish") free chain: a cantilever asset with no clamped node (floating base)."""
    a = cantilever_asset(n, L, r, E=E, z=z, direction=direction, damping=damping)
    a.stem.fixed[:] = False
    return a


def block_asset(side=0.06, plate_z=0.00025, areal_density=5.0):
    """Block-like payload: a floating 4 cm rod carrying a square rigid plate (leaf payload of
    areal density `areal_density`) whose 4 corners are the contact points. The default plate
    height equals the leaf contact radius (leaf_thickness / 2): it starts exactly touching."""
    hz = 0.014
    nodes = np.array([[-0.02, 0, hz], [0.02, 0, hz]])
    st = StemGraph(nodes, np.array([[0, 1]]), np.full(2, 0.003), np.zeros(2, bool), organ=np.zeros(2, int))
    s = side / 2
    V = np.array([[-s, -s, plate_z], [s, -s, plate_z], [s, s, plate_z], [-s, s, plate_z], [0.02, 0, hz]])
    leaf = Leaf(V, np.array([[0, 1, 2], [0, 2, 3]]), attach_node=1, attach_vertex=4)
    return PlantAsset(st, [leaf], PhysicsParams(stem_youngs_modulus=1e9, leaf_areal_density=areal_density,
                                                damping=0.0))


def pick_leaf(asset, zmin=0.35, zmax=0.6):
    """Outermost leaflet with centroid height in [zmin, zmax]: (index, centroid, tangential
    unit vector, half width along it)."""
    cent = np.array([L.vertices.mean(0) for L in asset.leaves])
    rad = np.linalg.norm(cent[:, :2], axis=1)
    cand = np.flatnonzero((cent[:, 2] > zmin) & (cent[:, 2] < zmax))
    li = int(cand[np.argmax(rad[cand])])
    c0 = cent[li]
    er = np.r_[c0[:2] / np.linalg.norm(c0[:2]), 0.0]
    et = np.cross([0.0, 0.0, 1.0], er)
    half = float(np.abs((asset.leaves[li].vertices - c0) @ et).max())
    return li, c0, et, half


@dataclass
class FingerPush:
    """Vertical capsule finger sweeping tangentially through a leaf, holding, retracting, then
    parked far away. finger(t) -> CapsuleObstacle with the matching velocity."""
    start: np.ndarray
    vel: np.ndarray
    t_in: float = 0.6
    t_hold: float = 0.3
    t_out: float = 0.6
    radius: float = 0.008
    half_len: float = 0.03
    mu: float = 0.3
    mode: str = "implicit"

    def center(self, t):
        if t < self.t_in:
            return self.start + self.vel * t, self.vel
        if t < self.t_in + self.t_hold:
            return self.start + self.vel * self.t_in, 0 * self.vel
        if t < self.t_in + self.t_hold + self.t_out:
            return self.start + self.vel * (self.t_in - (t - self.t_in - self.t_hold)), -self.vel
        return self.start + np.array([10.0, 10.0, 10.0]), 0 * self.vel

    def __call__(self, t):
        c, v = self.center(t)
        dz = np.array([0, 0, self.half_len])
        return CapsuleObstacle(c - dz, c + dz, self.radius, v, self.mu, self.mode)

    @property
    def released(self):
        return self.t_in + self.t_hold + self.t_out


def finger_for_leaf(asset, speed=0.15, **kw):
    li, c0, et, half = pick_leaf(asset)
    R = kw.get("radius", 0.008)
    return FingerPush(c0 + et * (half + R + 0.01), -speed * et, **kw), li


FALL_DIR = np.array([0.95593315, -0.29358444])   # fall direction of the seed-1 tomato at E = 1e8


def stake_in_front(sim, mu=1.0, radius=0.006, gap=0.003, direction=FALL_DIR, mode="implicit"):
    """Vertical stake just in front of the main stem's top along `direction`, at the first
    position with >= gap clearance to the main axis (which is the only part it touches)."""
    T = sim.topo
    ax = main_axis(sim)
    a_, b_ = T.X0[T.node_a[ax], :2], T.X0[T.node_b[ax], :2]
    r_ax = T.r[ax]
    top = T.X0[T.node_b[ax]][np.argmax(T.X0[T.node_b[ax], 2])]
    ab = b_ - a_
    for d in np.arange(0.0, 0.1, 0.001):
        q = top[:2] + direction * d
        t = np.clip(((q - a_) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0, 1)
        if (np.linalg.norm(a_ + t[:, None] * ab - q, axis=-1) - r_ax - radius).min() > gap:
            break
    return stake(q[0], q[1], radius, 1.0, mu=mu, mode=mode, bodies=ax)


def points(sim):
    """(B, P, 3) rod nodes + leaf vertices."""
    return np.concatenate([sim.state.x] + sim.leaf_vertices(), 1)


def min_gap_ground(sim, height=0.0):
    """(B,) smallest signed distance of any contact feature (rod node spheres, leaf vertices
    with half the leaf thickness, fruit) to the ground plane z = height."""
    from .contact import _contact_model
    cm, _ = _contact_model(sim)
    k = sim._kin()
    pw = k.o[:, cm.p_body] + np.einsum("bnij,nj->bni", k.R[:, cm.p_body], cm.p_local)
    return (pw[..., 2] - height - cm.p_rad).min(1)
