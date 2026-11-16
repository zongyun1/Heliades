# SPDX-License-Identifier: LGPL-2.1-or-later
"""GPU side: a Scene snapshot as a Warp BVH mesh, with batched ray casting.

This is the base the radiation / LiDAR / synthetic-camera ports will build on (Helios uses OptiX
or Vulkan ray tracing for the same job).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..parity.snapshot import Snapshot


class DeviceScene:
    def __init__(self, snap: Snapshot, device: str | None = None):
        self.device = device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
        pts, tri, owner = snap.triangles()
        self.points = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=self.device)
        self.indices = wp.array(tri.reshape(-1).astype(np.int32), dtype=wp.int32, device=self.device)
        self.tri_owner = wp.array(owner.astype(np.int32), dtype=wp.int32, device=self.device)
        self.uuid = snap.uuid
        self.mesh = wp.Mesh(points=self.points, indices=self.indices)

    def cast(self, origins: np.ndarray, directions: np.ndarray, t_max: float = 1e6):
        """Returns (t, primitive_index) per ray; index -1 / t=inf on miss."""
        n = len(origins)
        o = wp.array(origins.astype(np.float32), dtype=wp.vec3, device=self.device)
        d = wp.array(directions.astype(np.float32), dtype=wp.vec3, device=self.device)
        t = wp.zeros(n, dtype=wp.float32, device=self.device)
        hit = wp.zeros(n, dtype=wp.int32, device=self.device)
        wp.launch(_cast, dim=n, inputs=[self.mesh.id, o, d, float(t_max), self.tri_owner, t, hit], device=self.device)
        return t.numpy(), hit.numpy()


@wp.kernel
def _cast(mesh: wp.uint64, o: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3), t_max: float,
          owner: wp.array(dtype=wp.int32), t_out: wp.array(dtype=wp.float32), hit_out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    q = wp.mesh_query_ray(mesh, o[i], wp.normalize(d[i]), t_max)
    if q.result:
        t_out[i] = q.t
        hit_out[i] = owner[q.face]
    else:
        t_out[i] = wp.inf
        hit_out[i] = -1
