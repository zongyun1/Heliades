# SPDX-License-Identifier: LGPL-2.1-or-later
"""Backend-neutral dense dump of a scene, used for differential comparison and bulk export."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectTable:
    oid: np.ndarray                 # (m,) int64, sorted
    type: np.ndarray                # (m,) int32 (OBJECT_TYPE_*)
    uuids: list[np.ndarray]         # member primitive UUIDs, in object order
    center: np.ndarray              # (m,3) float32, getObjectCenter
    transform: np.ndarray           # (m,16) float32, getObjectTransformationMatrix

    @staticmethod
    def empty() -> "ObjectTable":
        return ObjectTable(np.zeros(0, np.int64), np.zeros(0, np.int32), [], np.zeros((0, 3), np.float32),
                           np.zeros((0, 16), np.float32))

    def __len__(self):
        return len(self.oid)


@dataclass
class Snapshot:
    uuid: np.ndarray       # (n,) int64, sorted
    type: np.ndarray       # (n,) int32: 0 patch, 1 triangle, 2 voxel
    vertices: np.ndarray   # (n,4,3) float32, padded
    nverts: np.ndarray     # (n,) int32
    area: np.ndarray       # (n,) float32
    normal: np.ndarray     # (n,3) float32
    color: np.ndarray      # (n,4) float32 RGBA
    parent: np.ndarray = None           # (n,) int64 parent object ID (0 = none)
    objects: ObjectTable = field(default_factory=ObjectTable.empty)

    def __post_init__(self):
        if self.parent is None:
            self.parent = np.zeros(len(self.uuid), np.int64)

    def __len__(self):
        return len(self.uuid)

    def triangles(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(points (m,3), tri indices (k,3), tri->primitive (k,)): patches split 0-1-2, 0-2-3."""
        pts = self.vertices.reshape(-1, 3)
        base = np.arange(len(self)) * 4
        tri = [np.c_[base, base + 1, base + 2]]
        owner = [np.arange(len(self))]
        q = np.nonzero(self.nverts == 4)[0]
        tri.append(np.c_[base[q], base[q] + 2, base[q] + 3])
        owner.append(q)
        return pts, np.concatenate(tri).astype(np.int32), np.concatenate(owner)
