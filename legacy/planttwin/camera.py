"""Pinhole camera: the Pi(.) in the Real2Sim loss."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    K: np.ndarray  # (3,3)
    R: np.ndarray  # (3,3) world -> camera
    t: np.ndarray  # (3,)
    width: int = 1280
    height: int = 720

    @staticmethod
    def look_at(eye, target, f=1000.0, width=1280, height=720, up=(0, 0, 1)) -> "Camera":
        eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)  # image y points down
        R = np.stack([x, y, z])
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        return Camera(K, R, -R @ eye, width, height)

    def project(self, X: np.ndarray) -> np.ndarray:
        """(..., 3) world points -> (..., 2) pixels."""
        Xc = X @ self.R.T + self.t
        uv = Xc @ self.K.T
        return uv[..., :2] / uv[..., 2:3]
