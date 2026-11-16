"""Vectorised SO(3) helpers (numpy, broadcasting over leading dims).

Conventions: rotation vectors phi (..., 3); rotation matrices (..., 3, 3); quaternions
(w, x, y, z). Everything is written element-wise so it maps 1:1 onto Warp `wp.func`s.
"""
from __future__ import annotations

import numpy as np

_SMALL = 1e-4


def skew(v: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) with skew(a) @ b == cross(a, b)."""
    z = np.zeros(v.shape[:-1])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([np.stack([z, -w, y], -1),
                     np.stack([w, z, -x], -1),
                     np.stack([-y, x, z], -1)], -2)


def _theta(phi):
    th2 = (phi * phi).sum(-1)
    th = np.sqrt(th2)
    small = th < _SMALL
    ths = np.where(small, 1.0, th)
    return th, th2, ths, small


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Rodrigues: exp(skew(phi))."""
    th, th2, ths, small = _theta(phi)
    A = np.where(small, 1.0 - th2 / 6.0, np.sin(ths) / ths)
    Bc = np.where(small, 0.5 - th2 / 24.0, (1.0 - np.cos(ths)) / (ths * ths))
    K = skew(phi)
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + A[..., None, None] * K + Bc[..., None, None] * (K @ K)


def jr_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3): phi_dot = Jr^{-1}(phi) u, where
    u is the body-frame angular velocity of exp(phi)."""
    th, th2, ths, small = _theta(phi)
    c = np.where(small, 1.0 / 12.0 + th2 / 720.0,
                 1.0 / (ths * ths) - (1.0 + np.cos(ths)) / (2.0 * ths * np.sin(ths)))
    K = skew(phi)
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + 0.5 * K + c[..., None, None] * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Matrix log (principal branch, |phi| <= pi). Used only for diagnostics/init."""
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(tr)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(th)
    small = th < _SMALL
    f = np.where(small, 0.5 + th * th / 12.0, th / (2.0 * np.where(small, 1.0, s)))
    return f[..., None] * v


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) (w, x, y, z), w >= 0."""
    m = R
    t = np.trace(m, axis1=-2, axis2=-1)
    w = np.sqrt(np.maximum(0.0, 1.0 + t)) / 2
    x = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    q = np.stack([w, x, y, z], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def frame_from_tangent(t: np.ndarray, prev: np.ndarray | None = None) -> np.ndarray:
    """Rotation matrix whose third column is unit vector t. If prev (3,3) is given the
    frame is the parallel transport of prev (minimal rotation of prev's z onto t)."""
    t = t / np.linalg.norm(t)
    if prev is None:
        a = np.array([0.0, 0.0, 1.0])
        base = np.eye(3)
    else:
        a = prev[:, 2]
        base = prev
    c = np.cross(a, t)
    d = float(a @ t)
    if d < -0.999999:
        ax = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1.0, 0])
        Rm = exp_so3(np.pi * ax / np.linalg.norm(ax))
    else:
        K = skew(c)
        Rm = np.eye(3) + K + K @ K / (1.0 + d)
    R = Rm @ base
    # re-orthonormalise
    u, _, vt = np.linalg.svd(R)
    return u @ vt


# ---------------------------------------------------------------- quaternions (w, x, y, z)
def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def quat_from_rotvec(phi):
    th, th2, ths, small = _theta(phi)
    s = np.where(small, 0.5 - th2 / 48.0, np.sin(0.5 * ths) / ths)
    return np.concatenate([np.cos(0.5 * th)[..., None], s[..., None] * phi], -1)


def rotvec_from_quat(q):
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sn = np.linalg.norm(v, axis=-1)
    th = 2.0 * np.arctan2(sn, q[..., 0])
    f = np.where(sn < 1e-12, 2.0, th / np.maximum(sn, 1e-300))
    return f[..., None] * v


def quat_to_matrix(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
