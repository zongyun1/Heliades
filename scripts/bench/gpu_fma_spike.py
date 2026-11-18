# SPDX-License-Identifier: LGPL-2.1-or-later
"""GPU spike (P1): can a Warp kernel reproduce Helios's float32 transform arithmetic bit for bit?

Warp 1.19 compiles CUDA kernels with NVRTC and passes ``--fmad=true`` by default (module option
``fuse_fp``), which lets the compiler contract ``a*b + c`` into one FMA (one rounding instead of two). With
``wp.set_module_options({"fuse_fp": False})`` (or ``@wp.kernel(module_options={"fuse_fp": False})``) it
passes ``--fmad=false``. This script runs, on ~1e6 random vertices / matrices:

* ``helios::matmult`` (M = sum_k ML[i,k] MR[k,j] from 0, k in order) for a batch of matrices, and
* the vertex transform of Patch/Triangle::getVertices (``T[0]*x + T[1]*y + T[2]*z + T[3]`` left to right),

on the GPU with fuse_fp on and off, and compares bitwise with the numpy float32 reference (elementwise
numpy float32 ops are IEEE-correctly rounded, which is what the x86-64 SSE build of Helios does).

    python scripts/bench/gpu_fma_spike.py [N]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import warp as wp

f32 = np.float32


def _ref_matmult(L, R):
    """(n,4,4) x (n,4,4), Helios matmult order."""
    M = np.zeros_like(R)
    for k in range(4):
        M = M + L[:, :, k:k + 1] * R[:, k:k + 1, :]
    return M


def _ref_apply(T, Y):
    """(n,4,4) matrices on (n,3) points."""
    out = np.empty_like(Y)
    for r in range(3):
        out[:, r] = ((T[:, r, 0] * Y[:, 0] + T[:, r, 1] * Y[:, 1]) + T[:, r, 2] * Y[:, 2]) + T[:, r, 3]
    return out


def make_kernels(fuse: bool):
    opts = {"fuse_fp": fuse, "fast_math": False}

    @wp.kernel(module_options=opts, module="unique")
    def matmult_k(L: wp.array(dtype=wp.mat44f), R: wp.array(dtype=wp.mat44f), out: wp.array(dtype=wp.mat44f)):
        t = wp.tid()
        a = L[t]
        b = R[t]
        m = wp.mat44f()
        for i in range(4):
            for j in range(4):
                acc = float(0.0)
                for k in range(4):
                    acc = acc + a[i, k] * b[k, j]
                m[i, j] = acc
        out[t] = m

    @wp.kernel(module_options=opts, module="unique")
    def apply_k(T: wp.array(dtype=wp.mat44f), Y: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
        t = wp.tid()
        m = T[t]
        y = Y[t]
        v = wp.vec3f()
        for r in range(3):
            v[r] = ((m[r, 0] * y[0] + m[r, 1] * y[1]) + m[r, 2] * y[2]) + m[r, 3]
        out[t] = v

    return matmult_k, apply_k


def run(n: int = 1_000_000, seed: int = 0) -> dict:
    wp.init()
    dev = "cuda:0" if wp.is_cuda_available() else "cpu"
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n, 4, 4)).astype(f32) * f32(3)
    R = rng.standard_normal((n, 4, 4)).astype(f32)
    L[:, 3, :3] = R[:, 3, :3] = 0
    L[:, 3, 3] = R[:, 3, 3] = 1
    Y = rng.standard_normal((n, 3)).astype(f32)
    t = time.perf_counter()
    refM, refV = _ref_matmult(L, R), _ref_apply(L, Y)
    t_np = time.perf_counter() - t
    res = {"device": dev, "n": n, "numpy_s": t_np}
    for fuse in (True, False):
        mk, ak = make_kernels(fuse)
        dL, dR = wp.array(L, dtype=wp.mat44f, device=dev), wp.array(R, dtype=wp.mat44f, device=dev)
        dY = wp.array(Y, dtype=wp.vec3f, device=dev)
        oM, oV = wp.empty(n, dtype=wp.mat44f, device=dev), wp.empty(n, dtype=wp.vec3f, device=dev)
        wp.launch(mk, n, inputs=[dL, dR, oM], device=dev)
        wp.launch(ak, n, inputs=[dL, dY, oV], device=dev)
        wp.synchronize()
        t = time.perf_counter()
        wp.launch(mk, n, inputs=[dL, dR, oM], device=dev)
        wp.launch(ak, n, inputs=[dL, dY, oV], device=dev)
        wp.synchronize()
        t_gpu = time.perf_counter() - t
        gM, gV = oM.numpy(), oV.numpy()
        dm = int((gM.view(np.uint32) != refM.view(np.uint32)).sum())
        dv = int((gV.view(np.uint32) != refV.view(np.uint32)).sum())
        res[f"fuse_fp={fuse}"] = {"matmult_diff_entries": dm, "apply_diff_entries": dv, "kernel_s": t_gpu}
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    for k, v in run(n).items():
        print(f"{k}: {v}")
