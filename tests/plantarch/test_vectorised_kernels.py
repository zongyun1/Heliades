"""P1 vectorisations of plantarchitecture loops must give the statement-by-statement results bit for bit."""
import math

import numpy as np
import pytest

from heliades.core import vec as V
from heliades.plantarchitecture import inflorescence as INF
from heliades.plantarchitecture import leaf_prototype as LP
from heliades.plantarchitecture.random_parameter import PlantArchitectureError

f32 = np.float32


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, f32)).view(np.uint32)


def _lattice(rng, Nx, Ny):
    i = np.arange(Nx + 1, dtype=f32)
    j = np.arange(Ny + 1, dtype=f32) - f32(Ny // 2)
    x = (i / f32(Nx))[None, :] * f32(0.3)
    y = (j[:, None] * f32(0.02)) * np.sin(np.pi * i / Nx).astype(f32)[None, :]
    z = (x * x * f32(rng.uniform(-1, 1))) + y * y
    P = np.stack(np.broadcast_arrays(x, y, z), -1).astype(f32)
    return (P + rng.standard_normal(P.shape).astype(f32) * f32(1e-4)).reshape(-1, 3)


@pytest.mark.parametrize("seed", range(6))
def test_deform_leaf_lattice_matches_scalar(seed):
    rng = np.random.default_rng(seed)
    Nx, Ny = int(rng.integers(3, 14)), 2 * int(rng.integers(1, 4))
    rest = _lattice(rng, Nx, Ny)
    for flex in (0.0, 0.3, 2.0, 8.0):
        args = (rest, Nx, Ny, f32(rng.uniform(0.2, 1.0)), f32(1.0), f32(flex), f32(rng.uniform(0.3, 1.0)))
        try:
            ref = LP._deform_leaf_lattice_scalar(*args)
        except PlantArchitectureError as e:
            with pytest.raises(PlantArchitectureError, match=str(e)[:40]):
                LP.deformLeafLattice(*args)
            continue
        got = LP.deformLeafLattice(*args)
        assert np.array_equal(_bits(got), _bits(ref))


def _slow_weld(Vt, nv, tolerance):
    welded, cells = [], {}

    def weld(v):
        c = tuple(int(math.floor(float(f32(v[k] / tolerance)))) for k in range(3))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for index in cells.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        if V.magnitude(welded[index] - v) <= tolerance:
                            return index
        welded.append(np.asarray(v, f32).copy())
        cells.setdefault(c, []).append(len(welded) - 1)
        return len(welded) - 1

    return [[weld(np.asarray(v, f32)) for v in Vt[i, :nv[i]]] for i in range(len(nv))], welded


@pytest.mark.parametrize("seed", range(5))
def test_fast_weld_matches_sequential_weld_or_declines(seed):
    rng = np.random.default_rng(seed)
    tol = f32(1e-3)
    base = rng.uniform(-0.05, 0.05, (60, 3)).astype(f32)
    # vertices: exact repeats, 1-ulp neighbours (cliques) and, for odd seeds, chains of near points
    pts = [base[rng.integers(0, 60)] for _ in range(300)]
    pts += [np.nextafter(p, f32(1)) for p in pts[:40]]
    if seed % 2:
        p0 = base[0]
        pts += [p0 + f32(0.0007) * f32(k) for k in range(4)]
    pts = np.array(pts, f32)
    rng.shuffle(pts)
    m = len(pts) // 3
    Vt = np.zeros((m, 4, 3), f32)
    Vt[:, :3] = pts[: 3 * m].reshape(m, 3, 3)
    nv = np.full(m, 3, np.int32)
    fast = INF._weld_exact_duplicates(Vt, nv, tol)
    faces, welded = _slow_weld(Vt, nv, tol)
    if fast is None:
        assert seed % 2          # only the chains may need the sequential weld
        return
    assert fast[0] == faces
    assert np.array_equal(_bits(np.array(fast[1])), _bits(np.array(welded)))
