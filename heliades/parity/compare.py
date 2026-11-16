# SPDX-License-Identifier: LGPL-2.1-or-later
"""Comparison primitives for differential testing against Helios.

Three tiers, strictest first:
  exact   bitwise equal (RNG streams, integer topology)
  ulp     float32 values within N units-in-last-place (geometry from the same op sequence; libm
          sin/cos/sqrt may differ by an ulp)
  dist    statistical agreement over many seeds (stochastic models whose draw order is not yet
          reproduced): two-sample KS test plus relative mean difference per metric
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .snapshot import Snapshot


def ulp_distance(a, b) -> np.ndarray:
    """Units-in-last-place between float32 arrays (0 = identical; sign-aware ordering)."""
    a = np.ascontiguousarray(a, np.float32).view(np.int32).astype(np.int64)
    b = np.ascontiguousarray(b, np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, -(2**31) - a, a)
    b = np.where(b < 0, -(2**31) - b, b)
    return np.abs(a - b)


@dataclass
class Report:
    ok: bool
    checks: dict = field(default_factory=dict)   # name -> dict(max_ulp / max_abs / n_bad / ...)
    first_divergence: str | None = None

    def __str__(self):
        lines = [f"{'PASS' if self.ok else 'FAIL'}" + (f"  first divergence: {self.first_divergence}" if self.first_divergence else "")]
        for k, v in self.checks.items():
            lines.append(f"  {k:<10} " + "  ".join(f"{a}={b}" for a, b in v.items()))
        return "\n".join(lines)


def compare_snapshots(a: Snapshot, b: Snapshot, max_ulp: int = 4, abs_floor: float = 1e-6,
                      tex_max_ulp: int = 0) -> Report:
    """Primitive-by-primitive comparison. A float passes if within max_ulp OR within abs_floor
    (values that should be 0 come out as +-1e-8 depending on rounding, which is huge in ulps).
    UV and solid fraction use ``tex_max_ulp`` (default 0) and no floor; see ``_compare_texture``."""
    r = Report(True)
    if len(a) != len(b) or not np.array_equal(a.uuid, b.uuid):
        r.ok = False
        r.checks["uuid"] = {"n_a": len(a), "n_b": len(b)}
        common = np.intersect1d(a.uuid, b.uuid)
        missing = np.setxor1d(a.uuid, b.uuid)
        r.first_divergence = f"uuid sets differ ({len(common)} common, first unmatched {missing[:5].tolist()})"
        return r
    for name in ("type", "nverts", "parent"):
        bad = np.nonzero(getattr(a, name) != getattr(b, name))[0]
        r.checks[name] = {"n_bad": len(bad)}
        if len(bad):
            r.ok = False
            r.first_divergence = r.first_divergence or f"{name} at uuid {a.uuid[bad[0]]}"
    for name in ("vertices", "area", "normal", "color"):
        x, y = getattr(a, name).copy(), getattr(b, name).copy()
        if name == "vertices":  # zero the padding so it never counts
            pad = np.arange(4)[None, :] >= a.nverts[:, None]
            x[pad], y[pad] = 0, 0
        x, y = x.reshape(len(a), -1), y.reshape(len(a), -1)
        u = ulp_distance(x, y)
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        bad = ((u > max_ulp) & (d > abs_floor)).any(1)
        r.checks[name] = {"max_ulp": int(u.max()) if u.size else 0, "max_abs": float(d.max()) if d.size else 0.0,
                          "n_bad": int(bad.sum())}
        if bad.any():
            r.ok = False
            i = int(np.nonzero(bad)[0][0])
            r.first_divergence = r.first_divergence or f"{name} at uuid {a.uuid[i]}: {x[i]} vs {y[i]}"
    _compare_texture(a, b, r, tex_max_ulp)
    _compare_objects(a.objects, b.objects, r, max_ulp, abs_floor)
    return r


def texture_key(path: str) -> str:
    """How texture files are compared: by basename.

    Helios stores the texture string verbatim (Material::texture_file is whatever was passed to
    addPatch/addTriangle/addTileObject/addTubeObject; only the Texture registry entry holds the resolved
    path). PyHelios then makes patch/triangle paths absolute (os.path.abspath) but passes tile/tube paths
    through unchanged, so the full string depends on the binding and the cwd, not on Helios semantics.
    The basename is what both backends agree on for the same file."""
    return path.replace("\\", "/").rsplit("/", 1)[-1] if path else ""


def _compare_texture(a: Snapshot, b: Snapshot, r: Report, max_ulp: int):
    """UV count exact, texture basename exact, UV and solid fraction at the ulp tier with N = max_ulp
    (0 by default) and no absolute floor. Neither goes through libm: UVs are a few IEEE float32 ops on the
    inputs and the solid fraction is a ratio of two pixel counts, so the only slack the geometry tier
    needs (sinf/cosf) does not apply, and a one-pixel change in a count can be smaller than 1e-6."""
    bad = np.nonzero(a.nuv != b.nuv)[0]
    r.checks["nuv"] = {"n_bad": len(bad)}
    if len(bad):
        r.ok = False
        r.first_divergence = r.first_divergence or f"nuv at uuid {a.uuid[bad[0]]}: {a.nuv[bad[0]]} vs {b.nuv[bad[0]]}"
    ka, kb = [texture_key(t) for t in a.texture], [texture_key(t) for t in b.texture]
    bad = [i for i, (x, y) in enumerate(zip(ka, kb)) if x != y]
    r.checks["texture"] = {"n_textured": sum(1 for t in ka if t), "n_bad": len(bad)}
    if bad:
        r.ok = False
        r.first_divergence = r.first_divergence or f"texture at uuid {a.uuid[bad[0]]}: {ka[bad[0]]!r} vs {kb[bad[0]]!r}"
    for name in ("uv", "solid_fraction"):
        x, y = getattr(a, name).copy(), getattr(b, name).copy()
        if name == "uv":
            pad = np.arange(4)[None, :] >= np.minimum(a.nuv, b.nuv)[:, None]
            x[pad], y[pad] = 0, 0
        x, y = x.reshape(len(a), -1), y.reshape(len(a), -1)
        u = ulp_distance(x, y)
        bad = (u > max_ulp).any(1)
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        r.checks[name] = {"max_ulp": int(u.max()) if u.size else 0, "max_abs": float(d.max()) if d.size else 0.0,
                          "n_bad": int(bad.sum())}
        if bad.any():
            r.ok = False
            i = int(np.nonzero(bad)[0][0])
            r.first_divergence = r.first_divergence or f"{name} at uuid {a.uuid[i]}: {x[i]} vs {y[i]}"


def _compare_objects(a, b, r: Report, max_ulp: int, abs_floor: float):
    if not np.array_equal(a.oid, b.oid) or not np.array_equal(a.type, b.type):
        r.ok = False
        r.checks["objects"] = {"n_a": len(a), "n_b": len(b)}
        r.first_divergence = r.first_divergence or f"object ids/types differ: {a.oid[:5]} {a.type[:5]} vs {b.oid[:5]} {b.type[:5]}"
        return
    bad_members = [int(o) for o, x, y in zip(a.oid, a.uuids, b.uuids) if not np.array_equal(x, y)]
    r.checks["obj_uuids"] = {"n": len(a), "n_bad": len(bad_members)}
    if bad_members:
        r.ok = False
        r.first_divergence = r.first_divergence or f"member UUIDs of object {bad_members[0]}"
    for name in ("center", "transform"):
        x, y = getattr(a, name), getattr(b, name)
        if not x.size:
            continue
        u = ulp_distance(x, y)
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        bad = ((u > max_ulp) & (d > abs_floor)).any(1)
        r.checks["obj_" + name] = {"max_ulp": int(u.max()), "max_abs": float(d.max()), "n_bad": int(bad.sum())}
        if bad.any():
            r.ok = False
            i = int(np.nonzero(bad)[0][0])
            r.first_divergence = r.first_divergence or f"object {a.oid[i]} {name}: {x[i]} vs {y[i]}"


def compare_streams(a, b, max_ulp: int = 0) -> Report:
    """Sequences of scalars (e.g. RNG draws). Reports the index of the first mismatch."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape:
        return Report(False, {"len": {"a": len(a), "b": len(b)}}, "length differs")
    u = ulp_distance(a, b)
    bad = np.nonzero(u > max_ulp)[0]
    r = Report(len(bad) == 0, {"stream": {"n": len(a), "n_bad": len(bad), "max_ulp": int(u.max()) if len(u) else 0}})
    if len(bad):
        i = bad[0]
        r.first_divergence = f"draw {i}: {a[i]!r} vs {b[i]!r}"
    return r


def compare_distributions(a: dict[str, np.ndarray], b: dict[str, np.ndarray], alpha: float = 1e-3,
                          rel_mean_tol: float = 0.05) -> Report:
    """Per-metric two-sample KS test (fail if p < alpha) and relative mean difference."""
    from scipy.stats import ks_2samp
    r = Report(True)
    for k in a:
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        p = ks_2samp(x, y).pvalue if len(x) > 1 and len(y) > 1 else 1.0
        rel = abs(x.mean() - y.mean()) / max(abs(y.mean()), 1e-12)
        good = p >= alpha and rel <= rel_mean_tol
        r.checks[k] = {"ks_p": round(float(p), 4), "rel_mean": round(float(rel), 4), "ok": good}
        if not good:
            r.ok = False
            r.first_divergence = r.first_divergence or k
    return r
