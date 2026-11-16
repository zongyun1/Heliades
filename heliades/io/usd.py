# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenUSD export: a Scene snapshot becomes one UsdGeom.Mesh with per-face primvars.

Patches stay quads (faceVertexCount 4), triangles stay triangles, so UUIDs map 1:1 to faces and
any primitive data can ride along as a uniform primvar (e.g. organ labels, absorbed PAR).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..parity.snapshot import Snapshot


def write_usd(snap: Snapshot, path: str | Path, face_data: dict[str, np.ndarray] | None = None,
              up_axis: str = "Z", meters_per_unit: float = 1.0) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    path = Path(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Scene/primitives")

    counts = snap.nverts.astype(np.int32)
    mask = np.arange(4)[None, :] < counts[:, None]
    pts = snap.vertices[mask]
    idx = np.arange(len(pts), dtype=np.int32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(idx))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = pts.min(0), pts.max(0)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])

    pv = UsdGeom.PrimvarsAPI(mesh)
    col = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform)
    col.Set(Vt.Vec3fArray.FromNumpy(snap.color[:, :3].astype(np.float32)))
    pv.CreatePrimvar("heliosUUID", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.uniform).Set(
        Vt.IntArray.FromNumpy(snap.uuid.astype(np.int32)))
    for name, vals in (face_data or {}).items():
        vals = np.asarray(vals)
        tn = Sdf.ValueTypeNames.FloatArray if vals.dtype.kind == "f" else Sdf.ValueTypeNames.IntArray
        arr = Vt.FloatArray.FromNumpy(vals.astype(np.float32)) if vals.dtype.kind == "f" else Vt.IntArray.FromNumpy(vals.astype(np.int32))
        pv.CreatePrimvar(name, tn, UsdGeom.Tokens.uniform).Set(arr)
    stage.GetRootLayer().Save()
    return path
