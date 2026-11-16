import math

import numpy as np
import pytest

from heliades.core.scene import Scene


def _scene():
    s = Scene()
    s.add_patch((0, 0, 1), (2, 2), (0, 0), (0.2, 0.8, 0.2))              # horizontal at z=1
    s.add_triangle((3, -1, 0), (5, -1, 0), (4, 1, 0), (0.6, 0.4, 0.2))    # in z=0 plane
    s.add_patch((0, 3, 0.5), (1, 1), (math.pi / 2, 0))                   # vertical (normal along y)
    return s


def test_geometry_basics():
    s = _scene()
    assert s.primitive_area(0) == pytest.approx(4.0)
    assert s.primitive_area(1) == pytest.approx(2.0)
    assert np.allclose(np.abs(s.primitive_normal(2)), [0, 1, 0], atol=1e-6)


def test_usd_roundtrip(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from heliades.io.usd import write_usd
    snap = _scene().snapshot()
    p = write_usd(snap, tmp_path / "s.usda", face_data={"organ": np.array([1, 2, 3])})
    stage = Usd.Stage.Open(str(p))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Scene/primitives"))
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [4, 3, 4]
    pv = UsdGeom.PrimvarsAPI(mesh)
    assert list(pv.GetPrimvar("heliosUUID").Get()) == [0, 1, 2]
    assert list(pv.GetPrimvar("organ").Get()) == [1, 2, 3]
    pts = np.array(mesh.GetPointsAttr().Get())
    assert np.allclose(pts, snap.vertices[np.arange(4)[None] < snap.nverts[:, None]])


def test_device_raycast_hits_right_primitive():
    pytest.importorskip("warp")
    from heliades.core.device import DeviceScene
    ds = DeviceScene(_scene().snapshot())
    o = np.array([[0.5, 0.5, 5], [4, 0, 5], [0, -2, 0.5], [9, 9, 5]], np.float32)
    d = np.array([[0, 0, -1], [0, 0, -1], [0, 1, 0], [0, 0, -1]], np.float32)
    t, hit = ds.cast(o, d)
    assert hit.tolist() == [0, 1, 2, -1]
    assert t[:3] == pytest.approx([4.0, 5.0, 5.0], abs=1e-5)
