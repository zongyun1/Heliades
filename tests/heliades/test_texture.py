"""W1-A textures: path resolution, PNG alpha decoding, registry semantics, and harness sensitivity."""
import os

import numpy as np
import pytest

from heliades.core.scene import Scene
from heliades.core.texture import (Texture, get_file_extension, read_png_alpha, resolve_file_path,
                                   try_resolve_file_path)
from heliades.parity.cases_texture import ALPHA_TEXTURES, _PA, synthetic_pngs, write_png

f32 = np.float32
LEAF = _PA + "TomatoLeaf.png"


def test_file_extension_matches_std_filesystem():
    assert get_file_extension("a/b/leaf.png") == ".png"
    assert get_file_extension("a/b.dir/leaf") == ""
    assert get_file_extension(".png") == ""          # a leading-dot name has no extension
    assert get_file_extension("x.tar.JPG") == ".JPG"
    assert get_file_extension("..") == ""


def test_resolve_order(tmp_path, monkeypatch):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "sub").mkdir()
    (tmp_path / "t.png").write_bytes(b"")
    monkeypatch.delenv("HELIOS_BUILD", raising=False)
    monkeypatch.chdir(tmp_path / "sub")               # cwd misses; parent holds plugins/ -> build dir
    assert resolve_file_path("t.png") == os.path.realpath(tmp_path / "t.png")
    monkeypatch.setenv("HELIOS_BUILD", str(tmp_path / "sub"))
    assert try_resolve_file_path("t.png") == ""
    with pytest.raises(FileNotFoundError):
        resolve_file_path("t.png")
    assert resolve_file_path(str(tmp_path / "t.png")) == os.path.realpath(tmp_path / "t.png")


def test_alpha_threshold_and_strip16(tmp_path):
    p = str(tmp_path / "a16.png")
    a = np.array([[63999, 64000, 64255, 64256]])        # high bytes 249, 250, 250, 251
    write_png(p, np.dstack([np.zeros((1, 4, 3)), a]), 16, 6)
    assert read_png_alpha(p).tolist() == [[False, True, True, True]]
    p = str(tmp_path / "a8.png")
    write_png(p, np.dstack([np.zeros((1, 3, 3)), [[249, 250, 255]]]), 8, 6)
    assert read_png_alpha(p).tolist() == [[False, True, True]]


def test_rgb16_trns_ambiguous_raises(tmp_path):
    p = str(tmp_path / "rgb16.png")
    import struct
    write_png(p, np.array([[[0x1200, 0x0000, 0x0000]]]), 16, 2, trns=struct.pack(">HHH", 0x1234, 0, 0))
    with pytest.raises(NotImplementedError):
        read_png_alpha(p)


def test_textured_patch_semantics():
    s = Scene()
    u = s.add_patch_textured((0, 0, 0), (2, 3), (0, 0), LEAF)
    p = s.primitives[u]
    assert s.primitive_has_texture(u) and s.primitive_texture_file(u) == LEAF
    assert p.color.tolist() == [0, 0, 0, 1]
    assert 0 < s.primitive_solid_fraction(u) < 1
    assert s.primitive_area(u) == f32(f32(f32(2) * f32(3)) * s.primitive_solid_fraction(u))
    t = s.add_triangle_textured((0, 0, 0), (1, 0, 0), (0, 1, 0), LEAF, (-0.5, 0), (1.5, 0), (0.5, 2))
    assert s.primitive_texture_uv(t).tolist() == [[-0.5, 0], [1, 0], [0.5, 1]]   # only the upper end clamps
    with pytest.raises(ValueError):
        s.add_patch_textured((0, 0, 0), (1, 1), (0, 0), LEAF, (0.9, 0.5), (0.4, 0.2))
    assert s.primitive_texture_uv(s.add_patch((0, 0, 0), (1, 1))).shape == (0, 2)


def test_registry_keyed_by_given_string(monkeypatch):
    monkeypatch.chdir(_PA)
    s = Scene()
    s.add_patch_textured((0, 0, 0), (1, 1), (0, 0), "TomatoLeaf.png")
    s.add_patch_textured((0, 0, 0), (1, 1), (0, 0), LEAF)
    assert sorted(s.textures) == sorted(["TomatoLeaf.png", LEAF])
    assert s.textures["TomatoLeaf.png"].get_texture_file() == os.path.realpath(LEAF)


def test_tile_resolution_check():
    s = Scene()
    with pytest.raises(ValueError, match="resolution"):
        s.add_tile_object_textured((0, 0, 0), (1, 1), (0, 0), (2, 1), synthetic_pngs()[-1])   # 1x1 image


def test_textured_tube_deletes_transparent_triangles():
    s = Scene()
    nodes = [(0, 0, 0), (0, 0, 0.3), (0, 0.05, 0.6), (0, 0.1, 1.0)]
    oid = s.add_tube_object_textured(8, nodes, [0.01] * 4, ALPHA_TEXTURES[0])
    members = s.object_primitive_uuids(oid)
    assert s.current_uuid == 2 * 3 * 8                    # every triangle consumed a UUID ...
    assert 0 < len(members) < 2 * 3 * 8                   # ... but transparent ones were deleted
    assert sorted(members) == s.all_uuids()
    assert all(s.primitive_area(u) > 0 for u in members)


def test_cache_reuses_first_value():
    t = Texture(LEAF)
    uv = np.array([[0.3, 0.3], [0.6, 0.31], [0.45, 0.62]], f32)
    a = t.get_solid_fraction(uv)
    b = t.get_solid_fraction(uv + f32(1e-5))           # same rounded pixel key
    assert a == b


def test_compare_detects_uv_and_texture_divergence():
    from heliades.parity.backends import ForgeBackend
    from heliades.parity.compare import compare_snapshots
    a, b = ForgeBackend(), ForgeBackend()
    for B in (a, b):
        B.add_patch_textured((0.1, 0.2, 0.3), (1.0, 2.0), (0.4, 0.5), LEAF, (0.5, 0.5), (0.3, 0.3))
    sa = a.snapshot()
    assert compare_snapshots(sa, b.snapshot()).ok
    sb = b.snapshot()
    sb.uv[0, 1, 0] = np.nextafter(sb.uv[0, 1, 0], f32(2))
    r = compare_snapshots(sa, sb)
    assert not r.ok and "uv" in r.first_divergence
    sb = b.snapshot()
    sb.solid_fraction[0] = np.nextafter(sb.solid_fraction[0], f32(2))
    assert "solid_fraction" in compare_snapshots(sa, sb).first_divergence
    sb = b.snapshot()
    sb.texture[0] = "/elsewhere/" + os.path.basename(LEAF)    # same basename: equal by design
    assert compare_snapshots(sa, sb).ok
    sb.texture[0] = "/x/OtherLeaf.png"
    assert "texture" in compare_snapshots(sa, sb).first_divergence


def test_relative_paths_match_helios(monkeypatch):
    """Relative texture strings resolve against the cwd in both, and are reported verbatim (tile, tube)."""
    pytest.importorskip("pyhelios")
    from heliades.parity.backends import ForgeBackend, HeliosBackend
    from heliades.parity.compare import compare_snapshots
    monkeypatch.chdir(_PA)
    snaps = []
    for B in (HeliosBackend(), ForgeBackend()):
        B.add_tile_textured((0, 0, 0), (1, 1), (0.2, 0.3), (3, 2), "TomatoLeaf.png", (2, 1))
        B.add_tube_textured(6, [(0, 0, 0), (0, 0, 1), (0, 0.1, 2)], [0.1, 0.1, 0.05], "CherryTomatoLeaf.png")
        snaps.append(B.snapshot())
    h, f = snaps
    assert set(h.texture) == set(f.texture) == {"TomatoLeaf.png", "CherryTomatoLeaf.png"}
    r = compare_snapshots(f, h)
    assert r.ok, str(r)
