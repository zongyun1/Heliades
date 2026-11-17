"""W2-A loadOBJ: stream semantics, number parsing, triangulation, materials, polymesh topology."""
import numpy as np
import pytest

from heliades.core import obj as O
from heliades.core.scene import Scene
from heliades.parity.suites.core.cases_fileio import TOMATO

f32 = np.float32


def _write(tmp_path, name, text):
    p = tmp_path / name
    with open(p, "w", newline="") as fh:
        fh.write(text)
    return str(p)


def test_cstream_failed_read_keeps_string():
    s = O.CStream("ab  \n")
    assert s.token("x") == "ab" and s.good()
    assert s.token("keep") == "keep" and s.fail          # only whitespace left: unchanged
    s = O.CStream("o")
    assert s.token("") == "o" and s.eof
    assert s.getline("o") == "o" and s.fail               # getline on an eof stream leaves it alone
    s = O.CStream("a\n")
    s.token("")
    assert s.getline("zz") == ""                           # good stream: erased, then the newline


def test_number_parsing_matches_stod_stoi():
    assert O.parse_float("0x1p-2") == (True, f32(0.25))
    assert O.parse_float("1e999")[0] is False             # ERANGE -> out_of_range
    assert O.parse_float("1e-400")[0] is False
    assert O.parse_float("1.5x")[0] is False
    assert O.parse_float(" -0.1 ")[1] == f32(float("-0.1"))
    assert O.parse_int("+5") == (True, 5)
    assert O.parse_int("5/")[0] is False
    assert O.parse_int("99999999999")[0] is False
    assert O.string2rgbcolor("0.1 0.2 0.3")[3] == f32(1)
    with pytest.raises(ValueError):
        O.string2rgbcolor("1 2 3 4 5")


def test_trailing_whitespace_rereads_last_reference(tmp_path):
    p = _write(tmp_path, "q.obj", "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4 \n")
    F = O.parse_obj(p, (0, 0, 1))
    assert F.face_inds["none"] == [[1, 2, 3, 4, 4]]
    S = Scene()
    uu = S.load_obj(p)
    assert len(uu) == 2                                     # the (1,4,4) fan triangle is skipped


def test_ngon_fan_and_polymesh(tmp_path):
    n = 32
    ang = np.arange(n) * 2 * np.pi / n
    txt = "".join(f"v {np.cos(a):.6f} {np.sin(a):.6f} 0\n" for a in ang)
    txt += "f " + " ".join(str(-k) for k in range(n, 0, -1)) + "\n"          # negative indices
    S = Scene()
    uu = S.load_obj(_write(tmp_path, "n.obj", txt), (0, 0, 0), None, (0, 0), (1, 0, 0))
    assert len(uu) == n - 2
    (oid,) = S.all_object_ids()
    assert S.polymesh_object_face_count(oid) == n - 2 and S.polymesh_object_vertex_count(oid) == n
    assert S.polymesh_object_vertex_normal_source(oid) == 2      # computed at 180 degrees
    assert [S.polymesh_object_primitive_uuid_for_face(oid, f) for f in range(n - 2)] == uu
    assert np.allclose(S.polymesh_object_vertex_normals(oid), [0, 0, 1])
    assert all(tuple(S.primitives[u].color[:3]) == (1, 0, 0) for u in uu)


def test_materials_sorted_and_renamed(tmp_path):
    _write(tmp_path, "m.mtl", "newmtl b\nKd 0 1 0\nnewmtl a\nKd 1 0 0\n")
    p = _write(tmp_path, "m.obj", "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl b\nf 1 2 3\nusemtl a\nf 1 3 2\n")
    S = Scene()
    u = S.load_obj(p)
    assert [S.get_primitive_material_label(x) for x in u] == ["a", "b"]           # std::map order, not file order
    u2 = S.load_obj(p)
    assert [S.get_primitive_material_label(x) for x in u2] == ["a_1", "b_1"]


def test_tomato_flower_counts():
    S = Scene()
    uu = S.load_obj(TOMATO[0], (0, 0, 0), 0.75, (0, 0), (0, 0, 0), "ZUP")
    assert len(uu) == 433
    (oid,) = S.all_object_ids()
    assert S.polymesh_object_vertex_count(oid) == 295 and S.does_polymesh_object_have_vertex_normals(oid)
    z = np.concatenate([S.primitive_vertices(u)[:, 2] for u in uu])
    assert np.isclose(z.max() - z.min(), 0.75, atol=1e-6)


def test_copy_object_keeps_topology_on_new_primitives():
    S = Scene()
    uu = S.load_obj(TOMATO[0], (0, 0, 0), 0.75, (0, 0), (0, 0, 0), "ZUP")
    (oid,) = S.all_object_ids()
    new = S.copy_object(oid)
    S.translate_object(new, (1, 0, 0))
    nu = S.object_primitive_uuids(new)
    assert S.polymesh_object_primitive_uuid_for_face(new, 0) == nu[0] != uu[0]
    d = S.polymesh_object_vertices(new) - S.polymesh_object_vertices(oid)
    assert np.allclose(d, [1, 0, 0], atol=1e-6)


def test_errors(tmp_path):
    S = Scene()
    with pytest.raises(ValueError):
        S.load_obj(_write(tmp_path, "x.txt", "v 0 0 0\n"))
    with pytest.raises(ValueError):
        S.load_obj(_write(tmp_path, "b.obj", "v 0 0 0\nf 1 2 3\n"))
    with pytest.raises(ValueError):
        S.load_obj(_write(tmp_path, "c.obj", "v 0 0 0\n"), upaxis="UP")


def test_parse_float_rounds_through_double():
    """helios::parse_float is (float)std::stod: double rounding, not a direct decimal->float32."""
    from heliades.parity.suites.core.cases_fileio import double_rounding_trap
    s = double_rounding_trap(0.3)
    ok, v = O.parse_float(s)
    assert ok and v == f32(float(s))
    assert v.view(np.int32) % 2 == 0                      # tie went to the even float, below the decimal
    assert float(v) < float(s)


def test_harness_catches_one_ulp_in_topology():
    from heliades.parity.backends import ForgeBackend
    from heliades.parity.compare import compare_streams
    a, b = ForgeBackend(), ForgeBackend()
    for B in (a, b):
        B.load_obj(TOMATO[0], (0.0, 0.0, 0.0), 0.75, (0.0, 0.0), (0.0, 0.0, 0.0), upaxis="ZUP", overload="height_up")
    o = b.scene.all_object_ids()[0]
    t = b.scene.objects[o].extra["topology"]
    t["normals"][7, 1] = np.nextafter(t["normals"][7, 1], f32(2))
    assert not compare_streams(a.obj_queries(), b.obj_queries()).ok
