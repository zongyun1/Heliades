"""The batched (structure-of-arrays) kernels of heliades.core.store must agree bit for bit with the scalar,
op-for-op ports they replace (docs/core/PERFORMANCE.md)."""
import numpy as np

from heliades.core import objects as O
from heliades.core import store as ST
from heliades.core.primitives import Primitive, make_rotation_matrix, matmult, triangle_transform
from heliades.core.scene import Scene

f32 = np.float32


def _bits(a):
    return np.ascontiguousarray(np.asarray(a, f32)).view(np.uint32)


def _rand(rng, *shape, special=True):
    x = (rng.standard_normal(shape) * rng.choice([1e-3, 1.0, 1e3], shape)).astype(f32)
    if special:
        flat = x.reshape(-1)
        k = max(1, flat.size // 50)
        flat[rng.integers(0, flat.size, k)] = 0.0
        flat[rng.integers(0, flat.size, k)] = -0.0
    return x


def test_matmult_rows_matches_matmult_small_and_large():
    rng = np.random.default_rng(1)
    for n in (1, 7, 300, 2000):
        R = _rand(rng, n, 16)
        L = _rand(rng, 16)
        got = ST.matmult_rows(L, R)
        ref = np.stack([matmult(L, r) for r in R])
        assert np.array_equal(_bits(got), _bits(ref))
        L2 = _rand(rng, n, 16)
        got2 = ST.matmult_rows(L2, R)
        ref2 = np.stack([matmult(a, b) for a, b in zip(L2, R)])
        assert np.array_equal(_bits(got2), _bits(ref2))


def test_triangle_transform_rows_matches_scalar():
    rng = np.random.default_rng(2)
    v0, v1, v2 = (_rand(rng, 500, 3) for _ in range(3))
    got = ST.triangle_transform_rows(v0, v1, v2)
    ref = np.stack([triangle_transform(a, b, c) for a, b, c in zip(v0, v1, v2)])
    assert np.array_equal(_bits(got), _bits(ref))


def test_row_getters_match_primitive_methods():
    rng = np.random.default_rng(3)
    S = Scene()
    for i in range(200):
        if i % 2:
            S.add_patch(_rand(rng, 3, special=False), np.abs(_rand(rng, 2, special=False)) + f32(0.01),
                        tuple(rng.uniform(-3, 3, 2)), (0.1, 0.2, 0.3))
        else:
            S.add_triangle(*(_rand(rng, 3) for _ in range(3)), (0.5, 0.5, 0.5))
    st = S._store
    ix = np.arange(st.n)
    V_, nv = st.vertices_rows(ix)
    A, N = st.areas_rows(ix), st.normals_rows(ix)
    for u in range(st.n):
        p = Primitive(st, u)
        assert np.array_equal(_bits(V_[u, :nv[u]]), _bits(p.vertices()))
        assert _bits(A[u]) == _bits(p.area())
        with np.errstate(all="ignore"):
            assert np.array_equal(_bits(N[u]), _bits(p.normal()))


def test_translate_and_left_multiply_rows_match_per_primitive():
    rng = np.random.default_rng(4)
    st = ST.PrimitiveStore()
    for u in range(64):
        st.add(u, ST.TRIANGLE, _rand(rng, 16) if u % 9 == 0 else matmult(_rand(rng, 16) * 0 + np.eye(4, dtype=f32).reshape(16),
                                                                         triangle_transform(*(_rand(rng, 3) for _ in range(3)))),
               (0, 0, 0, 1))
    T0 = st.T[:64].copy()
    s = _rand(rng, 3)
    idx = np.array([3, 9, 10, 40, 18, 27], np.int64)       # some rows are not affine (fall back on T*M)
    st.translate_rows(idx, s)
    st.translate_rows(slice(0, 64), s)
    Tt = np.eye(4, dtype=f32).reshape(16).copy()
    Tt[3], Tt[7], Tt[11] = s
    for u in range(64):
        ref = T0[u]
        for _ in range(2 if u in idx else 1):
            aff = ref[12] == 0 and ref[13] == 0 and ref[14] == 0 and ref[15] == 1
            if aff:
                ref = ref.copy()
                ref[3], ref[7], ref[11] = ref[3] + s[0], ref[7] + s[1], ref[11] + s[2]
            else:
                ref = matmult(Tt, ref)
        assert np.array_equal(_bits(st.T[u]), _bits(ref)), u
    R = make_rotation_matrix(0.7, "y")
    T1 = st.T[:64].copy()
    st.left_multiply_rows(idx, R)
    for u in range(64):
        ref = matmult(R, T1[u]) if u in idx else T1[u]
        assert np.array_equal(_bits(st.T[u]), _bits(ref))


def test_tube_frames_match_scalar_reference():
    rng = np.random.default_rng(5)
    cases = []
    for _ in range(60):
        n = int(rng.integers(2, 12))
        P = np.cumsum(rng.standard_normal((n, 3)).astype(f32) * f32(0.05), axis=0).astype(f32)
        cases.append(P)
    straight = np.array([[0, 0, z] for z in range(6)], f32)                # no turn: fallback frames
    kinked = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 0]], f32)   # repeats, reversals
    back = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1e-7, 1]], f32)
    cases += [straight, kinked, back, straight[:2]]
    for P in cases:
        nodes = [p for p in P]
        with np.errstate(all="ignore"):
            got = O._tube_frames(nodes)
            ref = O._tube_frames_scalar(nodes)
        for (r1, o1), (r2, o2) in zip(got, ref):
            assert np.array_equal(_bits(r1), _bits(r2)) and np.array_equal(_bits(o1), _bits(o2))


def test_add_triangles_equals_single_calls():
    rng = np.random.default_rng(6)
    v = [_rand(rng, 40, 3) for _ in range(3)]
    A, B = Scene(), Scene()
    a = [A.add_triangle(v[0][i], v[1][i], v[2][i], (0.2, 0.4, 0.6)) for i in range(40)]
    b = B.add_triangles(v[0], v[1], v[2], (0.2, 0.4, 0.6))
    assert a == b
    assert np.array_equal(_bits(A._store.T[:40]), _bits(B._store.T[:40]))
    assert [A.get_primitive_material_label(u) for u in a] == B.get_primitive_material_labels(b)
    assert list(A.primitive_order) == list(B.primitive_order)
    assert {m: x.reference_count for m, x in A.materials.items()} == {m: x.reference_count for m, x in B.materials.items()}


def test_primitive_map_behaves_like_the_dict_it_replaced():
    S = Scene()
    u = S.add_patch((0, 0, 0), (1, 1))
    t = S.add_triangle((0, 0, 0), (1, 0, 0), (0, 1, 0))
    assert list(S.primitives) == [u, t] and len(S.primitives) == 2 and t in S.primitives and 99 not in S.primitives
    assert S.primitives.get(99) is None and S.primitives[np.int64(t)].uuid == t
    S.delete_primitive(u)
    assert list(S.primitives) == [t] and u not in S.primitives and S.primitive_count() == 1
