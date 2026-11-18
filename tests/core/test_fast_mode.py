# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fast mode (``Scene(mode="fast")``): deferred transforms and tube rebuilds against exact mode."""
from __future__ import annotations

import numpy as np
import pytest

from heliades.core.fast import FastScene
from heliades.core.scene import Scene

f32 = np.float32


def _close(a, b, tol=2e-6):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))                  # e.g. scaleTubeGirth of a collapsed ring
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    assert np.abs(a - b).max(initial=0) <= tol * max(1.0, np.abs(a).max(initial=0))


def _same_scene(A, B, tol=2e-6):
    sa, sb = A.snapshot(True), B.snapshot(True)
    assert (sa.uuid == sb.uuid).all() and (sa.type == sb.type).all() and (sa.parent == sb.parent).all()
    _close(sa.vertices, sb.vertices, tol)
    _close(sa.area, sb.area, tol)
    assert np.array_equal(sa.objects.oid, sb.objects.oid)
    _close(sa.objects.transform, sb.objects.transform, tol)


def test_mode_selection():
    assert type(Scene()) is Scene and Scene().mode == "exact"
    s = Scene(mode="fast")
    assert isinstance(s, FastScene) and s.mode == "fast"
    assert isinstance(FastScene(), FastScene)
    with pytest.raises(ValueError):
        Scene(mode="quick")


def _ops(S):
    """A mixed sequence of object edits and reads (same calls on both scenes)."""
    t = S.add_tile_object((0, 0, 0), (1, 2), (0.3, 0.2), (3, 2))
    tube = S.add_tube_object(6, [(0, 0, 0), (0, 0.1, 1), (0.2, 0.1, 2)], [0.1, 0.08, 0.05])
    S.translate_object([t, tube], (0.5, -0.25, 1.0))
    S.rotate_object(t, 0.7, "z")
    S.rotate_object([t], 0.3, (1, 1, 0), (0.2, 0.0, 0.1))
    S.scale_object_about_point(t, (1.5, 1.5, 0.5), (0.1, 0.2, 0.3))
    c = S.copy_object(t)                                         # copy of an object with pending transforms
    S.translate_object(c, (2, 0, 0))
    a1 = S.object_area(t)                                       # targeted read
    S.set_tube_nodes(tube, [(0, 0, 0), (0.1, 0.1, 1.1), (0.3, 0.1, 2.2)])
    S.set_tube_radii(tube, [0.12, 0.1, 0.0])                     # a zero radius collapses the ring
    S.translate_object(tube, (0, 1, 0))
    S.append_tube_segment(tube, (0.3, 1.2, 3.0), 0.04, (0.5, 0.2, 0.1))
    S.set_tube_radii(tube, [0.1, 0.09, 0.07, 0.03])               # the collapsed ring stays collapsed
    S.rotate_object(tube, 0.4, "x")                               # rings no longer frame rings
    S.set_tube_radii(tube, [0.1, 0.09, 0.08, 0.03])
    ct = S.copy_object(tube)
    S.scale_object(ct, (2, 2, 2))
    S.scale_tube_girth(ct, 1.5)
    S.prune_tube_nodes(tube, 3)
    p = S.primitives[S.objects[c].uuids[0]]
    v = p.vertices()                                             # proxy read: flushes the parent only
    S.delete_object(t)
    return a1, v, S.tube_object_triangle_vertices(ct), S.tube_object_nodes(ct), S.tube_object_node_radii(ct)


def test_object_and_tube_edits_match_exact():
    A, B = Scene(), Scene(mode="fast")
    ra, rb = _ops(A), _ops(B)
    for x, y in zip(ra, rb):
        _close(x, y)
    _same_scene(A, B)
    assert B.fast_stats["flush_objects"] > 0


def test_raw_store_read_flushes_everything():
    A, B = Scene(), Scene(mode="fast")
    for S in (A, B):
        o = S.add_tile_object((0, 0, 0), (1, 1), (0, 0), (2, 2))
        S.translate_object(o, (1, 2, 3))
        S.rotate_object(o, 0.5, "y")
    assert B._store._dirty
    _close(A._store.T[: A._store.n], B._store.T[: B._store.n])        # the property flushed
    assert not B._store._dirty and B._store.flush_reads == 1


def test_object_transforms_stay_on_the_exact_chain():
    """Object matrices (and the tube nodes / radii baked from them) are bit-identical to exact mode."""
    A, B = Scene(), Scene(mode="fast")
    for S in (A, B):
        o = S.add_tube_object(5, [(0, 0, 0), (0, 0, 1)], [0.1, 0.1])
        for k in range(20):
            S.translate_object(o, (0.01 * k, -0.003, 0.002))
            S.rotate_object(o, 0.01 * k, (0.3, 0.2, 1.0))
            S.scale_object_about_point(o, (1.01, 1.01, 1.01), (0.1, 0.0, 0.0))
        S.set_tube_radii(o, [0.2, 0.1])
    oa, ob = A.objects[1], B.objects[1]
    assert np.array_equal(oa.transform.view(np.uint32), ob.transform.view(np.uint32))
    assert np.array_equal(oa.extra["nodes"].view(np.uint32), ob.extra["nodes"].view(np.uint32))
    assert np.array_equal(oa.extra["radius"].view(np.uint32), ob.extra["radius"].view(np.uint32))
    _same_scene(A, B)


@pytest.mark.parametrize("model,age", [("tomato", 45), ("bean", 30), ("maize", 40)])
@pytest.mark.parametrize("seed", [1, 2])
def test_plant_fast_vs_exact(model, age, seed):
    from heliades.parity.fastmode import check
    r = check(model, age, seed)
    assert r["draws"], r
    assert r["topology"], r
    assert r["rel_dev"] < 1e-4, r


def test_moving_a_plant_in_fast_mode():
    from heliades.parity.fastmode import check
    r = check("tomato", 40, 3, move=(1.5, -2.0, 0.25), grow_after=8)
    assert r["draws"] and r["topology"], r
    assert r["rel_dev"] < 1e-4, r


def test_batch_builder_matches_in_process_build():
    from heliades.plantarchitecture.batch import PlantJob, build_plant, build_plants
    jobs = [PlantJob("bean", 20, seed=s) for s in (1, 2, 3)]
    res = build_plants(jobs, workers=2)
    assert [r.job for r in res] == jobs
    for r in res:
        sc, _, _ = build_plant("bean", 20, r.job.seed, "fast")
        assert r.primitives == sc.primitive_count() and r.objects == sc.object_count()
    assert len({r.pid for r in res}) >= 1


# ---- mutation checks: the comparison must catch a broken fast mode ----------------------------------------------

def test_mutation_dropped_flush_is_detected(monkeypatch):
    from heliades.core import fast as F
    from heliades.parity.fastmode import check
    monkeypatch.setattr(F.FastSceneMixin, "_apply_matrices", lambda self, groups, mats: None)
    r = check("tomato", 30, 1)
    assert r["draws"]
    assert not (r.get("rel_dev", 1.0) < 1e-4 and r["topology"]), r


def test_mutation_extra_draw_is_detected(monkeypatch):
    from heliades.core import fast as F
    from heliades.parity.fastmode import check
    orig = F.FastSceneMixin.translate_object

    def drawing(self, oids, shift):
        if not getattr(self, "_drew", False):
            self._drew = True
            self.rng.generator()
        return orig(self, oids, shift)
    monkeypatch.setattr(F.FastSceneMixin, "translate_object", drawing)
    r = check("tomato", 30, 1)
    assert not r["draws"], r


def test_repeated_and_missing_objects_in_a_call():
    A, B = Scene(), Scene(mode="fast")
    for S in (A, B):
        t = S.add_tile_object((0, 0, 0), (1, 1), (0, 0), (2, 1))
        u = S.add_tile_object((1, 0, 0), (1, 1), (0, 0), (1, 1))
        S.translate_object([t, u, t], (0.5, 0.25, -1.0))           # t moves twice, as in Helios
        S.rotate_object([u, u], 0.3, "z")
        with pytest.raises(ValueError):
            S.translate_object([u, 999], (1.0, 0.0, 0.0))          # u moves, then the missing object raises
    _same_scene(A, B)
