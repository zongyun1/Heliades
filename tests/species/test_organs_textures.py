# SPDX-License-Identifier: LGPL-2.1-or-later
"""Organ prototypes (heliades.species.organs) and procedural leaf textures (heliades.species.leaf_texture)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from heliades.species import leaf_texture as LT
from heliades.species import organs as O


def _scene():
    from heliades.core.scene import Scene
    sc = Scene()
    sc.seed_random_generator(7)
    return sc


def _obj_vertices(sc, oid):
    uu = sc.object_primitive_uuids(oid)
    V, n = sc.primitive_vertices_rows(uu)
    return np.concatenate([V[i, :n[i]] for i in range(len(n))])


def test_obj_variants_reproduce_the_cucumber_prototypes():
    """``{obj_variants: Cucumber/fruit_{index:02d}.obj, count: 6}`` is the cucumber's fruit prototype: the same draw and
    the same mesh."""
    from heliades.plantarchitecture.library import cucumber
    fruit = O.organ_hook("t", "fruit", O.validate_organ({"obj_variants": "assets/obj/Cucumber/fruit_{index:02d}.obj",
                                                         "count": 6}, "fruit", "f"))
    flower = O.organ_hook("t", "flower", O.validate_organ(
        {"obj_variants": "assets/obj/Cucumber/flower_{state}_{index:02d}.obj", "count": 6}, "flower", "f"))
    for k in range(4):
        a, b = _scene(), _scene()
        for _ in range(k):
            a.randu(), b.randu()
        oa, ob = cucumber.CucumberFruitPrototype(a, 1), fruit(b, 1)
        assert np.array_equal(_obj_vertices(a, oa), _obj_vertices(b, ob))
        oa, ob = cucumber.CucumberFlowerPrototype(a, 1, k % 2 == 0), flower(b, 1, k % 2 == 0)
        assert np.array_equal(_obj_vertices(a, oa), _obj_vertices(b, ob))
        assert a.randu() == b.randu()


def test_procedural_fruit_and_flower():
    p = O.validate_organ({"ellipsoid": {"diameter": 0.4, "bulge": 0.2, "color": [0.2, 0.0, 0.2],
                                        "calyx_color": [0.2, 0.5, 0.2], "calyx_fraction": 0.1}}, "fruit", "f")
    sc = _scene()
    h = O.organ_hook("t", "fruit", p)
    assert h is O.organ_hook("t", "fruit", p)            # one function pointer per description
    oid = h(sc, 1)
    V = _obj_vertices(sc, oid)
    assert V[:, 0].min() == pytest.approx(0) and V[:, 0].max() == pytest.approx(1)
    assert 0.35 < 2 * np.abs(V[:, 1]).max() < 0.5          # diameter 0.4, widened towards the tip by the bulge
    assert sc.randu() == _scene().randu()                 # no draws
    f = O.organ_hook("t", "flower", O.validate_organ({"corolla": {"petals": 5, "color": [0.5, 0.3, 0.8],
                                                                  "center_color": [1, 1, 0]}}, "flower", "f"))
    for is_open in (True, False):
        oid = f(sc, 1, is_open)
        V = _obj_vertices(sc, oid)
        span = 2 * np.abs(V[:, 1:]).max()
        assert (span > 0.9) if is_open else (span < 0.2)


def test_organ_validation():
    assert O.validate_organ("template", "fruit", "f") == ("template",)
    assert O.validate_organ("none", "fruit", "f") == ("none",)
    with pytest.raises(O.OrganSpecError, match="exactly one"):
        O.validate_organ({"obj": "a.obj", "library": "X"}, "fruit", "f")
    with pytest.raises(O.OrganSpecError, match="outside"):
        O.validate_organ({"ellipsoid": {"diameter": 9.0, "color": [0, 0, 0]}}, "fruit", "f")


def _with_textures():
    from heliades.species import available_species, load_spec
    return [n for n in available_species() if load_spec(n).leaf_textures]


@pytest.mark.parametrize("name", _with_textures())
def test_committed_textures_are_regenerated_exactly(name):
    """The PNGs under heliades/species/assets are the tool's output for the species file's description."""
    from heliades.species import load_spec
    from heliades.species.assets import leaf_texture_path
    spec = load_spec(name)
    assert spec.leaf_textures
    for tex, desc in spec.leaf_textures.items():
        path = leaf_texture_path(name, tex)
        assert os.path.exists(path)
        assert np.array_equal(LT.read_png_rgba(path), LT.render(desc))


def test_texture_tool(tmp_path):
    d = {"aspect": 0.5, "length_px": 200, "widest_at": 0.4, "lobes": {"count": 3, "depth": 0.2},
         "serration": {"count": 20, "depth": 0.05}, "veins": 5}
    img = LT.render(d)
    assert img.shape == (100, 200, 4) and img.dtype == np.uint8
    assert set(np.unique(img[..., 3])) == {0, 255}
    a = img[..., 3] > 0
    assert not a[:, 0].any() and not a[:, -1].any()              # zero width at the base and the tip
    mid = a[:, int(0.4 * 200)]
    assert mid.sum() > 0.8 * 100                                 # nearly full width at the widest point
    assert np.array_equal(img, LT.render(dict(d)))               # deterministic
    p = LT.write_png(str(tmp_path / "leaf.png"), img)
    assert np.array_equal(LT.read_png_rgba(p), img)
    from heliades.core.texture import read_png_alpha
    assert np.array_equal(read_png_alpha(p), img[..., 3] >= 250)  # what Helios's loader sees as solid
    with pytest.raises(LT.LeafOutlineError, match="aspect"):
        LT.validate_outline({"length_px": 64})
    with pytest.raises(LT.LeafOutlineError, match="unknown key"):
        LT.validate_outline({"aspect": 0.3, "lobe": 3})
    prof = {"aspect": 0.3, "profile": [[0, 0], [0.3, 1], [1, 0]]}
    t = np.array([0.0, 0.15, 0.3, 0.65, 1.0])
    assert np.allclose(LT.profile(LT.validate_outline(prof), t), [0, 0.5, 1, 0.5, 0])
