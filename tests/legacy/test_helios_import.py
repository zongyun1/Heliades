import numpy as np
import pytest

pytest.importorskip("pyhelios")

from planttwin.asset import PlantAsset  # noqa: E402
from planttwin.helios_import import build_helios_plant  # noqa: E402
from planttwin.sim import PlantModel  # noqa: E402


@pytest.mark.parametrize("species,age", [("tomato", 75), ("capsicum", 80), ("maize", 50), ("strawberry", 60)])
def test_conversion_is_simulation_ready(species, age, tmp_path):
    asset, render = build_helios_plant(species, age, seed=0)
    st = asset.stem
    # one rooted tree
    assert (st.parent() < 0).sum() == 1
    assert len(st.edges) == len(st.nodes) - 1
    # no very short segments (they make rod / attachment constraints creep)
    L = np.linalg.norm(st.nodes[st.edges[:, 1]] - st.nodes[st.edges[:, 0]], axis=1)
    assert L.min() >= 4.9e-3
    # every Helios leaflet is kept, and every sim leaf mesh is one connected piece
    assert len(asset.leaves) == len(render["leaf"])
    for leaf in asset.leaves:
        seen, stack = {leaf.attach_vertex}, [leaf.attach_vertex]
        while stack:
            v = stack.pop()
            for f in leaf.faces[(leaf.faces == v).any(1)]:
                for w in f:
                    if w not in seen:
                        seen.add(w); stack.append(w)
        assert len(seen) == len(leaf.vertices)
    # fruit mass is lumped on its node
    for f in asset.fruits:
        assert st.point_mass[f.node] == pytest.approx(f.mass)
    # round-trips and builds a model
    asset.save(tmp_path / "a.json")
    PlantModel(PlantAsset.load(tmp_path / "a.json"))
