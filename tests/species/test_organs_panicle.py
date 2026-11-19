# SPDX-License-Identifier: LGPL-2.1-or-later
"""The procedural ``panicle`` fruit shape (heliades.species.organs, S1: oat)."""
from __future__ import annotations

import numpy as np
import pytest

from heliades.species import organs as O

SPEC = {"panicle": {"whorls": 5, "branches_base": 4, "branches_tip": 2, "color": [0.6, 0.7, 0.4],
                    "rachis_color": [0.5, 0.55, 0.3]}}


def test_panicle_validation():
    p = O.validate_organ(SPEC, "fruit", "f")
    assert p[0] == "panicle" and p[1]["droop"] == 40.0 and p[1]["seed"] == 1
    with pytest.raises(O.OrganSpecError, match="fruit"):
        O.validate_organ(SPEC, "flower", "f")
    with pytest.raises(O.OrganSpecError, match="unknown key"):
        O.validate_organ({"panicle": {**SPEC["panicle"], "awns": 1}}, "fruit", "f")
    with pytest.raises(O.OrganSpecError, match="required"):
        O.validate_organ({"panicle": {"whorls": 3}}, "fruit", "f")
    with pytest.raises(O.OrganSpecError, match="outside"):
        O.validate_organ({"panicle": {**SPEC["panicle"], "whorls": 40}}, "fruit", "f")


def test_panicle_geometry_is_deterministic_and_unit_length():
    params = O.validate_organ(SPEC, "fruit", "f")[1]
    V1, T1, C1 = O.panicle_mesh(params)
    V2, T2, C2 = O.panicle_mesh(params)
    assert np.array_equal(V1, V2) and T1 == T2
    assert len(T1) == len(C1) and max(max(t) for t in T1) < len(V1)
    assert V1[:, 0].min() > -0.05 and 0.9 < V1[:, 0].max() < 1.2        # base at the origin, about 1 long along +x
    # spikelets: 4 triangles each; basal whorl 4 branches x 5, top whorl 2 branches, plus the terminal one
    spikelet_tris = sum(1 for c in C1 if c == tuple(params["color"]))
    assert spikelet_tris % 4 == 0 and spikelet_tris // 4 > 30
    other = O.panicle_mesh({**params, "seed": 2})[0]
    assert other.shape != V1.shape or not np.array_equal(other, V1)          # the local seed changes the jitter


def test_panicle_hook_draws_nothing_from_the_context():
    from heliades.core.scene import Scene
    h = O.organ_hook("t", "fruit", O.validate_organ(SPEC, "fruit", "f"))
    a, b = Scene(), Scene()
    a.seed_random_generator(3)
    b.seed_random_generator(3)
    oid = h(a, 1)
    assert len(a.object_primitive_uuids(oid)) == len(O.panicle_mesh(O.validate_organ(SPEC, "fruit", "f")[1])[1])
    assert a.randu() == b.randu()
