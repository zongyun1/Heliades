# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for ``PlantArchitecture::setPlantBasePosition`` (moving a grown plant), 0 float32 ulp.

A plant is grown at the origin, moved, and grown on, so every stored position that later growth builds on must have
moved with it (shoot bases, internode / petiole / peduncle nodes, leaf and fruit bases), and for trees the shadow grid
(``shadow_grid_origin``) and the season-peak leaf positions that drive growth priority and shade shedding.

* ``plantarch.move.tomato``: tomato to 60 d, moved, on to 110 d (flowers and fruit after the move), moved again;
* ``plantarch.move.almond`` (slow): almond to 300 d, moved, on to 740 d (two season boundaries: growth priority at
  365 d and 730 d, the first shade shedding at 730 d), all read through the moved shadow grid;
* ``plantarch.move.errors`` (exact): moving a plant that does not exist is rejected by both.

Each observation is the engine-call probe, per shoot its node count and stored nodes and radii, the plant XML and the
whole hidden-inclusive snapshot (see ``cases_nut_trees.observe``).
"""
from __future__ import annotations

from ...cases import case
from .cases_nut_trees import _start, observe


@case("plantarch.move.tomato", tier="ulp", max_ulp=0, fuzz=4)
def move_tomato(B, s):
    seed = _start(B, "tomato", s, 60.0)
    B.pa_set_base_position(0.37, -1.21, 0.05)
    out = observe(B, seed)
    B.pa_advance(50.0)
    out += [-300.0] + observe(B, seed)
    B.pa_set_base_position(-2.5, 0.75, 0.0)
    return out + [-301.0] + observe(B, seed)


@case("plantarch.move.almond", tier="ulp", max_ulp=0, fuzz=2)
def move_almond(B, s):
    seed = _start(B, "almond", s, 300.0)
    B.pa_set_base_position(1.53, -0.71, 0.0)
    B.pa_advance(440.0)
    return observe(B, seed)


@case("plantarch.move.errors", tier="exact", fuzz=1)
def move_errors(B, s):
    _start(B, "tomato", s, 0.0)
    return [B.pa_set_base_position_error(B.pid + 7)]
