# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantLibrary.cpp:382-430, 511-518 and
# src/Assets.cpp:940-1020 (Copyright (C) 2016-2026 Brian Bailey).
"""Shared infrastructure of the library tree models (W5-D).

Library level (this module):

* the trained-tree build-parameter declarations (``tree_trunk_height``, ``tree_num_scaffolds``,
  ``tree_scaffold_angle``, ``walnut_scaffold_angle``) and their registration for almond, almond_independence,
  apple, pistachio and walnut (PL:383-386, 511-518);
* trunk internode lengths, ``trunkMaxNodes``, ``checkScaffoldsFitTrunk`` (PL:400-430);
* the training helpers of Assets.cpp shared by the trained trees (AS:940-1020): ``nodesToKeepForLength``,
  ``cutShootToLength``, ``releaseBudsBelowCut``, ``growSpursFromUnbrokenBuds``;
* :func:`kill_trunk_buds` (the "trunk built, apical meristem killed, its leaves and buds removed" block every
  tree builder repeats) and :func:`scaffold_yaw` (``(i + randu(-0.2, 0.2)) / n * 2 * M_PI``).

Engine level (``heliades.plantarchitecture.seasons``): the multi-year cycle (``Shoot::makeDormant``,
``makePlantDormant``, ``harvestPlant``), pruning (``pruneBranch``, deferred cuts, ``deletePhytomer``) and the
season-boundary self-pruning (shadow grid, ``shedShadedBranches``, ``updateGrowthPriority``).
"""
from __future__ import annotations

import math

import numpy as np

from ...migration_registry import ports
from ..params import BUD_ACTIVE, BUD_DEAD
from ..random_parameter import PlantArchitectureError
from .build_parameters import BuildParameterDeclaration, registerPlantModelBuildParameters

f32 = np.float32

# ---- build parameters (PL:383-386) ---------------------------------------------------------------------
tree_trunk_height = BuildParameterDeclaration("trunk_height", 0.1, 3.0, False, "total trunk height in meters")
tree_num_scaffolds = BuildParameterDeclaration("num_scaffolds", 2.0, 8.0, True, "number of scaffold branches")
tree_scaffold_angle = BuildParameterDeclaration("scaffold_angle", 20.0, 70.0, False, "scaffold branch angle in degrees")
walnut_scaffold_angle = BuildParameterDeclaration("scaffold_angle", 25.0, 60.0, False, "scaffold branch angle in degrees")

TREE_TRAINING_PARAMETERS = [tree_trunk_height, tree_num_scaffolds, tree_scaffold_angle]

# Internode length of the trunk the apple, pistachio and walnut builders lay out to trunk_height (PL:400-404).
apple_trunk_internode_length = f32(0.04)
pistachio_trunk_internode_length = f32(0.05)
walnut_trunk_internode_length = f32(0.04)
olive_trunk_internode_length = f32(0.03)
almond_trunk_internode_length = f32(0.03)


def register_build_parameters(pa):
    """PL:512-518 (the tree part). Apple is registered here too: it shares the declarations."""
    for name in ("almond", "almond_independence", "apple", "pistachio"):
        registerPlantModelBuildParameters(pa, name, list(TREE_TRAINING_PARAMETERS))
    registerPlantModelBuildParameters(pa, "walnut", [tree_trunk_height, tree_num_scaffolds, walnut_scaffold_angle])


@ports("trunkMaxNodes")
def trunkMaxNodes(trunk_height: BuildParameterDeclaration, trunk_internode_length) -> int:
    """PL:416-418: ``uint(std::ceil(max_value / internode_length)) + 1`` (float division, ceil of a float)."""
    return int(math.ceil(float(f32(trunk_height.max_value / f32(trunk_internode_length))))) + 1


@ports("checkScaffoldsFitTrunk")
def checkScaffoldsFitTrunk(builder: str, num_scaffolds: int, trunk_nodes: int, trunk_height):
    """PL:425-430."""
    if num_scaffolds > trunk_nodes + 1:
        raise PlantArchitectureError(
            f"ERROR (PlantArchitecture::{builder}): {num_scaffolds} scaffolds need {num_scaffolds} trunk nodes to attach "
            f"to, but a trunk_height of {float(f32(trunk_height)):f} m gives only {trunk_nodes + 1}. Increase trunk_height "
            "or reduce num_scaffolds.")


def trunk_node_count(trunk_height, internode_length) -> int:
    """``uint trunk_nodes = uint(trunk_height / internode_length); if (trunk_nodes < 1) trunk_nodes = 1;``"""
    n = int(f32(f32(trunk_height) / f32(internode_length)))
    return max(n, 1)


def scaffold_yaw(ctx, i: int, num_scaffolds: int) -> np.float32:
    """``(float(i) + randu(-0.2f, 0.2f)) / float(num_scaffolds) * 2 * M_PI`` (float, then the double M_PI),
    narrowed to the float argument of make_AxisRotation."""
    r = ctx.randu(f32(-0.2), f32(0.2))
    q = f32(f32(f32(i) + r) / f32(num_scaffolds))
    return f32(float(f32(q * f32(2))) * math.pi)


def kill_trunk_buds(pa, plantID: int, uID_trunk: int):
    """The block every tree builder runs after building the trunk: kill its apex, remove its leaves, kill its
    vegetative and (non-terminal) floral buds (e.g. PL:1200-1207)."""
    shoot = pa.plant_instances[plantID].shoot_tree[uID_trunk]
    shoot.meristem_is_alive = False
    for phytomer in list(shoot.phytomers):
        phytomer.removeLeaf()
        phytomer.setVegetativeBudState(BUD_DEAD)
        phytomer.setFloralBudState(BUD_DEAD)


def _no_shoot(vbud) -> bool:
    """``vbud.shoot_ID == uint(-1)``: the bud has not grown a shoot (the uint field may hold -1 or 2**32-1)."""
    return int(vbud.shoot_ID) in (-1, 0xFFFFFFFF)


# ---- training helpers (AS:940-1020) ----------------------------------------------------------------------

@ports("nodesToKeepForLength")
def nodesToKeepForLength(shoot, length) -> int:
    """AS:943-961: the first node whose far end reaches ``length`` of arc is kept."""
    from ...core import vec as V
    length = f32(length)
    arc = f32(0)
    first_vertex = True
    previous = None
    for node, segment in enumerate(shoot.shoot_internode_vertices):
        for vertex in segment:
            if not first_vertex:
                arc = f32(arc + V.magnitude(np.asarray(vertex, f32) - previous))
            previous = np.asarray(vertex, f32)
            first_vertex = False
        if arc >= length:
            return node + 1
    return len(shoot.shoot_internode_vertices)


@ports("cutShootToLength")
def cutShootToLength(shoot, length) -> int:
    """AS:965-972: cut back to about ``length`` (deferred inside a callback); the tip is stopped at once."""
    from ..seasons import pruneBranch
    keep = nodesToKeepForLength(shoot, length)
    if keep < len(shoot.phytomers):
        pruneBranch(shoot.plantarchitecture_ptr, shoot.plantID, shoot.ID, keep)
        shoot.meristem_is_alive = False
    return keep


@ports("releaseBudsBelowCut")
def releaseBudsBelowCut(shoot, keep: int, node_count: int, shoot_type_label: str):
    """AS:978-996: one bud on each of the top ``node_count`` nodes below ``keep`` (node 0 skipped) is set active
    with the given shoot type, unless it already grew a shoot."""
    released = 0
    node = int(keep) - 1
    while node >= 1 and released < node_count:
        phytomer = shoot.phytomers[node]
        released_here = False
        for petiole in phytomer.axillary_vegetative_buds:
            for vbud in petiole:
                if not released_here and _no_shoot(vbud):
                    vbud.shoot_type_label = shoot_type_label
                    vbud.state = BUD_ACTIVE                    # static Phytomer::setVegetativeBudState(state, vbud)
                    released_here = True
        released += 1
        node -= 1


@ports("growSpursFromUnbrokenBuds")
def growSpursFromUnbrokenBuds(phytomer, spur_fraction):
    """AS:1003-1017: a fraction of the dead, never-grown lateral buds of a non-dormant limb grow a ``spur``;
    every such bud is marked considered (its type set to ``spur``) whether or not it draws."""
    spur_fraction = f32(spur_fraction)
    shoot = phytomer.parent_shoot_ptr
    if spur_fraction <= 0 or shoot.isdormant or shoot.shoot_type_label == "spur" or int(phytomer.shoot_index[0]) == 0:
        return
    for petiole in phytomer.axillary_vegetative_buds:
        for vbud in petiole:
            if vbud.state == BUD_DEAD and _no_shoot(vbud) and vbud.shoot_type_label != "spur":
                vbud.shoot_type_label = "spur"
                if shoot.context_ptr.randu() < spur_fraction:
                    vbud.state = BUD_ACTIVE


__all__ = ["tree_trunk_height", "tree_num_scaffolds", "tree_scaffold_angle", "walnut_scaffold_angle",
           "TREE_TRAINING_PARAMETERS", "register_build_parameters", "trunkMaxNodes", "checkScaffoldsFitTrunk",
           "trunk_node_count", "scaffold_yaw", "kill_trunk_buds", "nodesToKeepForLength", "cutShootToLength",
           "releaseBudsBelowCut", "growSpursFromUnbrokenBuds"]
