# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp (Copyright (C) 2016-2026 Brian Bailey):
#   Shoot::makeDormant                          PA:2264-2294
#   PlantArchitecture::harvestPlant             PA:9980-9996
#   PlantArchitecture::makePlantDormant         PA:10058-10067
"""The multi-year cycle of perennial plants (W5-D): dormancy, pruning and season-boundary self-pruning.

Everything a tree reaches that the herbaceous crops did not:

* dormancy: :func:`makeDormant` (``Shoot::makeDormant``), :func:`makePlantDormant`, :func:`harvestPlant`;
"""
from __future__ import annotations

import math

import numpy as np

from ..migration_registry import ports
from .params import BUD_DEAD, BUD_DORMANT
from .random_parameter import PlantArchitectureError

f32 = np.float32


# ---- dormancy ---------------------------------------------------------------------------------------------

@ports("Shoot::makeDormant")
def makeDormant(shoot):
    """PA:2264-2294: every bud that is not dormant dies, leaves fall (unless evergreen), every phytomer is dormant,
    and a living apex of a type whose flowers require dormancy gets its terminal floral buds."""
    pa = shoot.plantarchitecture_ptr
    shoot.isdormant = True
    shoot.dormancy_cycles += 1
    shoot.nodes_this_season = 0
    evergreen = pa.plant_instances[shoot.plantID].is_evergreen
    for phytomer in shoot.phytomers:
        for petiole in phytomer.floral_buds:
            for fbud in petiole:
                if fbud.state != BUD_DORMANT:
                    phytomer.setFloralBudState(BUD_DEAD, fbud)
        for petiole in phytomer.axillary_vegetative_buds:
            for vbud in petiole:
                if vbud.state != BUD_DORMANT:
                    phytomer.setVegetativeBudState(BUD_DEAD, vbud)
        if not evergreen:
            phytomer.removeLeaf()
        phytomer.isdormant = True
    sp = shoot.shoot_parameters
    if shoot.meristem_is_alive and sp.flowers_require_dormancy and sp.max_terminal_floral_buds.val() > 0:
        shoot.addTerminalFloralBud()


@ports("PlantArchitecture::makePlantDormant")
def makePlantDormant(pa, plantID):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::makePlantDormant): Plant with ID of {plantID} does not "
                                     "exist.")
    for shoot in pa.plant_instances[plantID].shoot_tree:
        makeDormant(shoot)
    pa.plant_instances[plantID].time_since_dormancy = f32(0)


@ports("PlantArchitecture::harvestPlant")
def harvestPlant(pa, plantID):
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::harvestPlant): Plant with ID of {plantID} does not "
                                     "exist.")
    for shoot in pa.plant_instances[plantID].shoot_tree:
        for phytomer in shoot.phytomers:
            for petiole in phytomer.floral_buds:
                for fbud in petiole:
                    if fbud.state != BUD_DORMANT:
                        phytomer.setFloralBudState(BUD_DEAD, fbud)


# ---- libstdc++ std::sort -------------------------------------------------------------------------------------

def std_sort(seq, less):
    """``std::sort(first, last, comp)`` exactly as libstdc++ implements it (introsort with a threshold of 16,
    median-of-three pivot, heap sort at the depth limit, final insertion sort). std::sort is not stable, so a
    sort with ties (shoots of equal rank, siblings in the same voxel) must reproduce its element moves to put the
    tied elements in the C++ order. Returns a new list."""
    a = list(seq)
    n = len(a)
    if n < 2:
        return a

    def adjust_heap(first, hole, length, value):
        top = hole
        second = hole
        while second < (length - 1) // 2:
            second = 2 * (second + 1)
            if less(a[first + second], a[first + second - 1]):
                second -= 1
            a[first + hole] = a[first + second]
            hole = second
        if (length & 1) == 0 and second == (length - 2) // 2:
            second = 2 * (second + 1)
            a[first + hole] = a[first + second - 1]
            hole = second - 1
        parent = (hole - 1) // 2
        while hole > top and less(a[first + parent], value):
            a[first + hole] = a[first + parent]
            hole = parent
            parent = (hole - 1) // 2
        a[first + hole] = value

    def heap_sort(first, last):
        length = last - first
        if length >= 2:
            parent = (length - 2) // 2
            while True:
                adjust_heap(first, parent, length, a[first + parent])
                if parent == 0:
                    break
                parent -= 1
        while last - first > 1:
            last -= 1
            value = a[last]
            a[last] = a[first]
            adjust_heap(first, 0, last - first, value)

    def move_median_to_first(result, i, j, k):
        if less(a[i], a[j]):
            if less(a[j], a[k]):
                a[result], a[j] = a[j], a[result]
            elif less(a[i], a[k]):
                a[result], a[k] = a[k], a[result]
            else:
                a[result], a[i] = a[i], a[result]
        elif less(a[i], a[k]):
            a[result], a[i] = a[i], a[result]
        elif less(a[j], a[k]):
            a[result], a[k] = a[k], a[result]
        else:
            a[result], a[j] = a[j], a[result]

    def unguarded_partition(first, last, pivot):
        while True:
            while less(a[first], a[pivot]):
                first += 1
            last -= 1
            while less(a[pivot], a[last]):
                last -= 1
            if not first < last:
                return first
            a[first], a[last] = a[last], a[first]
            first += 1

    def introsort_loop(first, last, depth):
        while last - first > 16:
            if depth == 0:
                heap_sort(first, last)
                return
            depth -= 1
            mid = first + (last - first) // 2
            move_median_to_first(first, first + 1, mid, last - 1)
            cut = unguarded_partition(first + 1, last, first)
            introsort_loop(cut, last, depth)
            last = cut

    def unguarded_linear_insert(last):
        val = a[last]
        nxt = last - 1
        while less(val, a[nxt]):
            a[last] = a[nxt]
            last = nxt
            nxt -= 1
        a[last] = val

    def insertion_sort(first, last):
        if first == last:
            return
        for i in range(first + 1, last):
            if less(a[i], a[first]):
                val = a[i]
                a[first + 1:i + 1] = a[first:i]
                a[first] = val
            else:
                unguarded_linear_insert(i)

    introsort_loop(0, n, (n.bit_length() - 1) * 2)
    if n > 16:
        insertion_sort(0, 16)
        for i in range(16, n):
            unguarded_linear_insert(i)
    else:
        insertion_sort(0, n)
    return a


# ---- pruning (PA:6805-6930, 10123-10200) ----------------------------------------------------------------------

@ports("Phytomer::deletePhytomer")
def deletePhytomer(phytomer):
    """PA:6809-6929 (no detached material): remove this phytomer and every one above it on its shoot, with their
    leaves, inflorescences and child shoots; prune the internode tube."""
    from .tube_utils import NO_INTERNODE_TUBE_OBJID
    pa = phytomer.plantarchitecture_ptr
    ctx = phytomer.context_ptr
    pa.invalidateOrganIndex()
    if pa.phytomer_callback_depth > 0:
        return                                          # skipStructuralOperationInCallback
    shoot = phytomer.parent_shoot_ptr
    first_deleted_node = int(phytomer.shoot_index[0])
    node_count = int(phytomer.shoot_index[1])
    if ctx.does_object_exist(shoot.internode_tube_objID):
        tube_nodes = ctx.tube_object_node_count(shoot.internode_tube_objID)
        tube_segments = int(shoot.shoot_parameters.phytomer_parameters.internode.length_segments)
        tube_prune_index = 0 if first_deleted_node == 0 else first_deleted_node * tube_segments + 1
        if tube_prune_index < tube_nodes:
            ctx.prune_tube_nodes(shoot.internode_tube_objID, tube_prune_index)
        shoot.terminateApicalBud()
    tree = pa.plant_instances[phytomer.plantID].shoot_tree
    for node in range(first_deleted_node, node_count):
        ph = shoot.phytomers[node]
        ph.removeLeaf()
        for petiole in ph.floral_buds:
            for fbud in petiole:
                for p in range(len(fbud.inflorescence_objIDs) - 1, -1, -1):
                    ctx.delete_object([fbud.inflorescence_objIDs[p]])
                    del fbud.inflorescence_objIDs[p]
                    del fbud.inflorescence_bases[p]
                    if p < len(fbud.inflorescence_rotation):
                        del fbud.inflorescence_rotation[p]
                    if p < len(fbud.inflorescence_base_scales):
                        del fbud.inflorescence_base_scales[p]
                if fbud.peduncle_objIDs:
                    ctx.delete_object(list(fbud.peduncle_objIDs))
                    ctx.delete_object(list(fbud.inflorescence_objIDs))
                    fbud.peduncle_objIDs = []
                    fbud.inflorescence_objIDs = []
                    fbud.inflorescence_bases = []
                    fbud.inflorescence_rotation = []
                    fbud.inflorescence_base_scales = []
        for childID in list(shoot.childIDs.get(node, [])):
            child = tree[childID]
            if not child.isPruned():
                deletePhytomer(child.phytomers[0])
    del shoot.shoot_internode_radii[first_deleted_node:]
    del shoot.shoot_internode_vertices[first_deleted_node:]
    del shoot.phytomers[first_deleted_node:]
    for key in [k for k in shoot.childIDs if k >= first_deleted_node]:
        del shoot.childIDs[key]
    if not ctx.does_object_exist(shoot.internode_tube_objID):
        shoot.internode_tube_objID = NO_INTERNODE_TUBE_OBJID
    if not shoot.phytomers:
        shoot.terminateApicalBud()
    for ph in shoot.phytomers:
        ph.shoot_index[1] = len(shoot.phytomers)
    shoot.current_node_number = len(shoot.phytomers)


@ports("PlantArchitecture::applyPruneBranch")
def applyPruneBranch(pa, plantID, shootID, node_index):
    tree = pa.plant_instances[plantID].shoot_tree
    shoot = tree[shootID]
    pa.invalidateOrganIndex()
    deletePhytomer(shoot.phytomers[node_index])
    if shoot.isPruned() and shoot.parent_shoot_ID >= 0:
        parent_childIDs = tree[shoot.parent_shoot_ID].childIDs
        for key in list(parent_childIDs):
            siblings = [c for c in parent_childIDs[key] if c != shootID]
            if siblings:
                parent_childIDs[key] = siblings
            else:
                del parent_childIDs[key]


@ports("PlantArchitecture::pruneBranch")
def pruneBranch(pa, plantID, shootID, node_index, keep_trimmed_material=False):
    """PA:10159-10200. Inside a phytomer callback the cut is queued (``deferred_prunes``) and applied by
    :func:`flushDeferredPrunes` at the end of the time step."""
    if plantID not in pa.plant_instances:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::pruneBranch): Plant with ID of {plantID} does not exist.")
    tree = pa.plant_instances[plantID].shoot_tree
    if shootID >= len(tree):
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::pruneBranch): Shoot with ID of {shootID} does not exist "
                                     f"on plant {plantID}.")
    shoot = tree[shootID]
    if shoot.isPruned():
        return []
    if keep_trimmed_material:
        raise NotImplementedError("pruneBranch(keep_trimmed_material) is not ported")
    if pa.phytomer_callback_depth > 0:
        if node_index >= len(shoot.phytomers):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::pruneBranch): Node index {node_index} is out of range "
                                         f"for shoot {shootID}.")
        if not hasattr(pa, "deferred_prunes"):
            pa.deferred_prunes = []
        pa.deferred_prunes.append((plantID, shootID, int(node_index)))
        return []
    if node_index >= shoot.current_node_number:
        raise PlantArchitectureError(f"ERROR (PlantArchitecture::pruneBranch): Node index {node_index} is out of range for "
                                     f"shoot {shootID}.")
    applyPruneBranch(pa, plantID, shootID, int(node_index))
    return []


@ports("PlantArchitecture::flushDeferredPrunes")
def flushDeferredPrunes(pa):
    """PA:10139-10157: cuts requested from phytomer callbacks, applied in request order."""
    pending = getattr(pa, "deferred_prunes", None)
    if not pending:
        return
    if pa.phytomer_callback_depth > 0:
        raise PlantArchitectureError("ERROR (PlantArchitecture::flushDeferredPrunes): Deferred cuts cannot be applied "
                                     "while a phytomer callback is running. This is an internal error in the "
                                     "PlantArchitecture plug-in.")
    pa.deferred_prunes = []
    for plantID, shootID, node_index in pending:
        shoot = pa.plant_instances[plantID].shoot_tree[shootID]
        if shoot.isPruned() or node_index >= len(shoot.phytomers):
            continue
        applyPruneBranch(pa, plantID, shootID, node_index)


# ---- shadow grid (PA:8784-9266) ---------------------------------------------------------------------------------
# Palubicki et al. (2009) constants (PA:8813-8920)
shadow_strength_a = f32(0.018)
shadow_decay_b = f32(2)
shadow_depth_qmax = 3
shadow_full_exposure_C = f32(1)
shadow_extinction_k = f32(1)
shadow_lateral_weight = f32(0.55)
shadow_horizontal_falloff = f32(1.2)
branch_shedding_relative_light_threshold = f32(0.85)
branch_shedding_foliage_share_scale = f32(0.02)
branch_shedding_voxel_size = f32(0.25)
branch_shedding_projection_G = f32(0.5)
branch_shedding_probability_max = f32(0.95)
branch_shedding_low_crown_probability = f32(0)
branch_shedding_min_dormancy_cycles = 1
growth_priority_promoted_fraction = f32(0.25)
growth_priority_gain = f32(1.6)
growth_priority_max = f32(4)
growth_priority_min = f32(0.5)
growth_priority_min_siblings = 3


def _pyramid_offsets():
    """(q, di, dk, increment) of the shadow pyramid in the C++ loop order; increment = a * b^-q * falloff (float)."""
    from ..core import vec as V
    out = []
    for q in range(shadow_depth_qmax + 1):
        shadow_increment = f32(shadow_strength_a * V.powf(shadow_decay_b, f32(-float(q))))
        for di in range(-q, q + 1):
            for dk in range(-q, q + 1):
                horizontal_offset = f32(np.sqrt(f32(di * di + dk * dk)))
                lateral_falloff = f32(f32(1) / f32(f32(1) + f32(shadow_horizontal_falloff * horizontal_offset)))
                out.append((q, di, dk, f32(shadow_increment * lateral_falloff)))
    return out


_PYRAMID = None


class ShadowGrid:
    """The shadow-propagation grid of one set of leaf bases (``PlantInstance::shadow_grid``, ``crown_depth_grid``,
    ``shadow_grid_voxel_size``), built from those bases on first use. Helios rebuilds the live grid every time step
    the leaf count changes, but only ever reads the copy taken at the season's peak canopy; building the grid when
    it is read gives the same values (the same bases, the same arithmetic) at a fraction of the cost."""

    __slots__ = ("bases", "_built", "grid", "depth", "voxel_size")

    def __init__(self, bases):
        self.bases = bases                       # (N, 3) float32, the leaf bases the grid is built from
        self._built = False
        self.grid = {}
        self.depth = {}
        self.voxel_size = f32(0)

    def build(self):
        if self._built:
            return self
        self._built = True
        bases = self.bases
        if len(bases) == 0:
            return self
        global _PYRAMID
        if _PYRAMID is None:
            _PYRAMID = _pyramid_offsets()
        crown_min, crown_max = bases.min(axis=0), bases.max(axis=0)
        ext = crown_max - crown_min
        crown_extent = max(max(ext[0], ext[1]), ext[2])
        voxel_size = max(f32(0.02), f32(crown_extent / f32(20)))
        self.voxel_size = voxel_size
        # base = leaf_base - shadow_grid_origin (the origin stays at 0: plants are never translated here)
        li = np.floor(bases[:, 0] / voxel_size).astype(np.int64)
        lj = np.floor(bases[:, 2] / voxel_size).astype(np.int64)
        lk = np.floor(bases[:, 1] / voxel_size).astype(np.int64)
        q = np.array([p[0] for p in _PYRAMID], np.int64)
        di = np.array([p[1] for p in _PYRAMID], np.int64)
        dk = np.array([p[2] for p in _PYRAMID], np.int64)
        inc = np.array([p[3] for p in _PYRAMID], f32)
        I = (li[:, None] + di[None, :]).ravel()
        J = (lj[:, None] - q[None, :]).ravel()
        K = (lk[:, None] + dk[None, :]).ravel()
        off = 100000
        span = 2 * off
        keys = ((I + off) * span + (J + off)) * span + (K + off)
        uniq, inverse = np.unique(keys, return_inverse=True)
        values = np.zeros(len(uniq), f32)
        np.add.at(values, inverse, np.tile(inc, len(bases)))        # sequential float32 adds, in loop order
        self.grid = dict(zip(uniq.tolist(), values.tolist()))
        # crown depth: multi-source BFS from the surface voxels of the occupied set
        occupied = set(zip(li.tolist(), lj.tolist(), lk.tolist()))
        nbrs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        depth = {}
        queue = []
        for key in sorted(occupied):
            i, j, k = key
            if any((i + a, j + b, k + c) not in occupied for a, b, c in nbrs):
                depth[key] = 0
                queue.append(key)
        qi = 0
        while qi < len(queue):
            key = queue[qi]
            qi += 1
            d = depth[key]
            i, j, k = key
            for a, b, c in nbrs:
                nb = (i + a, j + b, k + c)
                if nb in occupied and nb not in depth:
                    depth[nb] = d + 1
                    queue.append(nb)
        self.depth = {((i + off) * span + (j + off)) * span + (k + off): d for (i, j, k), d in depth.items()}
        return self

    def light_exposure(self, position) -> np.float32:
        """``shadowGridLightExposure`` (PA:9196-9227)."""
        from ..core import vec as V
        self.build()
        if not self.grid or self.voxel_size <= 0:
            return shadow_full_exposure_C
        p = np.asarray(position, f32)
        vs = self.voxel_size
        i = int(np.floor(f32(p[0] / vs)))
        j = int(np.floor(f32(p[2] / vs)))
        k = int(np.floor(f32(p[1] / vs)))
        off = 100000
        span = 2 * off
        key = ((i + off) * span + (j + off)) * span + (k + off)
        shadow = f32(self.grid.get(key, 0.0))
        crown_depth = self.depth.get(key, 0)
        shadow = f32(shadow + f32(shadow_lateral_weight * f32(crown_depth)))
        shadow_from_others = max(f32(shadow - shadow_strength_a), f32(0))
        return f32(shadow_full_exposure_C * V.expf(f32(-shadow_extinction_k * shadow_from_others)))


EMPTY_SHADOW_GRID = ShadowGrid(np.zeros((0, 3), f32))


def _season_state(inst):
    """The PlantInstance members of the shadow grid and the season peak (H:3427-3487 defaults)."""
    if not hasattr(inst, "season_peak_leaf_area"):
        inst.season_peak_leaf_area = f32(0)
        inst.shadow_grid = EMPTY_SHADOW_GRID
        inst.shadow_grid_leaf_count = -1                 # SIZE_MAX
        inst.season_peak_shadow_grid = EMPTY_SHADOW_GRID
        inst.season_peak_leaf_positions = []
        inst.season_peak_leaf_areas = []
        inst.season_peak_leaf_shootIDs = []
    return inst


@ports("PlantArchitecture::getPlantLeafBases")
def getPlantLeafBases(pa, plantID):
    out = []
    for shoot in pa.plant_instances[plantID].shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.leaf_bases:
                out.extend(petiole)
    return np.asarray(out, f32).reshape(-1, 3)


@ports("PlantArchitecture::updateShadowGrid")
def updateShadowGrid(pa, plantID):
    """PA:9073-9194: rebuilt only when the leaf count changes (built lazily, see :class:`ShadowGrid`)."""
    inst = _season_state(pa.plant_instances[plantID])
    leaf_bases = getPlantLeafBases(pa, plantID)
    if len(leaf_bases) == 0:
        inst.shadow_grid = EMPTY_SHADOW_GRID
        inst.shadow_grid_leaf_count = 0
        return
    if len(leaf_bases) == inst.shadow_grid_leaf_count:
        return
    inst.shadow_grid_leaf_count = len(leaf_bases)
    inst.shadow_grid = ShadowGrid(leaf_bases)


@ports("PlantArchitecture::sumPlantLeafArea")
def sumPlantLeafArea(pa, plantID) -> np.float32:
    from .phytomer import get_leaf_blade_area
    area = f32(0)
    for shoot in pa.plant_instances[plantID].shoot_tree:
        for ph in shoot.phytomers:
            for petiole in ph.leaf_objIDs:
                for objID in petiole:
                    area = f32(area + get_leaf_blade_area(pa, objID))
    return area


@ports("PlantArchitecture::captureSeasonPeakLeaves")
def captureSeasonPeakLeaves(pa, plantID):
    """PA:9244-9266."""
    ctx = pa.context_ptr
    inst = pa.plant_instances[plantID]
    positions, areas, shootIDs = [], [], []
    for shoot in inst.shoot_tree:
        for ph in shoot.phytomers:
            for petiole in range(min(len(ph.leaf_objIDs), len(ph.leaf_bases))):
                for leaf in range(min(len(ph.leaf_objIDs[petiole]), len(ph.leaf_bases[petiole]))):
                    objID = ph.leaf_objIDs[petiole][leaf]
                    if not ctx.does_object_exist(objID):
                        continue
                    positions.append(np.asarray(ph.leaf_bases[petiole][leaf], f32))
                    areas.append(ctx.object_area(objID))
                    shootIDs.append(int(shoot.ID))
    inst.season_peak_leaf_positions = positions
    inst.season_peak_leaf_areas = areas
    inst.season_peak_leaf_shootIDs = shootIDs


def season_step(pa, plantID):
    """The per-sub-step part of advanceTime for a plant without the carbon model (PA:11866-11880)."""
    if pa.carbon_model_enabled:
        return
    inst = _season_state(pa.plant_instances[plantID])
    updateShadowGrid(pa, plantID)
    current_leaf_area = sumPlantLeafArea(pa, plantID)
    if current_leaf_area > inst.season_peak_leaf_area:
        inst.season_peak_leaf_area = current_leaf_area
        inst.season_peak_shadow_grid = inst.shadow_grid
        captureSeasonPeakLeaves(pa, plantID)


@ports("PlantArchitecture::getShadowLightExposureAtPoint")
def getShadowLightExposureAtPoint(pa, plantID, position) -> np.float32:
    return _season_state(pa.plant_instances[plantID]).shadow_grid.light_exposure(position)


@ports("PlantArchitecture::getSeasonPeakLightExposureAtPoint")
def getSeasonPeakLightExposureAtPoint(pa, plantID, position) -> np.float32:
    return _season_state(pa.plant_instances[plantID]).season_peak_shadow_grid.light_exposure(position)


@ports("PlantArchitecture::isStructuralShoot")
def isStructuralShoot(pa, plantID, shootID) -> bool:
    label = pa.plant_instances[plantID].shoot_tree[shootID].shoot_type_label
    return label != "proleptic" and label != "sylleptic"


@ports("PlantArchitecture::updateGrowthPriority")
def updateGrowthPriority(pa, plantID):
    """PA:8945-9071."""
    inst = _season_state(pa.plant_instances[plantID])
    tree = inst.shoot_tree
    grid = inst.season_peak_shadow_grid.build()
    if not grid.grid or grid.voxel_size <= 0:
        return
    siblings_by_parent = {}
    for shoot in tree:
        if shoot.parent_shoot_ID < 0 or not shoot.phytomers or not shoot.shoot_internode_vertices:
            continue
        if isStructuralShoot(pa, plantID, shoot.ID):
            continue
        siblings_by_parent.setdefault(shoot.parent_shoot_ID, []).append(shoot.ID)
    for parent in sorted(siblings_by_parent):
        siblings = siblings_by_parent[parent]
        if len(siblings) < growth_priority_min_siblings:
            continue
        ranked = [(grid.light_exposure(tree[sid].shoot_internode_vertices[-1][-1]), sid) for sid in siblings]
        ranked = std_sort(ranked, lambda a, b: a[0] > b[0])
        promoted_count = int(np.ceil(f32(growth_priority_promoted_fraction * f32(len(ranked)))))
        promoted_count = max(1, min(promoted_count, len(ranked) - 1))
        for i in range(promoted_count, len(ranked)):
            shoot = tree[ranked[i][1]]
            _capture_priority_baseline(shoot)
            shoot.growth_priority = max(growth_priority_min, f32(_priority(shoot) / growth_priority_gain))
            f = f32(np.sqrt(f32(shoot.growth_priority)))
            sp = shoot.shoot_parameters
            sp.max_nodes = max(1, int(_roundf(f32(f32(shoot.growth_priority_base_max_nodes) * f))))
            sp.max_nodes_per_season = max(1, int(_roundf(f32(f32(shoot.growth_priority_base_max_nodes_per_season) * f))))
            sp.internode_length_max = f32(shoot.growth_priority_base_internode_length_max * f)
            sp.vegetative_bud_break_probability_max = min(max(
                f32(shoot.growth_priority_base_bud_break_max * shoot.growth_priority), f32(0)), f32(1))
        for i in range(promoted_count):
            shoot = tree[ranked[i][1]]
            _capture_priority_baseline(shoot)
            shoot.growth_priority = min(growth_priority_max, f32(_priority(shoot) * growth_priority_gain))
            f = f32(np.sqrt(f32(shoot.growth_priority)))
            sp = shoot.shoot_parameters
            sp.max_nodes = int(_roundf(f32(f32(shoot.growth_priority_base_max_nodes) * f)))
            sp.max_nodes_per_season = int(_roundf(f32(f32(shoot.growth_priority_base_max_nodes_per_season) * f)))
            sp.internode_length_max = f32(shoot.growth_priority_base_internode_length_max * f)
            sp.vegetative_bud_break_probability_max = min(max(
                f32(shoot.growth_priority_base_bud_break_max * shoot.growth_priority), f32(0)), f32(1))


def _priority(shoot) -> np.float32:
    """Shoot::growth_priority (in-class default 1)."""
    return f32(getattr(shoot, "growth_priority", f32(1)))


def _roundf(x) -> float:
    """std::round(float): half away from zero."""
    x = float(x)
    return math.copysign(math.floor(abs(x) + 0.5), x)


def _capture_priority_baseline(shoot):
    if not getattr(shoot, "growth_priority_baseline_set", False):
        sp = shoot.shoot_parameters
        shoot.growth_priority_base_max_nodes = int(sp.max_nodes.val())
        shoot.growth_priority_base_max_nodes_per_season = int(sp.max_nodes_per_season.val())
        shoot.growth_priority_base_internode_length_max = f32(sp.internode_length_max.val())
        shoot.growth_priority_base_bud_break_max = f32(sp.vegetative_bud_break_probability_max.val())
        shoot.growth_priority_baseline_set = True


def _sky_directions():
    from ..core import vec as V
    from .tube_utils import PI_F
    dirs, weights = [], []
    for ring, (zen_deg, n_az, w) in enumerate(((15.0, 4, 0.25), (45.0, 8, 0.5), (75.0, 8, 0.25))):
        for a in range(n_az):
            zenith = V.deg2rad(f32(zen_deg))
            azimuth = f32(f32(f32(f32(2) * PI_F) * f32(f32(a) + f32(f32(0.5) * f32(ring)))) / f32(n_az))
            sz = V.sinf(zenith)
            dirs.append(np.array([f32(sz * V.cosf(azimuth)), f32(sz * V.sinf(azimuth)), V.cosf(zenith)], f32))
            weights.append(f32(f32(w) / f32(n_az)))
    return dirs, weights


@ports("PlantArchitecture::shedShadedBranches")
def shedShadedBranches(pa, plantID):
    """PA:9268-9497: shade-based self-pruning of fine wood at the season boundary."""
    from ..core import vec as V
    inst = _season_state(pa.plant_instances[plantID])
    tree = inst.shoot_tree
    positions = inst.season_peak_leaf_positions
    areas = inst.season_peak_leaf_areas
    shootIDs = inst.season_peak_leaf_shootIDs
    if not positions:
        return
    voxel = branch_shedding_voxel_size
    P = np.asarray(positions, f32)
    crown_min = P.min(axis=0)
    crown_max = P.max(axis=0)
    gx, gy, gz = (int(f32(f32(crown_max[c] - crown_min[c]) / voxel)) + 1 for c in range(3))

    def voxel_of(p):
        return tuple(min(max(int(f32(f32(p[c] - crown_min[c]) / voxel)), 0), g - 1)
                     for c, g in zip(range(3), (gx, gy, gz)))

    def vindex(i, j, k):
        return (k * gy + j) * gx + i

    lad = np.zeros(gx * gy * gz, f32)
    v3 = f32(f32(voxel * voxel) * voxel)
    cells = [voxel_of(p) for p in positions]
    np.add.at(lad, np.array([vindex(*c) for c in cells], np.int64),
              np.array([f32(a / v3) for a in areas], f32))
    lad_list = lad.tolist()
    sky_dirs, sky_w = _sky_directions()
    step = f32(f32(0.5) * voxel)
    voxel_light = {}

    def light_at_voxel(cell):
        idx = vindex(*cell)
        got = voxel_light.get(idx)
        if got is not None:
            return got
        centre = crown_min + np.array([f32(voxel * f32(f32(cell[c]) + f32(0.5))) for c in range(3)], f32)
        light = f32(0)
        cmx, cmy, cmz = (float(x) for x in crown_min)
        for d, w in zip(sky_dirs, sky_w):
            optical_depth = f32(0)
            distance = f32(f32(0.5) * step)
            while True:
                pt = centre + distance * d
                i = int(math.floor(float(f32(f32(pt[0] - crown_min[0]) / voxel))))
                j = int(math.floor(float(f32(f32(pt[1] - crown_min[1]) / voxel))))
                k = int(math.floor(float(f32(f32(pt[2] - crown_min[2]) / voxel))))
                if i < 0 or j < 0 or i >= gx or j >= gy or k >= gz:
                    break
                optical_depth = f32(optical_depth + f32(f32(branch_shedding_projection_G * f32(lad_list[vindex(i, j, k)])) * step))
                distance = f32(distance + step)
            light = f32(light + f32(w * V.expf(f32(-optical_depth))))
        voxel_light[idx] = light
        return light

    n = len(tree)
    subtree_area = [f32(0)] * n
    subtree_light = [f32(0)] * n
    for leaf, sid in enumerate(shootIDs):
        subtree_area[sid] = f32(subtree_area[sid] + areas[leaf])
        subtree_light[sid] = f32(subtree_light[sid] + f32(areas[leaf] * light_at_voxel(cells[leaf])))
    plant_leaf_area = f32(0)
    plant_light = f32(0)
    for sid in range(n):
        plant_leaf_area = f32(plant_leaf_area + subtree_area[sid])
        plant_light = f32(plant_light + subtree_light[sid])
    if plant_leaf_area <= 0 or plant_light <= 0:
        return
    plant_mean_light = f32(plant_light / plant_leaf_area)
    for sid in range(n - 1, 0, -1):
        parent = tree[sid].parent_shoot_ID
        if parent < 0:
            continue
        if parent >= sid:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::shedShadedBranches): Shoot {sid} has parent shoot "
                                         f"{parent}, which was created after it. Subtree light cannot be accumulated.")
        subtree_area[parent] = f32(subtree_area[parent] + subtree_area[sid])
        subtree_light[parent] = f32(subtree_light[parent] + subtree_light[sid])

    # crown_radius / crown_height only feed the low-crown term, which is disabled
    # (branch_shedding_low_crown_probability == 0): no draws, nothing to compute.
    shed_list = []
    ctx = pa.context_ptr
    thr = branch_shedding_relative_light_threshold
    for shoot in tree:
        if shoot.parent_shoot_ID < 0 or not shoot.phytomers:
            continue
        if isStructuralShoot(pa, plantID, shoot.ID):
            continue
        if shoot.dormancy_cycles < branch_shedding_min_dormancy_cycles:
            continue
        sid = shoot.ID
        relative_light = f32(0)
        if subtree_area[sid] > 0:
            relative_light = f32(f32(subtree_light[sid] / subtree_area[sid]) / plant_mean_light)
        if relative_light >= thr:
            continue
        shortfall = f32(f32(thr - relative_light) / thr)
        foliage_share = f32(subtree_area[sid] / plant_leaf_area)
        size_weight = V.expf(f32(-foliage_share / branch_shedding_foliage_share_scale))
        if ctx.randu() < f32(f32(branch_shedding_probability_max * shortfall) * size_weight):
            shed_list.append(sid)
    shed_list = std_sort(shed_list, lambda a, b: tree[a].rank > tree[b].rank)
    for sid in shed_list:
        if tree[sid].phytomers:
            pruneBranch(pa, plantID, sid, 0)


def season_boundary(pa, plantID):
    """The season boundary of advanceTime (PA:11888-11914), reached when ``time_since_dormancy`` reaches
    ``dd_to_dormancy_break + dd_to_dormancy``: shade shedding and growth priority (without the carbon model),
    then every shoot goes dormant and the plant is harvested. The caller skips the rest of the plant's step."""
    inst = _season_state(pa.plant_instances[plantID])
    inst.time_since_dormancy = f32(0)
    if not pa.carbon_model_enabled:
        shedShadedBranches(pa, plantID)
        updateGrowthPriority(pa, plantID)
    inst.season_peak_leaf_area = f32(0)
    for shoot in inst.shoot_tree:
        shoot.makeDormant()
        shoot.phyllochron_counter = f32(0)
    harvestPlant(pa, plantID)
