# SPDX-License-Identifier: LGPL-2.1-or-later
# Ported from Helios plugins/plantarchitecture/src/PlantArchitecture.cpp (Copyright (C) 2016-2026 Brian Bailey):
#   buildPlantCanopyFromLibrary                        PA:1165-1200 (spacing overload)
#   Phytomer::getPetioleLength                         PA:1342-1373
#   getPlantBasePosition / sumPlantLeafArea            PA:8768-8797
#   getPlantStemHeight / getPlantHeight                PA:9509-9553
#   getPlantLeafCount / getPlantLeafBases / Areas / InternodeLengths  PA:9671-9748
#   isPlantDormant                                     PA:9841-9853
#   getPlantName / getPlantAge / getPlantMaxAge        PA:9924-9958
#   getShootNodeCount                                  PA:10741-10748
#   getAllShootIDs .. getShootTaper (shoot topology)   PA:10759-10966
#   getAllPlantIDs .. getAllObjectIDs (organ queries)  PA:10968-11400
# and the per-phytomer getters of the PyHelios wrapper (native/src/pyhelios_wrapper_plantarchitecture.cpp:4409-4900).
"""The public query API of :class:`~heliades.plantarchitecture.PlantArchitecture` (W3-E).

Method names are Helios / PyHelios's (``getPlantLeafObjectIDs``); each also has a snake_case alias
(``get_plant_leaf_object_ids``), listed in :data:`SNAKE_ALIASES`. Return types follow PyHelios: plain
``list[int]`` for IDs, ``np.float32`` for scalars, ``np.ndarray`` (3,) float32 for vectors.

None of these draws from the random generator or touches geometry, so calling them between growth steps
never changes a plant. Orders are the C++ traversal orders (shoot tree, then phytomers, then petioles, ...),
so a list is element-for-element the oracle's, not only the same set.

Beyond Helios: ``getPlantUUIDsByObjectLabel`` / ``getAllUUIDsByObjectLabel`` select primitives by their
``object_label`` primitive data (``"leaf"``, ``"petiolule"``, ``"shoot"``, ``"petiole"``, ...), the C++
``filterPrimitivesByData(getAllPlantUUIDs(..), "object_label", label)`` idiom; and ``plant_structure_dict``.
"""
from __future__ import annotations

import numpy as np

from ..migration_registry import ports
from .params import BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING
from .random_parameter import PlantArchitectureError

f32 = np.float32


def _flatten(nested):
    return [x for row in nested for x in row]


class PlantArchitectureAPI:
    """Mixin: everything here reads ``self.plant_instances`` and ``self.context_ptr`` only."""

    # ---- checks ---------------------------------------------------------------------------------

    def _plant(self, plantID, fn: str):
        inst = self.plant_instances.get(int(plantID))
        if inst is None:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{fn}): Plant with ID of {plantID} does not exist.")
        return inst

    def _shoot(self, plantID, shootID, fn: str, msg=None):
        inst = self._plant(plantID, fn)
        if shootID < 0 or len(inst.shoot_tree) <= shootID:
            raise PlantArchitectureError(msg or f"ERROR (PlantArchitecture::{fn}): Shoot with ID of {shootID} does not "
                                                f"exist on plant {plantID}.")
        return inst.shoot_tree[shootID]

    def _phytomer(self, plantID, shootID, node_index, fn: str):
        shoot = self._shoot(plantID, shootID, fn)
        if node_index < 0 or node_index >= len(shoot.phytomers):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{fn}): Node index {node_index} is out of range for "
                                         f"shoot {shootID}, which has {len(shoot.phytomers)} phytomers.")
        return shoot.phytomers[node_index]

    def _exists(self, objID) -> bool:
        return self.context_ptr.does_object_exist(objID)

    # ---- canopy ---------------------------------------------------------------------------------

    @ports("PlantArchitecture::buildPlantCanopyFromLibrary")
    def buildPlantCanopyFromLibrary(self, canopy_center_position, plant_spacing_xy, plant_count_xy, age,
                                    germination_rate=1.0, build_parameters=None) -> list[int]:
        """PA:1170-1200: plants on a grid, row (y) major; one ``randu() < germination_rate`` per site, each plant
        built at age 0, then ``advanceTime(plantIDs, age)`` grows them together (interleaved per sub-step)."""
        nx, ny = int(plant_count_xy[0]), int(plant_count_xy[1])
        if nx <= 0 or ny <= 0:
            raise PlantArchitectureError("ERROR (PlantArchitecture::buildPlantCanopyFromLibrary): Plant count must be "
                                         "greater than zero.")
        c = np.asarray(canopy_center_position, f32).reshape(3)
        sx, sy = f32(plant_spacing_xy[0]), f32(plant_spacing_xy[1])
        ex, ey = f32(sx * f32(nx - 1)), f32(sy * f32(ny - 1))
        germination_rate = f32(germination_rate)
        plantIDs = []
        for j in range(ny):
            for i in range(nx):
                if self.context_ptr.randu() < germination_rate:
                    offset = np.array([f32(f32(f32(-0.5) * ex) + f32(f32(i) * sx)),
                                       f32(f32(f32(-0.5) * ey) + f32(f32(j) * sy)), f32(0)], f32)
                    plantIDs.append(self.buildPlantInstanceFromLibrary(c + offset, 0, build_parameters))
        if f32(age) > 0:
            self.advanceTime(plantIDs, age)
        return plantIDs

    # ---- plant-level scalars ----------------------------------------------------------------------

    @ports("PlantArchitecture::getAllPlantIDs")
    def getAllPlantIDs(self) -> list[int]:
        return sorted(self.plant_instances)                      # std::map order

    @ports("PlantArchitecture::getPlantAge")
    def getPlantAge(self, plantID) -> np.float32:
        inst = self._plant(plantID, "getPlantAge")
        if not inst.shoot_tree:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::getPlantAge): Plant with ID of {plantID} has no "
                                         "shoots.")
        return f32(inst.current_age)

    @ports("PlantArchitecture::getPlantMaxAge")
    def getPlantMaxAge(self, plantID) -> np.float32:
        return f32(self._plant(plantID, "getPlantMaxAge").max_age)

    @ports("PlantArchitecture::setPlantMaxAge")
    def setPlantMaxAge(self, plantID, max_age):
        if f32(max_age) < 0:
            raise PlantArchitectureError("ERROR (PlantArchitecture::setPlantMaxAge): Maximum age must be non-negative.")
        self._plant(plantID, "setPlantMaxAge").max_age = f32(max_age)

    @ports("PlantArchitecture::getPlantName")
    def getPlantName(self, plantID) -> str:
        return self._plant(plantID, "getPlantName").plant_name

    @ports("PlantArchitecture::getPlantBuildParameters")
    def getPlantBuildParameters(self, plantID) -> dict:
        return dict(self._plant(plantID, "getPlantBuildParameters").build_parameters)

    @ports("PlantArchitecture::getPlantBasePosition")
    def getPlantBasePosition(self, plantIDs):
        if isinstance(plantIDs, (list, tuple)):
            return [self.getPlantBasePosition(p) for p in plantIDs]
        return self._plant(plantIDs, "getPlantBasePosition").base_position.copy()

    @ports("PlantArchitecture::isPlantDormant")
    def isPlantDormant(self, plantID) -> bool:
        return all(s.isdormant for s in self._plant(plantID, "isPlantDormant").shoot_tree)

    def _bbox_z_extent(self, objIDs) -> np.float32:
        sc = self.context_ptr
        lo = hi = None
        for o in objIDs:
            a, b = sc.object_bounding_box(o)
            lo = a if lo is None else np.minimum(lo, a)
            hi = b if hi is None else np.maximum(hi, b)
        if lo is None:
            raise PlantArchitectureError("ERROR (Context::getObjectBoundingBox): No objects were given.")
        return f32(hi[2] - lo[2])

    @ports("PlantArchitecture::getPlantHeight")
    def getPlantHeight(self, plantID) -> np.float32:
        """``getObjectBoundingBox(getAllPlantObjectIDs(plantID))``: max.z - min.z over every vertex."""
        self._plant(plantID, "getPlantHeight")
        return self._bbox_z_extent(self.getAllPlantObjectIDs(plantID))

    @ports("PlantArchitecture::getPlantStemHeight")
    def getPlantStemHeight(self, plantID) -> np.float32:
        inst = self._plant(plantID, "getPlantStemHeight")
        base = inst.shoot_tree[0]
        stem = [base.internode_tube_objID]
        if not self._exists(stem[0]):
            raise PlantArchitectureError("ERROR (PlantArchitecture::getPlantStemHeight): The plant does not contain any "
                                         "geometry.")
        for childID in base.childIDs.get(base.current_node_number - 1, []):
            child = inst.shoot_tree[childID]
            if child.rank == base.rank and self._exists(child.internode_tube_objID):
                stem.append(child.internode_tube_objID)
        return self._bbox_z_extent(stem)

    @ports("PlantArchitecture::getLeafBladeArea")
    def getLeafBladeArea(self, leaf_objID) -> np.float32:
        from .phytomer import get_leaf_blade_area
        return get_leaf_blade_area(self, int(leaf_objID))

    @ports("PlantArchitecture::sumPlantLeafArea")
    def sumPlantLeafArea(self, plantID) -> np.float32:
        self._plant(plantID, "sumPlantLeafArea")
        area = f32(0)
        for o in self.getPlantLeafObjectIDs(plantID):
            area = f32(area + self.getLeafBladeArea(o))
        return area

    def getPlantLeafArea(self, plantID) -> np.float32:
        """PyHelios ``getPlantLeafArea`` = ``sumPlantLeafArea`` (blade area, petiolules excluded)."""
        return self.sumPlantLeafArea(plantID)

    @ports("PlantArchitecture::getPlantLeafCount")
    def getPlantLeafCount(self, plantID) -> int:
        self._plant(plantID, "getPlantLeafCount")
        return len(self.getPlantLeafObjectIDs(plantID))

    @ports("PlantArchitecture::getPlantLeafAreas")
    def getPlantLeafAreas(self, plantID) -> list:
        inst = self._plant(plantID, "getPlantLeafAreas")
        return [self.getLeafBladeArea(o) for s in inst.shoot_tree for ph in s.phytomers for pet in ph.leaf_objIDs
                for o in pet if self._exists(o)]

    @ports("PlantArchitecture::getPlantInternodeLengths")
    def getPlantInternodeLengths(self, plantID) -> list:
        inst = self._plant(plantID, "getPlantInternodeLengths")
        return [ph.getInternodeLength() for s in inst.shoot_tree for ph in s.phytomers]

    @ports("PlantArchitecture::getPlantLeafBases")
    def getPlantLeafBases(self, plantID) -> list:
        inst = self._plant(plantID, "getPlantLeafBases")
        return [np.asarray(b, f32).copy() for s in inst.shoot_tree for ph in s.phytomers for b in _flatten(ph.leaf_bases)]

    # ---- shoot topology ---------------------------------------------------------------------------

    @ports("PlantArchitecture::getAllShootIDs")
    def getAllShootIDs(self, plantID) -> list[int]:
        return list(range(len(self._plant(plantID, "getAllShootIDs").shoot_tree)))

    @ports("PlantArchitecture::getShootNodeCount")
    def getShootNodeCount(self, plantID, shootID) -> int:
        return int(self._shoot(plantID, shootID, "getShootNodeCount",
                               "ERROR (PlantArchitecture::getShootNodeCount): Shoot ID is out of range.").current_node_number)

    @ports("PlantArchitecture::getParentShootID")
    def getParentShootID(self, plantID, shootID) -> int:
        return int(self._shoot(plantID, shootID, "getParentShootID").parent_shoot_ID)

    @ports("PlantArchitecture::getShootRank")
    def getShootRank(self, plantID, shootID) -> int:
        return int(self._shoot(plantID, shootID, "getShootRank").rank)

    @ports("PlantArchitecture::getShootDepth")
    def getShootDepth(self, plantID, shootID) -> int:
        tree = self._plant(plantID, "getShootDepth").shoot_tree
        cur = self._shoot(plantID, shootID, "getShootDepth").parent_shoot_ID
        depth = 0
        while cur >= 0:
            depth += 1
            cur = tree[cur].parent_shoot_ID
        return depth

    @ports("PlantArchitecture::getPathToRoot")
    def getPathToRoot(self, plantID, shootID) -> list[int]:
        self._shoot(plantID, shootID, "getPathToRoot")
        tree = self.plant_instances[plantID].shoot_tree
        path, cur = [], int(shootID)
        while cur >= 0:
            path.append(cur)
            cur = tree[cur].parent_shoot_ID
        return path

    @ports("PlantArchitecture::getChildShootIDs")
    def getChildShootIDs(self, plantID, shootID) -> list[int]:
        shoot = self._shoot(plantID, shootID, "getChildShootIDs")
        tree = self.plant_instances[plantID].shoot_tree
        return [int(c) for node in sorted(shoot.childIDs) for c in shoot.childIDs[node] if not tree[c].isPruned()]

    getShootChildIDs = getChildShootIDs     # PyHelios alias

    @ports("PlantArchitecture::getAllDescendantShootIDs")
    def getAllDescendantShootIDs(self, plantID, shootID) -> list[int]:
        out, pending = [], self.getChildShootIDs(plantID, shootID)
        while pending:
            cur = pending.pop()
            out.append(cur)
            pending.extend(reversed(self.getChildShootIDs(plantID, cur)))
        return out

    @ports("PlantArchitecture::getShootIDsByRank")
    def getShootIDsByRank(self, plantID) -> dict[int, list[int]]:
        """PyHelios form: ``{rank: [shootIDs]}`` for the ranks that have live shoots (C++ returns a vector indexed by
        rank with empty entries; PyHelios drops those)."""
        out: dict[int, list[int]] = {}
        for s in self._plant(plantID, "getShootIDsByRank").shoot_tree:
            if not s.isPruned():
                out.setdefault(int(s.rank), []).append(int(s.ID))
        return dict(sorted(out.items()))

    @ports("PlantArchitecture::getShootHierarchyMap")
    def getShootHierarchyMap(self, plantID) -> dict[int, list[int]]:
        out = {}
        for s in self._plant(plantID, "getShootHierarchyMap").shoot_tree:
            if s.isPruned():
                continue
            ch = self.getChildShootIDs(plantID, s.ID)
            if ch:
                out[int(s.ID)] = ch
        return out

    @ports("PlantArchitecture::getTerminalShootIDs")
    def getTerminalShootIDs(self, plantID) -> list[int]:
        return [int(s.ID) for s in self._plant(plantID, "getTerminalShootIDs").shoot_tree
                if not s.isPruned() and not self.getChildShootIDs(plantID, s.ID)]

    @ports("PlantArchitecture::isShootPruned")
    def isShootPruned(self, plantID, shootID) -> bool:
        return bool(self._shoot(plantID, shootID, "isShootPruned",
                                "ERROR (PlantArchitecture::isShootPruned): Shoot ID is out of range.").isPruned())

    @ports("PlantArchitecture::isShootGeometryPrescribed")
    def isShootGeometryPrescribed(self, plantID, shootID) -> bool:
        return bool(self._shoot(plantID, shootID, "isShootGeometryPrescribed").growth_state.geometry_is_prescribed)

    @ports("PlantArchitecture::getShootTaper")
    def getShootTaper(self, plantID, shootID) -> np.float32:
        s = self._shoot(plantID, shootID, "getShootTaper")
        if not s.shoot_internode_radii or len(s.shoot_internode_radii[0]) == 0:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::getShootTaper): Shoot {shootID} of plant {plantID} "
                                         "has no internodes, so its taper is undefined.")
        r0, r1 = f32(s.shoot_internode_radii[0][0]), f32(s.shoot_internode_radii[-1][-1])
        taper = f32(f32(r0 - r1) / r0)
        return f32(0) if taper < 0 else (f32(1) if taper > 1 else taper)

    @ports("PlantArchitecture::getShootInternodeVertices")
    def getShootInternodeVertices(self, plantID, shootID) -> list:
        s = self._shoot(plantID, shootID, "getShootInternodeVertices")
        return [np.asarray(v, f32).copy() for row in s.shoot_internode_vertices for v in row]

    @ports("PlantArchitecture::getShootInternodeRadii")
    def getShootInternodeRadii(self, plantID, shootID) -> list:
        s = self._shoot(plantID, shootID, "getShootInternodeRadii")
        return [f32(r) for row in s.shoot_internode_radii for r in row]

    # ---- per-phytomer getters (PyHelios wrapper) ---------------------------------------------------

    def getPhytomerAge(self, plantID, shootID, node_index) -> np.float32:
        return f32(self._phytomer(plantID, shootID, node_index, "getPhytomerAge").age)

    def getInternodeLength(self, plantID, shootID, node_index) -> np.float32:
        return f32(self._phytomer(plantID, shootID, node_index, "getInternodeLength").getInternodeLength())

    def getInternodeRadius(self, plantID, shootID, node_index) -> np.float32:
        return f32(self._phytomer(plantID, shootID, node_index, "getInternodeRadius").getInternodeRadius())

    def getInternodeNodePositions(self, plantID, shootID, node_index) -> list:
        ph = self._phytomer(plantID, shootID, node_index, "getInternodeNodePositions")
        return [np.asarray(v, f32).copy() for v in ph.getInternodeNodePositions()]

    def getInternodeAxisVector(self, plantID, shootID, node_index, stem_fraction) -> np.ndarray:
        ph = self._phytomer(plantID, shootID, node_index, "getInternodeAxisVector")
        return np.asarray(ph.getInternodeAxisVector(f32(stem_fraction)), f32)

    def _petiole_index(self, ph, petiole_index, n, fn):
        if petiole_index < 0 or petiole_index >= n:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::{fn}): Petiole index {petiole_index} is out of range "
                                         f"({n} petioles).")

    def getPetioleAxisVector(self, plantID, shootID, node_index, stem_fraction, petiole_index) -> np.ndarray:
        ph = self._phytomer(plantID, shootID, node_index, "getPetioleAxisVector")
        self._petiole_index(ph, petiole_index, len(ph.petiole_vertices), "getPetioleAxisVector")
        return np.asarray(ph.getPetioleAxisVector(f32(stem_fraction), petiole_index), f32)

    def getPetioleVertices(self, plantID, shootID, node_index, petiole_index) -> list:
        ph = self._phytomer(plantID, shootID, node_index, "getPetioleVertices")
        self._petiole_index(ph, petiole_index, len(ph.petiole_vertices), "getPetioleVertices")
        return [np.asarray(v, f32).copy() for v in ph.petiole_vertices[petiole_index]]

    def getPetioleRadii(self, plantID, shootID, node_index, petiole_index) -> list:
        ph = self._phytomer(plantID, shootID, node_index, "getPetioleRadii")
        self._petiole_index(ph, petiole_index, len(ph.petiole_radii), "getPetioleRadii")
        return [f32(r) for r in ph.petiole_radii[petiole_index]]

    @ports("Phytomer::getPetioleLength")
    def getPetioleLength(self, plantID, shootID, node_index, petiole_index=None) -> np.float32:
        """No ``petiole_index``: the mean over the phytomer's petioles (float sum / count), 0 if it has none."""
        ph = self._phytomer(plantID, shootID, node_index, "getPetioleLength")
        if petiole_index is None:
            if not ph.petiole_length:
                return f32(0)
            total = f32(0)
            for L in ph.petiole_length:
                total = f32(total + L)
            return f32(total / f32(len(ph.petiole_length)))
        if petiole_index >= len(ph.petiole_length):
            raise PlantArchitectureError(f"ERROR (Phytomer::getPetioleLength): Petiole index of {petiole_index} was given, "
                                         f"but this phytomer has {len(ph.petiole_length)} petioles.")
        return f32(ph.petiole_length[petiole_index])

    def getPhytomerLeafObjectIDs(self, plantID, shootID, node_index) -> list[list[int]]:
        ph = self._phytomer(plantID, shootID, node_index, "getPhytomerLeafObjectIDs")
        return [[int(o) for o in pet] for pet in ph.leaf_objIDs]

    def getLeafBasePosition(self, plantID, shootID, node_index, petiole_index, leaf_index) -> np.ndarray:
        ph = self._phytomer(plantID, shootID, node_index, "getLeafBasePosition")
        self._petiole_index(ph, petiole_index, len(ph.leaf_bases), "getLeafBasePosition")
        if leaf_index < 0 or leaf_index >= len(ph.leaf_bases[petiole_index]):
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::getLeafBasePosition): Leaf index {leaf_index} is out "
                                         "of range.")
        return np.asarray(ph.leaf_bases[petiole_index][leaf_index], f32).copy()

    # ---- organ object IDs (per plant) ------------------------------------------------------------

    @ports("PlantArchitecture::getAllPlantObjectIDs")
    def getAllPlantObjectIDs(self, plantID) -> list[int]:
        out = []
        for shoot in self._plant(plantID, "getAllPlantObjectIDs").shoot_tree:
            if self._exists(shoot.internode_tube_objID):
                out.append(int(shoot.internode_tube_objID))
            for ph in shoot.phytomers:
                out += [int(o) for o in ph.getExistingPetioleObjIDs()]
                out += [int(o) for o in _flatten(ph.leaf_objIDs)]
                for petiole in ph.floral_buds:
                    for fbud in petiole:
                        out += [int(o) for o in fbud.inflorescence_objIDs]
                        out += [int(o) for o in fbud.peduncle_objIDs]
        return out

    @ports("PlantArchitecture::getAllPrototypeObjectIDs")
    def getAllPrototypeObjectIDs(self) -> list[int]:
        """The hidden prototype objects, in std::map key order (leaf cache, then closed flower, open flower, fruit)."""
        out = []
        for key in sorted(self.unique_leaf_prototype_objIDs):
            out += [int(o) for proto in self.unique_leaf_prototype_objIDs[key] for o in proto if self._exists(o)]
        for attr in ("unique_closed_flower_prototype_objIDs", "unique_open_flower_prototype_objIDs",
                     "unique_fruit_prototype_objIDs"):
            cache = getattr(self, attr, None) or {}
            for key in sorted(cache):
                out += [int(o) for o in cache[key] if self._exists(o)]
        return out

    @ports("PlantArchitecture::getAllPlantUUIDs")
    def getAllPlantUUIDs(self, plantID, include_hidden=False) -> list[int]:
        oids = self.getAllPlantObjectIDs(plantID)
        if include_hidden:
            oids += self.getAllPrototypeObjectIDs()
        return self._object_uuids(oids)

    def _object_uuids(self, oids) -> list[int]:
        """``Context::getObjectPrimitiveUUIDs(vector)``: concatenated member UUIDs, object order."""
        sc = self.context_ptr
        return [int(u) for o in oids for u in sc.object_primitive_uuids(o)]

    @ports("PlantArchitecture::getPlantInternodeObjectIDs")
    def getPlantInternodeObjectIDs(self, plantID, shoot_type_label=None) -> list[int]:
        tree = self._plant(plantID, "getPlantInternodeObjectIDs").shoot_tree
        if shoot_type_label is None:
            return [int(s.internode_tube_objID) for s in tree if self._exists(s.internode_tube_objID)]
        shoots = [s for s in tree if s.shoot_type_label == shoot_type_label]
        if not shoots:
            raise PlantArchitectureError(f"ERROR (PlantArchitecture::getPlantInternodeObjectIDs): No shoots with shoot "
                                         f"type label '{shoot_type_label}' exist for plant with ID {plantID}.")
        return [int(s.internode_tube_objID) for s in shoots if self._exists(s.internode_tube_objID)]

    getShootInternodeObjectIDs = getPlantInternodeObjectIDs

    @ports("PlantArchitecture::getPlantPetioleObjectIDs")
    def getPlantPetioleObjectIDs(self, plantID) -> list[int]:
        return [int(o) for s in self._plant(plantID, "getPlantPetioleObjectIDs").shoot_tree for ph in s.phytomers
                for o in ph.getExistingPetioleObjIDs()]

    @ports("PlantArchitecture::getPlantLeafObjectIDs")
    def getPlantLeafObjectIDs(self, plantIDs) -> list[int]:
        if isinstance(plantIDs, (list, tuple)):
            return [o for p in plantIDs for o in self.getPlantLeafObjectIDs(p)]
        return [int(o) for s in self._plant(plantIDs, "getPlantLeafObjectIDs").shoot_tree for ph in s.phytomers
                for pet in ph.leaf_objIDs for o in pet]

    @ports("PlantArchitecture::getPlantPeduncleObjectIDs")
    def getPlantPeduncleObjectIDs(self, plantID) -> list[int]:
        return [int(o) for s in self._plant(plantID, "getPlantPeduncleObjectIDs").shoot_tree for ph in s.phytomers
                for pet in ph.floral_buds for fb in pet for o in fb.peduncle_objIDs]

    def _inflorescence_objIDs(self, plantID, states, fn) -> list[int]:
        return [int(o) for s in self._plant(plantID, fn).shoot_tree for ph in s.phytomers
                for pet in ph.floral_buds for fb in pet if fb.state in states for o in fb.inflorescence_objIDs]

    @ports("PlantArchitecture::getPlantFlowerObjectIDs")
    def getPlantFlowerObjectIDs(self, plantID) -> list[int]:
        return self._inflorescence_objIDs(plantID, (BUD_FLOWER_OPEN, BUD_FLOWER_CLOSED), "getPlantFlowerObjectIDs")

    @ports("PlantArchitecture::getPlantFruitObjectIDs")
    def getPlantFruitObjectIDs(self, plantID) -> list[int]:
        return self._inflorescence_objIDs(plantID, (BUD_FRUITING,), "getPlantFruitObjectIDs")

    # ---- canopy-wide UUIDs / object IDs (std::map order over plants) -----------------------------

    def _over_plants(self, per_plant):
        return [x for pid in sorted(self.plant_instances) for x in per_plant(pid)]

    @ports("PlantArchitecture::getAllUUIDs")
    def getAllUUIDs(self) -> list[int]:
        return self._over_plants(self.getAllPlantUUIDs)

    @ports("PlantArchitecture::getAllLeafUUIDs")
    def getAllLeafUUIDs(self) -> list[int]:
        return self._over_plants(lambda p: self._object_uuids(self.getPlantLeafObjectIDs(p)))

    @ports("PlantArchitecture::getAllInternodeUUIDs")
    def getAllInternodeUUIDs(self) -> list[int]:
        return self._over_plants(lambda p: self._object_uuids(self.getPlantInternodeObjectIDs(p)))

    @ports("PlantArchitecture::getAllPetioleUUIDs")
    def getAllPetioleUUIDs(self) -> list[int]:
        return self._over_plants(lambda p: self._object_uuids(self.getPlantPetioleObjectIDs(p)))

    @ports("PlantArchitecture::getAllPeduncleUUIDs")
    def getAllPeduncleUUIDs(self) -> list[int]:
        return self._over_plants(lambda p: self._object_uuids(self.getPlantPeduncleObjectIDs(p)))

    @ports("PlantArchitecture::getAllFlowerUUIDs")
    def getAllFlowerUUIDs(self) -> list[int]:
        return self._over_plants(lambda p: self._object_uuids(self.getPlantFlowerObjectIDs(p)))

    @ports("PlantArchitecture::getAllFruitUUIDs")
    def getAllFruitUUIDs(self) -> list[int]:
        return self._over_plants(lambda p: self._object_uuids(self.getPlantFruitObjectIDs(p)))

    @ports("PlantArchitecture::getAllObjectIDs")
    def getAllObjectIDs(self) -> list[int]:
        return self._over_plants(self.getAllPlantObjectIDs)

    # ---- by object_label (primitive data) ---------------------------------------------------------

    def getPlantUUIDsByObjectLabel(self, plantID, object_label: str, include_hidden=False) -> list[int]:
        """``filterPrimitivesByData(getAllPlantUUIDs(plantID, include_hidden), "object_label", object_label)``."""
        return [int(u) for u in self.context_ptr.filter_primitives_by_data(
            self.getAllPlantUUIDs(plantID, include_hidden), "object_label", str(object_label))]

    def getAllUUIDsByObjectLabel(self, object_label: str) -> list[int]:
        """``filterPrimitivesByData(getAllUUIDs(), "object_label", object_label)`` over every plant."""
        return [int(u) for u in self.context_ptr.filter_primitives_by_data(self.getAllUUIDs(), "object_label",
                                                                           str(object_label))]

    # ---- structure ----------------------------------------------------------------------------------

    @ports("PlantArchitecture::writePlantStructureXML")
    def writePlantStructureXML(self, plantID, filename=None) -> str:
        """Writes the file if ``filename`` is given; always returns the XML text."""
        from .plant import plant_structure_xml
        text = plant_structure_xml(self, int(plantID))
        if filename is not None:
            with open(filename, "w") as f:
                f.write(text)
        return text

    def plant_structure_dict(self, plantID, organs: bool = True) -> dict:
        """The structured view of ``writePlantStructureXML`` (see :func:`structure.plant_structure_dict`)."""
        from .structure import plant_structure_dict
        return plant_structure_dict(self, int(plantID), organs)


def _snake(name: str) -> str:
    import re
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    s = re.sub(r"(?<=[A-Z])([A-Z][a-z])", r"_\1", s)
    return s.lower().replace("u_u_i_ds", "uuids").replace("i_ds", "ids").replace("x_m_l", "xml")


#: PyHelios name -> snake_case alias, for the public API (W3-E)
PUBLIC_API = (
    "buildPlantInstanceFromLibrary", "buildPlantCanopyFromLibrary", "loadPlantModelFromLibrary", "advanceTime",
    "getAvailablePlantModels", "getCurrentShootParameters", "defineShootType", "addPlantInstance",
    "getAllPlantIDs", "getPlantAge", "getPlantMaxAge", "setPlantMaxAge", "getPlantName", "getPlantBuildParameters",
    "getPlantBasePosition", "isPlantDormant", "getPlantHeight", "getPlantStemHeight", "getLeafBladeArea",
    "sumPlantLeafArea", "getPlantLeafArea", "getPlantLeafCount", "getPlantLeafAreas", "getPlantInternodeLengths",
    "getPlantLeafBases", "getAllShootIDs", "getShootNodeCount", "getParentShootID", "getShootRank", "getShootDepth",
    "getPathToRoot", "getChildShootIDs", "getShootChildIDs", "getAllDescendantShootIDs", "getShootIDsByRank",
    "getShootHierarchyMap", "getTerminalShootIDs", "isShootPruned", "isShootGeometryPrescribed", "getShootTaper",
    "getShootInternodeVertices", "getShootInternodeRadii", "getPhytomerAge", "getInternodeLength",
    "getInternodeRadius", "getInternodeNodePositions", "getInternodeAxisVector", "getPetioleAxisVector",
    "getPetioleVertices", "getPetioleRadii", "getPetioleLength", "getPhytomerLeafObjectIDs", "getLeafBasePosition",
    "getAllPlantObjectIDs", "getAllPrototypeObjectIDs", "getAllPlantUUIDs", "getPlantInternodeObjectIDs",
    "getShootInternodeObjectIDs", "getPlantPetioleObjectIDs", "getPlantLeafObjectIDs", "getPlantPeduncleObjectIDs",
    "getPlantFlowerObjectIDs", "getPlantFruitObjectIDs", "getAllUUIDs", "getAllLeafUUIDs", "getAllInternodeUUIDs",
    "getAllPetioleUUIDs", "getAllPeduncleUUIDs", "getAllFlowerUUIDs", "getAllFruitUUIDs", "getAllObjectIDs",
    "getPlantUUIDsByObjectLabel", "getAllUUIDsByObjectLabel", "writePlantStructureXML",
    "setPlantPhenologicalThresholds", "breakPlantDormancy",
)
SNAKE_ALIASES = {n: _snake(n) for n in PUBLIC_API}


def install_snake_aliases(cls):
    for camel, snake in SNAKE_ALIASES.items():
        if hasattr(cls, camel) and not hasattr(cls, snake):
            setattr(cls, snake, getattr(cls, camel))
    return cls


__all__ = ["PlantArchitectureAPI", "PUBLIC_API", "SNAKE_ALIASES", "install_snake_aliases"]
