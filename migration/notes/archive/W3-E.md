# W3-E: library build, public API and acceptance

Baseline: Helios 1.3.98. Scope from SPEC §8: the public `PlantArchitecture` API on the tomato path,
`plant_structure_dict`, multiple plants in one `PlantArchitecture` (`buildPlantCanopyFromLibrary`), the acceptance
grid `plantarchitecture.tomato`, and a performance report. W3-D (inflorescence) was merged into this branch before the
final run, so the grid runs in full (no gated pairs).

## What is ported

| file | contents | C++ |
|---|---|---|
| `plantarchitecture/api.py` (new) | `PlantArchitectureAPI` mixin (`PlantArchitecture` inherits it): canopy build; plant scalars and lists (`getPlantAge`, `getPlantMaxAge`/`setPlantMaxAge`, `getPlantName`, `getPlantBuildParameters`, `getPlantBasePosition`, `isPlantDormant`, `getPlantHeight`, `getPlantStemHeight`, `getLeafBladeArea`, `sumPlantLeafArea` = `getPlantLeafArea`, `getPlantLeafCount`, `getPlantLeafAreas`, `getPlantInternodeLengths`, `getPlantLeafBases`); shoot topology (`getAllShootIDs`, `getShootNodeCount`, `getParentShootID`, `getShootRank`, `getShootDepth`, `getPathToRoot`, `getChildShootIDs` (= `getShootChildIDs`), `getAllDescendantShootIDs`, `getShootIDsByRank`, `getShootHierarchyMap`, `getTerminalShootIDs`, `isShootPruned`, `isShootGeometryPrescribed`, `getShootTaper`, `getShootInternodeVertices`/`Radii`); the per-phytomer getters of the PyHelios wrapper (`getPhytomerAge`, `getInternodeLength`/`Radius`/`NodePositions`/`AxisVector`, `getPetioleAxisVector`/`Vertices`/`Radii`/`Length` (both overloads), `getPhytomerLeafObjectIDs`, `getLeafBasePosition`); organ queries (`getAllPlantObjectIDs`, `getAllPrototypeObjectIDs`, `getAllPlantUUIDs(include_hidden)`, `getPlantInternodeObjectIDs` (+ shoot-type overload), `getPlantPetiole/Leaf/Peduncle/Flower/FruitObjectIDs`, canopy-wide `getAllUUIDs`, `getAll{Leaf,Internode,Petiole,Peduncle,Flower,Fruit}UUIDs`, `getAllObjectIDs`); by `object_label`: `getPlantUUIDsByObjectLabel`, `getAllUUIDsByObjectLabel` (= `filterPrimitivesByData(..., "object_label", label)`); `writePlantStructureXML`; `plant_structure_dict`; snake_case aliases for all of these (`SNAKE_ALIASES`, e.g. `get_all_leaf_uuids`) | PA:1165-1200, 1342-1373, 8768-8810, 9509-9553, 9671-9748, 9841-9853, 9924-9958, 10741-11400; wrapper 4409-4900 |
| `plantarchitecture/architecture.py` | inherits the mixin; accepts PyHelios `vec3`s; `advanceTime(dt, plant_id=/plant_ids=/years=)` keyword forms next to the C++ positional ones; creates the inflorescence prototype caches (`unique_{closed,open}_flower/fruit_prototype_objIDs`) in the constructor like the C++ members (W3-D created them lazily; `inflorescence._caches` still tolerates either) | PA:382-403 |
| `plantarchitecture/structure.py` | `xml_to_structure_dict(text)` (any writer's XML, Helios or Heliades) and `plant_structure_dict(pa, plantID, organs=True)` | InputOutput.cpp:519-1066 |

`plant_structure_dict` is the parsed `writePlantStructureXML` of the port: every number is the float32 of what Helios
prints (6 significant digits), angles in degrees as in the XML. Plant → `shoots[]` → `phytomers[]` →
`internode{..., petioles[] → {..., leaves[], floral_buds[] → {bud_state, ..., peduncle{}, inflorescence{flower_offset,
flowers[]}}}], internode_floral_buds[]}`. Bud states are lists per petiole. With `organs=True` the live object IDs are
added: per shoot `organs.internode`, per phytomer `organs.{petioles, leaves, peduncles, inflorescences}`, per plant
`organs.{internode, petiole, leaf, peduncle, flower, fruit}`. Full-precision geometry is not in the dict (the XML has
6 digits). Use the getters for that.

Return types follow PyHelios: ID lists as `list[int]`, scalars as `np.float32`, vectors as float32 `(3,)` arrays,
`getShootIDsByRank` as PyHelios's `{rank: [ids]}` (ranks without live shoots are dropped; C++ returns a vector).
Lists come out in the C++ traversal order, not merely as the same set. Not ported, because they are off the tomato path or need a
subsystem Heliades lacks: leaf-angle distributions and inclinations, pruning/harvest, collision, carbon/nitrogen,
USD/QSM writers, `readPlantStructureXML`.

## Parity cases (`cases_plantarch_acceptance.py`, `ops_plantarch_acceptance.py`): all pass at 0 ulp

| case | grid | checks |
|---|---|---|
| `plantarchitecture.tomato` | ages {10, 30, 50, 75, 100, 120} × Context seeds 1..8 (48) | `snapshot(include_hidden=True)`: every primitive (UUID, type, vertices, normal, area, color, UV, texture, solid fraction, parent, hidden) and object (members, center, transform, hidden) |
| `plantarchitecture.tomato.state` | same 48 | materials and data of every primitive and object, hidden ones included (material label + color/texture/two-sided; every primitive datum; every object datum with type and value, e.g. `leaf_area`, `volume_ratio`, `old_shoot_volume`); the whole XML; `plant_structure_dict` vs the oracle XML through the same parser + organ IDs; the API (`pa_accept_api`: everything above that PyHelios exposes, including `object_label` selections through `filterPrimitivesByData` on the oracle); finally the engine-call count of the build |
| `plantarchitecture.tomato.two_plants` (+`_state`) | seeds 1..8 | two 30-day tomatoes from two `buildPlantInstanceFromLibrary` calls in one PlantArchitecture |
| `plantarchitecture.tomato.canopy` (+`_state`) | seeds 1..4 | `buildPlantCanopyFromLibrary` 2×2, 0.5 m, germination 0.75, 20 days (age-0 builds, then one interleaved `advanceTime(plantIDs)`) |
| `plantarchitecture.tomato.staggered` (+`_state`) | seeds 1..4 | plant A; `advanceTime(6)` for all; plant B built at age 4; `advanceTime([A,B], 5)`; `advanceTime(3, plant_id=B)`; `advanceTime(2)` |

The snapshot case caches its build's state stream, so the 48 pairs are built once per backend. Full grid
(`python -m heliades.parity plantarchitecture.tomato`, 8 cases): **8/8 pass, 0 ulp, all 48 pairs** (after merging
W3-D). `python -m heliades.parity plantarch`: 38/38. `pytest -q tests`: 360 passed, 1 skipped.

Flowering pairs (oracle plant built inflorescence geometry: a peduncle, or the flower/fruit prototypes, which
outlive a flower that died): `python -m heliades.parity.cases_plantarch_acceptance` prints them.

| age | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|
| 10, 30, 50 | . | . | . | . | . | . | . | . |
| 75 | . | F | . | F | F | F | . | F |
| 100, 120 | F | F | F | F | F | F | F | F |

Before W3-D merged the 27 flower-free pairs ran (all passing) and the 21 F pairs were gated. Now all 48 run.
The gate is kept as an escape hatch: `HELIADES_PARITY_FLOWERING=0` (shared with W3-D's `plantarch.tomato.flowering`)
makes `inflorescence_geometry_implemented()` false, and then the F pairs return the same sentinel on both backends.
Without the variable, the capability check is "`inflorescence` module present and the Phytomer seam no longer raises
the W3-D placeholder" (true now; `test_gating` asserts it).

**Multi-plant findings.** A second plant creates no hidden objects and no private draws: the 35 blade prototypes and
the 3 inflorescence prototypes are per `PlantArchitecture`. The UUIDs and object IDs of plant 2 therefore follow
plant 1's directly, and they match the oracle. Canopy and staggered builds match, including the interleaved sub-steps
of a multi-plant `advanceTime` (the phyllochron minimum over all listed plants sets the shared step size).

Unit tests (`tests/heliades/test_plantarch_api.py`): aliases, metric consistency (leaf area = float sum of blade
areas, organ UUIDs partition the plant), topology, the structure dict against the live tree and the parser, the
shared prototype cache, canopy layout, `advanceTime` keyword forms, gating, flowering pair (seed 2, day 75) builds
flowers and the queries see them.

## Performance (one tomato, `buildPlantInstanceFromLibrary`, wall time; 16 cores, load average ~12 from other agents' runs, so both columns are inflated alike)

| age | seed | primitives | Helios s | Heliades s | ratio |
|---|---|---|---|---|---|
| 10 | 1 / 3 | 11 592 | 0.05 | 0.99 | 20 |
| 30 | 1 / 3 | 22 512 / 23 604 | 0.10 / 0.08 | 2.5 | 26-32 |
| 50 | 1 / 3 | 31 248 / 51 996 | 0.13 / 0.22 | 3.4 / 5.6 | 25-27 |
| 75 | 1 / 3 | 82 572 / 117 516 | 0.37 / 0.67 | 10.5 / 15.5 | 23-28 |
| 100 | 1 / 3 | 174 650 / 242 490 | 0.93 / 1.39 | 22.5 / 33.1 | 24 |
| 120 | 1 / 3 | 256 730 / 334 500 | 1.69 / 2.25 | 37.6 / 55.7 | 22-25 |

Heliades is a steady ~25× slower: cost is linear in primitives × transforms. Profile at age 120, seed 1 (cProfile 71 s;
the plant-architecture logic itself is < 10%):

| hot spot | cum. s | cause |
|---|---|---|
| `core.object_transforms.translate_object` → `_apply_translation` | 24 | 8.0 M per-primitive calls (`setPetioleBase` moves every leaflet/flower every sub-step) |
| `_left_multiply_members` → `primitives.matmult` | 19 | 2.6 M per-primitive 4×4 products (scale/rotate of objects) |
| `copy_primitive` → `copy.deepcopy` | 15 | 276 k deep copies (leaflet and flower copies) |
| `Primitive.area` / `vertices` | 5 | blade area for girth, recomputed from the transform each sub-step |

Proposed speed-ups (bit-exact; none implemented, all are in `heliades/core`, outside this package):
1. **Batch object transforms.** Stack the members' transforms `(n,16)` and apply `translate` as one float32 add on
   the translation column (affine members; per-primitive fallback otherwise). Run `matmult` as the same k-ordered
   `M = M + ML[:,k] * MR[k,:]` broadcast over `(n,4,4)`. These are the same IEEE float32 operations element by
   element, so the bits are unchanged. Expected gain: about 40 of 71 s.
2. **`copy_primitive` without `deepcopy`.** Copy the known `Primitive` fields explicitly (a shallow `__dict__` copy
   plus `.copy()` of the arrays). The data is identical. Expected gain: about 13 s.
3. **Cache vertices and area per transform version.** Invalidate them on any transform write. This is exact because
   the same function is evaluated on the same input. Expected gain: about 4 s.
4. Longer term: a structure-of-arrays primitive store (transforms as one float32 array per object). This makes (1)
   the default and lets `snapshot` avoid Python loops (snapshot of 330 k primitives takes ~10 s now).

After (1)-(3) the expected ratio is about 8-10×. The rest is Python overhead of the per-phytomer logic.

## Discrepancies / findings

1. **PLAN.md's "first flowers between days 75 and 100" is wrong** (W3-C found day 56). In the oracle, seeds 2, 4, 5,
   6 and 8 have inflorescence geometry by day 75, and every seed has it by day 100. W3-C counted 6 of 8 at day 75 by
   bud state. By geometry, seed 1, 3 and 7 are flower-free at 75.
2. **`filterPrimitivesByData` is not a stable filter.** It swap-and-pops (Heliades reproduces this), so the
   `object_label` selections are a permutation of the plant's UUID order. The oracle side of the case uses the
   real `filterPrimitivesByData`, not a Python filter.
3. **PyHelios vs C++ return shapes.** `getShootIDsByRank` returns a dict in PyHelios and a vector in C++; the port
   follows PyHelios. `getPlantLeafObjectIDs` includes IDs of leaves that no longer exist (C++ does not filter);
   `getPlantLeafAreas` skips them. Both are ported as written.
4. **`getAllPrototypeObjectIDs`** is not exposed by PyHelios. It is checked indirectly: the hidden objects of the
   snapshot are exactly the prototypes.
5. `getLeafBladeArea` (used by `getPlantLeafArea`) subtracts the petiolule primitives, and the float sum is
   accumulated in leaf order.

## Requests for core (not done)

- The batched transforms, the `copy_primitive` fast path and the vertex/area cache above.
- `snapshot()` vectorisation (it dominates the parity-case wall time at ages 100-120).
