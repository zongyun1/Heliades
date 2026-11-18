# W5-D: nut trees (almond x4, pistachio, walnut) and the shared tree infrastructure

Baseline: Helios 1.3.98 (`14860044f`). PA = PlantArchitecture.cpp, PL = PlantLibrary.cpp, AS = Assets.cpp,
IO = InputOutput.cpp.

## Result

All six models match the oracle at **0 ulp** (full hidden-inclusive snapshot, plant XML, every shoot's stored nodes
and radii, engine draw counts) at every age of the ladder below, for Context seeds 1..8 unless stated.

| model | ages verified (seeds) | phases covered |
|---|---|---|
| `almond` | 0, 95, 130, 250, 366, 548 (1..8); 732 (1..4); 913 (1..2); 5-day steps 85..135 (1..4); non-default build parameters at 0 (1..4) and 200 (1..2) | planting (dormant, trained trunk + scaffolds), dormancy break (day 90), flowering, fruit set and growth, spurs, season boundary (leaf fall, harvest, growth priority), second season, first shade shedding (day 730), third summer |
| `almond_aldrich` | 0, 95, 200, 366, 548 (1..8) | same, first two seasons |
| `almond_independence` | 0, 95, 200, 366, 548 (1..8); non-default build parameters at 0 / 200 | same |
| `almond_wood_colony` | 0, 95, 200, 366, 548 (1..8) | same |
| `pistachio` | 0, 170, 220, 300, 366, 548 (1..8); 5-day steps 160..230 (1..4); non-default build parameters at 0 / 200 | dormancy break (day 165), leaf-angle tracking, scaffold tipping + regrowth (deferred cuts), fruit, first-winter thinning/heading of secondaries, second season |
| `walnut` | 0, 170, 220, 366, 548 (1..8); 732 (1..2); non-default build parameters at 0 / 200 | dormancy break (day 165), compound leaves, fruit, boundary, second season, first shedding |

`548` is the reference age of Helios's own plausibility test (`selfTest.cpp`, "every model builds a plausible plant
at its reference age": trees in the summer of their second season) and `913` its trunk-thickening reference age.
There is no tree in `scripts/demos/helios_gallery.py`.

**Not verified: the full `max_age`** (almond / Aldrich / Independence 3000 d, Wood Colony and pistachio 1825 d,
walnut 1460 d). Heliades is ~25-60x slower than Helios on these trees (core performance, not this package), so a
3000-day almond would take hours; the deepest verified age is 913 d (almond), 732 d (walnut), 548 d (others). The
seasonal machinery repeats every 365 days, and every mechanism of a later year (dormancy, break, spurs, priority,
shedding, pruning) is exercised and verified within the verified ages.

### Build time and size (seed 1, Helios vs Heliades, this machine, shared with other agents)

| model | age | primitives | Helios | Heliades |
|---|---|---|---|---|
| almond | 0 | 1 700 | 0.02 s | 0.3 s |
| almond | 548 | 32 138 | 0.55 s | 27 s |
| almond | 913 | 84 618 | 2.3 s | 103 s |
| almond_aldrich | 548 | 32 530 | 0.8 s | 32 s |
| almond_independence | 548 | 23 814 | 0.7 s | 22 s |
| almond_wood_colony | 548 | 21 318 | 0.5 s | 18 s |
| pistachio | 548 | 78 244 | 0.8 s | 52 s (before the prim_batch speed-up; ~30 s after) |
| walnut | 548 (seed 2) | 39 434 | 0.65 s | 14 s |

Full `max_age` on Helios (not run on Heliades; at the observed ~40x the 3000-day almond would take ~4 h):
almond 1825 d 44 s / 179 384 prims, 3000 d 337 s / 824 424; almond_aldrich 3000 d 131 s / 391 924;
almond_independence 3000 d 143 s / 430 394; almond_wood_colony 1825 d 19 s / 98 328; pistachio 1825 d 20 s /
1 299 310; walnut 1460 d 7.9 s / 195 726.

The whole nut-tree suite (49 cases, all seeds) takes ~45 min on Heliades + Helios on this (shared) machine; the
fast suite gains the six `params` cases, `trees.build_parameters`, two age-0 builds and the unit tests (~25 s).
`grapevine_VSP` now also runs `season_step` (its 365-day period equals its max_age): no change in results
(`grape.age76`, `grape.steps` pass), ~20% more time at age 76.

## Shared tree infrastructure (API for W5-E: apple, apple_fruitingwall, olive, easternredbud)

### Engine: `heliades/plantarchitecture/seasons.py` (new)

* dormancy: `makeDormant(shoot)` (also `Shoot.makeDormant()`), `makePlantDormant(pa, plantID)`, `harvestPlant(pa, plantID)`;
* pruning: `pruneBranch(pa, plantID, shootID, node_index)` (inside a phytomer callback the cut is queued and applied by
  `flushDeferredPrunes(pa)` at the end of the sub-step, as in C++), `applyPruneBranch`, `deletePhytomer(phytomer)`;
* the season boundary: `season_step(pa, plantID)` (shadow grid, plant leaf area, season-peak capture; called every
  sub-step by `growth.py` for a plant whose dormancy period fits in `max_age`) and `season_boundary(pa, plantID)`
  (`shedShadedBranches`, `updateGrowthPriority`, every shoot dormant, harvest);
* light: `ShadowGrid` (Palubicki shadow pyramid + crown depth, built lazily from the leaf bases of the last rebuild:
  same values as Helios, which rebuilds it every time the leaf count changes but only reads the season-peak copy),
  `getShadowLightExposureAtPoint`, `getSeasonPeakLightExposureAtPoint` (apple's flower-bud thinning needs this one),
  `isStructuralShoot`, `sumPlantLeafArea`, `captureSeasonPeakLeaves`, `getPlantLeafBases`;
* `std_sort(seq, less)`: libstdc++'s `std::sort` move for move (introsort, threshold 16, median of three, heap sort at
  the depth limit, final insertion sort). Use it wherever the C++ sorts with possible ties (`std::sort` is unstable).

### Engine: `heliades/plantarchitecture/leaf_angle_tracking.py` (new)

`enablePlantLeafElevationAngleDistributionTracking(pa, plantID, mu, nu, lambda_deg)` (called by
`buildPlantInstanceFromLibrary` for every model with a leaf-inclination Beta: cowpea, pistachio, easternredbud),
`updatePlantLeafAngleDistributionTracking`, `add_to_cohort`, `Phytomer::setLeafNormal` (`setLeafNormal`),
`evaluate_Beta_distribution_CDF` (glibc `lgamma` through ctypes), `cart2sphere` / `sphere2cart` / `atan2_2pi`.
Elevation tracking only; azimuth tracking (ellipsoidal CDF) raises `NotImplementedError` (no library model uses it).

### Engine: `heliades/plantarchitecture/prim_batch.py` (new)

Batched triangle areas/normals bit-identical to `Primitive.area()`/`normal()`, and left-to-right row sums over a
zero-padded table. Used by the season bookkeeping and the tracker, which read every leaf facet every sub-step (the
per-primitive path cost 70% of a pistachio build).

### Library: `heliades/plantarchitecture/library/trees.py` (new)

* declarations `tree_trunk_height`, `tree_num_scaffolds`, `tree_scaffold_angle`, `walnut_scaffold_angle`,
  `TREE_TRAINING_PARAMETERS`; `register_build_parameters(pa)` registers them for almond, almond_independence,
  **apple**, pistachio and walnut (PL:512-518), so W5-E does not need to register apple's;
* trunk internode lengths (`apple_/pistachio_/walnut_/olive_/almond_trunk_internode_length`), `trunkMaxNodes`,
  `checkScaffoldsFitTrunk`, `trunk_node_count` (the `uint(h / L)`, at least 1, idiom);
* `kill_trunk_buds(pa, plantID, uID_trunk)` (the "kill apex, remove leaves, kill buds" block of every tree builder),
  `scaffold_yaw(ctx, i, n)` (`(i + randu(-0.2, 0.2)) / n * 2 * M_PI`);
* AS:940-1017 training helpers: `nodesToKeepForLength`, `cutShootToLength`, `releaseBudsBelowCut`,
  `growSpursFromUnbrokenBuds` (apple uses all four).

Also `Shoot.calculateShootLength()` (shoot.py) and the prescribed-leaf-pose branch of
`Phytomer.setPetioleAndLeafScaleFraction` (a steered leaf keeps its offset from the petiole).

Things W5-E will still need: `getSeasonPeakLightExposureAtPoint` is ported but not yet oracle-checked on its own
(it is the same `ShadowGrid.light_exposure` that growth priority reads, which is verified through the 548-day trees);
`addEpicormicShoot`/`sampleEpicormicShoot` (still raise; none of my models enables epicormic shoots);
`appendWoodDome` (Wye), `appendShootFromNodePositions`; `pruneBranch(keep_trimmed_material=True)`; leaf texture stages.

## Models (files)

* `library/almond.py`: the four almond initializers and builders, `almondTrunkPath`, AlmondFruit/FlowerPrototype,
  AlmondPhytomerCreationFunction, AlmondPhytomerCallbackFunction (empty), AlmondSpurPhytomerCallbackFunction.
  The capsicum's `AlmondFlowerPrototype` hook is the same object (`_hook`), so both bind one implementation.
* `library/pistachio.py`: initializer, builder (opposite scaffold pairs), Pistachio{Fruit,Flower}Prototype,
  PistachioPhytomerCreationFunction, PistachioPhytomerCallbackFunction (training).
* `library/walnut.py`: initializer, builder, Walnut{Fruit,Flower}Prototype, Walnut creation/callback functions.

## Parity cases (`suites/plantarch/{ops,cases}_nut_trees.py`) and tests (`tests/plantarch/test_nut_trees.py`)

Per model `plantarch.<m>.load`, `.params`, `.age<A>`; `plantarch.trees.build_parameters` (names of every model and
the rejections: unknown key, out of range, a non-integer scaffold count, a model that declares no parameters,
`checkScaffoldsFitTrunk`, pistachio's pair check); `.build_parameters_age{0,200}`; `almond.steps`, `pistachio.steps`.
Every case observes the probe + shoot summary + XML + full snapshot stream (`observe`).

Mutation checks (`test_mutations_are_caught`; each caught):

| mutation | caught by |
|---|---|
| trunk sway `addComponent` drawn left to right | `almond.age0` |
| almond spur fraction 0 | `almond.age95` |
| pistachio scaffold roll drawn after the pitch | `pistachio.age0` |
| walnut nodes not blind | `walnut.age170` |
| leaf-angle tracking chooses no target | `pistachio.age170` |
| pistachio tipping length 0.40 | `pistachio.age300` (not yet tipped by day 220 on seed 1) |
| growth priority gain 1.5 | `almond.age548` |
| shedding threshold 0.80 | `almond.age732` |

## Shared-file changes (all additive or filling a branch that raised)

* `core/vec.py`: `tanf`, `expf`, `powf`, `atan2f` (glibc through ctypes).
* `shoot.py`: `Shoot.makeDormant` delegates to `seasons.makeDormant` (it raised); new `Shoot.calculateShootLength`.
* `growth.py`: the season-boundary raise is replaced by `seasons.season_boundary` (+ `return`, the C++ `continue`);
  `seasons.season_step` runs each sub-step only when `0 < dd_to_dormancy_break + dd_to_dormancy <= max_age` (never for
  the herbaceous crops, whose 1000-day period exceeds `max_age`, so their cost and results are unchanged);
  `flushDeferredPrunes` after the shoot loop; leaf-angle tracking: cohort entries at bud break and phytomer
  append (the raise is gone) and the per-sub-step update, all gated on `leaf_angle_tracking_enabled`.
* `architecture.py`: `buildPlantInstanceFromLibrary` enables leaf elevation tracking instead of raising.
* `phytomer.py`: the prescribed-leaf-pose branch of `setPetioleAndLeafScaleFraction` (it raised) and
  `_nearest_point_on_polyline` (PA:278).
* `xml_io.py`: writes `<plant_model>`/`<build_parameter>` (structure parameters, `%.9g`),
  `<leaf_inclination_Beta_distribution>`, `<prescribed_internode_nodes/radii>`, `<leaf_prescribed>`/`<leaf_base>`
  (each raised).
* `library/__init__.py`: registers the six models; `library/build_parameters.py`: calls `trees.register_build_parameters`.
* `tests/test_parity.py`: the slow nut-tree cases in `SLOW_CASES`.

Fast suite: 390 passed, 172 skipped. Spot-checked existing slow cases `grape.age76` and `grape.steps` still pass.

## Helios findings / discrepancies

1. **Unstable sorts decide which branches are shed and promoted.** `shedShadedBranches` sorts the shed list by rank and
   `updateGrowthPriority` sorts siblings by light exposure with `std::sort` (PA:9489, 8995). Ties are common (equal
   ranks; siblings whose tips share a voxel get identical exposure), so which sibling is promoted depends on the
   standard library's introsort. The results are libstdc++-specific (selfTest.cpp already notes libc++ vs libstdc++
   differences). Heliades replicates libstdc++'s `std::sort` exactly (`seasons.std_sort`).
2. **`almond_wood_colony` typo** (PL:2038): `shoot_parameters_proleptic.vegetative_bud_break_probability_max = 0.6`
   in the sylleptic block, after sylleptic was copied, so the proleptic (and the later-copied scaffold) types get 0.6
   and sylleptic keeps 0.5. Reproduced.
3. `buildAlmondTreeAldrich` / `buildAlmondTreeWoodColony` fall back to `initializeAlmondTreeShoots` (Nonpareil) when
   no shoot types are loaded (PL:1564, 2092). Unreachable through the library; reproduced.
4. `isStructuralShoot` treats everything but proleptic/sylleptic as structural, so **spurs are never shed** (PA:8936).
   The low-crown shedding term is disabled (`branch_shedding_low_crown_probability = 0`), so the crown radius/height it
   computes is dead code (not computed in Heliades; no draws, no state).
5. `WalnutPhytomerCallbackFunction` and `PistachioFlowerPrototype` are defined but never referenced by the models
   (ported anyway).
6. The documentation disagrees with the code: the phenology table in PlantArchitecture.dox (almond dormancy break
   165 d, max age 1825; walnut max age 1095; pistachio flower opening 3) vs the builders (almond 90/-1/3/7/20/275 with
   max_age 3000; walnut max_age 1460; pistachio opening -1); the shoot-type table omits almond's sylleptic/spur and
   pistachio's scaffold/spur.
7. The shadow grid is rebuilt every sub-step the leaf count changes, but only its season-peak copy is ever read
   (by growth priority; apple's light thinning) — wasted work in Helios; Heliades builds it lazily with the same
   values.
