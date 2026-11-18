# W5-E: fruit / ornamental trees (apple, apple_fruitingwall, olive, easternredbud)

Baseline: Helios 1.3.98 (`14860044f`). PA = PlantArchitecture.cpp, PL = PlantLibrary.cpp, AS = Assets.cpp.
Built on W5-D's tree infrastructure (`seasons.py`, `library/trees.py`, `leaf_angle_tracking.py`).

## Result

All four models match the oracle at **0 ulp** (full hidden-inclusive snapshot, plant XML, every shoot's stored nodes
and radii, engine draw counts; plus every primitive's material label and primitive data at one age) at every age
of the ladder below.

| model | ages verified (Context seeds) | phases covered |
|---|---|---|
| `apple` | 0, 170, 200, 300, 366, 548 (1..8); 732 (1..2); 913, **1460 = max_age** (1); `data250` (1..2); non-default build parameters at 0 (1..4) and 200 (1..2); 2-day steps 360..372 (1..2) | trained tree, break, flowering, fruit, spurs, short shoots, boundary, light thinning, second season, shedding, to max_age |
| `apple_fruitingwall` | 0, 170, 250, 366 (1..8); 548 (1..4); 732 (1..2); 913, 1460, **3650 = max_age** (1); `short_wall` (tree_height 2.0) 400 (1..4), 800 (1..2), 1100 (1); `data250` (1..2); non-default build parameters at 0 (1..4) / 200 (1..2); steps 360..372 (1..2) | nursery whip + feathers, all training rules but two (below), ten seasons |
| `olive` | 0, 170, 250, 366, 548 (1..8); 800 (1..2); 1100 (1); `data250` (1..2); non-default trunk height at 0 (1..4) / 200 (1..2) | evergreen boundary, 600-day leaf lifespan (first shed ~day 770), third season |
| `easternredbud` | 0, 30, 120, 250, 366, 548 (1..8); 732 (1..2); 1100, **1460 = max_age** (1); `data250` (1..2); steps 360..372 (1..2) | immediate growth from day 0, global-RNG callback from the first dormancy, leaf-angle tracking, to max_age |

`548` is the reference age of Helios's plausibility test (selfTest.cpp; trees in the summer of their second
season). None of the four is in `scripts/demos/helios_gallery.py`.

Phases covered: planting (dormant trained tree / nursery whip, or the redbud's immediately broken dormancy),
dormancy break (day 165), flowering, fruit set and growth, the season boundary (leaf fall except for the evergreen
olive, harvest, growth priority), second season, the first shade shedding (day 730), olive leaf shedding at the
600-day leaf lifespan (first ~day 770). Apple: spur release (`growSpursFromUnbrokenBuds`), short shoots, the
axillary flowering zones, flower-bud thinning by the season-peak light (`getSeasonPeakLightExposureAtPoint`, now
oracle-checked: thousands of thinning calls below full exposure per tree from day 366 on). Fruiting wall: the
prescribed whip and feathers, hedging, side-shoot removal and heading, branch length cuts, leader heading at the wall
height, bud release below the heading cut, bevel-cut renewal and stub regrowth (the cheaper `short_wall` variant,
`tree_height = 2.0`, heads its leader in the first winter: by day 800 it has 1 heading, hedging, 6 side-shoot
removals, 5 length cuts, a renewal cut and ~100 release calls). The apple's light thinning only bites below 0.8 of
full exposure, which the young tree first reaches around day 740 (17 partly thinned clusters on seed 1 by day 740),
so partial thinning is checked by the 913 and 1460-day apples; above that every cluster is kept whole. Competitor pinching and the low-feather removal did
not trigger in any verified build, even over the 3650-day wall (ported line by line, but not oracle-exercised).

**The full `max_age` is verified for apple (1460), apple_fruitingwall (3650) and easternredbud (1460)** (seed 1
each; ~7, ~13 and ~3 min per case). **Not for the olive** (max_age 1825): Helios has 1.30 M primitives there (19 s);
the Heliades build passed 5.7 GB of memory before reaching it on a machine with 5 GB free and was stopped. The olive
is verified to day 1100 (third season, after the first leaf-lifespan shedding); its later years repeat the same
mechanisms. After P1, `AGES["olive"]` can take `(1825, 1)`.

Events in the verified 3650-day fruiting wall (seed 1, counted in Heliades): 1 leader heading, 79 hedging cuts,
61 side-shoot removals, 57 branch / side-shoot length cuts, 19 bevel-cut renewals and 19 regrown `renewal` shoots,
788 bud-release calls, 4 shade-shedding cuts; 529 of 796 shoots pruned by day 3650.

### Build time and size (seed 1, Helios vs Heliades, shared machine)

| model | age | primitives (Helios, visible) | Helios | Heliades |
|---|---|---|---|---|
| apple | 548 | 41 917 | 0.9 s | 15 s |
| apple | 913 | 128 554 | 3.3 s | 63 s |
| apple | 1460 | 548 384 | 22 s | ~6.5 min (case 433 s incl. Helios + snapshots) |
| apple_fruitingwall | 548 | 71 079 | 1.7 s | 30 s |
| apple_fruitingwall | 913 | 114 837 | 4.4 s | 83 s |
| apple_fruitingwall | 1460 | 177 570 | 11 s | ~3 min (case 222 s) |
| apple_fruitingwall | 3650 | 164 031 | 47 s | ~13 min (case 844 s) |
| olive | 548 | 46 936 | 0.3 s | 11 s |
| olive | 913 | 146 136 | 1.0 s | 40 s |
| olive | 1825 | 1 301 764 | 19 s | not run (memory) |
| easternredbud | 548 | 14 778 | 0.3 s | 9 s |
| easternredbud | 913 | 38 832 | 0.9 s | 32 s |
| easternredbud | 1460 | 139 776 | 5.3 s | ~3 min (case 190 s) |

Heliades is 15-40x slower than Helios here (pre-P1 core). The whole fruit-tree suite (all cases, all seeds) takes
roughly 1 h of Heliades + Helios on this shared machine, most of it in the max-age cases.

## Files

* `library/apple.py`: `initializeAppleTreeShoots`, `buildAppleTree`, `initializeAppleFruitingWallShoots`,
  `buildAppleFruitingWall` (+ its five build-parameter declarations), `AppleFruitPrototype`, `AppleFlowerPrototype`,
  `ApplePhytomerCreationFunction`, `ApplePhytomerCallbackFunction` (`applePhytomerCallback`,
  `thinAppleFlowerClusterForLight`), `AppleFruitingWallPhytomerCreationFunction`,
  `AppleFruitingWallPhytomerCallbackFunction` (`appleFruitingWallDimensions/HalfWidth/IsStub/Branches`,
  `appleFruitingWallWinterPruning`, `appleFruitingWallSummerTraining`).
* `library/olive.py`: initializer, builder, `olive_trunk_height`, `OliveLeafPrototype`, `OliveFruitPrototype`,
  `OliveFlowerPrototype`, `OlivePhytomerCreationFunction` (the last two not referenced by the model),
  `OlivePhytomerCallbackFunction`.
* `library/easternredbud.py`: initializer, builder, `RedbudFlowerPrototype`, `RedbudFruitPrototype`,
  `RedbudPhytomerCreationFunction` (empty), `RedbudPhytomerCallbackFunction`.
* `suites/plantarch/{ops,cases}_fruit_trees.py`, `tests/plantarch/test_fruit_trees.py`.

What W5-D listed as missing for W5-E: none of the four models enables epicormic shoots, uses `appendWoodDome`
(ported by W5-C anyway), `pruneBranch(keep_trimmed_material)` or leaf texture stages (only rice has stages), so none
of these were needed or ported. `getSeasonPeakLightExposureAtPoint` is now exercised and verified (apple thinning).

## Shared-file changes (additive, or filling a branch that raised)

* `core/random.py`: `MT19937` (std::mt19937), `GLOBAL_GENERATOR`, `seed_global_random_generator`, `global_randu`
  (the process-wide generator behind the free `helios::randu()`, global.cpp:360-388).
* `core/shapes.py`: `add_tile_textured` (`Context::addTile` with a texture: loose patches, zero-solid-fraction
  sub-patches skipped before they take a UUID; Context.cpp:2005-2104). It manipulates the new primitives the way
  the existing `add_tile_object_textured` does (`self.primitives[u]` + `scale/translate/rotate`): when P1 lands,
  rewrite it like P1's version of `add_tile_object_textured`.
* `plantarchitecture/inflorescence.py`: `record_fruit_class_map` calls `recordSpectralClassMap` (it raised).
* `plantarchitecture/phytomer.py`: the internode class map (`recordInternodeClassMap`, PA:3259-3265) after the tube is
  created or appended (it raised).
* `library/trees.py`: `setShootInternodeLengthMax` (PA:7787); `register_build_parameters` also registers
  `apple_fruitingwall` and `olive` (PL:516, 519).
* `library/__init__.py`: registers the four models.
* `tests/test_parity.py`: the slow fruit-tree cases in `SLOW_CASES` (fast: the four `params`, the build-parameter
  check, `olive.age0`, `easternredbud.age0`).

## Parity cases and tests

Per model `plantarch.<m>.load`, `.params`, `.age<A>`, `.data250` (primitive data + material labels, paths reduced
to base names), `.build_parameters_age{0,200}` (apple, fruiting wall, olive), `.steps` (apple, fruiting wall, redbud:
2-day steps across the first season boundary); `plantarch.apple_fruitingwall.short_wall_age{400,800,1100}`;
`plantarch.fruit_trees.build_parameters` (names, rejected maps incl. the float max of the wall's trunk height).
Every case seeds the Context and the process-wide generator with the same value (`global_seed` op).

Mutation checks (`test_mutations_are_caught`, each caught):

| mutation | caught by |
|---|---|
| redbud trunk rotation drawn with the float `randu` | `easternredbud.age0` |
| feather node by truncation instead of `std::round` | `apple_fruitingwall.age0` (seeds 1..2) |
| apple full-cluster light exposure 2.0 instead of 0.8 | `apple.age548` (bud states show in the next spring) |
| leader heading window 0.2 m instead of 0.5 m | `apple_fruitingwall.short_wall_age400` |
| olive lower leaf tile flipped about y instead of x | `olive.age170` |
| redbud `helios::randu(2, 4)` fixed to 2 | `easternredbud.age548` |

## Helios findings / discrepancies

1. **Redbud trunk rotation uses the integer `randu`** (PL:5343): `context_ptr->randu(0, 0.1 * M_PI)`,
   `randu(0, 2 * M_PI)`: an `int` and a `double` argument select `Context::randu(int, int)` (exact match on the
   first argument), so the trunk pitch is `randu(0, 0)` = 0 (no draw) and the yaw and roll are whole radians in
   {0, ..., 6}, not a uniform angle. Reproduced (the mutation with a float draw is caught at age 0).
2. **`RedbudPhytomerCallbackFunction` draws from the unseeded process-wide generator** (AS:2292,
   `int Nchild_shoots = randu(2, 4)`: the free `helios::randu`, a `std::mt19937` seeded from `std::random_device`).
   A redbud past its first dormancy is therefore not reproducible from the Context seed: two Helios runs with the
   same Context seed differ unless `helios::seedRandomGenerator` / PyHelios `Global.seedRandomGenerator` is called.
   It also draws on every dormant time step of every phytomer, for nothing after the first. Heliades has the same
   process-wide generator (`core.random.global_randu`, bit-identical mt19937 + `uniform_int_distribution`); the
   parity cases seed both.
3. **`RedbudLeaf.obj` is not used**: the redbud leaf is the generic prototype on `RedbudLeaf.png` (PL:5236), so the
   OBJ that does not load (HELIOS_ISSUES 17) never matters for the library model.
4. The olive's flower prototype is commented out (PL:4739): olive inflorescences carry fruit only (no flower
   geometry); `OliveFlowerPrototype` and `OlivePhytomerCreationFunction` are defined but unreferenced, and
   `RedbudPhytomerCreationFunction` is empty. Ported anyway.
5. `apple_fruitingwall` trunk height range: its max is the constexpr float `1.5f - 0.4f - 0.05f - 0.05f` =
   1.0000001, not 1.0 (accepted by both; 1.0000002 rejected by both).
6. The fruiting wall's competitor pinching (AS:1418) and low-feather removal (AS:1412) did not fire in any build
   checked (default wall to day 3650, short wall to day 1100, seed 1, with an event counter): the first needs
   a branch of exactly one dormancy cycle within 0.2 m of a still-growing leader's tip, the second a branch below the
   trunk height, which the builder never makes. Not a bug, but the rules are untested by the oracle.
7. The shadow grid's origin (`shadow_grid_origin`) only moves when a plant is translated; the light lookup of
   Heliades (`ShadowGrid.light_exposure`) ignores it, which is exact for plants built at the origin (all cases).

## Checks run

Fast suite `pytest -q tests`: 455 passed, 276 skipped (5.6 min on the shared machine; W5-E adds ~15 s: the unit
tests, two fast mutation checks, the four `params`, the build-parameter case and two age-0 builds). The shared-file
edits only act for models with internode or fruit class maps (apple only) or a textured `addTile` (olive only), so
other models are unaffected; the fast suite covers the existing fast cases.
