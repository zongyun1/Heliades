# W3-C: shoot growth (tomato `advanceTime`)

Baseline: Helios 1.3.98. Scope from SPEC §8: `advanceTime` and everything it drives on the tomato path,
`appendPhytomerToShoot`, `addChildShoot`, the per-phytomer growth setters, girth, and
`buildPlantInstanceFromLibrary(age > 0)`. Flower and fruit geometry is W3-D. Leaf blades are W3-A.

## What is ported

| file | contents | C++ |
|---|---|---|
| `plantarchitecture/growth.py` | `advanceTime` (sub-step schedule, max-age guard, the per-shoot loop: phytomer ageing, dormancy break, floral state machine with determinate growth, internode elongation, girth, petiole elongation + blade expansion with its `prototype_scale` draw, fruit-growth call, lateral bud break, node cap, phyllochron loop; `updateShootNodes(false)` per step, `volume_ratio` / `old_shoot_volume` object data, final `updateShootNodes(true)`), `addChildShoot`, `appendPhytomerToShoot` (node counters, `shoot_index.y` refresh, terminal floral buds), `incrementPhytomerInternodeGirth` (incl. `leaf_area` object data) | PA:11715-12465, 7239-7283, 7523-7593, 7625-7680 |
| `plantarchitecture/shoot_min.py` (extended) | `Shoot.updateShootNodes` (cached outward offset, `setTubeRadii`/`setTubeNodes`, `setPetioleBase`, children in `std::map` order), `childShootOutwardAxis`, `terminateApicalBud`, `terminateAxillaryVegetativeBuds`, `addTerminalFloralBud`, `getGrowthNodeCapPerSeason`, `calculateShootInternodeVolume`, `refreshDownstreamInflorescenceArea`, `sumDownstreamInflorescenceArea`, `isPruned`; `makeDormant` raises; new Shoot members (`cached_base_offset`, `last_parent_internode_axis`, ...) | PA:2264-2547, 2598-2640, 6999 |
| `plantarchitecture/phytomer.py` (extended) | `setInternodeLengthScaleFraction`, `setPetioleBase`, `setPetioleAndLeafScaleFraction` (+ `setPetioleScaleFraction`, `setLeafScaleFraction`), `getInternodeRadius(float)`, `hasLeaf`, `getExistingPetioleObjIDs`, `calculatePhytomerVolume`, `getInflorescenceArea`, `peduncleShouldMatchCulm`, `inflorescenceLoadsStem`, `deformLeafUnderSelfWeight` (rigid guard); `setFloralBudState` now calls `updateInflorescence` and the 0.25 fruit scale; W3-D seam (`updateInflorescence`, `setInflorescenceScaleFraction`, `updatePeduncleRadii`); `stub_draws_only` split out of the stub blade | PA:1371-1431, 1491-1547, 3603-3621, 3847-3880, 4199-4530, 5214, 6744, 6868 |
| `plantarchitecture/architecture.py` | entry points `advanceTime` (all plants / one / several), `addChildShoot`, `appendPhytomerToShoot`, `incrementPhytomerInternodeGirth`; `buildPlantInstanceFromLibrary(age > 0)` calls `advanceTime(plantID, age)` | PL:753-755, PA:11715-11726 |
| `plantarchitecture/plant.py` | `plant_structure_xml`: `writePlantStructureXML` text-identical for the tomato (`cout` = `ostream << float`) | InputOutput.cpp:519-1066 |

Skipped because they have no effect on the tomato (documented in `growth.py`): `bakePlantPose` (no pose
state), BVH cache/rebuild, the shadow grid / `sumPlantLeafArea` / `captureSeasonPeakLeaves` (feed only the
season-boundary shedding, 1000 days away), `flushDeferredPrunes`, leaf-angle tracking, epicormic shoots,
`adjustFruitForObstacleCollision`. Every optional subsystem raises `NotImplementedError` if enabled.

### W3-D seam (inflorescence)

The floral state machine is complete: clocks, initiation (`randu() < 0.2`), opening, fruit set (`randu() < 0.8`),
`terminateApicalBud` + `terminateAxillaryVegetativeBuds` on every event (determinate growth). A non-dead
reproductive state calls `Phytomer.updateInflorescence`, which **raises `NotImplementedError("... W3-D")`**.
With `pa.inflorescence_geometry_draws_only = True` it instead replays only the reads/resamples of
`updateInflorescence` + `createInflorescenceGeometry` in their order (6 `phyllotactic_angle` draws per build for
the tomato, the `randu(0, unique_prototypes-1)` prototype pick, all constant parameters' `val()`/`resample()`),
and `setInflorescenceScaleFraction` only records the scale. Then `pa.inflorescence_geometry_is_stub` is set.
No flower before day 56 for seeds 1..8 (`test_no_flowers_before_day_55`), so ages <= 54 never reach the seam.

## Parity cases (`cases_plantarch_growth.py`, `ops_plantarch_growth.py`): all pass, 0 ulp

| case | seeds | checks |
|---|---|---|
| `plantarch.tomato.steps` | 1..8 | ages 2..40 in 2-day calls; after each: engine calls (probe), the whole XML, base-shoot nodes (full precision), base-shoot petiole tube vertices; area-dependent checks gated (below) |
| `plantarch.tomato.steps_replay` | 1..8 | the same with *replayed blades*: every check every step: XML incl. radii, all shoots' nodes + radii, every internode/petiole tube object (shoot, objID, UUIDs, vertices), every leaflet (objID + 4x4), leaflet vertices at age 40 |
| `plantarch.tomato.split` | 1..4 | `advanceTime(20)` vs `10+10` vs `5x4` (replayed blades, full checks per variant) + flags 20==10+10 (true), 20==5x4 (false) |
| `plantarch.tomato.age50` | 1..8 | direct build at 50 (replayed blades, full checks incl. leaflet vertices) |
| `plantarch.tomato.age75_vegetative` | 1..8 | direct build at 75, draws-only inflorescence: engine calls, XML minus `<flower>`/`<peduncle>` records, all tube and leaflet geometry by value (UUIDs/objIDs shift by the missing flower objects) |

**Blade-area gating.** Blade area reaches the geometry only through the girth (`downstream_leaf_area`), and
the girth through (a) every internode radius and (b) the seat of each lateral (0.9 parent radii off the parent
axis), so everything on a lateral moves by ulps. The blade primitive count also sets every later UUID. The draw
stream and the shoot/phytomer/bud state do not depend on blade area. `_area_exact` = replayed blades, or the
real `GenericLeafPrototype` is bound (W3-A merged, `leaf_geometry_is_stub` false). With stub blades the case
masks `<internode_radius>` everywhere and, on laterals, `<internode_length>`, `<petiole_axis_initial>`,
`<petiole_rotation_axis>`, `<gravitropic_angles>`, `<azimuthal_restoring_angles>`, and skips tube UUIDs, the
lateral tubes and the leaflet frames. When W3-A lands, `plantarch.tomato.steps` runs the full checks unchanged.

**Replayed blades** (`ops_plantarch_growth.pa_build_replay_blades`): the oracle's 35 hidden prototypes of the same
seed are rebuilt primitive by primitive in Heliades: primitive transform set directly, texture + UV (solid
fraction computed by Heliades), the petiolule label, and the oracle's UUIDs (its builder leaves gaps, 152 UUIDs
per 144-primitive blade). The private-stream draws are the stub's. Blade areas then match the oracle bit for bit,
and so does everything downstream.

Engine calls (oracle = Heliades, fresh build at that age, `table.py`-style probe):

| age | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|
| 2 | 71 | 71 | 69 | 80 | 66 | 71 | 75 | 72 |
| 4 | 111 | 101 | 99 | 113 | 101 | 105 | 105 | 117 |
| 6 | 148 | 135 | 178 | 157 | 136 | 141 | 148 | 151 |
| 8 | 182 | 172 | 214 | 195 | 174 | 182 | 183 | 185 |
| 10 | 227 | 210 | 261 | 229 | 208 | 225 | 217 | 226 |
| 12 | 263 | 252 | 296 | 269 | 242 | 263 | 250 | 266 |
| 14 | 299 | 286 | 334 | 320 | 277 | 301 | 300 | 300 |
| 16 | 337 | 321 | 374 | 357 | 313 | 340 | 342 | 336 |
| 18 | 374 | 361 | 407 | 397 | 353 | 377 | 380 | 374 |
| 20 | 426 | 404 | 441 | 431 | 389 | 418 | 413 | 409 |
| 22 | 470 | 437 | 489 | 473 | 425 | 453 | 456 | 445 |
| 24 | 515 | 471 | 526 | 519 | 461 | 491 | 493 | 480 |
| 26 | 555 | 516 | 562 | 556 | 503 | 527 | 527 | 514 |
| 28 | 593 | 559 | 596 | 593 | 539 | 562 | 567 | 558 |
| 30 | 627 | 611 | 662 | 631 | 575 | 597 | 606 | 597 |
| 32 | 632 | 616 | 697 | 662 | 580 | 628 | 611 | 628 |
| 34 | 636 | 646 | 734 | 722 | 584 | 671 | 641 | 665 |
| 36 | 639 | 679 | 777 | 788 | 587 | 732 | 676 | 702 |
| 38 | 641 | 714 | 843 | 889 | 589 | 769 | 721 | 761 |
| 40 | 642 | 749 | 942 | 993 | 590 | 807 | 760 | 837 |
| 50 | 923 | 1105 | 1627 | 1578 | 812 | 1205 | 965 | 1594 |
| 56 | 1356 | 1407 | 2175 | 2004 | 1194 | 1550 | 1103 | 2177 |
| 60 | 1647 | 1663 | 2570 | 2326 | 1514 | 1780 | 1190 | 2596 |
| 75 | 2752 | 2474 | 3917 | 3439 | 2672 | 2591 | 1456 | 3882 |

Ages 56-75 with the draws-only inflorescence on Heliades. Every entry is equal on both backends.

Mutation checks (`tests/heliades/test_plantarch_growth.py::test_mutations_are_caught`), all caught:
- drop the per-blade `prototype_scale.resample()` (PA:12133): `steps`;
- lateral bud break at `>=` instead of `>`: `steps`;
- girth full step instead of `0.5 * (target - r)`: `steps_replay` (not visible with stub blades);
- `updateShootNodes` keeps the old lateral offset: `steps_replay`;
- `advanceTime(T <= pmin)` as one step of T instead of `int(T)` one-day steps: `steps`;
- `setInternodeLengthScaleFraction` moves only its own internode: `steps`.

## Discrepancies / findings

1. **SPEC §6 / §0: the first inflorescence is not "between days 75 and 100".** Seed 6 initiates (and, with
   p = 0.2, builds) a closed flower on day 56: the node-8 bud is created on day 16 and its `time_counter` reaches
   40 on day 56. By day 75, 6 of 8 seeds carry flowers or fruit. Ages <= 54 are flower-free for seeds 1..8.
2. **`advanceTime(2)` is two one-day sub-steps**, not one 2-day step: `T <= phyllochron_min` gives `Nsteps =
   int(T)`, `dt = 1` (PA:11789). `advanceTime(20)` = 10 steps of 2 (so 20 == 10+10), `advanceTime(5)` = 2+2+1.
3. **Petioles and blades grow separately** in 1.3.98 (SPEC §6 (vi) predates this): the petiole fraction advances on
   `elongation_rate_max`, the blade on `leaf_expansion_rate_max` (unset for the tomato, so the same rate), and a
   petiole that outgrows its blade makes no draw. `setPetioleAndLeafScaleFraction` floors the radius scale with
   `petioleRadiusFraction`.
4. **Laterals depend on girth**: `addChildShoot` and every `updateShootNodes` seat a lateral at the parent node
   + `0.9 * outward * parent_radius(1)`; the offset is cached and re-derived when the parent axis moves (> 1e-7)
   or the radius changes (> 1e-9); the shift is applied only above 1e-6.
5. **The stored peduncle radius of a culm-tip bud is reconciled** with the culm (`updatePeduncleRadii`, 0.9 x
   tip radius), so the XML `<peduncle><radius>` differs from `peduncle.radius` there (W3-D).
6. The oracle blade builder **consumes 8 UUIDs per blade that it deletes again** (152 UUIDs, 144 primitives).
   W3-A must reproduce this, or every UUID after the leaf cache shifts.
7. Literal types: girth `girth_area_factor * area * 1e-4` (double, narrowed), `radius + 0.5 * (...)` (double),
   `sqrtf` (correctly rounded); fruit growth `fmin(1, 0.25f + 0.75f * tc / dd)` is the double `fmin` (W3-D).
   The re-seat of lateral leaflets uses the constructor's `fabs` (double) offset; for the tomato's constants the
   float form gives the same bits (unverified, as in W3-B).
8. `make_AxisRotation(deg2rad(insertion), deg2rad(base_yaw.val()), deg2rad(base_roll.val()))` for a lateral is
   evaluated right to left (GCC), ported so; only `base_yaw` draws for the tomato, so the oracle cannot tell.
9. Not ported (raise): seasonal dormancy (`makeDormant`, shedding, harvest), leaf lifespan (`removeLeaf`),
   phytomer callbacks, epicormic shoots, pinned (posed) lateral offsets, prescribed leaf poses, flexible leaves.

## Outside my edit list

- `tests/heliades/test_plantarch_phytomer.py::test_age_past_zero_is_w3c` asserted the W3-B placeholder (age > 0
  raises "W3-C"). It is replaced by `test_age_past_zero_grows`; nothing else in that file changed.

## Requests for core (not done; `heliades/core` is owned by others)

- A public way to add a primitive with a given transform (the replay uses `Scene._new` / `_new_textured`).
- (from W3-B, still open) `powf` / `atan2f` in `core/vec.py`.

## What W3-D (inflorescence) needs

- Replace `Phytomer.updateInflorescence`, `setInflorescenceScaleFraction`, `updatePeduncleRadii` (now seams in
  `phytomer.py`), plus `getReconciledPeduncleRadius`, `bendPeduncleAxis`, `createInflorescenceGeometry`,
  `ensureInflorescencePrototypesInitialized`. Keep `_update_inflorescence_draws`'s read/resample order: it is
  oracle-verified by the draw counts at day 75 (8 seeds). Store `peduncle_length/radius/...` for the XML.
- `setPetioleBase` already moves `inflorescence_objIDs`, `inflorescence_bases`, peduncle objects and vertices.
- The XML writer raises on `inflorescence_bases`; add the `<flower>` records.
- Cases: drop `pa_draws_only_inflorescence` from `age75_vegetative` and compare flowers too; ages 100/120.

## What W3-E (library build + acceptance) needs

- `buildPlantInstanceFromLibrary(pos, age)` grows to any age; `plant.plant_structure_xml` is ready as the
  structure oracle.
- Two plants in one PlantArchitecture: the code paths exist (shared leaf cache), not yet parity-tested.
- Per-step cost: ~0.4 s for a stub-blade tomato at 40 days, ~3 s with full blades (leaf translations dominate).
