# W4-A: cherry tomato and capsicum

Baseline: Helios 1.3.98. Scope: the `cherrytomato` and `capsicum` library entries (PL:6226-6359, PL:3207-3331), their
phytomer creation / callback functions and fruit prototype (AS:4111-4140, AS:1650-1683), and every shared branch they
reach that the tomato did not.

## Code paths these crops reach that the tomato does not (written before porting)

Unported in Heliades (raised `NotImplementedError` or absent):

| path | who reaches it | status before |
|---|---|---|
| `Phytomer::removeLeaf` (PA:6761-6803) | capsicum creation function (main-stem nodes 0-5, at creation); cherry tomato callback (day >= 101); leaf lifespan in `advanceTime` | absent / raised |
| phytomer callback call in `advanceTime` (`PhytomerCallbackScope`, PA:11935-11938) | cherry tomato, every phytomer every sub-step | raised |
| `writePlantStructureXML` `<shed_petiole>` (InputOutput.cpp:1020-1042) | capsicum (a phytomer with `petiole_length` empty, `petiole_vertices` kept) | raised |
| `initializeCherryTomatoShoots`, `buildCherryTomatoPlant`, `CherryTomatoPhytomerCreationFunction`, `CherryTomatoPhytomerCallbackFunction` | cherry tomato | absent |
| `initializeCapsicumShoots`, `buildCapsicumPlant`, `CapsicumPhytomerCreationFunction`, `CapsicumFruitPrototype` | capsicum | absent |
| `AlmondFlowerPrototype` (capsicum's flower function; its prototypes are built with the fruit cache even though capsicum flowers never open) | capsicum inflorescence | absent |

Ported already but never exercised by a tomato case (now oracle-checked by the W4-A cases):

* `orientLeaf` single-leaf branch (`leaves_per_petiole == 1`: sign-alternating roll, the blade correction about the
  petiole tip axis), a blade without petiolule, a vestigial petiole (0.1 mm), negative petiole pitch, zero petiole
  curvature (capsicum);
* two shoot types whose child type differs from their own (`mainstem` -> `lateral` / `secondary`), a shoot type
  without creation function (capsicum `secondary`), `determinate_shoot_growth = false` (cherry tomato),
  `sampleVegetativeBudBreak` with decay 0 (`p = probability_min`, cherry tomato `lateral` p = 0 still draws);
* distributed `leaf.roll` (cherry tomato, one draw per leaflet), 9 leaflets, 14 internode radial subdivisions;
* `advanceTime` schedules with `phyllochron_min` 3 / 4 / 6 (direct builds: `floor(T / pmin)` steps + remainder).

## What is ported

| file | contents | C++ |
|---|---|---|
| `library/cherrytomato.py` | `initializeCherryTomatoShoots`, `buildCherryTomatoPlant`, `CherryTomatoPhytomerCreationFunction`, `CherryTomatoPhytomerCallbackFunction` (hooks of the same names) | PL:6226-6359, AS:4111-4140 |
| `library/capsicum.py` | `initializeCapsicumShoots`, `buildCapsicumPlant`, `CapsicumPhytomerCreationFunction`, `CapsicumFruitPrototype` (`randn() < 0.4` picks the green/red OBJ, `loadOBJ` height 0 Z-up, `addPolymeshObject`), `AlmondFlowerPrototype` (bound only if no almond port bound it first; same code) | PL:3207-3331, AS:1650-1683, AS:934-938 |
| `phytomer.py` | `Phytomer.removeLeaf` | PA:6761-6803 |
| `growth.py` | callback call (depth counter = `PhytomerCallbackScope`), leaf-lifespan `removeLeaf` | PA:11935-11938, 11959-11965 |
| `plant.py` | `<shed_petiole>` with `<centerline_offsets>` and the floral buds of that petiole index | InputOutput.cpp:1020-1042 |

Hooks: `hooks.py` is not edited. `library/cherrytomato._hook(name)` returns `hooks.<name>`, creating and installing it
on the `hooks` module the first time, so a crop ported elsewhere that uses the same function (e.g. the almond's
`AlmondFlowerPrototype`) gets the same identity-compared pointer as long as it looks it up the same way (or W3-D adds it
to `hooks.py` before; `_hook` then reuses it).

### Shared-file hunks (all of them)

1. `heliades/plantarchitecture/phytomer.py`: new method `Phytomer.removeLeaf(detached_objIDs=None)` inserted before
   `getExistingPetioleObjIDs` (absent before). Nothing else.
2. `heliades/plantarchitecture/growth.py`, `_advance_plant_one_step`: `raise NotImplementedError("phytomer callback
   functions ...")` -> the callback call inside `pa.phytomer_callback_depth += 1 / -= 1`.
3. `heliades/plantarchitecture/growth.py`, `_advance_plant_one_step`: `raise NotImplementedError("Phytomer::removeLeaf
   (leaf lifespan) ...")` -> `phytomer.removeLeaf()` (raises if the nitrogen model is on: `resorbLeafNitrogen`).
4. `heliades/plantarchitecture/plant.py`, `_write_phytomer`: `raise NotImplementedError("writing shed petioles ...")`
   -> the `<shed_petiole>` writer. **`plant.py` is not in my list of shared files** (it is W3-D's); the hunk replaces
   one raise, 10 lines, and is needed for any capsicum XML. W3-D: keep it when you add `<flower>`.
5. `heliades/plantarchitecture/library/__init__.py`: 4 lines appended (import, `SHOOT_INITIALIZERS.update`,
   `PLANT_BUILDERS.update`). `architecture.py` untouched (`register_all` picks the builders up).

No tomato-covered behaviour changed: the tomato has no callback, never exceeds its leaf lifespan and never sheds a leaf.

## Parity cases (`cases_plantarch_w4a.py`, `ops_plantarch_w4a.py`): all pass, 0 ulp

Real W3-A blades on both sides (no stub, no replay): blade areas, girth, lateral seating and UUIDs are compared as is.
Context seeds 1..8 (case seed s -> Context seed s+1).

| case | cherrytomato | capsicum | checks |
|---|---|---|---|
| `plantarch.w4a.C.load` (exact) | 8 seeds | 8 seeds | engine calls after ctor / load / second load; every shoot type's ShootParameters (PyHelios JSON, float32) |
| `plantarch.w4a.C.age0_full` | 8 seeds, 0 ulp | 8 seeds, 0 ulp | `snapshot(include_hidden=True)`: every primitive and object incl. the hidden blade prototypes |
| `plantarch.w4a.C.age0_full_attrs` (exact) | 8 | 8 | materials, primitive data, polymesh face sets of every polymesh object |
| `plantarch.w4a.C.steps` | 8 seeds, ages 2..40 | 8 seeds, ages 2..40 | after every 2-day call: engine calls, whole XML, all shoots' nodes + radii, every internode/petiole tube (objID, UUIDs, vertices), every leaf (objID, 4x4), inflorescence object count; leaf vertices at 40 |
| `plantarch.w4a.C.steps_late` | 8 seeds, ages 42..80 | 8 seeds, ages 42..80 | the same, continuing the sweep (cherry tomato laterals: Context seeds 2, 3; capsicum `secondary` shoots) |
| `plantarch.w4a.C.gallery` | 8 seeds, direct build at 75 | 8 seeds, direct build at 80 | the same observations, leaf vertices included |
| `plantarch.w4a.C.gallery_full` | 8 seeds, 75 | 8 seeds, 80 | `snapshot(include_hidden=True)` of that build |
| `plantarch.w4a.C.reproductive` (after W3-D) | 8 seeds x 90, 110 d (flowers, fruit; 110 = callback pruning below 1 m) | 8 seeds x 130, 160 d (fruit) | the observation above + every peduncle / flower / fruit object (ID, UUIDs, 4x4, vertices) |
| `plantarch.w4a.C.reproductive_full` | the same builds | the same builds | `snapshot(include_hidden=True)` incl. the hidden flower/fruit prototypes |
| `plantarch.w4a.literal_probe` | ages 1, 3, 7, 13 | age 1 (8 nodes) | `addPlantInstance(pos, age)` + `addBaseStemShoot` + 2 days: the plant ages where the float and double forms of the creation functions' scales differ |

Max ulp everywhere: 0. Gated pairs: none. W3-D was merged during this package: the reproductive pairs (cherry tomato
90 / 110, capsicum 130 / 160, Context seeds 1..8) were added and run ungated. The gate: a (crop, age, seed) that would build
inflorescence geometry returns `[-999]` on both sides unless `heliades.plantarchitecture.inflorescence` (W3-D) exists;
the inflorescence object count is in every observation, so a pair that should have been gated fails instead of passing
silently. Measured with the draws-only seam: no peduncle/flower/fruit up to day 80 (1- or 2-day steps) for either crop,
nor in a direct build at 75 / 80; the first cherry tomato flowers come in a direct build at day 82 (Context seeds 3, 4,
6, 7).

Engine calls / shoots / UUIDs in use after a direct build (Context seeds 1..8; oracle = Heliades):

| crop, age | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|
| cherrytomato 0 | 52/1/1646 | 48/1/1646 | 48/1/1646 | 48/1/1646 | 50/1/1646 | 48/1/1646 | 56/1/1646 | 48/1/1646 |
| cherrytomato 20 | 313/1/9876 | 316/1/9876 | 295/1/9876 | 293/1/9876 | 298/1/9876 | 290/1/9876 | 323/1/9876 | 296/1/9876 |
| cherrytomato 40 | 563/1/18106 | 572/1/18106 | 549/1/18106 | 540/1/18106 | 560/1/18106 | 555/1/18106 | 577/1/18106 | 547/1/18106 |
| cherrytomato 75 | 982/1/31274 | 976/1/31274 | 955/1/31274 | 945/1/31274 | 975/1/31274 | 972/1/31274 | 1019/1/31274 | 980/1/31274 |
| capsicum 0 | 25/1/14 | 21/1/14 | 21/1/14 | 21/1/14 | 23/1/14 | 21/1/14 | 29/1/14 | 21/1/14 |
| capsicum 20 | 154/1/130 | 162/1/130 | 141/1/130 | 146/1/130 | 163/1/130 | 192/1/130 | 160/1/130 | 140/1/130 |
| capsicum 40 | 340/1/494 | 352/1/480 | 307/1/480 | 318/1/494 | 339/1/494 | 353/1/480 | 353/1/466 | 308/1/466 |
| capsicum 80 | 1087/9/2166 | 1330/10/2626 | 1171/10/2382 | 1066/9/2226 | 1212/10/2572 | 1246/10/2572 | 1391/12/2750 | 1260/10/2534 |

(Engine calls include the load; capsicum age 0 has 14 UUIDs because its only leaf is removed at creation.)

## Literal types (each proven)

| expression | evaluated as | proof |
|---|---|---|
| cherry `fmin(1.f, 0.7 + 0.3 * plant_age / 15.f)` (leaf) | double throughout (left to right, `0.3 * plant_age` is double), double `fmin`, narrowed | `literal_probe` ages 1, 7, 13; mutation `cherry_leaf_scale_float` |
| cherry `fmin(1.f, 0.7 + 0.3 * plant_age / 10.f)` (internode) | the same | `literal_probe` ages 3, 7; mutation `cherry_inode_scale_float` |
| capsicum `std::min(1.f, 0.05f + 0.95f * plant_age / 15.f)` | float throughout, `std::min<float>` | `literal_probe` capsicum age 1; mutation `capsicum_inode_scale_double` |
| capsicum `std::min(1.f, 0.6f + 0.4f * shoot_node_index / 5.f)` | float (`uint` converted to float) | **not distinguishable**: the float and double forms differ only at node 3, whose main-stem leaf is removed, and `secondary` shoots have no creation function |

A library build never shows the first three either: phytomers are created only at multiples of the phyllochron
(4 / 3 days), where both forms round the same. Hence the direct-build probe.

## Mutation checks (`tests/heliades/test_plantarch_w4a.py::test_mutations_are_caught`), all caught

- cherry tomato leaf scale in float: `literal_probe`;
- cherry tomato internode scale in float: `literal_probe`;
- capsicum internode scale in double: `literal_probe`;
- capsicum creation function without `removeLeaf()`: `capsicum.age0_full`;
- `removeLeaf` also clears `petiole_vertices` (C++ keeps them): `capsicum.steps`;
- capsicum `secondary` keeps the main stem's phytomer parameters (with the creation function): `capsicum.gallery`;
- `<centerline_offsets>` measured from the tip instead of the base: `capsicum.steps`;
- cherry tomato main stem with child type `mainstem` instead of `lateral`: `cherrytomato.load`.

- `CapsicumFruitPrototype` without its `randn()` draw: `capsicum.reproductive`;
- cherry tomato callback pruning below 0.8 m instead of 1 m: `cherrytomato.reproductive` (110 days).

Unit tests also check that it runs once per phytomer per sub-step inside the callback depth, and that
it removes the leaves of a node below 1 m once the plant is 101 days old.

## Discrepancies / findings

1. **Capsicum's shed leaves keep counting for girth.** `removeLeaf` does not reduce `downstream_leaf_area` (the
   `propagateDownstreamLeafArea(..., -getLeafArea())` call is commented out, PA:6767), and the capsicum creation function
   removes the leaf *after* `appendPhytomer` has added its area. The first six main-stem internodes are thickened by
   leaves that never existed. Ported as is; candidate for `HELIOS_ISSUES.md` (outside my edit list).
2. **Capsicum `secondary` shoots have no phytomer creation function.** `phytomer_parameters_secondary` is copied before
   `CapsicumPhytomerCreationFunction` is assigned, and `shoot_parameters_secondary.phytomer_parameters =
   phytomer_parameters_secondary` overwrites the copied pointer with null. So secondaries get no leaf-size ramp, and
   their rank >= 2 children would not be killed (they never break before day 80 anyway: p = 0.06). Ported as is; looks
   unintended.
3. **Cherry tomato `lateral.vegetative_bud_break_probability_max = 0` has no effect**: with decay rate 0,
   `sampleVegetativeBudBreak` uses `probability_min` (already 0). Each lateral bud still makes one `randu()`.
4. **Few cherry tomato laterals.** With p = 0.05 and a 40-day break time, only Context seeds 2 and 3 grow a lateral by
   day 80 in 2-day steps, and none does in a direct build up to day 82 (4-day sub-steps). `steps_late` is the case
   that exercises the `lateral` type.
5. **The draws-only inflorescence seam does not replay prototype-function draws.** `CapsicumFruitPrototype` makes a
   Context `randn()` each time it builds a prototype (10 per cache fill). With W3-D's real geometry this is exact
   (`capsicum.reproductive`); the draws-only mode would drift for capsicum.
6. Capsicum's flower prototypes (10 closed + 10 open `AlmondFlowerPrototype`) are built with the fruit cache even
   though its flowers never appear (flower initiation / opening thresholds are -1: buds go straight to fruit set).
   No fruit before day 123 for seeds 1..8 (main-stem nodes 0-5 are dead; fruit set needs 75 days of bud time).
7. `buildCapsicumPlant` passes `shoot_types.at("mainstem").internode_length_max.val()` (constant, no draw) where the
   tomato passes a literal.

## Requests for core (not done)

None new. (W3-B/W3-C's `powf` / `atan2f` in `core/vec.py` still stand.)

## After W3-D

Merged mid-package; the reproductive cases exercise, beyond the tomato: 10 unique inflorescence prototypes,
distributed `peduncle.pitch`, `inflorescence.roll` and `fruit_prototype_scale`, a non-zero
`fruit_gravity_factor_fraction` (0.9 capsicum, 0.2 cherry tomato: the gravity rotation W3-D could not test), the
`CapsicumFruitPrototype` draws, and the cherry tomato callback's pruning (`removeLeaf` from inside a callback).
