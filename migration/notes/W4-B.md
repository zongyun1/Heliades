# W4-B: bean and soybean

Baseline: Helios 1.3.98. Scope: `initializeBeanShoots` / `buildBeanPlant` (PL:2911-3069), `initializeSoybeanShoots` /
`buildSoybeanPlant` (PL:5745-5897), their Assets.cpp functions (AS:1578-1637, 4031-4066), and every shared branch
they reach that the tomato did not.

## Code paths the legumes reach and the tomato does not (written before porting)

Status: **[new]** absent from Heliades and ported here; **[first]** already ported (W3-B/C) but never oracle-checked,
now covered by these cases; **[W3-D]** inflorescence geometry, gated.

| # | path | C++ | status |
|---|---|---|---|
| 1 | `PlantArchitecture::appendShoot`: parent growth cap + `max_nodes` = node count, `terminateApicalBud`, base on the parent tip, `is_axis_continuation`, inherited `internode_radius_cap` | PA:7178-7237 | **[new]** `shoot_min.appendShoot` + `architecture.PlantArchitecture.appendShoot` |
| 2 | axis continuation in `Shoot::updateShootNodes` (zero cached offset), child of node 0 of the unifoliate shoot | PA:2429-2547 | [first] |
| 3 | `growth_state.max_nodes_growth_cap >= 0` in `getGrowthNodeCap` (the unifoliate stops at 1 node) | PA:2598 | [first] |
| 4 | two petioles per internode (unifoliate node): the `petiole > 0` bud rotation `2*pi*p/npet`, per-petiole buds and child-type samples, two leaf caches of 1 leaflet | PA:3300-3600 | [first] |
| 5 | `leaves_per_petiole == 1`: `orientLeaf` single-leaf roll sign and the blade elevation correction (`atan2f`, petiole/leaf length ratio, clamp) | PA:5016-5197 | [first] |
| 6 | petiole too small to build (bean unifoliate: 0.1 mm long, `makePetioleTube` returns `no_petiole_objID`), then **lazily created** in `setPetioleAndLeafScaleFraction` when it has grown | PA:4465-4477, 12879 | [first] |
| 7 | per-leaflet textures `{-1: right, 0: tip, 1: left}` (bean), one automatic material per texture | AS:640-924 | [first] (W3-A edge case 8 with tomato) |
| 8 | blade shapes outside the tomato's: lateral curvature -1 / +0.45, no wave, 8 subdivisions, positive longitudinal curvature | AS:517-924 | [first] |
| 9 | `leaf.pitch` normal distribution (bean: Context `randn`), distributed `leaf.roll` (soybean) | PA:3510-3530 | [first] |
| 10 | leaflet offsets 0.3 / 0.5 (`(fabs(ind) - 0.5f) * offset * length` in double, W3-B finding 2) | PA:3555 | [first] |
| 11 | petiole taper 0 (bean), 0.25 (soybean); petiole radius 0.4 mm | PA:3406 | [first] |
| 12 | base rotation with a non-zero pitch (3 `randu`, right to left): the `shoot_base_rotation.pitch != 0` branch of the first phytomer and of `childShootOutwardAxis` | PL:3058, PA:2900-2955, 2366 | [first] |
| 13 | no tortuosity (gravitropism only, OU drive 0, but the 2 `randn` per segment are still drawn) | PA:3000-3080 | [first] |
| 14 | null `phytomer_creation_function` (both unifoliates) | PA:2203 | [first] |
| 15 | `BeanPhytomerCreationFunction` / `SoybeanPhytomerCreationFunction` (rank- and node-based bud kills, double `fmin`) | AS:1616, 4048 | **[new]** library |
| 16 | `RandomParameter_int` uniform `max_nodes` (4-7, 16-19, 3-7), distributed `insertion_angle_tip` / `flower_bud_break_probability` | | [first] |
| 17 | `vegetative_bud_break_probability_decay_rate == 0` (soybean) and `< 0` with `min = 1` (bean unifoliate) | PA:7065 | [first] |
| 18 | `determinate_shoot_growth = false` (soybean) | PA:11970 | [first] |
| 19 | a child shoot type different from the parent's (soybean `trifoliate -> branch`, both `unifoliate -> trifoliate`), three shoot types in the snapshot map | PA:7013 | [first] |
| 20 | `max_floral_buds_per_petiole = 0` (unifoliate) | PA:2666 | [first] |
| 21 | `BeanFlowerPrototype`, `BeanFruitPrototype`, `SoybeanFlowerPrototype`, `SoybeanFruitPrototype` (OBJ, the 2-argument `loadOBJ` = RGB::blue default colour for the bean, RGB::black for the soybean) | AS:1599-1614, 4031-4046 | **[new]** library; gated until W3-D merged, now oracle-checked |
| 22 | distributed inflorescence parameters (`flowers_per_peduncle` 1-4, pitch, roll, fruit scale, gravity factor), peduncle of 1 segment | | [first] (W3-D's code) |
| 23 | `BeanLeafPrototype_unifoliate_OBJ` / `_trifoliate_OBJ` (OBJ leaves) | AS:1578-1597 | **[new]**, but no library entry uses them (the bean uses `GenericLeafPrototype` blades); not oracle-tested |

`cowpea` (out of scope) is the bean's structure with other values: the same two shoot types, `appendShoot`, the
same creation-function shape and `GenericLeafPrototype` blades. `library.bean.build_legume` and `appendShoot` would serve
it unchanged. It also declares a leaf-inclination Beta (1.398, 1.574), so `buildPlantInstanceFromLibrary` reaches the
leaf-inclination tracking, which still raises `NotImplementedError` (not ported by anyone yet).

## What is ported

| file | contents | C++ |
|---|---|---|
| `plantarchitecture/library/bean.py` | `initializeBeanShoots`, `buildBeanPlant` (via `build_legume`, shared with the soybean), `base_rotation_draws` (3 `randu`, right to left), `BeanPhytomerCreationFunction`, `BeanFlowerPrototype`, `BeanFruitPrototype`, `BeanLeafPrototype_{unifoliate,trifoliate}_OBJ`; new `hooks.Hook` slots defined in the module (hooks.py untouched) | PL:2911-3069, AS:1578-1637 |
| `plantarchitecture/library/soybean.py` | `initializeSoybeanShoots` (3 shoot types), `buildSoybeanPlant`, `SoybeanPhytomerCreationFunction`, `SoybeanFlowerPrototype`, `SoybeanFruitPrototype` | PL:5745-5897, AS:4031-4066 |

Literal types: `0.05f * M_PI` and `0.5f * M_PI` are double products narrowed by the float parameter (the float form gives
the same bits, `0x1.41b2f8p-3`, so it is not observable). The creation functions' `fmin(1.f, 0.6 + 0.4 * age / 8.f)` and
the like are double (caught when made float: `bean_leaf_scale_float`). In `appendShoot(..., radius_initial.val(),
internode_length_max.val(), ...)` the arguments are evaluated right to left; both are constants, so no draw.

## Shared-file hunks (all append-only / previously absent)

| file | hunk |
|---|---|
| `plantarchitecture/shoot_min.py` | new function `appendShoot` (PA:7178-7237) inserted before `breakPlantDormancy`; `"appendShoot"` added to `__all__` |
| `plantarchitecture/architecture.py` | new method `PlantArchitecture.appendShoot` delegating to `shoot_min.appendShoot` (inserted before `breakPlantDormancy`) |
| `plantarchitecture/library/__init__.py` | 6 appended lines: import `bean`, `soybean`; `SHOOT_INITIALIZERS` / `PLANT_BUILDERS` entries for both |

No edit to `phytomer.py`, `growth.py`, `leaf_prototype.py`, `tube_utils.py`: every other path in the table above was
already ported and is exact as it stands.

## Parity cases (`cases_plantarch_w4b.py`, `ops_plantarch_w4b.py`). All pass, 0 ulp, Context seeds 1..8

| case | bean | soybean | checks |
|---|---|---|---|
| `plantarch.<crop>.load` | PASS | PASS | engine calls after load (16 = 5 identifiers, both crops, all seeds), every shoot type's ShootParameters (PyHelios JSON, float32 canonical) |
| `.age0_full` | PASS | PASS | `snapshot(include_hidden=True)`: bean 3566 prims / 25 objects (17 hidden blades = 5x3 + 2x1), soybean 5100-5220 prims / 29 objects (20 hidden) |
| `.age0_full_attrs` | PASS | PASS | materials, labels, primitive / object data, polymesh face sets of every polymesh |
| `.steps` | PASS | PASS | 2-day `advanceTime` to day 30: after each call engine calls, whole XML, every shoot's nodes + radii, every internode / petiole tube (IDs, UUIDs, vertices), every leaflet (ID, 4x4); leaflet vertices on day 30 |
| `.steps_full` | PASS | PASS | full snapshot after the 15 steps |
| `.gallery` / `.gallery_full` | PASS (45 d) | PASS (50 d) | direct build: `steps` observation with leaflet vertices / full snapshot |
| `.flowering` / `.flowering_full` | PASS (60 d) | PASS (70 d) | direct build past the first flowers: number of flower/fruit buds, observation + every peduncle / flower / fruit object (`inflo_plant_organs`) / full snapshot |

Engine calls (Heliades = oracle for every entry; probe after the build):

| crop, age | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
|---|---|---|---|---|---|---|---|---|
| bean 45 | 1613 | 1449 | 1465 | 1173 | 2360 | 2170 | 1520 | 1606 |
| bean 60 | 1795 | 1565 | 1676 | 1373 | 2484 | 2338 | 1713 | 1772 |
| soybean 0 | 71 | 65 | 65 | 68 | 69 | 65 | 80 | 65 |
| soybean 50 | 1962 | 2402 | 2481 | 2544 | 1766 | 1708 | 1677 | 1360 |
| soybean 70 | 2363 | 2726 | 3030 | 3078 | 1951 | 2184 | 2053 | 1590 |

Flower / fruit buds (states 2-4) at 60 d (bean) 7, 6, 8, 5, 4, 6, 7, 7; at 70 d (soybean) 8, 10, 12, 7, 7, 6, 8, 7.
No seed has a flower by the gallery ages (bean: the first floral bud is at trifoliate node 2, about day 6, plus 40 days
to initiation; soybean: node 5 about day 17), so the gallery pairs and all steps are vegetative.

**Inflorescence gate.** Before W3-D merged, a (seed, age) pair whose plant held a flower or fruit was compared without the
flower geometry (Heliades in draws-only mode: draws, XML minus `<flower>` / `<peduncle>`, tubes and leaves by value);
it passed for all 16 flowering pairs. With W3-D merged, `ops_plantarch_w4b.inflorescence_ported()` is true, no pair is
gated, and every flowering pair runs all checks including the organs and the full snapshot.

## Mutation checks (`tests/heliades/test_plantarch_w4b.py::test_mutations_are_caught`), all caught

| mutation | caught by |
|---|---|
| `appendShoot` without `is_axis_continuation` | bean.steps |
| `appendShoot` without the parent growth cap | soybean.steps |
| base rotation drawn left to right | bean.age0_full |
| second petiole of a node not rotated by `2*pi*p/npet` (shared `phytomer.py`) | bean.age0_full |
| single-leaf blade correction reversed (shared `phytomer.py`) | soybean.age0_full |
| bean leaf scale in float instead of double | bean.steps |
| soybean branching nodes `< 4` instead of `< 5` | soybean.steps |
| soybean closed flower with its own open mesh instead of the bean's closed one | soybean.flowering_full |
| bean open / closed flower meshes swapped | bean.flowering_full |

Not observable (so not listed): `0.05f*M_PI` in float (same bits); the bean pod's 2-argument `loadOBJ` (RGB::blue
default) vs the black overload (every face of `BeanPod.obj` has a material); the single-leaf correction's node sign (the
unifoliate is always phytomer 0, sign +1).

## Discrepancies / findings

1. **Parity-op bug (W3-C, `ops_plantarch_growth.py`, not edited):** the Helios side of `pa_tube_objects("petiole")` zips
   the petiole objects with shoot labels derived from the node counts (one petiole per phytomer). On a plant with
   more petiole objects than phytomers (the unifoliate node has two), `zip` silently truncates the list, so the cases
   disagreed in length although the plants were identical. The legume cases use `w4b_petiole_objects` (plant order,
   no shoot label) instead. The tomato is unaffected (one petiole per node, every petiole built).
2. **W3-B finding 2 (`fabs` in the leaflet offset) is still unverified:** for a trifoliate `|ind| - 0.5 = 0.5`, so the
   double and float forms are bit-identical for any offset and length (probe: 0 of 200 000 random lengths differ). It
   needs a species with 5+ leaflets and non-tomato constants.
3. The bean unifoliate petiole (0.1 mm) builds no tube at age 0 and is created lazily by `setPetioleAndLeafScaleFraction`
   (between days 8 and 10 for seed 1): the object IDs of every later organ depend on it, and they match.
4. `cowpea` shares the structure (see the table above) but also needs leaf-inclination tracking, which raises.
5. No core requests.

## Outside my edit list

- `tests/heliades/test_plantarch_params.py::test_registry_and_errors` used `bean` as its example of an unported model
  (`loadPlantModelFromLibrary` raises `NotImplementedError`). It now uses `cowpea` (line 243); nothing else changed.

## Verification

`pytest -q tests`: 376 passed, 1 skipped (after the line above; the run before it had that 1 failure).
`python -m heliades.parity plantarch`: 48/48 cases pass, including the 18 W4-B cases and W3-D's `tomato.flowering`.
