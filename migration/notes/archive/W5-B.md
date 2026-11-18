# W5-B: herbaceous vegetables (cowpea, cucumber, butterlettuce, sugarbeet, asparagus)

Baseline: Helios 1.3.98 (`14860044f`). Scope: `initialize<X>Shoots` / `build<X>Plant` and their Assets.cpp functions for
the five models, plus every engine path they reach that no earlier package had ported.

## What is ported

| file | contents | C++ |
|---|---|---|
| `library/butterlettuce.py` | `initializeButterLettuceShoots`, `buildButterLettucePlant`, `ButterLettucePhytomerCreationFunction` (`rotateLeaf` by -60 deg x (max-node)/max, uint arithmetic) | PL:5510-5588, AS:2407-2417 |
| `library/sugarbeet.py` | `initializeSugarbeetShoots`, `buildSugarbeetPlant` | PL:6017-6099 |
| `library/asparagus.py` | `initializeAsparagusShoots`, `buildAsparagusPlant`, `AsparagusLeafPrototype` (tube cladode shaped from the *prototype's* generator, i.e. the private `minstd_rand0(identifier)` stream of the cache build), `AsparagusPhytomerCreationFunction` | PL:2696-2798, AS:1531-1576 |
| `library/cucumber.py` | `initializeCucumberShoots`, `buildCucumberPlant`, `CucumberFruitPrototype` / `CucumberFlowerPrototype` (6 baked OBJ variants, 1 `randu(int,int)` each), `CucumberMainPhytomerCreationFunction` (tomato's + `setPrimitiveBumpMap`), `CucumberMainPhytomerCallbackFunction` (harvest, high-wire lowering and leaning), `cucumberLoweredStemPath`, `CucumberStringAttractionPoints`; `PlantArchitecture::enableAttractionPoints` / `updateAttractionPoints` (module functions) | PL:6689-6824, AS:4262-4458, PA:13455-13511 |
| `library/cowpea.py` | `initializeCowpeaShoots`, `buildCowpeaPlant` (bean structure, `appendShoot`), `CowpeaPhytomerCreationFunction`, `CowpeaFlowerPrototype`, `CowpeaFruitPrototype`, `CowpeaLeafPrototype_{unifoliate,trifoliate}_OBJ` (unused by the library) | PL:3409-3566, AS:1692-1757 |
| `plantarchitecture/leaf_angle_tracking.py` (new) | leaf inclination tracking: `enablePlantLeafElevationAngleDistributionTracking(_private)`, `disable...`, `updatePlantLeafAngleDistributionTracking`, `leafBladeNormalCached`, `depositLeafFacets`, `chooseLeafAngleTarget`, `rodrigues`, `minimalRotationBetween`, `Phytomer::setLeafNormal`, `nearestPointOnPolyline`; `helios::cart2sphere`, `sphere2cart`, `atan2_2pi`, `evaluate_Beta_distribution_CDF` (Lentz, libm `lgamma`/`log`/`log1p`/`exp` via ctypes). Elevation tracking only (azimuth raises). | PA:4616-4842, 8090-8754, global.cpp |

## Shared-file changes (all additive)

| file | hunk |
|---|---|
| `core/materials.py` | `Material` gains `normal_map_file/kind/strength/flip_green`; `generate_material_label(..., normal_map)` hashes the 4 extra fields when kind != NONE (Context.cpp:4159); `_set_primitive_appearance(..., normal_map=None)` (None = keep the current material's, as `getMaterialAppearance(current)` does); `_copy_material_for_primitive` copies the map; new `set_primitive_bump_map`, `get_primitive_normal_map_file`, `get_material_normal_map_{file,kind,strength}`. Every existing material has kind NONE, so no existing label or behaviour changes. |
| `plantarchitecture/architecture.py` | `buildPlantInstanceFromLibrary`: enables elevation tracking (lambda 180) for a model with a registered Beta instead of raising (PL:743-746) |
| `plantarchitecture/growth.py` | pending-cohort appends after a child shoot (PA:12218) and after a new phytomer (PA:12265), `updatePlantLeafAngleDistributionTracking` before the end-of-substep `updateShootNodes` (PA:12310); all no-ops unless tracking is on |
| `plantarchitecture/phytomer.py` | `setPetioleAndLeafScaleFraction`: the prescribed-pose branch (nearest point on the previous centerline, PA:4490-4512) instead of `NotImplementedError` |
| `plantarchitecture/xml_io.py` | writes `<leaf_inclination_Beta_distribution>` (IO:612) and `<leaf_prescribed>` / `<leaf_base>` (IO:1002-1007) instead of raising |
| `plantarchitecture/library/__init__.py` | registration of the five models (appended block) |
| `tests/plantarch/test_plantarch_params.py` | `test_registry_and_errors` used `cowpea` as its unported example; it now picks the first unported model from the registry, or skips that assertion when every model is ported |
| `tests/test_parity.py` | W5-B entries in `SLOW_CASES` |

The hooks of the new models are created in the `hooks` module namespace (`strawberry.named_hook`), so the W4-C
`_patch_source` mutation helper rebinds them.

## Parity cases (`cases_vegetables.py`, `ops_vegetables.py`). All 0 ulp, Context seeds 1..8

Per crop: `load` (draw counts after construct / load / second load + every shoot type's ShootParameters), `age0_full`
(snapshot incl. hidden), `age0_attrs` (materials, data, polymesh face sets), `steps` (2-day steps to day 30: draws,
whole XML, shoot nodes/radii, internode and petiole tubes with UUIDs, every leaf 4x4 + vertices; full snapshot at the
end), `ages` (direct builds on a ladder: bud states, the same observation, full snapshot). Plus
`plantarch.cucumber.bump_maps` (material label + normal map file/kind/strength of every leaf primitive, ages 0 and 30).

| crop | `ages` ladder (days) | stages covered |
|---|---|---|
| butterlettuce | 10, 30, 50, 120 | rosette growing; all 25 leaves (from ~50 d); still afterwards |
| sugarbeet | 10, 30, 60, 120 | rosette growing; 30 nodes reached; still |
| asparagus | 10, 20, 40, 120 | branching fern; `max_age` 20 (growth stops after the step that crosses it) |
| cucumber | 20, 40, 60, 90, 120, 165 | vegetative; flowers + fruit (40); head at the wire, lowering/leaning and string slide (60+: laid stem reaches y = 7.8 m at 165 d), harvest of full-size fruit, leaves shed at 45 d (32 leaves kept from 60 d on); `max_age` 165 |
| cowpea | 20, 50, 60, 80, 100 | vegetative with tracking (63/77 leaves steered by 20 d, 186/194 by 50 d); flowering (80: 4 flowers); pods (90-100) |

Results (every case PASS for all its seeds; wall time of the whole case, both backends, shared machine):

| case | butterlettuce | sugarbeet | asparagus | cucumber | cowpea |
|---|---|---|---|---|---|
| `load` (8) | PASS 0.1 s | PASS 0.1 s | PASS 0.1 s | PASS 0.1 s | PASS 0.1 s |
| `age0_full` (8) | PASS 27 s | PASS 6 s | PASS 0.4 s | PASS 5 s | PASS 6 s |
| `age0_attrs` (8) | PASS 31 s | PASS 7 s | PASS 0.5 s | PASS 5 s | PASS 7 s |
| `steps` (8, to 30 d) | PASS 68 s | PASS 24 s | PASS 42 s | PASS 98 s | PASS 176 s |
| `ages` (8 x ladder) | PASS 32 pairs, 203 s | PASS 32, 72 s | PASS 32, 80 s | PASS 48, 1230 s | PASS 40, 868 s |
| `bump_maps` (16) | | | | PASS 69 s | |

Fast suite additions: the five `load` cases, `asparagus.age0_full` / `age0_attrs` and the unit tests + 4 fast mutation
checks of `test_vegetables.py` (~10 s together); everything else is in `SLOW_CASES` / `@pytest.mark.slow`.

Observation op notes: internode tubes are compared without the shoot label (the Helios side of `pa_tube_objects`
numbers internode objects 0, 1, 2, ... which shifts once a lateral's tube is not built yet: asparagus), and petioles
come from W4-B's `w4b_petiole_objects` (W4-B finding 1: `pa_tube_objects("petiole")` truncates at the cowpea's
two-petiole unifoliate node).

## Mutation checks (`tests/plantarch/test_vegetables.py::test_mutations_are_caught`), all caught

| mutation | caught by |
|---|---|
| lettuce creation function without the `rotateLeaf` pitch | butterlettuce.age0_full |
| sugar beet base-rotation draws left to right (pitch before the discarded yaw) | sugarbeet.age0_full |
| asparagus cladode without its private-stream roll about x | asparagus.age0_full |
| asparagus blind nodes `<= 1` instead of `<= 2` | asparagus.steps |
| cucumber bump strength 12.5886 instead of 12.58861 | cucumber.bump_maps |
| cucumber fruit not harvested | cucumber.ages (60 d) |
| cucumber stem never lowered | cucumber.ages (60 d) |
| cowpea first floral node `max(5, n/3)` instead of `max(5, n/2)` | cowpea.steps |
| tracker target at the bin's lower edge (no within-bin draw) | cowpea.steps |
| tracker turns a leaf all the way at once (no great-circle fraction) | cowpea.steps |
| `setLeafNormal` recorded roll sign flipped (`atan2(-y.z, z.z)`) | cowpea.steps (XML) |

(`pytest --runslow tests/plantarch/test_vegetables.py -k mutations`: 11 passed, 75 s; steps shortened to 16 days.)

## Build times (Context seed 1, one plant, this machine under shared load)

| model, age | Helios | Heliades | primitives |
|---|---|---|---|
| butterlettuce 50 d | 1.27 s | 5.5 s | 35650 |
| sugarbeet 60 d | 0.23 s | 4.4 s | 13940 |
| asparagus 30 d | 0.06 s | 5.3 s | 18448 |
| cucumber 165 d (max_age) | 1.48 s | 46.3 s | 71200 |
| cowpea 100 d | 0.77 s | 21.8 s | 63438 |

The tracker costs about 4 s of the cowpea's 22 s (per-substep facet areas/normals of every blade, vectorised per leaf).
One `ages` run of the cowpea took 47 min while the machine was swapping (other agents' jobs); rerun alone it took 12 s.

## Helios / PyHelios findings

1. **Sugar beet OBJ leaf is never loaded.** `assets/obj/SugarBeetLeaf.obj` (whose `mtllib SugarbeetLeaf.mtl` does not
   resolve on a case-sensitive filesystem, HELIOS_ISSUES) is referenced by no source file in 1.3.98: `initializeSugarbeetShoots`
   (PL:6017) builds `GenericLeafPrototype` blades textured `SugarbeetLeaf.png` (which exists). The sugar beet therefore
   builds without error on Helios at every age; Heliades does the same. Nothing to reproduce.
2. `buildSugarbeetPlant` (PL:6090) passes `0.f * context_ptr->randu(0.f, 2.f * M_PI)` as the yaw: the draw is taken
   (first, right to left) and discarded, so every sugar beet has yaw 0 but the stream advances. Reproduced.
3. Asparagus: `max_age = 20` but `setPlantPhenologicalThresholds(..., 5, 100)`: the plant stops growing after the step
   crossing day 20 and never reaches dormancy; `max_floral_buds_per_petiole = 0`, so its peduncle / inflorescence
   parameters are dead. Builds at 30..120 d are identical apart from the last partial step. Reproduced.
4. Cucumber: the species definition in PlantLibrary.cpp says it was "Ported by Heliades"; `CucumberMainPhytomerCreationFunction`
   calls `setPrimitiveBumpMap` on every new leaf, each leaf primitive of the named `cucumber_main_leaf` material then gets
   its own `__copy_<id>_<UUID>` material (renamed-from-automatic materials are copied per primitive). Reproduced (labels compared).
5. Parity-op quirk (not Helios): `pa_tube_objects("internode")` (W3-C) labels the Helios internodes with consecutive
   shoot indices, which disagree with Heliades once a shoot has no tube yet (asparagus laterals). The W5-B cases omit the label.
6. No discrepancy found in the leaf-angle tracker: it matches at 0 ulp including its private `minstd_rand0`
   (seeded `plantID*2654435761 + llround(1000 mu)*40503 + llround(1000 nu)*2246822519` mod 2^32).

## Not matched / not ported

Nothing on the five models' paths is unmatched. Not ported (not reached): azimuth tracking, epicormic cohort entries,
`leaf_stalk_base_placed` in the XML writer, `appendAttractionPoints` / `setAttractionParameters`, decoding of
normal/bump map images (the map is carried by the material only; Helios's texel cache has no observable effect here).
