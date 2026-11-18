# W5-C: weeds, bougainvillea, grapevine_Wye

Baseline: Helios 1.3.98 (`14860044f`). Abbreviations as in SPEC: PA = PlantArchitecture.cpp, PL = PlantLibrary.cpp,
AS = Assets.cpp.

## What is ported

| file | contents | C++ |
|---|---|---|
| `library/bindweed.py` | `initializeBindweedShoots` (3 shoot types), `buildBindweedPlant`, `BindweedFlowerPrototype` (also the groundcherry flower) | PL:2800-2909, AS:1638 |
| `library/cheeseweed.py` | `initializeCheeseweedShoots`, `buildCheeseweedPlant`, `CheeseweedLeafPrototype` (declared, unused by the library entry, not oracle-tested) | PL:3336-3405, AS:1686 |
| `library/groundcherryweed.py` | `initializeGroundCherryWeedShoots`, `buildGroundCherryWeedPlant` | PL:4459-4573 |
| `library/puncturevine.py` | `initializePuncturevineShoots`, `buildPuncturevinePlant`, `PuncturevineFlowerPrototype` | PL:5107-5227, AS:2269 |
| `library/bougainvillea.py` | `initializeBougainvilleaShoots` (4 shoot types), `buildBougainvilleaPlant`, `BougainvilleaFlowerPrototype` | PL:3072-3205, AS:1644 |
| `library/grapevine_wye.py` | `initializeGrapevineWyeShoots`, `buildGrapevineWye`, `appendWoodDome`, the Wye build-parameter declarations and registration | PL:111-130, 433-435, 523-524, 4033-4457 |

The Wye reuses W4-D unchanged: `BoundedWander`, `_add_component` (right-to-left `addComponent` draws),
`grapevineTrunkPath`, `_linspace3`, `addShootFromNodePositions`, `setPlantAttractionPoints`, `removeShootLeaves`,
the creation / callback functions and `GrapevineFruitPrototype`. The only edit to `grapevine.py` is two lines in
`register_build_parameters` that also register the Wye's declarations (so the shared `build_parameters.py` is not
touched). VSP parity is unchanged (`plantarch.grape.*` not affected: no line of the VSP path changed).

Paths these models reach that earlier models did not:

| path | model | status |
|---|---|---|
| `leaf.prototype.OBJ_model_file` (OBJ leaves through `GenericLeafPrototype`) | bindweed (on the default prototype of `PhytomerParameters`), cheeseweed | already ported, first oracle check |
| `petioles_per_internode = 0` (crown without leaves) | bindweed, puncturevine `base_*` | first oracle check |
| 10 leaflets per petiole, `leaflet_offset` 0.18, two petioles per internode | puncturevine | first oracle check (W3-B finding 2, the double `fabs` in the leaflet offset, is now exercised with `|ind| - 0.5` = 1.5, 2.5, ... and matches) |
| positive `vegetative_bud_break_probability_decay_rate` | puncturevine (1), bougainvillea (0.8) | **bug fixed in the port** (below) |
| `removeShootVegetativeBuds`, `removeShootFloralBuds` | bougainvillea | **new** in `shoot.py` |
| distributed `flowers_per_peduncle`, `flower_offset`, 3-segment curved peduncles | bougainvillea | first oracle check |
| null `fruit_prototype_function` with a fruit-set threshold (`GroundCherryFruitPrototype` is commented out in Helios): the fruit transitions are skipped, and with `time_to_flower_opening = -1` the flowers stay closed | groundcherryweed | first oracle check |
| `max_age` reached (no-growth branch) | all weeds | first oracle check |
| arms forking from a hidden node inside the trunk head, `appendWoodDome`, cordons forking from the arm top, `roundf` gable levels | grapevine_Wye | **new** |

## Shared-file changes

- `plantarchitecture/shoot.py`:
  - new `removeShootVegetativeBuds`, `removeShootFloralBuds` (PA:10014-10044), with a small `_shoot_or_raise`
    helper, inserted before `setPlantAttractionPoints`;
  - `Shoot.sampleVegetativeBudBreak`, the `decay > 0` branch: `current_node_number - node_index - 1` is now wrapped
    to uint32 as in C++. A bud sampled when its phytomer is appended has `node_index == current_node_number`, so C++
    computes `float(2^32 - 1)` and the probability is `probability_min`; the port computed `-1`, giving
    `probability_max + decay` (always break). No earlier model has a positive decay rate (tomato -0.25, bean -0.4,
    strawberry -0.5, the others 0), so their behaviour is unchanged. Found as the puncturevine growing 868 shoots
    instead of 76 by day 30 (first divergence on day 13, Context seed 1).
- `plantarchitecture/library/__init__.py`: two appended registration blocks (weeds + bougainvillea; grapevine_Wye).
- `tests/test_parity.py`: the W5-C heavy cases appended to `SLOW_CASES`.

No edit to `phytomer.py`, `growth.py`, `inflorescence.py`, `leaf_prototype.py`, `params.py`, `api.py`, `core/*`.

## Parity cases: all pass at 0 ulp

`cases_weeds_ornamentals.py` (no new ops: reuses `gv_build`, `gv_shoot_summary`, `pa_xml_text`,
`w4a_inflorescence_objects`) and `cases_grapevine_wye.py` / `ops_grapevine_wye.py` (`gvw_build_parameter_names`).

| case | seeds | checks |
|---|---|---|
| `plantarch.<M>.load` (5 models) | 1..8 | engine calls after ctor / load / second load, every shoot type's ShootParameters |
| `plantarch.<M>.age0` | 1..8 | full snapshot incl. hidden prototypes |
| `plantarch.<M>.steps` | 1..8 | 2-day steps to bindweed 56, cheeseweed 46, groundcherry 60, puncturevine 30, bougainvillea 90 days: probe, whole XML, shoot nodes/radii, inflorescence object count after every step, full snapshot at the end |
| `plantarch.<M>.ages` | 1..8 per age | direct builds; same observation + full snapshot. bindweed 12, 24, 45 (ref), 50 (max), 60; cheeseweed 10, 24, 40 (ref = max), 50; groundcherry 15, 30, 45, 60 (ref), 80 (max), 90; puncturevine 12, 20, 30 (ref), 45 (max); bougainvillea 40, 80, 120, 180 (ref) |
| `plantarch.puncturevine.ages_late` | 1..2 | 55 days (past max_age) |
| `plantarch.grapewye.load` | 1..8 | probes incl. builds with / without build parameters |
| `plantarch.grapewye.params` | 1..2 | the 5 ShootParameters trees |
| `plantarch.grapewye.build_parameters` | 1 | names; rejection of an unknown key, each structure parameter out of range, an organ value, NaN, `catch_wire_height <= trunk_height` (both `<` and `==`); acceptance of `trunk_height` 1.9 with the derived default catch wire 2.5, and of the fixed set |
| `plantarch.grapewye.age0` | 1..8 | full snapshot |
| `plantarch.grapewye.age0_build_parameters` | 1..4 x 2 sets | wide low trellis (0.8 / 1.6 / 3.3 / 2.1 + organ keys) and the smallest accepted trellis (0.5 / 0.2 / 0.5 / 0.55) |
| `plantarch.grapewye.steps` | 1..8 | 2..40 days, full snapshot stream after every step (buds break on day 30, gable wires steer) |
| `plantarch.grapewye.age76` | 1..8 | fruit-zone leaves pulled, no cluster |
| `plantarch.grapewye.age84` | 1..2 | clusters just set (601k primitives, seed 1) |
| `plantarch.grapewye.age100` | 1..8 | clusters growing (866k primitives, seed 1) |
| `plantarch.grapewye.small_age150` | 1..2 | `vine_spacing` 0.5 vine past fruit maturity (273k primitives) |

Stages reached (Helios, Context seeds 1..8; peduncle / flower / fruit objects): bindweed vegetative at 12 d, open
flowers from ~day 20 (4-9 at 24 d, 14-72 at 45-60 d), no growth past 50 d (primitive count constant 45 -> 60 for seed
1); cheeseweed never flowers (`flower_bud_break_probability` 0), 8 nodes by ~day 24; groundcherry closed flowers
from ~day 25 (1-5 at 30 d, 10-169 at 80 d), never opens or fruits (finding 2); puncturevine open flowers from ~day 15
(3-10 at 20 d, 171-328 at 45 d); bougainvillea first inflorescences on ~day 80 (seeds 2, 7, 8), 17-56 flowers at 120 d,
69-253 at 180 d; no fruit (no fruit prototype). The weeds' max_age (bindweed 50, cheeseweed 40, groundcherry 80,
puncturevine 45) is in each ladder with an age past it.

**grapevine_Wye at the reference age (120 days)** was verified once, Context seed 1: 866 154 primitives, 143 shoots,
snapshot and shoot summary at 0 ulp. It takes 3015 s on Heliades (8.8 s on Helios), so it is not a case. Ages 84 and
100 take 95 s and ~230 s: the 100 -> 120 stretch costs ~45 min. A cProfile of the age-100 build puts most of the time in
`object_transforms` (per-primitive `matmult` in `scale_object` / `copy_object` of cluster prototypes,
`_left_multiply`) under `setInflorescenceScaleFraction` / `updateInflorescence`, i.e. the per-sub-step rescaling of
~100 clusters of ~7-8k primitives each. That is core performance (another package), not optimised here. Fruit
maturity (fruit set + 45 days, ~125) is verified on the small vine instead (`small_age150`). The season boundary at
365 days (`time_since_dormancy >= 165 + 200`) raises `NotImplementedError` in `growth.py` (not ported, as for VSP).

## Timings (Context seed 1, build + snapshot; Helios / Heliades)

| model, age | primitives | Helios | Heliades |
|---|---|---|---|
| bindweed 45 (ref) | 9 938 | 0.1 s | 1.3 s |
| cheeseweed 40 (ref) | 942 | <0.1 s | 0.2 s |
| groundcherryweed 60 (ref) / 90 | 24 542 / 48 776 | 0.3 / 0.7 s | 6.8 / 14 s |
| puncturevine 30 (ref) / 45-55 (max) | 96 500 / 672 940 | 1.9 / 12 s | 20 / 150 s |
| bougainvillea 180 (ref) | 143 044 | 2.3 s | 20 s |
| grapevine_Wye 0 / 84 / 100 / 120 (ref) | 5 060 / 600 988 / 866 154 / 866 154 | 0.1 / 4.6 / ~6 / 8.8 s | 1.4 / 95 / ~230 / 3015 s |
| grapevine_Wye, vine_spacing 0.5, 150 | 273 348 | 3.0 s | 67 s |

Whole case runs: bindweed + cheeseweed 3.7 min; groundcherry + bougainvillea ~25 min; puncturevine ~30 min;
Wye `steps` + `age76` ~20 min, `age84` + `age100` + `small_age150` ~35 min. All W5-C cases with a build past age 0 are in
`SLOW_CASES`. Fast-suite cost of W5-C: ~19 s (the `load`, `age0`, `params`, `build_parameters` cases, unit tests and 5
cheap mutation checks). `pytest -q tests`: 401 passed, 139 skipped.

## Mutation checks (all caught)

| mutation | caught by |
|---|---|
| puncturevine `leaflet_offset` 0.17 | `puncturevine.ages` (12 d) |
| bougainvillea stem buds not removed | `bougainvillea.ages` (40 d) |
| bud-break `decay > 0` difference not wrapped to uint (shared `shoot.py`) | `puncturevine.ages` (20 d) |
| Wye arm bowed upward | `grapewye.age0` |
| `appendWoodDome` last angle 80 instead of 85 | `grapewye.age0` |
| `arm_radius` drawn before `arm_bow` | `grapewye.age0` |
| gable spread 0.4 instead of 0.6 (slow) | `grapewye.steps` |
| cordon buds left dead (slow) | `grapewye.steps` |

## Helios findings / discrepancies

1. **uint wrap in `Shoot::sampleVegetativeBudBreak`** (PA:7083): `float(this->current_node_number - node_index - 1)`
   with both `uint`. In `appendPhytomer` (PA:2098) the bud of the phytomer being appended is sampled before
   `current_node_number` is incremented (PA:7541), so the difference is `-1` -> `4294967295` -> probability
   `probability_min`. With a positive decay rate the "maximum at apex" profile therefore never applies to newly
   appended buds, only to buds re-sampled in `breakDormancy`. Likely unintended in Helios; reproduced (puncturevine,
   bougainvillea depend on it).
2. **groundcherryweed never fruits**: `fruit_prototype_function` is commented out (PL:4512), and every fruit-set
   transition in `advanceTime` is gated on a non-null fruit function, so `time_to_fruit_set` 20, `fruit_set_probability`
   0.5, `fruit_prototype_scale` 0.06 and `fruit_gravity_factor_fraction` 0.75 have no effect; with
   `time_to_flower_opening = -1` its flowers stay closed for life. Faithful.
3. `CheeseweedLeafPrototype` (AS:1686) is defined but the library sets `OBJ_model_file` instead; faithful.
4. `initializeGroundCherryWeedShoots` has a stray `;` statement (PL:4468); harmless.
5. The Wye attraction points have no PyHelios getter; they are verified through the growth they steer (`steps`,
   and the `gable_spread` mutation).

## Not done / remaining

- grapevine_Wye at 120 days as a standing case (50 min on Heliades; see above) and the 365-day season boundary.
- `BuildParameterDistribution` sampling for the Wye parameters is shared with VSP and still not oracle-checked
  (PyHelios passes fixed values only).
