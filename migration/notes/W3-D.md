# W3-D: inflorescence geometry

Baseline: Helios 1.3.98. Scope from SPEC §8: the flower / fruit prototypes, the peduncle tube, flower and fruit
placement, fruit scaling and the geometry side of `setFloralBudState`. The floral-bud *state machine*
(`advanceTime`) is W3-C's; nothing here decides when a bud changes state.

## What is ported

| file | contents | C++ |
|---|---|---|
| `plantarchitecture/inflorescence.py` | `set_floral_bud_state` (whole `Phytomer::setFloralBudState(BudState, FloralBud&)`), `update_inflorescence`, `create_inflorescence_geometry`, `ensure_inflorescence_prototypes_initialized`, `sample_organ_build_parameters`, `make_closed_fruit_surfaces_one_sided`, `record_fruit_class_map` (empty file only), `get_reconciled_peduncle_radius`, `peduncle_should_match_culm`, `get_peduncle_node_colors`, `bend_peduncle_axis`, `update_peduncle_radii`, `set_inflorescence_scale_fraction`, `get_floral_bud_storage_index`; `fruit_growth_fraction` (the advanceTime fruit-growth expression, for W3-C) | PA:54-179, 790-796, 1295-1305, 1491-1547, 3623-4197, 6744-6759, 12155 |
| `plantarchitecture/library/tomato_inflorescence.py` | `TomatoFlowerPrototype`, `TomatoFruitPrototype` bound to `hooks.*` (`loadOBJ(asset, (0,0,0), 0.75, nullrotation, black, "ZUP", silent)` + `addPolymeshObject`); imported lazily on first use | AS:4080-4090 |

The prototype caches are `pa.unique_closed_flower_prototype_objIDs`, `pa.unique_open_flower_prototype_objIDs`,
`pa.unique_fruit_prototype_objIDs` (C++ names), keyed by `(hook, organ build parameters)`. They are created on
first use (`inflorescence._caches`); W3-C/W3-E may move them into `shoot_min.init_plant_state`.

Not ported (raise `NotImplementedError`): carbon model costs in `setFloralBudState`, fruit-branch collision
(`fruit_collision_detection_enabled`), fruit class maps (`fruit_class_map_file`), organ build parameters
(`BuildParameterDistribution.sample`; tomato declares none). The fruit gravity rotation (step 6 of
`createInflorescenceGeometry`) is ported but not oracle-tested (tomato's `fruit_gravity_factor_fraction` is 0).

## Float / double / libm choices

- `std::atan2`, `std::acos`, `std::abs` on floats are the float overloads (glibc `atan2f`, `acosf`): `phytomer.atan2f`,
  `vec.acosf`. `asin_safe` is `asinf`. `rotatePointAboutLine` uses double sin/cos (`vec.rotate_point_about_line`).
- PlantArchitecture.cpp has `using namespace helios` only, so unqualified `fabs` is C's double one; on
  `fruit - float(n-1)/float(p)` (a float) that is exact and changes nothing.
- `floor(0.5*float(n-1)) + 1` (equal spacing, offset 0) is double; `-float(PI_F) + dphi*(fruit + 0.5f)` float.
- `std::abs(angle) > 1e-6` compares the float magnitude in double; `fmin(1, 0.25f + 0.75f*tc/dd)` is a float
  expression through double `fmin`.
- Literals keep their C++ types (`0.9f`, `1e-6f`, `1e-9f`, `0.25`).

## RNG (draw-count checks)

Per rebuild (`setFloralBudState` to CLOSED / OPEN / FRUITING) tomato makes exactly **6 engine calls**: one
`internode.phyllotactic_angle.resample()` per flower (PA:4130). `randu(0, unique_prototypes-1)` is `randu(0, 0)`:
no draw. All other resamples (peduncle length/radius/pitch/yaw/curvature/roll, inflorescence roll/pitch/scales/
gravity, flowers_per_peduncle) are of constants. Flower 0 uses the cached phyllotactic value: the phytomer's own
angle for the first rebuild, the previous rebuild's last resample afterwards (checked by `plantarch.inflo.draws`).
`ensureInflorescencePrototypesInitialized` draws nothing (no organ build parameters).

## Parity cases (`cases_plantarch_inflo.py`, `ops_plantarch_inflo.py`)

Growth is not merged, so no Heliades plant flowers. Two isolating strategies:

1. **Prototypes** straight from the oracle: tomato grown to 100/110/120 days, hidden objects > 37 (the three
   inflorescence prototypes, objects 700-702 for seed 1 / 100 days). Heliades: age-0 build, then
   `ensure_inflorescence_prototypes_initialized` on phytomer 0's parameters, exactly as the first peduncle does.
2. **Forced states with oracle inputs** via an oracle trace driver (C++ source embedded in
   `ops_plantarch_inflo._DRIVER_SRC`, compiled once with g++ against the pinned `libhelios.so` into
   `$HELIADES_ORACLE_CACHE` or `$TMPDIR/heliades_oracle_cache`; Helios itself is untouched). It is a
   `PlantArchitectureTestHelper` (the friend Helios declares for its self-tests); RandomParameter caches are
   private, so the header is included with `private`/`protected` defined to `public` after every std header.
   For each oracle plant (seeds 1..8 x ages 100/110/120) and each bud with flowers or fruit, it dumps
   everything `setFloralBudState` / `updateInflorescence` read (bud base and rotation, internode nodes and radii,
   petiole nodes, `inflorescence_bending_axis`, shoot index, every RandomParameter with its cache state, colours,
   storage sizes), then forces DEAD -> CLOSED -> OPEN -> FRUITING (Context reseeded before each), fruit growth
   (tc = 2..32) and `updatePeduncleRadii`, dumping the result after each step. The XML `<floral_bud>` of the
   same plant agrees with the driver's (bud states, `flower_yaw`): `test_driver_plant_matches_pyhelios_xml`.
   Heliades fills a stand-in phytomer (`StandInPhytomer`) from the dump and replays the same sequence.

| case | tier | seeds | checks | max ulp |
|---|---|---|---|---|
| `plantarch.inflo.prototypes` | ulp 0 | 3 (Context 1/4/7, ages 100/110/120) | 3 hidden objects: primitive count, transform; per primitive vertices, UVs, colour, area | 0 |
| `plantarch.inflo.prototypes_attrs` | exact | 3 | per primitive material (label, two-sided flag, colour, texture, override) and `object_label` and data labels; object data; hidden flags | — |
| `plantarch.inflo.placement` | ulp 0 | 24 (Context 1..8 x 100/110/120 d; @@NBUDS@@ buds) | after each forced step, growth and radii update: state, counts, fruit scale, peduncle nodes / radii / triangles / colours, per flower or fruit: transform, `inflorescence_rotation` (pitch, yaw, roll, azimuth, peduncle axis, gravity), base, base scale, triangles / colours; stored `peduncle_vertices` / radii / sampled parameters | 0 |
| `plantarch.inflo.placement_labels` | exact | 24 | material label and `object_label` of every peduncle / flower / fruit primitive | — |
| `plantarch.inflo.draws` | exact | 24 | engine calls per rebuild (6), phyllotactic cache (value, sampled) at the start of each rebuild | — |
| `plantarch.tomato.flowering` | ulp 0 | 24 | **gated**: registered only when `PlantArchitecture.advanceTime` exists and a 2-day build works (`growth_available()`, override with `HELIADES_PARITY_FLOWERING=0/1`). Engine calls after the build, then every peduncle / flower / fruit object (sorted IDs): primitive count, transform, triangles, colours | — (not run yet) |

Unit tests (`tests/heliades/test_plantarch_inflo.py`): bend branches, fruit growth formula, hook binding,
storage index, DORMANT/ACTIVE shortcut, a **real age-0 Phytomer** forced through CLOSED/OPEN/FRUITING (6 draws per
rebuild, 3 hidden prototypes built once, old copies deleted, 0.25 initial fruit scale, DEAD cleans up), scale about
each base, peduncle re-anchoring and culm matching.

## Mutation check (`test_mutations_are_caught`, re-executing the module source with one edit)

@@MUTATIONS@@

## Discrepancies / findings

1. **SPEC §5.5 vs 1.3.98 code** (no draw-count change):
   - `ind_from_tip = fabs(fruit - float(n-1)/float(petioles_per_internode))` and the compound rotation is
     `deg2rad(phyl)*ind + 2*PI_F/float(petioles_per_internode)*fruit` (SPEC's `|f-5|`, `2π f` for tomato's single petiole).
   - The base fraction is `frac = 1 - ((ind-0.5f)*offset*length)/length` in float, not `1-(ind-0.5)*0.15`.
   - `pitch = -asin_safe(axis.z) + deg2rad(80)`, `azimuth = +atan2f(y, x)` (prototype along +x), yaw =
     `deg2rad(peduncle_roll) + compound`, all from `getAxisVector(frac, peduncle)` at the flower's own fraction.
   - Step 6 (fruit gravity) exists; 0 for tomato.
2. **`makeClosedFruitSurfacesOneSided` is not "materials only" in effect**: it welds the fruit's vertices,
   finds the closed outward-facing body and moves its 1024 triangles to a new one-sided material
   `Material_closed` (two-sided flag 0); the calyx (`Material.003_2`, 658 triangles) stays two-sided.
3. **Peduncle colours** blend from the internode colour only for a culm-matched peduncle (last node of a
   max-node shoot); otherwise all nodes have the peduncle colour (= internode colour for tomato).
4. **Grown vs rebuilt peduncles differ** in the oracle (e.g. seed 1 / 100 d: node 1 `3d239194` grown vs `3d239198`
   rebuilt): the grown tube was built when the internode tip sat elsewhere and was re-anchored by
   `updatePeduncleRadii` (`setTubeNodes` of the shifted nodes) as the stem grew. W3-C must call
   `update_peduncle_radii` from the girth step every sub-step, exactly as PA:7683 does, for end-to-end parity.
5. `getInternodeRadius(1.f)` interpolates a 1-element radius list for phytomers > 0 (`1.f/0 = inf`, loop
   skipped, returns the only radius); numpy warns, so the call is wrapped in `np.errstate`.

## Wiring W3-C must add (phytomer.py / shoot_min.py / growth; W3-D did not edit them)

```python
from .inflorescence import (set_floral_bud_state, update_inflorescence, set_inflorescence_scale_fraction,
                            update_peduncle_radii, fruit_growth_fraction)
```

1. `Phytomer._set_floral_bud_state(state, fbud)` (phytomer.py, PA:1491): replace the body with
   `set_floral_bud_state(self.plantarchitecture_ptr, self, fbud, state)`. It already covers the
   DORMANT/ACTIVE shortcut, the deletes and the DEAD case; `setFloralBudState(state)` / `(state, p, b)` keep
   dispatching to it. Optionally add `Phytomer.updateInflorescence(fbud)`, `setInflorescenceScaleFraction(fbud, f)`,
   `updatePeduncleRadii()` as one-line delegates.
2. `advanceTime` floral state machine (PA:11972-12052), per phytomer -> petiole -> bud, after
   `fbud.time_counter += dt` (and `fbud.age += dt` if not ACTIVE):
   - initiation: `randu() < flower_bud_break_probability.val()` -> `phytomer.setFloralBudState(BUD_FLOWER_CLOSED, fbud)`
     else `BUD_DEAD` (PA:11992/11994);
   - opening: `BUD_FLOWER_OPEN` (PA:12007/12010), DEAD branch PA:12012;
   - fruit set: `randu() < fruit_set_probability` -> `BUD_FRUITING` else `BUD_DEAD` (PA:12042/12044).
   The state is set *inside* the same bud iteration (each non-DEAD change draws 6 and creates objects there).
3. Fruit growth (PA:12142-12158), per phytomer after the petiole/leaf scaling of the same phytomer:
   ```python
   dd = fb.dd_to_inflorescence_maturity if fb.dd_to_inflorescence_maturity > 0 else inst.dd_to_fruit_maturity
   if fb.state == BUD_FRUITING and fb.time_counter > 0 and dd > 0 and fb.dd_to_inflorescence_maturity != 0:
       fb.previous_fruit_scale_factor = fb.current_fruit_scale_factor
       set_inflorescence_scale_fraction(phytomer, fb, fruit_growth_fraction(fb.time_counter, dd))
   ```
4. End of `incrementPhytomerInternodeGirth` (PA:7683), every sub-step, not gated on `update_context_geometry`:
   `update_peduncle_radii(phytomer)`.
5. `Phytomer::setPetioleBase` (PA:4199-4250, W3-C): besides petioles and leaves, add `shift` to every
   `phytomer.peduncle_vertices[e][b][k]`, and per floral bud to `fb.base_position`, every
   `fb.inflorescence_bases[i]`, and `translate_object(fb.inflorescence_objIDs, shift)`,
   `translate_object(fb.peduncle_objIDs, shift)` (if `build_context_geometry_peduncle`), in that order.
6. `appendPhytomerToShoot` terminal buds (PA:7556-7590): `update_inflorescence(pa, phytomer, fb)` and, for
   FRUITING, `set_inflorescence_scale_fraction(phytomer, fb, 0.25)`. Not reached by tomato
   (`max_terminal_floral_buds` = 0).
7. A stand-in phytomer must provide `required_phytomer_members()`; the real `Phytomer` already does
   (`peduncle_*` storage, `inflorescence_bending_axis`, `build_context_geometry_peduncle`, ...).

For W3-E: the end-to-end case `plantarch.tomato.flowering` activates by itself once
`buildPlantInstanceFromLibrary(age > 0)` works; its Heliades op (`inflo_plant_organs`) walks
`shoot_tree -> phytomers -> floral_buds` and classifies inflorescence objects by bud state, while the oracle uses
`getPlantPeduncleObjectIDs` / `getPlantFlowerObjectIDs` / `getPlantFruitObjectIDs`.

## Requests for core (not done)

- None blocking. `load_obj`, `copy_object`, the object transforms, `hide_object`, tubes (`set_tube_radii`,
  `set_tube_nodes`, node getters), materials (two-sided flag, add/assign) and data sufficed.
- `atan2f` still lives in `phytomer.py` (W3-B's request to move it, with `powf`, into `core/vec.py` stands).
