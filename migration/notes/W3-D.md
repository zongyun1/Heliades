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

The isolating cases (written before growth merged) use two strategies:

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
| `plantarch.inflo.placement` | ulp 0 | 24 (Context 1..8 x 100/110/120 d; 179 buds: 33 closed, 34 open, 112 fruiting at build time, 11 of them naturally culm-matched; 537 forced rebuilds + 24 culm-matched variants) | after each forced step, growth and radii update: state, counts, fruit scale, peduncle nodes / radii / triangles / colours, per flower or fruit: transform, `inflorescence_rotation` (pitch, yaw, roll, azimuth, peduncle axis, gravity), base, base scale, triangles / colours; stored `peduncle_vertices` / radii / sampled parameters | 0 |
| `plantarch.inflo.placement_labels` | exact | 24 | material label and `object_label` of every peduncle / flower / fruit primitive | — |
| `plantarch.inflo.draws` | exact | 24 | engine calls per rebuild (6), phyllotactic cache (value, sampled) at the start of each rebuild | — |
| `plantarch.tomato.flowering` | ulp 0 | 24 (Context 1..8 x 75/100/120 d) | end to end, real W3-A blades: `buildPlantInstanceFromLibrary(age)`, then W3-C's full observation (engine calls, whole XML incl. `<flower>`/`<peduncle>`, all shoots' nodes/radii, every internode/petiole tube and leaflet with UUIDs/IDs) and every peduncle / flower / fruit object (sorted ID, primitive count, transform, triangles, colours). Gated on `growth_available()` (`advanceTime` exists and a 2-day build works; `HELIADES_PARITY_FLOWERING=0/1` overrides) -- active now | 0 |

Unit tests (`tests/heliades/test_plantarch_inflo.py`): bend branches, fruit growth formula, hook binding,
storage index, DORMANT/ACTIVE shortcut, a **real age-0 Phytomer** forced through CLOSED/OPEN/FRUITING (6 draws per
rebuild, 3 hidden prototypes built once, old copies deleted, 0.25 initial fruit scale, DEAD cleans up), scale about
each base, peduncle re-anchoring and culm matching.

## Mutation check (`test_mutations_are_caught`, re-executing the module source with one edit)

All caught (`plantarch.inflo.*`, case seed 0 = Context seed 1, 100 days); `test_unmutated_source_passes` checks the
re-execution itself is neutral.

| mutation | caught by |
|---|---|
| compound yaw before the translation | placement |
| pitch rotation before roll | placement |
| `ind_from_tip = fruit` (unsigned distance from the tip lost) | placement |
| bend sign flipped in `bendPeduncleAxis` | placement |
| flower azimuth `-atan2f` | placement |
| fruit growth scaled about the origin instead of each base | placement |
| no initial 0.25 fruit scale in `setFloralBudState` | placement |
| drop the per-flower `phyllotactic_angle.resample()` | placement, draws |
| one extra `phyllotactic_angle.resample()` per rebuild | placement, draws |
| skip `makeClosedFruitSurfacesOneSided` | placement_labels |
| peduncle primitives not labelled `"peduncle"` | placement_labels |

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

## Wiring (done after W3-C merged; documented for W3-E)

W3-C's state machine already calls the C++ call sites (PA:11992-12044 initiation / opening / fruit set through
`Phytomer.setFloralBudState` -> `updateInflorescence` + 0.25 fruit scale; fruit growth PA:12155 through
`setInflorescenceScaleFraction`; `updatePeduncleRadii` at the end of `incrementPhytomerInternodeGirth`, PA:7683;
`setPetioleBase` moves the inflorescence and peduncle objects, PA:4199-4250). W3-D replaced the three seams:

- `heliades/plantarchitecture/phytomer.py`: `Phytomer.updateInflorescence(fbud)` ->
  `inflorescence.update_inflorescence(pa, self, fbud)`; `setInflorescenceScaleFraction(fbud, f)` ->
  `inflorescence.set_inflorescence_scale_fraction`; `updatePeduncleRadii()` -> `inflorescence.update_peduncle_radii`.
  The `pa.inflorescence_geometry_draws_only` path is kept unchanged (opt-in; its read/resample order is the same
  as the real one: `test_flowering_geometry_and_draws_only_agree` checks equal engine state at day 56, seed 6).
  `Phytomer._set_floral_bud_state` (W3-C's) is kept; it is the same code as `inflorescence.set_floral_bud_state`,
  which stays the stand-alone entry point for stand-ins.
- `heliades/plantarchitecture/plant.py`: `writePlantStructureXML` now writes the `<flower>` records
  (`flower_pitch/yaw/roll/azimuth` in degrees, `flower_gravity_fraction` if > 0, `flower_base_scale`; contact tilts
  raise). The `<peduncle>` records were already written from the stored values.
- `heliades/parity/cases_plantarch_growth.py`: `plantarch.tomato.age75_vegetative` no longer sets the draws-only flag
  or strips `<flower>`/`<peduncle>`; it compares the whole XML and the tube / leaflet geometry *with* UUIDs and IDs.
- `tests/heliades/test_plantarch_growth.py`: `test_flowering_needs_w3d_or_draws_only` (expected the W3-D raise)
  became `test_flowering_geometry_and_draws_only_agree`.
- Not touched: `growth.py`, `shoot_min.py`, `architecture.py`.

The prototype caches stay lazily created (`inflorescence._caches`); `init_plant_state` may adopt them later.

## Requests for core (not done)

- None blocking. `load_obj`, `copy_object`, the object transforms, `hide_object`, tubes (`set_tube_radii`,
  `set_tube_nodes`, node getters), materials (two-sided flag, add/assign) and data sufficed.
- `atan2f` still lives in `phytomer.py` (W3-B's request to move it, with `powf`, into `core/vec.py` stands).
