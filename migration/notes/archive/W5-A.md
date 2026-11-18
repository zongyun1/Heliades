# W5-A: cereals (maize tassel / maize to maturity, wheat, sorghum, rice)

Baseline: Helios 1.3.98. PA = PlantArchitecture.cpp, PL = PlantLibrary.cpp, AS = Assets.cpp.

## What is ported

| file | contents | C++ |
|---|---|---|
| `library/grass_inflorescence.py` (new) | organ build-parameter key tables (`OrganKeyTables`, `blendOrganParameters`, `applyOrganBuildParameters`, `appendOrganBuildParameterDeclarations`, the shared `fromBuildParameters` body), raw `getBuildParameterIntegerValue`, `perpendicularTo`, `quantizedBrightness`, `GrassInflorescenceMesh`, `addGrassAnther`, `Context::addTube` (colored, object-less), `add_triangle` with the RGBA clamp of `Context::addTriangle(RGBcolor)` | AS:2458-2540, 2616-2625, 3196-3300; Context.cpp:2114-2206; Context_primitive.cpp:181 |
| `library/maize.py` | `MaizeTasselParameters` (presets, key tables, declarations), `addMaizeSpikelet`, `buildMaizeTassel`, `MaizeTasselPrototype` (the `NotImplementedError` is gone), maize organ build parameters, `setInternodeMaxLength` (free function) | AS:3086-3553, PL:528, PA:4360 |
| `library/wheat.py` (new) | `initializeWheatShoots`, `buildWheatPlant`, `WheatPhytomerCreationFunction` (+ blade / sheath profiles, tillers), `WheatSpikePrototype` (OBJ), `tiller_count` | PL:6542-6687, AS:4166-4260 |
| `library/sorghum.py` (new) | `initializeSorghumShoots`, `buildSorghumPlant`, `SorghumPhytomerCreationFunction`, `SorghumPanicleParameters`, `addSorghumGrain`, `addSorghumGlumes`, `buildSorghumPanicle` (packing vectorised, see below), `SorghumPaniclePrototype`, `tiller_count` + panicle organ parameters | PL:5590-5743, AS:2419-3084, 3949-4029 |
| `library/rice.py` (new) | see the rice section (rice cases / tests live in `cases_cereals_rice.py`, `ops_cereals_rice.py`, `tests/plantarch/test_cereals_rice.py`, written in parallel and kept as separate files) | |
| `library/cereals_registry.py` (new) | the `registerPlantModelBuildParameters` calls of the four grasses | PL:525-535 |

### Shared-file hunks

1. `library/__init__.py`: appended registrations of wheat, sorghum, rice (shoot initializer + builder).
2. `library/build_parameters.py`, `register_build_parameters`: 2 lines calling `cereals_registry.register_build_parameters`.
3. `tests/test_parity.py`: the slow W5-A cases added to `SLOW_CASES`. Fast suite: 395 passed, 166 skipped in 5:25 on a loaded machine (W5-A adds only load / age0 / build_parameters cases, the cheap wheat mutations and unit tests, ~30 s).
4. Rice: see the rice section (`core/scene.py`, `shoot.py`).

No hunk changes a path an existing model takes: the new branches are reached only by the new models (the existing
maize cases, including `flag_leaf` with its draws-only inflorescence, still pass).

## Parity (`cases_cereals.py`, `ops_cereals.py`): Context seeds 1..8 unless noted, all 0 ulp / exact

Observation of a build: draw-count probe, flower/fruit/leaf/primitive counts, whole plant XML (left out when build
parameters are given: the Heliades XML writer does not write them), every shoot's nodes and radii, every internode and
petiole tube, every leaf (4x4 + vertices), then the full hidden-inclusive snapshot as a stream.

The Helios gallery age (`utilities/render_plant_library.cpp`: 5-day steps up to `max_age`, the last full-leaf step with
fruit) is 365 days for maize, wheat, sorghum and rice: none of them sheds leaves and their geometry stops changing
after ~90-125 days, so that is the age verified as `gallery365`.

| case | ages (d) | seeds | result |
|---|---|---|---|
| `maize.tassel` / `anthesis` / `mature` / `gallery365` | 58 (tassel just formed, day 57) / 64 (full size, 6-day maturity period) / 120 (ear mature) / 365 | 8 | pass |
| `maize.tassel_steps` | 54..66 in 2-day `advanceTime` calls (bud, prototypes, expansion) | 4 | pass |
| `maize.tassel_build_parameters` | 60, one non-default organ map per seed (tassel_type 0 / 1 / 0.3 / 0.8 + keys; no branches; no anthers; spikelet_subdivisions 3 and 5; ...) | 8 | pass |
| `maize.build_parameters` | names + error texts of 7 rejected maps + 1 accepted | 1 | pass |
| W4-C maize cases (load, age0, steps, gallery 50, flag_leaf 57, ...) | unchanged | 8 | pass |
| `wheat.load` / `age0` / `steps` (0..60, 2-day) | | 8 | pass |
| `wheat.vegetative` / `heading` / `ear` / `ripe` / `gallery365` | 30 / 46 / 58 / 90 / 365 | 8 | pass |
| `wheat.tillers_vegetative` / `tillers_heading` / `tillers_ripe` | 36 / 56 / 100 with tiller_count 1,2,3,4,4,3,2,1 | 8 | pass |
| `wheat.build_parameters` | names + error texts | 1 | pass |
| `sorghum.load` / `age0` / `steps` (0..50, 2-day) | | 8 | pass |
| `sorghum.vegetative` / `flag_leaf` / `panicle` / `grain_fill` / `mature` / `gallery365` | 30 / 50 / 54 (panicle set day ~53) / 70 / 100 / 365 | 8 | pass |
| `sorghum.tillers_vegetative` / `tillers_panicle` | 40 / 66 with tiller_count 1,2,3,4,5,6,3,6 (up to 7 panicles, 538k primitives) | 8 | pass |
| `sorghum.panicle_steps` | 50..60 in 2-day calls | 4 | pass |
| `sorghum.panicle_build_parameters` | 56, one non-default organ map per seed (panicle_type 0 / 1 / 0.3 / 0.7, rachis curvature 60, grain_count 300/600, no glumes, grain_subdivisions 3 / 5, ...) | 8 | pass |
| `sorghum.build_parameters` | names + error texts of 9 rejected maps + 1 accepted | 1 | pass |
| `rice.*` | see the rice section | | pass |

Build times (seed 1, direct build, Heliades vs oracle): maize 60 d 17 s vs 0.9 s, maize 365 d 70 s vs 1 s; wheat 60 d
2 s vs 0.1 s; sorghum 53 d 96 s, 365 d 173 s vs ~5 s, sorghum 66 d with 6 tillers 424 s (538k primitives) vs ~7 s.
The Heliades time is dominated by the flexible-leaf deflection (`deformLeafLattice`, re-run every step; core, not
optimised here); one sorghum panicle prototype (2000 grains, 45k triangles, packing included) takes 2.9 s, one maize
tassel prototype ~1-2 s.

## Mutation checks (`tests/plantarch/test_cereals.py::test_mutations_are_caught`)

| mutation | caught by |
|---|---|
| tassel anther scatter `make_vec3(0, randu, randu)` drawn left to right | maize.tassel |
| no RGBA clamp in `Context::addTriangle(RGBcolor)` | maize.tassel |
| tassel secondary-branch draws left to right (insertion before droop) | maize.tassel |
| wheat internode without the sheath below | wheat.vegetative |
| wheat flag leaf at 30 instead of 33 degrees | wheat.heading |
| wheat tiller leaf number off by one | wheat.tillers_heading |
| sorghum tiller order 1..6 instead of T3, T4, T2, T1, T5, T6 | sorghum.tillers_vegetative |
| sorghum `internode_length_max.resample()` dropped | sorghum.vegetative |
| sorghum packing: a grain's own additions before the subtractions it receives | sorghum.panicle (so the C++ accumulation order is observable) |
| sorghum glume spread term dropped | sorghum.panicle |

Plus `test_resolve_overlaps_matches_loop`: the vectorised overlap pass is bit-identical to the C++ triple loop.

## Float / evaluation-order findings

- GCC right-to-left argument evaluation decides: the anther scatter and tilt (`make_vec3(0, randu(), randu())`: z
  before y), `addGrassAnther(..., filament * randu(0.7, 1.3), anther * randu(0.85, 1.15), ...)` (anther length first),
  the tassel secondary branch `add_branch(..., deg2rad(randu(20, 35)), deg2rad(randu(0, 15)), ...)` (droop first), and
  the wheat / maize base rotations (roll, yaw, pitch). Sorghum's lean, lean azimuth and phyllotaxis azimuth are
  separate statements (in order).
- `std::pow(float, 1.5f)` is glibc `powf`; `std::round(float)` is `roundf` (half away from zero); `std::sqrt(float)`,
  `std::floor(float)` are the float versions; `std::sin(float)` in the glumes is `sinf`; `cosf` / `sinf` are glibc's.
- `Context::addTriangle(v0, v1, v2, RGBcolor)` builds `make_RGBAcolor(color, 1)`, whose constructor clamps every
  channel to [0, 1]. `RGBcolor::scale` does not clamp, so a brightened anther or grain color (> 1) is clamped when the
  triangle is made. The Heliades core `add_triangle` does not clamp; `grass_inflorescence.add_triangle` does (found on
  the first tassel comparison: 93 color values of 1.0032 vs 1.0).
- Sorghum packing (`AS:2966-3000`) is vectorised over grain pairs (sorted cell keys, `searchsorted`); each grain's
  displacement is accumulated term by term in the C++ order: first the subtractions from pairs (i, k), i < k, i
  ascending, then its own additions in cell-visit order (dx, dy, dz nested), j ascending within a cell.

## Helios / PyHelios discrepancies

1. **PyHelios rejects an undeclared build-parameter key itself** ("Unknown build parameter(s) 'tiller_count'. Plant
   model 'maize' accepts only: ...") before calling C++, with a text different from the C++
   `validateBuildParameters` message ("Build parameter 'x' is not accepted by plant model 'y'. Accepted parameters
   are: ..."). Heliades raises the C++ text; the case compares both as `UNKNOWN_KEY`.
2. **PyHelios prefixes C++ errors** with "ERROR (PlantArchitecture::buildPlantInstanceFromLibrary): "; the case
   compares the innermost `ERROR (...)` message, which matches the C++ text exactly.
3. Test infrastructure (not Helios): the oracle side of `pa_tube_objects("petiole")` (W3-C) infers a petiole's shoot
   from per-shoot node counts, which is wrong once a tiller exists; the cereal cases compare petiole objects without
   the inferred shoot index.

## Requests for core (not done)

- `Scene.add_triangle(..., RGBcolor)` should clamp the color as `Context::addTriangle` does.
- `Context::addTube` (the colored, object-less overload) belongs in the core (`grass_inflorescence.context_add_tube`).
- `Phytomer::setInternodeMaxLength` as a method (`maize.setInternodeMaxLength` for now).
- Performance of the flexible-leaf deflection: a tillered sorghum at 66 d takes ~7 min.

## Rice

### Code paths beyond maize/wheat

| path | C++ | state before | now |
|---|---|---|---|
| `initializeRiceShoots` / `buildRicePlant` (2 shoot types; upright base, 2 `randu` right to left) | PL:5366-5508 | absent | `library/rice.py` |
| `RicePhytomerCreationFunction` (+ blade / sheath / culm tables, `rice_tiller_positions`, crown, drawn upper/crown leaf angles via `setGrassFlagLeafAngle`) | AS:2302-2405 | absent | ported |
| `tiller_count` (0-18, default 12) and secondary tillers (rank 1) | PL:531, AS:2335-2380 | absent | ported |
| `RicePanicleParameters` (erect / lax presets, 28 float + 2 int + 9 color keys, `panicle_type`, `spikelet_subdivisions`) | AS:3555-3666 | absent | ported (shared `grass_inflorescence` tables) |
| `addRiceSpikelet`, `buildRicePanicle` (heading / anthesis / ripening), `RicePaniclePrototype`, `RiceFloweringPaniclePrototype` | AS:3668-3947 | absent | ported |
| leaf texture series + stage ages: `Phytomer::updateLeafTextureStage`, `Context::validateColorTexture`, `Context::setPrimitiveColorTexture` / `getPrimitiveColorTextureFile` | PA:4855-4903, Context.cpp:3366-3436 | raised | ported |

### Shared-file hunks (rice)

1. `heliades/core/scene.py`: three new Scene methods after `primitive_texture_file`: `validate_color_texture` (resolution / transparency-channel / alpha-mask checks, cached per pair), `set_primitive_color_texture` (stored as the primitive attribute `color_texture_file`, deep-copied with the primitive like Helios's `color_texture_ID`; geometry, material, solid fraction untouched), `primitive_color_texture_file`. Nothing calls them except the texture-stage path, so no existing model changes.
2. `heliades/plantarchitecture/shoot.py`, `Shoot._update_leaf_texture_stage`: the `raise` for `len(leaf_texture_stage_ages) >= 2` replaced by the port of `Phytomer::updateLeafTextureStage` (stage = max(stage at age, min(plant minimum stage, last)); per existing leaf, the blade primitives whose texture is the variant's first image get the stage image as color texture, "" at stage 0). The no-stage path is unchanged (returns). `leaf_texture_stage` still starts at 0 (C++: UINT_MAX): the first C++ call at stage 0 only clears a color texture fresh blades do not have, so this is equivalent.
3. `heliades/plantarchitecture/library/__init__.py`: 4 lines appended (rice import + registrations).
   `leaf_prototype.py` is unchanged: its texture-series branch looked up `ctx.validate_color_texture` and raised when absent; it now finds it.

### Parity (`cases_cereals_rice.py`, `ops_cereals_rice.py`): 0 ulp

Every build case compares the W5-A observation (draw-count probe, organ counts, XML, shoot nodes/radii, tubes, leaves with vertices, full hidden-inclusive snapshot) plus the color texture of every leaf primitive (`getPrimitiveColorTextureFile`, basename).

| case | tier | seeds | ages / content | result |
|---|---|---|---|---|
| `plantarch.rice.load` | exact | 8 | engine calls (ctor, load, 2nd load) + mainstem and tiller ShootParameters JSON | 0 |
| `plantarch.rice.age0` | ulp 0 | 8 | full hidden snapshot (10 blade prototypes) | 0 ulp |
| `plantarch.rice.steps` | ulp 0 | 4 | 6..60 d in 6-day calls (tillering), observation per step + final snapshot | 0 ulp |
| `plantarch.rice.tillering` / `stage1` | ulp 0 | 8 | 40 d / 66 d (first stage-1 leaf images) | 0 ulp |
| `plantarch.rice.heading` / `anthesis` / `ripening` | ulp 0 | 8 | 79 d (closed flowering panicles) / 83 d (open, anthers) / 90 d (ripening panicles) | 0 ulp |
| `plantarch.rice.mature` / `gallery365` | ulp 0 | 8 | 120 d (stage-2 leaves) / 365 d (Helios gallery age) | 0 ulp |
| `plantarch.rice.tillers_vegetative` / `tillers_ripening` | ulp 0 | 8 | tiller_count 0,1,4,7,9,13,16,18 at 50 d / 90 d | 0 ulp |
| `plantarch.rice.panicle_build_parameters` | ulp 0 | 8 | panicle_type 0 / 1 / 0.3 / 0.7, branch_count, spikelets_per_secondary, spikelet_subdivisions 3/4, pedicel 0, ripeness, anther_fraction 0/1, colors, filament 0, no secondaries, rachis_droop 170 + branch_hang 1 (with tiller_count 0-2); 90 d (seeds 0-3) / 83 d (4-7) | 0 ulp |
| `plantarch.rice.build_parameters` | exact | 1 | accepted names (42); error texts of tiller_count 19 / 2.5, an undeclared key, panicle_type 1.5, branch_count 0, spikelet_subdivisions 13, hull_ripe_color_b -0.1, ripeness NaN; two accepted maps | 0 |

All 14 cases pass at 0 ulp (full run ~80 min wall-clock on a shared machine).

Oracle schedule (seed 1, 2-day steps): tillers from day 24; 93 leaves on 13 culms by day 78; 13 heading panicles (closed flowers) on day 78, open (anthesis, anthers) day 82, ripening panicles from day 88; nothing changes after ~day 90 (162 694 primitives). First stage-1 leaf images on day ~56-66, stage 2 after 72 days of leaf age.

Build times (seed 1, direct build): Helios 79 d 0.3 s, 365 d ~1.3 s; Heliades 40 d 1.8 s, 66 d ~11 s, 77 d 8.5 s, 79 d 31 s, 120 d 38 s, 200 d 50 s, 365 d ~75 s (the day panicles appear costs ~12 s: 6 flower + later 3 fruit prototypes and 13 placed copies, 150k triangles; after heading `deformLeafLattice` takes ~5x more iterations per leaf, ~64 s per simulated day around 78-80, then settles).

### Mutation checks (`tests/plantarch/test_cereals_rice.py`)

| mutation | caught by |
|---|---|
| primary tiller positions shifted by one leaf | rice.tillering |
| crown leaf angle randu(3, 13) instead of (3, 12) | rice.tillering |
| no leaf texture stage (Phytomer::updateLeafTextureStage a no-op) | rice.stage1 |
| anther scatter drawn left to right (y before z) | rice.anthesis (slow) |
| heading spread fraction 0.4 instead of 0.35 | rice.heading (slow) |

Not distinguishable (left out): drawing the base yaw before the roll. With pitch 0 both rotate the upright culm about z, so swapping the two draws gives the same plant.

### Findings

1. Rice's leaves have a texture series and no class map; the texture-stage path (color textures) is invisible in a geometry snapshot (solid fraction and material come from the shape texture), so the cases compare `getPrimitiveColorTextureFile` directly.
2. `Context::addTriangle(..., RGBcolor)` clamps through `make_RGBAcolor`; rice hull colors after `quantizedBrightness` scaling stay in range by default, but the clamp (in `grass_inflorescence.add_triangle`) matters for brightened anthers/sterile lemmas (scale 1.2) — covered by the shared helper.
3. A first heading run took 47 min wall-clock while other heavy runs shared the machine; measured alone the same case is 42 s (Heliades 40 s, Helios 1.6 s).
