# W2-C: PlantArchitecture data layer

Baseline: Helios 1.3.98. Scope from SPEC §8: parameter structs, RNG semantics, tomato library entry and the
draw-count probe. Geometry and growth are W3.

## What is ported (`heliades/plantarchitecture/`)

| file | contents | C++ |
|---|---|---|
| `random_parameter.py` | `RandomParameterFloat` / `RandomParameterInt` (alias `RandomParameter_float/_int`): constant/uniform/normal/weibull and constant/uniform/discretevalues, lazy `val()`, `resample()`, `set()` (= `operator=(scalar)`), `copy()` / `_assign_from()` (implicit copy operations: the cache and the generator pointer travel with them), both `initialize` overloads | H:57-306 |
| `params.py` | `AxisRotation`, `make_AxisRotation`, `BudState`, `VegetativeBud`, `FloralBud`, `LeafShapeValues`, `LeafPrototype` (ctor, defaulted ctor, `duplicate`, `operator=`, `sampleIdentifier`, `setRandomGenerator`, `resolveFlexibility`, texture-series helpers), the five `PhytomerParameters` sub-structs with their custom `operator=`, `PhytomerParameters` (ctor + `resample` order), `ShootParameters` (ctor, custom `operator=`, `defineChildShootTypes`, `inheritCustomFunctionsFrom`), `deg2rad`/`rad2deg`/`sum` | H:430-1646, PA:798-1150 |
| `hooks.py` | identity-compared, named stand-ins for the C++ function pointers (`GenericLeafPrototype`, `TomatoFlowerPrototype`, `TomatoFruitPrototype`, `TomatoPhytomerCreationFunction`); W3 binds them with `.implement(fn)` | |
| `architecture.py` | `PlantArchitecture(scene)`: `generator` **is** the Scene's `MinstdRand0` object (PA:383); registry (`registerPlantModel`, `getAvailablePlantModels`, leaf-inclination Beta map), `loadPlantModelFromLibrary`, `initializeDefaultShoots`, `defineShootType`, `getCurrentShootParameters` (both overloads), `updateCurrentShootParameters` (both overloads) | PA:382-403, 1225; PL:444-520, 631-660, 755-880 |
| `library/__init__.py` | all 32 Helios models registered in C++ order with plant type and Beta; only tomato has a shoot initializer | PL:444-505 |
| `library/tomato.py` | `initializeTomatoShoots` line for line, literals with their C++ types; `base_rotation_draws` (the right-to-left `make_AxisRotation(0, randu, randu)` of PL:6214) for W3-E | PL:6101-6214 |
| `structure.py` | `shoot_parameters_to_dict` mirroring the PyHelios JSON of `getCurrentShootParameters`, plus `canonical()` (float32 JSON). Parameter part of `plant_structure_dict` | |

Value semantics: Python objects are references, so `_Struct.__setattr__` turns `s.field = v` into the
C++ assignment when `field` holds a value type. `pp.internode.pitch = 10` is `operator=(float)`;
`pp.leaf.prototype = proto` is `LeafPrototype::operator=`, which draws an identifier. Use `object.__setattr__`
or `_raw_set` to rebind on purpose.

Tooling: `heliades/parity/tools/draw_count.py`, the §3.5 probe for both backends. It reads 3 `randu()` values
and finds them in the seeded stream (vectorised LCG jumps, ~1 s per 10^7 calls). One value is ambiguous
past a few million calls. CLI: `python -m heliades.parity.tools.draw_count --seeds 1,5 load build:0`.
`calls_between(seed, generator)` is a Heliades-only shortcut.

## Parity cases (`cases_plantarch_load.py`, `ops_plantarch_load.py`). All exact and all pass.

| case | seeds | checks |
|---|---|---|
| `plantarch.load.tomato` | Context 1..8 | engine calls after ctor (0), after load (8/10/12/16), after a second load |
| `plantarch.load.tomato_params` | 2 | entire `mainstem` ShootParameters tree vs PyHelios JSON (float32 bytes) |
| `plantarch.load.models` | 1 | registry list |
| `plantarch.load.identifier` | 1..8 | 4th identifier via what it seeds: per leaflet of phytomer 0, `<leaf_prototype>` index (XML) and `getLeafShape().longitudinal_curvature` (full float) = draw `p*7+l` of `minstd_rand0(identifier)` |
| `plantarch.load.sampled_age0` | 1..8 | base rotation (right to left), phyllotactic angle, petiole pitch and curvature from the XML (6 digits; Heliades formats with `%g` like `ostream<<float`) |
| `plantarch.load.redefined_distributions` | 1..8 | `getCurrentShootParameters`/`defineShootType` round trip with normal, weibull, float uniform, int uniform with a==b, and discretevalues parameters, then the age-0 resample sequence and the 7 prototype choices |

The age-0 cases use `ops_plantarch_load.age0_draw_plan`, an explicit replay of the Phytomer constructor's
parameter draws, built on the ported structs. It is a W2-C stand-in, not the geometry port, and W3-B
should replace it with the real constructor and keep the cases.

Mutation checks (`tests/heliades/test_plantarch_params.py::test_mutations_are_caught`), all caught:
- drop the PA:983 `operator=` identifier draw: caught by `load.tomato`;
- make `LeafParameters::operator=` use `operator=` instead of `duplicate()`: caught by `load.tomato`;
- drop the wasted `longitudinal_curvature` resample (f): caught by `load.identifier`, while `load.tomato` still passes;
- draw the base rotation left to right: caught by `load.sampled_age0`.

## Discrepancies / findings

1. **SPEC §3.3 gap (no effect on the default tomato).** The Phytomer constructor also resamples
   `petiole.taper` (PA:3406) and `internode.max_vegetative_buds_per_petiole` (PA:3439, and PA:3584 for
   terminal buds). `createPetioleFloralBuds` resamples `max_floral_buds_per_petiole` (PA:2678). All of these
   are constant for tomato, so they make no draws and the 26/27 count holds. A redefined shoot type that
   distributes them does draw; this was found by `load.redefined_distributions`. W3-B must keep these calls
   in this order.
2. **`AxisRotation::operator-` is reversed** (H:479: `x - a` returns `{a.pitch - pitch, ...}`). Ported as
   written. Candidate for `HELIOS_ISSUES.md` (outside my edit scope).
3. **Members left without a generator** by `PhytomerParameters(gen)`:
   - `leaf.leaflet_scale` and `leaf.intercalary_leaflet_scale` are set with `operator=(float)`;
   - `internode.max_vegetative_buds_per_petiole` and `max_floral_buds_per_petiole` are never initialised.

   Giving any of them a distribution through the C++ API throws on the first `val()`. Heliades does the same.
   PyHelios's `defineShootType` works because its wrapper rebuilds every parameter with `initialize(gen)`,
   and the Heliades op mirrors that.
4. `PhytomerParameters()` assigns `petiole.color` twice (PA:996). `peduncle.color` is never set and stays
   `RGBcolor()` = black. Tomato sets it.
5. The PyHelios `defineShootType` / `getCurrentShootParameters` wrappers make **no** engine draws in 1.3.98
   (measured with the probe). The redefined case still reseeds, to stay independent of the wrapper.
6. The `ShootParameters::operator=` comment (H:1585) says it resamples. It does not: confirmed by
   `load.tomato` together with the unit tests.
7. Not ported:
   - `skipStructuralOperationInCallback` and `rejectStructuralOperationInCallback`, because phytomer
     callbacks do not exist yet;
   - `registerPlantModelBuildParameters` and build-parameter distributions, because tomato declares none.
8. Heliades `Scene()` starts at seed 1, whereas Helios seeds from the wall clock. Always seed explicitly.
9. `migration/inventory.py` needs `third_party/` inside the worktree, so it was not run here.

## Requests for core (not done; `heliades/core` is owned by others)

- A `Scene.get_random_generator()` accessor, the counterpart of `Context::getRandomGenerator()`. Until it
  exists, `plantarchitecture.context_generator()` falls back to `scene.rng.generator`. It relies on
  `ContextRandom.seed()` reseeding that object in place, never replacing it.

## What W3 can build on

- `PhytomerParameters.copy()` followed by `.resample()` is the constructor's first step.
- `pp.leaf.prototype.setRandomGenerator(MinstdRand0(identifier))` plus `longitudinal_curvature.val()` and
  `.resample()` per blade gives the private stream (W3-A).
- `library.tomato.base_rotation_draws`, `hooks.*.implement`, `BudState`/buds/`AxisRotation`.
- `shoot_parameters_to_dict` for the snapshot state.
- The draw-count probe after every `advanceTime` sub-step (W3-C).
