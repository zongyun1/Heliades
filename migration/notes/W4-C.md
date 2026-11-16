# W4-C: strawberry and maize

Baseline: Helios 1.3.98. Scope: the `strawberry` and `maize` library plants (shoot parameters, builders,
phytomer creation function, organ prototype functions) and every shared branch they reach that the tomato did not.

## Code paths these crops reach beyond the tomato (written before porting)

Diffed `initializeStrawberryShoots` / `buildStrawberryPlant` / `initializeMaizeShoots` / `buildMaizePlant` /
`MaizePhytomerCreationFunction` (PL:5900-6012, 4574-4692; AS:2007-2110) against the tomato path.

| crop | path | state in Heliades before W4-C |
|---|---|---|
| both | registration (`SHOOT_INITIALIZERS` / `PLANT_BUILDERS`) | absent (`_not_ported`) |
| strawberry | 3 leaflets with yaw -30 / roll -30, no petiolule, 10 unique prototypes, distributed petiole length, distributed `gravitropic_curvature` (Shoot ctor quirk), no creation function, bud break p_min 0 / decay -0.5 | all ported (W3-B/C), never oracle-checked |
| strawberry | `StrawberryFlowerPrototype` / `StrawberryFruitPrototype` (OBJ) | absent; reached only from W3-D's inflorescence |
| maize | base rotation with a pitch (3 `randu`, right to left), radius taper 0.2, `internode_length_max.val()` argument | builder absent; taper path ported, untested |
| maize | sessile leaf (`petiole.length = 0`): leaf pitch as petiole pitch, sessile `orientLeaf` branch | ported (W3-B), untested |
| maize | leaf class map (`leaf_class_map_file`) -> `recordSpectralClassMap` + `helios::classMapAreaFractions` + `readPNGSingleChannel` | **raised `NotImplementedError`** |
| maize | flexible leaf (`flexibility` U(1.5, 2.3), taper 150, ageing 11 / 6): `recordLeafPrototypeRestGeometry` flexible branch | **raised** |
| maize | `Phytomer::deformLeafUnderSelfWeight` (+ `flexibilityAgingProgress/Multiplier`, `deformLeafLattice`, `Context::setPolymeshObjectVertices`) | **raised**; `deformLeafLattice` ported by W3-A but never oracle-checked; `setPolymeshObjectVertices` absent from core |
| maize | `MaizePhytomerCreationFunction`: internode / leaf scale ramps, `leaf_flexibility` taper by rank, ear bud ACTIVE at node `max_nodes - 6`, other buds DEAD | absent |
| maize | `setGrassFlagLeafAngle` -> `Phytomer::rotateLeaf` (node 20) | **absent** |
| maize | 2 internode length segments, 10 radial subdivisions, `max_vegetative_buds_per_petiole = 0` (no laterals), `max_floral_buds_per_petiole = 1` | ported, untested |
| maize | `max_terminal_floral_buds = 1`: terminal tassel bud at the node cap (`appendPhytomerToShoot`) | ported (W3-C), reaches `updateInflorescence` (W3-D) |
| maize | phenology `(0, -1, -1, 4, 58, 1000)`: ACTIVE -> FRUITING directly (no flower prototype) | ported (W3-C), reaches `updateInflorescence` |
| maize | `MaizeEarPrototype` (OBJ), `MaizeTasselPrototype` (procedural, ~10^5 Context draws), organ build-parameter declarations (`registerPlantModelBuildParameters`) | absent; inflorescence-only (W3-D) |

Not reached by either crop: stolons (1.3.98's strawberry has a single `mainstem` shoot type and no stolon), leaf
twist, texture stages, phytomer callbacks, determinate tassel growth beyond `inflorescence_maturity_period`.

## What is ported

| file | contents | C++ |
|---|---|---|
| `library/strawberry.py` | `initializeStrawberryShoots`, `buildStrawberryPlant`, `StrawberryFlowerPrototype`, `StrawberryFruitPrototype` (OBJ via `obj_prototype`), `named_hook` (get-or-create a `hooks.<name>` slot without editing `hooks.py`) | PL:5900-6012, AS:4068-4078 |
| `library/maize.py` | `initializeMaizeShoots`, `buildMaizePlant`, `MaizePhytomerCreationFunction`, `setGrassFlagLeafAngle`, `MaizeEarPrototype` (OBJ); `MaizeTasselPrototype` is a bound hook that raises (procedural tassel, inflorescence-only) | PL:4574-4692, AS:2007-2110, 3549 |
| `leaf_prototype.py` | `recordSpectralClassMap`, `class_map_covers_a_class` (= `!classMapAreaFractions(...).empty()`, same cell walk and Sutherland-Hodgman clipping in double, early exit), `_read_png_single_channel` | PA:714-787, global.cpp:2001-2220 |
| `phytomer.py` | `record_leaf_prototype_rest_geometry` flexible branch; `Phytomer.deformLeafUnderSelfWeight`; `Phytomer.rotateLeaf`; `flexibility_aging_progress` / `_multiplier`; `set_polymesh_object_vertices` (Polymesh::setVertices); glibc `expf` | PA:4548-4595, 5191-5352, 4252-4306; Context_object.cpp:4459-4503 |

## Shared-file hunks (every one fills a branch that raised or was absent; tomato leaves are rigid, so none runs for it)

1. `library/__init__.py`: 6 lines appended at the end (import + 4 registrations, via lambdas so they resolve the
   module attribute at call time). `architecture.py`, `hooks.py` untouched.
2. `leaf_prototype.py`, `GenericLeafPrototype`: the class-map branch's `raise NotImplementedError(...)` replaced by
   the `recordSpectralClassMap(...)` call; the class-map functions appended after the function.
3. `phytomer.py`, module level: `expf`, `flexibility_aging_progress`, `flexibility_aging_multiplier`,
   `set_polymesh_object_vertices` added before `_v`.
4. `phytomer.py`, `record_leaf_prototype_rest_geometry`: the `leaf_flexibility > 0` `raise` replaced by the flexible
   branch.
5. `phytomer.py`, `Phytomer.deformLeafUnderSelfWeight`: the `raise` after the rigid guard replaced by the port
   (reshaped leaves, `leaf_shape_state`, still raise).
6. `phytomer.py`, new method `Phytomer.rotateLeaf` (after `deformLeafUnderSelfWeight`).

## Parity (`cases_plantarch_w4c.py`, `ops_plantarch_w4c.py`): Context seeds 1..8, all 0 ulp

PARITY_TABLE

Inflorescence gate: `inflorescence_ported()` = `heliades.plantarchitecture.inflorescence` importable (W3-D). The
oracle decides per (age, seed) whether flowers / fruit exist (`w4c_inflorescence_gate`, PyHelios on both sides),
and a gated pair returns `[-999, n_organs]` on both backends. Today: `gallery` is gated for every seed (strawberry
day 60: 6 flowers; maize day 50: the ear); `gallery_vegetative` and `flag_leaf` run the same builds with W3-C's
draws-only inflorescence on Heliades and compare everything vegetative. When W3-D lands, `gallery` runs in full
without edits. The oracle's floral schedule is seed-independent (seeds 1..8): strawberry's first flower on day 40
(1, 2, 3 flowers on days 40, 42, 44), maize's ear set on day 46.

## Mutation checks (`tests/heliades/test_plantarch_w4c.py::test_mutations_are_caught`)

MUTATION_TABLE

## Float / double choices (each proven or found indistinguishable)

| expression | evaluated as | evidence |
|---|---|---|
| `leaf_flexibility *= 1.f - 0.75f * fmax(0.f, r - 0.45f) / 0.55f` (AS) | C `fmax` -> double chain, narrowed once | mutation `maize_flex_factor_float` |
| `fmin(1.f, 0.7 + 0.3 * float(i) / 5.f)` (AS) | double | mutation `maize_inode_scale_float` |
| `fmin(scale, fmin(1.f, 0.475f + 0.092f * r))` | float arithmetic; `fmin` of floats is exact in either type | by construction |
| `flexibilityAgingProgress`: `expf` | glibc `expf` (ctypes) | mutation `aging_exp_double` |
| `randu(0.f, 0.035f * M_PI)` | double product, narrowed | not distinguishable: float and double products give the same float (0.10995574) |
| `(fabs(ind) - 0.5f) * offset * length` (W3-B open item) | ported as double | still not distinguishable: with 3 leaflets `fabs(ind) - 0.5 = 0.5` is a power of two, so both forms round identically (checked: the float mutation passes strawberry.steps) |

## Discrepancies / findings

1. **No stolons in 1.3.98.** The strawberry is one `mainstem` shoot type with `vegetative_bud_break_probability_min = 0`
   and decay -0.5 (bud break p = 0: no laterals); the brief's "stolons" are not in the pinned source.
2. **`deformLeafLattice` (W3-A, unit-tested only) is oracle-exact**, and so is the whole flexible-leaf chain: rest
   lattice in the prototype frame, gravity carried into it through the inverse of the leaf transform, ageing, the
   `1e-3` relative redeform tolerance, `setPolymeshObjectVertices` (global -> local through `makeNormalMatrix`, then
   `setTriangleVertices` from `transformPointByMatrix`).
3. **The maize tassel draws ~10^5 numbers from the Context stream** (`MaizeTasselPrototype`): a draws-only
   inflorescence cannot reproduce the engine-call count once the tassel exists (day 57 onward: 549 vs 104121 for
   seed 1). `flag_leaf` therefore skips the probe; the geometry, which is all built before the tassel, matches.
4. The ear (`MaizeEarPrototype`) and the strawberry flower/fruit are plain OBJ prototypes, bound here; their
   placement, scaling and the peduncle are W3-D's.
5. Maize leaves are re-deflected every step (`deformLeafUnderSelfWeight` from `setPetioleAndLeafScaleFraction` and
   from the mature-leaf path of the expansion loop). This dominates the maize cost: ~40 s per seed for 15 two-day
   steps (each step rewrites 800 triangles per leaf through `triangle_transform`).
6. Not ported (raise): `MaizeTasselPrototype`; `deformLeafUnderSelfWeight` of a reshaped leaf (`reshapeLeaf`);
   organ build parameters (`buildPlantInstanceFromLibrary` with maize tassel keys still raises in `architecture.py`).

## Requests for core (not done; `heliades/core` is owned by others)

- `Scene.set_polymesh_object_vertices` (Context::setPolymeshObjectVertices / Polymesh::setVertices). It lives in
  `phytomer.set_polymesh_object_vertices` for now and reaches into `scene._polymesh`, `polymesh.topology` and
  `scene.primitives[u].transform`.
- `Scene.set_triangle_vertices` (Context::setTriangleVertices) as a public method.
- `helios::readPNGSingleChannel` and `helios::classMapAreaFractions` (`leaf_prototype.py` has them for now).
- (still open from W3-B) `powf` / `atan2f` / `expf` in `core/vec.py`.
