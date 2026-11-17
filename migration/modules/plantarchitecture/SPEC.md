# PlantArchitecture: tomato porting spec (W1-D)

Baseline: Helios 1.3.98 (`third_party/pyhelios_src/helios-core`, see `migration/STATUS.md`).
Abbreviations:
- `PA` = `plugins/plantarchitecture/src/PlantArchitecture.cpp`
- `PL` = `.../src/PlantLibrary.cpp`
- `AS` = `.../src/Assets.cpp`
- `H` = `.../include/PlantArchitecture.h`
- `core/` = `helios-core/core/`

Line numbers are from the pinned tree. "Oracle" means PyHelios built from that tree
(`.venv/bin/python`), run single-process on Linux/GCC (Release `libhelios.so`).

Status tags used in the dependency checklist:
- **[P]** already ported in `heliades/core`
- **[A]** in progress in W1-A (textures/UV)
- **[B]** in progress in W1-B (object copy/transform/delete)
- **[C]** in progress in W1-C (shapes)
- **[M]** missing: no package owns it yet

Machine-readable symbol list: `tomato_calls.txt` (same directory).

---

## 0. Executive summary

- **One RNG.** Every draw on the tomato path comes from the Context's `std::minstd_rand0`. The plugin keeps a pointer to it (`PA:383`). There is one exception: the 35 unique leaflet blades are drawn from a *private* `std::minstd_rand0` seeded with a `uint` identifier, and that identifier was itself drawn from the Context generator when the model was loaded (`PA:3475`).
- **When to seed.** Seeding before or after `PlantArchitecture(ctx)` is equivalent. Seeding **after** `loadPlantModelFromLibrary` is not reproducible. Seeds 0 and 1 are the same plant, because of minstd's seed semantics (§3.4).
- **Exact draw counts.** RNG draw counts per stage were measured on the oracle with an external fingerprint method (§3.5) and they match the static analysis exactly:
  - load = 4 identifier draws (8-12 engine calls);
  - first phytomer = 26 or 27 calls.
- **Geometry kinds.** Tomato geometry is only:
  - textured triangles (leaf blades);
  - untextured triangles (petiolules);
  - color tubes (internode, petiole, peduncle);
  - OBJ polymeshes (flower, fruit);
  - object copy, scale, rotate and translate operations;
  - tube node and radius updates.
- **Core APIs still missing:** `loadOBJ` (W2-A), `appendTubeSegment`, `setTubeNodes`, `setTubeRadii`, `hideObject`, `scaleObjectAboutPoint`, polymesh topology and normals, `filterPrimitivesByData`, object data, and the material system.
- **Optional subsystems.** Carbohydrate, nitrogen, collision, leaf-angle tracking and optional object data are all OFF for a default tomato. They can be skipped for geometry parity.
- **Acceptance grid gap.** The proposed acceptance ages {10, 30, 50, 75} build **no flower or fruit**: the oracle shows the first inflorescence between 75 and 100 days. Add ages 100 and 110 or 120.
- **Main parity risks:**
  - double-precision `sincos` inside `rotatePointAboutLine` and `makeRotationMatrix`, and a double `atan` in the leaf lattice;
  - glibc `sinf`, `atanf` and `atan2f` versus numpy;
  - path-dependent incremental Context transforms over hundreds of sub-steps;
  - C++ argument evaluation order;
  - `Context::getAllUUIDs` iterating an `unordered_map`.

---

## 1. Entry points and call graph

```
Context()                                           core/src/Context.cpp:24    generator.seed(wall clock)  (:37-38)
Context::seedRandomGenerator(s)                     Context.cpp:57             generator.seed(s); unif/norm .reset()
PlantArchitecture(ctx)                              PA:382  generator = &ctx.generator (PA:383)
  initializePlantModelRegistrations()               PL:444  registerPlantModel("tomato", initializeTomatoShoots, buildTomatoPlant) PL:501
                                                            (no leaf-inclination Beta -> no tracking, PL:863-880)
  output_object_data[*] = false                     PA:389-402
loadPlantModelFromLibrary("tomato")                 PL:631 -> initializeDefaultShoots PL:848 (shoot_types.clear()) -> initializeTomatoShoots PL:6101
  LeafPrototype(gen)                 [draw id#1]    PA:798, sampleIdentifier H:994
  PhytomerParameters(gen)            [id#2, id#3]   PA:951; PA:983 `leaf.prototype = LeafPrototype(generator)` (ctor + operator=)
  phytomer_parameters.leaf.prototype = leaf_prototype  [id#4]  PL:6146 (LeafPrototype::operator=, H:953-991)
  ShootParameters(gen)               [no draw]      PA:1084 (member PhytomerParameters default-constructed, null gen)
  shoot_parameters.phytomer_parameters = ...        [no draw]  LeafParameters::operator= -> duplicate() H:1178
  defineChildShootTypes({"mainstem"},{1})           PA:1130
  defineShootType("mainstem", sp)                   PA:1225 (map emplace = copy ctor, no draw)
buildPlantInstanceFromLibrary(pos, age, {})         PL:649 -> PL:653
  validateBuildParameters (empty -> no-op)          PL:567
  build params sampling (empty -> none)             PL:672-679
  buildTomatoPlant(pos)                             PL:6205
    addPlantInstance(pos, 0)                        PA:11410  plant_instances[plant_count]; shoot_types_snapshot = shoot_types (copy)
    make_AxisRotation(0, randu(0,2pi), randu(0,2pi))  PL:6214  [2 float draws; GCC evaluates RIGHT-TO-LEFT: roll drawn first, oracle-verified §3.3]
    addBaseStemShoot(pid,1,rot,0.002,0.04,0.01,0.01,0,"mainstem")  PA:7146
      validateShootTypes                            PA:7512
      new Shoot(...)                                PA:6951  (val() of constants only)
      Shoot::buildShootPhytomers                    PA:6975 -> Shoot::appendPhytomer PA:2007 -> appendPhytomerInternal PA:2017
        make_shared<Phytomer>(...)                  PA:2075 -> Phytomer::Phytomer PA:2694-3592  (§5)
        sampleChildShootType (1 label: no draw)     PA:7013
        sampleVegetativeBudBreak  [1 randu]         PA:7065/7096
        floral buds DORMANT->ACTIVE (no draw)       PA:2109-2127
        propagateDownstreamLeafArea                 PA:2550
        optional setObjectData (all off)            PA:2135-2183
        updateLeafTextureStage (no-op)              PA:4855
        TomatoPhytomerCreationFunction              AS:4092 (via PhytomerCallbackScope, PA:2201)
          setFloralBudState(BUD_DEAD) rank0 && node<8; rank>1 kills both
          scaleLeafPrototypeScale(fmin(1, 0.5+0.5*age/10))   PA:5846 -> scaleObjectAboutPoint per leaflet
          scaleInternodeMaxLength(fmin(1, 0.7+0.3*age/10))   PA:4349
    breakPlantDormancy(pid)                         PA:10069 -> Shoot::breakDormancy PA:2221
          [1 randu per vegetative bud NOT dead, PA:2246-2252]  <- the 26/27 variance
    setPlantPhenologicalThresholds(pid,0,40,5,5,30,1000)  PA:11669
    max_age = 365
  plant_name set; output_object_data plant_name/plant_type off  PL:700-722
  collision registration off                       PL:725-731
  leaf-inclination tracking: tomato not in map     PL:743-746
  if age>0: advanceTime(pid, age)                  PL:753-755 -> PA:11723 -> PA:11728-12465 (§6)
```

PyHelios is a thin wrapper (`native/src/pyhelios_wrapper_plantarchitecture.cpp:1175-1262`). The only thing it adds is a cwd change around calls (`pyhelios/PlantArchitecture.py:157`).

---

## 2. Dependency checklist: Context API and `helios::` functions on the tomato path

Per-call overload details come from §5/§6. "Geometry-relevant" means the call changes vertices, UUID order, color or texture.

| API (overload) | used at | geometry-relevant | status |
|---|---|---|---|
| `Context::seedRandomGenerator`, `getRandomGenerator`, `randu()`, `randu(float,float)`, `randu(int,int)` (a==b → no draw), `randn()` (persistent polar spare) | PA:383, PL:6214, PA:3078-3079, 3503, 7096, 2248, 11991, 12041 | RNG | **[P]** (`core/random.py` `ContextRandom`; `randu(int,int)` a==b shortcut: verify) |
| `helios::uniform_int_distribution<uint>` full range (identifier) | H:994 | RNG | **[P]** (`uniform_integer(g, 2**32-1)`, verified vs oracle counts) |
| `helios::uniform_real_distribution<float>`, `normal_distribution<float>` (fresh per RandomParameter call) | H:179-189 | RNG | **[P]** |
| `addTubeObject(Ndiv, nodes, radii, vector<RGBcolor>)` | PA:3276 (internode), PA:12879-12938 `makePetioleTube` (petiole), PA:4063 (peduncle, Ndiv 8) | yes | **[P]** colored tube |
| `appendTubeSegment(obj, node, radius, RGBcolor)` | PA:3287 | yes | **[M]** |
| `setTubeNodes`, `setTubeRadii` (path-dependent radius-only!), `getTubeObjectNodes`, `getTubeObjectNodeRadii` | PA:2513-2516 (`updateShootNodes(true)`), 4402-4530, 3860-3930 | yes | **[M]** |
| `addTriangle(v0,v1,v2, texture, uv0,uv1,uv2)` + solid fraction from alpha | AS:~700-760 (GenericLeafPrototype blade) | yes | **[A]** |
| `addTriangle(v0,v1,v2)` (default color) | AS petiolule (104 per leaflet) | yes | **[P]** |
| `getPrimitiveArea` (keep triangle iff area>0, else `deletePrimitive`; UUID consumed) | AS GenericLeafPrototype | yes | **[P]** area (textured area×solid fraction **[A]**), `deletePrimitive` **[P]** |
| `translatePrimitive(vector, shift)` | AS:769, 790 | yes | **[P]** single; vector overload trivial |
| `addPolymeshObject`, `setPolymeshObjectTopology`, `computePolymeshObjectVertexNormals(obj,180)` | AS GenericLeafPrototype end; AS:4081/4087 | topology/normals only | `addPolymeshObject` **[P]**; topology + normals **[M]** (not vertex-relevant; needed for USD) |
| `loadOBJ(file, origin, height=0.75, nullrotation, black, "ZUP", silent)` | AS:4081, 4087 | yes | **[M]** → W2-A. TomatoFlower.obj: 360 faces (277 tri, 78 quad, n-gons of 5/7/15/32); TomatoFruit.obj: 863 faces (796 quad, 64 tri, three 32-gons); materials by `Kd`, no textures |
| `copyObject(obj)` (copies primitives in order, transform, material, primitive data; **not** hidden flag, unverified) | PA:3505, PA:3623ff | UUID order | **[B]** |
| `scaleObject(obj, vec3)` (about origin) | PA:3523, createInflorescenceGeometry | yes | **[B]** |
| `scaleObjectAboutPoint(obj, s, p)` (`makeScaleMatrix` uses `p*(1-s)`, core/src/global.cpp:754) | PA:5846-5868, 6744-6758 | yes | **[B]** (in scope list) |
| `rotateObject(obj, a, "x"/"y"/"z")` | orientLeaf PA:5016-5197; createInflorescenceGeometry | yes | **[B]** |
| `rotateObject(obj, a, origin, axis)` | PA:4252-4305 rotateLeaf, createInflorescenceGeometry (yaw about peduncle axis) | yes | **[B]** |
| `translateObject(obj / vector<obj>, shift)` (zero shift early-return, Context_object.cpp:2636/2646/2730) | PA:3561, 4199-4250 (every substep), 4402-4530 | yes | **[B]** |
| `deleteObject(vector)` | `setFloralBudState` PA:1491 | UUID gaps | **[B]** |
| `hideObject(obj)`; `getAllUUIDs` skips hidden (Context.cpp:540-555) | PA:3493 (35 blade prototypes), PA:3733 (3 inflorescence prototypes) | visibility | **[M]** |
| `getObjectPrimitiveUUIDs`, `doesObjectExist`, `getObjectTransformationMatrix`, `getObjectArea` | many | query | **[P]** (`getObjectArea` **[M]**, trivial) |
| `setPrimitiveData(uuids, "object_label", str)` | PA:3278/3290, 3427, AS petiolule, labelLeafPrototype PA:4968, PA:4063 | label | **[P]** single-UUID; vector overload trivial |
| `filterPrimitivesByData(uuids, "object_label", "petiolule")` | PA:4968-4985 | no | **[M]** (trivial) |
| `setObjectData(obj, "leaf_area"/"volume_ratio"/"old_shoot_volume", float)` (unconditional) | PA:7625ff, PA:12291-12355 | no | **[M]**; skip for geometry parity |
| Materials: `getPrimitiveMaterialLabel/ID`, `getMaterial`, `doesMaterialExist`, `addMaterial`, `renameMaterial`, `assignMaterialToPrimitive`, `setMaterial*` (`nameOrganMaterials` PA:181-264, `assignPetioluleMaterial` PA:4987, `makeClosedFruitSurfacesOneSided` PA:54ff) | every organ | color identical; labels only | **[M]**. Skip for geometry; color/texture per primitive must still match (petiolule takes petiole's material → petiole color) |
| `helios::rotatePointAboutLine` (**double `sincos`**, core/src/global.cpp:214-247) | PA ~40 sites | yes | **[P]** (`vec.rotate_point_about_line`: verify it uses double sin/cos) |
| `helios::makeRotationMatrix` (axis string / vector; **double `sincos`**, global.cpp:599-713), `makeScaleMatrix`, `makeTranslationMatrix`, `matmult` (naive i,j,k from 0.f, global.cpp:773) | via object transforms | yes | **[P]** axis-string; vector-axis **[B]** |
| `cross`, `normalize` (`*1/sqrtf`), `magnitude`, `vec3/a` (reciprocal multiply) | everywhere | yes | **[P]** |
| `deg2rad(d) = d*float(M_PI)/180.f` (global.cpp:855), `rad2deg`, `acos_safe`/`asin_safe` (acosf/asinf, global.cpp:389-403), `clamp`, `sum` | everywhere | yes | **[M]** (trivial; put in `core/vec.py`) |
| `cart2sphere`/`sphere2cart` | PA (collision/tracking paths only) | no | not needed |
| `make_AxisRotation`, `AxisRotation` | PL:6214, PA | yes | **[M]** (data class) |
| `helios::resolvePluginAsset` | AS:4081/4087 | path | **[M]** (trivial) |
| `readPNG` (palette PNG `TomatoLeaf_blade.png`, 988×527 mode `P` with alpha) | texture alpha mask | solid fraction | **[A]** |
| `PlantArchitecture::interpolateTube(vec3/float)` (`fplus = 1.f + 1e-3` double→float, `fabs(frac-fplus)<0.0001`) | PA:325-378 | yes | plugin-internal |

Not on the tomato default path: `CollisionDetection`, `addSphere*`, `addTile`, `splitTubeObject`, `pruneTubeNodes`, `setPolymeshObjectVertices` (placeLeafStalk), `getPolymesh*`, texture series and class maps.

---

## 3. RNG

### 3.1 Generators

1. **Context generator** `std::minstd_rand0 Context::generator` (`core/include/Context.h:3887`).
   - `PlantArchitecture::generator` is `&ctx.generator` (`PA:383`).
   - Every `RandomParameter_*` stores the same pointer (`initialize(val, gen)`, PA:951-1127).
   - Also used by `context_ptr->randu/randn`.
2. **Private blade stream** `std::minstd_rand0 prototype_generator(unique_prototype_identifier)` (`PA:3475`), a local in the first Phytomer that needs the leaf cache.
   - `LeafPrototype::setRandomGenerator` points every prototype parameter at it and sets `sampled=false` (PA:902-922).
   - The Context generator is restored afterwards (PA:3497).
3. **Not used by tomato:**
   - the global `helios::randu()` mt19937 (`core/src/global.cpp:358-385`; PyHelios `seedGlobalRandomGenerator`);
   - leaf-angle tracker generators (H:3585);
   - `noiseLatticeValue` hash (PA:442).

### 3.2 Distributions

- **`RandomParameter_float` uniform** (H:179-181): `p0 + u*(p1-p0)` in float, with `u = helios::uniform_real_distribution<float>()(g)`.
  - That is one engine call: `float(double(g()-1)/2147483646.0)`, clamped below 1 (`core/include/portable_random.h:121-127`).
  - Bounds are stored as float: `137.5-40 → 97.5f`, `0.12f*0.9031f`.
- **Lazy cache** (H:167-176): `val()` draws only if `!sampled`, and `resample()` always marks the parameter sampled.
  - **A constant parameter never draws.**
  - Copies carry `sampled` and `constval`.
- **`Context::randu(int a,int b)`**: returns `a` when `a==b` with no draw (Context.cpp:519).
  - Otherwise it is rejection sampling (`uniformInteger`, portable_random.h:61-91).
  - `randu(0,4)` costs 1 call (it rejects with probability ≈5e-10).
- **Identifier** `uniform_int_distribution<uint>(0, UINT_MAX)`: the range exceeds the engine span (2^31-2), so it takes the nested path (portable_random.h:79-87).
  - It costs 2 calls per attempt and is rejected about 1/3 of the time, so 2, 4, 6... calls per identifier.
- **`Context::randn()`**: Marsaglia polar in double. It costs ≥2 double uniforms, and each double uniform is 2 chunks of 30 bits with about 50% chunk rejection.
  - A persistent spare is kept in the Context `norm_distribution`, so two consecutive `randn()` usually consume one polar pair.
  - `seedRandomGenerator` resets the spare.
  - (A `RandomParameter` "normal" builds a fresh distribution each call. Tomato has none.)

### 3.3 Draw order (tomato)

**Distributed parameters.** Tomato has exactly these, each `uniform`:
- `internode.phyllotactic_angle` U(97.5, 177.5)
- `petiole.pitch` U(45, 60)
- `petiole.curvature` U(-150, -50)
- `leaf.pitch` U(-30, 5)
- `leaf.prototype_scale` U(0.108379, 0.162558)
- `leaf.prototype.longitudinal_curvature` U(-0.45, -0.2)
- `ShootParameters::base_yaw` U(-20, 20)

Everything else is constant (PL:6101-6203).

**Load** (`loadPlantModelFromLibrary`):
- 4 identifier draws, as in §1.
- The 4th identifier (from `PL:6146`) is the one stored in `shoot_types["mainstem"]` and seeds the blade stream.
- Oracle: engine calls = 10 (seed 5), 12 (seed 1), 8 (seeds 2 and 3). This matches `heliades.core.random.uniform_integer(g, 2**32-1)` ×4 exactly.

**Build at age 0, Context stream, in order:**
1. `randu(0, float(2π))` → **roll**, then `randu(0, float(2π))` → **yaw**.
   - GCC evaluates the arguments of `make_AxisRotation` right to left.
   - Oracle-verified: the XML `<base_rotation>` = (0, draw2°, draw1°) for seeds 5, 2 and 3.
2. Phytomer 0 constructor (`PA:2694`). These are `PhytomerParameters::resample()` (PA:1009-1078, fixed order), then the constructor's own draws:
   - (a) `phyllotactic_angle`, (b) `petiole.pitch`, (c) `petiole.curvature`, (d) `leaf.pitch`, (e) `prototype_scale`, (f) `longitudinal_curvature`. (f) is wasted: leaves are copies of cached blades.
     - Oracle-verified: XML `<internode_phyllotactic_angle>` = 97.5 + 80·u, where u is the first draw after the base rotation.
   - Phytomer index > 0 only: `randn()`, `randn()` for the OU tortuosity terms (PA:3078-3079). Gate: `(|gravitropic|>0 || tortuosity>0) && shoot_index.x>0`; once per internode segment (1).
   - `petiole.pitch.resample()` (PA:3321). Skipped on the last node (index 15), where pitch is forced to 5° (PA:3312-3316). The value is unused (1 petiole).
   - `petiole.curvature.resample()` (PA:3340). Unused.
   - **If the blade cache is missing (first phytomer of the PlantArchitecture):** the private stream builds 35 blades. There are 36 private draws of `longitudinal_curvature`; blade `b = p*7+l` uses draw b (AS:688 val, AS:868 resample).
   - For each leaflet `l = 0..6`:
     - `randu(0,4)` picks the prototype (PA:3503);
     - `leaf.pitch.resample()` (PA:3538). Leaflet l uses pitch draw l: draw 0 is (d), and the 7th resample is wasted.
     - `roll`/`yaw` resamples (PA:3533/3544) are constant.
   - `prototype_scale.resample()` (PA:3572). Wasted.
   - Subtotal: 23 calls + randn.
3. `sampleVegetativeBudBreak`: 1 `randu`. The bud is DEAD if `u > 0.25` (PA:7096).
4. `breakPlantDormancy` → `Shoot::breakDormancy`: 1 more `randu` per vegetative bud not DEAD (PA:2246-2252).

Total 2+23+1(+1) = 26 or 27. Oracle: 26 (seeds 1 and 5), 27 (seeds 2 and 3). ✓

**Per `advanceTime` sub-step** (PA:11926-12269; one plant, shoots in vector order, captured count). For each shoot i:
1. **Floral state machine**, per phytomer → petiole → bud:
   - initiation (ACTIVE, `tc≥40`): `randu() < 0.2` → CLOSED, else DEAD;
   - opening (CLOSED, `tc≥5`): no draw;
   - fruit set (OPEN, `tc≥5`): `randu() < 0.8` → FRUITING, else DEAD.
   - Each non-DEAD state change rebuilds the inflorescence (§5.5), which costs **6 `phyllotactic_angle.resample()` draws** (one per flower, PA:4130). `createInflorescenceGeometry` uses `randu(0,0)`, which draws nothing.
2. **Leaf expansion**: per still-expanding (phytomer, petiole), in node order, **`leaf.prototype_scale.resample()` → 1 draw** (PA:12133).
   - This is interleaved per phytomer with vegetative bud break (ACTIVE and `phytomer.age+dt > 30`):
     - `snapshot.base_yaw.val()`, which draws only the first time per plant because the snapshot starts unsampled;
     - `base_yaw.resample()`, 1 draw (PA:12174);
     - `addChildShoot` → that child's phytomer-0 constructor draws (§3.3 step 2, including the blade-cache check), then 1 bud-break `randu`. The child gets no `breakDormancy` (`isdormant=false` is set directly, PA:12224).
3. **Phyllochron**: one new phytomer per 2-day sub-step while nodes < 16 → constructor draws, then 1 bud `randu` (PA:12249-12269).

Engine calls measured on the oracle (seed 5): build age 0/2/4/10 = 36/66/101/208, including load.

### 3.4 Answer: does `seedRandomGenerator` change the tomato, and does timing matter?

Oracle `scripts/oracle/rng_exp.py`, age 20, vertex SHA1 over `getAllUUIDs`:

| seed set | seed 0 | seed 1 | seed 2 | repeat same seed |
|---|---|---|---|---|
| before `PlantArchitecture()` | fd70a3 | fd70a3 | ab79bd | identical |
| after ctor, before load | fd70a3 | fd70a3 | ab79bd | identical |
| after `loadPlantModelFromLibrary` | 807fc5 | 7d695 | 0b640 | **differs run to run** (same topology, 12 012 triangles) |
| never | — | — | — | differs |

- **Seeds 0 and 1 give the same plant.** `std::minstd_rand0::seed(s)` with `s mod (2^31-1) == 0` sets state 1 ([rand.eng.lcong]). This is the "older PyHelios gave identical tomatoes for seeds 0 and 1" effect. It is still true in 1.3.98, and Heliades's `MinstdRand0.seed` reproduces it. Use seeds 1..8 in the acceptance grid.
- **Before or after the PlantArchitecture constructor: no difference.** The plugin holds a pointer to the same engine.
- **After load: not reproducible.** The 4 identifier draws, and with them the private blade stream, came from the wall-clock seed.
- PyHelios `Context.seedRandomGenerator` → `context->seedRandomGenerator` (`native/src/pyhelios_wrapper_context.cpp:82`), not the global mt19937.

### 3.5 Oracle draw-count probe (no Helios changes)

1. Seed s, run a stage, then call `ctx.randu()`.
2. Find n such that `UniformReal(float)` applied to a copy of `MinstdRand0(s)` advanced n calls equals that value.

Script: `heliades/parity/tools/draw_count.py` (committed in W2-C; it replaced the first oracle-only prototype). It gives an exact engine-call count after any public-API stage, for example after each `advanceTime(2)`. It is the primary tool for localising an RNG divergence.

---

## 4. Data structures

### 4.1 `RandomParameter_float` / `RandomParameter_int` (H:57-306)

- Fields: `distribution` (string: constant/uniform/normal/weibull; int: constant/uniform/discretevalues), `distribution_parameters`, `mutable sampled`, `mutable constval`, `generator*`.
- Default copy semantics: the cache and the generator pointer are copied. `operator=(float)` makes the parameter constant.
- `RandomParameter_int` default `constval=1`.

### 4.2 `LeafPrototype` (H:701-1042)

Tomato values (PL:6104-6121):

| field | tomato value |
|---|---|
| `leaf_texture_file` | `{0:"TomatoLeaf_blade.png"}` |
| `leaf_aspect_ratio` | 0.5f/0.9031f, so Ny = ceil(6·0.5536) = 4 |
| `midrib_fold_fraction` | 0.1 |
| `longitudinal_curvature` | U(-0.45, -0.2) |
| `longitudinal_curvature_exponent` | 4 (default) |
| `lateral_curvature` | -0.3 |
| `wave_period` / `wave_amplitude` | 0.35 / 0.08 |
| `subdivisions` | 6 |
| `unique_prototypes` | 5 |
| `build_petiolule` | true |
| `petiolule_length` | `{0:0.085}` |
| `flexibility` / buckle / roll | 0 |
| `prototype_function` | `GenericLeafPrototype` (AS:640) |
| `leaf_offset` | 0 |

Hidden state:
- `unique_prototype_identifier`;
- private `generator`, `petiolule_radius_fraction` (set per blade by `buildLeafPrototype`, PA:4904-4966);
- `last_built_shape`.

Copy semantics: the copy constructor and `duplicate()` do not redraw the identifier; `operator=` does.

### 4.3 `PhytomerParameters` (H:1043-1375)

Sub-structs, each with a custom no-resample `operator=`:
- `internode` (pitch, phyllotactic_angle, radius_initial, max_*_buds_per_petiole, color, image_texture, length_segments, radial_subdivisions);
- `petiole` (petioles_per_internode, pitch, radius, length, curvature, taper, color, length_segments, radial_subdivisions, flexibility, flexibility_aging);
- `leaf` (leaves_per_petiole, pitch, yaw, roll, leaflet_offset, leaflet_scale, intercalary_leaflet_scale, prototype_scale, prototype);
- `peduncle` (length, radius, pitch, roll, yaw, curvature, color, length_segments, radial_subdivisions);
- `inflorescence` (flowers_per_peduncle, flower_offset, pitch, roll, flower/fruit prototype_scale and function pointers, fruit_class_map_file, fruit_gravity_factor_fraction, unique_prototypes, inflorescence_maturity_period).

Function pointers: `phytomer_creation_function`, `phytomer_callback_function`. `resample()` order: PA:1009-1078.

Tomato values: PL:6125-6168.
- internode: pitch 10, radial_subdivisions 7 (default), color (0.2495, 0.3162, 0.0657), length_segments 1.
- petiole: radius 0.002, length 0.2, taper 0.7, length_segments 5, radial 7.
- leaf: 7 leaflets, yaw 10, roll 0, leaflet_offset 0.15, leaflet_scale 0.7, intercalary 0.
- peduncle: length 0.18, radius 0.0015, pitch 20, roll 0, curvature -900, segments 5, radial 8.
- inflorescence: 6 flowers, offset 0.15, pitch 80, roll 180, flower scale 0.05, fruit scale 0.15, gravity 0, unique_prototypes 1 (default).

### 4.4 `ShootParameters` (H:1377-1646)

- Contains `phytomer_parameters` plus geometric and growth `RandomParameter`s, the three bools, and child labels/probabilities.
- Custom `operator=` does not resample. Its comment at H:1585 says otherwise and is stale (notes/W1-D.md).
- Tomato values (PL:6172-6200):
  - geometry: max_nodes 16, insertion_angle_tip 30, internode_length_max 0.04, base_roll 90, base_yaw U(-20,20), gravitropic_curvature 150, tortuosity 75, tortuosity_persistence_length 0.5 (default);
  - growth: phyllochron_min 2, elongation_rate_max 0.1, leaf_expansion_rate_max unset (→0.1), girth_area_factor 2, vegetative_bud_break_time 30, bud-break p_min 0.25, decay -0.25, flower_bud_break_probability 0.2, fruit_set_probability 0.8, determinate.

### 4.5 Runtime structures

- **`Shoot`** (H:2968-3364).
  - Identity and topology: `ID`, `parent_shoot_ID`, `parent_node_index`, `parent_petiole_index`, `rank`, `childIDs` (`std::map<int, vector<int>>`).
  - Growth state: `current_node_number`, `nodes_this_season`, `isdormant`, `meristem_is_alive`, `phyllochron_counter`, `phyllochron_instantaneous`, `elongation_rate_instantaneous`, `leaf_expansion_rate_instantaneous`, `gravitropic_curvature`, OU state `curvature_perturbation` / `yaw_perturbation`, `growth_state`.
  - Geometry: `base_position`, `base_rotation`, `internode_tube_objID` (sentinel 4294967294), `shoot_internode_vertices` / `shoot_internode_radii` (`vector<vector<>>`, per phytomer), `phytomers` (vector of `shared_ptr<Phytomer>`).
  - Its own copy of `shoot_parameters`.
- **`Phytomer`** (H:1691-2966).
  - Its own resampled `phytomer_parameters`.
  - Indices: `shoot_index` (x = index, y = node count at creation, z = max_nodes), `rank`, `age`.
  - Internode: `internode_phyllotactic_azimuth`, `internode_pitch`, length max/scale fraction, radius.
  - Petiole: `petiole_vertices`/radii per petiole, `petiole_objIDs`, `petiole_length`, scale factors.
  - Leaves: `leaf_objIDs[petiole][leaf]`, `leaf_bases`, `leaf_size_max`, leaf rotation records.
  - Buds: `vegetative_buds` and `floral_buds` (`vector<vector<>>`).
  - Inflorescence: `inflorescence_bending_axis`.
- **`VegetativeBud`** (H:616): state, shoot_type_label, shoot_ID.
- **`FloralBud`** (H:626): state, time_counter, age, isterminal, parent/bud index, fruit scale factors, base_position/rotation, bending_axis, inflorescence bases/rotations/base scales, `peduncle_objIDs`, `inflorescence_objIDs`, contact swing.
- **`PlantInstance`** (H:3366-3615).
  - `shoot_tree` (vector of `shared_ptr<Shoot>`, index = shootID), `base_position`, `current_age`, `time_since_dormancy`, `plant_name`.
  - Phenology: the `dd_*` thresholds, `max_leaf_lifespan`, `is_evergreen`, `max_age`.
  - Parameters: `shoot_types_snapshot`, build/organ parameters.
  - Other state: shadow grids (`unordered_map`, unused for tomato), carbon and nitrogen parameters/pools (unused), leaf-angle tracker (unused).

### 4.6 Per plant vs shared

**Per plant:** `PlantInstance`, which holds `shoot_types_snapshot`.
- The snapshot is mutated during growth: `base_yaw` is resampled on lateral creation, so its `sampled` cache carries over from one lateral to the next.
- Each `Shoot` has its own `ShootParameters` copy, and each `Phytomer` has its own resampled `PhytomerParameters`.

**Shared per `PlantArchitecture`:**
- `shoot_types` (H:6623);
- `unique_leaf_prototype_objIDs` / `unique_leaf_prototype_rest_geometry`, keyed by `(identifier, leaves_per_petiole)` (H:6632-6660);
- flower and fruit prototype caches, keyed by `(function pointer, organ build params)` (H:6700-6704);
- `organ_material_labels` (H:6718).

**Consequence:** a second tomato built by the same `PlantArchitecture` reuses the hidden prototypes. It makes no private-stream draws and creates no prototype UUIDs, so its UUIDs differ from those of a tomato built alone. The prototypes are deleted only by `deleteAllPrototypes` when no plants remain (PA:11655). The parity harness must build one plant per `PlantArchitecture`, or test both cases on purpose.

---

## 5. Geometry per organ

### 5.1 Phytomer constructor sequence (tomato, PA:2694-3592)

```
pp = params; pp.resample()                                    # §3.3 (a)-(f)
shoot_index=(k, cur_nodes, 16); Lint = frac*Lmax; petiole_len = lsf*0.2; r0 = max(lsf,0.3)*0.002   (unverified: lsf form)
pitch_i = deg2rad(10); phyl = deg2rad(pp.phyl.val()); azimuth = prev_azimuth + phyl (0 for k=0)
axis = parent_axis; rot_ax = cross(parent_axis, parent_petiole_axis) or (1,0,0)
k==0: axis = rotPAL(axis, rot_ax, 0.5*pitch_i); then roll (about parent axis), pitch (0, skipped), yaw (base_rotation) PA:2913-2955
k>0 : axis = rotPAL(axis, rot_ax, -1.25*pitch_i) PA:2965; OU tortuosity: 2 randn; gravitropic + transverse + azimuth rotations PA:3078-3137; normalize
v1 = v0 + Lint*axis -> shoot_internode_vertices; k==0 addTubeObject(7,...) else appendTubeSegment(...) ; setPrimitiveData(all tube UUIDs,"object_label","shoot")
petiole: pa = rotPAL(axis, rot_ax, |deg2rad(pitch)|) (5 deg on last node); pitch.resample(); rotate pa & rot_ax about axis by phyl (k!=0)
         curv = curvature.val(); curvature.resample()
         for j=1..5: pa = rotPAL(pa, rot_ax, -deg2rad(curv*dr)); v_j = v_{j-1} + dr*pa; r_j = r0*(1 - taper*j/5)   (dr = petiole_len/5)
         makePetioleTube: drop nodes < 1e-6 apart, clamp r>=1e-5, addTubeObject(7, v, r, colors); label "petiole"   PA:12879-12938
buds: data only (no geometry)
tip = getAxisVector(1.0, petiole) (frac 0.9..1, interpolateTube)                              PA:1307-1327, 351-378
if cache[(id,7)] missing: build 35 hidden blade polymeshes from the private stream (§5.3)
for l in 0..6: p = randu(0,4); obj = copyObject(cache[p][l]); scaleObject(lsf*0.7^|l-3|*ps)
               orientLeaf: rotate "x" roll, "y" -pitch, (laterals) "z" yaw, "z" azimuth+compound   PA:5016-5197
               translateObject(leaf_base_l); pitch.resample()
bendPetioleUnderLeafWeight -> early return (flexibility 0, PA:5476); prototype_scale.resample()
```

Leaf base:
- Terminal leaflet: at the petiole tip.
- Laterals: `offset = (|ind|-0.5f)*0.15f*length`, and the base is `interpolateTube(petiole_vertices, 1 - offset/length)` (PA:3555-3558).
- Size: `leafletSizeMax = powf(0.7f, slot)*prototype_scale` (PA:309-323).

`compoundLeafRotation` (PA:4597-4617) is 0 for the tip and `∓0.5*PI_F` for the laterals. That value is computed in double and narrowed to float.

`orientLeaf` angles:
- `petiole_elevation = asin_safe(tip.z)` (asinf) and `petiole_azimuth = atan2f(tip.y, tip.x)`.
- Past-vertical branch: if `stem.z > cosf(0.25f*PI_F)`.
- Terminal leaflet: roll = 0 and pitch = leaf_pitch + elevation.
- Laterals: roll = `(leaf_roll - elevation)*sign`, yaw = `-sign*deg2rad(10)`.
- Zero-angle and zero-shift transforms return early in the Context (they must early-return identically in Heliades, or the matrices differ by rounding).

### 5.2 Internode, petiole and peduncle tubes

- **Internode.** There is one tube object per shoot. It is created by `addTubeObject(7, …, vector<RGBcolor>)` with the first phytomer and extended by `appendTubeSegment` (PA:3270-3293).
  - Afterwards the vertices are edited **in place** only by `Shoot::updateShootNodes(true)` → `setTubeRadii` then `setTubeNodes` (PA:2513-2516).
  - `Tube::setTubeNodes` recomputes the cross-sections from scratch (core Context_object.cpp:3774-3857), so the final internode geometry depends only on the final node and radius arrays.
  - With collision off, `updateShootNodes(true)` runs once per plant at the end of each `advanceTime` call (PA:12408-12464).
  - In every sub-step it runs with `false`, which only recomputes the child-shoot origins and calls `setPetioleBase` (PA:2429-2535, 4199).
- **Girth** (PA:7625-7680):
  - `area = gaf*supported_leaf_area*1e-4`: float×float, then × double 1e-4, then narrowed to float;
  - `r = sqrtf(area/PI_F)`;
  - ratchet `r_i = r_i + 0.5*(r - r_i)` in double when r > r_i.
  - The radii are stored only; the Context is not updated during sub-steps.
- **Petiole.** One tube per petiole, built in the constructor.
  - Growth: `setPetioleAndLeafScaleFraction` (PA:4402-4530) recomputes the nodes and radii, then `setTubeNodes` and `setTubeRadii` (or `makePetioleTube` if it now qualifies).
  - Each leaflet is then moved with `translateObject(-old_base)`, `scaleObject(delta)`, `translateObject(new_base)`. This is **incremental and path-dependent**.
- **Peduncle** (`updateInflorescence`, PA:3934-4197).
  - Length 0.18 and `dr = L/5`.
  - Radius: `getReconciledPeduncleRadius` (PA:3783). It is 0.9×culm tip radius only on the last node of a max-node shoot; otherwise 0.0015.
  - Axis: the internode axis at frac 1, pitched by `deg2rad(20)+base_rotation.pitch` about the bending axis, then turned by the azimuth difference.
  - 5 segments by `bendPeduncleAxis(axis,-900,dr)` (PA:3820; `std::acos` clamp toward −z).
  - Then `addTubeObject(8, …)` with label "peduncle".
  - After that, `updatePeduncleRadii` (PA:3860) can call `setTubeRadii` alone. That is **path-dependent** (`Tube::setTubeRadii` rescales the existing offsets, Context_object.cpp:3732-3750), and so is `setTubeNodes`.

### 5.3 Leaf blade (`GenericLeafPrototype`, AS:640-924; lattice AS:516-638; deform AS:37-515)

**Cache build** (PA:3460-3497): `p = 0..4` outer, `l = 0..6` inner, `ind_from_tip = l-3`. For each blade:
- `buildLeafPrototype` (PA:4904):
  - `petiolule_radius_fraction` is built from the means of the shoot type's petiole radius, taper and leaf scale. The mean of a uniform is `0.5f*(a+b)`, and it draws nothing (PA:4844-4853).
- `GenericLeafPrototype` (Assets.cpp):
  - `Nx=6`, `Ny=ceil(aspect*6)=4`, `dx=1/6`, `dy=aspect/4`.
  - **Lattice** (`buildLeafBladeLattice`). `lateral_curvature` is nonzero, so the cross-section branch runs:
    - `fold = 0.5f*mf*1.5f*(1-x²)*float(M_PI)`;
    - 16 sub-samples per cell with `psi = fold + atanf(4*lc*s³/a⁴)`;
    - cos and sin summed (`sincosf`);
    - arc-length normalisation with `sqrtf`.
  - **Wave**: `phase = (x + wp*[j≥Ny/2])*M_PI/wp` in **double**, narrowed to float; `z_wave = 2|y|·wa·sinf(phase)`.
  - **Longitudinal curvature**: `dtheta -= atan(4*lc*powf(x,3)*dx)` with **double `atan`**, then `rotatePointAboutLine(v, 0, (0,1,0), dtheta)`, then the wave is added along `sinf/cosf(rot)`.
  - **Triangles**: two per cell (j outer 0..3, i inner 0..5), `addTriangle(v0,v1,v2,tex,uv…)` and `(v0,v2,v3,…)`, with UV = `(x, (y+0.5a)/a)`.
    - A triangle is kept only if `getPrimitiveArea>0`; otherwise `deletePrimitive`, and the UUID is still consumed.
    - Oracle: about 40 of the 48 survive per blade (28 840 textured triangles / 721 leaflets at age 75). The textured area uses the alpha-mask solid fraction **[A]**.
  - `translatePrimitive(0)`, then `translatePrimitive((0.085·blade_len?, 0, 0))` (AS:769, 790). The petiolule length is in prototype units, 0.085 (unverified: scaling).
  - **Petiolule**: 57 vertices, 104 untextured triangles (default color, later assigned the petiole material), label "petiolule" (`r*sqrtf(1-t²)`, `r*0.5f*(1+cosf(float(M_PI)*t))`, 8 spokes).
  - Trailing `resample()` of the shape parameters, drawn from the private stream.
  - `addPolymeshObject`, `setPolymeshObjectTopology`, `computePolymeshObjectVertexNormals(obj,180)`.
- `labelLeafPrototype` (blade → "leaf"), `assignPetioluleMaterial`, `recordLeafPrototypeRestGeometry` (only reads the transform), `nameOrganMaterials`, `hideObject`.
- The prototypes are **never deleted** and stay hidden: 35 objects, about 35×144 hidden UUIDs at the start of the UUID space.

Self-weight deflection is OFF: `flexibility=0` and the buckle parameters are 0, so `deformLeafUnderSelfWeight` returns at PA:5216.

### 5.4 Flower and fruit prototypes

- `TomatoFlowerPrototype` / `TomatoFruitPrototype` (AS:4080-4090): `loadOBJ(assets/obj/TomatoFlower.obj | TomatoFruit.obj, (0,0,0), height 0.75, nullrotation, RGB::black, "ZUP", silent)`, then `addPolymeshObject`.
  - `flower_is_open` is ignored, so open and closed flowers use the same mesh.
  - No textures; the colors come from the MTL `Kd`.
  - These are OBJ assets only (the `.blend` files are sources, not used).
- `ensureInflorescencePrototypesInitialized` (PA:3733-3773), on first use, **after** the first peduncle tube: closed flower, open flower, fruit.
  - Each: prototype function → `nameOrganMaterials` → (fruit: `makeClosedFruitSurfacesOneSided`, which touches materials only) → `hideObject`.
  - They are shared across plants.

### 5.5 Inflorescence placement (PA:3623-3730, 3934-4165)

- Rebuilt on every floral state change through `setFloralBudState` (PA:1491): `deleteObject` the old flowers and peduncle, then `updateInflorescence`.
- For flower `f = 0..5`:
  - `ind = |f-5|`;
  - base = `interpolateTube(peduncle, 1-(ind-0.5)*0.15)`, or the tip when `ind=0`;
  - `compound = deg2rad(phyl.val())*ind + 2π f`, then `phyllotactic_angle.resample()` (1 draw);
  - roll = `deg2rad(180)`, pitch = `-asin_safe(axis.z)+deg2rad(80)`, azimuth = `atan2(y,x)`, yaw = compound.
- `createInflorescenceGeometry` applies, in order:
  1. `copyObject(proto)`
  2. `scaleObject(s)` (0.05 flower / 0.15 fruit)
  3. `rotateObject("x", roll)`
  4. `rotateObject("y", pitch)`
  5. `rotateObject("z", azimuth)`
  6. `translateObject(base)`
  7. `rotateObject(yaw, base, peduncle_axis)`

  Each rotation is applied only if |angle| > 1e-6.
- Fruit: `setInflorescenceScaleFraction(0.25)` at set. Growth is `0.25+0.75·tc/30` through `scaleObjectAboutPoint(Δ)` (PA:6744-6758).

### 5.6 Object/UUID creation order

1. Plant → shoots in shootID order.
2. Within a sub-step, for each shoot:
   - (a) inflorescence rebuilds, in phytomer → petiole → bud order;
   - (b) child shoots born at each phytomer, in node order;
   - (c) the shoot's new apical phytomer.
3. Within a phytomer: internode tube triangles, petiole tube, [35 hidden blade prototypes, first time only], 7 leaflet copies.
4. Within an inflorescence: peduncle tube, [3 hidden prototypes, first time only], 6 copies.

Object and UUID counters only increase (unverified for deletions; W1-B should confirm).

---

## 6. Time stepping

`advanceTime(vector, T)` (PA:11728-12465). `buildPlantInstanceFromLibrary(age)` builds the plant at age 0 (a single phytomer), then calls exactly `advanceTime(pid, age)`.

**Sub-steps** (PA:11758-11800):
- `pmin` = minimum `phyllochron_min` over the existing shoots and the snapshot types (2 for tomato).
- If `T ≤ pmin`: `Nsteps=int(T)`, `dt=1`. Otherwise `Nsteps=floor(T/pmin)`, `dt=pmin`.
- A float remainder > 0 adds a last step with `dt=remainder`.
- So `advanceTime(20)` = `advanceTime(10)`×2, but ≠ `advanceTime(5)`×4. Oracle-verified (`scripts/oracle/grid.py split`).

**Per sub-step, per plant:**
1. **Max-age guard** (365) (PA:11853-11866).
2. `age += dt`.
3. **Shadow grid** (ON but without effect for tomato; read-only on the Context; skippable).
4. **Per shoot** (count captured first, PA:11926):
   - (i) phytomer `age += dt`, callback null.
   - (ii) dormancy break never triggers.
   - (iii) leaf lifespan 1e6, so OFF.
   - (iv) **floral state machine** (§3.3). Determinate growth: every initiation or fruit-set *event* calls `terminateApicalBud` and `terminateAxillaryVegetativeBuds` on that shoot, whatever the draw result (PA:11996-11999).
   - (v) **internode elongation** `setInternodeLengthScaleFraction(fmin(1,(L+dt·0.1·Lmax)/Lmax), false)`, then girth, then `updatePeduncleRadii`.
   - (vi) **petiole and leaf expansion** at rate 0.1/day toward 1, with one `prototype_scale` draw each. Fruit growth. Vegetative bud break at `phytomer.age+dt > 30` → `addChildShoot`:
     - pitch 30°, yaw = `base_yaw`, roll 90°;
     - length 0.04, radius `min(0.001, parent r_init)`;
     - rank+1;
     - `TomatoPhytomerCreationFunction` kills every bud at rank > 1.
   - (vii) node cap 16 → `terminateApicalBud`.
   - (viii) **phyllochron**: counter += dt, then a while loop appends phytomers (exactly 1 per 2-day step).
5. `updateShootNodes(false)`: child origins and `setPetioleBase` (`translateObject`, including zero shifts).
6. `setObjectData` volume_ratio / old_shoot_volume (unconditional, non-geometric).

**After all sub-steps:** `updateShootNodes(true)` writes the internode tubes into the Context.

**Phenology** (`dd_*`):
- flower initiation 40 days of an ACTIVE floral bud's `time_counter`;
- open +5;
- fruit set +5 (p = 0.8);
- maturity 30.

Rank-0 buds on nodes < 8 are dead, so the first inflorescence appears late. Oracle (seed 3): none at 75 d; at 100 d, 800 peduncle, 18 186 flower and 30 276 fruit triangles.

**Not exercised by default:** gravitropism and tortuosity are ON (per phytomer, §5.1); collision avoidance is OFF.

Oracle growth (seeds 0/3), in UUIDs/objects/shoots:

| age | seed 0 | seed 3 |
|---|---|---|
| 10 | 6 552 / 49 / 1 | 6 552 / 49 / 1 |
| 30 | 17 472 / 129 / 1 | 18 564 / 138 / 2 |
| 50 | 26 208 / 196 / 4 | 46 956 / 349 / 5 |
| 75 | 77 532 / 573 / 5 | 112 476 / 833 / 9 |

Each build takes ≤ 0.4 s on the oracle.

---

## 7. Optional subsystems (defaults for tomato)

| subsystem | flag / default | tomato | skip for geometry parity? |
|---|---|---|---|
| carbohydrate model (`CarbohydrateModel.cpp`) | `carbon_model_enabled=false` H:7009 | OFF | yes |
| nitrogen model (`NitrogenModel.cpp`) | `nitrogen_model_enabled=false` H:7024 | OFF | yes |
| collision / soft avoidance / solid obstacle / attraction points | H:7035, 7069, 7097; null ptr | OFF | yes (Phytomer early-returns, PA:1551, 1701, 1813-1834) |
| fruit-branch collision (`FruitCollision.cpp`) | H:7072, 7102 | OFF | yes |
| ground clipping | `ground_clipping_height=-99999` H:6743 | OFF | yes |
| leaf angle distribution tracking | not registered for tomato (PL:501) | OFF | yes |
| optional object data (`output_object_data`) | all false PA:389-402 | OFF | yes |
| unconditional object data `leaf_area`, `volume_ratio`, `old_shoot_volume` | always written | ON | yes for geometry; port later for USD/carbon |
| primitive data `object_label` | always | ON | **keep**: used by `filterPrimitivesByData` (petiolule) and by the organ queries |
| materials naming | always | ON | labels skippable; per-primitive color must match |
| shadow grid / shedShadedBranches / updateGrowthPriority | runs each step; effect only at dormancy (never for tomato) | no effect | yes |
| plant pose (`PlantPose.cpp`), USD/XML IO | user-invoked | OFF | yes (XML writer is useful as an oracle, §8) |
| leaf self-weight / petiole bending | flexibility 0 | no-op | yes, but keep the early-return guards |

---

## 8. Proposed Wave 2/3 work breakdown

All under `heliades/plantarchitecture/`. C++ sizes are the lines on the tomato path only (not whole files). Parity ops go in `heliades/parity/suites/plantarch/ops_*.py` and cases in `suites/plantarch/cases_*.py` (originally `heliades/parity/ops_plantarch_*.py` / `cases_plantarch_*.py`).

**Common intermediate-state oracle.** Write `plant_structure_dict()` in Heliades mirroring `PlantArchitecture::writePlantStructureXML` (InputOutput.cpp:519-1066). On the Helios side, parse the XML that PyHelios writes (`writePlantStructureXML`). Its tags give:
- per phytomer: sampled `internode_pitch`, `internode_phyllotactic_angle`, `petiole_pitch`, `petiole_curvature`, `leaf_pitch`, `leaf_yaw`, `leaf_roll`, `leaf_scale`, `internode_radii`, `internode_vertices`, bud states, `phyllochron_counter`;
- per shoot: `base_rotation`, `curvature_perturbations`, `yaw_perturbations`.

Together with the §3.5 engine-call probe after each `advanceTime(2)`, this localises a divergence to one sub-step, one phytomer and one parameter before any vertex is compared.

| pkg | scope | files | depends | parity cases (intermediate state) | C++ lines |
|---|---|---|---|---|---|
| **W2-C data layer** | `RandomParameter_float/int` (lazy cache, copy semantics), `LeafPrototype` (identifier draws on ctor/operator=, not copy), `PhytomerParameters` (+`resample` order), `ShootParameters`, `AxisRotation`, `BudState`, VegetativeBud/FloralBud, `initializeTomatoShoots`, `registerPlantModel`/`loadPlantModelFromLibrary`, §3.5 probe tool | `params.py`, `random_parameter.py`, `library/tomato.py`, `parity/tools/draw_count.py` | core RNG [P] | `plantarch.load.tomato` (exact): engine-call count after load = oracle, 8 seeds; identifier value vs. oracle (read back via XML `<leaf_prototype>` if exposed, else via blade hash after W3-A); `RandomParameter` uniform sample sequence | ~900 (H:57-306, 701-1646; PA:798-1232; PL:631-660, 848-880, 6101-6203) |
| **W2-D core gaps** (one core package) | `appendTubeSegment`, `setTubeNodes`, `setTubeRadii` (with the path-dependent rescale), `getTubeObjectNodes/Radii`, `hideObject` + hidden-aware `all_uuids`, `filterPrimitivesByData`, object data, `getObjectArea`, `setPolymeshObjectTopology` + `computePolymeshObjectVertexNormals`, `deg2rad`/`asin_safe`/`acos_safe`/`clamp`, `resolvePluginAsset`, minimal material table (label → color/texture, `assignMaterialToPrimitive`) | `core/objects.py` (tube methods), `core/scene.py` (hide/data), `core/vec.py`, `core/materials.py` | W1-A, W1-B | fuzz: tube append then setTubeNodes/Radii sequences (ulp); hidden prims excluded from `getAllUUIDs`; radius-only update repeated N times | ~700 core |
| **W2-A OBJ loader** (planned) | `loadOBJ(file, origin, height, rot, color, "ZUP", silent)`: face, n-gon triangulation and material order | `core/fileio.py` | W1-A | TomatoFlower.obj and TomatoFruit.obj: UUID count, per-triangle vertices and colors (ulp/exact) | (core) |
| **W3-A leaf prototype** | `GenericLeafPrototype`, `buildLeafBladeLattice`, petiolule, `deformLeafLattice` (no-op guards), `buildLeafPrototype`, `labelLeafPrototype`, `assignPetioluleMaterial`, cache build with private stream | `leaf_prototype.py` | W2-C, W2-D, W1-A, W1-B | `plantarch.leafproto` (ulp): seed a private minstd with identifier k, build 35 blades, compare each hidden object's vertices/UV/solid fraction; oracle side reads hidden prototype UUIDs from `getAllPrototypeObjectIDs` (PA:11009, if exposed; else C++ trace driver); also a fuzz over curvature/wave values | ~1100 (AS:28-924; PA:4904-5014, 4532-4597) |
| **W3-B phytomer geometry** | `Phytomer` constructor (internode axis, tortuosity OU, petiole), `makePetioleTube`, `interpolateTube`, `getAxisVector`, `orientLeaf`, `compoundLeafRotation`, `leafletSizeMax`, leaf copy/place, `scaleLeafPrototypeScale`, `scaleInternodeMaxLength`, `TomatoPhytomerCreationFunction` | `phytomer.py`, `library/tomato.py` (creation fn) | W3-A, W2-C, W1-B | `plantarch.tomato.age0` (ulp, seeds 1..8): engine calls (26/27), XML per-phytomer params, then the 1092 vertices sorted by UUID | ~1600 (PA:2694-3592, 1307-1330, 325-378, 4597-4617, 5016-5197, 4349-4402, 5846-5876, 12879-12938; AS:4092-4108) |
| **W3-C shoot & growth** | `Shoot`, `appendPhytomerInternal`, bud sampling, `breakDormancy`, `addBaseStemShoot`, `addChildShoot`, `appendPhytomerToShoot`, `advanceTime` (substeps, floral state machine, elongation, girth, petiole/leaf scale, child shoots, phyllochron), `setInternodeLengthScaleFraction`, `setPetioleAndLeafScaleFraction`, `updateShootNodes`, `setPetioleBase`, `incrementPhytomerInternodeGirth`, `addPlantInstance`, `setPlantPhenologicalThresholds` | `shoot.py`, `growth.py`, `plant.py` | W3-B | `plantarch.tomato.steps` (ulp): ages 2,4,…,40 in 2-day steps; after each step compare engine-call count, XML shoot/phytomer state, then vertices. Includes the `advanceTime` split case (20 vs 10+10 vs 5×4) | ~2300 (PA:2007-2694, 4308-4532, 6951-7300, 7523-7680, 10069-10082, 11410-11431, 11669-11688, 11715-12465 minus disabled branches) |
| **W3-D inflorescence** | `setFloralBudState`, `updateInflorescence`, `bendPeduncleAxis`, `getReconciledPeduncleRadius`, `updatePeduncleRadii`, `ensureInflorescencePrototypesInitialized`, `createInflorescenceGeometry`, `setInflorescenceScaleFraction`, Tomato flower/fruit prototype fns | `inflorescence.py`, `library/tomato.py` | W3-C, W2-A | `plantarch.tomato.flowering` (ulp): ages 100, 110, 120 × seeds 1..8 (flowers exist); per-bud state from XML `<floral_bud>`/`<bud_state>`; per-flower base, rotation and scale; then vertices | ~700 (PA:1449-1547, 3623-4197, 3783-3933, 6744-6760; AS:4080-4090) |
| **W3-E library build + acceptance** | `buildPlantInstanceFromLibrary`, `buildTomatoPlant`, organ queries (`getAllLeafUUIDs`, …, by `object_label`), `plant_structure_dict` | `library/__init__.py`, `api.py`, `structure.py` | all | `plantarchitecture.tomato` (ulp): ages {10,30,50,75,**100,120**} × seeds {**1..8**}; also two plants in one PlantArchitecture (shared prototype cache) | ~500 (PL:649-760, 6205-6224; InputOutput.cpp:519-1066 subset; PA:11066-11400 subset) |

Parallelism:
- W2-C, W2-D and W2-A can run in parallel.
- W3-A and the W3-B "internode/petiole" half can start in parallel once W2-C and W2-D land. The leaf placement in W3-B needs W3-A.
- W3-C is sequential after W3-B. W3-D can be developed against W3-C's floral state machine stubbed with forced states.
- Total ≈ 7.8k C++ lines on the path, plus core gaps.

**Recommended extra tool (W2-C):** an out-of-tree C++ trace driver linked against the oracle `libhelios.so`. It would print every engine call (wrap the generator through `Context::getRandomGenerator` before load, e.g. by reseeding and stepping) and dump hidden prototype vertices. It is not required, because the XML writer plus the §3.5 probe cover most needs.

---

## 9. Risks for exact parity

1. **libm.**
   - `rotatePointAboutLine` and all `makeRotationMatrix` overloads call double `sincos(theta)` and narrow to float (core/src/global.cpp:214-247, 599-713). Heliades must do `np.float32(math.sin(float(theta)))`, not `np.sin(float32)`.
   - The leaf lattice uses `atanf`, `sincosf`, `powf`, `sqrtf`, `asinf` and `atan2f` (glibc float). numpy's float32 ufuncs are not guaranteed to equal glibc's, so call libm through `ctypes` (`libm.sinf`, …) or prove equality by exhaustive tests.
   - Double `atan` (AS lattice dtheta), double `wave phase*M_PI`, and double `sqrt`/`log` in randn.
   - The tier is ulp, but differences compound through incremental transforms.
2. **Float/double mixing.**
   - `fmin(1.f, 0.5+0.5*age/10.f)` is evaluated in double (AS:4103-4107);
   - girth `*1e-4`, `0.5*(r-ri)` (PA:7656-7663);
   - `randu(0.f, 2.f*M_PI)` bound = `float(2π)`;
   - `compoundLeafRotation` `0.5*PI_F` in double;
   - `interpolateTube` `1.f+1e-3`;
   - `deg2rad` uses `float(M_PI)`.

   Port every literal with its C++ type.
3. **FMA / contraction.** No FMA instructions were found in the oracle `libhelios.so` (analyst disassembly). A rebuild with `-march=native` could change this, so pin the oracle build flags in `PORTING.md`.
4. **Path dependence.** Leaves and petioles are moved by an incremental `translate → scale → translate` every sub-step, and peduncle radii by `setTubeRadii`. Heliades must apply the same sequence of float32 4×4 matrix updates. Zero-shift and zero-angle early returns must match. Vectorising across leaves is fine; fusing steps is not.
5. **Argument evaluation order.** `make_AxisRotation(0, randu(), randu())` draws right to left with GCC (oracle-verified). Other multi-draw expressions (unverified; grep `randu(.*randu(`) would differ under Clang or MSVC. The selfTest notes platform-dependent seeds (tests/selfTest.cpp:202-204).
6. **Hidden prototypes and the shared cache.**
   - 35 blade prototypes plus 3 inflorescence prototypes are hidden objects that occupy UUIDs and object IDs.
   - A second plant in the same `PlantArchitecture` creates none of them and makes no private draws.
7. **UUID order in Helios queries.** `Context::primitives` and `objects` are `std::unordered_map` (core/include/Context.h:3548, 3622), so `getAllUUIDs()` returns bucket order. Compare by sorted UUID. No plugin code on the tomato path iterates them (only the collision paths at PA:1620, 1664, 13268-13275).
8. **Draw-count traps.**
   - `LeafPrototype::operator=` redraws the identifier while copies do not;
   - wasted draws (`longitudinal_curvature`, the extra `leaf.pitch` and `prototype_scale` resamples);
   - `randu(0,0)` makes no draw;
   - the bud-break randu happens before the creation function kills the bud;
   - the `base_yaw.val()` first-time draw on the snapshot;
   - the `randn` spare shared between consecutive calls and reset on seed.

   Each of these must be reproduced literally.
9. **Comparisons with constants** decide branches: `randu() > 0.25`, `< 0.2`, `< 0.8`, `tc >= 40`, `age+dt > 30`, `fabs(frac - fplus) < 0.0001`, `area > 0`. A 1-ulp difference upstream can flip topology, and from then on the plants diverge.
10. **Static state.**
    - `static class_map_cache` (PA:722) is unused by tomato;
    - the `static` collision counters (PA:1688-1693) are unused;
    - `Context` `norm_distribution` spare.

    No wall-clock dependence after seeding. No OpenMP in the plugin.
11. **Texture alpha.** `TomatoLeaf_blade.png` is a palette PNG with transparency, read by libpng in Helios. The solid fraction and the area > 0 test depend on matching the alpha decode (W1-A).
12. **Stale or misleading Helios comments** (H:1585 and others) contradict the code. Port from the code, not the comments (notes/W1-D.md §4).
