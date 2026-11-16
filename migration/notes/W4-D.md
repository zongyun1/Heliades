# W4-D: grapevine_VSP

Baseline: Helios 1.3.98. Abbreviations as in SPEC: PA = PlantArchitecture.cpp, PL = PlantLibrary.cpp,
AS = Assets.cpp.

## Diff against the tomato path (written before porting, prioritised)

These are the code paths the grapevine reaches that the tomato does not. P1 is needed for age 0, P2 for
growth before the clusters set (about day 79), and P3 for the gallery age of 120 days.

| prio | path | C++ | lines | status |
|---|---|---|---|---|
| P1 | build-parameter registry, `validateBuildParameters`, `BuildParameterDistribution`, `getBuildParameterValue`, sampling in `buildPlantInstanceFromLibrary` | PL:182-397, 538-629, 667-696 | ~300 | ported |
| P1 | `initializeGrapevineVSPShoots` (5 shoot types) | PL:3567-3785 | 219 | ported |
| P1 | `buildGrapevineVSP` | PL:3787-4030 | 244 | ported |
| P1 | `BoundedWander`, `grapevineTrunkPath` | PL:24-58, 132-181 | 85 | ported |
| P1 | `addShootFromNodePositions` / `prescribedShoot` (prescribed trunk and canes) | PA:7286-7465 | 180 | ported (axis-continuation variant raises) |
| P1 | `Shoot::appendPhytomerFromNodePositions`, plus the restore-measured-internode branch of `appendPhytomerInternal` | PA:2011-2015, 2187-2216 | 35 | ported |
| P1 | Phytomer from a prescribed internode | PA:2758-2770, 2993-3030 | 50 | ported |
| P1 | textured internode tubes (`GrapeBark.jpg`, uv wrapping every 0.25 m) | PA:3243-3293 | 50 | ported (class maps raise) |
| P1 | per-plant attraction points: `setPlantAttractionPoints`, `calculateAttractionPointDirection`, `detectAttractionPointsInCone` | PA:13556-13582, 1809-1831, 1929-2005 | ~130 | ported (global attraction points still raise) |
| P1 | `Phytomer::removeLeaf`, `removeShootLeaves` | PA:6765-6803, 9998-10012 | 55 | ported (detached-objects variant not ported) |
| P1 | `GrapevinePhytomerCreationFunction` | AS:1943-1975 | 33 | ported |
| P2 | attraction blend in the segment loop; the forced Context update (`collision_avoidance_applied && force_update_on_collision`) | PA:3162-3189, 12312-12326 | 45 | ported |
| P2 | phytomer callback dispatch; `GrapevinePhytomerCallbackFunction` (fruit-zone leaf pulling at a shoot age of 45 days) | PA:11935-11938, AS:1977-2005 | 33 | ported |
| P3 | `GrapeClusterParameters` | AS:1760-1772 | 13 | ported |
| P3 | `GrapevineFruitPrototype` (berry seeding, packing, `addSphere`) | AS:1774-1935 | 162 | **hook bound, raises** |
| P3 | inflorescence geometry: fruit prototypes, peduncles, `sampleOrganBuildParameters` used per prototype | W3-D | - | **W3-D** (`sampleOrganBuildParameters` is ported in `build_parameters.py`) |

These paths are not reached before day 120:
- seasonal dormancy (break at 165 days, dormancy period 365);
- leaf lifespan (`max_leaf_lifespan` defaults to 1e6);
- epicormic shoots;
- `max_age` (365).

## What is ported

| file | contents |
|---|---|
| `library/build_parameters.py` (new) | declarations, `getBuildParameterValue`/`IntegerValue`, `BuildParameterDistribution` (constant/uniform/normal/discrete, `sample`), `registerPlantModelBuildParameters`, `getPlantModelBuildParameterNames`, `validateBuildParameters`, `prepare_build_parameters` (PL:667-679), `sampleOrganBuildParameters` |
| `library/grapevine.py` (new) | everything in the grapevine rows of the table above; the hooks `GrapevineFruitPrototype`, `GrapevinePhytomerCreationFunction` and `GrapevinePhytomerCallbackFunction` |

### Shared-file hunks (each one fills a branch that used to raise or was absent)

- `library/__init__.py`: appended 4 lines that register the grapevine shoot initializer and builder.
- `architecture.py`:
  - `initializePlantModelRegistrations` gains 2 lines (`register_build_parameters`).
  - `buildPlantInstanceFromLibrary`: the `raise NotImplementedError("build parameters…")` is replaced by `prepare_build_parameters`. With no parameters this is the same as before (empty maps, no draws).
- `phytomer.py`:
  - new module functions `detectAttractionPointsInCone` and `calculateAttractionPointDirection`;
  - in the constructor:
    - the prescribed-internode raise is removed and the prescribed overrides are added (PA:2762);
    - the attraction raise is split: global attraction, collision and solid obstacles still raise, and per-plant attraction is computed;
    - the prescribed axis block is added, and the segment loop is skipped for prescribed internodes;
    - the attraction blend is added inside the segment loop;
    - the textured-tube raise is replaced by the uv computation, `add_tube_object_textured` and `append_tube_segment_textured`;
  - new method `removeLeaf`.
- `shoot_min.py`:
  - `PlantInstance` gets the attraction members;
  - `appendPhytomerInternal` passes `prescribed_internode` through, instead of raising, and gains the restore branch;
  - new `Shoot.appendPhytomerFromNodePositions`;
  - new module functions `PrescribedInternode`, `addShootFromNodePositions`, `removeShootLeaves` and `setPlantAttractionPoints`;
  - `init_plant_state` gains `collision_avoidance_applied` and `force_update_on_collision`.
- `growth.py`:
  - the phytomer-callback raise is replaced by the dispatch;
  - the end-of-sub-step `updateShootNodes(False)` becomes `(True)` when attraction was applied, as in C++, then the flag is reset.

None of these hunks changes the tomato path, because every new branch is gated on a feature the tomato does not use. The tomato plantarch tests pass (102 passed).

## Parity (`cases_plantarch_w4d.py`, `ops_plantarch_w4d.py`): all pass at 0 ulp

| case | seeds | checks |
|---|---|---|
| `plantarch.grape.load` | 1..8 | engine calls after the ctor, the load (12), a second load, an age-0 build (609 for seed 1) and an age-0 build with fixed build parameters |
| `plantarch.grape.params` | 1..2 | all 5 ShootParameters trees as PyHelios serialises them (33k bytes) |
| `plantarch.grape.build_parameters` | 1 | accepted names; rejection of: an unknown key, `vine_spacing` 6, `trunk_height` 0.01, `berry_diameter` 0.5 (organ, validated at build), `cluster_berry_count` 5, NaN; acceptance of the fixed set |
| `plantarch.grape.age0` | 1..8 | `snapshot(include_hidden=True)`, which covers:<br>- the textured trunk (uv, solid fraction);<br>- the canes;<br>- the 10 hidden grape-leaf prototypes;<br>- every object transform |
| `plantarch.grape.age0_build_parameters` | 1..4 | the same, with `vine_spacing` 1.8, `trunk_height` 0.6 and the organ keys |
| `plantarch.grape.steps` | 1..8 | ages 2..40 in 2-day calls. After each call:<br>- the probe;<br>- per shoot, the node count and the stored nodes and radii;<br>- the whole hidden-inclusive snapshot as a stream.<br>The shoots break on day 30 and the catch wires steer them from then on (attraction is active on 337 of 436 phytomers by day 60). |
| `plantarch.grape.age76` | 1..8 | direct build at 76 days: probe, shoot summary and full snapshot (55 658 primitives for seed 1). The fruit-zone leaves have been pulled and no cluster has set. |
| `plantarch.grape.age120` | 1..8 | **gated**: returns `[-1]` on both sides until `heliades.plantarchitecture.inflorescence` exists *and* `grapevine.FRUIT_PROTOTYPE_IS_STUB` is False |

The first cluster sets on day 79-81 in daily steps on the oracle (seeds 1..8), so every age up to 76 is vegetative.

Timing: Heliades takes about 0.6 s at age 0, about 9 s for the 2..40 sweep and about 19 s at age 76, against 0.3 s for the oracle.

Mutation checks (`tests/heliades/test_plantarch_w4d.py::test_mutations_are_caught`):

| mutation | caught by |
|---|---|
| `addComponent` randu evaluated left to right | `age0` |
| prescribed radius constant instead of interpolated | `age0` |
| attraction disabled | `steps` |
| cordon buds not woken | `steps` |
| leaf pulling disabled | `age76` |

## Findings and discrepancies

1. **GCC right-to-left evaluation** of `addComponent(randu(amp), randu(wl), randu(phase))` holds (the phase is drawn first). The age-0 snapshot proves it, and the mutation check confirms it.
2. **Prescribed shoots still build petioles and leaves.** The leaf-prototype cache (10 grape blades, built through the W3-A `GenericLeafPrototype` with `leaf_offset`) and the draws for every leaflet happen. Then `removeShootLeaves` deletes them. The petiole vertices are kept, and their axes decide `grapevine_shoot` versus `_mirrored`.
3. Two ported branches are faithful but not observable, so no mutation check covers them:
   - **The forced Context push.** Attraction forces a Context push every sub-step (`force_update_on_collision` defaults to true). Dropping the push changes no compared value, because the final `updateShootNodes(True)` rewrites the tube.
   - **The bark uv wrap.** Dropping the `floor` that wraps the bark uv leaves the primitive UVs unchanged as the core reports them: their values stay in [0, 1] either way.
4. `detectAttractionPointsInCone`:
   - `half_angle * M_PI / 180.0f` is evaluated in double;
   - `distance <= 1e-6` is compared in double;
   - `std::acos(float)` is glibc `acosf`.
5. `removeLeaf` does not reduce `downstream_leaf_area`, which is commented out in C++. Girth after leaf pulling therefore still counts the pulled leaves. This is faithful.
6. PyHelios 1.3.98 passes build parameters as fixed values only. So `BuildParameterDistribution::sample` (uniform/normal/discrete) is ported but **not oracle-checked**. Its draws use fresh `helios::` distributions, like the C++.
7. The unit check on a constant build parameter confirms that it is taken without a draw.

## Remaining work

- **`GrapevineFruitPrototype`** (AS:1774-1935):
  - Berry count, branch skeleton, berry seeding (`make_vec3(randn, randn, randn)`, which is right to left), 400 packing iterations.
  - The O(n²) overlap pass can be vectorised exactly: for each k, accumulate `-P[i,k]` for i ascending, then `+P[k,j]` for j ascending, through `np.where` to keep signed zeros.
  - `Context::addSphere` differs from `add_sphere_object`: the radius is inside `sphere2cart`, and the top cap is wound (v2, v1, v0). A local helper is needed.
  - Open question: whether the unqualified `sin`, `cos` and `std::pow(float, int)` resolve to the float or the double versions in that TU. Settle it with a compile probe (`typeid` of `sin(float)` with `global.h` included) before trusting the geometry.
  - When it is done, set `FRUIT_PROTOTYPE_IS_STUB = False`.
- **W3-D inflorescence**: with no flower prototype, the bud goes straight to FRUITING. This needs:
  - fruit prototypes built `unique_prototypes = 5` times through the hook, with `sampleOrganBuildParameters` per prototype;
  - the peduncle (yaw 180, pitch 50-90);
  - `fruit_gravity_factor_fraction` 0.7.

  After that, run `plantarch.grape.age120` (it un-gates by itself).
- **`plant.py` XML writer** for prescribed shoots and build parameters (owned by W3-D). This is why the grape cases compare the snapshot and the shoot summary instead of the XML.
- Not ported:
  - `appendShootFromNodePositions` (axis continuation);
  - global attraction points;
  - internode class maps;
  - `removeLeaf(detached)`.
- `grapevine_Wye` shares `grapevineTrunkPath`, `BoundedWander`, the prescribed shoots and attraction; only its shoot types and builder are missing.

## Requests for core (not done)

- `Context::addSphere` (primitive version, not the object version).
- `expf`, `cbrtf`, `powf`, `atan2f` in `core/vec.py`. `expf` is loaded locally in `grapevine.py`.
