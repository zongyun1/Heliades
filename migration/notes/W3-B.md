# W3-B: Phytomer geometry (age-0 tomato)

Baseline: Helios 1.3.98. Scope from SPEC §8: the `Phytomer` constructor, the tube helpers, leaf orientation and
placement, the tomato creation function, and the minimal plant-building path to age 0. Growth is W3-C; the
leaflet blades themselves are W3-A.

## What is ported

| file | contents | C++ |
|---|---|---|
| `plantarchitecture/tube_utils.py` | `interpolate_tube` (vec3 and float), `get_axis_vector`, `make_petiole_tube`, `clamp_offset`, `leaflet_size_max` (libm `powf`), `petiole_radius_fraction`, `compound_leaf_rotation`, `distribution_mean`, tube constants | PA:29-40, 266-378, 1307-1327, 4597-4617, 4844-4853, 12879-12938 |
| `plantarchitecture/phytomer.py` | `Phytomer.__init__` line for line (internode axis, base roll/pitch/yaw, gravitropism + OU tortuosity + azimuthal restoring force, internode tube create/append, petiole axis, curvature, taper, radii, `makePetioleTube`, buds, `createPetioleFloralBuds`, leaf cache build, leaflet placement); `orientLeaf` (procedural, sessile and prescribed branches); `buildLeafPrototype`, `labelLeafPrototype`, `assignPetioluleMaterial`; `getCachedLeafPrototypeObjID`, `recordLeafPrototypeRestGeometry` (rigid leaves); `getLeafShape`; `scaleLeafPrototypeScale`, `scaleInternodeMaxLength`; bud state setters; `getLeafArea`, `getLeafBladeArea`; `nameOrganMaterials`; internode/petiole axis getters | PA:2694-3601, 1239-1394, 1449-1547, 4349-4358, 4532-4595, 4904-5197, 5416-5480, 5846-5874, 6225-6250, 181-264, 8799-8810 |
| `plantarchitecture/shoot_min.py` | `Shoot` (ctor, `buildShootPhytomers`, `appendPhytomer(Internal)`, `sampleChildShootType`, `sampleVegetativeBudBreak`, `breakDormancy`, `propagateDownstreamLeafArea`), `PlantInstance`, `addPlantInstance`, `addBaseStemShoot`, `validateShootTypes`, `breakPlantDormancy`, `setPlantPhenologicalThresholds`, `init_plant_state` (PlantArchitecture members of this path) | PA:2007-2262, 2550, 6951-7102, 7146-7173, 7512, 10069, 11410, 11669 |
| `plantarchitecture/library/tomato.py` | `buildTomatoPlant`, `TomatoPhytomerCreationFunction` (bound to `hooks.TomatoPhytomerCreationFunction`) | PL:6205-6224, AS:4092-4108 |
| `plantarchitecture/architecture.py` | entry points `addPlantInstance`, `addBaseStemShoot`, `breakPlantDormancy`, `setPlantPhenologicalThresholds`, `invalidateOrganIndex`, `buildPlantInstanceFromLibrary` (age 0; `age > 0` raises "W3-C"); the tomato builder is registered in `initializePlantModelRegistrations` (`library/__init__.py` untouched) | PL:653-760 |

`ops_plantarch_load.age0_draw_plan` (the W2-C stand-in) is gone: `plantarch_age0_observables` and
`plantarch_build` on Heliades now build the plant and read the observables from the real objects. All six
`plantarch.load.*` cases still pass, including `redefined_distributions` (normal, weibull, int uniform a==b,
discretevalues).

Not ported (raise `NotImplementedError`): prescribed internodes (`addShootFromNodePositions`), textured
internodes, collision / attraction points / solid obstacles, carbohydrate model, optional object data, leaf
texture stages, flexible petioles (`bendPetioleUnderLeafWeight` past its rigid guard), flexible-leaf rest
geometry, `unique_prototypes == 0`, `updateInflorescence` (W3-D), `setInternodeLengthScaleFraction` (W3-C).

## Leaf-placement interface expected from W3-A

The only thing the Phytomer needs from W3-A is the **`GenericLeafPrototype` hook**, which the parameter layer
already stores as `LeafPrototype.prototype_function`:

```python
from heliades.plantarchitecture import hooks

@hooks.GenericLeafPrototype.implement
def GenericLeafPrototype(scene, prototype: LeafPrototype, compound_leaf_index: int) -> int:  # AS:640
    ...  # build the blade (+ petiolule labelled "petiolule"), return scene.add_polymesh_object(uuids)
```

Contract:
- It draws only from `prototype`'s `RandomParameter`s. During the cache build their generator is the private
  `MinstdRand0(identifier)`, which `Phytomer._build_leaf_prototype_cache` installs and restores. It keeps
  Helios's order, including the trailing resamples (AS:867-878).
- It sets `prototype.last_built_shape` / `last_built_shape_valid` (use `_raw_set` or plain assignment; both
  are value types).
- It returns a polymesh object. Its frame is the translation to the first vertex (`addPolymeshObject`).
- It is bound when `heliades.plantarchitecture.leaf_prototype` is imported. `phytomer._resolve_prototype_function`
  imports that module lazily. If the hook is still unbound, it falls back to `stub_generic_leaf_prototype` and
  sets `pa.leaf_geometry_is_stub = True`.

Everything around the hook is already here, in the C++ order of PA:3460-3497, for each of the 5×7 blades:
`buildLeafPrototype` (petiolule radius fraction from the shoot type's means), `labelLeafPrototype` (blade →
"leaf", petiolule label kept), `assignPetioluleMaterial`, the cache append, `recordLeafPrototypeRestGeometry`,
`nameOrganMaterials(..._leaf)` and `hideObject`. If W3-A ports any of these too, keep one copy: the ones here
are oracle-checked only for tubes and leaflet frames.

Placement (`Phytomer._place_leaflet`, PA:3499-3570) runs per leaflet:
1. `randu(0, 4)`, then `copyObject(cache[p][l])`;
2. `scaleObject(lsf·leafletSizeMax)`;
3. sample roll, pitch and yaw;
4. `orientLeaf`: `rotateObject` "x" roll, "y" −pitch, "z" yaw (if ≠ 0), "z" azimuth + compound;
5. `translateObject(leaf_base)`.

Then `TomatoPhytomerCreationFunction` calls `scaleObjectAboutPoint(0.5, leaf_base)`. The `age0_leaf_frames` case
verifies this chain at 0 ulp by giving the stub blades the oracle prototypes' frames.

## Parity cases (`cases_plantarch_phytomer.py`, `ops_plantarch_phytomer.py`). All pass.

| case | tier | seeds (Context) | checks | max ulp |
|---|---|---|---|---|
| `plantarch.tomato.age0_draws` | exact | 1..8 | engine calls after load, after the age-0 build, and their difference (26/27) | 0 |
| `plantarch.tomato.age0_params` | exact | 1..8 | 64 XML values per plant (`%g` as `ostream<<float`): base rotation, bud states, internode length/radius/pitch/phyllotaxy/length_max/perturbations, petiole length/radius/pitch/curvature/scale factors/taper/axes, leaflet scale, 7×(leaf scale, pitch, yaw, roll, prototype), floral bud. Then 77 full-precision values through the PyHelios getters: internode nodes, 6 petiole nodes, radii, 7 leaflet bases, petiole/internode axis vectors at 0, 0.5, 1, petiole length | 0 |
| `plantarch.tomato.age0_tubes` | ulp, max 0 | 1..8 | all 84 primitives labelled `shoot`/`petiole`: UUID, label, 9 vertex floats, color | 0 |
| `plantarch.tomato.age0_tube_materials` | exact | 1..2 | material label of each tube primitive (`tomato_mainstem_stem` / `_petiole`) | — |
| `plantarch.tomato.age0_leaf_frames` | ulp, max 0 | 1..8 | object ID and 4×4 transform of the 7 leaflet copies (stub blades in oracle frames) | 0 |

Unit tests (`tests/heliades/test_plantarch_phytomer.py`) cover the helpers, the object layout (35 hidden
blades 3..37, leaflets 38..44), and a constructor draw count of exactly 23 engine calls (§3.3).

Mutation checks (`test_mutations_are_caught`). All are caught:
- drop the unused `petiole.curvature.resample()`: caught by `age0_draws`;
- skip `breakDormancy`'s bud draw: caught by `age0_draws`;
- petiole radius `frac*(radius*taper)` instead of `(frac*radius)*taper`: caught by `age0_tubes`;
- `fplus = 1` instead of `float(1.f+1e-3)` in `interpolateTube`: caught by `age0_leaf_frames` and `age0_params`;
- `scaleLeafPrototypeScale` about the origin instead of the leaf base: caught by `age0_leaf_frames`;
- `orientLeaf` pitch before roll: caught by `age0_leaf_frames`.

## Discrepancies / findings

1. **SPEC §5.1 sketch vs 1.3.98 code** (no draw-count change):
   - The constructor also resamples `leaf.roll` and, for laterals, `leaf.yaw` per leaflet (constant for tomato).
   - `orientLeaf` takes the sampled angles as arguments.
   - The lateral roll is `(roll − elevation)·compound/|compound|`. The terminal roll is a separate
     `terminal_roll_angle` (0).
   - `leafletSizeMax` has an intercalary branch.
   - `compoundLeafRotation` takes the clamped leaflet offset.
   - `petioleRadiusFraction = max(lsf, 0.3)` is confirmed: r0 = 0.0006 in the XML. SPEC had it "unverified".
2. **`fabs` promotion in the leaflet offset** (PA:3555, `(fabs(ind) − 0.5f)·offset·length`). It is ported as C
   `fabs` (double product, narrowed once). For tomato's constants (0.15, 0.2) the float and double forms give
   identical bits, so the oracle cannot tell them apart. This is **unverified**. Re-check with a species whose
   offset or length differs.
3. **Shoot ctor quirk** (PA:6958): `gravitropic_curvature = shoot_params.gravitropic_curvature.val()` reads the
   *argument* (the local copy in `addBaseStemShoot`), not the member. A distributed value would be drawn and
   cached on that temporary. It is ported literally.
4. `TomatoPhytomerCreationFunction`'s `fmin(1.f, 0.5 + 0.5*age/10.f)` is evaluated in double. At age 0 this is
   exact either way; growth (W3-C) will exercise it.
5. `distribution_mean` uses Python's double `gamma` for `std::tgamma(float)` (weibull only, off the tomato path).
6. `nameOrganMaterials` is ported, and the tube material labels match the oracle. The core material table has
   no normal-map fields, so those members of the "same appearance" test are assumed equal.
7. `load.identifier` now reads `getLeafShape` through the real rest-geometry record. The stub prototype function
   reproduces the private-stream draws, so the case is unaffected by the missing blades.

## Requests for core (not done; `heliades/core` is owned by others)

- `powf` and `atan2f` via ctypes in `core/vec.py`, next to `sinf`/`cosf`/`asinf`. They are loaded locally in
  `tube_utils.py` and `phytomer.py` for now.
- No blocking gap: tubes (`add_tube_object`, `append_tube_segment`), object transforms, `hide_object`, primitive
  data and materials were sufficient.

## What W3-C (growth) needs

- `Phytomer.setInternodeLengthScaleFraction` raises; port it with `Shoot.updateShootNodes`,
  `setPetioleAndLeafScaleFraction`, `setPetioleBase` and girth. `scaleInternodeMaxLength` already calls it
  when the fraction reaches 1.
- The phytomer-index > 0 branch is ported but not yet oracle-checked. It covers −1.25·pitch, the OU tortuosity
  with 2 `randn`, gravitropic, transverse and azimuthal rotations, the petiole phyllotactic rotation, the
  azimuth accumulation and `appendTubeSegment`. The first 2-day step exercises all of it: use the draw probe
  there first.
- `appendPhytomerToShoot` (PA:7523) is not ported: `current_node_number++`, `nodes_this_season++`,
  `shoot_index.y` refresh, terminal floral bud. `Shoot.appendPhytomer` itself is ready.
- `addChildShoot`, `advanceTime`, `terminateApicalBud`, floral state machine. `Phytomer.setFloralBudState`
  handles DORMANT/ACTIVE/DEAD; other states need `updateInflorescence` (W3-D).
- Extend `shoot_min.Shoot` / `PlantInstance` rather than replacing them. `init_plant_state` holds the
  PlantArchitecture members of this path.
- `getLeafArea` (girth) depends on real blade areas, so girth parity needs W3-A merged.
- The creation function receives `plant_instances[pid].current_age`. Keep that member current per sub-step.
