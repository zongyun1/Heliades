# W3-A: leaf blade prototypes

Baseline: Helios 1.3.98. Scope from SPEC §8: `GenericLeafPrototype`, `buildLeafBladeLattice`, the petiolule,
`deformLeafLattice`, `buildLeafPrototype`, `labelLeafPrototype`, `assignPetioluleMaterial` and the prototype cache
on the private stream (AS:24-282, 517-924; PA:181-276, 3449-3497, 4532-4596, 4844-5014).

## What is ported (`heliades/plantarchitecture/leaf_prototype.py`)

| symbol | C++ |
|---|---|
| `GenericLeafPrototype(scene, prototype, compound_leaf_index)` (bound to `hooks.GenericLeafPrototype`, lazily) | AS:640-924 |
| `buildLeafBladeLattice(shape)` | AS:517-638 |
| `deformLeafLattice(...)` (rigid no-op guard, errors, moment iteration) | AS:37-282 |
| `petioluleVertexCount`, `petioluleRadialSubdivisions` | AS:24-35 |
| `resolveTextureFile` | PA:669-711 |
| `nameOrganMaterials(pa, objID(s), base)` + `OrganMaterial` | PA:181-264 |
| `getCachedLeafPrototypeObjID`, `recordLeafPrototypeRestGeometry` + `LeafRestGeometry` | PA:4532-4596 |
| `leafletSizeMax`, `_clamp_offset`, `_distribution_mean` (file-static helpers) | PA:266-323, 4844 |
| `PhytomerLeafMixin.buildLeafPrototype / labelLeafPrototype / assignPetioluleMaterial` | PA:4904-5014 |
| `PhytomerLeafMixin.buildLeafPrototypeCache(leaves_per_petiole, leaf_flexibility)` (the constructor block) | PA:3449-3497 |

PlantArchitecture state owned here, created on first use: `unique_leaf_prototype_objIDs`,
`unique_leaf_prototype_rest_geometry` (both keyed `(identifier, leaves_per_petiole)`), `organ_material_labels`
(iterated sorted, as the `std::map`).

Not ported (raise `NotImplementedError`): the OBJ-file leaf goes through `Scene.load_obj` but is untested;
`validateColorTexture` for texture series and `recordSpectralClassMap` for leaf class maps (both off for tomato).
`bendPetioleCenterline` (AS:284-515) is petiole bending, not in this package.

## Float/double choices (each proven by a mutation that the parity cases catch)

| expression (AS) | evaluated as | caught by |
|---|---|---|
| `dtheta -= atan(ce*lc*powf(x,ce-1)*dx)` | double `atan`, `dtheta` narrowed | tomato (all seeds) |
| `wave_phase = (x + wp*float(j>=Ny/2)) * M_PI / wp` | double, narrowed | tomato |
| `z_wave = 2.f*fabs(y)*wa*sinf(phase)` | double product chain (`fabs` is the C double one) | tomato |
| `cosf/sinf(0.5f*local_fold*M_PI)` (pure fold) | argument in double, narrowed | edges 0,1 |
| `fmin(0.1f, pr*powf(..)*exp(-70.f*x)) - 0.01*pr/fabs(pr)` | double `fmin`, `fabs`, `exp` | edges 7 (`fmin`/`fabs` term; a float `exp` alone is not distinguishable) |
| `atanf`, `cosf`, `sinf` (cross-section, wave) | glibc via ctypes, not correctly rounded | atanf: fuzz 3,5; cosf: edges 2,7; sinf: edges 7, fuzz 0,1,2,4,5,10 |
| `powf` | glibc via ctypes | not distinguishable by the cases (few calls); glibc and the correctly rounded value differ in ~0.05% of random arguments, so ctypes is kept |

So the unsuffixed math calls are the C `double` functions: no `using namespace std` reaches Assets.cpp and
libstdc++'s `<cmath>` does not put the float overloads in the global namespace. `std::atan2`, `std::remainder`,
`std::hypot` and `std::fabs` in `deformLeafLattice` are explicitly `std::`, i.e. the float overloads.

## Parity cases (`cases_plantarch_leafproto.py`, `ops_plantarch_leafproto.py`). All 0 ulp.

| case | seeds | checks |
|---|---|---|
| `plantarch.leafproto.tomato` | Context 1..8 | the 35 hidden prototypes: UUIDs, object IDs, vertices, normals, areas, colors, UVs, solid fractions, textures, hidden flags, parents, members, centers, transforms |
| `plantarch.leafproto.tomato_attrs` | Context 1..8 | per primitive: material label, color, texture, two-sided flag, `object_label` and data labels; per object: object data labels, polymesh vertices / normals / UVs / faces / face UUIDs / normal source |
| `plantarch.leafproto.edges`, `edges_attrs` | 10 specs | pure fold; fold + exponent < 1; roll only; petiole roll ±; 1 subdivision, odd Ny, no petiolule; per-leaflet petiolule lengths + leaf offset; every shape parameter distributed (private-stream val/resample order); per-leaflet textures (`<base>`, `<base>_1`, ... naming); intercalary leaflets with zero offset |
| `plantarch.leafproto.fuzz` (12), `fuzz_attrs` (6) | random | curvature (distributed), exponent, lateral, roll (sometimes distributed), fold, petiole roll, wave (amplitude sometimes distributed), subdivisions 1-10, aspect ratio (sometimes distributed), unique prototypes 1-5, leaflets 1/3/5/7, petiolule on/off + length, leaf offset, petiole radius (distributed) and taper, prototype scale |

How the oracle exposes the prototypes: `buildPlantInstanceFromLibrary(pos, 0)` builds the internode tube
(object 1, UUIDs 0-13), the petiole tube (object 2, UUIDs 14-83), then the 35 hidden blade prototypes
(objects 3-37, 152 UUIDs each of which 144 survive: 40 blade facets and 104 petiolule facets) and the 7 copies.
At age 0 the hidden objects of the Context are exactly the blade prototypes (the inflorescence prototypes come
with the first peduncle), so the cases select them by the hidden flag from `snapshot(include_hidden=True)`.

Heliades side, until W3-B: the identifier comes from `loadPlantModelFromLibrary`; the Context state before the
cache is reproduced with stand-in internode (14 triangles) and petiole (70 triangles) tubes named with
`nameOrganMaterials` (`tomato_mainstem_stem` / `tomato_mainstem_petiole`), then `buildLeafPrototypeCache` on a
phytomer stand-in (`ops_plantarch_leafproto._PhytomerStandIn`). The stand-in tubes are not compared; they only
fix the UUID/object footprint and the petiole material the petiolules take. W3-B should replace the stand-in
with the real Phytomer and keep the cases.

Redefinition: Helios goes through the PyHelios JSON round trip; Heliades rebuilds each listed RandomParameter
with `initialize(gen)` like `jsonToRandomParameterFloat`, and sets scalar fields directly.

Mutation checks (`tests/heliades/test_plantarch_leafproto.py::test_mutations_are_caught`, by re-executing the
module source with one edit), all caught: the seven float/double/libm choices above, dropping the trailing
`longitudinal_curvature.resample()` (tomato), moving `wave_amplitude.resample()` first (edges 7), not hiding the
prototypes, keeping transparent facets, the petiolule keeping its automatic material (attrs), not recording the
named organ material (attrs: `_1`.. labels), and a petiolule radius that ignores the petiole taper.

## Findings / discrepancies

1. SPEC §5.3 is out of date in places, the code wins:
   - the longitudinal term uses `powf(x, exponent - 1)` with the exponent clamped by `fmaxf(1, ...)`, not `powf(x,3)`;
   - the petiolule is 57 vertices / 104 triangles built in the blade's face set, with the radius
     `petiolule_radius_fraction` = petiole radius at the attachment point / mature leaflet size, both from the
     shoot type's distribution means (no draws);
   - all 48 blade facets consume UUIDs; 40 survive per tomato blade (8 fully transparent are deleted).
2. Material IDs: Helios numbers the named tomato materials stem 1, petiole 2, leaf 3 (it has no visible default
   material entry in that count). Labels match; IDs are not compared.
3. Draw counts: the cache makes no Context draws (checked); per blade the private stream gets one `val()` (first
   blade only, lazy) and one `resample()` for each distributed shape parameter, in the order `aspect_ratio` (Ny),
   `roll_fraction` (validation), then the `LeafShapeValues` fill order, then the trailing resamples
   aspect, fold, curvature, exponent, lateral, roll, petiole roll, [shape recorded], wave period, wave amplitude.
4. `resolveTextureFile`: Heliades follows the C++ order (absolute, `tryResolveFilePath`, plugin asset,
   `assets/textures/`). When neither cwd nor `$HELIOS_BUILD` holds `plugins/`, it falls back to the pinned
   `third_party/pyhelios_src/helios-core` found by walking up from the package. Textures are compared by basename.

## Requests for core (not done)

- `Scene.translate_primitive` returns early on a zero shift; Helios's `translatePrimitive(vector, shift)` always
  multiplies. Identical up to the sign of zero entries (not observable); left as is.
- Normal-map fields on `Material` are not modelled; `nameOrganMaterials` treats them as the defaults.

## What W3-B needs (placing a leaf)

- Make `Phytomer` inherit `PhytomerLeafMixin` and give it `context_ptr`, `plantarchitecture_ptr`,
  `phytomer_parameters`, `parent_shoot_ptr.shoot_type_label`, `petiole_objIDs`, `plantID`; give
  `PlantArchitecture` `plant_instances[plantID].plant_name` (the build name, `"tomato"`).
- In the constructor, after the petiole tube is built and named: `self.buildLeafPrototypeCache(leaves_per_petiole,
  leaf_flexibility)` (PA:3452-3497; it builds only on a cache miss and returns whether it did).
- Per leaflet (PA:3499-3512): `p = ctx.randu(0, unique_prototypes - 1)`;
  `oid = ctx.copy_object(getCachedLeafPrototypeObjID(pa, identifier, leaves_per_petiole, p, leaf, "PlantArchitecture::Phytomer"))`;
  then `nameOrganMaterials(pa, oid, f"{plant_name}_{shoot_type_label}_leaf")`, scale/orient/translate.
  With `unique_prototypes == 0`: `self.buildLeafPrototype(leaves_per_petiole, ind_from_tip)` (+ `recordBuiltLeafShape`,
  W3-B).
- `nameOrganMaterials(pa, objID, base)` is also what the internode (`_stem`) and petiole (`_petiole`) tubes need.
- `pa.unique_leaf_prototype_rest_geometry[key][p][l]` holds the rest shape (`.shape`, `.lattice_frame_offset`,
  and the lattice when the leaf is flexible) for `deformLeafLattice`.
