# W3-A: leaf blade geometry

Baseline: Helios 1.3.98. Scope (narrowed by the coordinator once W3-B landed): the blade geometry itself,
`GenericLeafPrototype` with its petiolule, `buildLeafBladeLattice` and `deformLeafLattice` (AS:24-282, 517-924).
The Phytomer side (buildLeafPrototype, labelLeafPrototype, assignPetioluleMaterial, recordLeafPrototypeRestGeometry,
getLeafShape, nameOrganMaterials and the private-stream cache build) is W3-B's, in `phytomer.py`; I had ported
these too and removed my copies after merging main (differences below).

## What is ported (`heliades/plantarchitecture/leaf_prototype.py`)

| symbol | C++ |
|---|---|
| `GenericLeafPrototype(scene, prototype, compound_leaf_index) -> objID`, bound with `@hooks.GenericLeafPrototype.implement` at import | AS:640-924 |
| `buildLeafBladeLattice(shape)` | AS:517-638 |
| `deformLeafLattice(...)` (rigid no-op guard, errors, moment iteration) | AS:37-282 |
| `petioluleVertexCount`, `petioluleRadialSubdivisions` | AS:24-35 |
| `resolveTextureFile` (called by GenericLeafPrototype) | PA:669-711 |

The hook draws only from the prototype's RandomParameters, in Helios order (first `val()`s: aspect ratio for Ny,
roll fraction for validation, then the `LeafShapeValues` fill; trailing resamples: aspect, fold, curvature,
exponent, lateral, roll, petiole roll, [shape recorded], wave period, wave amplitude), sets
`last_built_shape(_valid)`, labels the petiolule triangles "petiolule", and returns the polymesh with its face
set and 180-degree vertex normals.

Not ported (raise `NotImplementedError`): `validateColorTexture` for texture series and `recordSpectralClassMap`
for leaf class maps (both off for tomato). The OBJ-file leaf goes through `Scene.load_obj`, untested.
`bendPetioleCenterline` (AS:284-515) is petiole bending, not in this package. `deformLeafLattice` is ported and
unit-tested (rigid identity, droop, no stretching, errors) but not oracle-tested: no tomato path reaches it.

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

Both sides run `buildPlantInstanceFromLibrary(pos, 0)` (W3-B's real Phytomer on Heliades, with this hook bound;
the op refuses a stub build).

| case | seeds | checks |
|---|---|---|
| `plantarch.tomato.age0_full` (acceptance) | Context 1..8 | the whole age-0 tomato from `snapshot(include_hidden=True)`: all 6412 primitives and 44 objects (tubes, 35 hidden prototypes, 7 leaflet copies): UUIDs, vertices, normals, areas, colors, UVs, solid fractions, textures, hidden flags, parents, members, centers, transforms |
| `plantarch.tomato.age0_full_attrs` | Context 1..8 | for every polymesh object (prototypes and copies): per primitive material label / color / texture / two-sided flag, `object_label` and data labels; object data labels; polymesh vertices, normals, UVs, faces, face UUIDs, normal source |
| `plantarch.leafproto.tomato` / `tomato_attrs` | Context 1..8 | the same, restricted to the hidden prototypes |
| `plantarch.leafproto.edges`, `edges_attrs` | 10 specs | pure fold; fold + exponent < 1; roll only; petiole roll ±; 1 subdivision, odd Ny, no petiolule; per-leaflet petiolule lengths + leaf offset; every shape parameter distributed (private-stream val/resample order); per-leaflet textures (`<base>`, `<base>_1`, ... naming); intercalary leaflets with zero offset |
| `plantarch.leafproto.fuzz` (12), `fuzz_attrs` (6) | random | curvature (distributed), exponent, lateral, roll (sometimes distributed), fold, petiole roll, wave (amplitude sometimes distributed), subdivisions 1-10, aspect ratio (sometimes distributed), unique prototypes 1-5, leaflets 1/3/5/7, petiolule on/off + length, leaf offset, petiole radius (distributed) and taper, prototype scale |

Prototype selection: at age 0 the hidden objects of the Context are exactly the blade prototypes (objects 3-37 for
tomato, 152 UUIDs each of which 144 survive: 40 blade facets, 104 petiolule facets; the inflorescence prototypes
only come with the first peduncle), so the prototype-level cases select them by the hidden flag.

Redefinition: Helios goes through the PyHelios JSON round trip; Heliades rebuilds each listed RandomParameter
with `initialize(gen)` like `jsonToRandomParameterFloat`, and sets scalar fields directly.

Mutation checks (`tests/heliades/test_plantarch_leafproto.py::test_mutations_are_caught`, by re-executing the
module source with one edit), all caught: the eight float/double/libm choices above, dropping the trailing
`longitudinal_curvature.resample()` (tomato), moving `wave_amplitude.resample()` first (edges 7), keeping
transparent facets (tomato), not labelling the petiolule (attrs), a halved petiolule radius (age0_full).

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

## Differences between my removed Phytomer-side copies and W3-B's (kept W3-B's)

- `recordLeafPrototypeRestGeometry`: W3-B raises `NotImplementedError` for a flexible leaf; mine ported the
  flexible branch (polymesh vertices minus the frame origin, petiolule vertices split off, Ny recovered from the
  vertex count). Needed before any flexible species; the code is in this branch's history (commit "W3-A WIP").
- `buildLeafPrototype`: W3-B's defaults for `leaflet_size_own` etc. are `0.0`, the C++ ones `-1.f`; equivalent,
  since every use tests `> 0`.
- `OrganMaterial` in W3-B has no normal-map fields; the core material table has none, so identical in effect.
- Otherwise the same code path; W3-B's versions pass every case here, including the per-leaflet texture naming.

## Requests for core (not done)

- `Scene.translate_primitive` returns early on a zero shift; Helios's `translatePrimitive(vector, shift)` always
  multiplies. Identical up to the sign of zero entries (not observable); left as is.
- Normal-map fields on `Material` are not modelled; `nameOrganMaterials` treats them as the defaults.

## For later packages

- `GenericLeafPrototype` is all the Phytomer needs; it is bound when `leaf_prototype` is imported.
- `deformLeafLattice(rest_lattice, Nx, Ny, scale, mature_scale, flexibility, taper, gravity)` is ready for flexible
  leaves once `recordLeafPrototypeRestGeometry` records the lattice.
