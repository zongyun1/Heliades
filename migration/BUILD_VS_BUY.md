# Port, rebuild, or use a library: per-module decisions

Baseline Helios 1.3.98. Line counts are C++ source excluding tests and third-party code.
Every library listed is MIT, BSD or Apache, so all are compatible with LGPL-2.1. GPL packages
(pysolar, phydrus) are excluded.

## Deciding rule

1. **On the plant-generation path: port faithfully, bit-exact.** This covers RNG, geometry, compound
   objects, growth rules and organ prototypes. The same seed must give the same plant, because
   datasets, the calibration loop and reviewers compare against Helios. A library may be used here
   only if its operation sequence provably matches (e.g. PNG decoding).
2. **Downstream physics and measurement: re-implement the model, swap the engine.** Radiation,
   LiDAR and collision are defined by their physics and observables, not by their OptiX or Vulkan
   code. Port the model logic onto Warp and check parity statistically or within a tolerance.
3. **Small scientific formula modules: port them.** Photosynthesis, stomata and similar modules are
   200–3000 lines of equations with species coefficients. Porting is cheap and keeps the
   coefficients identical. Where a library exists (e.g. pvlib), use it only as an independent
   cross-check.
4. **Infrastructure with a mature library: use the library and do not port.** This covers
   optimization, visualization, GUI, generic mesh/point-cloud processing, file formats and math
   utilities.

## Module table

| Helios module | lines | decision | how | parity tier |
|---|---|---|---|---|
| core: RNG, primitives, objects, data | ~30k of 44k | **port** | `heliades/core/*` (in progress) | exact / ulp |
| core: file IO (OBJ, PLY, XML) | ~6k | **port the semantics, parse with libraries** | parse with `trimesh` (`process=False`, no vertex merging) or our own small OBJ reader. Face, UV and material order must match Helios's loader so UUIDs line up. XML via `xml.etree` | ulp |
| core: math utilities (fzero, interp1, etc.) | ~2k | **port** where results feed generation or parity-tested models; otherwise `scipy` | — | ulp / tol |
| core: textures and alpha masks | — | **library** `Pillow`. PNG decoding is lossless and identical; JPEG is used only for color | `solid_fraction` from the alpha mask must match | exact (PNG) |
| **plantarchitecture** | 43k | **port (the core of the project)** | species library, phytomer rules, carbohydrate and nitrogen models | ulp (same seed = same plant) |
| weberpenntree | 1.8k | port (low priority) | — | ulp |
| canopygenerator | 8k | port the parts still used; most is superseded by plantarchitecture | — | ulp |
| solarposition | 2k + Prague sky 1k | **port** (Spencer sun position, Gueymard broadband, cloud calibration). Prague Sky Model is Apache-2.0: port or wrap | cross-check sun position against `pvlib` | tol |
| photosynthesis (FvCB + empirical) | 2.9k | **port** | vectorize over primitives in numpy/Warp | tol |
| stomatalconductance (BWB, BBL, MOPT, BMF, Buckley) | 2.2k | **port** | — | tol |
| boundarylayerconductance | 1.3k | **port** (itself a port of an LGPL package) | — | tol |
| energybalance | 2.9k | **port the model**; the per-primitive solve becomes a Warp kernel | — | tol |
| planthydraulics | 1.6k | port | — | tol |
| leafoptics (PROSPECT-PRO/D) | 1.9k | **port**. The `prosail` package is from 2019, has no PROSPECT-PRO, and its license is unclear | spectral data files taken from Helios assets | tol |
| **radiation** | 42k (of which ~10k OptiX/Vulkan backends) | **rebuild on Warp**: port `RadiationModel` semantics (bands, sources, scattering iterations, emission, per-primitive absorbed flux) onto `wp.Mesh` ray queries. Backends are not ported | photoreal synthetic images optionally via **Mitsuba 3** (BSD, spectral) | statistical (Monte Carlo) |
| radiation camera | 3.5k | rebuild on Warp (Helios-equivalent camera); Mitsuba for photoreal | — | statistical |
| syntheticannotation | 1.1k | rebuild on our camera (organ masks, boxes from per-pixel UUID/object IDs) | — | exact (labels) |
| collisiondetection | 11.7k | **library**: Warp `wp.Mesh` / `wp.HashGrid` BVH queries | port only the query semantics that plantarchitecture's collision avoidance uses | tol |
| voxelintersection | 2k | library: Warp kernels | — | tol |
| lidar (synthetic scan + processing) | 16.7k | **rebuild scan** on Warp ray casting (port scan patterns and multi-return logic). **Library** for processing: `open3d`, `scipy` (triangulation, voxel gap fraction); `laspy` for LAS IO | — | statistical |
| aeriallidar | 4k | same as lidar | — | statistical |
| parameteroptimization (8 algorithms) | 4.7k | **library, not ported**: `scipy.optimize`, `cma`, `optuna`, `nevergrad` | — | n/a |
| soiltransport (Richards eq.) | 13.4k | **defer**. No permissive Python library exists (phydrus is GPL and wraps the HYDRUS executable). Port later if needed, possibly as a Warp solver | — | tol |
| visualizer | 15.3k | **replace**: USD export plus `polyscope` / usdview / Isaac Sim | — | n/a |
| projectbuilder (GUI) | 36.5k | **drop** | — | n/a |

## Effort implied

- **Ported faithfully:** about 95k lines of C++ (core, plantarchitecture, generators, physiology,
  solar). At roughly 0.4–0.5 Python lines per C++ line, that is about 40–45k lines of Python.
- **Rebuilt on Warp around ported model semantics:** radiation, camera, annotation and LiDAR scan,
  about 10–15k lines of new code instead of about 60k ported.
- **Replaced by libraries or dropped:** optimization, collision, voxel, LiDAR processing, visualizer
  and GUI, about 85k lines that we do not write.

## New capabilities (not in Helios), and what they are built on

| capability | build on |
|---|---|
| Deformable plant dynamics (rods + leaf shells, tree direct solver, inverse statics) | **our own**: no package targets plants. Warp kernels |
| Robot interaction / rigid contact | **Newton** (Apache-2.0, Warp-based) or **MuJoCo** (Apache-2.0); plants are coupled as deformable bodies |
| Photoreal rendering, 3DGS export | **Mitsuba 3** / Isaac Sim; Gaussian export from the mesh |
| Mesh utilities (remeshing, decimation for sim meshes) | **trimesh** (MIT) |
| Calibration and agent search | **cma**, **optuna**, `scipy` |
| Interchange | **OpenUSD** (`usd-core`) |
