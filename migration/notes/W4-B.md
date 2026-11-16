# W4-B: bean and soybean

Baseline: Helios 1.3.98. Scope: `initializeBeanShoots` / `buildBeanPlant` (PL:2911-3069), `initializeSoybeanShoots` /
`buildSoybeanPlant` (PL:5745-5897), their Assets.cpp functions (AS:1578-1637, 4031-4066), and every shared branch
they reach that the tomato did not.

## Code paths the legumes reach and the tomato does not (written before porting)

Status: **[new]** absent from Heliades and ported here; **[first]** already ported (W3-B/C) but never oracle-checked,
now covered by these cases; **[W3-D]** inflorescence geometry, gated.

| # | path | C++ | status |
|---|---|---|---|
| 1 | `PlantArchitecture::appendShoot`: parent growth cap + `max_nodes` = node count, `terminateApicalBud`, base on the parent tip, `is_axis_continuation`, inherited `internode_radius_cap` | PA:7178-7237 | **[new]** `shoot_min.appendShoot` + `architecture.PlantArchitecture.appendShoot` |
| 2 | axis continuation in `Shoot::updateShootNodes` (zero cached offset), child of node 0 of the unifoliate shoot | PA:2429-2547 | [first] |
| 3 | `growth_state.max_nodes_growth_cap >= 0` in `getGrowthNodeCap` (the unifoliate stops at 1 node) | PA:2598 | [first] |
| 4 | two petioles per internode (unifoliate node): the `petiole > 0` bud rotation `2*pi*p/npet`, per-petiole buds and child-type samples, two leaf caches of 1 leaflet | PA:3300-3600 | [first] |
| 5 | `leaves_per_petiole == 1`: `orientLeaf` single-leaf roll sign and the blade elevation correction (`atan2f`, petiole/leaf length ratio, clamp) | PA:5016-5197 | [first] |
| 6 | petiole too small to build (bean unifoliate: 0.1 mm long, `makePetioleTube` returns `no_petiole_objID`), then **lazily created** in `setPetioleAndLeafScaleFraction` when it has grown | PA:4465-4477, 12879 | [first] |
| 7 | per-leaflet textures `{-1: right, 0: tip, 1: left}` (bean), one automatic material per texture | AS:640-924 | [first] (W3-A edge case 8 with tomato) |
| 8 | blade shapes outside the tomato's: lateral curvature -1 / +0.45, no wave, 8 subdivisions, positive longitudinal curvature | AS:517-924 | [first] |
| 9 | `leaf.pitch` normal distribution (bean: Context `randn`), distributed `leaf.roll` (soybean) | PA:3510-3530 | [first] |
| 10 | leaflet offsets 0.3 / 0.5 (`(fabs(ind) - 0.5f) * offset * length` in double, W3-B finding 2) | PA:3555 | [first] |
| 11 | petiole taper 0 (bean), 0.25 (soybean); petiole radius 0.4 mm | PA:3406 | [first] |
| 12 | base rotation with a non-zero pitch (3 `randu`, right to left): the `shoot_base_rotation.pitch != 0` branch of the first phytomer and of `childShootOutwardAxis` | PL:3058, PA:2900-2955, 2366 | [first] |
| 13 | no tortuosity (gravitropism only, OU drive 0, but the 2 `randn` per segment are still drawn) | PA:3000-3080 | [first] |
| 14 | null `phytomer_creation_function` (both unifoliates) | PA:2203 | [first] |
| 15 | `BeanPhytomerCreationFunction` / `SoybeanPhytomerCreationFunction` (rank- and node-based bud kills, double `fmin`) | AS:1616, 4048 | **[new]** library |
| 16 | `RandomParameter_int` uniform `max_nodes` (4-7, 16-19, 3-7), distributed `insertion_angle_tip` / `flower_bud_break_probability` | | [first] |
| 17 | `vegetative_bud_break_probability_decay_rate == 0` (soybean) and `< 0` with `min = 1` (bean unifoliate) | PA:7065 | [first] |
| 18 | `determinate_shoot_growth = false` (soybean) | PA:11970 | [first] |
| 19 | a child shoot type different from the parent's (soybean `trifoliate -> branch`, both `unifoliate -> trifoliate`), three shoot types in the snapshot map | PA:7013 | [first] |
| 20 | `max_floral_buds_per_petiole = 0` (unifoliate) | PA:2666 | [first] |
| 21 | `BeanFlowerPrototype`, `BeanFruitPrototype`, `SoybeanFlowerPrototype`, `SoybeanFruitPrototype` (OBJ, the 2-argument `loadOBJ` = RGB::blue default colour for the bean, RGB::black for the soybean) | AS:1599-1614, 4031-4046 | **[new]** library, **[W3-D]** gated |
| 22 | distributed inflorescence parameters (`flowers_per_peduncle` 1-4, pitch, roll, fruit scale, gravity factor), peduncle of 1 segment | | **[W3-D]** |
| 23 | `BeanLeafPrototype_unifoliate_OBJ` / `_trifoliate_OBJ` (OBJ leaves) | AS:1578-1597 | **[new]**, but no library entry uses them (the bean uses `GenericLeafPrototype` blades); not oracle-tested |

`cowpea` (out of scope) is the bean's structure with other values: the same two shoot types, `appendShoot`, the
same creation-function shape and `GenericLeafPrototype` blades. `library.bean.build_legume` and `appendShoot` would serve
it unchanged. It also declares a leaf-inclination Beta (1.398, 1.574), so `buildPlantInstanceFromLibrary` reaches the
leaf-inclination tracking, which still raises `NotImplementedError` (not ported by anyone yet).
