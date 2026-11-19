# Species file format (`heliades-species/1`)

A species is a **data file** on top of one of the 32 ported Helios library models: the *template* supplies the shoot
types, the parameter values, the prototypes, the creation / callback functions and the builder; the file overrides
what differs, says where each value comes from, and adds rules and assets by name. No code goes in a species file.

Files live in `heliades/species/defs/<name>.yaml` (or a directory on `HELIADES_SPECIES_PATH`). The file name is the
species name. `PlantArchitecture.loadPlantModelFromLibrary("<name>")` registers the file on first use, so
`buildPlantInstanceFromLibrary`, `buildPlantCanopyFromLibrary`, the batch builder
(`heliades.plantarchitecture.batch`) and fast mode (`Scene(mode="fast")`) work with it unchanged.
`getAvailablePlantModels()` lists the Helios library until a species is loaded (it stays the Helios list for parity);
`heliades.species.available_species()` lists the species files.

Reference files: `cucumber_from_tomato.yaml` (Helios's cucumber re-expressed from the tomato, 0 ulp against Helios),
`eggplant.yaml` (tomato template, generated leaf, procedural flower and fruit, an extra shoot type),
`barley.yaml` (wheat template, template builder, build-parameter default).

## Why YAML

Species files are written by people and, later, by an agent, and read in review. YAML allows comments (the reasoning
behind a judged value belongs next to it), flow mappings keep one value per line
(`max_nodes: {value: 125, source: judged}`), and it is a superset of JSON, so a tool can emit JSON and it loads.
Only `yaml.safe_load` is used: no tags, no code. Watch one YAML trap: inside `{...}` a comma ends the value, so
quote any note that contains one (`note: "first flower, then fruit"`). The loader reports such a split as an
unknown key.

## Top level

```yaml
format: heliades-species/1          # required
name: eggplant                      # required; = the file name; not a Helios model name
template: tomato                    # required; a library model (see getAvailablePlantModels())
metadata: {...}                     # required
sources: {...}                      # required (may be {})
leaf_textures: {...}                # generated leaf images
shoot_types: {...}                  # overrides per shoot type
drop_shoot_types: [mainstem]        # template shoot types to remove
build: {...}                        # the plant builder
traits: [...]                       # expected trait ranges (plausibility harness)
```

Unknown keys are errors everywhere. Every error names the file and the key path, e.g.
`eggplant.yaml: shoot_types.mainstem.shoot.max_nodes.value: 0 is outside the valid range [1, 10000]`.

### `metadata`

`latin_name`, `common_name` (required), `cultivar`, `description`, `plant_type` (`herbaceous` / `tree` / `weed`;
default the template's), `leaf_inclination_beta` (`[mu, nu]` to enable Helios's leaf-angle tracking, `null` for none;
default the template's).

### `sources` and the `source` of every value

```yaml
sources:
  kahlen_chen_2015:
    citation: "Kahlen K, Chen T-W (2015) ... full reference"    # required, >= 10 characters
    doi: 10.xxxx/...                                           # optional; also url, note
```

Every value a file sets carries `source:` — a key of `sources`, or **`judged`** (set by judgement, with a `note:` if
the reasoning is not obvious). There is no third option: a value is either traceable to a reference or honestly
marked as a judgement. Do not invent references; an honest `judged` is better. Values the file does not set are the
template's and need no source.

## `shoot_types`

```yaml
shoot_types:
  main:                       # the label of the shoot type in the species
    from: mainstem            # copied from: a shoot type defined earlier in this file, else the template's
                              # (default: the same label)
    child_shoot_types: {value: {branch: 1.0}, source: judged}   # label -> probability (sum 1); {} = none
    shoot:          {<field>: <value>, ...}     # ShootParameters
    internode:      {...}                       # PhytomerParameters.internode
    petiole:        {...}
    leaf:           {...}
    peduncle:       {...}
    inflorescence:  {...}
    leaf_prototype: {...}                       # PhytomerParameters.leaf.prototype (LeafPrototype)
    flower: <organ>                             # flower prototype (see Organs)
    fruit:  <organ>
    phytomer_creation: template | none | [<rule>, ...]
    phytomer_callback: template | none | [<rule>, ...]
```

Template shoot types not listed stay as they are, unless listed in `drop_shoot_types`. A shoot type listed under its
own template label is modified in place (`mainstem:` with no `from`). Child shoot types must exist in the result.

### Values

| form | for | meaning |
|---|---|---|
| `{value: x, source: s}` | every field | a fixed value (`RandomParameter = x`) |
| `{uniform: [a, b], source: s}` | float, int fields | `uniformDistribution(a, b)` |
| `{normal: [mean, sd], source: s}` | float fields | `normalDistribution(mean, sd)` |
| `{weibull: [shape, scale], source: s}` | float fields | `weibullDistribution(shape, scale)` |
| `{discrete: [v1, v2, ...], source: s}` | int fields | `discreteValues(...)` |
| `{texture: <name>, source: s}` | `leaf_texture_file` | a generated leaf texture of this file |
| `{texture: {<index>: <name>, ...}, source: s}` | `leaf_texture_file` | one generated texture per compound-leaf index (`-n..n`, 0 = tip; the pea's tendrils / leaflets / stipules) |

Every form takes an optional `note:`. Applying an override is the C++ statement on the copied struct (`operator=`,
`*Distribution`): no random draws. A distribution on a parameter that has no generator yet (`leaflet_scale`,
`intercalary_leaflet_scale`) gets the Context generator.

### Fields (units: m, degrees, days; curvatures in degrees per metre)

Kinds: **float** / **int** = `RandomParameter` (value or distribution); **uint**, **real**, **bool**, **rgb**
(three numbers in [0, 1]), **vec3**, **str** (file name) = plain members; **texture_map** = file name or
`{index: file}` (or `texture:`); **length_map** = number or `{index: number}`. The range is a typo guard.

`shoot` — float: `girth_area_factor` [0,100], `insertion_angle_tip`, `base_roll`, `base_yaw` [-360,360],
`insertion_angle_decay_rate`, `internode_length_max`, `internode_length_min` [0,5], `internode_length_decay_rate`,
`gravitropic_curvature` [-5000,5000], `tortuosity` [0,1000], `tortuosity_persistence_length`, `phyllochron_min`
(0,365], `elongation_rate_max` [0,10], `leaf_expansion_rate_max` [-1,10] (-1 = unset),
`vegetative_bud_break_probability_min` / `_max` [0,1], `vegetative_bud_break_probability_decay_rate`,
`flower_bud_break_probability`, `fruit_set_probability` [0,1], `vegetative_bud_break_time` [0,3650];
int: `max_nodes` [1,10000], `max_nodes_per_season`, `max_terminal_floral_buds` [0,100];
bool: `flowers_require_dormancy`, `growth_requires_dormancy`, `determinate_shoot_growth`.

`internode` — float: `pitch`, `phyllotactic_angle`, `radius_initial`; int: `max_vegetative_buds_per_petiole`,
`max_floral_buds_per_petiole`; rgb `color`; str `image_texture`, `image_class_map_file`; uint `length_segments`,
`radial_subdivisions`.

`petiole` — uint `petioles_per_internode`; float `pitch`, `radius`, `length`, `curvature`, `taper` [0,1],
`flexibility`, `flexibility_aging`; rgb `color`; uint `length_segments`, `radial_subdivisions`.

`leaf` — int `leaves_per_petiole` [1,50]; float `pitch`, `yaw`, `roll`, `leaflet_offset` [0,1], `leaflet_scale`,
`intercalary_leaflet_scale` [0,2], `prototype_scale` (the blade length in m).

`peduncle` — float `length`, `radius`, `pitch`, `roll`, `yaw`, `curvature`; rgb `color`; uint `length_segments`,
`radial_subdivisions`.

`inflorescence` — int `flowers_per_peduncle`; float `flower_offset`, `pitch`, `roll`, `flower_prototype_scale`,
`fruit_prototype_scale` (organ size in m, see Organs), `fruit_gravity_factor_fraction` [0,1],
`inflorescence_maturity_period` (-1 = the plant's fruit-maturity threshold); str `fruit_class_map_file`;
uint `unique_prototypes`.

`leaf_prototype` — float `leaf_aspect_ratio` (blade width / length), `midrib_fold_fraction`, `longitudinal_curvature`,
`longitudinal_curvature_exponent`, `lateral_curvature`, `roll_fraction`, `petiole_roll`, `wave_period`,
`wave_amplitude`, `flexibility`, `flexibility_taper`, `flexibility_aging`, `flexibility_aging_max`,
`leaf_buckle_length`, `leaf_buckle_angle`; vec3 `leaf_offset`; uint `subdivisions`, `unique_prototypes`; bool
`build_petiolule`; length_map `petiolule_length`; real `petiolule_radius_fraction`; texture_map `leaf_texture_file`,
`leaf_class_map_file`; str `OBJ_model_file`.

The full table with ranges is `heliades/species/fields.py`.

## Organs (`flower:` / `fruit:`)

A prototype is built at the origin pointing along +x from its base, 1 long (fruit) or 1 across (flower); the
inflorescence scales it by `flower_prototype_scale` / `fruit_prototype_scale`, so those are the organ sizes in m.

| form | |
|---|---|
| `template` (default) | keep the template's |
| `none` | no geometry |
| `{library: CucumberFruitPrototype}` | a ported Helios prototype function by name |
| `{obj: WheatSpike.obj, height: 0}` | one Helios OBJ asset (`height` 0 keeps its size) |
| `{obj_variants: "assets/obj/Cucumber/fruit_{index:02d}.obj", count: 6}` | one of n OBJs, `randu(0, n-1)`; flowers need `{state}` (open / closed) |
| `{ellipsoid: {diameter, bulge, color, calyx_color, calyx_fraction, rings, segments}}` | procedural fruit (diameter relative to the length; `bulge` > 0 widens towards the tip) |
| `{corolla: {petals, petal_width, cup, color, center_color, center_diameter, center_length, bud_length}}` | procedural flower; closed = petals folded into a bud |
| `{panicle: {whorls, whorl_top, branches_base, branches_tip, branch_length, branch_length_tip, branch_angle, branch_angle_tip, droop, spikelets_per_branch, spikelet_zone, spikelet_length, spikelet_width, spikelet_hang, pedicel_length, rachis_radius, branch_radius, rachis_bend, color, rachis_color, jitter, seed}}` | procedural open grass panicle (fruit; S1, oat): a nodding rachis with whorls of drooping branches and pendant spikelets on pedicels; lengths are fractions of the panicle length, `seed` drives a local numpy generator (no Context draw); see `PANICLE_KEYS` in `organs.py` |
| `{capitulum: {disc_color, disc_center_color, disc_center_fraction, dome, thickness, back_color, rays, ray_length, ray_width, ray_angle, ray_color, bracts, bract_length, bract_color, bud_fraction, rings, segments}}` | procedural composite head (flower or fruit): disc 1 across facing +x on a green receptacle, involucral bracts, ray florets beyond the disc (`ray_length` in disc diameters, `ray_angle` > 0 forward); closed = a bract-covered bud `bud_fraction` across. The scale is the disc diameter (sunflower) |

Each takes `source:` / `note:`. Procedural organs draw nothing from the Context stream.

## Rules (`phytomer_creation`, `phytomer_callback`, `build.rules`)

A list, run in order: `- {rule: <name>, source: s, note: ..., <parameters>}`. A single `library_function` rule is the
library hook itself. `template` keeps the template's function, `none` removes it.

| rule | kind | parameters | |
|---|---|---|---|
| `kill_buds` | creation | `which` (floral / vegetative / both), `node_below`, `node_from`, `rank`, `rank_above` | kill the new phytomer's axillary buds when every given condition holds |
| `leaf_scale_ramp` | creation | `intercept`, `slope`, `days` | seedling leaves: scale by `min(1, intercept + slope * age / days)` |
| `internode_scale_ramp` | creation | `intercept`, `slope`, `days` | seedling internodes, the same ramp |
| `leaf_bump_map` | creation | `texture`, `strength` | bump map on the new leaves |
| `library_function` | creation, callback | `name` | a ported Helios creation / callback function (`TomatoPhytomerCreationFunction`, `WheatPhytomerCreationFunction`, ...) |
| `harvest_ripe_fruit` | callback | | pick full-size fruit (bud dead) |
| `high_wire_lowering` | callback | | lower and lean the main stem at the wire (with `high_wire_string`) |
| `high_wire_string` | build | `view_half_angle_deg`, `look_ahead_distance`, `attraction_weight` | attraction points up a string to a 3.8 m wire |

The tomato's creation function is `kill_buds {which: floral, node_below: 8, rank: 0}`,
`kill_buds {which: both, rank_above: 1}`, `leaf_scale_ramp {0.5, 0.5, 10}`, `internode_scale_ramp {0.7, 0.3, 10}`
(tested bit-identical). `heliades/species/rules.py` has the schemas; `ADDING.md` says how to add a rule.

## `build`

```yaml
build:
  builder: template | generic        # default template
  # both builders
  phenology: {<key>: {value: dd, source: s}, ...}   # dormancy_break, flower_initiation, flower_opening, fruit_set,
                                                     # fruit_maturity, dormancy, max_leaf_lifespan, evergreen (bool)
  max_age: {value: days, source: s}
  rules: [...]                                       # build rules, after the plant is built
  build_parameter_defaults: {tiller_count: {value: 3, source: s}}   # a template build parameter's default
  # generic only (all required except base_offset / base_rotation)
  source: s
  base_offset: [0, 0, -0.025]
  base_rotation: {pitch: 0, yaw: {randu: [0, 2pi]}, roll: {randu: [0, 2pi]}}    # radians; '<k>pi' = float(k) * pi
  base_stem: {shoot_type, nodes, internode_radius, internode_length_max (m or shoot_type),
              internode_length_scale_factor_fraction, leaf_scale_factor_fraction, radius_taper}
```

`template` calls the template's `build<Template>Plant` (so it needs the template's shoot-type labels), then applies
`phenology` (only the keys given), `max_age` and the rules. `generic` does what the library builders do:
`addPlantInstance(base + offset)`, the shoot type's `internode_length_max.val()` if asked, the base rotation (random
components drawn **right to left**, roll then yaw then pitch, as GCC evaluates `make_AxisRotation(randu(), ...)`),
`addBaseStemShoot`, `breakPlantDormancy`, `setPlantPhenologicalThresholds` (all six thresholds required;
`max_leaf_lifespan` 0 = none), `max_age`, the rules. The species inherits the template's build-parameter declarations
(wheat's `tiller_count`, grapevine's trellis, ...); `build_parameter_defaults` fills one that the caller did not pass.

## `leaf_textures`

```yaml
leaf_textures:
  eggplant_leaf:              # -> heliades/species/assets/eggplant/eggplant_leaf.png
    source: judged
    aspect: 0.62              # width / length; set leaf_aspect_ratio to the same
    widest_at: 0.42           # outline: see heliades/species/leaf_texture.py
    base: 0.5
    tip: 1.25
    lobes: {count: 4, depth: 0.10, sharpness: 0.8}
    color: [0.30, 0.40, 0.22]
    veins: 6
```

Keys: `length_px`, `aspect`, `widest_at`, `plateau`, `base`, `tip`, `asymmetry`, `lobes {count, depth, sharpness}`,
`serration {count, depth}`, `sinus {depth, width}` (cordate basal notch; the midrib stays solid across it), `profile [[t, p], ...]` (replaces the formula), `color`, `midrib_color`, `midrib_width`,
`vein_color`, `veins`, `vein_style` (pinnate / parallel), `vein_angle`, `vein_width`, `margin_shade`. The image is an
RGBA PNG, alpha 255 inside the blade and 0 outside, `u` along the leaf. Generate / check with
`python -m heliades.species.leaf_texture [--check] [species ...]`; the committed PNGs are tested to equal the tool's
output. Use it in `leaf_prototype: {leaf_texture_file: {texture: eggplant_leaf, source: ...}}`.

## `traits`

```yaml
traits:
  - {trait: plant_height, age: 60, range: [0.30, 0.90], source: judged, note: ..., seeds: 8}
```

Traits: `plant_height` (m), `leaf_count`, `leaf_area` (m^2), `main_stem_nodes`, `internode_length_mean` (main stem,
m), `leaf_length_max` (longest blade / leaflet, m), `fruit_count`, `flower_count`, `fruit_length_max` (m),
`shoot_count`, and the timing traits `first_flower_day` / `first_fruit_day` (`age` is the horizon; the plant grows
in one-day steps). `python -m heliades.species.traits <species>` prints measured (mean over seeds 1..n, fast mode)
against expected; `tests/species/test_traits.py` asserts every mean is in range. `slow: true` marks a trait whose
check is expensive (old, large plants): its test is `pytest.mark.slow` (run with `--runslow`); the traits command
always measures it.

## What the format cannot express yet

* New growth logic beyond the rule registry (a rule is Python in `heliades/species/rules.py`, reviewed and tested).
* Shoot-type-specific build logic of the template builders other than through `builder: template`.
* Leaf texture series (age-staged textures), class maps for generated leaves.
* Organ shapes other than OBJ assets, ellipsoids, corollas, open panicles and capitula (no pods, spikes with awns).
* Temperature-driven phenology: all thresholds are in degree-days at the model's one per day.
