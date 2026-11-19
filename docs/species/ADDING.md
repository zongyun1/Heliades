# How to add a species

A new species is one YAML file (`heliades/species/defs/<name>.yaml`) plus, if it has its own leaf image, the PNG the
leaf tool generates from that file. No Python unless the species needs a growth rule the registry does not have.
`FORMAT.md` is the reference; `eggplant.yaml` and `barley.yaml` are worked examples, `cucumber_from_tomato.yaml`
shows a file that reproduces a Helios model exactly.

## 1. Pick the template

The library model whose *architecture* is closest, not the closest relative by name: the template brings shoot types,
branching logic, creation functions and the builder.

| the species is like | template | why |
|---|---|---|
| sympodial Solanaceae, bushy fruiting herbs | `tomato`, `capsicum`, `cherrytomato` | trusses / cymes, seedling ramps, determinate shoots |
| vine trained on a string | `cucumber` (or `tomato` + rules, as `cucumber_from_tomato`) | one stem, attraction points, high-wire rules |
| trailing / sprawling vine, large simple leaves | `cucumber` with the high-wire rules removed, or `bindweed` | |
| small grain cereal | `wheat`, `rice` | culm with elongating upper internodes, tillers, terminal ear |
| tall C4 grass | `sorghum`, `maize` | |
| legume with trifoliate leaves | `bean`, `soybean`, `cowpea` | unifoliate + trifoliate shoot types |
| rosette | `butterlettuce`, `sugarbeet`, `strawberry` | |
| fruit tree | `apple`, `almond`, `olive`, `walnut`, `pistachio` | trunk / scaffold / spur shoot types, pruning, seasons |

Look at the template's shoot initializer (`heliades/plantarchitecture/library/<template>.py`) to see its shoot-type
labels and what each value is, and dump its parameters:

```python
from heliades.core.scene import Scene
from heliades.plantarchitecture import PlantArchitecture
from heliades.plantarchitecture.structure import shoot_parameters_to_dict
pa = PlantArchitecture(Scene()); pa.loadPlantModelFromLibrary("wheat")
print(sorted(pa.shoot_types)); print(shoot_parameters_to_dict(pa.shoot_types["mainstem"]))
```

## 2. Write the file

Start from a copy of `eggplant.yaml` (tomato-like) or `barley.yaml` (grass). Then:

1. `metadata`: latin and common name, cultivar / type, and what "age 0" means for this model (emergence,
   transplant).
2. `sources`: only references you can cite exactly. Everything else is `source: judged` with a `note:` that says
   why. Never invent a citation; reviewers will check them.
3. `shoot_types`: override what differs from the template, section by section. Keep the template's labels if you use
   `builder: template`. Use `from:` to derive a new shoot type (an earlier one of the file or a template one) and
   `child_shoot_types` to wire it in; `drop_shoot_types` to remove template ones.
4. Leaves: either reuse a Helios texture (`leaf_texture_file: {value: CapsicumLeaf.png, ...}`, LGPL Helios asset) or
   describe the outline under `leaf_textures` and use `{texture: <name>}`. Set `leaf_aspect_ratio` to the outline's
   `aspect`. `prototype_scale` is the blade length in m.
5. Flowers / fruit: `library:` a Helios prototype, `obj:` / `obj_variants:` a Helios OBJ asset, or a procedural
   `ellipsoid` / `corolla`. The `*_prototype_scale` values are the organ sizes in m.
6. `phytomer_creation` / `phytomer_callback`: `template`, or a list of rules from the registry (`FORMAT.md`).
7. `build`: `builder: template` and override `phenology` / `max_age`; or `builder: generic` with the base stem,
   rotation and all six phenological thresholds.
8. `traits`: at least six expected ranges with sources (height at two ages, leaf length, a node / leaf / shoot count,
   first flower or fruit day, fruit size or count). The ranges are about the real crop, not about the model.

Do not put commas inside unquoted notes of `{...}` mappings (YAML splits them).

## 3. Generate assets and validate

```bash
.venv/bin/python -m heliades.species.leaf_texture <name>          # writes heliades/species/assets/<name>/*.png
.venv/bin/python -c "from heliades.species import validate; print(validate('<name>').summary())"
```

`validate` parses the file, registers it, loads it and builds an age-0 plant: every error names the key path.

## 4. Check plausibility and look at it

```bash
.venv/bin/python -m heliades.species.traits <name>                 # measured vs expected, mean over seeds
.venv/bin/python scripts/species/gallery.py <name> --compare <template>   # outputs/species/<name>.png
```

Tune values until the traits are in range **and** the gallery looks like the plant. Typical levers: height =
`max_nodes` x `internode_length_max` (and the base stem); flowering date = `phyllochron_min` x the first flowering node
(`kill_buds node_below`) + `phenology.flower_initiation` + `flower_opening`; fruit load =
`flower_bud_break_probability` x `fruit_set_probability` x `flowers_per_peduncle`; bushiness =
`vegetative_bud_break_probability_min`, `child_shoot_types`, `kill_buds rank_above`; a `determinate_shoot_growth`
shoot stops when its first flower opens.

## 5. Test and commit

`tests/species/test_traits.py`, `test_loader.py::test_species_files_validate` and
`test_organs_textures.py::test_committed_textures_are_regenerated_exactly` (the PNGs equal the tool's output) pick up
every file in `defs/` automatically. Run `pytest -q tests/species`. Commit the YAML, the PNGs and, if useful, the gallery
image. Add a row to the trait table of your work-package note.

## Adding a rule

Only when no combination of existing rules does it. In `heliades/species/rules.py`:

```python
@rule("kill_buds_above_height", "creation",
      height=Param(float, REQUIRED, 0.0, 10.0, doc="m above the plant base"))
def _kill_buds_above_height(height):
    """One-line description: what it does, and which Helios function it comes from if any."""
    def fn(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
        ...
    return fn
```

Keep it small, deterministic, and explicit about any draw from the Context stream; test it in
`tests/species/test_rules.py`; add a row to the rule table in `FORMAT.md`. If it reproduces part of a Helios
function, test it against that function bit for bit (see `test_tomato_creation_function_as_rules`).
