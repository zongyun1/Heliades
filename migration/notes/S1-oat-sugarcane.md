# S1: oat and sugarcane

Two species files on the Heliades species format (`docs/species/FORMAT.md`), each a YAML file plus a generated leaf
texture: `heliades/species/defs/oat.yaml`, `heliades/species/defs/sugarcane.yaml`,
`heliades/species/assets/{oat/oat_leaf,sugarcane/sugarcane_leaf}.png`. Galleries: `outputs/species/oat.png`
(`--ages 30 50 70 90`), `outputs/species/sugarcane.png` (`--ages 60 120 200 330`).

Every value of both files is `judged` (general agronomic knowledge); no number is taken from a reference I could
cite exactly, so `sources: {}`. The `note:` fields give the reasoning. They are the first values to replace when
calibrating against data.

## Oat (*Avena sativa*, spring oat; template `wheat`)

**Template.** `wheat`, not `rice`: the wheat creation function gives the temperate cereal culm (8 phytomers, the
upper four internodes elongating ligule to ligule, flag leaf turned off the culm, peduncle shortened by the flag-leaf
sheath) and tillers from the lowest axils via `tiller_count`; `barley.yaml` is the worked example. The rice is a
panicle cereal too, but its culm and tillering are those of paddy rice. The wheat's ear OBJ is replaced by a
procedural **panicle** (format extension below).

Changes from the wheat: taller culm (`internode_length_max` 0.19 m) with a 0.40 m peduncle, longer, broader, laxer
blades (0.28 m, aspect 0.066, more flexible, generated blue-green texture), slower phyllochron (8 d), two
tillers by default (`tiller_count` 2), more erect tillers, 40 d grain fill, 120 d max age.

Measured: mean over Context seeds 1..8 (heading 1..2), fast mode, [min, max].

| trait | age (d) | measured | expected | source |
|---|---|---|---|---|
| plant_height (m) | 40 | 0.33 [0.31, 0.34] | 0.20-0.55 | judged |
| plant_height (m) | 90 | 1.14 [1.14, 1.15] | 0.80-1.25 | judged |
| main_stem_nodes | 90 | 8 [8, 8] | 7-9 | judged |
| leaf_length_max (m) | 50 | 0.28 [0.28, 0.28] | 0.18-0.35 | judged |
| shoot_count (culms) | 60 | 3 [3, 3] | 2-5 | judged |
| first_fruit_day (heading) | <=80 | 56 [56, 56] | 50-70 | judged |
| fruit_count (panicles) | 90 | 3 [3, 3] | 2-5 | judged |
| fruit_length_max (m) | 90 | 0.29 | 0.15-0.32 | judged (PCA extent: rachis plus the spread of the basal branches; the rachis is 0.22 m) |

Looks: an erect tuft of lax strap leaves that becomes three culms with open, spreading panicles of pendant spikelets
on drooping whorled branches, the rachis nodding at the tip. Still wrong: the panicle stays green (no colour change
at ripening, no age-staged organ colour in the format); all three panicles are identical copies of two prototypes
(the inflorescence copies `unique_prototypes` of one deterministic mesh); little seed-to-seed variation (inherited
from the wheat template, whose randomness is mostly in angles); no leaf senescence before harvest.

## Sugarcane (*Saccharum officinarum* hybrids, plant crop; template `sorghum`)

**Template.** `sorghum`, not `maize`. Sugarcane and sorghum are both Andropogoneae, and the sorghum template has
what a stool of cane needs: tillers from the lowest leaf axils (`tiller_count` up to 6, so seven stalks), a crown of
short basal internodes, long flexible sessile blades. The maize is one stalk with an ear and a tassel and no tillers.

Changes from the sorghum: 45 max nodes (not reached in a year) on a 9-day phyllochron, 0.10-0.115 m internodes,
slow elongation, thick yellow-green stalks (`girth_area_factor` 7: about 3 cm at mid-stalk, wider at the stool base),
1.3 m arching blades (generated texture with a broad pale midrib, more flexibility, faster droop with age, wider leaf
angle), a leaf lifespan of 90 d (`max_leaf_lifespan`: lower leaves drop, 10 green leaves per stalk), smaller first
leaves (`leaf_scale_ramp` after the sorghum function: 0.3 at age 0 to 1 at 70 d), flowering off
(`flower_bud_break_probability` 0, `max_terminal_floral_buds` 0), six tillers by default, 400 d max age. The sorghum
function sets the last internode to `peduncle.length`; that is set to an internode length, and the last node is not
reached anyway.

No phenology date is meaningful for a crop harvested vegetative; `flower_count = 0` at 330 d checks that flowering is
off, and `main_stem_nodes` at 330 d is the leaf-appearance clock.

| trait | age (d) | measured | expected | source |
|---|---|---|---|---|
| plant_height (m) | 120 | 1.28 [1.28, 1.29] (2 seeds) | 0.8-2.0 | judged |
| leaf_length_max (m) | 120 | 1.26 [1.26, 1.26] (2 seeds) | 1.0-1.6 | judged |
| shoot_count (stalks) | 150 | 7 [7, 7] (3 seeds, slow) | 5-13 | judged |
| plant_height (m) | 330 | 3.62 [3.41, 3.79] (3 seeds, slow) | 2.8-5.0 | judged |
| main_stem_nodes | 330 | 37 [37, 37] (slow) | 28-42 | judged |
| internode_length_mean (m) | 330 | 0.083 [0.080, 0.086] (slow) | 0.06-0.16 | judged (includes the six 8 mm crown internodes) |
| leaf_count (green, per stool) | 330 | 70 [70, 70] (slow) | 45-110 | judged |
| flower_count | 330 | 0 [0, 0] (slow) | 0 | judged |

Looks: a fountain-like young stool with arching leaves (60-120 d), then a clump of seven bare jointed cane stalks with
a spreading crown of arching leaves (200-330 d). Still wrong: dropped leaves leave the stalk bare (real cane keeps dry
leaves and sheaths, the "trash", on the lower stalk; the format has no senesced-leaf state); no leaf sheaths, so the
upper stalk looks thinner than the leafy cane top; stalk thickness tapers along the whole stalk (pipe model) where
cane is nearly uniform; seven stalks is the template's maximum (a field stool has 6-12 millable stalks, more tillers
at first); little seed-to-seed variation in counts.

## Format extensions

* **Organ shape `panicle`** (`heliades/species/organs.py`, fruit): a nodding rachis with `whorls` whorls of spreading
  branches that droop towards their tips, pendant spikelets (two crossed lanceolate glume planes) on pedicels along
  the distal part of each branch, and a terminal spikelet; 22 parameters (`PANICLE_KEYS`), lengths as fractions of the
  panicle length, vertex colours, a local numpy generator (`seed`, `jitter`) and no Context draw. Added through a
  small registry `EXTRA_FORMS[name] = (organs, keys, mesh_fn)` appended at the end of `organs.py` (three hooks in
  `validate_organ` / `organ_hook`), so other procedural shapes can register the same way without touching the
  dispatch. Row in the FORMAT.md organ table; tests `tests/species/test_organs_panicle.py`.
* **Trait key `slow: true`** (`loader.py` `TRAIT_KEYS`): the trait's test is `pytest.mark.slow`
  (`tests/species/test_traits.py` builds `pytest.param(..., marks=...)`); the traits command always measures it.
  Needed because a 330-day sugarcane stool costs seconds per seed. FORMAT.md traits paragraph; tests
  `tests/species/test_trait_slow.py`.

No change to Helios-ported code or to existing species files.

## Tests

`pytest -q tests/species`: 103 passed, 6 skipped (the sugarcane slow traits) before the last oat seed change; default
trait time oat about 33 s, sugarcane about 15 s. `python -m heliades.species.traits oat sugarcane`: all 16 traits in
range (slow ones included). `python -m heliades.parity plantarch.species`: 6/6 cases pass. Both species build in exact mode, the batch builder (exact and fast) and fast mode.
