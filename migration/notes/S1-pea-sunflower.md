# S1: pea and sunflower species files

Files: `heliades/species/defs/{pea,sunflower}.yaml`, generated leaves `heliades/species/assets/{pea,sunflower}/*.png`,
galleries `outputs/species/{pea,sunflower}.png`. Every value of both files is `judged` (general agronomic knowledge of
the crops; I could not tie any single number to a reference I am certain of, so none is cited).

Gallery commands (the gallery has no default ages for these species):

    .venv/bin/python scripts/species/gallery.py pea --ages 30 50 65 85 --compare soybean
    .venv/bin/python scripts/species/gallery.py sunflower --ages 30 45 65 90 --compare soybean

## Template choice

**Both on the soybean.** The soybean (like the bean and the cowpea) is a one-node `unifoliate` base with an opposite
pair of small leaves, the `trifoliate` main stem appended to it, petiolate GenericLeaf blades.

* **Pea**: the soybean rather than the bean or the cowpea because it has an indeterminate main stem of 16-19 nodes
  (pea: 18-25) and a `branch` shoot type for the basal branches; the bean's main stem has 4-7 nodes and the cowpea
  switches on leaf-angle tracking. The base node carries the two scale leaves of a hypogeal pea seedling.
* **Sunflower**: a dicot with petiolate blades on one upright stem, and the soybean's base pair is the pair of
  cotyledons. Branching killed (no child shoot types, no axillary buds, `kill_buds both` on every node), one terminal
  floral bud (`max_terminal_floral_buds: 1`) carries the capitulum. maize / sorghum also end in a terminal organ but
  are sheathed strap leaves on a culm, which the sunflower is not; reshaping a grass would mean fighting its creation
  function (sheath / blade logic) on every phytomer.

Consequence of the template: the plant has two shoots (base node + main stem), so `shoot_count` of an unbranched
sunflower is 2, and `main_stem_nodes` / `internode_length_mean` (shoot 0) would measure the one-node base; they are
not used.

## How the organs are expressed

**Pea leaf** (9 "leaflets" per petiole, per-leaflet generated textures, the new `{texture: {index: name}}` form):
tip and the first pair (indices 0, +-1) = tendrils (a thin strip texture in the same frame), +-2 and +-3 = two pairs
of ovate leaflets, +-4 = the two large stipules; `leaflet_offset` 0.2857 (Helios clamps to 1/3.5) puts the last pair
at offset 0, i.e. at the node. Flowers: the Helios bean's white flower OBJs (`obj_variants` with `{state}`, count 1;
the `{library: BeanFlowerPrototype}` form does not resolve: the bean hooks are not in `hooks`), pod: `BeanPod.obj`
scaled to 5-10 cm. Distichous phyllotaxis (180 deg), first flowers from about node 10 (`kill_buds floral node_below
10`), basal branches only (`kill_buds vegetative node_from 3`).

**Sunflower head** (the new `capitulum` organ): closed flower = the green bract-covered bud (R1-R4), open flower = the
yellow-rayed head (R5), fruit = the ripening head with wilted, drooping rays and a yellowing back (R6-R9). The neck is
the peduncle (16 cm, curvature -300..-450 deg/m: the head leans over), and `fruit_gravity_factor_fraction` 0.45-0.6
turns the ripening head face-down: the nodding head. `inflorescence_maturity_period: 0` makes the fruit head at full
size (Helios would grow a fruit from 25 % to 100 %, which would make the head shrink from flower to fruit). Leaf:
broadly ovate, cordate base (the new `sinus` key of the leaf outline), serrate, acuminate.

## Trait tables

Measured = mean over Context seeds 1..8 (timing traits: the listed seeds), fast mode; [min, max] over the seeds.

**Pea** (*Pisum sativum*, soybean template; age 0 = emergence)

| trait | age (d) | measured | expected | source |
|---|---|---|---|---|
| plant_height (m) | 30 | 0.29 [0.24, 0.32] | 0.12-0.40 | judged |
| plant_height (m) | 75 | 0.81 [0.55, 0.99] | 0.50-1.00 | judged |
| leaf_length_max (m) | 50 | 0.063 [0.058, 0.067] | 0.03-0.08 | judged |
| shoot_count (base + main + branches) | 60 | 3.5 [2, 5] | 2-5 | judged |
| first_flower_day (seeds 1..4) | <=70 | 45.3 [42, 47] | 38-60 | judged |
| first_fruit_day (pod, seeds 1..4) | <=80 | 56.8 [53, 60] | 48-70 | judged |
| fruit_count (pods) | 85 | 9.9 [5, 14] | 4-20 | judged |
| fruit_length_max (pod, m) | 85 | 0.069 [0.064, 0.084] | 0.05-0.10 | judged |

**Sunflower** (*Helianthus annuus*, single-headed hybrid, soybean template; age 0 = emergence)

| trait | age (d) | measured | expected | source |
|---|---|---|---|---|
| plant_height (m) | 30 | 0.53 [0.44, 0.58] | 0.25-0.70 | judged |
| plant_height (m) | 90 | 2.17 [1.91, 2.50] | 1.50-2.50 | judged |
| leaf_length_max (m) | 60 | 0.29 [0.27, 0.33] | 0.20-0.35 | judged |
| leaf_count (incl. cotyledons) | 60 | 30.9 [29, 33] | 25-37 | judged |
| shoot_count (base + main: unbranched; seeds 1..4) | 60 | 2 [2, 2] | 2 | judged |
| first_flower_day (head bud visible, R1; seeds 1..4) | <=60 | 40 [37, 42] | 30-50 | judged |
| first_fruit_day (end of flowering, R6; seeds 1..3) | <=100 | 70.7 [68, 73] | 62-82 | judged |
| fruit_count (heads; seeds 1..4) | 100 | 1 [1, 1] | 1 | judged |
| fruit_length_max (head diameter incl. wilted rays, m) | 100 | 0.24 [0.20, 0.28] | 0.15-0.30 | judged |

Trait-test time in the default run (`pytest --durations`): about 37 s (pea) and 33 s (sunflower; shoot_count and fruit_count on 4 seeds, they do not vary); nothing needed `slow`.

## What still looks wrong

* **Pea**: no climbing (a single unsupported plant stands upright with gravitropism; real field peas lean on
  neighbours / lodge). Tendrils are flat unbranched strips the length of a leaflet (real ones are longer, branched
  and coiled); in the matplotlib gallery they mostly disappear (triangles kept by the texel at their centroid).
  Stipules are the same size as the leaflets (the leaflets of one leaf share one `prototype_scale` and one aspect;
  real stipules are often larger than the leaflets and clasp the stem). No leaf senescence of the lower nodes. The
  bean's pod and flower OBJs are a fair stand-in, but the pod is a bean pod shape.
* **Sunflower**: leaves are all alternate (the lowest 2-3 pairs of a real sunflower are opposite; Helios has one
  `petioles_per_internode` per shoot type). Leaf size rises from the base (seedling ramp) but does not fall again
  towards the head (no rule for a size profile along the stem). At anthesis the head leans over on a bent neck
  rather than facing east; disc florets are a flat colour (no phyllotactic spiral texture). The fruit head appears at
  full size at the end of flowering, there is no seed-fill growth.

## Format extensions (all small, appended)

| where | what | test |
|---|---|---|
| `organs.py` | `capitulum` organ form (flower: bud / open head; fruit: ripening head), `CAPITULUM_KEYS`, `capitulum_mesh` | `test_organs_textures.py::test_capitulum_flower_and_fruit` |
| `leaf_texture.py` | `sinus {depth, width}`: a cordate basal notch, midrib kept solid across it | `test_organs_textures.py::test_sinus_cuts_a_cordate_base_and_keeps_the_midrib` |
| `loader.py` | `leaf_texture_file: {texture: {index: name}}`: generated textures per compound-leaf index | `test_loader.py::test_per_leaflet_generated_texture_map` |

`docs/species/FORMAT.md`: rows for the three, and the "cannot express" list updated. No Helios-ported code and no
existing species file changed. Exact mode builds both species (direct build + `advanceTime`, and the batch builder in
exact mode).

Tests: `pytest -q tests/species` 108 passed (before the last seeds change; the sunflower subset re-run: 10 passed);
`python -m heliades.parity plantarch.species` 6/6 PASS.
