# SPDX-License-Identifier: LGPL-2.1-or-later
"""The named growth rules a species file can use (no code in a species file: only rule names and parameters).

Three kinds:

``creation``
    run when a phytomer is created, in the order listed (``phytomer_creation`` of a shoot type); the arguments are
    those of ``PhytomerParameters::phytomer_creation_function``:
    ``(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age)``;
``callback``
    run on every phytomer after each growth step (``phytomer_callback``): ``(phytomer)``;
``build``
    run by the generic plant builder after the plant has its base stem and phenology: ``(pa, plantID,
    base_position)``.

Each rule is a :class:`Rule` with a parameter schema (type, default, range), registered with :func:`rule`. A rule is
small, deterministic and draws from the Context stream only where it says so. To add one: write the function here
(or in a module imported here), give it a schema and a docstring, add a test in ``tests/species/test_rules.py`` and
list it in ``docs/species/FORMAT.md``.

Float arithmetic follows the C++ the rules were taken from (``TomatoPhytomerCreationFunction``: the ramps are
evaluated in double and narrowed once), so a species file that lists the tomato rules builds bit-identically to the
library tomato's creation function.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

f32 = np.float32

REQUIRED = object()

RULES: dict[str, "Rule"] = {}


@dataclass
class Param:
    type: type | tuple
    default: object = REQUIRED
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    doc: str = ""


@dataclass
class Rule:
    name: str
    kind: str                       # creation | callback | build
    params: dict
    factory: object                 # factory(**params) -> callable
    doc: str = ""
    kinds: tuple = field(default=())

    def accepts(self, kind: str) -> bool:
        return kind == self.kind or kind in self.kinds


def rule(name: str, kind: str, /, also: tuple = (), **params):
    """Register ``factory`` (a function of the rule parameters returning the callable) as rule ``name``."""
    def deco(factory):
        RULES[name] = Rule(name, kind, params, factory, (factory.__doc__ or "").strip(), tuple(also))
        return factory
    return deco


# ---- creation rules -----------------------------------------------------------------------------------

_WHICH = ("floral", "vegetative", "both")


@rule("kill_buds", "creation",
      which=Param(str, choices=_WHICH, doc="floral, vegetative or both (floral first)"),
      node_below=Param(int, None, 0, 100000, doc="only phytomers whose node index on their shoot is < this"),
      node_from=Param(int, None, 0, 100000, doc="only phytomers whose node index is >= this"),
      rank=Param(int, None, 0, 100, doc="only phytomers on shoots of this rank (0 = main stem)"),
      rank_above=Param(int, None, 0, 100, doc="only phytomers on shoots of rank > this"))
def _kill_buds(which, node_below=None, node_from=None, rank=None, rank_above=None):
    """Kill (``BUD_DEAD``) the axillary floral and/or vegetative buds of a new phytomer that meets every given
    condition. ``{which: floral, node_below: 8, rank: 0}`` is the tomato's "no trusses below node 8 of the main
    stem"; ``{which: both, rank_above: 1}`` its "no branching beyond second-order shoots"."""
    from ..plantarchitecture.params import BUD_DEAD

    def fn(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
        if node_below is not None and not shoot_node_index < node_below:
            return
        if node_from is not None and not shoot_node_index >= node_from:
            return
        if rank is not None and phytomer.rank != rank:
            return
        if rank_above is not None and not phytomer.rank > rank_above:
            return
        if which in ("floral", "both"):
            phytomer.setFloralBudState(BUD_DEAD)
        if which in ("vegetative", "both"):
            phytomer.setVegetativeBudState(BUD_DEAD)
    return fn


def _ramp(intercept, slope, days):
    def scale(plant_age):
        age = float(f32(plant_age))
        return f32(min(1.0, intercept + slope * age / days))     # double, narrowed once (AS:4104-4107)
    return scale


_RAMP = dict(intercept=Param(float, REQUIRED, 0.0, 1.0, doc="scale of a phytomer made at plant age 0"),
             slope=Param(float, REQUIRED, 0.0, 10.0, doc="added scale per `days` of plant age"),
             days=Param(float, REQUIRED, 0.001, 3650.0, doc="age unit of the slope (days)"))


@rule("leaf_scale_ramp", "creation", **_RAMP)
def _leaf_scale_ramp(intercept, slope, days):
    """Seedling leaves: scale the new phytomer's leaves by ``min(1, intercept + slope * plant_age / days)``
    (``Phytomer::scaleLeafPrototypeScale``). The tomato's is ``{intercept: 0.5, slope: 0.5, days: 10}``."""
    scale = _ramp(intercept, slope, days)

    def fn(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
        phytomer.scaleLeafPrototypeScale(scale(plant_age))
    return fn


@rule("internode_scale_ramp", "creation", **_RAMP)
def _internode_scale_ramp(intercept, slope, days):
    """Seedling internodes: scale the new phytomer's maximum internode length by
    ``min(1, intercept + slope * plant_age / days)`` (``Phytomer::scaleInternodeMaxLength``). The tomato's is
    ``{intercept: 0.7, slope: 0.3, days: 10}``."""
    scale = _ramp(intercept, slope, days)

    def fn(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
        phytomer.scaleInternodeMaxLength(scale(plant_age))
    return fn


@rule("leaf_bump_map", "creation",
      texture=Param(str, doc="bump-map image (a Helios texture name, a species asset or a path)"),
      strength=Param(float, REQUIRED, 0.0, 1000.0, doc="bump strength"))
def _leaf_bump_map(texture, strength, species=None):
    """Give every leaf primitive of the new phytomer a bump map (``Context::setPrimitiveBumpMap``), as the
    cucumber's creation function does with ``CucumberLeaf_0_bump.png`` at strength 12.58861."""
    from .assets import resolve_asset
    path = resolve_asset(texture, species)
    strength32 = f32(strength)

    def fn(phytomer, shoot_node_index, parent_shoot_node_index, shoot_max_nodes, plant_age):
        context = phytomer.parent_shoot_ptr.context_ptr
        leaf_UUIDs = []
        for petiole in phytomer.leaf_objIDs:
            for objID in petiole:
                leaf_UUIDs.extend(context.object_primitive_uuids(objID))
        if leaf_UUIDs:
            context.set_primitive_bump_map(leaf_UUIDs, path, strength32)
    return fn


def _library_hook(name):
    from ..plantarchitecture import hooks
    from ..plantarchitecture import library  # noqa: F401  (binds the library hooks)
    from ..plantarchitecture import inflorescence
    inflorescence._bind_library_hooks()
    h = getattr(hooks, name, None)
    if not isinstance(h, hooks.Hook) or h.impl is None:
        raise ValueError(f"no library function named '{name}'")
    return h


@rule("library_function", "creation", also=("callback",),
      name=Param(str, doc="a Helios library creation / callback function, e.g. WheatPhytomerCreationFunction"))
def _library_function(name):
    """Call a ported Helios library phytomer creation or callback function by name (e.g.
    ``TomatoPhytomerCreationFunction``, ``WheatPhytomerCreationFunction``, ``CucumberMainPhytomerCallbackFunction``).
    For rules that are specific to one Helios species and not worth splitting into generic ones."""
    h = _library_hook(name)

    def fn(*args):
        return h(*args)
    return fn


# ---- callback rules ------------------------------------------------------------------------------------

@rule("harvest_ripe_fruit", "callback")
def _harvest_ripe_fruit():
    """Pick every fruit that has reached full size (``BUD_FRUITING`` with ``current_fruit_scale_factor >= 1``:
    the bud becomes ``BUD_DEAD``), as a greenhouse crop is harvested continuously. The cucumber's."""
    from ..plantarchitecture.library.cucumber import cucumberHarvestFullSizeFruit
    return cucumberHarvestFullSizeFruit


@rule("high_wire_lowering", "callback")
def _high_wire_lowering():
    """High-wire training: once the main-stem head is 3.8 m above the base, lower the stem and lay the old part along
    +y at 0.3 m, sliding the string's attraction points with it (the cucumber's ``CucumberMainPhytomerCallbackFunction``
    after the harvest). Use with the build rule ``high_wire_string``."""
    from ..plantarchitecture.library.cucumber import cucumberLowerAndLeanStem
    return cucumberLowerAndLeanStem


# ---- build rules ---------------------------------------------------------------------------------------

@rule("high_wire_string", "build",
      view_half_angle_deg=Param(float, 80.0, 0.0, 180.0),
      look_ahead_distance=Param(float, 0.15, 0.0, 10.0),
      attraction_weight=Param(float, 0.8, 0.0, 1.0))
def _high_wire_string(view_half_angle_deg, look_ahead_distance, attraction_weight):
    """A vertical string of attraction points from the base to 4.4 m (every 5 cm), the stem trained up it to a wire
    at 3.8 m (the cucumber's ``enableAttractionPoints(plantID, CucumberStringAttractionPoints(base), 80, 0.15,
    0.8)``)."""
    from ..plantarchitecture.library.cucumber import CucumberStringAttractionPoints, enableAttractionPoints

    def fn(pa, plantID, base_position):
        enableAttractionPoints(pa, plantID, CucumberStringAttractionPoints(base_position), view_half_angle_deg,
                               look_ahead_distance, attraction_weight)
    return fn


# ---- validation ----------------------------------------------------------------------------------------

def check_params(rule_: Rule, given: dict, where: str, error) -> dict:
    """Validate ``given`` against the rule's schema; returns the full parameter dict (defaults filled in)."""
    out = {}
    for k in given:
        if k not in rule_.params:
            error(f"{where}: rule '{rule_.name}' has no parameter '{k}' (it takes: "
                  f"{', '.join(rule_.params) or 'none'})")
    for k, p in rule_.params.items():
        if k not in given:
            if p.default is REQUIRED:
                error(f"{where}: rule '{rule_.name}' needs parameter '{k}'")
            out[k] = p.default
            continue
        v = given[k]
        types = p.type if isinstance(p.type, tuple) else (p.type,)
        if float in types and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if isinstance(v, bool) and bool not in types or not isinstance(v, types):
            error(f"{where}.{k}: expected {'/'.join(t.__name__ for t in types)}, got {v!r}")
        if p.choices is not None and v not in p.choices:
            error(f"{where}.{k}: must be one of {', '.join(p.choices)}, got {v!r}")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if (p.min is not None and v < p.min) or (p.max is not None and v > p.max):
                error(f"{where}.{k}: {v} is outside [{p.min}, {p.max}]")
        out[k] = v
    return out


def make(name: str, kind: str, params: dict, species=None):
    """The callable of rule ``name`` with validated ``params``."""
    r = RULES[name]
    if r.name == "leaf_bump_map":
        return r.factory(species=species, **params)
    return r.factory(**params)
