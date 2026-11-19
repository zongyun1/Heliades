# SPDX-License-Identifier: LGPL-2.1-or-later
"""Species files: parse, validate, and register them as plant models.

A species file (YAML, ``heliades/species/defs/<name>.yaml``; format in ``docs/species/FORMAT.md``) names a Helios
library model as its ``template`` and overrides, per shoot type, any parameter value (each with a ``source``: a key
of its ``sources`` list or ``judged``), the flower / fruit prototypes, the phytomer creation / callback rules and the
plant builder. :func:`register` turns it into a plant model of a PlantArchitecture:

* the **shoot initializer** runs the template's ``initialize<Template>Shoots`` (so the template's draws from the
  Context stream happen exactly as in the library: the leaf-prototype identifiers), then copies each listed shoot
  type from its template shoot type (copy constructor: no draw), applies the overrides in file order (assignments and
  ``*Distribution`` calls: no draw), drops ``drop_shoot_types`` and defines the result;
* the **plant builder** is either the template's builder (``builder: template``) followed by phenology / max-age
  overrides and build rules, or the generic builder (``builder: generic``): ``addPlantInstance``, the base rotation
  (its random components drawn right to left, as GCC evaluates ``make_AxisRotation(randu(), randu(), randu())``),
  ``addBaseStemShoot``, ``breakPlantDormancy``, ``setPlantPhenologicalThresholds``, ``max_age``, build rules.

``PlantArchitecture.loadPlantModelFromLibrary(name)`` registers a species on first use (see
:func:`register_if_species`), so the batch builder, fast mode and everything else that loads a model by name work
unchanged. The library models' registry (``getAvailablePlantModels``) is not touched until then, so it stays the
Helios list.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import fields as F
from . import rules as R

f32 = np.float32

FORMAT = "heliades-species/1"
DEFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defs")
ENV_PATH = "HELIADES_SPECIES_PATH"
RESERVED_SOURCES = ("judged",)

TOP_KEYS = ("format", "name", "template", "metadata", "sources", "leaf_textures", "shoot_types", "drop_shoot_types",
            "build", "traits", "note")
METADATA_KEYS = ("latin_name", "common_name", "cultivar", "plant_type", "leaf_inclination_beta", "description")
SOURCE_KEYS = ("citation", "doi", "url", "note")
SHOOT_TYPE_KEYS = ("from", "child_shoot_types", "flower", "fruit", "phytomer_creation", "phytomer_callback",
                   "note") + tuple(F.SECTIONS)
BUILD_KEYS = ("builder", "source", "note", "base_offset", "base_rotation", "base_stem", "phenology", "max_age",
              "rules", "build_parameter_defaults")
BASE_STEM_KEYS = ("shoot_type", "nodes", "internode_radius", "internode_length_max",
                  "internode_length_scale_factor_fraction", "leaf_scale_factor_fraction", "radius_taper")
PHENOLOGY_KEYS = ("dormancy_break", "flower_initiation", "flower_opening", "fruit_set", "fruit_maturity", "dormancy",
                  "max_leaf_lifespan", "evergreen")
TRAIT_KEYS = ("trait", "age", "range", "source", "note", "seeds", "slow")


class SpeciesError(ValueError):
    """A species file is invalid (the message starts with the file and the key path)."""


@dataclass
class Override:
    section: str
    field: str
    kind: str
    op: str
    args: object
    source: str
    note: str = ""


@dataclass
class RuleUse:
    name: str
    params: dict
    source: str
    note: str = ""


@dataclass
class ShootTypeSpec:
    label: str
    from_label: str
    overrides: list
    child_shoot_types: dict | None = None        # None: keep the template's
    flower: tuple = ("template",)
    fruit: tuple = ("template",)
    creation: object = "template"                # "template" | "none" | [RuleUse]
    callback: object = "template"


@dataclass
class BuildSpec:
    builder: str = "template"
    base_offset: tuple = (0.0, 0.0, 0.0)
    base_rotation: dict = field(default_factory=dict)   # pitch / yaw / roll -> ("const", v) | ("randu", lo, hi)
    base_stem: dict | None = None
    phenology: dict = field(default_factory=dict)
    max_age: float | None = None
    rules: list = field(default_factory=list)
    build_parameter_defaults: dict = field(default_factory=dict)


@dataclass
class SpeciesSpec:
    name: str
    path: str
    template: str
    metadata: dict
    sources: dict
    leaf_textures: dict
    shoot_types: list
    drop_shoot_types: list
    build: BuildSpec
    traits: list
    raw: dict = field(default_factory=dict, repr=False)

    def summary(self) -> dict:
        """Counts of what the file sets and how it is sourced."""
        n_cited = n_judged = 0
        for st in self.shoot_types:
            for ov in st.overrides:
                if ov.source == "judged":
                    n_judged += 1
                else:
                    n_cited += 1
        return {"name": self.name, "template": self.template, "overrides": n_cited + n_judged, "cited": n_cited,
                "judged": n_judged, "traits": len(self.traits)}


# ---- file lookup ---------------------------------------------------------------------------------------

def species_dirs() -> list[str]:
    dirs = [DEFS_DIR]
    for d in os.environ.get(ENV_PATH, "").split(os.pathsep):
        if d and os.path.isdir(d):
            dirs.append(d)
    return dirs


def find_species_file(name: str) -> str | None:
    if name.endswith((".yaml", ".yml")) and os.path.exists(name):
        return os.path.abspath(name)
    for d in species_dirs():
        for ext in (".yaml", ".yml"):
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return p
    return None


def available_species() -> list[str]:
    out = set()
    for d in species_dirs():
        for f in os.listdir(d):
            if f.endswith((".yaml", ".yml")) and not f.startswith("_"):
                out.add(os.path.splitext(f)[0])
    return sorted(out)


# ---- parsing -------------------------------------------------------------------------------------------

class _Ctx:
    def __init__(self, path):
        self.path = path

    def error(self, msg):
        raise SpeciesError(f"{os.path.basename(self.path)}: {msg}")


def _keys(ctx, d, allowed, where, required=()):
    if not isinstance(d, dict):
        ctx.error(f"{where}: expected a mapping, got {type(d).__name__}")
    for k in d:
        if k not in allowed:
            ctx.error(f"{where}: unknown key '{k}' (allowed: {', '.join(allowed)})")
    for k in required:
        if k not in d:
            ctx.error(f"{where}: '{k}' is required")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _source(ctx, d, where, sources, required=True):
    s = d.get("source")
    if s is None:
        if required:
            ctx.error(f"{where}: every value needs a 'source' (a key of 'sources', or 'judged')")
        return ""
    if not isinstance(s, str) or (s not in sources and s not in RESERVED_SOURCES):
        ctx.error(f"{where}.source: '{s}' is not in 'sources' (known: {', '.join(sorted(sources)) or 'none'}; or use "
                  "'judged')")
    note = d.get("note", "")
    if not isinstance(note, str):
        ctx.error(f"{where}.note: expected text")
    return s


def _range(ctx, v, lo, hi, where):
    if lo is not None and not (lo <= v <= hi):
        ctx.error(f"{where}: {v} is outside the valid range [{lo}, {hi}]")


OPS_BY_KIND = {
    "float": ("value", "uniform", "normal", "weibull"),
    "int": ("value", "uniform", "discrete"),
    "uint": ("value",), "real": ("value",), "bool": ("value",), "rgb": ("value",), "vec3": ("value",),
    "str": ("value",), "texture_map": ("value", "texture"), "length_map": ("value",),
}


def parse_override(ctx, raw, section, name, kind, lo, hi, where, sources, textures) -> Override:
    ops = OPS_BY_KIND[kind]
    if not isinstance(raw, dict):
        ctx.error(f"{where}: expected a mapping like {{value: ..., source: ...}}, got {raw!r}")
    _keys(ctx, raw, ops + ("source", "note"), where)
    given = [k for k in ops if k in raw]
    if len(given) != 1:
        ctx.error(f"{where}: give exactly one of {', '.join(ops)}")
    op = given[0]
    v = raw[op]
    w = f"{where}.{op}"
    source = _source(ctx, raw, where, sources)
    if kind == "float":
        if op == "value":
            if not _is_num(v):
                ctx.error(f"{w}: expected a number")
            _range(ctx, v, lo, hi, w)
            args = float(v)
        else:
            if not (isinstance(v, list) and len(v) == 2 and all(_is_num(x) for x in v)):
                ctx.error(f"{w}: expected two numbers")
            a, b = float(v[0]), float(v[1])
            if op == "uniform":
                if a > b:
                    ctx.error(f"{w}: the minimum {a} is above the maximum {b}")
                _range(ctx, a, lo, hi, w)
                _range(ctx, b, lo, hi, w)
            elif op == "normal":
                if b < 0:
                    ctx.error(f"{w}: the standard deviation must be >= 0")
                _range(ctx, a, lo, hi, w)
            else:
                if a <= 0 or b <= 0:
                    ctx.error(f"{w}: Weibull shape and scale must be > 0")
            args = (a, b)
    elif kind == "int":
        if op == "value":
            if not isinstance(v, int) or isinstance(v, bool):
                ctx.error(f"{w}: expected a whole number")
            _range(ctx, v, lo, hi, w)
            args = int(v)
        elif op == "uniform":
            if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                for x in v)) or v[0] > v[1]:
                ctx.error(f"{w}: expected [min, max] whole numbers")
            for x in v:
                _range(ctx, x, lo, hi, w)
            args = (int(v[0]), int(v[1]))
        else:
            if not (isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) for x in v)):
                ctx.error(f"{w}: expected a list of whole numbers")
            for x in v:
                _range(ctx, x, lo, hi, w)
            args = [int(x) for x in v]
    elif kind == "uint":
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            ctx.error(f"{w}: expected a whole number >= 0")
        _range(ctx, v, lo, hi, w)
        args = int(v)
    elif kind == "real":
        if not _is_num(v):
            ctx.error(f"{w}: expected a number")
        _range(ctx, v, lo, hi, w)
        args = float(v)
    elif kind == "bool":
        if not isinstance(v, bool):
            ctx.error(f"{w}: expected true or false")
        args = v
    elif kind == "rgb":
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) and 0 <= x <= 1 for x in v)):
            ctx.error(f"{w}: expected three numbers in [0, 1]")
        args = tuple(float(x) for x in v)
    elif kind == "vec3":
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)):
            ctx.error(f"{w}: expected three numbers")
        for x in v:
            _range(ctx, x, lo, hi, w)
        args = tuple(float(x) for x in v)
    elif kind == "str":
        if not isinstance(v, str):
            ctx.error(f"{w}: expected a file name")
        args = v
    elif kind == "texture_map":
        if op == "texture":
            if v not in textures:
                ctx.error(f"{w}: '{v}' is not one of this file's leaf_textures ({', '.join(textures) or 'none'})")
            args = v
        elif isinstance(v, str):
            args = {0: v}
        elif isinstance(v, dict) and all(isinstance(x, str) for x in v.values()):
            try:
                args = {int(k): x for k, x in v.items()}
            except (TypeError, ValueError):
                ctx.error(f"{w}: keys must be compound-leaf indices")
        else:
            ctx.error(f"{w}: expected a file name or {{index: file name}}")
    else:   # length_map
        if _is_num(v):
            _range(ctx, v, lo, hi, w)
            args = {0: float(v)}
        elif isinstance(v, dict) and all(_is_num(x) for x in v.values()):
            for x in v.values():
                _range(ctx, x, lo, hi, w)
            try:
                args = {int(k): float(x) for k, x in v.items()}
            except (TypeError, ValueError):
                ctx.error(f"{w}: keys must be compound-leaf indices")
        else:
            ctx.error(f"{w}: expected a number or {{index: number}}")
    return Override(section, name, kind, op, args, source, raw.get("note", ""))


def parse_rules(ctx, raw, kind, where, sources):
    if raw in ("template", "none"):
        return raw
    if not isinstance(raw, list):
        ctx.error(f"{where}: expected template, none or a list of rules")
    out = []
    for i, entry in enumerate(raw):
        w = f"{where}[{i}]"
        if not isinstance(entry, dict) or "rule" not in entry:
            ctx.error(f"{w}: expected {{rule: <name>, source: ..., <parameters>}}")
        name = entry["rule"]
        r = R.RULES.get(name)
        if r is None:
            ctx.error(f"{w}.rule: unknown rule '{name}' (known {kind} rules: "
                      f"{', '.join(n for n, x in sorted(R.RULES.items()) if x.accepts(kind))})")
        if not r.accepts(kind):
            ctx.error(f"{w}.rule: '{name}' is a {r.kind} rule, not a {kind} rule")
        source = _source(ctx, entry, w, sources)
        params = {k: v for k, v in entry.items() if k not in ("rule", "source", "note")}
        params = R.check_params(r, params, w, ctx.error)
        out.append(RuleUse(name, params, source, entry.get("note", "")))
    return out


def _angle(ctx, v, where):
    """A base-rotation component in radians: a number or '<k>pi' (``float(f32(k)) * M_PI`` narrowed to float)."""
    if _is_num(v):
        return f32(v)
    if isinstance(v, str) and v.replace(" ", "").endswith("pi"):
        k = v.replace(" ", "")[:-2].rstrip("*") or "1"
        try:
            return f32(float(f32(float(k))) * np.pi)
        except ValueError:
            pass
    ctx.error(f"{where}: expected radians (a number or '<k>pi'), got {v!r}")


def parse_build(ctx, raw, sources) -> BuildSpec:
    b = BuildSpec()
    if raw is None:
        return b
    _keys(ctx, raw, BUILD_KEYS, "build")
    b.builder = raw.get("builder", "template")
    if b.builder not in ("template", "generic"):
        ctx.error("build.builder: must be template or generic")
    if b.builder == "generic":
        for k in ("base_stem", "phenology", "max_age"):
            if k not in raw:
                ctx.error(f"build.{k}: required by the generic builder")
        _source(ctx, raw, "build", sources)
    else:
        for k in ("base_offset", "base_rotation", "base_stem"):
            if k in raw:
                ctx.error(f"build.{k}: only for builder: generic (the template builder sets it)")
    if "base_offset" in raw:
        v = raw["base_offset"]
        if not (isinstance(v, list) and len(v) == 3 and all(_is_num(x) for x in v)):
            ctx.error("build.base_offset: expected three numbers (m)")
        b.base_offset = tuple(float(x) for x in v)
    rot = raw.get("base_rotation", {}) or {}
    _keys(ctx, rot, ("pitch", "yaw", "roll"), "build.base_rotation")
    for k in ("pitch", "yaw", "roll"):
        v = rot.get(k, 0.0)
        w = f"build.base_rotation.{k}"
        if isinstance(v, dict):
            _keys(ctx, v, ("randu",), w, required=("randu",))
            lohi = v["randu"]
            if not (isinstance(lohi, list) and len(lohi) == 2):
                ctx.error(f"{w}.randu: expected [min, max]")
            lo, hi = _angle(ctx, lohi[0], w), _angle(ctx, lohi[1], w)
            if lo > hi:
                ctx.error(f"{w}.randu: min above max")
            b.base_rotation[k] = ("randu", lo, hi)
        else:
            b.base_rotation[k] = ("const", _angle(ctx, v, w))
    if "base_stem" in raw:
        bs = raw["base_stem"]
        _keys(ctx, bs, BASE_STEM_KEYS, "build.base_stem", required=BASE_STEM_KEYS[:6])
        if not isinstance(bs["shoot_type"], str):
            ctx.error("build.base_stem.shoot_type: expected a shoot type label")
        if not isinstance(bs["nodes"], int) or isinstance(bs["nodes"], bool) or bs["nodes"] < 1:
            ctx.error("build.base_stem.nodes: expected a whole number >= 1")
        il = bs["internode_length_max"]
        if not (il == "shoot_type" or (_is_num(il) and 0 <= il <= 5)):
            ctx.error("build.base_stem.internode_length_max: a length in m, or 'shoot_type' (the shoot type's "
                      "internode_length_max, sampled)")
        for k in ("internode_radius", "internode_length_scale_factor_fraction", "leaf_scale_factor_fraction",
                  "radius_taper"):
            v = bs.get(k, 0.0)
            if not _is_num(v) or v < 0:
                ctx.error(f"build.base_stem.{k}: expected a number >= 0")
        b.base_stem = {k: bs.get(k, 0.0) for k in BASE_STEM_KEYS}
    ph = raw.get("phenology", {}) or {}
    _keys(ctx, ph, PHENOLOGY_KEYS, "build.phenology",
          required=PHENOLOGY_KEYS[:6] if b.builder == "generic" else ())
    for k, v in ph.items():
        w = f"build.phenology.{k}"
        if not isinstance(v, dict) or "value" not in v:
            ctx.error(f"{w}: expected {{value: ..., source: ...}}")
        _keys(ctx, v, ("value", "source", "note"), w)
        _source(ctx, v, w, sources)
        x = v["value"]
        if k == "evergreen":
            if not isinstance(x, bool):
                ctx.error(f"{w}.value: expected true or false")
        elif not _is_num(x) or not (-1 <= x <= 100000):
            ctx.error(f"{w}.value: expected degree-days in [-1, 100000] (-1: never)")
        b.phenology[k] = x
    if "max_age" in raw:
        v = raw["max_age"]
        if not isinstance(v, dict) or "value" not in v:
            ctx.error("build.max_age: expected {value: ..., source: ...}")
        _keys(ctx, v, ("value", "source", "note"), "build.max_age")
        _source(ctx, v, "build.max_age", sources)
        if not _is_num(v["value"]) or not (0 < v["value"] <= 100000):
            ctx.error("build.max_age.value: expected days in (0, 100000]")
        b.max_age = float(v["value"])
    b.rules = parse_rules(ctx, raw.get("rules", []), "build", "build.rules", sources)
    bpd = raw.get("build_parameter_defaults", {}) or {}
    if not isinstance(bpd, dict):
        ctx.error("build.build_parameter_defaults: expected a mapping")
    for k, v in bpd.items():
        w = f"build.build_parameter_defaults.{k}"
        if not isinstance(v, dict) or "value" not in v or not _is_num(v["value"]):
            ctx.error(f"{w}: expected {{value: <number>, source: ...}}")
        _keys(ctx, v, ("value", "source", "note"), w)
        _source(ctx, v, w, sources)
        b.build_parameter_defaults[k] = float(v["value"])
    return b


def parse_spec(raw: dict, path: str = "<memory>") -> SpeciesSpec:
    """Validate a species document (already parsed from YAML) without building anything."""
    from .organs import OrganSpecError, validate_organ
    from .leaf_texture import LeafOutlineError, validate_outline
    ctx = _Ctx(path)
    if not isinstance(raw, dict):
        ctx.error("the file must be a mapping")
    _keys(ctx, raw, TOP_KEYS, "<top>", required=("format", "name", "template", "metadata", "sources"))
    if raw["format"] != FORMAT:
        ctx.error(f"format: expected '{FORMAT}', got {raw['format']!r}")
    name = raw["name"]
    if not isinstance(name, str) or not name.replace("_", "").isalnum() or not name[:1].isalpha():
        ctx.error(f"name: expected an identifier (letters, digits, _), got {name!r}")
    stem = os.path.splitext(os.path.basename(path))[0]
    if path != "<memory>" and stem != name:
        ctx.error(f"name: '{name}' must match the file name '{stem}'")
    from ..plantarchitecture.library import MODELS, SHOOT_INITIALIZERS
    template = raw["template"]
    if template not in SHOOT_INITIALIZERS:
        ctx.error(f"template: '{template}' is not a library model (one of: {', '.join(sorted(SHOOT_INITIALIZERS))})")
    if name in {m[0] for m in MODELS}:
        ctx.error(f"name: '{name}' is a Helios library model; pick another name")

    md = raw["metadata"]
    _keys(ctx, md, METADATA_KEYS, "metadata", required=("latin_name", "common_name"))
    for k in ("latin_name", "common_name", "cultivar", "description"):
        if k in md and not isinstance(md[k], str):
            ctx.error(f"metadata.{k}: expected text")
    if "plant_type" in md and md["plant_type"] not in ("herbaceous", "tree", "weed"):
        ctx.error("metadata.plant_type: herbaceous, tree or weed")
    if md.get("leaf_inclination_beta") is not None:
        b = md["leaf_inclination_beta"]
        if not (isinstance(b, list) and len(b) == 2 and all(_is_num(x) and x > 0 for x in b)):
            ctx.error("metadata.leaf_inclination_beta: two positive Beta parameters, or null")

    sources = raw["sources"] or {}
    if not isinstance(sources, dict):
        ctx.error("sources: expected a mapping key -> {citation: ...}")
    for k, v in sources.items():
        if k in RESERVED_SOURCES:
            ctx.error(f"sources.{k}: '{k}' is reserved")
        _keys(ctx, v, SOURCE_KEYS, f"sources.{k}", required=("citation",))
        if not isinstance(v["citation"], str) or len(v["citation"]) < 10:
            ctx.error(f"sources.{k}.citation: expected a full reference")

    textures = raw.get("leaf_textures", {}) or {}
    if not isinstance(textures, dict):
        ctx.error("leaf_textures: expected a mapping name -> outline")
    for tname, desc in textures.items():
        if not isinstance(tname, str) or not tname.replace("_", "").isalnum():
            ctx.error(f"leaf_textures: '{tname}' is not an identifier")
        d = dict(desc or {})
        d.pop("source", None)
        d.pop("note", None)
        _source(ctx, desc or {}, f"leaf_textures.{tname}", sources)
        try:
            validate_outline(d, f"leaf_textures.{tname}")
        except LeafOutlineError as e:
            ctx.error(str(e))
        textures[tname] = d

    shoot_types = []
    sts = raw.get("shoot_types", {}) or {}
    if not isinstance(sts, dict):
        ctx.error("shoot_types: expected a mapping label -> shoot type")
    for label, st in sts.items():
        w = f"shoot_types.{label}"
        if st is None:
            st = {}
        _keys(ctx, st, SHOOT_TYPE_KEYS, w)
        spec = ShootTypeSpec(label, st.get("from", label), [])
        if not isinstance(spec.from_label, str):
            ctx.error(f"{w}.from: expected a template shoot type label")
        for section, (_path, table) in F.SECTIONS.items():
            block = st.get(section)
            if block is None:
                continue
            _keys(ctx, block, tuple(table), f"{w}.{section}")
            for fname, raw_v in block.items():
                kind, lo, hi = table[fname]
                spec.overrides.append(parse_override(ctx, raw_v, section, fname, kind, lo, hi,
                                                     f"{w}.{section}.{fname}", sources, textures))
        if "child_shoot_types" in st:
            c = st["child_shoot_types"]
            ww = f"{w}.child_shoot_types"
            if not isinstance(c, dict) or "value" not in c:
                ctx.error(f"{ww}: expected {{value: {{label: probability}}, source: ...}}")
            _keys(ctx, c, ("value", "source", "note"), ww)
            _source(ctx, c, ww, sources)
            v = c["value"] or {}
            if not isinstance(v, dict) or not all(isinstance(k, str) and _is_num(p) and 0 <= p <= 1
                                                  for k, p in v.items()):
                ctx.error(f"{ww}.value: expected {{label: probability}}")
            if v and abs(sum(v.values()) - 1) > 1e-6:
                ctx.error(f"{ww}.value: the probabilities must sum to 1")
            spec.child_shoot_types = dict(v)
        for organ in ("flower", "fruit"):
            if organ in st:
                o = st[organ]
                if isinstance(o, dict):
                    _source(ctx, o, f"{w}.{organ}", sources)
                try:
                    setattr(spec, organ, validate_organ(o, organ, f"{w}.{organ}"))
                except OrganSpecError as e:
                    ctx.error(str(e))
        spec.creation = parse_rules(ctx, st.get("phytomer_creation", "template"), "creation",
                                    f"{w}.phytomer_creation", sources)
        spec.callback = parse_rules(ctx, st.get("phytomer_callback", "template"), "callback",
                                    f"{w}.phytomer_callback", sources)
        shoot_types.append(spec)

    drop = raw.get("drop_shoot_types", []) or []
    if not (isinstance(drop, list) and all(isinstance(x, str) for x in drop)):
        ctx.error("drop_shoot_types: expected a list of template shoot type labels")
    for x in drop:
        if x in sts:
            ctx.error(f"drop_shoot_types: '{x}' is also defined under shoot_types")

    build = parse_build(ctx, raw.get("build"), sources)

    from .traits import TRAITS
    traits = []
    for i, t in enumerate(raw.get("traits", []) or []):
        w = f"traits[{i}]"
        _keys(ctx, t, TRAIT_KEYS, w, required=("trait", "age", "range", "source"))
        if t["trait"] not in TRAITS:
            ctx.error(f"{w}.trait: unknown trait '{t['trait']}' (known: {', '.join(sorted(TRAITS))})")
        if not _is_num(t["age"]) or t["age"] < 0:
            ctx.error(f"{w}.age: expected days >= 0")
        rg = t["range"]
        if not (isinstance(rg, list) and len(rg) == 2 and all(_is_num(x) for x in rg) and rg[0] <= rg[1]):
            ctx.error(f"{w}.range: expected [min, max]")
        _source(ctx, t, w, sources)
        seeds = t.get("seeds", 8)
        if not isinstance(seeds, int) or seeds < 1:
            ctx.error(f"{w}.seeds: expected a whole number >= 1")
        slow = t.get("slow", False)
        if not isinstance(slow, bool):
            ctx.error(f"{w}.slow: expected true or false")
        traits.append({"trait": t["trait"], "age": float(t["age"]), "range": (float(rg[0]), float(rg[1])),
                       "source": t["source"], "note": t.get("note", ""), "seeds": seeds, "slow": slow})

    return SpeciesSpec(name, path, template, dict(md), dict(sources), textures, shoot_types, list(drop), build,
                       traits, raw)


_CACHE: dict = {}


def load_spec(name_or_path: str) -> SpeciesSpec:
    """Parse and validate a species file (by species name or path). Cached on the file's modification time."""
    path = find_species_file(name_or_path)
    if path is None:
        raise SpeciesError(f"no species file for '{name_or_path}' (looked in {', '.join(species_dirs())})")
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _CACHE:
        import yaml
        with open(path) as f:
            try:
                raw = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise SpeciesError(f"{os.path.basename(path)}: not valid YAML: {e}") from None
        _CACHE[key] = parse_spec(raw, path)
    return _CACHE[key]


# ---- building ------------------------------------------------------------------------------------------

def _template_functions(template):
    from ..plantarchitecture.library import PLANT_BUILDERS, SHOOT_INITIALIZERS
    from ..plantarchitecture.library.tomato import buildTomatoPlant
    builders = dict(PLANT_BUILDERS)
    builders.setdefault("tomato", buildTomatoPlant)
    return SHOOT_INITIALIZERS[template], builders[template]


def _template_registration(template):
    from ..plantarchitecture.library import MODELS
    for name, plant_type, beta in MODELS:
        if name == template:
            return plant_type, beta
    raise KeyError(template)


def apply_override(sp, ov: Override, generator, species: SpeciesSpec):
    """Apply one override to a ShootParameters, as the C++ statement would (no draws)."""
    from ..plantarchitecture.params import make_RGBcolor
    from ..plantarchitecture.random_parameter import RandomParameterFloat, RandomParameterInt
    obj = F.target(sp, ov.section)
    cur = getattr(obj, ov.field)
    if ov.kind in ("float", "int"):
        assert isinstance(cur, (RandomParameterFloat, RandomParameterInt)), (ov.section, ov.field)
        if ov.op == "value":
            cur.set(ov.args)
            return
        if cur.generator is None:
            cur.generator = generator        # e.g. leaflet_scale, which PhytomerParameters() sets by operator=
        if ov.op == "uniform":
            cur.uniformDistribution(*ov.args)
        elif ov.op == "normal":
            cur.normalDistribution(*ov.args)
        elif ov.op == "weibull":
            cur.weibullDistribution(*ov.args)
        else:
            cur.discreteValues(ov.args)
        return
    if ov.kind == "rgb":
        setattr(obj, ov.field, make_RGBcolor(*ov.args))
    elif ov.kind == "texture_map" and ov.op == "texture":
        from .assets import ensure_leaf_texture
        setattr(obj, ov.field, {0: ensure_leaf_texture(species.name, ov.args, species.leaf_textures[ov.args])})
    elif ov.kind == "texture_map":
        setattr(obj, ov.field, dict(ov.args))
    elif ov.kind == "length_map":
        setattr(obj, ov.field, {k: f32(v) for k, v in ov.args.items()})
    else:
        setattr(obj, ov.field, ov.args)


def _compose(species, label, kind, uses):
    """One function pointer that runs the rules in order."""
    from ..plantarchitecture import hooks
    if uses == "none" or not uses:
        return None
    if len(uses) == 1 and uses[0].name == "library_function":
        return R._library_hook(uses[0].params["name"])
    fns = [R.make(u.name, kind, u.params, species=species) for u in uses]
    h = hooks.Hook(f"{species}:{label}:{kind}")

    def run(*args):
        for fn in fns:
            fn(*args)
    h.implement(run)
    return h


def initialize_species_shoots(pa, spec: SpeciesSpec):
    """The species' shoot initializer (see the module docstring)."""
    from .organs import organ_hook
    init, _ = _template_functions(spec.template)
    pa.shoot_types.clear()
    init(pa)
    template_types = dict(pa.shoot_types)
    for x in spec.drop_shoot_types:
        if x not in template_types:
            raise SpeciesError(f"{os.path.basename(spec.path)}: drop_shoot_types: the template '{spec.template}' has no "
                               f"shoot type '{x}' (it has: {', '.join(sorted(template_types))})")
    new = {}
    for st in spec.shoot_types:
        # ``from``: a shoot type defined earlier in the file, else the template's
        base = new.get(st.from_label, template_types.get(st.from_label))
        if base is None:
            raise SpeciesError(f"{os.path.basename(spec.path)}: shoot_types.{st.label}.from: neither the template "
                               f"'{spec.template}' (it has: {', '.join(sorted(template_types))}) nor an earlier shoot "
                               f"type of this file has the label '{st.from_label}'")
        sp = base.copy()                                                  # copy constructor: no draw
        for ov in st.overrides:
            apply_override(sp, ov, pa.generator, spec)
        if st.child_shoot_types is not None:
            if st.child_shoot_types:
                sp.defineChildShootTypes(list(st.child_shoot_types), list(st.child_shoot_types.values()))
            else:
                sp._raw_set("child_shoot_type_labels", [])
                sp._raw_set("child_shoot_type_probabilities", [])
        infl = sp.phytomer_parameters.inflorescence
        for organ, attr in (("flower", "flower_prototype_function"), ("fruit", "fruit_prototype_function")):
            h = organ_hook(spec.name, organ, getattr(st, organ))
            if h != "template":
                setattr(infl, attr, h)
        pp = sp.phytomer_parameters
        if st.creation != "template":
            pp.phytomer_creation_function = _compose(spec.name, st.label, "creation", st.creation)
        if st.callback != "template":
            pp.phytomer_callback_function = _compose(spec.name, st.label, "callback", st.callback)
        new[st.label] = sp
    final = {k: v for k, v in template_types.items() if k not in spec.drop_shoot_types and k not in new}
    final.update(new)
    for label, sp in final.items():
        for child in sp.child_shoot_type_labels:
            if child not in final:
                raise SpeciesError(f"{os.path.basename(spec.path)}: shoot type '{label}' grows child shoots of type "
                                   f"'{child}', which the species does not define (it has: {', '.join(sorted(final))})")
    pa.shoot_types.clear()
    for label in final:
        if label in new:
            pa.defineShootType(label, final[label])
        else:
            pa.shoot_types[label] = final[label]


_PHENO_ATTR = {"dormancy_break": "dd_to_dormancy_break", "flower_initiation": "dd_to_flower_initiation",
               "flower_opening": "dd_to_flower_opening", "fruit_set": "dd_to_fruit_set",
               "fruit_maturity": "dd_to_fruit_maturity", "dormancy": "dd_to_dormancy",
               "max_leaf_lifespan": "max_leaf_lifespan", "evergreen": "is_evergreen"}


def _phenology_args(b: BuildSpec, inst=None):
    out = []
    for k in PHENOLOGY_KEYS:
        if k in b.phenology:
            out.append(b.phenology[k])
        elif inst is not None:
            out.append(getattr(inst, _PHENO_ATTR[k]))
        else:
            out.append(False if k == "evergreen" else 0.0)
    return out


def build_species_plant(pa, spec: SpeciesSpec, base_position):
    """The species' plant builder (see the module docstring)."""
    from ..plantarchitecture.params import make_AxisRotation
    b = spec.build
    if not pa.shoot_types:
        initialize_species_shoots(pa, spec)
    if b.build_parameter_defaults:
        given = dict(getattr(pa, "current_build_parameters", {}) or {})
        pa.current_build_parameters = {**{k: f32(v) for k, v in b.build_parameter_defaults.items()}, **given}
    if b.builder == "template":
        _, template_build = _template_functions(spec.template)
        plantID = template_build(pa, base_position)
        inst = pa.plant_instances[plantID]
        if b.phenology:
            pa.setPlantPhenologicalThresholds(plantID, *_phenology_args(b, inst))
        if b.max_age is not None:
            inst.max_age = f32(b.max_age)
    else:
        bs = b.base_stem
        pos = np.asarray(base_position, f32) + np.asarray(b.base_offset, f32)
        plantID = pa.addPlantInstance(pos, 0)
        if bs["internode_length_max"] == "shoot_type":
            internode_length = pa.shoot_types[bs["shoot_type"]].internode_length_max.val()
        else:
            internode_length = f32(bs["internode_length_max"])
        ang = {}
        for k in ("roll", "yaw", "pitch"):           # GCC: make_AxisRotation's arguments right to left
            r = b.base_rotation.get(k, ("const", f32(0)))
            ang[k] = pa.context_ptr.randu(r[1], r[2]) if r[0] == "randu" else r[1]
        rotation = make_AxisRotation(ang["pitch"], ang["yaw"], ang["roll"])
        pa.addBaseStemShoot(plantID, bs["nodes"], rotation, bs["internode_radius"], internode_length,
                            bs["internode_length_scale_factor_fraction"], bs["leaf_scale_factor_fraction"],
                            bs["radius_taper"], bs["shoot_type"])
        pa.breakPlantDormancy(plantID)
        pa.setPlantPhenologicalThresholds(plantID, *_phenology_args(b))
        pa.plant_instances[plantID].max_age = f32(b.max_age)
    for use in b.rules:
        R.make(use.name, "build", use.params, species=spec.name)(pa, plantID, base_position)
    return plantID


def register(pa, spec_or_name, label: str | None = None) -> str:
    """Register a species as plant model ``label`` (default: its name) of ``pa``. Returns the label."""
    spec = spec_or_name if isinstance(spec_or_name, SpeciesSpec) else load_spec(spec_or_name)
    label = label or spec.name
    plant_type, beta = _template_registration(spec.template)
    md = spec.metadata
    plant_type = md.get("plant_type", plant_type)
    if "leaf_inclination_beta" in md:
        beta = md["leaf_inclination_beta"]
    pa.registerPlantModel(label, lambda: initialize_species_shoots(pa, spec),
                          lambda pos: build_species_plant(pa, spec, pos), plant_type,
                          tuple(beta) if beta else (0.0, 0.0))
    decl = getattr(pa, "plant_model_build_parameters", {}).get(spec.template)
    if decl is not None:
        pa.plant_model_build_parameters[label] = decl
    for k, v in spec.build.build_parameter_defaults.items():
        names = [d.name for d in (decl.structure_parameters if decl else [])]
        if k not in names:
            raise SpeciesError(f"{os.path.basename(spec.path)}: build.build_parameter_defaults.{k}: the template "
                               f"'{spec.template}' declares no build parameter '{k}' (it declares: "
                               f"{', '.join(names) or 'none'})")
        d = next(d for d in decl.structure_parameters if d.name == k)
        if not d.min_value <= f32(v) <= d.max_value or (d.is_integer and v != round(v)):
            raise SpeciesError(f"{os.path.basename(spec.path)}: build.build_parameter_defaults.{k}: {v} is not a "
                               f"valid value ([{d.min_value}, {d.max_value}]{', whole' if d.is_integer else ''})")
    return label


def register_if_species(pa, name: str) -> bool:
    """Called by ``loadPlantModelFromLibrary`` for a name the library does not have."""
    if find_species_file(name) is None:
        return False
    register(pa, name)
    return True


def validate(name_or_path: str) -> SpeciesSpec:
    """Full check of a species file: parse it, then register and load it into a scratch PlantArchitecture and build
    one plant at age 0 (catches template shoot-type labels, rule names, assets, child shoot types)."""
    from ..core.scene import Scene
    from ..plantarchitecture import PlantArchitecture
    spec = load_spec(name_or_path)
    sc = Scene()
    pa = PlantArchitecture(sc)
    register(pa, spec)
    pa.loadPlantModelFromLibrary(spec.name)
    if spec.build.builder == "generic" and spec.build.base_stem["shoot_type"] not in pa.shoot_types:
        raise SpeciesError(f"{os.path.basename(spec.path)}: build.base_stem.shoot_type: no shoot type "
                           f"'{spec.build.base_stem['shoot_type']}' (have: {', '.join(sorted(pa.shoot_types))})")
    pa.buildPlantInstanceFromLibrary((0.0, 0.0, 0.0), 0.0)
    return spec


def describe(spec: SpeciesSpec) -> str:
    return json.dumps(spec.summary())
