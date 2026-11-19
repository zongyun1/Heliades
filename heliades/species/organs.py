# SPDX-License-Identifier: LGPL-2.1-or-later
"""Flower and fruit prototypes a species file can choose (``flower:`` / ``fruit:`` of a shoot type).

A prototype function builds one organ at the origin, pointing along +x from its base (the convention of the Helios
OBJ assets: ``Cucumber/fruit_00.obj`` spans x in [0, 1]); the inflorescence code scales it by
``flower_prototype_scale`` / ``fruit_prototype_scale`` and places it. So a unit prototype is 1 long (fruit) or 1
across (flower) and the scale is the organ's size in metres.

Forms (one key each):

``template``
    keep the template's function (the default when the key is absent);
``none``
    no organ geometry (null function pointer);
``{library: <name>}``
    a ported Helios prototype function (``TomatoFruitPrototype``, ``CucumberFlowerPrototype``, ...);
``{obj: <file>, height: h}``
    one Helios OBJ asset (``loadOBJ(file, origin, height, nullrotation, black, "ZUP")``; height 0 keeps its size);
``{obj_variants: <pattern>, count: n}``
    one of ``n`` OBJ files picked with ``Context::randu(0, n - 1)`` (one draw), ``{index}`` in the pattern for the
    variant and, for flowers, ``{state}`` for ``open`` / ``closed`` (the cucumber's);
``{ellipsoid: {...}}``
    procedural fruit: a spheroid of revolution about +x, ``length`` 1 by ``diameter``, optionally pear-shaped
    (``bulge`` > 0 moves the widest point towards the tip), with a ``calyx`` cap of another colour at the base;
``{corolla: {...}}``
    procedural flower: ``petals`` petals radiating from a central cone (``center_color``), cupped forward by ``cup``;
    a closed flower is the same petals folded into a bud.

Procedural organs are triangles with vertex colours (no texture) wrapped in a polymesh, and draw nothing from the
Context stream. The hooks are cached by (species, organ, description) so identical descriptions share one function
pointer, and therefore one prototype cache, as a C++ function would.
"""
from __future__ import annotations

import json
import math

import numpy as np

f32 = np.float32

FRUIT_ELLIPSOID_KEYS = {
    "diameter": (float, None, 0.01, 5.0),
    "bulge": (float, 0.0, -0.9, 0.9),
    "color": (list, None, None, None),
    "calyx_color": (list, None, None, None),
    "calyx_fraction": (float, 0.0, 0.0, 0.9),
    "rings": (int, 12, 3, 64),
    "segments": (int, 12, 3, 64),
}
COROLLA_KEYS = {
    "petals": (int, 5, 3, 20),
    "petal_width": (float, 0.5, 0.05, 1.0),        # fraction of the angular sector
    "cup": (float, 0.15, 0.0, 1.0),                # forward (x) rise of the petal tips, in diameters
    "color": (list, None, None, None),
    "center_color": (list, None, None, None),
    "center_diameter": (float, 0.2, 0.0, 0.9),
    "center_length": (float, 0.25, 0.0, 2.0),
    "bud_length": (float, 0.6, 0.05, 2.0),         # closed flower: length of the folded bud, in diameters
}

_HOOKS: dict = {}


class OrganSpecError(ValueError):
    pass


def _rgb(v, where):
    if not (isinstance(v, list) and len(v) == 3 and all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in v)):
        raise OrganSpecError(f"{where}: expected three numbers in [0, 1], got {v!r}")
    return tuple(float(c) for c in v)


def _block(d, keys, where):
    if not isinstance(d, dict):
        raise OrganSpecError(f"{where}: expected a mapping")
    for k in d:
        if k not in keys:
            raise OrganSpecError(f"{where}: unknown key '{k}' (allowed: {', '.join(keys)})")
    out = {}
    for k, (t, default, lo, hi) in keys.items():
        v = d.get(k, default)
        if v is None:
            if k == "calyx_color":
                out[k] = None
                continue
            raise OrganSpecError(f"{where}: '{k}' is required")
        if t is float and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        if isinstance(v, bool) or not isinstance(v, t):
            raise OrganSpecError(f"{where}.{k}: expected {t.__name__}, got {v!r}")
        if lo is not None and not lo <= v <= hi:
            raise OrganSpecError(f"{where}.{k}: {v} is outside [{lo}, {hi}]")
        if t is list:
            v = _rgb(v, f"{where}.{k}")
        out[k] = v
    return out


def validate_organ(spec, organ: str, where: str):
    """Normalise an organ description; returns ``("template",)``, ``("none",)`` or ``(form, params)``."""
    if spec == "template" or spec is None:
        return ("template",)
    if spec == "none":
        return ("none",)
    if not isinstance(spec, dict):
        raise OrganSpecError(f"{where}: expected template, none or a mapping")
    forms = {"library", "obj", "obj_variants", "ellipsoid", "corolla"} | set(EXTRA_FORMS)
    keys = set(spec) - {"source", "note", "height", "count"}
    form = keys & forms
    if len(form) != 1 or keys - forms:
        raise OrganSpecError(f"{where}: give exactly one of {', '.join(sorted(forms))} (got {', '.join(sorted(spec))})")
    form = form.pop()
    extra = set(spec) - {form, "source", "note"} - ({"height"} if form == "obj" else set()) \
        - ({"count"} if form == "obj_variants" else set())
    if extra:
        raise OrganSpecError(f"{where}: unknown key(s) {', '.join(sorted(extra))} for {form}")
    if form == "library":
        if not isinstance(spec["library"], str):
            raise OrganSpecError(f"{where}.library: expected a function name")
        return ("library", {"name": spec["library"]})
    if form == "obj":
        h = spec.get("height", 0.0)
        if not isinstance(spec["obj"], str) or isinstance(h, bool) or not isinstance(h, (int, float)) or h < 0:
            raise OrganSpecError(f"{where}: obj needs a file name and a height >= 0")
        return ("obj", {"file": spec["obj"], "height": float(h)})
    if form == "obj_variants":
        n = spec.get("count")
        if not isinstance(spec["obj_variants"], str) or not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise OrganSpecError(f"{where}: obj_variants needs a file pattern and a count >= 1")
        if organ == "flower" and "{state}" not in spec["obj_variants"]:
            raise OrganSpecError(f"{where}: a flower pattern needs {{state}} (open / closed)")
        return ("obj_variants", {"pattern": spec["obj_variants"], "count": n})
    if form == "ellipsoid":
        if organ != "fruit":
            raise OrganSpecError(f"{where}: ellipsoid is a fruit shape")
        return ("ellipsoid", _block(spec["ellipsoid"], FRUIT_ELLIPSOID_KEYS, where + ".ellipsoid"))
    if form in EXTRA_FORMS:
        organs_ok, keys, _ = EXTRA_FORMS[form]
        if organ not in organs_ok:
            raise OrganSpecError(f"{where}: {form} is a {' / '.join(organs_ok)} shape")
        return (form, _block(spec[form], keys, where + "." + form))
    if organ != "flower":
        raise OrganSpecError(f"{where}: corolla is a flower shape")
    return ("corolla", _block(spec["corolla"], COROLLA_KEYS, where + ".corolla"))


# ---- geometry ----------------------------------------------------------------------------------------

def _mesh(ctx, verts, tris, colors):
    uu = [ctx.add_triangle(verts[a], verts[b], verts[c], colors[i]) for i, (a, b, c) in enumerate(tris)]
    return ctx.add_polymesh_object(uu)


def ellipsoid_mesh(p):
    """(vertices (n, 3) float32, triangles, per-triangle colours) of the fruit."""
    rings, seg = p["rings"], p["segments"]
    R = 0.5 * p["diameter"]
    verts = [(0.0, 0.0, 0.0)]
    xs = []
    for i in range(1, rings):
        s = i / rings                                   # 0 at the base, 1 at the tip
        theta = math.pi * s
        x = 0.5 * (1 - math.cos(theta))
        r = R * math.sin(theta) * (1 + p["bulge"] * (2 * x - 1))      # bulge > 0: wider towards the tip
        xs.append(x)
        for j in range(seg):
            a = 2 * math.pi * j / seg
            verts.append((x, r * math.cos(a), r * math.sin(a)))
    verts.append((1.0, 0.0, 0.0))
    tip = len(verts) - 1
    tris, cols = [], []
    body, calyx = p["color"], p["calyx_color"] or p["color"]

    def ring(i, j):
        return 1 + i * seg + (j % seg)

    def col(x):
        return calyx if x <= p["calyx_fraction"] else body
    for j in range(seg):
        tris.append((0, ring(0, j + 1), ring(0, j)))
        cols.append(col(0.0))
    for i in range(rings - 2):
        c = col(0.5 * (xs[i] + xs[i + 1]))
        for j in range(seg):
            tris.append((ring(i, j), ring(i, j + 1), ring(i + 1, j + 1)))
            tris.append((ring(i, j), ring(i + 1, j + 1), ring(i + 1, j)))
            cols += [c, c]
    for j in range(seg):
        tris.append((ring(rings - 2, j), ring(rings - 2, j + 1), tip))
        cols.append(body)
    return np.asarray(verts, f32), tris, cols


def corolla_mesh(p, is_open: bool):
    n = p["petals"]
    R = 0.5
    verts, tris, cols = [(0.0, 0.0, 0.0)], [], []
    half = math.pi / n * p["petal_width"]
    for k in range(n):
        a = 2 * math.pi * k / n
        if is_open:
            tip = (p["cup"], R * math.cos(a), R * math.sin(a))
            r_side, x_side = 0.55 * R, 0.4 * p["cup"]
        else:
            rb = 0.12 * R
            tip = (p["bud_length"], rb * 0.2 * math.cos(a), rb * 0.2 * math.sin(a))
            r_side, x_side = rb, 0.5 * p["bud_length"]
        left = (x_side, r_side * math.cos(a - half), r_side * math.sin(a - half))
        right = (x_side, r_side * math.cos(a + half), r_side * math.sin(a + half))
        b = len(verts)
        verts += [left, tip, right]
        tris += [(0, b, b + 1), (0, b + 1, b + 2)]
        cols += [p["color"], p["color"]]
    if is_open and p["center_diameter"] > 0 and p["center_length"] > 0:
        rc, m = 0.5 * p["center_diameter"], 6
        b = len(verts)
        for j in range(m):
            a = 2 * math.pi * j / m
            verts.append((0.0, rc * math.cos(a), rc * math.sin(a)))
        verts.append((p["center_length"], 0.0, 0.0))
        apex = len(verts) - 1
        for j in range(m):
            tris.append((b + j, b + (j + 1) % m, apex))
            cols.append(p["center_color"])
    return np.asarray(verts, f32), tris, cols


# ---- hooks -------------------------------------------------------------------------------------------

def _resolve_obj(name, species):
    from .assets import resolve_asset
    if not name.startswith("assets/") and "/" not in name and not name.endswith(".obj"):
        name = name + ".obj"
    if not name.startswith("assets/") and not __import__("os").path.isabs(name):
        try:
            return resolve_asset(name, species)
        except FileNotFoundError:
            return resolve_asset("assets/obj/" + name, species)
    return resolve_asset(name, species)


def _load_obj(ctx, path, height):
    uu = ctx.load_obj(path, (0.0, 0.0, 0.0), height=f32(height), rotation=(0.0, 0.0), color=(0.0, 0.0, 0.0),
                      upaxis="ZUP", silent=True)
    return ctx.add_polymesh_object(list(uu))


def organ_hook(species: str, organ: str, norm):
    """The function pointer (a :class:`~heliades.plantarchitecture.hooks.Hook`) for a normalised organ
    description, or ``"template"`` / ``None``."""
    from ..plantarchitecture import hooks
    form = norm[0]
    if form == "template":
        return "template"
    if form == "none":
        return None
    params = norm[1]
    if form == "library":
        from .rules import _library_hook
        return _library_hook(params["name"])
    key = (species, organ, form, json.dumps(params, sort_keys=True))
    if key in _HOOKS:
        return _HOOKS[key]
    h = hooks.Hook(f"{species}:{organ}:{form}")
    if form == "obj":
        path = _resolve_obj(params["file"], species)
        if organ == "fruit":
            h.implement(lambda ctx, subdivisions, build_parameters=None: _load_obj(ctx, path, params["height"]))
        else:
            h.implement(lambda ctx, subdivisions, is_open, build_parameters=None: _load_obj(ctx, path, params["height"]))
    elif form == "obj_variants":
        n, pat = params["count"], params["pattern"]
        if organ == "fruit":
            paths = [_resolve_obj(pat.format(index=i), species) for i in range(n)]

            def fruit(ctx, subdivisions, build_parameters=None):
                return _load_obj(ctx, paths[int(ctx.randu(0, n - 1))], 0.0)
            h.implement(fruit)
        else:
            paths = {s: [_resolve_obj(pat.format(index=i, state=s), species) for i in range(n)]
                     for s in ("open", "closed")}

            def flower(ctx, subdivisions, is_open, build_parameters=None):
                return _load_obj(ctx, paths["open" if is_open else "closed"][int(ctx.randu(0, n - 1))], 0.0)
            h.implement(flower)
    elif form in EXTRA_FORMS:
        V, T, C = EXTRA_FORMS[form][2](params)
        if organ == "fruit":
            h.implement(lambda ctx, subdivisions, build_parameters=None: _mesh(ctx, V, T, C))
        else:
            h.implement(lambda ctx, subdivisions, is_open, build_parameters=None: _mesh(ctx, V, T, C))
    elif form == "ellipsoid":
        V, T, C = ellipsoid_mesh(params)
        h.implement(lambda ctx, subdivisions, build_parameters=None: _mesh(ctx, V, T, C))
    else:
        meshes = {s: corolla_mesh(params, s) for s in (True, False)}
        h.implement(lambda ctx, subdivisions, is_open, build_parameters=None: _mesh(ctx, *meshes[bool(is_open)]))
    _HOOKS[key] = h
    return h


# ---- extra procedural organ shapes ----------------------------------------------------------------------------
# A shape registered here is a form of its own: ``EXTRA_FORMS[name] = (organs, keys, mesh_fn)`` with ``organs`` the
# organs it may be used for ("fruit", "flower"), ``keys`` a ``_block`` schema and ``mesh_fn(params)`` returning
# (vertices, triangles, per-triangle colours) of the unit organ (base at the origin, along +x). Procedural, no
# Context draw.

EXTRA_FORMS: dict = {}


# ---- panicle (S1, oat) ----------------------------------------------------------------------------------------

PANICLE_KEYS = {
    "whorls": (int, 5, 1, 12),                     # rachis nodes bearing branches
    "whorl_top": (float, 0.8, 0.1, 1.0),           # position of the highest whorl, fraction of the length
    "branches_base": (int, 4, 1, 12),              # branches at the basal whorl ...
    "branches_tip": (int, 1, 1, 12),               # ... and at the highest (linear, rounded)
    "branch_length": (float, 0.45, 0.02, 1.5),     # basal branch, fraction of the panicle length
    "branch_length_tip": (float, 0.12, 0.01, 1.5),
    "branch_angle": (float, 45.0, 0.0, 120.0),     # basal branch angle from the rachis (degrees)
    "branch_angle_tip": (float, 25.0, 0.0, 120.0),
    "droop": (float, 40.0, 0.0, 150.0),            # extra bend of every branch from its base to its tip (degrees)
    "spikelets_per_branch": (int, 5, 1, 40),       # on a basal branch; shorter branches carry proportionally fewer
    "spikelet_zone": (float, 0.6, 0.05, 1.0),      # distal fraction of a branch that bears spikelets
    "spikelet_length": (float, 0.1, 0.005, 0.5),   # fraction of the panicle length
    "spikelet_width": (float, 0.35, 0.05, 1.0),    # fraction of the spikelet length
    "spikelet_hang": (float, 160.0, 0.0, 180.0),   # angle of a spikelet from the rachis direction (180 = straight down)
    "pedicel_length": (float, 0.04, 0.0, 0.5),     # fraction of the panicle length
    "rachis_radius": (float, 0.006, 0.0005, 0.05),
    "branch_radius": (float, 0.0025, 0.0002, 0.05),
    "rachis_bend": (float, 15.0, 0.0, 120.0),      # nod of the rachis tip (degrees, total)
    "color": (list, None, None, None),             # spikelets (glumes)
    "rachis_color": (list, None, None, None),      # rachis, branches and pedicels
    "jitter": (float, 0.15, 0.0, 1.0),             # relative random variation of angles and lengths
    "seed": (int, 1, 0, 1_000_000),                # of the local generator (numpy, not the Context stream)
}


def _tube(verts, tris, cols, pts, radius, color, sides=3):
    """A polyline as a prism of ``sides`` faces per segment (open ends)."""
    pts = [np.asarray(p, float) for p in pts]
    rings = []
    for i, p in enumerate(pts):
        d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        d = d / (np.linalg.norm(d) or 1.0)
        a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, a)
        u /= np.linalg.norm(u)
        w = np.cross(d, u)
        b = len(verts)
        for k in range(sides):
            t = 2 * math.pi * k / sides
            verts.append(p + radius * (math.cos(t) * u + math.sin(t) * w))
        rings.append(b)
    for i in range(len(rings) - 1):
        a0, a1 = rings[i], rings[i + 1]
        for k in range(sides):
            k1 = (k + 1) % sides
            tris += [(a0 + k, a0 + k1, a1 + k1), (a0 + k, a1 + k1, a1 + k)]
            cols += [color, color]


def _spikelet(verts, tris, cols, base, axis, side, length, width, color):
    """Two crossed lanceolate glume planes (a diamond each, widest at 0.35 of the length)."""
    axis = axis / np.linalg.norm(axis)
    side = side - axis * np.dot(side, axis)
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    for s in (side, other):
        b = len(verts)
        mid = base + axis * (0.35 * length)
        verts += [base, mid + s * (0.5 * width), base + axis * length, mid - s * (0.5 * width)]
        tris += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
        cols += [color, color]


def panicle_mesh(p):
    """Open grass panicle along +x: a rachis (nodding by ``rachis_bend``) with ``whorls`` whorls of spreading branches
    that droop towards their tips, pendant spikelets on pedicels along the distal part of each branch, and a terminal
    spikelet. Lengths are fractions of the panicle length (the organ is 1 long)."""
    rng = np.random.default_rng(p["seed"])
    J = p["jitter"]

    def jit(x):
        return x * (1 + J * (rng.random() * 2 - 1))

    verts, tris, cols = [], [], []
    rc, sc = p["color"], p["rachis_color"]
    ex, ez = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    nseg = 8
    bend = math.radians(p["rachis_bend"])

    def rachis_point(t):                              # arc of total turn ``bend`` in the x-z plane
        if bend < 1e-6:
            return ex * t, ex
        R = 1.0 / bend
        a = bend * t
        return np.array([R * math.sin(a), 0.0, R * (1 - math.cos(a))]), np.array([math.cos(a), 0.0, math.sin(a)])

    _tube(verts, tris, cols, [rachis_point(i / nseg)[0] for i in range(nseg + 1)], p["rachis_radius"], sc, 4)
    L_s, W_s = p["spikelet_length"], p["spikelet_width"] * p["spikelet_length"]
    hang = math.radians(p["spikelet_hang"])
    n = p["whorls"]
    az0 = 0.0
    for w in range(n):
        f = w / (n - 1) if n > 1 else 0.0
        t = p["whorl_top"] * f
        P, T = rachis_point(t)
        nb = int(round(p["branches_base"] + (p["branches_tip"] - p["branches_base"]) * f))
        blen = p["branch_length"] + (p["branch_length_tip"] - p["branch_length"]) * f
        bang = math.radians(p["branch_angle"] + (p["branch_angle_tip"] - p["branch_angle"]) * f)
        az0 += 2.399963                                # golden angle between whorls
        N1 = np.cross(T, ez if abs(T[2]) < 0.9 else ex)
        N1 /= np.linalg.norm(N1)
        N2 = np.cross(T, N1)
        for b in range(nb):
            az = az0 + 2 * math.pi * b / nb + J * (rng.random() - 0.5)
            radial = math.cos(az) * N1 + math.sin(az) * N2
            L = jit(blen)
            a0 = jit(bang)
            droop = math.radians(jit(p["droop"]))
            pts, q = [P], P.copy()
            for k in range(6):
                th = a0 + droop * (k + 0.5) / 6
                q = q + (L / 6) * (math.cos(th) * T + math.sin(th) * radial)
                pts.append(q)
            _tube(verts, tris, cols, pts, p["branch_radius"], sc)
            ns = max(1, int(round(p["spikelets_per_branch"] * L / max(p["branch_length"], 1e-9))))
            for s in range(ns):
                u = 1 - p["spikelet_zone"] * (s / ns)     # from the branch tip inwards
                idx = u * 6
                k = min(5, int(idx))
                at = pts[k] + (pts[k + 1] - pts[k]) * (idx - k)
                h = jit(hang) if hang < math.pi else hang
                sdir = math.cos(h) * T + math.sin(h) * radial
                side = np.cross(sdir, T) if abs(np.dot(sdir, T)) < 0.99 else N1
                ped_end = at + sdir * p["pedicel_length"]
                if p["pedicel_length"] > 0:
                    _tube(verts, tris, cols, [at, ped_end], 0.6 * p["branch_radius"], sc)
                _spikelet(verts, tris, cols, ped_end, sdir, side, jit(L_s), W_s, rc)
    P, T = rachis_point(1.0)
    _spikelet(verts, tris, cols, P, T, N1 if n else np.array([0, 1.0, 0]), L_s, W_s, rc)
    return np.asarray(verts, f32), tris, cols


EXTRA_FORMS["panicle"] = (("fruit",), PANICLE_KEYS, panicle_mesh)
