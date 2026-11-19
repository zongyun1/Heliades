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
``{capitulum: {...}}``
    procedural composite head (Asteraceae), flower or fruit: a disc of florets 1 across facing +x on a rounded green
    receptacle, a ring of involucral bracts and ``rays`` ray florets; a closed flower is a bud of bracts folded over
    the disc (``bud_fraction`` of the diameter). The scale is therefore the *disc* diameter (rays extend beyond it).

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
    forms = {"library", "obj", "obj_variants", "ellipsoid", "corolla", "capitulum"}
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
    if form == "capitulum":
        return ("capitulum", _block(spec["capitulum"], CAPITULUM_KEYS, where + ".capitulum"))
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
    elif form == "ellipsoid":
        V, T, C = ellipsoid_mesh(params)
        h.implement(lambda ctx, subdivisions, build_parameters=None: _mesh(ctx, V, T, C))
    elif form == "capitulum":
        if organ == "fruit":
            V, T, C = capitulum_mesh(params, True)
            h.implement(lambda ctx, subdivisions, build_parameters=None: _mesh(ctx, V, T, C))
        else:
            cap = {s: capitulum_mesh(params, s) for s in (True, False)}
            h.implement(lambda ctx, subdivisions, is_open, build_parameters=None: _mesh(ctx, *cap[bool(is_open)]))
    else:
        meshes = {s: corolla_mesh(params, s) for s in (True, False)}
        h.implement(lambda ctx, subdivisions, is_open, build_parameters=None: _mesh(ctx, *meshes[bool(is_open)]))
    _HOOKS[key] = h
    return h


# ---- capitulum (composite head; added in S1 for the sunflower) -----------------------------------------

CAPITULUM_KEYS = {
    "disc_color": (list, [0.30, 0.18, 0.07], None, None),         # the disc florets
    "disc_center_color": (list, [0.36, 0.30, 0.10], None, None),  # the young (central) florets
    "disc_center_fraction": (float, 0.3, 0.0, 1.0),               # radius of the central zone / disc radius
    "dome": (float, 0.04, -0.5, 0.5),                             # forward rise of the disc centre, in diameters
    "thickness": (float, 0.12, 0.0, 1.0),                         # depth of the receptacle behind the disc
    "back_color": (list, [0.30, 0.42, 0.16], None, None),         # receptacle (back of the head)
    "rays": (int, 0, 0, 200),                                     # ray florets
    "ray_length": (float, 0.3, 0.0, 2.0),                         # in disc diameters
    "ray_width": (float, 0.7, 0.05, 1.0),                         # fraction of the angular sector
    "ray_angle": (float, 0.0, -90.0, 90.0),                       # degrees forward (+) / back (-) of the disc plane
    "ray_color": (list, [0.98, 0.78, 0.05], None, None),
    "bracts": (int, 0, 0, 200),                                   # involucral bracts (phyllaries)
    "bract_length": (float, 0.12, 0.0, 1.0),                      # in disc diameters
    "bract_color": (list, [0.26, 0.40, 0.14], None, None),
    "bud_fraction": (float, 0.45, 0.05, 1.0),                     # closed head (bud): diameter / open disc diameter
    "rings": (int, 6, 2, 64),
    "segments": (int, 24, 6, 256),
}


def capitulum_mesh(p, is_open: bool):
    """(vertices, triangles, per-triangle colours) of a composite head, base at the origin, facing +x.

    Open: receptacle (x from 0 to ``thickness``, radius 0 -> 0.5), disc (radius 0.5, centre raised by ``dome``),
    bracts pointing out and back from the rim, rays out from the rim tilted by ``ray_angle``. Closed (a bud): the
    receptacle with the bracts folded forward over it, all scaled by ``bud_fraction``."""
    seg, nr = p["segments"], p["rings"]
    th = p["thickness"]
    verts, tris, cols = [], [], []

    def add(v):
        verts.append(tuple(float(c) for c in v))
        return len(verts) - 1

    def circ(x, r, j):
        a = 2 * math.pi * j / seg
        return (x, r * math.cos(a), r * math.sin(a))

    s = 1.0 if is_open else p["bud_fraction"]
    R = 0.5 * s
    # receptacle: r = R sin(pi/2 u), x = th s (1 - cos(pi/2 u)), u from 0 (base) to 1 (rim)
    back = [add((0.0, 0.0, 0.0))]
    for i in range(1, nr + 1):
        u = i / nr
        back.append([add(circ(th * s * (1 - math.cos(0.5 * math.pi * u)), R * math.sin(0.5 * math.pi * u), j))
                     for j in range(seg)])
    bc = p["back_color"]
    for j in range(seg):
        tris.append((back[0], back[1][(j + 1) % seg], back[1][j]))
        cols.append(bc)
    for i in range(1, nr):
        for j in range(seg):
            a, b, c, d = back[i][j], back[i][(j + 1) % seg], back[i + 1][(j + 1) % seg], back[i + 1][j]
            tris += [(a, c, b), (a, d, c)]
            cols += [bc, bc]
    x_rim = th * s
    if is_open:
        # disc: rings from the rim to the centre
        dc, cc, cf = p["disc_color"], p["disc_center_color"], p["disc_center_fraction"]
        prev = back[nr]
        for i in range(nr - 1, 0, -1):
            r = R * i / nr
            x = x_rim + p["dome"] * (1 - (r / R) ** 2)
            ring = [add(circ(x, r, j)) for j in range(seg)]
            c = cc if (r + 0.5 * R / nr) / R <= cf else dc
            for j in range(seg):
                a, b, c2, d = prev[j], prev[(j + 1) % seg], ring[(j + 1) % seg], ring[j]
                tris += [(a, b, c2), (a, c2, d)]
                cols += [c, c]
            prev = ring
        centre = add((x_rim + p["dome"], 0.0, 0.0))
        c = cc if cf > 0 else dc
        for j in range(seg):
            tris.append((prev[j], prev[(j + 1) % seg], centre))
            cols.append(c)
    # bracts: open = out and back from just behind the rim; closed = forward, converging over the bud
    nb, bl = p["bracts"], p["bract_length"] * s
    for k in range(nb):
        a = 2 * math.pi * (k + 0.5 * (k % 2)) / nb
        half = math.pi / nb * 0.9
        x0, r0 = 0.85 * x_rim, 0.97 * R
        if is_open:
            tip = (x0 - 0.55 * bl, (r0 + 0.8 * bl) * math.cos(a), (r0 + 0.8 * bl) * math.sin(a))
        else:
            reach = max(0.0, r0 - 1.6 * bl) + 0.08 * R
            tip = (x_rim + 1.1 * bl, reach * math.cos(a), reach * math.sin(a))
        lv = add((x0, r0 * math.cos(a - half), r0 * math.sin(a - half)))
        rv = add((x0, r0 * math.cos(a + half), r0 * math.sin(a + half)))
        tris.append((lv, rv, add(tip)))
        cols.append(p["bract_color"])
    if not is_open:
        # the top of the bud under the bract tips: a green cap
        cap = add((x_rim + bl, 0.0, 0.0))
        ring = back[nr]
        for j in range(seg):
            tris.append((ring[j], ring[(j + 1) % seg], cap))
            cols.append(p["bract_color"])
        return np.asarray(verts, f32), tris, cols
    # rays: a lanceolate quad + a tip triangle each
    n, L = p["rays"], p["ray_length"]
    tilt = math.radians(p["ray_angle"])
    r0, x0 = 0.95 * R, x_rim + 0.25 * p["dome"]
    for k in range(n):
        a = 2 * math.pi * (k + 0.25) / n
        half = math.pi / n * p["ray_width"]

        def at(t, w):
            rr = r0 + t * L * math.cos(tilt)
            xx = x0 + t * L * math.sin(tilt)
            return (xx, rr * math.cos(a + w), rr * math.sin(a + w))
        wm = 1.2 * half * r0 / (r0 + 0.55 * L * math.cos(tilt))
        b0, b1 = add(at(0.0, -0.6 * half)), add(at(0.0, 0.6 * half))
        m0, m1 = add(at(0.55, -wm)), add(at(0.55, wm))
        tip = add(at(1.0, 0.0))
        tris += [(b0, b1, m1), (b0, m1, m0), (m0, m1, tip)]
        cols += [p["ray_color"]] * 3
    return np.asarray(verts, f32), tris, cols
