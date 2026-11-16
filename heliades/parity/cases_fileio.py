# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for Context::loadOBJ and polymesh topology (W2-A).

Real assets: every plantarchitecture OBJ (the tomato flower and fruit through the exact
PlantArchitecture call, textured leaves with map_Kd / map_d, authored normals in OliveFlower_open,
the Cucumber set). Synthetic OBJs written at runtime cover what the assets do not: n-gons up to 32
sides, every face-reference form, negative indices, degenerate and sub-threshold triangles, faces
before any usemtl, unknown materials, duplicate newmtl blocks, CRLF and trailing whitespace (which
make Helios re-read the last face reference), odd number formats, no final newline, relative texture
paths, and repeated loads into one Context (material label collisions).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from .cases import case
from .cases_texture import write_png

HELIOS = "/home/yun/Desktop/plants/third_party/pyhelios_src/helios-core"
OBJ_DIR = f"{HELIOS}/plugins/plantarchitecture/assets/obj/"
TOMATO = [OBJ_DIR + "TomatoFlower.obj", OBJ_DIR + "TomatoFruit.obj"]
_BROKEN = {"RedbudLeaf.obj", "SugarBeetLeaf.obj"}     # Helios itself rejects these (see notes/W2-A.md)
ASSETS = sorted(OBJ_DIR + f for f in os.listdir(OBJ_DIR) if f.endswith(".obj") and f not in _BROKEN) + \
    sorted(OBJ_DIR + "Cucumber/" + f for f in os.listdir(OBJ_DIR + "Cucumber") if f.endswith(".obj"))
LEAF_PNG = f"{HELIOS}/plugins/plantarchitecture/assets/textures/TomatoLeaf.png"


def _rot(r):
    return (0.0, 0.0) if r.random() < 0.4 else (float(r.uniform(-1.5, 1.5)), float(r.uniform(0, 6.3)))


def _random_load(B, r, f):
    """One loadOBJ call through a randomly chosen overload."""
    k = r.random()
    if k < 0.15:
        return B.load_obj(f, overload="simple")
    origin, color, rot = r.uniform(-2, 2, 3), r.random(3), _rot(r)
    if k < 0.4:
        return B.load_obj(f, origin, float(r.choice([0.0, 0.75, r.uniform(0.01, 3)])), rot, color, overload="height")
    if k < 0.7:
        return B.load_obj(f, origin, float(r.choice([0.0, 0.75, r.uniform(0.01, 3)])), rot, color,
                          upaxis=str(r.choice(["XUP", "YUP", "ZUP"])), overload="height_up")
    s = r.uniform(0.05, 3, 3)
    s[r.random(3) < 0.2] = 0.0                                     # zero factors fall back to 1
    return B.load_obj(f, origin, rotation=rot, color=color, upaxis=str(r.choice(["XUP", "YUP", "ZUP"])), scale=s,
                      overload="scale")


# ---- tomato: the PlantArchitecture call ----------------------------------------------------

def _tomato(B, seed):
    for f in TOMATO:
        if seed == 0:   # AssetLibrary TomatoFlowerPrototype / TomatoFruitPrototype exactly
            B.load_obj(f, (0.0, 0.0, 0.0), 0.75, (0.0, 0.0), (0.0, 0.0, 0.0), upaxis="ZUP", overload="height_up")
        else:
            _random_load(B, np.random.default_rng(seed * 31 + len(f)), f)


@case("fileio.obj.tomato", fuzz=6)
def obj_tomato(B, seed):
    _tomato(B, seed)
    return B.snapshot()


@case("fileio.obj.tomato_topology", tier="exact", fuzz=6)
def obj_tomato_topology(B, seed):
    _tomato(B, seed)
    return B.obj_queries()


# ---- every plantarchitecture asset ---------------------------------------------------------

def _assets(B, seed):
    r = np.random.default_rng(1000 + seed)
    chosen = [ASSETS[i] for i in range(len(ASSETS)) if i % 6 == seed % 6]   # each asset once per 6 seeds
    for f in chosen:
        _random_load(B, r, f)
    _random_load(B, r, chosen[0])                                   # a second load: material names collide


@case("fileio.obj.assets", fuzz=6)
def obj_assets(B, seed):
    _assets(B, seed)
    return B.snapshot()


@case("fileio.obj.assets_topology", tier="exact", fuzz=6)
def obj_assets_topology(B, seed):
    _assets(B, seed)
    return B.obj_queries()


# ---- synthetic OBJs ------------------------------------------------------------------------

def double_rounding_trap(x) -> str:
    """A decimal just above the midpoint of two adjacent float32s near x (the lower one even). strtod
    rounds it to the midpoint double, (float) then ties to the even, *lower* float; a direct
    correctly-rounded decimal->float32 conversion gives the upper one. parse_float does the former."""
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    a = np.float32(x)
    if a.view(np.int32) & 1:
        a = np.nextafter(a, np.float32(-np.inf) if a > 0 else np.float32(np.inf))
    b = np.nextafter(a, np.float32(np.inf) if a >= 0 else np.float32(-np.inf))
    m = Decimal(float(a)) + (Decimal(float(b)) - Decimal(float(a))) / 2
    return f"{m + abs(m) * Decimal('1e-25') * (1 if b > a else -1):.40e}"


def _num(r, x):
    k = r.random()
    if k < 0.05:
        return double_rounding_trap(x)
    if k < 0.6:
        return f"{x:.6f}"
    if k < 0.75:
        return f"{x:.9g}"
    if k < 0.85:
        return f"{x:e}"
    if k < 0.9:
        return float(x).hex()                                       # strtod reads hex floats
    if k < 0.95:
        return f"{x:+.3f}"
    return repr(float(np.float32(x)))


def synthetic_obj(seed: int) -> str:
    """Write a random OBJ + MTL (+ a PNG) into a per-seed temp dir; content depends only on the seed."""
    r = np.random.default_rng(5000 + seed)
    d = os.path.join(tempfile.gettempdir(), f"heliades_w2a_obj_{seed}")
    os.makedirs(os.path.join(d, "tex"), exist_ok=True)
    H, W = 16, 24
    a = r.choice(np.array([0, 120, 249, 250, 255]), (H, W))
    write_png(os.path.join(d, "tex", "a.png"), np.dstack([r.integers(0, 256, (H, W, 3)), a]), 8, 6)

    def c3():
        return " ".join(f"{x:.6f}" for x in r.random(3))

    mtl = ["# synthetic", "", "newmtl red", "Ns 10", f"Ka {c3()}", f"Kd {c3()}", f"Ks {c3()}", "d 1.0", "illum 2",
           "newmtl texKd", f"Kd {c3()}", "map_Kd tex/a.png",
           "newmtl texKdD", f"Kd {c3()}", f"map_Kd {LEAF_PNG}", "map_d tex/a.png",
           "newmtl onlyD", f"Kd {c3()}", "map_d -o 0 0 tex/a.png",
           "newmtl my mat ", f"Kd {c3()} 0.5",
           "newmtl red", f"Kd {c3()}",                               # duplicate block: same entry, new Kd
           "newmtl plain"]
    with open(os.path.join(d, "m.mtl"), "w") as fh:
        fh.write("\n".join(mtl) + "\n")

    nv = int(r.integers(40, 160))
    has_n = r.random() < 0.5
    out = ["# synthetic OBJ", "mtllib m.mtl", "o first"]
    centre = r.normal(0, 1, 3)
    P = centre + r.normal(0, 0.5, (nv, 3))
    P[nv - 3:] = P[nv - 4] + r.normal(0, 3e-6, (3, 3))              # a sub-threshold triangle's corners
    for k, p in enumerate(P):
        if k == nv // 2:
            out.append("o  second object ")
        out.append("v " + " ".join(_num(r, x) for x in p) + (" 0.1 0.2 0.3" if r.random() < 0.1 else ""))
    nt = int(r.integers(10, 60))
    for _ in range(nt):
        out.append("vt " + " ".join(_num(r, x) for x in r.uniform(-0.2, 1.3, 2)) + (" 0" if r.random() < 0.2 else ""))
    nn = int(r.integers(5, 40)) if has_n else 0
    for _ in range(nn):
        n = r.normal(size=3)
        if r.random() < 0.8:
            n /= np.linalg.norm(n)
        out.append("vn " + " ".join(_num(r, x) for x in n))
    out += ["g group1", "s 1"]
    mats = ["red", "texKd", "texKdD", "onlyD", "my mat", "plain", "unknown_mat"]
    textured = {"texKd", "texKdD", "onlyD"}
    current = None
    n_faces = int(r.integers(30, 90))
    for fi in range(n_faces):
        if fi > 3 and r.random() < 0.15:
            current = str(r.choice(mats))
            out.append(f"usemtl {current}" + ("\r" if r.random() < 0.2 else ""))
        n = int(r.choice([3, 3, 3, 4, 4, 5, 6, 7, 15, 32]))
        if fi == n_faces - 1:
            idx = list(range(nv - 2, nv + 1))                        # the tiny triangle
            n = 3
        else:
            idx = list(r.choice(np.arange(1, nv - 3), size=n, replace=n > nv - 4))
            if r.random() < 0.1:
                idx[int(r.integers(1, n))] = idx[0]                  # coincident corners: skipped triangles
        tex = current in textured
        form = int(r.integers(0, 4))
        if tex:
            form = 1 if (not has_n or r.random() < 0.5) else 3
        elif not has_n and form >= 2:
            form = int(r.integers(0, 2))
        refs = []
        for v in idx:
            vs = str(v) if r.random() < 0.8 else str(v - nv - 1)     # negative = relative to the end
            t = int(r.integers(1, nt + 1))
            ts = str(t) if r.random() < 0.8 else str(t - nt - 1)
            if form == 0:
                refs.append(vs)
            elif form == 1:
                refs.append(f"{vs}/{ts}")
            else:
                nm = int(r.integers(1, nn + 1))
                if form == 2:
                    refs.append(f"{vs}//{nm}" if r.random() < 0.9 else vs)   # some corners without a normal
                else:
                    refs.append(f"{vs}/{ts}/{nm}")
        tail = str(r.choice(["", "", "", " ", "\r", " \t"]))       # trailing whitespace re-reads the last ref
        out.append("f " + " ".join(refs) + tail)
    path = os.path.join(d, "synthetic.obj")
    with open(path, "w", newline="") as fh:
        fh.write("\n".join(out) + ("\n" if r.random() < 0.7 else ""))
    return path


def _synthetic(B, seed):
    r = np.random.default_rng(seed)
    f = synthetic_obj(seed)
    _random_load(B, r, f)
    _random_load(B, r, f)                                           # material labels collide -> "red_1" ...


@case("fileio.obj.synthetic", fuzz=10)
def obj_synthetic(B, seed):
    _synthetic(B, seed)
    return B.snapshot()


@case("fileio.obj.synthetic_topology", tier="exact", fuzz=10)
def obj_synthetic_topology(B, seed):
    _synthetic(B, seed)
    return B.obj_queries()


# ---- errors and stream corner cases --------------------------------------------------------

_ERROR_FILES = {
    "ok_o_at_eof.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\no",                  # 'o' with no newline: name "o"
    "ok_crlf.obj": "v 0 0 0\r\nv 1 0 0\r\nv 0 1 0\r\nv 1 1 0\r\nf 1 2 3 4\r\n",
    "ok_comments.obj": "#c\n# c\nv 0 0 0 #x\nv 1 0 0\nv 0 1 0\n#\nf 1 2 3\n",
    "bad_index.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n",
    "bad_zero.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n",
    "bad_negative.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -1 -2 -4\n",
    "bad_fields.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1/1 2 3\n",
    "bad_int.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 x\n",
    "bad_overflow.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99999999999\n",
    "bad_float.obj": "v 0 0 zz\n",
    "bad_short.obj": "v 0 0\n",
    "bad_range.obj": "v 0 0 1e999\n",
    "bad_uv.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1 2/2 3/1\n",
    "bad_vn.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//2 3//1\n",
    "bad_f_at_eof.obj": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf",                           # getline fails: parses "f"
    "bad_mtl_missing.obj": "mtllib nothere.mtl\nv 0 0 0\n",
    "bad_tex_no_vt.obj": "mtllib e.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl t\nf 1 2 3\n",
    "bad_kd.obj": "mtllib k.mtl\nv 0 0 0\n",
    "bad_reserved.obj": "mtllib r.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl __x\nf 1 2 3\n",
    "bad_ext.txt": "v 0 0 0\n",
    "ok_after_errors.obj": "mtllib e.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl t\nf 1/1 2/2 3/3\n"
                           "usemtl __x\nf 1 3 2\n",
}
_ERROR_MTL = {"e.mtl": "newmtl t\nKd 1 0 0\nmap_Kd tex.png\n", "k.mtl": "newmtl k\nKd 1 0\n",
              "r.mtl": "newmtl __x\nKd 1 0 0\n"}


@case("fileio.obj.errors", tier="exact", fuzz=1)
def obj_errors(B, seed):
    """Which files raise (1) or load (0, then the UUID count), then the resulting Context."""
    d = os.path.join(tempfile.gettempdir(), "heliades_w2a_errors")
    os.makedirs(d, exist_ok=True)
    write_png(os.path.join(d, "tex.png"), np.full((4, 4, 4), 255), 8, 6)
    for n, s in {**_ERROR_FILES, **_ERROR_MTL}.items():
        with open(os.path.join(d, n), "w", newline="") as fh:
            fh.write(s)
    out = []
    for n in _ERROR_FILES:
        try:
            u = B.load_obj(os.path.join(d, n), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0), (0.1, 0.2, 0.3), upaxis="ZUP",
                           overload="height_up")
            out += [0.0, float(len(u))]
        except Exception:
            out += [1.0, 0.0]
    try:
        B.load_obj(os.path.join(d, "ok_crlf.obj"), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0), (0.1, 0.2, 0.3), upaxis="UP",
                   overload="height_up")
        out.append(0.0)
    except Exception:
        out.append(1.0)
    return out + B.obj_queries()


# ---- polymesh topology: normals, transforms, copies, setPolymeshObjectTopology ---------------

_NORMAL_ASSETS = TOMATO + [OBJ_DIR + f for f in ("CapsicumFruit_red.obj", "StrawberryFruit.obj", "OliveFlower_open.obj",
                                                  "AlmondLeaf.obj", "WalnutHull.obj")]


@case("fileio.polymesh.normals", tier="exact", fuzz=6)
def polymesh_normals(B, seed):
    """computePolymeshObjectVertexNormals at several crease angles (vertex splitting), on meshes moved by
    object transforms and on copies (the copy's face set is remapped onto its new primitives)."""
    r = np.random.default_rng(77 + seed)
    oids = []
    for f in _NORMAL_ASSETS[seed % 3::3] + [_NORMAL_ASSETS[seed % len(_NORMAL_ASSETS)]]:
        before = set(B.object_ids())
        _random_load(B, r, f)
        oids += sorted(set(B.object_ids()) - before)
    for o in list(oids):
        k = r.random()
        if k < 0.3:
            B.translate_object(o, r.uniform(-1, 1, 3))
        elif k < 0.6:
            B.rotate_object(o, float(r.uniform(-3, 3)), str(r.choice(["x", "y", "z"])))
        elif k < 0.8:
            B.scale_object(o, r.uniform(0.2, 3, 3))
        if r.random() < 0.5:
            oids.append(B.copy_object(o))
            B.rotate_object(oids[-1], float(r.uniform(-3, 3)), r.normal(size=3), r.normal(size=3))
    for o in oids:
        if r.random() < 0.7:
            B.compute_polymesh_normals(o, float(r.choice([0.0, 1.0, 20.0, 30.0, 45.5, 90.0, 135.0, 179.9, 180.0])))
    return B.obj_queries()


@case("fileio.polymesh.set_topology", tier="exact", fuzz=8)
def polymesh_set_topology(B, seed):
    """addPolymeshObject on loose triangles, then a face set attached through setPolymeshObjectTopology
    (with and without normals/UVs, after a translation so the local-frame inverse is exercised)."""
    r = np.random.default_rng(300 + seed)
    for _ in range(3):
        nv = int(r.integers(4, 30))
        P = r.normal(0, 1, (nv + 1, 3))                              # the last vertex is unreferenced
        F = [r.choice(nv, 3, replace=False) for _ in range(int(r.integers(2, 25)))]
        if r.random() < 0.3:
            F.append(F[0][[0, 2, 1]])                                # opposite orientation: a crease of 180
        uu = [B.add_triangle(P[a], P[b], P[c], r.random(3)) for a, b, c in F]
        o = B.add_polymesh(uu)
        if r.random() < 0.5:
            B.translate_object(o, r.uniform(-2, 2, 3))
        if r.random() < 0.5:
            B.scale_object(o, r.uniform(0.3, 2, 3))
        k = r.random()
        normals = uv = None
        source = 0
        if k < 0.4:
            normals, source = r.normal(size=(nv + 1, 3)), int(r.choice([1, 2]))
        if r.random() < 0.5:
            uv = r.random((nv + 1, 2))
        B.set_polymesh_topology(o, P, F, uu, normals, uv, source)
        if r.random() < 0.6:
            B.compute_polymesh_normals(o, float(r.uniform(0, 180)))
    return B.obj_queries()
