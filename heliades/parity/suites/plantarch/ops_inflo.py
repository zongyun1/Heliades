# SPDX-License-Identifier: LGPL-2.1-or-later
"""Inflorescence operations for both backends (W3-D).

Growth (``advanceTime``, W3-C) is not merged, so no Heliades plant reaches flowering age. The inflorescence
geometry is therefore checked in isolation, with every input taken from the oracle:

* ``inflo_prototypes``: the three hidden inflorescence prototypes (closed flower, open flower, fruit). Helios:
  build a tomato to age 100..120 (the first inflorescence appears between 75 and 100 days) and read the hidden
  objects created after the leaf blades (objects > 37). Heliades: an age-0 build, then
  ``ensureInflorescencePrototypesInitialized`` on the first phytomer's parameters, exactly as the first peduncle
  triggers it.
* ``inflo_forced``: an **oracle trace driver** (``_DRIVER_SRC``, compiled once against the pinned
  ``libhelios.so``; Helios itself is not modified). It builds the tomato with ``buildPlantInstanceFromLibrary``,
  and for every floral bud carrying flowers or fruit it dumps, at full precision, what
  ``Phytomer::setFloralBudState`` / ``updateInflorescence`` read (bud base and rotation, the phytomer's internode
  nodes and radii, petiole nodes, ``inflorescence_bending_axis``, shoot index, every RandomParameter with its
  cache state, colours, storage sizes). It then forces the bud DEAD -> FLOWER_CLOSED -> FLOWER_OPEN -> FRUITING,
  reseeding the Context with a known seed before each step, then grows the fruit (``setInflorescenceScaleFraction``
  with the advanceTime formula, tc = 2..32) and calls ``updatePeduncleRadii``; on the first bud of each plant also a
  culm-matched variant (``shoot_index.z = x + 1``: reconciled radius, blended colours, then the culm radii x1.5 and
  x2 with ``updatePeduncleRadii`` -> path-dependent ``setTubeRadii``); after every step it dumps the engine
  state, the peduncle tube (nodes, radii, triangles, colours, material and object labels), every flower / fruit
  object (transform, triangles, colours, labels), ``inflorescence_rotation`` / bases / base scales and the
  phytomer's stored peduncle values.
  Heliades rebuilds a stand-in phytomer from the dumped inputs and replays the same sequence through
  ``heliades.plantarchitecture.inflorescence`` (forced states: W3-C's state machine is not involved).

The driver is a ``PlantArchitectureTestHelper`` (the friend class Helios declares for its own self-tests); the
RandomParameter caches are private, so it includes the header with ``private``/``protected`` defined to ``public``
after every standard header (layout unchanged with GCC). Floats are written as IEEE bit patterns.
"""
from __future__ import annotations

import functools
import hashlib
import os
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np

from ...ops import op

f32 = np.float32

HELIOS_SRC = "/home/yun/Desktop/plants/third_party/pyhelios_src"

_DRIVER_SRC = r'''// Oracle trace driver for the W3-D inflorescence parity cases (heliades/parity/ops_plantarch_inflo.py).
// Builds a tomato with the pinned libhelios.so, then for every floral bud with flower/fruit geometry
// dumps the inputs of Phytomer::setFloralBudState / updateInflorescence and forces the bud through
// DEAD -> FLOWER_CLOSED -> FLOWER_OPEN -> FRUITING (reseeding the Context before each), then fruit
// growth and updatePeduncleRadii, dumping the geometry after each step. Floats are printed as their
// IEEE-754 bit patterns (hex). Read-only use of Helios: no Helios source is modified.
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <omp.h>
#define private public
#define protected public
#include "PlantArchitecture.h"
#undef private
#undef protected

using namespace helios;

static std::ostream *O;

static void F(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    char b[16];
    std::snprintf(b, sizeof b, " %08x", u);
    *O << b;
}
static void V3(const vec3 &v) { F(v.x); F(v.y); F(v.z); }

static void engine(Context &ctx, const char *tag) {
    std::ostringstream s;
    s << *ctx.getRandomGenerator();
    *O << tag << " " << s.str() << "\n";
}

static void rp(const char *name, const RandomParameter_float &p) {
    *O << "RPF " << name << " " << p.distribution << " " << p.distribution_parameters.size();
    for (float x : p.distribution_parameters) F(x);
    F(p.constval);
    *O << " " << int(p.sampled) << "\n";
}
static void rpi(const char *name, const RandomParameter_int &p) {
    *O << "RPI " << name << " " << p.distribution << " " << p.distribution_parameters.size();
    for (int x : p.distribution_parameters) *O << " " << x;
    *O << " " << p.constval << " " << int(p.sampled) << "\n";
}

static void prims(Context &ctx, uint objID) {
    // one line of vertex+colour bits (hex, no separators), then run-length material and object labels
    std::vector<uint> UUIDs = ctx.getObjectPrimitiveUUIDs(objID);
    std::string hex;
    hex.reserve(UUIDs.size() * 12 * 8);
    std::vector<std::pair<std::string, int>> mats, labels;
    auto push = [](std::vector<std::pair<std::string, int>> &runs, const std::string &s) {
        if (!runs.empty() && runs.back().first == s) runs.back().second++;
        else runs.emplace_back(s, 1);
    };
    char b[16];
    auto H = [&](float v) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        std::snprintf(b, sizeof b, "%08x", u);
        hex += b;
    };
    for (uint u : UUIDs) {
        std::vector<vec3> v = ctx.getPrimitiveVertices(u);
        if (v.size() != 3) {
            std::cerr << "non-triangle primitive " << u << "\n";
            std::exit(3);
        }
        for (auto &x : v) { H(x.x); H(x.y); H(x.z); }
        RGBcolor c = ctx.getPrimitiveColor(u);
        H(c.r); H(c.g); H(c.b);
        std::string label = "-";
        if (ctx.doesPrimitiveDataExist(u, "object_label")) ctx.getPrimitiveData(u, "object_label", label);
        push(mats, ctx.getPrimitiveMaterialLabel(u));
        push(labels, label);
    }
    *O << "PV " << UUIDs.size() << " " << (hex.empty() ? "-" : hex) << "\n";
    *O << "PM " << mats.size();
    for (auto &r : mats) *O << " " << r.first << " " << r.second;
    *O << "\n";
    *O << "PL " << labels.size();
    for (auto &r : labels) *O << " " << r.first << " " << r.second;
    *O << "\n";
}

static void geometry(Context &ctx, Phytomer &ph, FloralBud &fbud, uint entry, uint pos, const char *tag, bool with_prims = true) {
    *O << "G " << tag << " " << int(fbud.state) << " " << fbud.peduncle_objIDs.size() << " " << fbud.inflorescence_objIDs.size() << "\n";
    *O << "FS";
    F(fbud.current_fruit_scale_factor); F(fbud.time_counter);
    *O << "\n";
    for (uint objID : fbud.peduncle_objIDs) {
        *O << "PED " << objID << " " << ctx.getTubeObjectNodeCount(objID);
        for (auto &n : ctx.getTubeObjectNodes(objID)) V3(n);
        for (float r : ctx.getTubeObjectNodeRadii(objID)) F(r);
        *O << "\n";
        if (with_prims) prims(ctx, objID);
    }
    for (size_t i = 0; i < fbud.inflorescence_objIDs.size(); i++) {
        uint objID = fbud.inflorescence_objIDs[i];
        float T[16];
        ctx.getObjectTransformationMatrix(objID, T);
        *O << "OBJ " << objID;
        for (float t : T) F(t);
        const AxisRotation &r = fbud.inflorescence_rotation.at(i);
        F(r.pitch); F(r.yaw); F(r.roll); F(r.azimuth); V3(r.peduncle_axis); F(r.gravity_fraction);
        V3(fbud.inflorescence_bases.at(i));
        F(fbud.inflorescence_base_scales.at(i));
        *O << "\n";
        if (with_prims) prims(ctx, objID);
    }
    *O << "PSTORE";
    if (entry < ph.peduncle_vertices.size() && pos < ph.peduncle_vertices[entry].size()) {
        *O << " " << ph.peduncle_vertices[entry][pos].size();
        for (auto &v : ph.peduncle_vertices[entry][pos]) V3(v);
        *O << " " << ph.peduncle_radii[entry][pos].size();
        for (float r : ph.peduncle_radii[entry][pos]) F(r);
        F(ph.peduncle_length[entry][pos]); F(ph.peduncle_radius[entry][pos]); F(ph.peduncle_pitch[entry][pos]);
        F(ph.peduncle_curvature[entry][pos]); F(ph.peduncle_roll[entry][pos]); F(ph.peduncle_yaw[entry][pos]);
    } else {
        *O << " 0 0";
    }
    *O << "\n";
}

static void inputs(Phytomer &ph, FloralBud &fbud) {
    *O << "BASE";
    V3(fbud.base_position); F(fbud.base_rotation.pitch); F(fbud.base_rotation.yaw); F(fbud.base_rotation.roll);
    F(fbud.current_fruit_scale_factor); F(fbud.dd_to_inflorescence_maturity); F(fbud.time_counter); F(fbud.age);
    *O << " " << fbud.parent_index << " " << fbud.bud_index << " " << int(fbud.isterminal) << "\n";
    *O << "SIDX " << ph.shoot_index.x << " " << ph.shoot_index.y << " " << ph.shoot_index.z << " " << ph.rank << "\n";
    std::vector<vec3> nodes = ph.getInternodeNodePositions();
    *O << "INODES " << nodes.size();
    for (auto &n : nodes) V3(n);
    *O << "\n";
    const auto &radii = ph.parent_shoot_ptr->shoot_internode_radii.at(ph.shoot_index.x);
    *O << "IRADII " << radii.size();
    for (float r : radii) F(r);
    *O << "\n";
    *O << "PETIOLES " << ph.petiole_vertices.size() << "\n";
    for (auto &pv : ph.petiole_vertices) {
        *O << "PET " << pv.size();
        for (auto &v : pv) V3(v);
        *O << "\n";
    }
    *O << "BEND";
    V3(ph.inflorescence_bending_axis);
    *O << "\n";
    *O << "SHOOT " << ph.parent_shoot_ptr->shoot_type_label << " " << int(ph.parent_shoot_ptr->shoot_parameters.phytomer_parameters.internode.image_texture.empty()) << "\n";
    const PhytomerParameters &pp = ph.phytomer_parameters;
    rp("peduncle.length", pp.peduncle.length);
    rp("peduncle.radius", pp.peduncle.radius);
    rp("peduncle.pitch", pp.peduncle.pitch);
    rp("peduncle.roll", pp.peduncle.roll);
    rp("peduncle.yaw", pp.peduncle.yaw);
    rp("peduncle.curvature", pp.peduncle.curvature);
    rpi("inflorescence.flowers_per_peduncle", pp.inflorescence.flowers_per_peduncle);
    rp("inflorescence.flower_offset", pp.inflorescence.flower_offset);
    rp("inflorescence.pitch", pp.inflorescence.pitch);
    rp("inflorescence.roll", pp.inflorescence.roll);
    rp("inflorescence.flower_prototype_scale", pp.inflorescence.flower_prototype_scale);
    rp("inflorescence.fruit_prototype_scale", pp.inflorescence.fruit_prototype_scale);
    rp("inflorescence.fruit_gravity_factor_fraction", pp.inflorescence.fruit_gravity_factor_fraction);
    rp("internode.phyllotactic_angle", pp.internode.phyllotactic_angle);
    *O << "MISC " << pp.petiole.petioles_per_internode << " " << pp.peduncle.length_segments << " " << pp.peduncle.radial_subdivisions << " "
       << pp.inflorescence.unique_prototypes;
    F(pp.peduncle.color.r); F(pp.peduncle.color.g); F(pp.peduncle.color.b);
    F(pp.internode.color.r); F(pp.internode.color.g); F(pp.internode.color.b);
    *O << "\n";
    *O << "FBUDS " << ph.floral_buds.size();
    for (auto &e : ph.floral_buds) *O << " " << e.size();
    *O << "\n";
    *O << "PEDSTORE " << ph.peduncle_length.size();
    for (auto &e : ph.peduncle_length) *O << " " << e.size();
    *O << " " << ph.peduncle_vertices.size();
    for (auto &e : ph.peduncle_vertices) *O << " " << e.size();
    *O << "\n";
}

class PlantArchitectureTestHelper {
public:
    static std::vector<std::shared_ptr<Shoot>> &tree(PlantArchitecture &pa, uint pid) { return pa.plant_instances.at(pid).shoot_tree; }
};

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "usage: inflo_driver SEED AGE RESEED OUT\n";
        return 2;
    }
    uint seed = uint(std::atoi(argv[1]));
    float age = float(std::atof(argv[2]));
    uint reseed = uint(std::atoi(argv[3]));
    std::ofstream out(argv[4]);
    O = &out;
    Context ctx;
    ctx.seedRandomGenerator(seed);
    PlantArchitecture pa(&ctx);
    pa.disableMessages();
    pa.loadPlantModelFromLibrary("tomato");
    uint pid = pa.buildPlantInstanceFromLibrary(make_vec3(0, 0, 0), age);
    out << "PLANT " << seed << " " << age << " " << ctx.getPrimitiveCount() << " " << ctx.getObjectCount() << "\n";
    engine(ctx, "ENGBUILD");
    uint k = 0;
    bool did_culm = false;
    auto &tree = PlantArchitectureTestHelper::tree(pa, pid);
    for (auto &shoot : tree) {
        for (uint ip = 0; ip < shoot->phytomers.size(); ip++) {
            Phytomer &ph = *shoot->phytomers[ip];
            for (uint entry = 0; entry < ph.floral_buds.size(); entry++) {
                for (uint pos = 0; pos < ph.floral_buds[entry].size(); pos++) {
                    FloralBud &fbud = ph.floral_buds[entry][pos];
                    if (fbud.state != BUD_FLOWER_CLOSED && fbud.state != BUD_FLOWER_OPEN && fbud.state != BUD_FRUITING) continue;
                    out << "BUD " << shoot->ID << " " << ip << " " << entry << " " << pos << " " << int(fbud.state) << "\n";
                    geometry(ctx, ph, fbud, entry, pos, "GROWN", false);
                    ph.setFloralBudState(BUD_DEAD, fbud);
                    inputs(ph, fbud);
                    const BudState seq[3] = {BUD_FLOWER_CLOSED, BUD_FLOWER_OPEN, BUD_FRUITING};
                    for (BudState target : seq) {
                        ctx.seedRandomGenerator(reseed + k);
                        out << "STEP " << int(target) << " " << (reseed + k) << "\n";
                        k++;
                        rp("internode.phyllotactic_angle", ph.phytomer_parameters.internode.phyllotactic_angle);
                        ph.setFloralBudState(target, fbud);
                        engine(ctx, "ENG");
                        geometry(ctx, ph, fbud, entry, pos, "FORCED");
                    }
                    for (int tc = 2; tc <= 32; tc += 2) {
                        float scale = fmin(1, 0.25f + 0.75f * float(tc) / 30.f);
                        ph.setInflorescenceScaleFraction(fbud, scale);
                    }
                    geometry(ctx, ph, fbud, entry, pos, "GROW");
                    ph.updatePeduncleRadii();
                    geometry(ctx, ph, fbud, entry, pos, "RADII");
                    out << "ENDBUD\n";
                    if (!did_culm) {
                        // Culm-matched variant on the first bud: as if this were the last node of a max-node
                        // shoot (reconciled radius 0.9 * culm tip, blended colours), then the culm thickens
                        // twice and updatePeduncleRadii rescales the tube (path-dependent setTubeRadii).
                        did_culm = true;
                        std::vector<float> saved_radii = ph.parent_shoot_ptr->shoot_internode_radii.at(ph.shoot_index.x);
                        int saved_z = ph.shoot_index.z;
                        ph.setFloralBudState(BUD_DEAD, fbud);
                        ph.shoot_index.z = ph.shoot_index.x + 1;
                        ctx.seedRandomGenerator(reseed + 500);
                        out << "CULM " << (reseed + 500) << "\n";
                        rp("internode.phyllotactic_angle", ph.phytomer_parameters.internode.phyllotactic_angle);
                        ph.setFloralBudState(BUD_FLOWER_CLOSED, fbud);
                        engine(ctx, "ENG");
                        geometry(ctx, ph, fbud, entry, pos, "CULM");
                        const float factors[2] = {1.5f, 2.f};
                        for (float factor : factors) {
                            for (float &r : ph.parent_shoot_ptr->shoot_internode_radii.at(ph.shoot_index.x)) r *= factor;
                            ph.updatePeduncleRadii();
                            geometry(ctx, ph, fbud, entry, pos, "CULMR");
                        }
                        ph.parent_shoot_ptr->shoot_internode_radii.at(ph.shoot_index.x) = saved_radii;
                        ph.shoot_index.z = saved_z;
                        out << "ENDCULM\n";
                    }
                }
            }
        }
    }
    out << "END\n";
    return 0;
}
'''


# ---- driver build / run ------------------------------------------------------------------------------

def _cache_dir() -> str:
    d = os.environ.get("HELIADES_ORACLE_CACHE") or os.path.join(tempfile.gettempdir(), "heliades_oracle_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _helios_paths():
    src = os.environ.get("HELIOS_SRC_ROOT", HELIOS_SRC)
    build = os.path.join(src, "pyhelios_build", "build")
    try:
        from pyhelios.assets import get_asset_manager
        p = get_asset_manager()._get_helios_build_path()
        if p and os.path.isdir(os.path.join(str(p), "lib")):
            build = str(p)
    except Exception:
        pass
    return src, build


def _src_hash() -> str:
    return hashlib.sha1(_DRIVER_SRC.encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=None)
def driver_executable() -> str:
    """Compile the trace driver once (cached by source hash) and return its path."""
    src, build = _helios_paths()
    exe = os.path.join(_cache_dir(), f"inflo_driver_{_src_hash()}")
    if os.path.exists(exe):
        return exe
    cpp = exe + ".cpp"
    with open(cpp, "w") as fh:
        fh.write(_DRIVER_SRC)
    core = os.path.join(src, "helios-core")
    lib = os.path.join(build, "lib")
    cmd = ["g++", "-O1", "-std=c++17", "-fopenmp", "-DNDEBUG", "-DHELIOS_NO_OPTIX", "-DUSE_OPENMP",
           "-DPLANTARCHITECTURE_PLUGIN_AVAILABLE",
           f"-I{core}/plugins/plantarchitecture/include", f"-I{core}/plugins/collisiondetection/include",
           f"-I{core}/core/include", f"-I{core}/core/lib/pugixml", f"-I{core}/core/lib/json", f"-I{lib}/zlib",
           cpp, f"-L{lib}", "-lhelios", f"-Wl,-rpath,{lib}", "-o", exe + ".tmp"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("compiling the inflorescence oracle driver failed:\n" + r.stderr[-4000:])
    os.replace(exe + ".tmp", exe)
    return exe


@functools.lru_cache(maxsize=4)
def oracle_dump(seed: int, age: float, reseed: int) -> list:
    """Run the driver (cached on disk) and parse its output into one record per floral bud."""
    out = os.path.join(_cache_dir(), f"inflo_{_src_hash()}_{seed}_{age:g}_{reseed}.txt")
    if not os.path.exists(out):
        _src, build = _helios_paths()
        r = subprocess.run([driver_executable(), str(seed), repr(float(age)), str(reseed), out + ".tmp"],
                           cwd=build, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"inflorescence oracle driver failed ({r.returncode}): {r.stderr[-2000:]}")
        os.replace(out + ".tmp", out)
    with open(out) as fh:
        return parse_dump(fh.read().splitlines())


# ---- dump parsing ----------------------------------------------------------------------------------

def _hexf(tokens) -> np.ndarray:
    """Hex bit patterns -> float32."""
    return np.array([int(t, 16) for t in tokens], np.uint32).view(np.float32)


def _hexrun(s: str) -> np.ndarray:
    if s == "-":
        return np.zeros(0, np.float32)
    return np.frombuffer(bytes.fromhex(s), ">u4").astype(np.uint32).view(np.float32)


def _runs(tok) -> list:
    n = int(tok[1])
    return [(tok[2 + 2 * i], int(tok[3 + 2 * i])) for i in range(n)]


def _parse_prims(lines, i):
    pv = lines[i].split(" ")
    assert pv[0] == "PV"
    return {"n": int(pv[1]), "pv": _hexrun(pv[2]), "pm": _runs(lines[i + 1].split()),
            "pl": _runs(lines[i + 2].split())}, i + 3


def _parse_block(lines, i):
    t = lines[i].split()
    assert t[0] == "G", lines[i][:80]
    blk = {"tag": t[1], "state": int(t[2]), "npeds": int(t[3]), "nobjs": int(t[4]), "peds": [], "objs": []}
    i += 1
    blk["fs"] = _hexf(lines[i].split()[1:])
    i += 1
    for _ in range(blk["npeds"]):
        t = lines[i].split()
        n = int(t[2])
        vals = _hexf(t[3:])
        ped = {"nodes": vals[:3 * n], "radii": vals[3 * n:]}
        i += 1
        if lines[i].startswith("PV"):
            ped["prims"], i = _parse_prims(lines, i)
        blk["peds"].append(ped)
    for _ in range(blk["nobjs"]):
        t = lines[i].split()
        vals = _hexf(t[2:])
        obj = {"T": vals[:16], "rot": vals[16:24], "base": vals[24:27], "scale": vals[27:28]}
        i += 1
        if i < len(lines) and lines[i].startswith("PV"):
            obj["prims"], i = _parse_prims(lines, i)
        blk["objs"].append(obj)
    t = lines[i].split()
    assert t[0] == "PSTORE"
    nv = int(t[1])
    if nv == 0:
        blk["pstore"] = np.zeros(0, np.float32)
    else:
        verts = _hexf(t[2:2 + 3 * nv])
        k = 2 + 3 * nv
        nr = int(t[k])
        rest = _hexf(t[k + 1:k + 1 + nr + 6])
        blk["pstore"] = np.concatenate([verts, rest])
    return blk, i + 1


def _parse_rp(t):
    kind, name, dist, n = t[0], t[1], t[2], int(t[3])
    if kind == "RPF":
        params = _hexf(t[4:4 + n])
        return name, {"kind": "f", "dist": dist, "params": params, "constval": _hexf([t[4 + n]])[0],
                      "sampled": bool(int(t[5 + n]))}
    params = [int(x) for x in t[4:4 + n]]
    return name, {"kind": "i", "dist": dist, "params": params, "constval": int(t[4 + n]), "sampled": bool(int(t[5 + n]))}


def parse_dump(lines) -> list:
    buds = []
    i = 0
    while i < len(lines):
        t = lines[i].split()
        if not t or t[0] != "BUD":
            i += 1
            continue
        bud = {"where": tuple(int(x) for x in t[1:5]), "state": int(t[5]), "rp": {}, "steps": []}
        i += 1
        bud["grown"], i = _parse_block(lines, i)
        inp = {}
        while not lines[i].startswith("STEP"):
            t = lines[i].split()
            k = t[0]
            if k == "BASE":
                v = _hexf(t[1:11])
                inp.update(base_position=v[0:3], base_rotation=v[3:6], current_fruit_scale_factor=v[6],
                           dd_to_inflorescence_maturity=v[7], time_counter=v[8], age=v[9],
                           parent_index=int(t[11]), bud_index=int(t[12]), isterminal=bool(int(t[13])))
            elif k == "SIDX":
                inp["shoot_index"] = [int(t[1]), int(t[2]), int(t[3])]
                inp["rank"] = int(t[4])
            elif k == "INODES":
                inp["internode_nodes"] = _hexf(t[2:]).reshape(-1, 3)
            elif k == "IRADII":
                inp["internode_radii"] = list(_hexf(t[2:]))
            elif k == "PETIOLES":
                inp["petioles"] = []
            elif k == "PET":
                inp["petioles"].append(_hexf(t[2:]).reshape(-1, 3))
            elif k == "BEND":
                inp["bending_axis"] = _hexf(t[1:4])
            elif k == "SHOOT":
                inp["shoot_type_label"] = t[1]
                inp["image_texture_empty"] = bool(int(t[2]))
            elif k in ("RPF", "RPI"):
                name, rp = _parse_rp(t)
                bud["rp"][name] = rp
            elif k == "MISC":
                inp.update(petioles_per_internode=int(t[1]), peduncle_length_segments=int(t[2]),
                           peduncle_radial_subdivisions=int(t[3]), unique_prototypes=int(t[4]))
                c = _hexf(t[5:11])
                inp.update(peduncle_color=c[0:3], internode_color=c[3:6])
            elif k == "FBUDS":
                inp["fbuds"] = [int(x) for x in t[2:]]
            elif k == "PEDSTORE":
                n = int(t[1])
                inp["pedstore_length"] = [int(x) for x in t[2:2 + n]]
                m = int(t[2 + n])
                inp["pedstore_vertices"] = [int(x) for x in t[3 + n:3 + n + m]]
            i += 1
        bud["inputs"] = inp
        while lines[i].startswith("STEP"):
            t = lines[i].split()
            step = {"target": int(t[1]), "reseed": int(t[2])}
            _name, step["phyl_before"] = _parse_rp(lines[i + 1].split())
            step["engine"] = int(lines[i + 2].split()[1])
            step["block"], i = _parse_block(lines, i + 3)
            bud["steps"].append(step)
        bud["grow"], i = _parse_block(lines, i)
        bud["radii"], i = _parse_block(lines, i)
        assert lines[i] == "ENDBUD"
        i += 1
        if i < len(lines) and lines[i].startswith("CULM "):
            culm = {"reseed": int(lines[i].split()[1])}
            _name, culm["phyl_before"] = _parse_rp(lines[i + 1].split())
            culm["engine"] = int(lines[i + 2].split()[1])
            culm["blocks"] = []
            i += 3
            while lines[i].startswith("G "):
                blk, i = _parse_block(lines, i)
                culm["blocks"].append(blk)
            assert lines[i] == "ENDCULM"
            i += 1
            bud["culm"] = culm
        buds.append(bud)
    return buds


# ---- Heliades replay ---------------------------------------------------------------------------------

class StandInPhytomer:
    """The members of ``Phytomer`` that the inflorescence functions read, filled from an oracle dump."""

    def __init__(self, pa, scene, plantID, phytomer_parameters, inputs, parent_shoot):
        from ....plantarchitecture.params import FloralBud
        self.plantarchitecture_ptr = pa
        self.context_ptr = scene
        self.plantID = plantID
        self.phytomer_parameters = phytomer_parameters
        self.parent_shoot_ptr = parent_shoot
        self.shoot_index = list(inputs["shoot_index"])
        self.rank = inputs["rank"]
        self._internode_nodes = [np.asarray(n, f32).copy() for n in inputs["internode_nodes"]]
        self.petiole_vertices = [[np.asarray(v, f32).copy() for v in pv] for pv in inputs["petioles"]]
        self.inflorescence_bending_axis = np.asarray(inputs["bending_axis"], f32).copy()
        self.build_context_geometry_peduncle = pa.build_context_geometry_peduncle
        self.floral_buds = [[FloralBud() for _ in range(n)] for n in inputs["fbuds"]]
        ns = inputs["pedstore_length"]
        self.peduncle_length = [[f32(0)] * n for n in ns]
        self.peduncle_radius = [[f32(0)] * n for n in ns]
        self.peduncle_pitch = [[f32(0)] * n for n in ns]
        self.peduncle_curvature = [[f32(0)] * n for n in ns]
        self.peduncle_roll = [[f32(0)] * n for n in ns]
        self.peduncle_yaw = [[f32(0)] * n for n in ns]
        self.peduncle_vertices = [[[] for _ in range(n)] for n in inputs["pedstore_vertices"]]
        self.peduncle_radii = [[[] for _ in range(n)] for n in inputs["pedstore_vertices"]]

    def getInternodeNodePositions(self):
        return [n.copy() for n in self._internode_nodes]

    def getPetioleAxisVector(self, stem_fraction, petiole_index):
        from ....plantarchitecture.tube_utils import get_axis_vector
        return get_axis_vector(stem_fraction, self.petiole_vertices[petiole_index])


def _apply_rp(param, rp, name):
    if param.distribution != rp["dist"]:
        raise AssertionError(f"{name}: distribution {param.distribution} vs oracle {rp['dist']}")
    if rp["kind"] == "f":
        if not np.array_equal(np.asarray(param.distribution_parameters, f32), np.asarray(rp["params"], f32)):
            raise AssertionError(f"{name}: distribution parameters differ")
        param.constval = f32(rp["constval"])
    else:
        param.constval = int(rp["constval"])
    param.sampled = bool(rp["sampled"])


def _get_param(pp, dotted):
    obj = pp
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _fill_bud(fbud, inp):
    from ....plantarchitecture.params import BUD_DEAD
    fbud.state = BUD_DEAD
    fbud.base_position = tuple(inp["base_position"])
    fbud.base_rotation.pitch, fbud.base_rotation.yaw, fbud.base_rotation.roll = inp["base_rotation"]
    fbud.current_fruit_scale_factor = inp["current_fruit_scale_factor"]
    fbud.dd_to_inflorescence_maturity = inp["dd_to_inflorescence_maturity"]
    fbud.time_counter = inp["time_counter"]
    fbud.age = inp["age"]
    fbud.parent_index = inp["parent_index"]
    fbud.bud_index = inp["bud_index"]
    fbud.isterminal = inp["isterminal"]


class Replay:
    """One Heliades scene per oracle plant; one stand-in phytomer per bud, replayed in the driver's order."""

    def __init__(self, B):
        from ....plantarchitecture import PlantArchitecture
        self.B = B
        self.scene = B.scene
        self.pa = PlantArchitecture(self.scene)
        self.pa.loadPlantModelFromLibrary("tomato")
        self.pa.current_build_plant_name = "tomato"
        self.pid = self.pa.addPlantInstance((0.0, 0.0, 0.0), 0.0)
        self.pa.current_build_plant_name = ""

    def stand_in(self, bud):
        pa = self.pa
        inp = bud["inputs"]
        st = pa.plant_instances[self.pid].shoot_types_snapshot.get(inp["shoot_type_label"]) \
            or pa.shoot_types[inp["shoot_type_label"]]
        pp = st.phytomer_parameters.copy()
        for name, rp in bud["rp"].items():
            _apply_rp(_get_param(pp, name), rp, name)
        if (int(pp.petiole.petioles_per_internode) != inp["petioles_per_internode"]
                or int(pp.peduncle.length_segments) != inp["peduncle_length_segments"]
                or int(pp.peduncle.radial_subdivisions) != inp["peduncle_radial_subdivisions"]
                or int(pp.inflorescence.unique_prototypes) != inp["unique_prototypes"]
                or not np.array_equal(np.asarray(pp.peduncle.color, f32), inp["peduncle_color"])
                or not np.array_equal(np.asarray(pp.internode.color, f32), inp["internode_color"])):
            raise AssertionError("tomato shoot parameters differ from the oracle phytomer's")
        sp = SimpleNamespace(phytomer_parameters=SimpleNamespace(
            internode=SimpleNamespace(image_texture="" if inp["image_texture_empty"] else "bark.png")))
        shoot = SimpleNamespace(shoot_type_label=inp["shoot_type_label"], shoot_parameters=sp,
                                shoot_internode_radii={inp["shoot_index"][0]: [f32(r) for r in inp["internode_radii"]]})
        ph = StandInPhytomer(pa, self.scene, self.pid, pp, inp, shoot)
        entry, pos = bud["where"][2], bud["where"][3]
        fbud = ph.floral_buds[entry][pos]
        _fill_bud(fbud, inp)
        return ph, fbud, entry, pos

    def run(self, bud, on_step=None):
        """Replay the driver's sequence for one bud; returns the Heliades blocks and engine states."""
        from ....plantarchitecture import inflorescence as I
        from ....plantarchitecture.architecture import context_generator
        ph, fbud, entry, pos = self.stand_in(bud)
        out = {"steps": []}
        gen = context_generator(self.scene)
        for step in bud["steps"]:
            self.scene.seed_random_generator(step["reseed"])
            phyl = ph.phytomer_parameters.internode.phyllotactic_angle
            before = (f32(phyl.constval), bool(phyl.sampled))
            I.set_floral_bud_state(self.pa, ph, fbud, step["target"])
            out["steps"].append({"phyl_before": before, "engine": int(gen.x),
                                 "block": heliades_block(self.scene, ph, fbud, entry, pos)})
        for tc in range(2, 33, 2):
            I.set_inflorescence_scale_fraction(ph, fbud, I.fruit_growth_fraction(f32(tc), f32(30)))
        out["grow"] = heliades_block(self.scene, ph, fbud, entry, pos)
        I.update_peduncle_radii(ph)
        out["radii"] = heliades_block(self.scene, ph, fbud, entry, pos)
        if "culm" in bud:
            out["culm"] = self._culm(bud["culm"], ph, fbud, entry, pos)
        return out

    def _culm(self, culm, ph, fbud, entry, pos):
        """The driver's culm-matched variant, on the same stand-in (its state carries over, as in Helios)."""
        from ....plantarchitecture import inflorescence as I
        from ....plantarchitecture.architecture import context_generator
        from ....plantarchitecture.params import BUD_DEAD, BUD_FLOWER_CLOSED
        radii = ph.parent_shoot_ptr.shoot_internode_radii[ph.shoot_index[0]]
        saved_radii, saved_z = list(radii), ph.shoot_index[2]
        I.set_floral_bud_state(self.pa, ph, fbud, BUD_DEAD)
        ph.shoot_index[2] = ph.shoot_index[0] + 1
        self.scene.seed_random_generator(culm["reseed"])
        phyl = ph.phytomer_parameters.internode.phyllotactic_angle
        res = {"phyl_before": (f32(phyl.constval), bool(phyl.sampled))}
        I.set_floral_bud_state(self.pa, ph, fbud, BUD_FLOWER_CLOSED)
        res["engine"] = int(context_generator(self.scene).x)
        res["blocks"] = [heliades_block(self.scene, ph, fbud, entry, pos)]
        for factor in (f32(1.5), f32(2)):
            radii[:] = [f32(r * factor) for r in radii]
            I.update_peduncle_radii(ph)
            res["blocks"].append(heliades_block(self.scene, ph, fbud, entry, pos))
        radii[:] = saved_radii
        ph.shoot_index[2] = saved_z
        return res


def _runs_of(seq):
    runs = []
    for s in seq:
        if runs and runs[-1][0] == s:
            runs[-1] = (s, runs[-1][1] + 1)
        else:
            runs.append((s, 1))
    return runs


def _heliades_prims(scene, objID):
    uu = scene.object_primitive_uuids(objID)
    vals = []
    mats, labels = [], []
    for u in uu:
        v = scene.primitive_vertices(u)
        if len(v) != 3:
            raise AssertionError("non-triangle primitive")
        vals.append(np.asarray(v, f32).reshape(9))
        vals.append(np.asarray(scene.get_primitive_color(u), f32)[:3])
        mats.append(scene.get_primitive_material_label(u))
        labels.append(scene.get_primitive_data(u, "object_label") if scene.does_primitive_data_exist(u, "object_label")
                      else "-")
    pv = np.concatenate(vals).astype(f32) if vals else np.zeros(0, f32)
    return {"n": len(uu), "pv": pv, "pm": _runs_of(mats), "pl": _runs_of(labels)}


def heliades_block(scene, ph, fbud, entry, pos) -> dict:
    """The driver's ``geometry()`` dump, read from the Heliades scene."""
    blk = {"state": int(fbud.state), "npeds": len(fbud.peduncle_objIDs), "nobjs": len(fbud.inflorescence_objIDs),
           "fs": np.array([fbud.current_fruit_scale_factor, fbud.time_counter], f32), "peds": [], "objs": []}
    for objID in fbud.peduncle_objIDs:
        blk["peds"].append({"nodes": np.asarray(scene.tube_object_nodes(objID), f32).reshape(-1),
                            "radii": np.asarray(scene.tube_object_node_radii(objID), f32),
                            "prims": _heliades_prims(scene, objID)})
    for i, objID in enumerate(fbud.inflorescence_objIDs):
        r = fbud.inflorescence_rotation[i]
        rot = np.array([r.pitch, r.yaw, r.roll, r.azimuth, *r.peduncle_axis, r.gravity_fraction], f32)
        blk["objs"].append({"T": np.asarray(scene.object_transform(objID), f32).reshape(16), "rot": rot,
                            "base": np.asarray(fbud.inflorescence_bases[i], f32),
                            "scale": np.array([fbud.inflorescence_base_scales[i]], f32),
                            "prims": _heliades_prims(scene, objID)})
    if entry < len(ph.peduncle_vertices) and pos < len(ph.peduncle_vertices[entry]) and ph.peduncle_vertices[entry][pos]:
        verts = np.asarray(ph.peduncle_vertices[entry][pos], f32).reshape(-1)
        rad = np.asarray(ph.peduncle_radii[entry][pos], f32)
        rest = np.array([ph.peduncle_length[entry][pos], ph.peduncle_radius[entry][pos], ph.peduncle_pitch[entry][pos],
                         ph.peduncle_curvature[entry][pos], ph.peduncle_roll[entry][pos], ph.peduncle_yaw[entry][pos]], f32)
        blk["pstore"] = np.concatenate([verts, rad, rest])
    else:
        blk["pstore"] = np.zeros(0, f32)
    return blk


# ---- streams -----------------------------------------------------------------------------------------

def block_floats(blk, with_prims=True) -> list:
    """A geometry block as one float32 sequence (counts first, so a structural difference shows at once)."""
    parts = [np.array([blk["state"], blk["npeds"], blk["nobjs"]], f32), np.asarray(blk["fs"], f32)]
    for ped in blk["peds"]:
        parts += [np.array([len(ped["nodes"]) // 3], f32), ped["nodes"], ped["radii"]]
        if with_prims:
            parts += [np.array([ped["prims"]["n"]], f32), ped["prims"]["pv"]]
    for obj in blk["objs"]:
        parts += [obj["T"], obj["rot"], obj["base"], obj["scale"]]
        if with_prims:
            parts += [np.array([obj["prims"]["n"]], f32), obj["prims"]["pv"]]
    parts.append(blk["pstore"])
    return parts


def block_labels(blk) -> list:
    out = []
    for item in blk["peds"] + blk["objs"]:
        p = item["prims"]
        out += [f"M:{s}x{n}" for s, n in p["pm"]] + [f"L:{s}x{n}" for s, n in p["pl"]]
    return out


def _encode(strings) -> list:
    return [float(b) for b in "\n".join(strings).encode()]


def engine_calls_between(reseed: int, state: int, nmax: int = 100000) -> int:
    from ....core.random import MinstdRand0
    g = MinstdRand0(reseed)
    for n in range(nmax):
        if g.x == state:
            return n
        g()
    return -1


AGES = (100.0, 110.0, 120.0)


def grid(s: int):
    """Case seed s (0..23) -> (Context seed 1..8, age 100/110/120, reseed base)."""
    seed = s % 8 + 1
    age = AGES[s // 8 % 3]
    return seed, age, 1000 * seed + int(age)


def _streams(buds, results) -> dict:
    """The three comparison streams of one plant (``results``: per bud, the oracle record or the replay)."""
    out = {"labels": [float(len(buds))], "draws": [float(len(buds))]}
    floats = [np.array([len(buds)], f32)]
    for bud, res in zip(buds, results):
        culm_blocks = res["culm"]["blocks"] if "culm" in res else []
        for blk in [step["block"] for step in res["steps"]] + [res["grow"], res["radii"]] + culm_blocks:
            floats += [np.asarray(p, f32).reshape(-1) for p in block_floats(blk)]
        labs = []
        for blk in [step["block"] for step in res["steps"]] + culm_blocks[:1]:
            labs += block_labels(blk)
        out["labels"] += _encode(labs)
        pairs = list(zip(bud["steps"], res["steps"]))
        if "culm" in res:
            pairs.append((bud["culm"], res["culm"]))
        for ostep, step in pairs:
            cv, sampled = step["phyl_before"]
            out["draws"] += [float(engine_calls_between(ostep["reseed"], step["engine"])), float(cv), float(sampled)]
    out["floats"] = np.concatenate(floats)
    return out


def _oracle_results(buds) -> list:
    res = []
    for bud in buds:
        steps = [{"block": st["block"], "engine": st["engine"],
                  "phyl_before": (st["phyl_before"]["constval"], st["phyl_before"]["sampled"])} for st in bud["steps"]]
        r = {"steps": steps, "grow": bud["grow"], "radii": bud["radii"]}
        if "culm" in bud:
            c = bud["culm"]
            r["culm"] = {"blocks": c["blocks"], "engine": c["engine"],
                         "phyl_before": (c["phyl_before"]["constval"], c["phyl_before"]["sampled"])}
        res.append(r)
    return res


# Small streams (labels, draws) are kept per (backend, case seed) so the three placement cases replay once.
_SMALL: dict = {}


def _forced(backend: str, B, s: int, what: str) -> list:
    key = (backend, s)
    if what != "floats" and key in _SMALL:
        return _SMALL[key][what]
    seed, age, reseed = grid(s)
    buds = oracle_dump(seed, age, reseed)          # Heliades reads only the dumped *inputs*
    if backend == "helios":
        results = _oracle_results(buds)
    else:
        rep = Replay(B)
        results = [rep.run(bud) for bud in buds]
    st = _streams(buds, results)
    _SMALL[key] = {"labels": st["labels"], "draws": st["draws"]}
    return st[what]


@op("inflo_forced", "helios")
def _(B, s, what="floats"):
    return _forced("helios", B, s, what)


@op("inflo_forced", "heliades")
def _(B, s, what="floats"):
    return _forced("heliades", B, s, what)


# ---- end-to-end organs (plantarch.tomato.flowering, gated on growth) ----------------------------------

def _organ_stream(objs, prims, transform):
    """``prims(o)`` -> (n, float32 array of the n triangles' 9 vertex floats + 3 colour floats each)."""
    out = []
    for kind, ids in enumerate(objs):
        out.append(np.array([len(ids)], f32))
        for o in sorted(ids):
            n, vals = prims(o)
            out += [np.array([kind, o, n], f32), np.asarray(transform(o), f32).reshape(16), vals]
    return np.concatenate(out)


@op("inflo_plant_organs", "helios")
def _(B):
    ctx, pa, pid = B.ctx, B.pa, B.pid
    objs = [list(pa.getPlantPeduncleObjectIDs(pid)), list(pa.getPlantFlowerObjectIDs(pid)),
            list(pa.getPlantFruitObjectIDs(pid))]

    def prims(o):
        uu = [int(u) for u in ctx.getObjectPrimitiveUUIDs(o)]
        flat, _ = ctx.getPrimitiveVertices(uu)
        v = np.asarray(flat, f32).reshape(len(uu), 9)
        c = np.asarray(ctx.getPrimitiveColor(uu), f32).reshape(len(uu), -1)[:, :3]
        return len(uu), np.concatenate([v, c], axis=1).reshape(-1)

    return _organ_stream(objs, prims, lambda o: np.asarray(ctx.getObjectTransformationMatrix(o), f32))


@op("inflo_plant_organs", "heliades")
def _(B):
    from ....plantarchitecture.params import BUD_FRUITING
    S = B.scene
    peds, flowers, fruit = [], [], []
    for shoot in B.pa.plant_instances[B.pid].shoot_tree:
        for ph in shoot.phytomers:
            for buds in ph.floral_buds:
                for fb in buds:
                    peds += list(fb.peduncle_objIDs)
                    (fruit if fb.state == BUD_FRUITING else flowers).extend(fb.inflorescence_objIDs)
    def prims(o):
        uu = S.object_primitive_uuids(o)
        if not uu:
            return 0, np.zeros(0, f32)
        v = np.array([np.asarray(S.primitive_vertices(u), f32).reshape(9) for u in uu], f32)
        c = np.array([np.asarray(S.get_primitive_color(u), f32)[:3] for u in uu], f32)
        return len(uu), np.concatenate([v, c], axis=1).reshape(-1)

    return _organ_stream([peds, flowers, fruit], prims, S.object_transform)


# ---- prototypes ------------------------------------------------------------------------------------

def _proto_record(get):
    """Flatten one prototype (via accessor callbacks) into floats + strings."""
    floats, strings = [], []
    for o in get["objects"]():
        uu = get["uuids"](o)
        floats += [float(len(uu))] + list(get["T"](o))
        for u in uu:
            floats += list(get["verts"](u)) + list(get["uv"](u)) + list(get["color"](u)) + [get["area"](u)]
            strings += [get["material"](u), get["label"](u), ",".join(sorted(get["pdata"](u)))]
        strings += sorted(get["odata"](o))
    return floats, strings


@op("inflo_prototypes", "helios")
def _(B, seed, age):
    from pyhelios.types import vec3
    ctx = B.ctx
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.pid = B.pa.buildPlantInstanceFromLibrary(vec3(0, 0, 0), float(age))
    hidden = [o for o in range(38, ctx.getObjectCount() + 64) if ctx.doesObjectExist(o) and ctx.isObjectHidden(o)]
    mats = {}

    def material(u):
        lab = ctx.getPrimitiveMaterialLabel(u)
        if lab not in mats:
            c = ctx.getMaterialColor(lab)
            mats[lab] = (lab, ctx.getMaterialTwosidedFlag(lab), (c.r, c.g, c.b, c.a), ctx.getMaterialTexture(lab),
                         ctx.isMaterialTextureColorOverridden(lab))
        return repr(mats[lab])

    def verts(u):
        flat, _ = ctx.getPrimitiveVertices([u])
        return [float(x) for x in np.asarray(flat, f32)]

    def uv(u):
        flat, _ = ctx.getPrimitiveTextureUV([u])
        return [float(x) for x in np.asarray(flat, f32)]

    get = {"objects": lambda: hidden, "uuids": lambda o: list(ctx.getObjectPrimitiveUUIDs(o)),
           "T": lambda o: [float(x) for x in np.asarray(ctx.getObjectTransformationMatrix(o), f32).reshape(16)],
           "verts": verts, "uv": uv,
           "color": lambda u: [float(x) for x in np.asarray(ctx.getPrimitiveColor([u]), f32).reshape(-1)[:3]],
           "area": lambda u: float(f32(ctx.getPrimitiveArea(u))),
           "material": material,
           "label": lambda u: ctx.getPrimitiveData(u, "object_label") if ctx.doesPrimitiveDataExist(u, "object_label") else "-",
           "odata": lambda o: list(ctx.listObjectData(o)), "pdata": lambda u: list(ctx.listPrimitiveData(u))}
    return _proto_record(get), [ctx.isObjectHidden(o) for o in hidden]


@op("inflo_prototypes", "heliades")
def _(B, seed, age):
    from ....plantarchitecture import inflorescence as I
    S = B.scene
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.plantarch_build((0.0, 0.0, 0.0), 0.0)
    ph = B.pa.plant_instances[B.pid].shoot_tree[0].phytomers[0]
    before = set(S.all_object_ids(include_hidden=True))
    I.ensure_inflorescence_prototypes_initialized(B.pa, ph.phytomer_parameters, B.pid)
    hidden = sorted(o for o in S.all_object_ids(include_hidden=True) if o not in before)
    mats = {}

    def material(u):
        lab = S.get_primitive_material_label(u)
        if lab not in mats:
            c = np.asarray(S.get_material_color(lab), f32)
            mats[lab] = (lab, S.get_material_twosided_flag(lab), tuple(float(x) for x in c[:4]),
                         S.get_material_texture(lab), S.is_material_texture_color_overridden(lab))
        return repr(mats[lab])

    def uv(u):
        t = S.primitive_texture_uv(u)
        return [] if t is None else [float(x) for x in np.asarray(t, f32).reshape(-1)]

    get = {"objects": lambda: hidden, "uuids": lambda o: list(S.object_primitive_uuids(o)),
           "T": lambda o: [float(x) for x in np.asarray(S.object_transform(o), f32).reshape(16)],
           "verts": lambda u: [float(x) for x in np.asarray(S.primitive_vertices(u), f32).reshape(-1)],
           "uv": uv,
           "color": lambda u: [float(x) for x in np.asarray(S.get_primitive_color(u), f32)[:3]],
           "area": lambda u: float(f32(S.primitive_area(u))),
           "material": material,
           "label": lambda u: S.get_primitive_data(u, "object_label") if S.does_primitive_data_exist(u, "object_label") else "-",
           "odata": lambda o: list(S.list_object_data(o)), "pdata": lambda u: list(S.list_primitive_data(u))}
    return _proto_record(get), [S.is_object_hidden(o) for o in hidden]
