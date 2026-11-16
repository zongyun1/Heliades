# SPDX-License-Identifier: LGPL-2.1-or-later
"""Parity cases for tomato growth (W3-C): ``advanceTime`` and everything it drives.

Seeds: case seed ``s`` -> Context seed ``s + 1`` (fuzz=8 covers Context seeds 1..8). One tomato per
PlantArchitecture, built at age 0 with ``buildPlantInstanceFromLibrary`` and then grown.

What is compared, per observation (``_observe``):

* ``draws``: engine calls since seeding (the external probe, SPEC §3.5; it reads 3 ``randu`` on both
  backends, so later steps continue from the same, probed, stream);
* ``xml``: every element of ``writePlantStructureXML`` in document order (tag code, then its numbers; the
  Heliades text is a port of the writer, ``plantarchitecture/plant.py``);
* ``shoot0``: the base shoot's stored internode nodes at full precision (independent of blade area);
* ``petiole0``: the triangle vertices of the base shoot's petiole tubes (independent of blade area);
* area-dependent checks, on only when the blade areas are real (``_area_exact``): all internode radii
  (``<internode_radius>`` in the XML and the stored radii), the position-derived XML values of the laterals
  (``AREA_MASK_LATERAL``; a lateral is seated by its parent's radius), every internode and petiole tube object with its
  primitive UUIDs and vertices (girth sets the internode radii, the parent radius seats each lateral, the
  blade primitive count sets the UUIDs), and every leaflet object's ID and transform.

``_area_exact`` is true when the real ``GenericLeafPrototype`` is bound (W3-A merged: ``leaf_geometry_is_stub``
false) or the case uses *replayed blades* (``pa_build_replay_blades``: the oracle's 35 prototypes rebuilt
primitive by primitive, so every blade area and UUID is the oracle's). The stub blades of W3-B have other
areas, which only change girth and what hangs off it; the draw stream and the shoot / phytomer / bud state do
not depend on blade area.

Cases:

* ``plantarch.tomato.steps`` (ulp, 0): ages 2, 4, ..., 40 in 2-day ``advanceTime`` calls, observed after each
  call, with the default blades (stub until W3-A: area checks switch on by themselves when it lands);
* ``plantarch.tomato.steps_replay`` (ulp, 0): the same with replayed blades: every check, every step, and the
  leaflet vertices at the last step;
* ``plantarch.tomato.split`` (ulp, 0): ``advanceTime(20)`` vs ``10 + 10`` vs ``5 x 4`` (replayed blades; each
  variant against the oracle's same variant, plus the oracle-side identities 20 == 10+10 and 20 != 5x4 as flags);
* ``plantarch.tomato.age50`` (ulp, 0): a direct build at 50 days with replayed blades (no flowers yet for any
  of the 8 seeds; the earliest initiation is day 56);
* ``plantarch.tomato.age75_vegetative`` (ulp, 0): a direct build at 75 days, where most seeds flower. The
  inflorescence geometry is W3-D, so Heliades runs ``updateInflorescence`` in its draws-only mode: the draw
  count, the XML without the ``<flower>`` records, and the internode / petiole / leaflet geometry by value
  (not by UUID or object ID, which the missing flower objects shift).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zlib

import numpy as np

from .cases import case
from .tools.draw_count import engine_calls

f32 = np.float32

STEP_AGES = tuple(range(2, 41, 2))       # module-level so tests can shorten the sweep
STEP_DAYS = 2.0


def _seed(s):
    return s + 1


def _code(text: str) -> float:
    return float(zlib.crc32(text.encode()) & 0xFFFFFF)


_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$|^[-+]?(inf|nan)$")


#: Area-dependent XML values with stub blades. On the base shoot only the girth radius; on a lateral also
#: everything its position feeds (it is seated 0.9 parent radii off the parent's axis, so its node positions,
#: hence its axes and the gravitropic / azimuthal corrections built from them, move by ulps).
AREA_MASK_BASE = frozenset({"internode_radius"})
AREA_MASK_LATERAL = AREA_MASK_BASE | {"internode_length", "petiole_axis_initial", "petiole_rotation_axis",
                                      "gravitropic_angles", "azimuthal_restoring_angles"}


def xml_stream(text: str, mask_base=(), mask_lateral=()) -> list[float]:
    """Every element in document order: its tag code, then each whitespace/';'-separated token of its text
    (numbers as numbers, anything else as a code). Masked tags (``mask_base`` inside shoot 0,
    ``mask_lateral`` inside any other shoot) keep only their tag code."""
    out: list[float] = []

    def walk(el, shoot_id):
        if el.tag == "shoot":
            shoot_id = int(el.attrib.get("ID", -1))
        out.append(_code(el.tag))
        for k, v in sorted(el.attrib.items()):
            out.extend([_code(k), float(v) if _NUM.match(v) else _code(v)])
        mask = mask_base if shoot_id == 0 else (mask_lateral if shoot_id > 0 else ())
        if el.tag in mask:
            out.append(-1.0)
        else:
            for tok in (el.text or "").replace(";", " ; ").split():
                out.append(float(tok) if _NUM.match(tok) else _code(tok))
        for child in el:
            walk(child, shoot_id)

    walk(ET.fromstring(text), -1)
    return out


def strip_flowers(text: str) -> str:
    """Drop the per-flower records and the peduncle records (W3-D geometry: the stored peduncle radius of a
    culm-tip bud is reconciled with the culm by ``updatePeduncleRadii``) from a plant XML."""
    text = re.sub(r"\s*<flower>.*?</flower>", "", text, flags=re.S)
    return re.sub(r"\s*<peduncle>.*?</peduncle>", "", text, flags=re.S)


def _probe(B, seed):
    n = engine_calls(seed, [B.randu() for _ in range(3)])
    return -1 if n is None else n


def _hook_bound() -> bool:
    from ..plantarchitecture import hooks
    if hooks.GenericLeafPrototype.impl is None:
        try:
            from ..plantarchitecture import leaf_prototype  # noqa: F401
        except ImportError:
            pass
    return hooks.GenericLeafPrototype.impl is not None


def _area_exact(B, replay: bool) -> bool:
    """Same answer on both backends: replayed blades, or the real W3-A blades are bound."""
    exact = replay or _hook_bound()
    pa = getattr(B, "pa", None)
    if hasattr(B, "scene") and pa is not None and not replay:
        assert exact == (not pa.leaf_geometry_is_stub), "leaf stub flag and hook state disagree"
    return exact


def _start(B, s, replay: bool, age=0.0):
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    if replay:
        B.pa_build_replay_blades(seed, age)
    else:
        B.plantarch_build((0.0, 0.0, 0.0), age)
    return seed


def _tubes(B, kind, shoots=None, uuids=True):
    out = []
    for sid, oid, uu, verts in B.pa_tube_objects(kind):
        if shoots is not None and sid not in shoots:
            continue
        out += [float(sid)] + ([float(oid), float(len(uu))] + [float(u) for u in uu] if uuids else []) + verts
    return out


def _observe(B, exact: bool, *, draws_seed=None, xml_filter=None, ids=True, leaf_vertices=False) -> list[float]:
    out: list[float] = []
    if draws_seed is not None:
        out += [-101.0, float(_probe(B, draws_seed))]
    text = B.pa_xml_text()
    if xml_filter is not None:
        text = xml_filter(text)
    out += [-102.0] + (xml_stream(text) if exact else xml_stream(text, AREA_MASK_BASE, AREA_MASK_LATERAL))
    nodes, radii = B.pa_shoot_internodes([0])[0]
    out += [-103.0] + nodes
    out += [-104.0] + _tubes(B, "petiole", shoots={0}, uuids=exact and ids)
    if exact:
        for verts, rad in B.pa_shoot_internodes(B.pa_shoot_ids()):
            out += [-105.0] + verts + rad
        out += [-106.0] + _tubes(B, "internode", uuids=ids) + [-107.0] + _tubes(B, "petiole", uuids=ids)
        out.append(-108.0)
        for oid, T, verts in B.pa_leaf_objects(leaf_vertices):
            out += ([float(oid)] if ids else []) + T + verts
    return out


def _steps(B, s, replay: bool):
    seed = _start(B, s, replay)
    exact = _area_exact(B, replay)
    out = []
    for k, age in enumerate(STEP_AGES):
        B.pa_advance(STEP_DAYS)
        out += [-100.0, float(age)] + _observe(B, exact, draws_seed=seed,
                                               leaf_vertices=replay and k == len(STEP_AGES) - 1)
    return out


@case("plantarch.tomato.steps", tier="ulp", max_ulp=0, fuzz=8)
def tomato_steps(B, s):
    return _steps(B, s, replay=False)


@case("plantarch.tomato.steps_replay", tier="ulp", max_ulp=0, fuzz=8)
def tomato_steps_replay(B, s):
    return _steps(B, s, replay=True)


SPLITS = ((20.0,), (10.0, 10.0), (5.0, 5.0, 5.0, 5.0))


@case("plantarch.tomato.split", tier="ulp", max_ulp=0, fuzz=4)
def tomato_split(B, s):
    """advanceTime(20) vs 10+10 vs 5x4, each compared with the oracle's same variant. No probe between calls."""
    out, texts = [], []
    for split in SPLITS:
        B.pa_fresh_context()
        seed = _start(B, s, replay=True)
        for days in split:
            B.pa_advance(days)
        texts.append(B.pa_xml_text())
        out += [-110.0, float(len(split))] + _observe(B, True, draws_seed=seed)
    out += [-111.0, float(texts[0] == texts[1]), float(texts[0] == texts[2])]
    return out


@case("plantarch.tomato.age50", tier="ulp", max_ulp=0, fuzz=8)
def tomato_age50(B, s):
    seed = _start(B, s, replay=True, age=50.0)
    return _observe(B, True, draws_seed=seed, leaf_vertices=True)


@case("plantarch.tomato.age75_vegetative", tier="ulp", max_ulp=0, fuzz=8)
def tomato_age75_vegetative(B, s):
    seed = _seed(s)
    B.seed(seed)
    B.plantarch_new()
    B.plantarch_load("tomato")
    B.pa_draws_only_inflorescence()
    B.pa_build_replay_blades(seed, 75.0)
    return _observe(B, True, draws_seed=seed, xml_filter=strip_flowers, ids=False)
