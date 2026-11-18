# SPDX-License-Identifier: LGPL-2.1-or-later
"""Fruit / ornamental tree (W5-E) operations for both backends (the tree build / summary ops are W5-D's,
``ops_nut_trees``).

* ``global_seed(seed)``: seed the process-wide generator behind ``helios::randu()`` (PyHelios
  ``Global.seedRandomGenerator``; Heliades ``core.random.seed_global_random_generator``). The redbud callback draws
  from it.
* ``ft_prim_data()``: per primitive (hidden ones included) its material label and every primitive-data entry (label,
  value; file paths reduced to their base name, since the two asset roots differ) as a stream.
"""
from __future__ import annotations

import os
import zlib

import numpy as np

from ...ops import op

f32 = np.float32


@op("global_seed", "helios")
def _(B, seed):
    from pyhelios import Global
    Global.seedRandomGenerator(int(seed))


@op("global_seed", "heliades")
def _(B, seed):
    from ....core.random import seed_global_random_generator
    seed_global_random_generator(int(seed))


def _code(s) -> float:
    s = str(s)
    if "/" in s:
        s = os.path.basename(s)
    return float(zlib.crc32(s.encode()) & 0xFFFFFF)


def _flat(v, out):
    if isinstance(v, str):
        out.append(_code(v))
    elif isinstance(v, (list, tuple, np.ndarray)) or hasattr(v, "__iter__"):
        for x in v:
            _flat(x, out)
    elif hasattr(v, "x"):                    # PyHelios vec types
        for c in ("x", "y", "z", "w"):
            if hasattr(v, c):
                out.append(float(f32(getattr(v, c))))
    else:
        out.append(float(f32(v)))


def _encode(rows) -> list:
    out = [float(len(rows))]
    for u, material, data in rows:
        out += [float(u), _code(material), float(len(data))]
        for label, value in data:
            out.append(_code(label))
            _flat(value, out)
    return out


@op("ft_prim_data", "helios")
def _(B):
    from ...backends import _all_ids_including_hidden
    ctx = B.ctx
    uu = _all_ids_including_hidden(ctx.getPrimitiveCount(), ctx.doesPrimitiveExist, 0)
    labels = ctx.getPrimitiveMaterialLabel(uu) if uu else []
    return _encode([(u, m, [(lab, ctx.getPrimitiveData(u, lab)) for lab in sorted(ctx.listPrimitiveData(u))])
                    for u, m in zip(uu, labels)])


@op("ft_prim_data", "heliades")
def _(B):
    sc = B.scene
    uu = sc.all_uuids(include_hidden=True)
    return _encode([(u, sc.get_primitive_material_label(u),
                     [(lab, sc.get_primitive_data(u, lab)) for lab in sorted(sc.list_primitive_data(u))]) for u in uu])
