# SPDX-License-Identifier: LGPL-2.1-or-later
"""Rice operations (W5-A), both backends.

* ``rice_color_textures(oids)``: for every primitive of each object, the file name (basename) of its color texture
  (``getPrimitiveColorTextureFile``: the stage image a leaf texture stage gave it, else its own texture), encoded as
  one stream: per primitive the bytes of the name, then -1.
"""
from __future__ import annotations

import os

from ...ops import op


def _encode(names):
    out = []
    for n in names:
        out += [float(b) for b in os.path.basename(n).encode()] + [-1.0]
    return out


@op("rice_color_textures", "helios")
def _(B, oids):
    names = []
    for oid in oids:
        for u in B.ctx.getObjectPrimitiveUUIDs(int(oid)):
            names.append(B.ctx.getPrimitiveColorTextureFile(int(u)))
    return _encode(names)


@op("rice_color_textures", "heliades")
def _(B, oids):
    s = B.scene
    names = [s.primitive_color_texture_file(u) for oid in oids for u in s.object_primitive_uuids(int(oid))]
    return _encode(names)
