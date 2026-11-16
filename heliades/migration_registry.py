# SPDX-License-Identifier: LGPL-2.1-or-later
"""Records which Helios symbols each Heliades function ports.

``@ports("Context::addPatch")`` tags a function; ``migration/inventory.py`` compares the tags against the
Helios public API to report coverage. The tag is metadata only and does not change behaviour.
"""
from __future__ import annotations

PORTED: dict[str, list[str]] = {}


def ports(*helios_symbols: str):
    def deco(fn):
        name = f"{fn.__module__}.{fn.__qualname__}"
        for s in helios_symbols:
            PORTED.setdefault(s, []).append(name)
        fn.__helios_ports__ = helios_symbols
        return fn
    return deco
