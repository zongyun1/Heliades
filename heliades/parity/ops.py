# SPDX-License-Identifier: LGPL-2.1-or-later
"""Per-module backend operations, so parallel porting work never edits a shared adapter.

In ``heliades/parity/suites/<group>/ops_<area>.py`` (``<group>`` is ``core``, ``plantarch``, ...)::

    from ...ops import op

    @op("add_sphere", "helios")
    def _(B, ndiv, center, radius, color=None): return B.ctx.addSphereObject(...)

    @op("add_sphere", "heliades")
    def _(B, ndiv, center, radius, color=None): return B.scene.add_sphere_object(...)

Cases then call ``B.add_sphere(...)`` on either backend. Modules named ``ops_*`` and ``cases_*`` anywhere
under ``heliades.parity.suites`` are imported automatically.
"""
from __future__ import annotations

import importlib
import pkgutil

OPS: dict[str, dict[str, object]] = {}


def op(name: str, backend: str):
    if backend not in ("helios", "heliades"):
        raise ValueError("backend must be 'helios' or 'heliades'")

    def deco(fn):
        OPS.setdefault(name, {})[backend] = fn
        return fn
    return deco


_loaded = False


def load_plugins():
    """Import every ops_* and cases_* module under heliades.parity.suites (idempotent)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import heliades.parity.suites as pkg
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        if not m.ispkg and m.name.rsplit(".", 1)[-1].startswith(("ops_", "cases_")):
            importlib.import_module(m.name)
