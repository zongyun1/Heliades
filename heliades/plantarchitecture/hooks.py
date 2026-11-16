# SPDX-License-Identifier: LGPL-2.1-or-later
"""Stand-ins for the C++ function pointers stored in the parameter structs.

Helios stores raw function pointers (``LeafPrototype::prototype_function``,
``InflorescenceParameters::flower_prototype_function``, ``PhytomerParameters::phytomer_creation_function``,
...). They matter in two ways that a plain Python callable would not reproduce:

* identity: the flower/fruit prototype caches are keyed by the pointer (H:6700-6704), and
  ``validateTextureSeries`` compares ``prototype_function != GenericLeafPrototype``;
* naming: PyHelios serialises them by name (``"GenericLeafPrototype"``).

A :class:`Hook` is a named, identity-compared slot. The geometry packages (W3-A leaf prototype, W3-B
phytomer, W3-D inflorescence) bind the implementation with ``HOOK.implement(fn)``; until then calling the
hook raises ``NotImplementedError``. Copying a parameter struct copies the reference, like a pointer.
"""
from __future__ import annotations


class Hook:
    __slots__ = ("name", "impl", "lazy")

    def __init__(self, name: str, lazy: str | None = None):
        self.name = name
        self.impl = None
        self.lazy = lazy          # "module:function" bound on first call (avoids an import cycle with params)

    def implement(self, fn):
        self.impl = fn
        return fn

    def __call__(self, *a, **k):
        if self.impl is None and self.lazy:
            import importlib
            mod, fn = self.lazy.split(":")
            self.impl = getattr(importlib.import_module(mod), fn)
        if self.impl is None:
            raise NotImplementedError(f"{self.name} is not ported yet (bind it with {self.name}.implement(fn))")
        return self.impl(*a, **k)

    def __repr__(self):
        return f"<Hook {self.name}{'' if self.impl else ' (unbound)'}>"


# ---- Assets.cpp -----------------------------------------------------------------------------------
GenericLeafPrototype = Hook("GenericLeafPrototype",               # AS:640 (W3-A)
                            lazy="heliades.plantarchitecture.leaf_prototype:GenericLeafPrototype")
TomatoFlowerPrototype = Hook("TomatoFlowerPrototype")            # AS:4080 (W3-D)
TomatoFruitPrototype = Hook("TomatoFruitPrototype")              # AS:4087 (W3-D)
TomatoPhytomerCreationFunction = Hook("TomatoPhytomerCreationFunction")   # AS:4092 (W3-B)


def hook_name(fn) -> str:
    """Name PyHelios would print for a function pointer ("" for null)."""
    if fn is None:
        return ""
    return getattr(fn, "name", getattr(fn, "__name__", repr(fn)))
