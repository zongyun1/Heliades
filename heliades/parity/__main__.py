# SPDX-License-Identifier: LGPL-2.1-or-later
"""Run parity cases: ``python -m heliades.parity [substring ...] [--json out.json]``."""
from __future__ import annotations

import json
import sys

from . import run_all


def main(argv):
    out = None
    if "--json" in argv:
        i = argv.index("--json")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    results = run_all(argv or None, verbose=True)
    n_ok = sum(r["ok"] for r in results)
    print(f"\n{n_ok}/{len(results)} cases pass")
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)
    return 0 if n_ok == len(results) else 1


sys.exit(main(sys.argv[1:]))
