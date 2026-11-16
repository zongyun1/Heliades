# SPDX-License-Identifier: LGPL-2.1-or-later
"""Differential testing of Heliades against a pinned Helios build (the oracle)."""
from __future__ import annotations

from .compare import Report, compare_distributions, compare_snapshots, compare_streams  # noqa: F401
from .snapshot import Snapshot


def run_case(name: str, seed: int = 0):
    from .ops import load_plugins
    load_plugins()
    from .backends import ForgeBackend, HeliosBackend
    from .cases import CASES
    c = CASES[name]
    a = c["fn"](HeliosBackend(), seed)
    b = c["fn"](ForgeBackend(), seed)
    if isinstance(a, Snapshot):
        return compare_snapshots(b, a, max_ulp=c["max_ulp"])
    return compare_streams(b, a, max_ulp=0 if c["tier"] == "exact" else c["max_ulp"])


def run_all(patterns=None, verbose=False) -> list[dict]:
    from .cases import CASES
    from .ops import load_plugins
    load_plugins()
    results = []
    for name, c in CASES.items():
        if patterns and not any(p in name for p in patterns):
            continue
        reps = [run_case(name, s) for s in range(c["fuzz"])]
        bad = [(s, r) for s, r in enumerate(reps) if not r.ok]
        res = {"case": name, "tier": c["tier"], "ok": not bad, "seeds": len(reps),
               "checks": reps[0].checks, "failure": None}
        if bad:
            s, r = bad[0]
            res["failure"] = f"seed {s}: {r.first_divergence}"
            res["checks"] = r.checks
        results.append(res)
        if verbose:
            print(f"{'PASS' if res['ok'] else 'FAIL'}  {name:<32} [{c['tier']}, {len(reps)} seeds]"
                  + (f"  {res['failure']}" if bad else ""))
    return results
