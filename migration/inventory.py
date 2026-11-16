# SPDX-License-Identifier: LGPL-2.1-or-later
"""Inventory of the Helios public API at the pinned baseline, and Heliades's coverage of it.

  python migration/inventory.py            -> migration/inventory.json, migration/STATUS.md

Public API = public member functions of classes declared in core/include/*.h and
plugins/*/include/*.h, plus free functions in namespace helios (global.h). Overloads are collapsed
to one name. The header parser is deliberately simple (brace and access tracking after stripping
comments and strings); it is for progress accounting, not for code generation.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELIOS = ROOT / "third_party" / "pyhelios_src" / "helios-core"
SKIP_PLUGINS = {"projectbuilder", "visualizer"}  # replaced, not ported (see PORTING.md)

_comment = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_string = re.compile(r'"(?:\\.|[^"\\])*"')
_cls = re.compile(r"\b(class|struct)\s+(\w+)\s*(?:final\s*)?(?::[^{;]*)?\{")
_meth = re.compile(r"(?:^|[;{}\s])(~?\w+)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?(?:=\s*\w+\s*)?[;{]")
_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "static_assert", "operator", "decltype", "catch"}


def parse_header(text: str) -> dict[str, set[str]]:
    """{class_name: {public method names}} for top-level classes; '<free>' for namespace functions."""
    text = _string.sub('""', _comment.sub("", text))
    out: dict[str, set[str]] = defaultdict(set)
    i, n = 0, len(text)
    depth_stack: list[tuple[str, int, bool]] = []  # (class, depth_at_open, is_public)
    depth = 0
    seg_start = 0
    ns_interior = {0}  # depths whose scope is a namespace (or file scope)
    while i < n:
        m = _cls.match(text, i) if text[i] in "cs" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")) else None
        if m and not depth_stack:
            depth_stack.append((m.group(2), depth, m.group(1) == "struct"))
            depth += 1
            i = m.end()
            seg_start = i
            continue
        c = text[i]
        if c == "{":
            seg = text[seg_start:i]
            if depth_stack and depth == depth_stack[-1][1] + 1:
                _collect(seg + "{", depth_stack, out)
            elif not depth_stack and depth in ns_interior:
                if re.search(r"\bnamespace\b[\w\s:]*$", seg) or re.search(r'extern\s*""\s*$', seg):
                    ns_interior.add(depth + 1)
                else:
                    _collect_free(seg + "{", out)
            depth += 1
            seg_start = i + 1
        elif c == "}":
            ns_interior.discard(depth)
            depth -= 1
            if depth_stack and depth == depth_stack[-1][1]:
                depth_stack.pop()
            seg_start = i + 1
        elif c == ";":
            if depth_stack and depth == depth_stack[-1][1] + 1:
                _collect(text[seg_start:i + 1], depth_stack, out)
            elif not depth_stack and depth in ns_interior:
                _collect_free(text[seg_start:i + 1], out)
            seg_start = i + 1
        elif c == ":" and depth_stack and depth == depth_stack[-1][1] + 1:
            word = text[seg_start:i].strip().split()[-1:] or [""]
            if word[0] in ("public", "private", "protected") and text[i + 1:i + 2] != ":":
                cls, d0, _ = depth_stack[-1]
                depth_stack[-1] = (cls, d0, word[0] == "public")
                seg_start = i + 1
        i += 1
    return out


def _collect(seg: str, stack, out):
    cls, _, public = stack[-1]
    if not public:
        return
    for m in _meth.finditer(" " + seg):
        name = m.group(1)
        if name not in _KEYWORDS and name != cls and not name.startswith("~"):
            out[cls].add(name)


def _collect_free(seg: str, out):
    for m in _meth.finditer(" " + seg):
        if m.group(1) not in _KEYWORDS:
            out["helios"].add(m.group(1))


def helios_api() -> dict[str, dict[str, list[str]]]:
    """{module: {class: [methods]}} at the pinned baseline."""
    mods = {"core": sorted((HELIOS / "core" / "include").glob("*.h"))}
    for p in sorted((HELIOS / "plugins").iterdir()):
        if p.is_dir() and p.name not in SKIP_PLUGINS:
            mods[p.name] = sorted((p / "include").glob("*.h"))
    api = {}
    for mod, headers in mods.items():
        classes: dict[str, set[str]] = defaultdict(set)
        for h in headers:
            for cls, ms in parse_header(h.read_text(errors="ignore")).items():
                classes[cls] |= ms
        api[mod] = {c: sorted(m) for c, m in classes.items() if m}
    return api


def ported() -> dict[str, list[str]]:
    sys.path.insert(0, str(ROOT))
    import heliades
    from heliades.migration_registry import PORTED
    for m in pkgutil.walk_packages(heliades.__path__, "heliades."):
        if ".parity" not in m.name:
            importlib.import_module(m.name)
    return PORTED


def main():
    base = subprocess.run(["git", "-C", str(HELIOS), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    api, done = helios_api(), ported()
    done_names = {k.split("(")[0] for k in done}
    rows, inv = [], {"baseline": base, "modules": {}}
    tot_all = tot_done = 0
    for mod, classes in api.items():
        entries = []
        for cls, ms in classes.items():
            for m in ms:
                sym = f"{cls}::{m}"
                entries.append({"symbol": sym, "ported": sym in done_names, "by": done.get(sym, [])})
        k = sum(e["ported"] for e in entries)
        inv["modules"][mod] = entries
        rows.append((mod, len(entries), k))
        tot_all += len(entries)
        tot_done += k
    (ROOT / "migration" / "inventory.json").write_text(json.dumps(inv, indent=1))
    lines = ["# Migration status", "", f"Helios baseline: `{base}`. Generated by `migration/inventory.py`; do not edit.", "",
             f"**{tot_done} / {tot_all} public API symbols ported ({100 * tot_done / max(tot_all, 1):.1f}%)**", "",
             "| module | symbols | ported | % |", "|---|---|---|---|"]
    lines += [f"| {m} | {n} | {k} | {100 * k / max(n, 1):.0f} |" for m, n, k in rows]
    unknown = sorted(done_names - {e["symbol"] for es in inv["modules"].values() for e in es})
    if unknown:
        lines += ["", "Tags that match no parsed Helios symbol (check spelling or parser): " + ", ".join(f"`{u}`" for u in unknown)]
    (ROOT / "migration" / "STATUS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
