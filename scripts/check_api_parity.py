#!/usr/bin/env python3
"""Guard that the engine's public operation surface stays mirrored by the
Node and Python bindings.

The engine's public API is snapshotted in `public-api.txt` (guarded by
`cargo public-api`). This adds a second gate: every user-facing *operation*
method — the free `connect` / `connect_with` and the `Connection::*` /
`Supertable::*` methods — must be listed in `api-parity.txt` as either:

  <method>\tcovered\t<engine signature>   # wrapped in BOTH bindings
  <method>\texempt\t<reason>              # deliberately not wrapped

The check fails (non-zero exit) when:
  (a) an engine method is missing from api-parity.txt        — new, undecided
  (b) a `covered` method has no wrapper in a binding          — not exposed
  (c) a `covered` method's engine signature drifted           — re-verify wrappers
  (d) an api-parity.txt entry is no longer in public-api.txt  — stale

It verifies *presence* (a wrapper fn with the matching name exists) and *signature
stability* (the engine signature hasn't changed since the wrappers were last
verified). It does NOT prove the wrapper's own signature is equivalent — the
types differ by design across languages — but a drift forces a conscious review.

Usage:
  check_api_parity.py            # verify; non-zero exit on any gap
  check_api_parity.py --update   # rewrite api-parity.txt with current signatures
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_API = ROOT / "public-api.txt"
PARITY = ROOT / "api-parity.txt"
NODE_SRC = ROOT / "infino-node" / "src"
PY_SRC = ROOT / "infino-python" / "src"

# Trait-derive / plumbing methods that aren't part of the operation surface.
NOISE = {"clone", "fmt", "eq", "from", "default", "serialize", "deserialize",
         "source", "hash", "cmp", "partial_cmp"}

# Bindings that fold one engine method into a differently-named wrapper.
WRAPPER_OVERRIDES = {"connect_with": "connect"}

# Path prefixes stripped so signatures read cleanly and compare stably.
PREFIXES = ("core::result::", "core::option::", "core::convert::", "core::time::",
            "alloc::vec::", "alloc::string::", "arrow_array::record_batch::",
            "arrow_schema::schema::", "datafusion_expr::expr::", "std::path::",
            "uuid::", "infino::")

_FN = re.compile(r"^pub fn infino::([A-Za-z0-9_:]+)\((.*)\)(?: -> (.*))?$")


def is_tracked(method: str) -> bool:
    """User-facing operation methods only: connect(_with) and Connection/Supertable."""
    if method.split("::")[-1] in NOISE:
        return False
    if method in ("connect", "connect_with"):
        return True
    return method.split("::")[0] in ("Connection", "Supertable")


def normalize_sig(sig: str) -> str:
    for pfx in PREFIXES:
        sig = sig.replace(pfx, "")
    return re.sub(r"\s+", " ", sig).strip()


def parse_public_api() -> dict:
    """{method: normalized_sig} for tracked operation methods."""
    methods = {}
    for line in PUBLIC_API.read_text().splitlines():
        m = _FN.match(line.strip())
        if not m:
            continue
        name, args, ret = m.group(1), m.group(2), m.group(3)
        if not is_tracked(name):
            continue
        sig = f"({args})" + (f" -> {ret}" if ret else "")
        methods[name] = normalize_sig(sig)
    return methods


def load_parity():
    covered, exempt = {}, {}
    if not PARITY.exists():
        return covered, exempt
    for line in PARITY.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        method, status = parts[0], parts[1]
        detail = parts[2] if len(parts) > 2 else ""
        (covered if status == "covered" else exempt)[method] = detail
    return covered, exempt


def wrapper_name(method: str) -> str:
    return WRAPPER_OVERRIDES.get(method, method.split("::")[-1])


def binding_has(src_dir: Path, fn_name: str) -> bool:
    pat = re.compile(rf"\bfn {re.escape(fn_name)}\b")
    return any(pat.search(f.read_text()) for f in src_dir.rglob("*.rs"))


def render(covered_sigs: dict, exempt: dict) -> str:
    lines = [
        "# Parity map for the engine's public operation surface — enforced by",
        "# scripts/check_api_parity.py. Every method in public-api.txt must be here as",
        "# `covered` (wrapped in Node AND Python) or `exempt` (with a reason).",
        "# Regenerate signatures with `make api-parity-update`.",
        "#",
        "# <method>\\tcovered\\t<engine signature>",
        "# <method>\\texempt\\t<reason>",
        "",
    ]
    for method in sorted(covered_sigs):
        lines.append(f"{method}\tcovered\t{covered_sigs[method]}")
    lines.append("")
    for method in sorted(exempt):
        lines.append(f"{method}\texempt\t{exempt[method]}")
    return "\n".join(lines) + "\n"


def update() -> None:
    engine = parse_public_api()
    _, exempt = load_parity()
    covered_sigs = {m: sig for m, sig in engine.items() if m not in exempt}
    # Drop exempt entries that no longer exist in the engine.
    exempt = {m: r for m, r in exempt.items() if m in engine}
    PARITY.write_text(render(covered_sigs, exempt))
    print(f"wrote {PARITY.relative_to(ROOT)} — {len(covered_sigs)} covered, {len(exempt)} exempt")


def check() -> int:
    engine = parse_public_api()
    covered, exempt = load_parity()
    mapped = set(covered) | set(exempt)
    errors = []

    for method in sorted(engine):
        if method not in mapped:
            errors.append(
                f"NEW public method not in api-parity.txt: {method}\n"
                f"      {engine[method]}\n"
                f"    -> wrap it in Node + Python and mark it `covered`, or add it as `exempt` with a reason."
            )
    for method in sorted(mapped):
        if method not in engine:
            errors.append(f"STALE api-parity.txt entry (not in public-api.txt): {method}  -> remove it.")

    for method, recorded in sorted(covered.items()):
        if method not in engine:
            continue
        name = wrapper_name(method)
        if not binding_has(NODE_SRC, name):
            errors.append(f"COVERED but no Node wrapper `fn {name}`: {method}")
        if not binding_has(PY_SRC, name):
            errors.append(f"COVERED but no Python wrapper `fn {name}`: {method}")
        if normalize_sig(recorded) != engine[method]:
            errors.append(
                f"SIGNATURE CHANGED: {method}\n"
                f"      engine: {engine[method]}\n"
                f"      recorded: {normalize_sig(recorded)}\n"
                f"    -> re-verify the Node + Python wrappers, then run `make api-parity-update`."
            )

    if errors:
        print("API parity check FAILED:\n", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"API parity OK — {len(covered)} covered, {len(exempt)} exempt.")
    return 0


if __name__ == "__main__":
    if "--update" in sys.argv[1:]:
        update()
    else:
        sys.exit(check())
