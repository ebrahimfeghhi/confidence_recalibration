#!/usr/bin/env python
"""Every resume-guard field that is COMPARED must also be SAVED. Static, CPU, no model.

WHY THIS EXISTS. phase_k_train.py keeps two dictionaries: the settings a run compares against a
resume bundle, and the settings it writes INTO the bundle. Adding a flag to one and not the other
leaves the guard half-wired -- a bundle moved without its args.json carries no record of the
setting and falls through to a migration default, so a resume that genuinely changed the objective
reads as a match.

This has now happened five times, and the file's own comments say so: frame_direction ("Third time
a guard field has been half-wired"), then pose_perblock_share, l2_sp and three ranking fields
("FOURTH TIME, five fields at once"), then pae_row_frac/mass/floor. Each was caught by a human
reading the diff. This makes it a gate instead.

Usage:  python -I run_logging/test_guard_fields.py
"""
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "phases", "phase_k_train.py")

# NOT FLAGS. These live in the compared or saved dict but are derived, not settings: they cannot
# be "missing" from the other side because they are not read from argparse at all.
DERIVED = {"drift_history", "manifest_digest", "world", "epochs", "crop_size",
           # PROVENANCE, not settings: written on save and only ever reported back.
           "commit", "written",
           # COVERED BY manifest_digest, which is computed over the population AFTER the skip
           # list is applied -- so a changed skip list already shows up as a digest mismatch.
           "manifest_skip"}

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def dict_keys_at(tree, src, want_value_prefix):
    """Every dict literal in the file, as (lineno, {key: is_attr_of_a}), largest first."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys, dupes = {}, []
        for k, v in zip(node.keys, node.values):
            if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
                continue
            if k.value in keys:
                dupes.append(k.value)
            keys[k.value] = (isinstance(v, ast.Attribute)
                             and isinstance(v.value, ast.Name)
                             and v.value.id == want_value_prefix)
        if len(keys) >= 10:
            out.append((node.lineno, keys, dupes))
    return sorted(out, key=lambda t: -len(t[1]))


def main():
    src = io.open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    cands = dict_keys_at(tree, src, "a")
    check("found at least two large settings dicts", len(cands) >= 2, f"{len(cands)}")
    if len(cands) < 2:
        return 1
    # the SAVED one is the dict holding drift_history; the COMPARED one is the other big one
    saved = next((c for c in cands if "drift_history" in c[1]), None)
    compared = next((c for c in cands if c is not saved and "lr_decay_shape" in c[1]), None)
    check("identified the saved guard dict", saved is not None,
          f"line {saved[0]}, {len(saved[1])} keys" if saved else "not found")
    check("identified the compared settings dict", compared is not None,
          f"line {compared[0]}, {len(compared[1])} keys" if compared else "not found")
    if not saved or not compared:
        return 1

    missing = sorted(k for k in compared[1] if k not in saved[1] and k not in DERIVED)
    check("every COMPARED field is also SAVED", not missing,
          f"half-wired: {missing}" if missing else
          f"{len(compared[1])} compared, {len(saved[1])} saved")

    extra = sorted(k for k in saved[1] if k not in compared[1] and k not in DERIVED)
    check("every SAVED flag is also COMPARED", not extra,
          f"saved but never checked: {extra}" if extra else "")

    for lbl, (ln, keys, dupes) in (("saved", saved), ("compared", compared)):
        check(f"the {lbl} dict has no duplicate keys", not dupes,
              f"line {ln}: {sorted(set(dupes))}" if dupes else "")

    for f in ("pae_row_frac", "pae_row_mass", "pae_row_floor"):
        check(f"{f} is in BOTH dicts", f in saved[1] and f in compared[1],
              f"saved={f in saved[1]} compared={f in compared[1]}")

    # the migration defaults must cover anything compared, or an old bundle reads as a mismatch
    pre = next((n for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_PRE_CROP" for t in n.targets)), None)
    check("_PRE_CROP migration block found", pre is not None)
    if pre is not None and isinstance(pre.value, ast.Dict):
        pk = {k.value for k in pre.value.keys
              if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        for f in ("pae_row_frac", "pae_row_mass", "pae_row_floor"):
            check(f"{f} has a migration default for pre-flag bundles", f in pk)

    print(f"\n  {len(FAIL)} failure(s)" + (f": {FAIL}" if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
