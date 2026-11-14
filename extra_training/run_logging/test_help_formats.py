#!/usr/bin/env python
"""Every --help string must survive argparse's %-formatting.

    python -I run_logging/test_help_formats.py

WHY THIS EXISTS. argparse runs `help % params` on EVERY action when --help is requested, so one
literal percent anywhere kills --help for the whole file, not just that flag. It has happened
three times in this project: `epoch % n` in phase_k_eval's --crop-pool, and "2.7% of", "0.6% of"
and "~2% of" in phase_k_train. The last one mattered: a chain script used
`phase_k_train.py --help | grep -- --pae-split-accept` as its readiness check, and a broken
--help made a landed flag look absent, so the run would not launch.

IT PARSES RATHER THAN RUNS, so it costs no imports and works on files that need a GPU, an env var
or a label store before they will start.
"""
import ast
import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(ROOT, "phases", "*.py"))
               + glob.glob(os.path.join(ROOT, "run_logging", "*.py"))
               + glob.glob(os.path.join(ROOT, "eval", "*.py")))

bad = []
checked = 0
for f in FILES:
    try:
        tree = ast.parse(io.open(f, encoding="utf-8").read())
    except SyntaxError as e:
        bad.append((f, "?", f"will not parse: {e}", []))
        continue
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        name = (node.args[0].value if node.args and isinstance(node.args[0], ast.Constant)
                else "<positional>")
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                h = ast.literal_eval(kw.value)
            except Exception:
                continue          # an f-string or a name: argparse will format whatever it gets
            checked += 1
            try:
                # THE SAME SUBSTITUTION argparse performs, with the keys it supplies.
                h % {"default": 0, "prog": "p", "const": 0, "type": "t",
                     "choices": "c", "metavar": "m"}
            except Exception as e:
                ctx = [h[max(0, m.start() - 14):m.start() + 14] for m in re.finditer("%", h)]
                bad.append((os.path.relpath(f, ROOT), name, str(e), ctx[:4]))

for f, name, err, ctx in bad:
    print(f"FAIL {f} {name}: {err}")
    for c in ctx:
        print(f"       ...{c}...")
print(f"checked {checked} help string(s) across {len(FILES)} file(s); {len(bad)} broken")
if bad:
    print("FIX: double the literal percent -- '2.7%' becomes '2.7%%'.")
sys.exit(1 if bad else 0)
