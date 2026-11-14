#!/usr/bin/env python
"""What is actually reachable from the entry points that run today.

WHY NOT grep. docs/code_inventory.json counts how many files MENTION a module's name, which catches
documentation and comments and misses nothing-but-imports. That is a ranking signal, not a call
graph: a module named only in a comment scores 1, and a module imported under a different alias
scores 0. Deleting on that basis would remove live code.

This walks real edges instead:
  ast.Import / ast.ImportFrom            python imports, resolved against phases/ and run_logging/
  subprocess / os.system string args     the trainer shells out to the evaluator
  .py and .sh paths inside shell scripts the launcher invokes python by path

Roots are the things that run today: the launcher, the trainer, the evaluator and its sharded
driver, the two canonical plotters and their refresh script, and every test (a test is reachable by
definition -- it is run by hand).

Anything NOT reachable is a CANDIDATE. Dynamic imports, importlib and names built at runtime are
invisible to this, so the output says "no static edge found", not "dead".
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEARCH = ["phases", "run_logging", "shell_scripts"]
# THE LIVE LAUNCHERS ARE ROOTS. Review of b70fa96b: this list named launch_default_arm.sh but
# not the scripts that actually start today's arm, so reachability computed from it would have
# called the live memory gate unreachable.
ROOTS = ["shell_scripts/launch_fullhead_5site.sh", "shell_scripts/mem_probe_fullhead.sh",
         "shell_scripts/lr_sweep_ep5.sh",
         "shell_scripts/launch_default_arm.sh", "phases/phase_k_train.py",
         "phases/phase_k_eval.py", "phases/phase_k_eval_sharded.py",
         "run_logging/refresh_plots.sh", "run_logging/plot_rel_arm.py",
         "run_logging/plot_eval_series.py"]


def collisions():
    """Basenames that exist in more than one searched directory.

    index() resolves a bare name to ONE path. Where a basename is ambiguous that choice is
    arbitrary, so an edge drawn to it may point at the wrong file and reachability computed
    through it is not evidence. Reported rather than silently resolved.
    """
    seen = {}
    for d in SEARCH:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith((".py", ".sh")):
                seen.setdefault(f, []).append(f"{d}/{f}")
    return {k: v for k, v in seen.items() if len(v) > 1}


def index():
    """{module_or_basename: relpath} for everything we could resolve an edge to.

    FIRST PATH WINS, and that is a real limitation, not a detail: see collisions().
    """
    out = {}
    for d in SEARCH:
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith((".py", ".sh")):
                rel = f"{d}/{f}"
                out.setdefault(f, rel)
                if f.endswith(".py"):
                    out.setdefault(f[:-3], rel)
    return out


def edges(rel, idx):
    p = os.path.join(ROOT, rel)
    try:
        src = open(p, errors="replace").read()
    except OSError:
        return set()
    out = set()
    if rel.endswith(".py"):
        try:
            for n in ast.walk(ast.parse(src)):
                if isinstance(n, ast.Import):
                    for al in n.names:
                        out.add(al.name.split(".")[0])
                elif isinstance(n, ast.ImportFrom) and n.module:
                    out.add(n.module.split(".")[0])
        except SyntaxError:
            pass
    # shelled-out scripts, in both .py and .sh: any bare filename that we know
    for m in re.finditer(r"[\w./-]+\.(?:py|sh)", src):
        out.add(os.path.basename(m.group(0)))
    return {idx[e] for e in out if e in idx}


def main():
    idx = index()
    tests = [v for k, v in idx.items() if k.startswith("test_") and k.endswith(".py")]
    seen, stack = set(), list(ROOTS) + tests
    while stack:
        cur = stack.pop()
        if cur in seen or not os.path.exists(os.path.join(ROOT, cur)):
            continue
        seen.add(cur)
        stack.extend(edges(cur, idx))
    allf = sorted({v for v in idx.values()})
    unreach = sorted(set(allf) - seen)
    print(f"roots: {len(ROOTS)} + {len(tests)} tests")
    print(f"reachable: {len(seen)} of {len(allf)} files in {', '.join(SEARCH)}")
    col = collisions()
    if col:
        print(f"AMBIGUOUS BASENAMES: {len(col)} -- any edge through these is UNKNOWN, "
              f"not safe-to-remove")
        for k, v in sorted(col.items()):
            print(f"  {k}: {v}")
        print()
    print("THIS IS TRIAGE, NOT A REACHABILITY PROOF. It regex-scans text, so comments and "
          "strings make false edges; it resolves dotted imports by first component; and it "
          "cannot see subprocess calls or shell variables. Absence of an edge is a question, "
          "never a verdict.\n")
    print(f"NO STATIC EDGE: {len(unreach)}\n")
    by = {}
    for f in unreach:
        by.setdefault(f.split("/")[0], []).append(f)
    for d, fs in sorted(by.items()):
        print(f"  {d}/  {len(fs)} file(s)")
    if "--list" in sys.argv:
        print()
        for f in unreach:
            n = len(open(os.path.join(ROOT, f), errors="replace").read().splitlines())
            print(f"  {n:>6}  {f}")
    print("\nDynamic imports and runtime-built names are invisible here, so this is a candidate "
          "list,\nnot a dead list. Check each before removing it from the copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
