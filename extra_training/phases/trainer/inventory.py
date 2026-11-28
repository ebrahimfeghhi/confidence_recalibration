#!/usr/bin/env python
"""What each closure inside phase_k_train.main() actually depends on, measured from the AST.

WHY THIS EXISTS. main() is 8,380 lines holding 68 nested functions in one scope, so a function can
read anything defined before it without saying so. "Pure helper" is therefore a claim that has to
be checked, not read off the name: a parser that quietly consults the filesystem or a module-level
manifest is not extractable as a pure function, and finding that out after moving it is the
expensive way.

For each closure this reports its source span, a hash of its source (for drift detection against
the original after extraction), the names it uses that are NOT its own parameters or locals -- i.e.
what it closes over -- and whether it touches the filesystem, torch, or the distributed backend.

    python -I phases/trainer/inventory.py            # the big ones
    python -I phases/trainer/inventory.py _skipset   # one function, every free name
"""
import ast
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # extra_training/
SRC = os.path.join(ROOT, "phases", "phase_k_train.py")

IO_HINTS = {"open", "os", "glob", "shutil", "json", "csv", "np_load", "Path", "makedirs"}
GPU_HINTS = {"torch", "cuda", "dist", "nn", "F"}


class Scope(ast.NodeVisitor):
    """Names a function body READS that it did not bind itself."""

    def __init__(self, fn):
        self.bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        if fn.args.vararg:
            self.bound.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            self.bound.add(fn.args.kwarg.arg)
        self.free, self.attrs = set(), set()
        for st in fn.body:
            self.visit(st)

    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Store):
            self.bound.add(n.id)
        elif n.id not in self.bound:
            self.free.add(n.id)

    def visit_Attribute(self, n):
        if isinstance(n.value, ast.Name):
            self.attrs.add(f"{n.value.id}.{n.attr}")
        self.generic_visit(n)

    def visit_FunctionDef(self, n):          # nested defs bind their own name
        self.bound.add(n.name)
        self.generic_visit(n)


def walk():
    src = open(SRC).read()
    lines = src.splitlines()
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a, b = node.lineno, (node.end_lineno or node.lineno)
        body = "\n".join(lines[a - 1:b])
        sc = Scope(node)
        free = {n for n in sc.free if not n.startswith("__")}
        out.append({
            "name": node.name, "line": a, "lines": b - a + 1,
            "sha": hashlib.sha256(body.encode()).hexdigest()[:12],
            "free": sorted(free),
            "io": sorted(n for n in free | {x.split(".")[0] for x in sc.attrs} if n in IO_HINTS),
            "gpu": sorted(n for n in free | {x.split(".")[0] for x in sc.attrs} if n in GPU_HINTS),
        })
    return sorted(out, key=lambda d: -d["lines"])


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    rows = walk()
    if want:
        for r in rows:
            if r["name"] == want:
                print(f"{r['name']}  line {r['line']}  {r['lines']} lines  sha {r['sha']}")
                print(f"  filesystem: {r['io'] or 'none'}")
                print(f"  torch/dist: {r['gpu'] or 'none'}")
                print(f"  closes over {len(r['free'])} name(s):")
                for n in r["free"]:
                    print(f"    {n}")
                return 0
        print(f"no function named {want}")
        return 1
    print(f"{len(rows)} functions in {os.path.relpath(SRC, ROOT)}\n")
    print(f"  {'lines':>6} {'line':>6} {'free':>5} {'io':>4} {'gpu':>4}  name")
    for r in rows:
        if r["lines"] < 80:
            continue
        print(f"  {r['lines']:>6} {r['line']:>6} {len(r['free']):>5} "
              f"{'yes' if r['io'] else '-':>4} {'yes' if r['gpu'] else '-':>4}  {r['name']}")
    print("\n'free' = names read from the enclosing 8,380-line scope. Low free + no io/gpu is a "
          "clean extraction;\nhigh free means the function is entangled and needs a contract first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
