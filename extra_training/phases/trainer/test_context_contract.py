"""Every `rt.<attr>` a module reads must be written by some earlier stage.

THIS IS THE TEST THAT WOULD HAVE STOPPED ME CLAIMING THE PACKAGE WAS READY. The other tests ask
whether the code parses, whether names resolve lexically, and whether the extraction is faithful.
All three pass on a package that cannot complete a single step, because the thing missing is not
syntax or imports -- it is the ~1,300 lines of `main()` BETWEEN the extracted regions, which is
where TERMS, the module handles, the caches and the per-run callbacks are built.

`Ctx.SLOTS` does not help: it is a tuple, not `__slots__`, and nothing validates against it. An
unset attribute is an AttributeError at the moment of first read, which on this code is somewhere
inside the first optimiser step.

The test prints the count and the first offenders. It fails while any remain.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ("ddp.py", "bootstrap.py", "dataset.py", "trainable.py", "optimiser.py", "resume.py",
           "unit.py", "evaluate.py", "loop.py", "helpers.py", "main.py", "step_plan.py",
           "runtime.py")

#: supplied by the interpreter, not by a stage
EXEMPT = {"__file__", "__name__", "__doc__"}


def scan():
    reads, writes = {}, set()
    for m in MODULES:
        p = os.path.join(HERE, m)
        if not os.path.exists(p):
            continue
        tree = ast.parse(open(p).read())
        for n in ast.walk(tree):
            if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                    and n.value.id == "rt"):
                if isinstance(n.ctx, ast.Store):
                    writes.add(n.attr)
                else:
                    reads.setdefault(n.attr, set()).add(m)
        if m == "main.py":
            for n in ast.walk(tree):
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "Ctx":
                    for kw in n.keywords:
                        if kw.arg:
                            writes.add(kw.arg)
                # attributes set through setattr(rt, name, value) in a wiring loop
                if (isinstance(n, ast.Call) and getattr(n.func, "id", "") == "setattr"
                        and n.args and getattr(n.args[0], "id", "") == "rt"):
                    writes.add("*setattr*")
        # A STAGE MAY EXPORT A WHOLE MODULE'S TOP LEVEL: `for _n in dir(constants): setattr(...)`.
        # The module is importable here, so its public names are statically knowable.
        for n in ast.walk(tree):
            if (isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                    and getattr(n.iter.func, "id", "") == "dir" and n.iter.args
                    and isinstance(n.iter.args[0], ast.Name)):
                modname = n.iter.args[0].id
                mp = os.path.join(HERE, modname + ".py")
                if os.path.exists(mp):
                    mt = ast.parse(open(mp).read())
                    for mn in ast.walk(mt):
                        if isinstance(mn, (ast.FunctionDef, ast.ClassDef)):
                            writes.add(mn.name)
                        elif isinstance(mn, (ast.Import, ast.ImportFrom)):
                            for al in mn.names:
                                writes.add((al.asname or al.name).split(".")[0])
                        elif isinstance(mn, ast.Name) and isinstance(mn.ctx, ast.Store):
                            writes.add(mn.id)

        # A STAGE MAY EXPORT A LITERAL LIST OF NAMES. runtime.py ends with
        #     for _n in ('TERMS', 'BC', ...): setattr(rt, _n, locals()[_n])
        # The names are literals, so they are statically visible and count as writes. A dynamic
        # vars() export would not, which is exactly why the export is written out by name.
        for n in ast.walk(tree):
            if (isinstance(n, ast.For) and isinstance(n.iter, (ast.Tuple, ast.List))
                    and any(isinstance(b, ast.Expr) and isinstance(b.value, ast.Call)
                            and getattr(b.value.func, "id", "") == "setattr"
                            for b in ast.walk(n))):
                for e in n.iter.elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        writes.add(e.value)
    return reads, writes


def main():
    reads, writes = scan()
    missing = {k: v for k, v in sorted(reads.items())
               if k not in writes and k not in EXEMPT}
    print(f"  rt attributes read:     {len(reads)}")
    print(f"  rt attributes written:  {len(writes)}")
    print(f"  READ BUT NEVER WRITTEN: {len(missing)}")
    if missing:
        print()
        for k in list(missing)[:12]:
            print(f"    {k:<26} read in {', '.join(sorted(missing[k]))[:52]}")
        if len(missing) > 12:
            print(f"    ... and {len(missing) - 12} more")
        print()
        print("  Each is an AttributeError at first read. These are the glue between the")
        print("  extracted regions -- the part of main() that builds TERMS, the module handles,")
        print("  the caches and the per-run callbacks. NOT RUNNABLE until this is 0.")
        return 1
    print("\n  every context attribute read is written by some stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
