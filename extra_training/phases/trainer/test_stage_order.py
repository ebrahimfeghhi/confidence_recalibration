"""A context attribute must be written by a stage that runs BEFORE its reader.

WHY THIS IS A DIFFERENT TEST. `test_context_contract.py` proves a writer exists somewhere in the
package. It is order-blind: a name written by the last stage and read by the first passes it
cleanly. Review on issue #9 drew that distinction, and it is the difference between a static scan
and an initialisation that fails at the right moment.

This walks `main()` in source order, tracking which stage each call belongs to, and checks that
every `rt.require(...)` names only attributes some earlier stage provides. It also checks the
require gates exist at all -- a stage boundary with no gate is a place where a misordered export
would surface deep inside a GPU run instead of at startup.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: stage module -> the attributes it exports, read from its own literal export tuple
def exports_of(mod):
    p = os.path.join(HERE, mod)
    if not os.path.exists(p):
        return set()
    out = set()
    tree = ast.parse(open(p).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.For) and isinstance(n.iter, (ast.Tuple, ast.List)):
            if any(isinstance(b, ast.Expr) and isinstance(b.value, ast.Call)
                   and getattr(b.value.func, "id", "") == "setattr" for b in ast.walk(n)):
                for e in n.iter.elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        out.add(e.value)
    # a module exported wholesale (constants.py) contributes its public top level
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            # MODULE-LEVEL ASSIGNMENTS COUNT. constants.py defines TERMS and ROOT this way, and
            # omitting them made the first gate report them as not yet provided when they are
            # set two lines earlier.
            out.add(n.id)
    return out


def main():
    src = open(os.path.join(HERE, "main.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "main")

    provided, fails, gates = set(), [], 0
    # names the entrypoint sets directly, in order
    for stmt in fn.body:
        for n in ast.walk(stmt):
            # a stage call: <mod>.<fn>(...) -- credit that module's exports
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and isinstance(n.func.value, ast.Name):
                mod = n.func.value.id.replace("_mod", "")
                for cand in (mod + ".py", mod.replace("config", "config") + ".py"):
                    provided |= exports_of(cand)
            # a wholesale module export: for _n in dir(constants)
            if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) \
                    and getattr(n.iter.func, "id", "") == "dir" and n.iter.args:
                provided |= exports_of(getattr(n.iter.args[0], "id", "") + ".py")
            # direct assignment onto the context
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                    and n.value.id == "rt" and isinstance(n.ctx, ast.Store):
                provided.add(n.attr)
            # the gate itself
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == "require" \
                    and getattr(n.func.value, "id", "") == "rt":
                gates += 1
                want = [a.value for a in n.args
                        if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                missing = [w for w in want if w not in provided]
                label = f"gate {gates} ({', '.join(want[:3])}...)"
                print(("  PASS  " if not missing else "  FAIL  ") + label +
                      ("" if not missing else f"   not yet provided: {missing}"))
                if missing:
                    fails.append(label)

    print()
    if gates == 0:
        print("no rt.require() gates in main() -- a misordered export would surface "
              "inside the run, not at startup")
        return 1
    if fails:
        print(f"FAILED {len(fails)} of {gates} gates")
        return 1
    print(f"all {gates} stage gates satisfied in source order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
