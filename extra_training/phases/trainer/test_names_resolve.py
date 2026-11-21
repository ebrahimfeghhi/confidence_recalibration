"""Every name a function body reads must actually resolve.

WHY THIS EXISTS. `import x` only executes module-level code, so it proves nothing about the
bodies of functions: a name used inside one fails at CALL time, not import time. I checked
"16/16 modules import cleanly" and reported the package ready; a separate audit then found 40+
used-but-unimported names, a call with the wrong arity, and an UnboundLocalError on exactly the
live configuration. None of that was visible to an import check.

So this walks every function body and resolves each load against: the module's own definitions
and imports, its parameters and locals, builtins, and `rt.` attributes (which are checked
separately by context.py's SLOTS). Anything left over is a name that will raise when that line
is first reached.
"""
import ast
import builtins
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
MODULES = ("ddp.py", "bootstrap.py", "dataset.py", "trainable.py", "optimiser.py",
           "resume.py", "unit.py", "evaluate.py", "loop.py", "helpers.py",
           "context.py", "main.py", "step_plan.py", "checkpoints.py", "config.py", "cli.py")


def module_scope(tree):
    """Names bound at module level: defs, classes, imports, assignments."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Import):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for al in n.names:
                out.add(al.asname or al.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
    return out


def unresolved(path):
    """[(function, lineno, name)] -- loads that resolve in no enclosing scope.

    SCOPES NEST. A nested def sees its enclosing function's parameters and locals, so the walk
    carries a stack of scopes rather than checking each function in isolation. The first version
    did not, and reported 248 "unresolved" names of which most were simply `rt` read from the
    enclosing closure -- an instrument that cries wolf is worse than none.
    """
    tree = ast.parse(open(path).read())
    bad = []

    def bind(node, into):
        """Every name this node binds, not descending into nested function bodies."""
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                into.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    into.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, ast.ExceptHandler) and n.name:
                into.add(n.name)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                into.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                into.update(n.names)
            elif isinstance(n, ast.comprehension):
                for t in ast.walk(n.target):
                    if isinstance(t, ast.Name):
                        into.add(t.id)

    def walk_fn(fn, outer):
        scope = set(outer)
        for a in (list(fn.args.args) + list(fn.args.kwonlyargs) + list(fn.args.posonlyargs)):
            scope.add(a.arg)
        if fn.args.vararg:
            scope.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            scope.add(fn.args.kwarg.arg)
        bind(fn, scope)

        # SCAN THIS FUNCTION ONLY. A nested def's parameters belong to the nested scope, so
        # walking into it from here reports them as unresolved in the parent -- which is how the
        # first version produced 18 phantom hits in dataset.py from one `def _no_lddt(...)`.
        def own_loads(node):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    continue
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    yield child
                yield from own_loads(child)

        for n in own_loads(fn):
            if n.id not in scope:
                bad.append((fn.name, n.lineno, n.id))
        for n in fn.body:
            for sub in ast.walk(n):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not fn:
                    walk_fn(sub, scope)

    top = set(BUILTINS)
    bind(tree, top)
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            walk_fn(n, top)

    seen, out = set(), []
    for f, ln, nm in sorted(bad, key=lambda x: x[1]):
        if (f, nm) in seen:
            continue
        seen.add((f, nm))
        out.append((f, ln, nm))
    return out


def main():
    total, worst = 0, []
    for m in MODULES:
        p = os.path.join(HERE, m)
        if not os.path.exists(p):
            continue
        bad = unresolved(p)
        total += len(bad)
        mark = "PASS " if not bad else "FAIL "
        print(f"  {mark} {m:<16} {len(bad):>3} unresolved")
        for f, ln, nm in bad[:4]:
            print(f"          {m}:{ln} in {f}(): {nm}")
        if bad:
            worst.append((len(bad), m))
    print()
    if total:
        worst.sort(reverse=True)
        print(f"{total} unresolved name(s). Worst: " +
              ", ".join(f"{m} ({n})" for n, m in worst[:5]))
        print("Each one raises the first time its line is reached. The package cannot run.")
        return 1
    print("every name in every function body resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
