"""No module may read a flag `trainer/cli.py` cannot express.

Such a read is not a dead branch that evaluates false -- it raises AttributeError the first time
its line is reached. So every one is a landmine, and this test is the definition of whether the
package can run at all.

The automatic pass (`lean.py`) removes the shapes it can prove dead: whole `if` blocks, guard-dict
entries, banner prints, literal elements. What it cannot prove, it leaves: ternaries and keyword
arguments, where removing the reference means deciding what the expression should do instead.
Those need a person, and this test counts them so the number is visible rather than discovered on
a GPU.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lean  # noqa: E402

MODULES = ("ddp.py", "bootstrap.py", "dataset.py", "trainable.py", "optimiser.py", "resume.py",
           "unit.py", "evaluate.py", "loop.py", "helpers.py", "step_plan.py", "config.py",
           "checkpoints.py", "main.py", "context.py")


def main():
    total, rows = 0, []
    for m in MODULES:
        p = os.path.join(HERE, m)
        if not os.path.exists(p):
            continue
        left = lean.culled_in(ast.parse(open(p).read()))
        n = sum(left.values())
        total += n
        print(("  PASS  " if not n else "  FAIL  ") + f"{m:<16} {n:>3}" +
              ("" if not n else "   " + ", ".join(sorted(left))[:92]))
        if n:
            rows.append((n, m))
    print()
    if total:
        rows.sort(reverse=True)
        print(f"{total} read(s) of a flag cli.py does not define. Each raises AttributeError the")
        print("first time its line runs, so the package cannot complete a step.")
        print("Worst: " + ", ".join(f"{m} ({n})" for n, m in rows[:5]))
        return 1
    print("no module reads a culled flag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
