"""The detectors must FAIL on a planted fault. Otherwise they only prove they ran.

WHY. Two tests in this suite reported clean while the thing they exist to catch was present:
`test_no_culled_flags.py` had a hand-written flag list that omitted `crop_log`, and both it and
the context contract had a hand-written module tuple that omitted `derive.py`. Both passed on a
package with a live `AttributeError` in it.

So each detector is run here against a temporary copy of the package with a fault planted, and
must report it. A detector that cannot be made to fail is not evidence.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(test, pkg):
    """Run one detector against a package directory. Returns (rc, output)."""
    p = subprocess.run([PY, "-I", os.path.join(pkg, test)],
                       capture_output=True, text=True, cwd=pkg)
    return p.returncode, p.stdout + p.stderr


def planted(fault):
    """A temp copy of the package with `fault(dir)` applied."""
    d = tempfile.mkdtemp(prefix="trainer_fault_")
    pkg = os.path.join(d, "trainer")
    shutil.copytree(HERE, pkg)
    fault(pkg)
    return d, pkg


def main():
    fails = []

    def check(name, ok, detail=""):
        print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # ---- 1. a read of a flag the parser cannot produce -------------------------------------
    def plant_culled(pkg):
        p = os.path.join(pkg, "unit.py")
        s = open(p).read()
        s = s.replace("def run_unit(rt, ",
                      "def run_unit(rt, ", 1)
        # a live read of a flag that does not exist in cli.py
        s += "\n\ndef _planted(rt, a):\n    return a.a_flag_that_does_not_exist\n"
        open(p, "w").write(s)

    d, pkg = planted(plant_culled)
    rc, out = run("test_no_culled_flags.py", pkg)
    check("culled-flag detector fails on a planted unknown flag", rc != 0,
          f"exit {rc}" + ("" if rc else " -- it did not notice"))
    check("  and it names the planted flag", "a_flag_that_does_not_exist" in out)
    shutil.rmtree(d, ignore_errors=True)

    # ---- 2. a context read with no writer anywhere -----------------------------------------
    def plant_unwritten(pkg):
        p = os.path.join(pkg, "loop.py")
        s = open(p).read()
        s += "\n\ndef _planted(rt):\n    return rt.a_name_no_stage_ever_sets\n"
        open(p, "w").write(s)

    d, pkg = planted(plant_unwritten)
    rc, out = run("test_context_contract.py", pkg)
    check("context contract fails on a planted unwritten attribute", rc != 0,
          f"exit {rc}" + ("" if rc else " -- it did not notice"))
    check("  and it names the planted attribute", "a_name_no_stage_ever_sets" in out)
    shutil.rmtree(d, ignore_errors=True)

    # ---- 3. A NEW MODULE, which is how both detectors went blind before ---------------------
    def plant_new_module(pkg):
        open(os.path.join(pkg, "newly_added_stage.py"), "w").write(
            "def thing(rt, a):\n    return a.another_flag_that_does_not_exist\n")

    d, pkg = planted(plant_new_module)
    rc, out = run("test_no_culled_flags.py", pkg)
    check("detectors see a module added after they were written", rc != 0,
          f"exit {rc}" + ("" if rc else " -- a hand-written module list would miss this"))
    shutil.rmtree(d, ignore_errors=True)

    # ---- 4. and the clean package must still pass ------------------------------------------
    for t in ("test_no_culled_flags.py", "test_context_contract.py"):
        rc, _ = run(t, HERE)
        check(f"{t} passes on the real package", rc == 0, f"exit {rc}")

    print()
    if fails:
        print(f"FAILED {len(fails)}: {', '.join(fails)}")
        return 1
    print("every detector fails on a planted fault and passes on the real package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
