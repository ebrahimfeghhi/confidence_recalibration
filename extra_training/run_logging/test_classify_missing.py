#!/usr/bin/env python
"""Tests for phase_i_check.classify_missing -- the three-way split of a missing DockQ row.

THE BUG THIS EXISTS FOR. parse() returns {} both for a truncated json and for a complete OST run
whose dockq_interfaces is [], and classify_missing treated them alike: with invalidate=True it
moved a VALID result aside and sent the unit back through OST, which on the same input returns the
same empty answer. 800 interface-samples in the training fold are in that state. A rescore that
"repairs" them by re-running the identical call is not a repair.

THE SECOND BUG. An interface absent from dockq_interfaces was always called a no-contact. OST's
dockq_interfaces holds the reference interfaces it COULD MAP, so absence also happens when a
reference chain was never mapped to a model chain -- a limitation of the mapping, not a statement
about the pose. chain_mapping tells them apart.

Run: python -I run_logging/test_classify_missing.py
"""
import importlib.util, json, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "pic", os.path.join(os.path.dirname(HERE), "phases", "phase_i_check.py"))
pic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pic)
FAILED = []


def check(label, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  {extra}" if not cond and extra else ""))
    if not cond:
        FAILED.append(label)


def mkrun(root, name, pose=True, doc="omit"):
    """a run dir with sample 0's pose cif and, unless doc is 'omit', an ost_json/0.json"""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "predictions"), exist_ok=True)
    if pose:
        open(os.path.join(d, "predictions", "x_sample_0.cif"), "w").write("data_x\n")
    if doc != "omit":
        os.makedirs(os.path.join(d, "ost_json"), exist_ok=True)
        p = os.path.join(d, "ost_json", "0.json")
        if isinstance(doc, str):
            open(p, "w").write(doc)                       # raw text, for the truncated case
        else:
            json.dump(doc, open(p, "w"))
    return d


SCORED = {"dockq_interfaces": [["A", "B", "A", "B"]], "dockq": [0.5],
          "chain_mapping": {"A": "A", "B": "B"}}

root = tempfile.mkdtemp()
try:
    print("the pose and the scorer cache decide, not the absence of a row")
    d = mkrun(root, "nopose", pose=False)
    check("no pose cif -> no_pose", pic.classify_missing(d, "ost", "0")[0] == "no_pose")

    d = mkrun(root, "nojson")
    check("pose but no cache -> unscored", pic.classify_missing(d, "ost", "0")[0] == "unscored")

    d = mkrun(root, "truncated", doc="{")
    b, _ = pic.classify_missing(d, "ost", "0")
    check("truncated cache -> unscored", b == "unscored")
    check("read-only classification leaves it in place",
          os.path.exists(os.path.join(d, "ost_json", "0.json")))
    b, _ = pic.classify_missing(d, "ost", "0", invalidate=True)
    check("invalidate moves a truncated cache aside",
          os.path.exists(os.path.join(d, "ost_json", "0.json.invalid")))

    d = mkrun(root, "ostfailed", doc={"status": "FAILURE", "traceback": "RuntimeError\n"})
    check("an OST FAILURE doc -> unscored (it really did not score)",
          pic.classify_missing(d, "ost", "0")[0] == "unscored")

    print("\na successful run that mapped nothing is NOT an unusable cache")
    d = mkrun(root, "empty", doc={"dockq_interfaces": [], "dockq": [], "status": "SUCCESS"})
    b, why = pic.classify_missing(d, "ost", "0", iface="x_AB")
    check("empty interface list -> unmapped", b == "unmapped", f"got {b!r}: {why}")
    b, why = pic.classify_missing(d, "ost", "0", iface="x_AB", invalidate=True)
    check("invalidate does NOT move a valid empty result aside -- the regression",
          os.path.exists(os.path.join(d, "ost_json", "0.json"))
          and not os.path.exists(os.path.join(d, "ost_json", "0.json.invalid")),
          str(sorted(os.listdir(os.path.join(d, "ost_json")))))

    print("\nabsent is two outcomes, and chain_mapping tells them apart")
    d = mkrun(root, "present", doc=SCORED)
    b, why = pic.classify_missing(d, "ost", "0", iface="x_AB")
    check("the cache reports this pair -> unscored (the row went missing)", b == "unscored",
          f"got {b!r}: {why}")

    both = dict(SCORED, chain_mapping={"A": "A", "B": "B", "C": "C"})
    d = mkrun(root, "nocontact", doc=both)
    b, why = pic.classify_missing(d, "ost", "0", iface="x_AC")
    check("pair absent but both chains mapped -> no contact", b == "", f"got {b!r}: {why}")

    d = mkrun(root, "unmapped", doc=SCORED)          # chain_mapping has no C
    b, why = pic.classify_missing(d, "ost", "0", iface="x_AC")
    check("pair absent because a chain was never mapped -> unmapped", b == "unmapped",
          f"got {b!r}: {why}")
    check("and it says which chain", "['C']" in why, why)
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
print(f"FAILURES: {len(FAILED)}" + (f" -> {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)
