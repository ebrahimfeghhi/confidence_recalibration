#!/usr/bin/env python
"""Tests for the rescue route out of an unreadable OST chain mapping. CPU only, no fixtures on disk.

WHAT IS UNDER TEST. phase_k_eval.ost_chain_pairs is the one place ten callers resolve a pose's chain
pair, and until now a pose whose ost_json is a FAILURE record was unresolvable forever: the mapping
was never lost, only never written down, because the scorer that wrote it had a bug (it sent OST the
reference's AUTH chain names and OST refused the whole call). run_logging/rescue_ost_mapping.py
re-runs the FIXED scorer and writes OST's own answer to ost_json/<k>.json.rescue, and
ost_chain_pairs reads that file when and only when the original carries no mapping.

THE ASSERTIONS THAT MATTER MOST ARE THE NEGATIVE ONES:
  * the feature CHANGES something -- the same directory is unresolvable without the rescue file and
    resolved with it, checked in one test rather than assumed;
  * a recorded mapping still WINS, so a stale or wrong rescue file cannot override a real decision;
  * the rescue file is held to the SAME sample-index assertion as the original, because a mapping
    read off the wrong pose is the exact defect ost_chain_pairs exists to prevent, and it fails
    silently;
  * a truncated json is a refusal with a reason, not a ValueError out of the function -- that used
    to kill a training run with a traceback that named no pose.

Run: python -I run_logging/test_ost_rescue.py
"""
import importlib.util, json, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "phases"))
spec = importlib.util.spec_from_file_location(
    "pke", os.path.join(ROOT, "phases", "phase_k_eval.py"))
PKE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PKE)

FAILED = []


def check(label, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  {extra}" if not cond and extra else ""))
    if not cond:
        FAILED.append(label)


# 4zto's real shape: reference D/E/F against model A/B/C, pool interfaces DF and EF.
REFS = [("4zto_EF", "E", "F"), ("4zto_DF", "D", "F")]
GOOD = {"model": "/x/4zto_sample_0.cif",
        "chain_mapping": {"D": "B", "F": "C", "E": "A"},
        "reference_chains": ["D", "E", "F"], "model_chains": ["A", "B", "C"]}
FAILREC = {"status": "FAILURE",
           "traceback": 'RuntimeError: Target chain "M" is not available in target processed '
                        'for chain mapping ({\'D\', \'F\', \'E\'})\n'}
# D->B is asym 1, F->C is 2, E->A is 0. EF = (0, 2), DF = (1, 2).
WANT = [("4zto_EF", 0, 2), ("4zto_DF", 1, 2)]


def unit(root, name, orig=None, rescue=None, k=0):
    d = os.path.join(root, name, "ost_json")
    os.makedirs(d, exist_ok=True)
    for suffix, doc in ((f"{k}.json", orig), (f"{k}.json.rescue", rescue)):
        if doc is None:
            continue
        p = os.path.join(d, suffix)
        if isinstance(doc, str):
            open(p, "w").write(doc)                 # raw text, for the truncated case
        else:
            json.dump(doc, open(p, "w"))
    return os.path.join(root, name)


root = tempfile.mkdtemp()
try:
    print("the rescue file resolves a pose the failure record cannot -- and it CHANGES the outcome")
    d = unit(root, "nofix", orig=FAILREC)
    before, why_before = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
    check("a FAILURE record alone is unresolvable", before is None, f"got {before!r}")
    # A FAILURE RECORD HAS NO "model" KEY, so the sample-index assertion fires before the
    # chain_mapping one -- which is exactly the line the evaluator printed for 4zto,
    # "1.json describes , not sample 1". The reason must still account for BOTH candidates, or a
    # reader is left thinking only one file was looked for.
    check("and the reason names both files it tried",
          "0.json describes" in why_before and "no 0.json.rescue" in why_before, why_before)

    d = unit(root, "fixed", orig=FAILREC, rescue=GOOD)
    after, why_after = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
    check("with the rescue file beside it the pose resolves", after == WANT,
          f"got {after!r} ({why_after})")
    check("so the rescue file is what changed the outcome, not the refs",
          before is None and after is not None)

    print()
    print("a recorded mapping always wins -- a rescue file cannot override a real decision")
    swapped = dict(GOOD, chain_mapping={"D": "C", "F": "B", "E": "A"})
    d = unit(root, "origwins", orig=GOOD, rescue=swapped)
    got, _ = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
    check("the original's mapping is used, not the rescue file's", got == WANT, f"got {got!r}")
    check("and the two really do differ, so the test is not vacuous",
          PKE.ost_chain_pairs(unit(root, "swaponly", orig=swapped), "4zto", 0, REFS)[0] != WANT)

    print()
    print("the rescue file is held to the SAME checks as the original")
    d = unit(root, "wrongsample", orig=FAILREC,
             rescue=dict(GOOD, model="/x/4zto_sample_3.cif"))
    got, why = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
    check("a rescue file describing another sample is REFUSED", got is None, f"got {got!r}")
    check("and it says which pose it actually describes",
          "4zto_sample_3.cif" in why and "not sample 0" in why, why)

    d = unit(root, "nomapping", orig=FAILREC, rescue={"model": "/x/4zto_sample_0.cif"})
    got, why = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
    check("a rescue file with no chain_mapping is refused", got is None, f"got {got!r}")
    check("and the reason names the rescue file", "0.json.rescue has no chain_mapping" in why, why)

    print()
    print("a corrupt json is a refusal with a reason, not a traceback out of the function")
    d = unit(root, "truncated", orig="{")
    try:
        got, why = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
        raised = ""
    except Exception as e:                                            # noqa: BLE001
        got, why, raised = "", "", f"{type(e).__name__}: {e}"
    check("a truncated original does not raise", not raised, raised)
    check("it refuses and names the defect", got is None and "unreadable" in why, why)

    d = unit(root, "truncfix", orig="{", rescue=GOOD)
    got, why = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
    check("a truncated original still falls through to the rescue file", got == WANT,
          f"got {got!r} ({why})")

    print()
    print("no ost_json at all is still a refusal that names what was looked for")
    d = unit(root, "empty")
    got, why = PKE.ost_chain_pairs(d, "4zto", 0, REFS)
    check("missing on both paths -> None", got is None, f"got {got!r}")
    check("and both candidate names appear", "no 0.json" in why and "no 0.json.rescue" in why, why)

    print()
    print("the rescue sidecar must not be caught by the ost_json/*.json globs that exist")
    import glob as _g
    d = unit(root, "globcheck", orig=FAILREC, rescue=GOOD)
    hits = sorted(os.path.basename(f) for f in _g.glob(os.path.join(d, "ost_json", "*.json")))
    check("ost_json/*.json still sees exactly one file per pose", hits == ["0.json"], str(hits))

    print()
    print("the callers that used to swallow an unmapped pose now name it")
    for rel, needle in (("run_logging/build_crop_pool.py", "OST MAPPING UNREADABLE"),
                        ("run_logging/check_crop_pairs.py", "OST MAPPING UNREADABLE"),
                        ("run_logging/insample_fullset.py", "OST MAPPING RESCUED")):
        src = open(os.path.join(ROOT, rel)).read()
        check(f"{rel} reports it", needle in src)
        check(f"{rel} points at the recovery command", "rescue_ost_mapping.py" in src)
    for rel in ("phases/phase_k_train.py", "phases/trainer/dataset.py",
                "phases/trainer/helpers.py", "run_logging/itm_ceiling.py",
                "phases/phase_k_eval.py"):
        src = open(os.path.join(ROOT, rel)).read()
        check(f"{rel}'s refusal names the recovery command", "rescue_ost_mapping.py" in src)
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
print(f"FAILURES: {len(FAILED)}" + (f" -> {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)
