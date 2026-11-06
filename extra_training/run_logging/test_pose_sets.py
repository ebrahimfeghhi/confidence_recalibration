#!/usr/bin/env python
"""Tests for pose_sets.classify -- the shared definition of a CPU-recoverable unit.

Both directions are tested, because both are costly: a unit wrongly called recoverable lets the
validation fold take the cards while training still needs them, and a unit wrongly called
GPU-blocking holds the validation fold indefinitely.

Run: python -I run_logging/test_pose_sets.py
"""
import json, os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_sets                                                           # noqa: E402

FAILED = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f"  (got {got!r}, want {want!r})"))
    if not ok:
        FAILED.append(label)


def mkrun(root, pdb, seed, tag, samples, arm="free", conf=True, conf_body=None,
          run_json=True, run_pdb=None, run_seed=None, run_arm=None, bad_conf_json=False):
    d = os.path.join(root, pdb, f"{seed}-{tag}")
    os.makedirs(os.path.join(d, "predictions"), exist_ok=True)
    if run_json:
        with open(os.path.join(d, "run.json"), "w") as fh:
            json.dump({"pdb": run_pdb or pdb,
                       "seed": int(run_seed if run_seed is not None else seed),
                       "arm": run_arm or arm, "n_sample": 5}, fh)
    for k in samples:
        with open(os.path.join(d, "predictions", f"{pdb}_sample_{k}.cif"), "w") as fh:
            fh.write(f"data_{pdb}_sample_{k}\n")
        if conf:
            p = os.path.join(d, "predictions", f"{pdb}_summary_confidence_sample_{k}.json")
            if bad_conf_json:
                open(p, "w").write("{not json")
            else:
                body = conf_body if conf_body is not None else {"chain_pair_iptm": [[0.9, 0.5],
                                                                                    [0.5, 0.9]]}
                json.dump(body, open(p, "w"))
    return d


def run(name, build, want_status, arm="free"):
    root = tempfile.mkdtemp()
    try:
        build(root)
        st, d, detail = pose_sets.classify(root, "8abc", "500000", 5, arm=arm)
        check(name, st, want_status)
        return st, d, detail
    finally:
        shutil.rmtree(root, ignore_errors=True)


print("recoverable, and must NOT be called GPU-blocking")
run("complete: one run, samples 0-4, good summaries",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5)), pose_sets.COMPLETE)
run("complete: extra samples beyond nsample are harmless",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(7)), pose_sets.COMPLETE)
_st, _d, _dt = run("complete: a PARTIAL run sorting FIRST does not hide the complete one",
                   lambda r: (mkrun(r, "8abc", "500000", "aaa", [0, 1]),
                              mkrun(r, "8abc", "500000", "zzz", range(5))), pose_sets.COMPLETE)
check("   and it returns the complete run, not the partial one", os.path.basename(_d or ""),
      "500000-zzz")
run("complete: arm not requested -> provenance arm is not checked",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), arm="tmpl"), pose_sets.COMPLETE, arm=None)

print("NOT recoverable -- needs a GPU or a human")
run("partial: 2 of 5 poses",
    lambda r: mkrun(r, "8abc", "500000", "aaa", [0, 1]), pose_sets.PARTIAL)
run("partial: a missing middle sample",
    lambda r: mkrun(r, "8abc", "500000", "aaa", [0, 1, 3, 4]), pose_sets.PARTIAL)
run("partial: summary json absent for every pose",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), conf=False), pose_sets.PARTIAL)
run("partial: ONE summary json absent",
    lambda r: (mkrun(r, "8abc", "500000", "aaa", range(5)),
               os.remove(os.path.join(r, "8abc", "500000-aaa", "predictions",
                                      "8abc_summary_confidence_sample_2.json"))),
    pose_sets.PARTIAL)
run("partial: summary json unreadable",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), bad_conf_json=True), pose_sets.PARTIAL)
run("partial: summary json lacks chain_pair_iptm",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), conf_body={"iptm": 0.9}),
    pose_sets.PARTIAL)
run("partial: chain_pair_iptm present but empty",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), conf_body={"chain_pair_iptm": []}),
    pose_sets.PARTIAL)
run("partial: no run.json at all",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), run_json=False), pose_sets.PARTIAL)
run("partial: run.json names a different pdb",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), run_pdb="9xyz"), pose_sets.PARTIAL)
run("partial: run.json names a different seed",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), run_seed=500001), pose_sets.PARTIAL)
run("partial: run.json names a different ARM",
    lambda r: mkrun(r, "8abc", "500000", "aaa", range(5), run_arm="tmpl"), pose_sets.PARTIAL)
run("none: no saved run", lambda r: None, pose_sets.NONE)

print("ambiguity is reported, never resolved")
_st, _d, _dt = run("ambiguous: two complete eligible runs",
                   lambda r: (mkrun(r, "8abc", "500000", "aaa", range(5)),
                              mkrun(r, "8abc", "500000", "bbb", range(5))), pose_sets.AMBIGUOUS)
check("   and no run is chosen", _d, None)
run("not ambiguous: a second run that fails provenance",
    lambda r: (mkrun(r, "8abc", "500000", "aaa", range(5)),
               mkrun(r, "8abc", "500000", "bbb", range(5), run_arm="tmpl")), pose_sets.COMPLETE)

print("seed is matched as a string, since run.json stores an int")
check("provenance_ok on an int seed", pose_sets.provenance_ok.__name__, "provenance_ok")

print()
print(f"FAILURES: {len(FAILED)}" + (f" -> {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)
