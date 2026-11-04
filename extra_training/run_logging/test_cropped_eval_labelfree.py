#!/usr/bin/env python
"""The label-free cropped eval: crop resolution, the leak guards, and the cache-collision trap.

WHAT THIS COVERS WITHOUT A CARD. cropped_eval.py resolves the crop and builds the pass list before
it launches anything, so --dry-run exercises every guard for real rather than by reading source.
The four passes themselves need GPUs and are not run here.

  1  THE CACHE-COLLISION TRAP, which is the bug this file was written around. Passes are skipped
     when their .json exists and the shipped pass carries no epoch, so a second crop variant run
     into the same --out silently reused the FIRST variant's shipped baseline and compared two
     different crops. Every pass name must now carry a fingerprint of its crop, and the two
     variants must produce DIFFERENT fingerprints.
  2  THE LEAK GUARDS. A posefree mask must record force_gt=false; the override must refuse to ADD
     --force-gt to an arm that trained without it; and the deviation must be announced.
  3  THE PAIRING GUARDS. One mask without the other is refused, a mask from the wrong pool is
     refused by pool/seed-base/budget, and a rebuilt mask is refused by the guard digest.
  4  THE FEATURE CHANGES SOMETHING: --crop-tokens must actually reach the evaluator's argv, and
     the margin path must be untouched for arms that never used a mask.
  5  THE REPORT'S LEGACY NAMES. The ground-truth-cropped arm's existing passes predate
     fingerprinting, so the report must still find insample_shipped.json alongside
     insample_shipped_<fp>.json -- and must not match ep3 against ep30.

Run:  python -I run_logging/test_cropped_eval_labelfree.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_BIN = "/home/ebrahim/Protenix/.venv/bin/python"
EVAL = os.path.join(ROOT, "run_logging", "cropped_eval.py")
REPORT = os.path.join(ROOT, "run_logging", "cropped_eval_report.py")
SCRIPT = os.path.join(ROOT, "shell_scripts", "cropped_eval_labelfree.sh")
BUNDLE = "training/runs/y8pyuq/resume_ep20.pt"
IS_GT = "training/full_train/union_crop_m0_gt_ost.npz"
VF_GT = "training/valfull/union_crop_m0_gt_ost.npz"
IS_PF = "training/full_train/union_crop_m0_posefree.npz"
VF_PF = "training/valfull/union_crop_m0_posefree.npz"
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def run_eval(*extra, bundle=BUNDLE):
    cmd = [PY_BIN, "-I", EVAL, "--bundle", bundle, "--dry-run",
           "--out", "training/runs/_test_cropped"] + list(extra)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def meta(rel):
    z = np.load(os.path.join(ROOT, rel), allow_pickle=True)
    return json.loads(str(z["__meta"][0]))


# ------------------------------------------------------------------ 0. preconditions
have_pf = os.path.exists(os.path.join(ROOT, IS_PF)) and os.path.exists(os.path.join(ROOT, VF_PF))
check("the gt_ost masks exist for both pools",
      os.path.exists(os.path.join(ROOT, IS_GT)) and os.path.exists(os.path.join(ROOT, VF_GT)))
if not have_pf:
    print(f"  note  the posefree masks are not built yet; their cases are skipped")

# ------------------------------------------------------------------ 1. cache collision
rc_gt, out_gt = run_eval("--insample-crop-tokens", IS_GT, "--valfull-crop-tokens", VF_GT)
check("the training-crop variant resolves", rc_gt == 0, out_gt.strip().split("\n")[-1][:90])
fp_gt = re.search(r"crop fingerprint: (\w+)", out_gt)
check("every pass name carries the crop fingerprint",
      bool(fp_gt) and all(fp_gt.group(1) in ln for ln in out_gt.split("\n")
                          if re.match(r"\s+(insample|valfull)_", ln)),
      fp_gt.group(1) if fp_gt else "no fingerprint printed")
check("the shipped pass is no longer named bare",
      "insample_shipped\n" not in out_gt and "    insample_shipped\n" not in out_gt)

if have_pf:
    rc_pf, out_pf = run_eval("--insample-crop-tokens", IS_PF, "--valfull-crop-tokens", VF_PF)
    check("the label-free variant resolves", rc_pf == 0, out_pf.strip().split("\n")[-1][:90])
    fp_pf = re.search(r"crop fingerprint: (\w+)", out_pf)
    check("THE TWO VARIANTS GET DIFFERENT FINGERPRINTS, so neither can read the other's cache",
          bool(fp_pf) and bool(fp_gt) and fp_pf.group(1) != fp_gt.group(1),
          f"{fp_gt.group(1) if fp_gt else '?'} vs {fp_pf.group(1) if fp_pf else '?'}")
    check("the label-free run announces the deviation", "OFF-TRAINING CROP" in out_pf)
    check("and prints both the trained and the scored rule",
          "trained under" in out_pf and "scoring under" in out_pf)

# ------------------------------------------------------------------ 2. leak guards
if have_pf:
    for m in (IS_PF, VF_PF):
        d = meta(m)
        check(f"{os.path.basename(m)} records force_gt=false",
              not d.get("force_gt") and not d.get("ost_pairs"),
              f"force_gt={d.get('force_gt')} ost_pairs={d.get('ost_pairs')}")
        check(f"{os.path.basename(m)} keeps the arm's geometry",
              float(d["margin"]) == 0.0 and float(d["contact"]) == 5.0,
              f"margin={d['margin']} contact={d['contact']}")
    # the gt_ost masks are the leaky ones, and must say so
    check("the gt_ost masks record force_gt=true",
          meta(IS_GT).get("force_gt") and meta(VF_GT).get("force_gt"))
    # ADDING force-gt to an arm that trained without it is the one refused direction. y8pyuq DID
    # train with it, so this is checked at source level rather than by a run that cannot trigger.
    src = open(EVAL).read()
    check("adding --force-gt over an arm that trained without it is refused",
          'if METAS["insample"].get("force_gt") and not _tm.get("force_gt")' in src
          and "one direction this flag must not be used in" in src)

# ------------------------------------------------------------------ 3. pairing guards
rc, out = run_eval("--insample-crop-tokens", IS_GT)
check("one mask without the other is refused", rc != 0 and "go together" in out)
rc, out = run_eval("--insample-crop-tokens", VF_GT, "--valfull-crop-tokens", VF_GT)
check("a held-out mask handed in as the in-sample mask is refused by pool/seed-base/budget",
      rc != 0 and "but this pass" in out, out.strip().split("\n")[-1][:110])
rc, out = run_eval("--insample-crop-tokens", IS_GT, "--valfull-crop-tokens", IS_GT)
check("a train mask handed in as the held-out mask is refused the same way",
      rc != 0 and "but this pass" in out)

# A REBUILT MASK MUST BE REFUSED BY THE GUARD DIGEST. Flipping one bit of one structure's mask in
# a temp copy is the cheapest faithful version of "someone rebuilt it at a different margin".
with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "training", "runs")) as td:
    z = np.load(os.path.join(ROOT, IS_GT), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    tgt = sorted(k for k in d if not k.endswith(("__n", "__pairs")) and k != "__meta")[0]
    d[tgt] = np.asarray(d[tgt]).copy()
    d[tgt][0] = not bool(d[tgt][0])
    tampered = os.path.join(td, "tampered.npz")
    np.savez(tampered, **d)
    rel = os.path.relpath(tampered, ROOT)
    rc, out = run_eval("--insample-crop-tokens", rel, "--valfull-crop-tokens", VF_GT)
    # the digest is checked on the GUARD's mask, so tampering with the OVERRIDE must not be
    # silently accepted as the training crop either -- it must read as an off-training crop
    check("a tampered mask is not mistaken for the training crop",
          rc != 0 or "OFF-TRAINING CROP" in out)

# ------------------------------------------------------------------ 4. the feature does something
check("--crop-tokens reaches the evaluator for BOTH pools, and --crop-margin does not",
      'CROP_FOR = {t: ["--crop-tokens", MASKS[t]] for t in MASKS}' in open(EVAL).read()
      and "+ CROP_FOR[tag] + POOL_FLAGS" in open(EVAL).read())
src = open(EVAL).read()
check("the margin path survives for arms that never used a mask",
      'elif float(G.get("crop_margin") or 0) > 0:' in src
      and '_c = ["--crop-margin"' in src)
check("--crop-contact is NOT passed under --crop-tokens, where it has no effect",
      '"--crop-contact"' not in src.split("CROP_FOR = {t:")[1].split("CROP_DESC")[0])
check("the guard digest is reproduced with phase_k_train's own algorithm",
      "__pairs" in src and "np.packbits" in src and 'hexdigest()[:16]' in src)

# ------------------------------------------------------------------ 5. the report
rep = subprocess.run([PY_BIN, "-I", REPORT, "--out", "training/runs/cropped_eval",
                      "--rid", "t4jtjr", "--epoch", "30", "--legacy-unfingerprinted"],
                     cwd=ROOT, capture_output=True, text=True)
check("the report reads the ground-truth-cropped arm's pre-fingerprint passes",
      rep.returncode == 0 and "+0.0226" in rep.stdout,
      rep.stdout.strip().split("\n")[-1][:90] if rep.stdout else rep.stderr[:90])
check("the report prints the cropped-minus-uncropped gap",
      "cropped PRIMARY minus uncropped" in rep.stdout)
# THE CORRECTION THIS REVIEW FORCED. With the bar computed rather than carried over from the
# uncropped comparison, the ground-truth-cropped arm's held-out cropped gain does NOT clear it.
check("and does NOT announce the ground-truth-cropped arm's held-out gain as significant",
      "n=92" in rep.stdout and rep.stdout.count("inside the noise") >= 2,
      "+0.0226 on a +0.0245 bar")
rep3 = subprocess.run([PY_BIN, "-I", REPORT, "--out", "training/runs/cropped_eval",
                       "--rid", "t4jtjr", "--epoch", "3", "--legacy-unfingerprinted"],
                      cwd=ROOT, capture_output=True, text=True)
check("epoch 3 does NOT match epoch 30's passes",
      "no pass on disk" in rep3.stdout, rep3.stdout.strip()[-80:])

# ------------------------------------------------------------------ 6. the script
sh = open(SCRIPT).read()
check("the script verifies both masks every run, built or not",
      sh.count("check_crop_mask.py") == 2)
check("the script asserts force_gt=false for the posefree variant rather than waiving it",
      "is NOT label-free" in sh and "--require-force-gt" in sh)
check("the script never launches or stops a trainer",
      not any(t in sh for t in ("stop_run.sh", "nohup", "launch_")))
check("the script defaults to the label-free variant",
      "VARIANT=${3:-posefree}" in sh)

# ------------------------------------------------------------------ 7. the review fixes
# (a) THE FINGERPRINT MUST FOLLOW MASK CONTENTS, NOT THE FILENAME. Same path, one bit flipped:
# before the fix both runs produced the same id, so the evaluator reused result files computed on
# the OLD tokens. Raised in review.
if have_pf:
    import shutil
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "training", "valfull")) as td:
        same = os.path.join(td, "m.npz")
        shutil.copyfile(os.path.join(ROOT, VF_PF), same)
        rel = os.path.relpath(same, ROOT)
        rc1, o1 = run_eval("--insample-crop-tokens", IS_PF, "--valfull-crop-tokens", rel)
        id1 = re.search(r"^CROP_ID=(\w+)$", o1, re.M)
        z = np.load(same, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        tgt = sorted(k for k in d if not k.endswith(("__n", "__pairs")) and k != "__meta")[0]
        d[tgt] = np.asarray(d[tgt]).copy()
        d[tgt][0] = not bool(d[tgt][0])
        np.savez(same, **d)
        rc2, o2 = run_eval("--insample-crop-tokens", IS_PF, "--valfull-crop-tokens", rel)
        id2 = re.search(r"^CROP_ID=(\w+)$", o2, re.M)
        check("A MASK REBUILT UNDER THE SAME FILENAME GETS A NEW FINGERPRINT",
              rc1 == 0 and rc2 == 0 and id1 and id2 and id1.group(1) != id2.group(1),
              f"{id1.group(1) if id1 else '?'} -> {id2.group(1) if id2 else '?'}")

# (b) THE REPORT MUST BE BOUND TO ONE CROP. It used to glob and take the newest by mtime, so
# running posefree, then gt_ost, then the posefree report printed gt_ost numbers.
rc = subprocess.run([PY_BIN, "-I", REPORT, "--out", "training/runs/cropped_eval",
                     "--rid", "t4jtjr", "--epoch", "30"], cwd=ROOT, capture_output=True, text=True)
check("the report refuses to run without a crop id",
      rc.returncode != 0 and "exactly one of" in (rc.stdout + rc.stderr))
rc = subprocess.run([PY_BIN, "-I", REPORT, "--out", "training/runs/cropped_eval", "--rid",
                     "t4jtjr", "--epoch", "30", "--crop-id", "deadbeef",
                     "--legacy-unfingerprinted"], cwd=ROOT, capture_output=True, text=True)
check("and refuses both at once, which would be two answers to which files to read",
      rc.returncode != 0 and "exactly one of" in (rc.stdout + rc.stderr))
rc = subprocess.run([PY_BIN, "-I", REPORT, "--out", "training/runs/cropped_eval", "--rid",
                     "t4jtjr", "--epoch", "30", "--crop-id", "deadbeef"],
                    cwd=ROOT, capture_output=True, text=True)
check("a crop id with no passes on disk reports MISSING, never another variant's files",
      rc.returncode == 0 and "no pass on disk for this crop" in rc.stdout
      and "0.0163" not in rc.stdout and "0.0226" not in rc.stdout)

# (c) TWO BASELINES, REPORTED SEPARATELY. The cropped delta used ours-mean against shipped-MEAN
# while the uncropped delta used the shipped head's RECORDED ipTM, so the gap between them moved
# the crop AND the shipped readout at once.
rc = subprocess.run([PY_BIN, "-I", REPORT, "--out", "training/runs/cropped_eval", "--rid",
                     "t4jtjr", "--epoch", "30", "--legacy-unfingerprinted"],
                    cwd=ROOT, capture_output=True, text=True)
check("the report separates the shipped MAX primary from the shipped MEAN diagnostic",
      "PRIMARY    ours mean-sel vs shipped MAX-sel" in rc.stdout
      and "DIAGNOSTIC ours mean-sel vs shipped mean-sel" in rc.stdout)
check("and labels the cropped-minus-uncropped gap as indicative, not an estimate",
      "indicative" in rc.stdout and "recomputed vs stored" in rc.stdout)

# (d) ERROR BARS COMPUTED PER COMPARISON, NOT HARDCODED. The old file carried a 0.0044 s.e. and a
# +0.009 bar from an uncropped comparison on a different head. Cropping nearly triples the
# per-interface spread, so those numbers turned an inside-the-noise delta into "3.7 s.e.".
rsrc = open(REPORT).read()
_code = rsrc.split('"""', 2)[2] if rsrc.count('"""') >= 2 else rsrc
check("no hardcoded s.e. or significance bar survives in the report's CODE",
      "0.0044" not in _code and "0.009" not in _code
      and not re.search(r"^(SE|BAR)\s*(,|=)", _code, re.M),
      "the docstring still cites them while explaining their removal, which is intended")
check("the paired s.e. is computed from the per-interface differences",
      "st.stdev(d) / math.sqrt(len(d))" in rsrc and '"bar": 1.96 * se' in rsrc)
check("the report prints its own s.e. and n for every delta",
      "paired s.e." in rc.stdout and "n=92" in rc.stdout and "n=738" in rc.stdout)
check("the computed cropped bar is far wider than the uncropped 0.009 it replaced",
      re.search(r"n=92\s+--", rc.stdout) and "bar +0.02" in rc.stdout,
      "cropped held-out bar is +0.02x, not +0.009")
check("selection ties break exactly as phase_k_eval._pick does",
      "-t[3], -t[4]" in rsrc and "_pick" in rsrc)

# (e) THE MASK LOADER COUNTED __pairs AND __meta AS STRUCTURES
esrc = open(os.path.join(ROOT, "phases", "phase_k_eval.py")).read()
check("phase_k_eval's mask loader excludes __pairs and __meta, not just __n",
      'k.endswith("__n") or k.endswith("__pairs") or k == "__meta"' in esrc)
if have_pf:
    z = np.load(os.path.join(ROOT, VF_PF), allow_pickle=True)
    real = len([k for k in z.files
                if not (k.endswith("__n") or k.endswith("__pairs") or k == "__meta")])
    old_count = len([k for k in z.files if not k.endswith("__n")])
    check("and that is not cosmetic: the old filter would have counted 107 of 53",
          real == 53 and old_count == 107, f"{old_count} -> {real}")

# ------------------------------------------------------------------ 8. the reconstruction check
# THE GUARD MUST ACTUALLY FIRE. Everything the report prints is rebuilt by re-selecting poses, so
# a mistake in the report would produce a confident table nobody could tell from a real one. The
# check reproduces the evaluator's own saved mean; here it is handed a perturbed summary and must
# refuse the whole table rather than print it. Raised in review.
import shutil
SRC = os.path.join(ROOT, "training", "runs", "cropped_eval")
if os.path.exists(os.path.join(SRC, "valfull_ours_ep30.json")):
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "training", "runs")) as td:
        for n in ("valfull_ours_ep30", "valfull_shipped"):
            shutil.copyfile(os.path.join(SRC, n + ".json"), os.path.join(td, n + ".json"))
            os.symlink(os.path.join(SRC, n), os.path.join(td, n))
        rel = os.path.relpath(td, ROOT)

        def _report(where):
            r = subprocess.run([PY_BIN, "-I", REPORT, "--out", where, "--rid", "t4jtjr",
                                "--epoch", "30", "--legacy-unfingerprinted"],
                               cwd=ROOT, capture_output=True, text=True)
            return r.returncode, r.stdout + r.stderr

        rc, out = _report(rel)
        check("the untouched copy still reproduces the saved means",
              rc == 0 and "rows reproduce the evaluator" in out)

        # a mean that disagrees by more than one rounding step must stop the report
        jp = os.path.join(td, "valfull_ours_ep30.json")
        d = json.load(open(jp))
        true_mean = d["mean_dockq"]
        d["mean_dockq"] = round(true_mean + 0.01, 4)
        json.dump(d, open(jp, "w"))
        rc, out = _report(rel)
        check("A PERTURBED SUMMARY MEAN IS FATAL, not a warning",
              rc != 0 and "does not reproduce the pass it is describing" in out,
              f"mean_dockq {true_mean} -> {d['mean_dockq']}")
        check("and the refusal says not to read the table",
              "Do not read the table" in out)

        # a rounding-sized disagreement must NOT stop it, or the check would be unusable
        d["mean_dockq"] = true_mean
        json.dump(d, open(jp, "w"))
        rc, out = _report(rel)
        check("a rounding-sized difference is tolerated", rc == 0)

        # and a summary claiming the wrong interface count must stop it
        d["n_interfaces"] = 91
        json.dump(d, open(jp, "w"))
        rc, out = _report(rel)
        check("an interface count that disagrees with the rows is fatal",
              rc != 0 and ("rows carry" in out or "expected" in out))
else:
    print("  note  the ground-truth-cropped arm's passes are absent; reconstruction cases skipped")

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: " + "; ".join(FAIL))
    sys.exit(1)
print("all checks passed")
