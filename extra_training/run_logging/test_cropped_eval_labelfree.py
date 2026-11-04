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
                      "--rid", "t4jtjr", "--epoch", "30"],
                     cwd=ROOT, capture_output=True, text=True)
check("the report reads the ground-truth-cropped arm's pre-fingerprint passes",
      rep.returncode == 0 and "+0.0163" in rep.stdout,
      rep.stdout.strip().split("\n")[-1][:90] if rep.stdout else rep.stderr[:90])
check("the report prints the cropped-minus-uncropped gap, which is the result",
      "the crop is worth" in rep.stdout)
rep3 = subprocess.run([PY_BIN, "-I", REPORT, "--out", "training/runs/cropped_eval",
                       "--rid", "t4jtjr", "--epoch", "3"],
                      cwd=ROOT, capture_output=True, text=True)
check("epoch 3 does NOT match epoch 30's passes",
      "no cropped pass on disk" in rep3.stdout, rep3.stdout.strip()[-80:])

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

print()
if FAIL:
    print(f"{len(FAIL)} FAILED: " + "; ".join(FAIL))
    sys.exit(1)
print("all checks passed")
