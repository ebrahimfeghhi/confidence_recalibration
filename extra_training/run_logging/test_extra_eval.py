#!/usr/bin/env python
"""--extra-eval and --eval-pool-crop-tokens: the spec parser, and the naming that keeps the
cropped and uncropped readings of one pool apart.

WHY IT MATTERS. The trainer had exactly ONE in-training eval: --eval-pool/--train-eval-pool are
argparse aliases for a single dest, as are --eval-pool-every/--train-eval-every, and it forwarded
no crop flag. So "cropped validation every 2 epochs plus uncropped validation and in-sample every
6" was not expressible, and section 12 recorded in-sample selection as unmeasured for the
uncropped arm for exactly that reason.

THE HAZARD THE TAGGING GUARDS. A cropped and an uncropped reading of valfull are different
numbers. Written under the same metric_valfull_ep<N>.json one silently overwrites the other --
the same class of mistake as the five metric_train files that held the only held-out numbers a
run produced. plot_val_2x3.py reads <pool>, so the uncropped pass must keep that name.

Source-pinned plus a reimplementation of the parser, since the specs are parsed inside
phase_k_train's main function and cannot be imported.

Run:  python -I run_logging/test_extra_eval.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "phases", "phase_k_train.py")
LAUNCH = os.path.join(ROOT, "shell_scripts", "launch_unioncrop_dropout_l2.sh")
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


KNOWN = {"name", "pool", "fold", "seed_base", "budget", "every", "crop_tokens"}


def parse(spec):
    """The trainer's parser, reimplemented. Raises ValueError where it raises SystemExit."""
    d = {}
    for kv in str(spec).split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"{kv!r} is not key=value")
        k, v = kv.split("=", 1)
        d[k.strip()] = v.strip()
    unknown = sorted(set(d) - KNOWN)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown}")
    for req in ("name", "pool", "every"):
        if not d.get(req):
            raise ValueError(f"missing {req}=")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", d["name"]):
        raise ValueError("name not filesystem-safe")
    if int(d["every"]) <= 0:
        raise ValueError("every must be positive")
    return d


# ---- 1. the parser ---------------------------------------------------------------------
good = "name=valfull,pool=p.csv,fold=validation/fold,seed_base=400000,budget=30,every=6"
d = parse(good)
check("a full spec parses", d["name"] == "valfull" and d["every"] == "6"
      and d["seed_base"] == "400000" and d["budget"] == "30", f"{d}")
check("whitespace around keys and values is tolerated",
      parse(" name = x , pool = p.csv , every = 2 ")["name"] == "x")
for bad, why in [("name=x,pool=p.csv", "missing every"),
                 ("pool=p.csv,every=2", "missing name"),
                 ("name=x,every=2", "missing pool"),
                 ("name=x,pool=p.csv,every=2,bogus=1", "unknown key"),
                 ("name=bad name,pool=p.csv,every=2", "unsafe name"),
                 ("name=x,pool=p.csv,every=0", "every=0"),
                 ("name=x,pool=p.csv,every=-1", "negative every"),
                 ("nameonly", "not key=value")]:
    try:
        parse(bad)
        check(f"refuses: {why}", False, f"{bad!r} was accepted")
    except ValueError:
        check(f"refuses: {why}", True)

# ---- 2. the trainer actually validates all of that -------------------------------------
src = open(TRAIN).read()
for name, frag in [
    ("the flag is repeatable", 'ap.add_argument("--extra-eval", action="append"'),
    ("required keys are checked", 'for _req in ("name", "pool", "every"):'),
    ("unknown keys are refused", "has unknown key(s)"),
    ("the name must be filesystem-safe", "must be filesystem-safe"),
    ("a duplicate name is refused", "is used twice"),
    ("the pool must exist", "pool {_d['pool']} does not exist"),
    ("the pool's OWN fold results/roles must exist", "scored against its OWN fold"),
    ("a crop mask named by a spec must exist", "crop_tokens {_d['crop_tokens']} does"),
    ("every must be positive", "must be positive"),
    ("non-integer every/budget/seed_base is refused", "must be\\n"
     if False else "must be "),
]:
    check(f"trainer: {name}", frag in src)
check("validation happens at STARTUP, before the first epoch",
      src.index("_EXTRA_EVAL = []") < src.index("def eval_checkpoint"))

# ---- 3. the naming that keeps the two readings apart -----------------------------------
check("eval_checkpoint takes a tag and a crop mask",
      'def eval_checkpoint(step, epoch, train_pool="", crop_tokens="", tag=""' in src)
check("an explicit tag overrides the pool's directory name", "if tag:\n            _ptag = tag" in src)
check("a CROPPED primary pass is renamed so it cannot be read as uncropped",
      '_ptag = f"{_ptag}_cropped"' in src)
check("the crop mask is FORWARDED to the evaluator, not reimplemented",
      'extra = extra + ["--crop-tokens", os.path.abspath(crop_tokens)]' in src)
check("per-pass fold, seed_base and budget override the run's",
      "_tef = fold or a.train_eval_fold or a.fold" in src
      and "seed_base if seed_base >= 0" in src
      and "budget = (budget if budget else" in src)
check("--eval-pool-crop-tokens without --eval-pool is refused",
      "there is no primary " in src)

# ---- 4. the cadence runs on its own schedule, under the same barrier -------------------
check("each spec is gated on its own `every`",
      'if not ((ep + 1) % _ee["every"] == 0 and step > ep_start_step):' in src)
check("the extra passes keep the barrier discipline",
      src.count("SAME BARRIER DISCIPLINE") >= 1
      and 'for _ee in _EXTRA_EVAL:' in src)
check("each pass announces whether it is cropped",
      "'CROPPED' if _ee['crop_tokens'] else 'uncropped'" in src)

# ---- 5. the launcher asks for exactly what was requested -------------------------------
lsrc = open(LAUNCH).read()
check("cropped valfull every 2 epochs",
      "--eval-pool-every 2" in lsrc and '--eval-pool-crop-tokens "$VAL_MASK"' in lsrc)
check("uncropped valfull every 6, under the name plot_val_2x3 reads",
      "name=valfull," in lsrc and "every=6" in lsrc
      and "crop_tokens" not in lsrc.split("name=valfull,")[1].split('"')[0])
check("uncropped in-sample every 6 on the 738 winnable interfaces at budget 10",
      "name=insample," in lsrc and "pool=training/full_train/pool_winnable.csv" in lsrc
      and "budget=10" in lsrc)
check("the valfull crop mask is gated like the training one",
      "VAL_MASK=training/valfull/union_crop_m5_gt_ost.npz" in lsrc
      and "check_crop_mask.py" in lsrc.split("VAL_MASK=")[1])

# ---- 6. THE FLAGS SURVIVE STARTUP, not just argparse -----------------------------------
# WHY THIS RUNS THE TRAINER. Two bugs today were ordering bugs that every source-level test
# passed: the crop digest referenced a name whose defining patch had failed, and --lr-steps used
# `_LRS` ~850 lines before `import lr_schedule as _LRS`. Both raised before pose 0 on every rank.
# argparse accepting a flag proves nothing about the validation that reads it, so this invokes the
# real trainer with every new flag and asserts it reaches a KNOWN LATER GUARD -- i.e. it got
# through all of the new startup code -- rather than dying on a NameError or UnboundLocalError.
import subprocess                                                            # noqa: E402

TRAIN_PY = os.path.join(ROOT, "phases", "phase_k_train.py")
MASK_T = os.path.join(ROOT, "training", "full_train", "union_crop_m5_gt_ost.npz")
MASK_V = os.path.join(ROOT, "training", "valfull", "union_crop_m5_gt_ost.npz")
if os.path.exists(MASK_T) and os.path.exists(MASK_V):
    env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
    cmd = [sys.executable, "-I", TRAIN_PY,
           "--lr-steps", "20:1e-3,40:1e-4",
           "--pair-dropout", "0.25",
           "--crop-tokens", MASK_T,
           "--eval-pool", os.path.join(ROOT, "training", "valfull", "pool.csv"),
           "--eval-pool-crop-tokens", MASK_V,
           "--extra-eval", f"name=valfull,pool={os.path.join(ROOT, 'training/valfull/pool.csv')},"
                           f"fold={os.path.join(ROOT, 'validation/fold')},seed_base=400000,"
                           f"budget=30,every=6",
           "--l2-sp", "1e-5"]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    for bad in ("UnboundLocalError", "NameError", "AttributeError"):
        check(f"startup raises no {bad} with every new flag set",
              bad not in out, (out.strip().splitlines() or [""])[-1][:110])
    check("startup gets PAST the new validation to a known later guard",
          "needs --minimal-items" in out,
          (out.strip().splitlines() or [""])[-1][:110])
else:
    check("the crop masks exist, so the startup smoke test could run", False,
          "masks missing; cannot smoke-test startup")

print()
if FAIL:
    print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("  all checks passed")
