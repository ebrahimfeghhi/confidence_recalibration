#!/usr/bin/env python
"""A SHIPPED pass must install NOTHING, and must be refused before the GPU if it would.

THE FAILURE THIS FILE EXISTS FOR, MEASURED 2026-11-18. The shipped head was scored on val through
a copy of shell_scripts/score_checkpoint.sh with only its `CKPT=${CKPT:?}` guard relaxed. That
script's COMMON hardcodes --pose-perblock --pose-perblock-copy --frame-direction, because every arm
in this project trains with them -- so the pass printed "using the SHIPPED head (no --ckpt)" and
then installed three pose-injection pathways on top of the shipped weights. --pose-perblock-copy
copies block 0's pretrained injection into blocks 1-3, which IS a different model, so its ipTM no
longer reproduced the fold's stored values and the merge's self-check refused: 21,045 of 21,150
rows over tolerance, after twelve minutes on three cards. The number that pass was aiming at --
the shipped head's MEAN-pooled val DockQ, which nothing in the repo can derive from `stored_iptm`
because that column is max-pooled -- was not produced.

SO THERE ARE THREE THINGS TO HOLD, and a section each:

  1. THE SHIPPED ARGV IS THE TRAINED ARGV MINUS THE PATHWAY FLAGS, derived rather than retyped.
     Section 1 runs both scripts under DRY=1 and compares their argv token by token. Two
     hand-maintained command lines drifting apart is the defect itself, not a hypothetical.
  2. A SHIPPED PASS THAT WOULD INSTALL A PATHWAY IS REFUSED AT STARTUP, before the model, the
     trunks or a card. Section 2 asserts this for EVERY pose_/frame_ flag phase_k_eval declares,
     read out of its own argparse -- so a pathway flag added next month is covered the day it is
     added, which is the property a hand-typed list cannot have.
  3. THE POST-HOC SELF-CHECK SURVIVES. It is what caught the mistake, and nothing here may
     weaken it. Section 3 merges real rows from that failed pass -- kept as a fixture precisely
     because they are a known-wrong measurement -- and asserts the refusal still fires and now
     names the likely cause instead of sending the reader to debug three correct joins.

NO GPU ANYWHERE. Every check is argv parsing, a startup refusal, or the aggregate half of the
evaluator over rows already on disk.
"""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
EVAL = os.path.join(ROOT, "phases", "phase_k_eval.py")
SHIPPED_SH = os.path.join(ROOT, "shell_scripts", "score_shipped_head.sh")
CKPT_SH = os.path.join(ROOT, "shell_scripts", "score_checkpoint.sh")
VAL_POOL = os.path.join(ROOT, "training/valfull83/pool.csv")
VAL_MAN = os.path.join(ROOT, "training/valfull83/seed_manifest.json")
VAL_ROLES = os.path.join(ROOT, "validation/fold/chain_roles_all83.csv")
VAL_RESULTS = os.path.join(ROOT, "validation/fold/results.csv")
VAL_FOLD = os.path.join(ROOT, "validation/fold")
# THE NEGATIVE EXAMPLE, KEPT. 24 rows of the 2026-11-18 shipped-plus-injections pass: real shipped
# weights, real pose injections, so `iptm` and `stored_iptm` disagree by 0.1-0.4 on every row. The
# scratchpad they came from is session-scoped and would have taken the only evidence with it.
BAD_ROWS = os.path.join(ROOT, "run_logging/fixtures/shipped_pathway_rows.csv")
# WHAT THAT PASS INSTALLED, verbatim from COMMON in score_checkpoint.sh.
PATHWAY = ["--pose-perblock", "--pose-perblock-copy", "--frame-direction"]

FAIL = []
TMP = tempfile.mkdtemp(prefix="test_shipped_eval_")


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def dry(script, env):
    """The DRY=1 lines of a scoring script, as {side: [argv tokens]}."""
    r = subprocess.run(["bash", script], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "DRY": "1", **env})
    out = {}
    for line in (r.stdout + r.stderr).splitlines():
        m = re.match(r"DRY ([^:]+):\s+(.*)", line.strip())
        if m:
            out[m.group(1).split()[0]] = m.group(2)
    return out, r


def extra_of(cmdline):
    """The tokens inside --extra "..." of a printed argv, plus the argv's own flags."""
    m = re.search(r'--extra "([^"]*)"', cmdline)
    inner = m.group(1).split() if m else []
    outer = cmdline[:m.start()].split() if m else cmdline.split()
    return outer, inner


def eval_pathway_flags():
    """Every pose_/frame_ option phase_k_eval declares, and whether it takes a value.

    READ OUT OF THE PARSER'S OWN SOURCE, the same way run_logging/pathway_flags.py does it. A list
    typed here would be a second copy to keep in step -- and a second copy drifting from the first
    is the whole defect under test.
    """
    src = open(EVAL).read()
    out = []
    for m in re.finditer(r'ap\.add_argument\("(--(?:pose|frame)-[a-z0-9-]+)"([^)]*)', src):
        flag, rest = m.group(1), m.group(2)
        out.append((flag, "type=int" in rest))
    return out


def main():
    print(__doc__.strip().splitlines()[0])

    # ------------------------------------------------------------------------------------------
    print("\n1  THE SHIPPED ARGV IS THE TRAINED ARGV MINUS THE PATHWAY FLAGS")
    fake = os.path.join(TMP, "head_ep9_step9.pt")      # no args.json beside it -> no derived flags
    open(fake, "w").close()
    ck, rck = dry(CKPT_SH, {"CKPT": fake, "TAG": "argvcheck", "SIDES": "both",
                            "SD": os.path.join(TMP, "sd")})
    sh, rsh = dry(SHIPPED_SH, {"SIDES": "both"})
    ok(rck.returncode == 0 and {"slice", "val"} <= set(ck),
       "score_checkpoint.sh prints both sides' argv under DRY=1", f"exit {rck.returncode}")
    ok(rsh.returncode == 0 and len(sh) == 2,
       "score_shipped_head.sh prints both sides' argv under DRY=1", f"exit {rsh.returncode}")
    if rck.returncode or rsh.returncode:
        print((rck.stdout + rck.stderr + rsh.stdout + rsh.stderr)[-1500:])
        return 1
    pairs = [("val", [k for k in sh if k.startswith("val")][0], 30),
             ("slice", [k for k in sh if k.startswith("in-sample")][0], 15)]
    for ckside, shside, budget in pairs:
        _co, ci = extra_of(ck[ckside])
        _so, si = extra_of(sh[shside])
        # THE SHIPPED SIDE IS THE TRAINED SIDE WITH THE PATHWAY FLAGS DELETED AND NOTHING ELSE
        # TOUCHED. Compared as multisets of tokens, which is what argparse sees.
        ok(sorted(t for t in ci if t not in PATHWAY) == sorted(si),
           f"{ckside}: shipped --extra == trained --extra minus {len(PATHWAY)} pathway flag(s)",
           f"{sorted(set(ci) ^ set(si))}")
        ok(not [t for t in si if t.startswith(("--pose", "--frame"))],
           f"{ckside}: the shipped --extra carries no --pose-* or --frame-* token")
        ok("--pool-mean" in si and "--pair-iptm-only" in si,
           f"{ckside}: both poolings come out of the one pass (--pool-mean, --pair-iptm-only)")
        ok("--ckpt" not in _so, f"{ckside}: and no --ckpt, which is what selects the shipped head")
        ok(f"--budget {budget}" in sh[shside], f"{ckside}: at budget {budget}")
    ok("--seed-base 400000" in " ".join(extra_of(sh[pairs[0][1]])[1])
       and os.path.basename(VAL_MAN) in sh[pairs[0][1]],
       "val is seed base 400000 with the pool's own seed manifest")
    ok("--seed-base 500000" in " ".join(extra_of(sh[pairs[1][1]])[1])
       and "/training/fold " in sh[pairs[1][1]] + " ",
       "in-sample is seed base 500000 on the training fold")
    # AND THE TWO COMMONs ARE ONE EDIT APART, checked in the source as well as in the argv: the
    # argv check above would still pass if someone added a pathway flag to BOTH scripts.
    c_common = re.search(r'^COMMON="([^"]*)"', open(CKPT_SH).read(), re.M).group(1).split()
    s_common = re.search(r'^COMMON="([^"]*)"', open(SHIPPED_SH).read(), re.M).group(1).split()
    ok(sorted(t for t in c_common if t not in PATHWAY and not t.startswith("${")) == sorted(s_common),
       "and the two COMMON strings differ only by the pathway flags and POSE_FLAGS",
       " ".join(s_common))

    # ------------------------------------------------------------------------------------------
    print("\n2  A SHIPPED PASS THAT WOULD INSTALL A PATHWAY IS REFUSED AT STARTUP")
    # A POOL THAT DOES NOT EXIST, deliberately. It makes the run die quickly either way, so the
    # question each case answers is WHICH refusal comes first -- which is the whole point: the
    # flag-vs-checkpoint check has to come before anything is read, loaded or scored.
    base = [PY, "-I", EVAL, "--budget", "30", "--fold", VAL_FOLD, "--results", VAL_RESULTS,
            "--pool", os.path.join(TMP, "no_such_pool.csv"), "--roles", VAL_ROLES,
            "--seed-base", "400000", "--pair-iptm-only", "--pool-mean"]
    env = {**os.environ, "LAYERNORM_TYPE": "openfold"}
    clean = subprocess.run(base, capture_output=True, text=True, env=env, cwd=ROOT)
    ok("STOP (startup)" not in (clean.stdout + clean.stderr),
       "a clean shipped pass is NOT refused by the guard (it fails later, on the bogus pool)",
       (clean.stdout + clean.stderr).strip().splitlines()[-1][:70] if clean.returncode else "ran on")
    flags = eval_pathway_flags()
    ok(len(flags) >= 10, f"phase_k_eval declares {len(flags)} pose_/frame_ option(s) to cover")
    missed, allowed = [], []
    for flag, takes_value in flags:
        argv = base + [flag] + (["0"] if takes_value else [])
        r = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT)
        txt = r.stdout + r.stderr
        refused = r.returncode != 0 and "STOP (startup)" in txt and flag in txt
        if flag.replace("-", "_").lstrip("_") in ("pose_split_check",):
            # THE ONE DOCUMENTED EXCEPTION: --pose-split-check scores one pose per forward with
            # the same modules and the same weights. It installs nothing, so it is legal here --
            # and NOT_ARCHITECTURE in phase_k_eval is where that judgement lives.
            allowed.append((flag, refused))
            continue
        if not refused:
            missed.append((flag, r.returncode, txt.strip().splitlines()[-1][:70] if txt else ""))
    ok(not missed, "every architecture-installing flag is refused on a shipped pass, by name",
       f"missed {missed}" if missed else f"{len(flags) - len(allowed)} flag(s)")
    ok(all(not refused for _f, refused in allowed),
       "while --pose-split-check, which installs nothing, is still allowed")
    # AND A TRAINED PASS IS UNTOUCHED: the same flags with a --ckpt must not hit this guard.
    r = subprocess.run(base + ["--ckpt", fake] + PATHWAY, capture_output=True, text=True,
                       env=env, cwd=ROOT)
    ok("STOP (startup)" not in (r.stdout + r.stderr),
       "a pass WITH --ckpt and the same flags is not refused -- that is the normal arm path")
    # AND THE DELIBERATE ABLATION IS AVAILABLE, but only fully asked for.
    r = subprocess.run(base + PATHWAY + ["--shipped-with-pathways"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    ok(r.returncode != 0 and "--shipped-with-pathways needs --no-self-check" in (r.stdout + r.stderr),
       "--shipped-with-pathways without --no-self-check is refused at startup, not at the merge")
    r = subprocess.run(base + PATHWAY + ["--shipped-with-pathways", "--no-self-check"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    ok("STOP (startup)" not in (r.stdout + r.stderr),
       "and with both flags the ablation is allowed to run")

    # ------------------------------------------------------------------------------------------
    print("\n3  THE POST-HOC GUARD IS STILL THE LAST LINE, AND NOW NAMES THE CAUSE")
    rows = list(csv.DictReader(open(BAD_ROWS)))
    ok(len(rows) >= 20 and all(abs(float(r["iptm"]) - float(r["stored_iptm"])) > 0.05 for r in rows),
       f"the fixture is {len(rows)} genuinely wrong row(s) from that pass",
       f"worst {max(abs(float(r['iptm']) - float(r['stored_iptm'])) for r in rows):.4f}")

    def merge(meta_extra, argv_extra):
        """Merge the fixture as ONE shard with the given meta, and return the output."""
        d = os.path.join(TMP, "merge%d" % len(os.listdir(TMP)))
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "rows_b30_shipped_s0.csv")
        open(p, "w").write(open(BAD_ROWS).read())
        meta = {"shard": 0, "nshards": 1, "units": len(rows), "rows": len(rows), "n_ost": 0,
                "n_unresolvable": 0, "n_differ": 0, "differ_pdbs": {}, "n_unmapped": 0,
                "unmapped": [], "budget": 30, "ckpt": ""}
        meta.update(meta_extra)
        json.dump(meta, open(p + ".meta.json", "w"))
        r = subprocess.run([PY, "-I", EVAL, "--rows-in", p, "--budget", "30", "--fold", VAL_FOLD,
                            "--results", VAL_RESULTS, "--pool", VAL_POOL, "--roles", VAL_ROLES,
                            "--seed-base", "400000", "--seed-manifest", VAL_MAN,
                            "--pair-iptm-only", "--pool-mean"] + argv_extra,
                           capture_output=True, text=True, env=env, cwd=ROOT)
        return r, r.stdout + r.stderr

    # 3a. THE SHARDS' OWN RECORD. Rows written by a shipped pass that installed pathways carry the
    # flag list in their meta, so the merge can name them instead of guessing.
    r, txt = merge({"pathway_flags": PATHWAY}, [])
    ok(r.returncode != 0 and "STOP (merge)" in txt and all(f in txt for f in PATHWAY),
       "3a a merge of shipped rows whose shards installed pathways is refused, naming the flags",
       txt.strip().splitlines()[0][:80])

    # 3b. THE 2026-11-18 ROWS AS THEY REALLY ARE: written before the meta carried that field, so
    # the only evidence is the merge's own argv -- which phase_k_eval_sharded.py passes through
    # from the shards. Caught there too, so the rows already on disk are not a blind spot.
    r, txt = merge({}, PATHWAY)
    ok(r.returncode != 0 and "STOP (merge)" in txt,
       "3b legacy rows with no recorded flags are still caught, from the merge's own argv",
       txt.strip().splitlines()[0][:80])

    # 3c. AND WITH NO FLAG ANYWHERE the self-check is what fires -- the guard that actually caught
    # this. It must still refuse, and its message must point at the model before the joins.
    r, txt = merge({}, [])
    ok(r.returncode != 0 and "does not reproduce" in txt,
       "3c with no flag recorded anywhere, the stored_iptm self-check still refuses")
    ok("changed model or a changed input" in txt,
       "3c and names near-total disagreement as a changed model, not a broken join")
    ok("three joins" in txt, "3c while still naming the three joins as the other suspect")

    # 3d. THE CLEAN SHIPPED PASS STILL PASSES ITS SELF-CHECK. Without this, every check above is
    # satisfied by an evaluator that refuses everything.
    good = os.path.join(ROOT, "training/valfull83/shipped_rho_shards")
    rowfiles = sorted(os.path.join(good, f) for f in os.listdir(good)
                      if re.match(r"rows_.*_s\d\.csv$", f)) if os.path.isdir(good) else []
    if len(rowfiles) == 3:
        r = subprocess.run([PY, "-I", EVAL, "--rows-in", ",".join(rowfiles), "--budget", "30",
                            "--fold", VAL_FOLD, "--results", VAL_RESULTS, "--pool", VAL_POOL,
                            "--roles", VAL_ROLES, "--seed-base", "400000",
                            "--seed-manifest", VAL_MAN, "--pair-iptm-only", "--pool-mean"],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        txt = r.stdout + r.stderr
        ok(r.returncode == 0 and "the joins reproduce the fold's own ipTM" in txt,
           "3d the real pathway-free shipped val rows pass the self-check",
           re.search(r"max \|diff\| [0-9.]+", txt).group(0) if "max |diff|" in txt else "")
        ok("METRIC (shipped," in txt and "mean_dockq 0.5813" in txt,
           "3d and still report the shipped val number this project quotes",
           "0.5813 max / 0.5735 mean")
    else:
        ok(False, "3d the clean shipped val rows are on disk to check against",
           f"{len(rowfiles)} row file(s) in {good}")

    # ------------------------------------------------------------------------------------------
    print("\n4  THE DERIVATION CANNOT BE ASKED ABOUT THE SHIPPED HEAD")
    r = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging/pathway_flags.py"), ""],
                       capture_output=True, text=True)
    ok(r.returncode == 2 and "EMPTY checkpoint path" in (r.stdout + r.stderr),
       "pathway_flags.py refuses an empty --ckpt instead of deriving from the current directory",
       f"exit {r.returncode}")
    r = subprocess.run([PY, "-I", os.path.join(ROOT, "run_logging/pathway_flags.py"),
                        os.path.join(ROOT, "training/runs/w57hjo/head_ep20_step2900.pt")],
                       capture_output=True, text=True)
    ok(r.returncode == 0 and "--pose-readout-prenorm" in r.stdout,
       "while a trained head's derivation is unchanged -- Arm B's flags still come back",
       r.stdout.strip()[:70])

    # ------------------------------------------------------------------------------------------
    print("\n5  THE SHIPPED SCRIPT REFUSES THE WAYS IT WOULD BE MISUSED")
    for env_extra, needle, name in (
            ({"POSE_FLAGS": "--pose-perblock"}, "STOP: POSE_FLAGS is set",
             "POSE_FLAGS is refused, not silently dropped"),
            ({"CKPT": "/tmp/x.pt"}, "STOP: CKPT is set",
             "a CKPT in the environment is refused, pointing at score_checkpoint.sh"),
            ({"SIDES": "valfull"}, "is not both|val|insample",
             "an unknown SIDES is refused before either side runs")):
        r = subprocess.run(["bash", SHIPPED_SH], capture_output=True, text=True, cwd=ROOT,
                           env={**os.environ, "DRY": "1", **env_extra})
        ok(r.returncode != 0 and needle in (r.stdout + r.stderr), f"5 {name}",
           f"exit {r.returncode}")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
