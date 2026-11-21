#!/usr/bin/env python
"""Correctness gate for --pae-split-accept: the per-epoch PAE loss split by pose group.

CPU only, no model, no cards -- safe to run while arms train.

THE BLOCKS ARE EXTRACTED FROM phase_k_train.py VERBATIM and exec'd here against hand-built
inputs, rather than reimplemented. A reimplementation of a classifier cannot catch the one defect
this classifier is most likely to have, because the defect would be copied into the test: the
POSE index is not the SAMPLE index, and _rank_lab is indexed by the sample. This file has paid for
that twice -- the rel target regressed pose 1 onto pose 4's true iTM, and --emit-iptm-rows wrote
`i` while 73.8% of its rows carried another sample's DockQ. So the fixture's sample_index is a
NON-IDENTITY permutation whose labels disagree with the ones the pose index would reach, and the
test asserts the answer changes when the permutation does.

WHAT IS ASSERTED
  1. The tally CHANGES something: the accumulators move, and the two means are the hand-computed
     1.5 and 5.0 with a difference of -3.5 over 2 and 2 poses.
  2. A pose that STRADDLES the threshold across its interfaces is in NEITHER mean and is counted.
     Its loss is 100.0, so either mean would be visibly wrong if it leaked in.
  3. A missing DockQ does NOT raise -- it is counted unclassified. Same for an unresolvable
     sample index and for the SystemExit _pose_pairs throws on an unreadable OST mapping. A
     diagnostic that can kill a training run in epoch 40,000 is worse than no diagnostic.
  4. Reading the labels by the POSE index instead of the sample index gives a DIFFERENT answer,
     so the join is actually being exercised.
  5. The csv keeps its old columns, appends the new ones, writes exactly ONE line per epoch, and
     the header field count equals the row field count on both a ranking epoch and a PAE-only one.
  6. The accumulators are reset at the boundary, so epoch 2 is not epochs 1 and 2 pooled.

Usage:  python -I run_logging/test_pae_split_accept.py
"""
import math
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "phases", "phase_k_train.py")
PY_BIN = "/home/ebrahim/Protenix/.venv/bin/python"

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def extract(first_line_startswith, lines, occurrence=0):
    """The block beginning at the line that starts with this text, to the end of its indent.

    Indentation decides the end, not a line count: any edit above or below the block moves the
    numbers, and a hard-coded range would silently test the wrong code after the next commit.
    """
    seen = -1
    for n, ln in enumerate(lines):
        if ln.strip().startswith(first_line_startswith):
            seen += 1
            if seen != occurrence:
                continue
            ind = len(ln) - len(ln.lstrip())
            out = [ln]
            for ln2 in lines[n + 1:]:
                if ln2.strip() and (len(ln2) - len(ln2.lstrip())) <= ind:
                    break
                out.append(ln2)
            return textwrap.dedent("".join(out))
    raise SystemExit(f"STOP: {first_line_startswith!r} is no longer in {SRC}; the test is "
                     f"pointing at code that has moved.")


LINES = open(SRC).readlines()
TALLY_SRC = extract("def _pae_split_tally(", LINES)
KEY_SRC = extract("def _rank_key(", LINES)
CSV_SRC = extract("if _sp_rank_due or _ps_on:", LINES)
# OCCURRENCE 2: `if _ps_on:` appears three times -- the label loader at startup, the per-pose
# hook in _one_pose, and this print block. The middle one is asserted below, because a tally that
# is never called is the one failure none of the arithmetic above could see.
HOOK_SRC = extract("if _ps_on:", LINES, occurrence=1)
PRINT_SRC = extract("if _ps_on:", LINES, occurrence=2)

# ---------------------------------------------------------------- the fixture
# REVERSED, so pose i reads sample 7 - i. 99.3% of the training label files carry a non-identity
# permutation, and the labels below are chosen so the pose index reaches a DIFFERENT class.
SAMP_REV = np.array([7, 6, 5, 4, 3, 2, 1, 0])
PDB, LAB_P, SEED = "test", "/x/free_500000.npz", "free_500000"
DQ = {
    # iface -> {sample: DockQ}; sample 2 is missing from test_AB on purpose
    "test_AB": {7: 0.90, 6: 0.80, 5: 0.10, 4: 0.05, 3: 0.90, 1: 0.50, 0: 0.50},
    "test_AC": {7: 0.70, 6: 0.60, 5: 0.20, 4: 0.00, 3: 0.10, 2: 0.40, 1: 0.50, 0: 0.50},
}
# POSE LOSSES. The two acceptable poses mean 1.5, the two unacceptable ones mean 5.0; every
# excluded pose carries 100.0 so a leak into either mean is unmissable.
V = {0: 1.0, 1: 2.0, 2: 4.0, 3: 6.0, 4: 100.0, 5: 100.0, 6: 100.0, 7: 100.0}
THRESH = 0.23


def build_ns(samp_index=SAMP_REV, raise_on_sample=(), thresh=THRESH):
    """The names _pae_split_tally closes over in phase_k_train.main()."""
    class A:
        pae_split_accept = thresh

    stat = {"ps_acc_sum": 0.0, "ps_acc_n": 0, "ps_un_sum": 0.0, "ps_un_n": 0,
            "ps_straddle": 0, "ps_unclass": 0, "ps_batched": 0}

    def _pose_pairs(pdb, lab_p, samp):
        if int(samp) in raise_on_sample:
            # EXACTLY WHAT THE REAL ONE DOES on an unreadable OST mapping: SystemExit, which is
            # not an Exception, which is why the tally's except names it.
            raise SystemExit("STOP: no OST chain mapping")
        # the per-pose asym pair deliberately differs from the static one the DockQ is keyed by
        return [("test_AB", 1, 0), ("test_AC", 2, 0)]

    ns = {"np": np, "math": math, "a": A(), "_rel_stat": stat,
          "_pose_pairs": _pose_pairs,
          "_rel_dqkey": {"test_AB": (0, 1), "test_AC": (0, 2)},
          "_rank_lab": {("test", 500000, (0, 1)): dict(DQ["test_AB"]),
                        ("test", 500000, (0, 2)): dict(DQ["test_AC"])}}
    exec(KEY_SRC, ns)
    exec(TALLY_SRC, ns)
    ns["lab"] = {"sample_index": samp_index}
    return ns


def run_all(ns, poses=range(8), nb_of=lambda i: 2 if i == 6 else 1, hint=None):
    for i in poses:
        ns["_pae_split_tally"](PDB, LAB_P, ns["lab"], SEED, i, hint, nb_of(i), V[i])
    return ns["_rel_stat"]


print("the split itself, on a non-identity sample_index")
ns = build_ns(raise_on_sample=(0,))          # pose 7 -> sample 0 -> unreadable mapping
before = dict(ns["_rel_stat"])
st = run_all(ns)
check("the tally changes the accumulators", st != before)
check("acceptable mean is 1.5 over 2 poses",
      st["ps_acc_n"] == 2 and abs(st["ps_acc_sum"] / st["ps_acc_n"] - 1.5) < 1e-12,
      f"n={st['ps_acc_n']} sum={st['ps_acc_sum']}")
check("unacceptable mean is 5.0 over 2 poses",
      st["ps_un_n"] == 2 and abs(st["ps_un_sum"] / st["ps_un_n"] - 5.0) < 1e-12,
      f"n={st['ps_un_n']} sum={st['ps_un_sum']}")
check("difference is -3.5",
      abs((st["ps_acc_sum"] / st["ps_acc_n"]) - (st["ps_un_sum"] / st["ps_un_n"]) + 3.5) < 1e-12)
# pose 4 -> sample 3: DockQ 0.90 on test_AB and 0.10 on test_AC
check("the straddler is excluded and counted", st["ps_straddle"] == 1, str(st))
check("no excluded pose leaked into a mean",
      st["ps_acc_sum"] == 3.0 and st["ps_un_sum"] == 10.0, str(st))
# pose 5 -> sample 2 (no test_AB label), pose 6 (--pose-batch row), pose 7 (SystemExit)
check("missing label, batched row and SystemExit are unclassified, not fatal",
      st["ps_unclass"] == 3 and st["ps_batched"] == 1, str(st))
check("every pose landed somewhere exactly once",
      st["ps_acc_n"] + st["ps_un_n"] + st["ps_straddle"] + st["ps_unclass"] == 8, str(st))

print("the pose index is not the sample index")
ns_id = build_ns(samp_index=np.arange(8), raise_on_sample=())
st_id = run_all(ns_id)
check("an identity permutation gives a different answer",
      (st_id["ps_acc_n"], st_id["ps_un_n"], st_id["ps_acc_sum"], st_id["ps_un_sum"])
      != (st["ps_acc_n"], st["ps_un_n"], st["ps_acc_sum"], st["ps_un_sum"]),
      f"identity={st_id}")
# BY HAND, pose -> sample identity: poses 0, 1 and 7 are acceptable (0.50/0.50, 0.50/0.50,
# 0.90/0.70) for 1.0 + 2.0 + 100.0; poses 4 and 5 unacceptable (0.05/0.00, 0.10/0.20) for 200.0;
# pose 3 straddles (0.90/0.10); pose 2 has no test_AB label and pose 6 is a batched row.
check("identity run matches the hand count: 3 acceptable, 2 unacceptable, 1 straddler",
      (st_id["ps_acc_n"], st_id["ps_acc_sum"], st_id["ps_un_n"], st_id["ps_un_sum"],
       st_id["ps_straddle"], st_id["ps_unclass"]) == (3, 103.0, 2, 200.0, 1, 2), str(st_id))

print("the sample index is only re-read when nothing resolved it")
ns_h = build_ns(samp_index=np.array([99] * 8), raise_on_sample=())
ns_h["_pae_split_tally"](PDB, LAB_P, ns_h["lab"], SEED, 3, 7, 1, 1.0)
check("a passed-in _samp_i wins over the label array",
      ns_h["_rel_stat"]["ps_acc_n"] == 1, str(ns_h["_rel_stat"]))
ns_b = build_ns(samp_index=None, raise_on_sample=())
ns_b["lab"] = {}                                  # no sample_index at all
ns_b["_pae_split_tally"](PDB, LAB_P, ns_b["lab"], SEED, 0, None, 1, 1.0)
check("a label with no sample_index is unclassified, not fatal",
      ns_b["_rel_stat"]["ps_unclass"] == 1, str(ns_b["_rel_stat"]))

print("a threshold is a threshold")
ns_t = build_ns(raise_on_sample=(0,), thresh=0.95)
st_t = run_all(ns_t)
# AT 0.95 the five classifiable poses (0, 1, 2, 3, 4) are all unacceptable and none straddles,
# so the sum is 1 + 2 + 4 + 6 + 100 = 113; the other three are the same exclusions as above.
check("at DockQ 0.95 nothing is acceptable and the straddler stops straddling",
      (st_t["ps_acc_n"], st_t["ps_un_n"], st_t["ps_un_sum"], st_t["ps_straddle"],
       st_t["ps_unclass"]) == (0, 5, 113.0, 0, 3), str(st_t))

# -------------------------------------------------------- the csv row and its header
print("spread.r<rank>.csv: one line per epoch, one header, old columns kept")
OLD_COLS = ["epoch", "n_ifaces", "rho", "sd_pred", "sd_true", "sd_ratio",
            "insample_dq", "insample_frac", "oracle_dq"]
NEW_COLS = ["pae_acc", "pae_unacc", "n_pae_acc", "n_pae_unacc",
            "n_pae_straddle", "n_pae_unclassified", "pae_acc_minus_unacc"]


def write_row(tmp, ep, stat, rank_due, ps_on, rho=()):
    def _med(v):
        w = sorted(v)
        return w[len(w) // 2] if w else float("nan")
    class A:
        out = tmp
    g = {"os": os, "a": A(), "rank": 0, "ep": ep - 1, "_med": _med,
         "_rel_stat": stat, "_sp_rank_due": rank_due, "_ps_on": ps_on}
    stat.setdefault("rho", list(rho))
    exec(CSV_SRC, g)


tmp = tempfile.mkdtemp()
st2 = dict(st)
write_row(tmp, 1, st2, rank_due=False, ps_on=True)
rows = open(os.path.join(tmp, "spread.r0.csv")).read().strip().split("\n")
hdr = rows[0].split(",")
check("header keeps the old columns in order and appends the new ones",
      hdr == OLD_COLS + NEW_COLS, str(hdr))
check("one header plus exactly one row for one epoch", len(rows) == 2, str(rows))
check("the row has as many fields as the header", len(rows[1].split(",")) == len(hdr),
      rows[1])
vals = dict(zip(hdr, rows[1].split(",")))
check("the row carries the hand-computed means",
      (float(vals["pae_acc"]), float(vals["pae_unacc"]),
       int(vals["n_pae_acc"]), int(vals["n_pae_unacc"]),
       int(vals["n_pae_straddle"]), int(vals["n_pae_unclassified"]),
       float(vals["pae_acc_minus_unacc"])) == (1.5, 5.0, 2, 2, 1, 3, -3.5), rows[1])
check("the ranking columns are nan on a PAE-only epoch, not 0",
      vals["rho"] == "nan" and vals["insample_dq"] == "nan", rows[1])
check("the accumulators were reset at the boundary",
      (st2["ps_acc_n"], st2["ps_un_n"], st2["ps_acc_sum"], st2["ps_un_sum"],
       st2["ps_straddle"], st2["ps_unclass"], st2["ps_batched"]) == (0, 0, 0.0, 0.0, 0, 0, 0),
      str(st2))

# A SECOND EPOCH ON THE SAME FILE: no second header, no pooling with epoch 1.
st3 = run_all(build_ns(raise_on_sample=(0,)))
write_row(tmp, 2, st3, rank_due=False, ps_on=True)
rows = open(os.path.join(tmp, "spread.r0.csv")).read().strip().split("\n")
check("epoch 2 appends one line and no second header",
      len(rows) == 3 and rows[0] == ",".join(OLD_COLS + NEW_COLS)
      and rows[2].startswith("2,"), str(rows))
check("epoch 2's mean is epoch 2's poses alone",
      float(dict(zip(hdr, rows[2].split(",")))["pae_acc"]) == 1.5, rows[2])

# A RANKING EPOCH, where the gate used to be the only writer: same header, ranking columns filled.
tmp2 = tempfile.mkdtemp()
st4 = run_all(build_ns(raise_on_sample=(0,)))
st4.update({"sd_s": [0.1], "sd_g": [0.2], "sd_ratio": [0.5],
            "sel_dq": [0.4, 0.1], "orc_dq": [0.9, 0.9]})
write_row(tmp2, 7, st4, rank_due=True, ps_on=True, rho=[0.3, 0.5, 0.7])
rows2 = open(os.path.join(tmp2, "spread.r0.csv")).read().strip().split("\n")
v2 = dict(zip(rows2[0].split(","), rows2[1].split(",")))
check("a ranking epoch writes one line with the same header",
      rows2[0].split(",") == OLD_COLS + NEW_COLS and len(rows2) == 2, str(rows2))
check("the ranking columns are unchanged by the append",
      (int(v2["epoch"]), int(v2["n_ifaces"]), float(v2["rho"]), float(v2["sd_ratio"]),
       float(v2["insample_dq"]), float(v2["insample_frac"]), float(v2["oracle_dq"]))
      == (7, 3, 0.5, 0.5, 0.25, 0.5, 0.9), rows2[1])
check("and the split columns are there too on a ranking epoch",
      float(v2["pae_acc"]) == 1.5 and int(v2["n_pae_straddle"]) == 1, rows2[1])

print("the per-pose hook is actually wired into the pose loop")
check("the hook calls the tally with the POSE loop's own sample index and nb",
      "_pae_split_tally(pdb, lab_p, lab, arm_seed, i, _samp_i, nb, out[\"pae_loss\"])"
      in HOOK_SRC, HOOK_SRC)
check("it is tallied from out[\"pae_loss\"], the number the run reports and differentiates",
      'out["pae_loss"]' in HOOK_SRC and "met[" not in HOOK_SRC, HOOK_SRC)

print("the epoch line itself")
st5 = run_all(build_ns(raise_on_sample=(0,)))


def say(stat, thresh=THRESH):
    """The real print block, with print captured. An f-string that cannot format is a crash at
    the epoch boundary of a run that has already paid for the epoch, so it is exercised here."""
    class A:
        pae_split_accept = thresh
    said = []
    exec(PRINT_SRC, {"a": A(), "rank": 2, "ep": 4, "_rel_stat": stat, "_ps_on": True,
                     "print": lambda *x, **k: said.append(x[0])})
    return said


line = say(st5)
check("one line, naming the rank and the epoch", len(line) == 1
      and line[0].startswith("  [rank 2] epoch 5 PAE BY POSE GROUP"), str(line))
check("the line carries both means, both counts, the straddler and the difference",
      all(t in line[0] for t in ("acceptable 1.5000 over 2 pose(s)",
                                 "unacceptable 5.0000 over 2 pose(s)",
                                 "acceptable minus unacceptable -3.5000",
                                 "1 pose(s) STRADDLED", "3 unclassified",
                                 "1 of them --pose-batch rows")), line[0])
empty = say({"ps_acc_sum": 0.0, "ps_acc_n": 0, "ps_un_sum": 0.0, "ps_un_n": 0,
             "ps_straddle": 0, "ps_unclass": 0, "ps_batched": 0})
check("an empty group prints nan rather than dividing by zero",
      "acceptable nan over 0 pose(s)" in empty[0] and "--pose-batch" not in empty[0], empty[0])

# ------------------------------------------------- the startup refusals, by RUNNING the trainer
print("startup refusals (the trainer is actually run; no cards are touched)")


def stop_text(extra):
    p = subprocess.run([PY_BIN, "-I", os.path.join(ROOT, "phases", "phase_k_train.py")] + extra,
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    return (p.stdout + p.stderr)


pool = os.path.join(tmp, "pool.csv")
open(pool, "w").write("pdb,iface\n")
base = ["--pae-split-accept", "0.23", "--fold", tmp]
txt = stop_text(base + ["--pae-scored-pairs", pool])
check("a missing label source refuses at startup, naming the flag",
      "STOP: --pae-split-accept 0.23 needs DockQ labels" in txt
      and os.path.join(tmp, "results.csv") in txt, txt.strip()[-400:])
txt = stop_text(base)
check("no --pae-scored-pairs refuses at startup, naming the flag",
      "STOP: --pae-split-accept needs --pae-scored-pairs" in txt, txt.strip()[-400:])
txt = stop_text(["--pae-split-accept", "23", "--fold", tmp, "--pae-scored-pairs", pool])
check("a DockQ outside (0, 1) refuses at startup",
      "STOP: --pae-split-accept 23.0 is not a DockQ threshold" in txt, txt.strip()[-400:])
# OFF BY DEFAULT: the same argv without the flag must get PAST these checks, or the refusal would
# be firing on every run in the repo.
txt = stop_text(["--fold", tmp, "--pae-scored-pairs", pool])
check("the flag is off by default", "pae-split-accept" not in txt, txt.strip()[-300:])

print()
if FAIL:
    print(f"FAILED {len(FAIL)}: " + "; ".join(FAIL))
    sys.exit(1)
print("all checks passed")
