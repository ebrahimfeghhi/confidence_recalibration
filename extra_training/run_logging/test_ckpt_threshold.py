#!/usr/bin/env python
"""CPU regression for --ckpt-token-threshold: the RULE, and the guards that refuse a no-op.

NO GPU AND NO MODEL. What can be wrong here is the predicate -- off-by-one at the boundary, the
wrong sense, a unit with an unknown token count (-1) silently taking the uncheckpointed path --
and all of that is arithmetic. The memory question is bench_ckpt_by_size.py's and needs cards.
"""
import subprocess
import sys

PY_BIN = "/home/ebrahim/Protenix/.venv/bin/python"
TRAIN = "phases/phase_k_train.py"


def decide(n_tok, threshold, blocks_per_ckpt):
    """The trainer's rule, kept here so a change to one has to change the other."""
    if threshold <= 0:
        return blocks_per_ckpt or None
    return None if 0 <= n_tok <= threshold else (blocks_per_ckpt or None)


FAIL = []


def eq(got, want, what):
    if got != want:
        FAIL.append(f"{what}: got {got!r}, want {want!r}")


# AT the threshold is UNCHECKPOINTED -- "at or below" in the help, so 600 with --threshold 600
# must not checkpoint. One off here silently leaves the whole median bucket checkpointed.
eq(decide(600, 600, 1), None, "n_tok == threshold")
eq(decide(599, 600, 1), None, "n_tok just below")
eq(decide(601, 600, 1), 1, "n_tok just above")
eq(decide(1543, 600, 1), 1, "the largest unit keeps checkpointing")

# A UNIT WHOSE TOKEN COUNT IS UNKNOWN (-1, no asym_id) MUST KEEP CHECKPOINTING. -1 <= threshold
# is true for every positive threshold, so a naive `n_tok <= threshold` would run the one unit we
# know least about with the most memory.
eq(decide(-1, 600, 1), 1, "unknown n_tok keeps checkpointing")

# THRESHOLD OFF is the live behaviour, unchanged, for every size.
eq(decide(100, 0, 1), 1, "threshold 0 keeps checkpointing on small units")
eq(decide(1543, 0, 1), 1, "threshold 0 keeps checkpointing on large units")
eq(decide(100, 0, 0), None, "threshold 0, blocks 0 is off everywhere")

# ---- the parse-time guards ----
def refuses(args, needle):
    r = subprocess.run([PY_BIN, "-I", TRAIN] + args,
                       capture_output=True, text=True, timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    if needle not in out:
        FAIL.append(f"{' '.join(args[:4])}: expected a refusal mentioning {needle!r}, got "
                    f"rc={r.returncode} {out.strip()[:200]!r}")


refuses(["--ckpt-token-threshold", "600", "--blocks-per-ckpt", "0"],
        "needs --blocks-per-ckpt > 0")
refuses(["--ckpt-token-threshold", "-1"], "cannot be negative")

# THE DUPLICATED RULE IS ONLY WORTH TESTING IF IT IS STILL THE TRAINER'S RULE. decide() is a copy,
# so a change to phase_k_train.py that this file does not follow would leave eight checks passing
# against code that no longer runs. Pin the expression itself.
_EXPR = ("_bpc = (None if 0 <= n_tok <= a.ckpt_token_threshold\n"
         "                    else (a.blocks_per_ckpt or None))")
_src = open(TRAIN).read()
if _EXPR not in _src:
    FAIL.append("the trainer's selection expression has changed and decide() still copies the old "
                "one -- update both, or these checks pass against code that no longer runs")

if FAIL:
    print("FAIL")
    for f in FAIL:
        print("  " + f)
    sys.exit(1)
print(f"all checks pass ({8} rule case(s), 2 guard(s))")
