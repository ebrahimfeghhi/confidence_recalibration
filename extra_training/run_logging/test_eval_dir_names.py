#!/usr/bin/env python
"""Gate: the during-training eval files are named after the POOL, and both conventions read back.

WHAT THIS PROTECTS. The trainer computes the pool tag inline (one basename/dirname expression) and
run_logging/eval_dirs.pool_tag computes it again for readers. Two copies of one rule drift, and the
failure is silent: the trainer writes metric_valfull_ep30.json, the reader looks for something else,
and the epoch simply goes missing from the plot rather than erroring. So the first test asserts the
two implementations agree on the paths this project actually uses, by executing the trainer's own
source line rather than a restatement of it.

ASSERTS THE RENAME CHANGES SOMETHING. A naming gate that passes on the old behaviour is worthless,
so test_tag_is_not_train pins that a held-out pool does NOT produce "train".
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import eval_dirs                                                             # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ---- the trainer's expression and the reader's function are the same rule --------------------
def trainer_tag(train_pool):
    """The trainer's own line, lifted from phase_k_train.py by regex so it cannot be paraphrased."""
    src = open(os.path.join(ROOT, "phases", "phase_k_train.py")).read()
    m = re.search(r"_ptag = \((.*?)\) if train_pool else \"\"", src, re.S)
    assert m, "the trainer's _ptag expression was not found -- this gate is stale"
    # the expression is wrapped across source lines, so flatten it before eval
    expr = " ".join(m.group(1).split())
    return eval(expr, {"os": os}, {"train_pool": train_pool})                # noqa: S307


POOLS = ["training/valfull/pool.csv", "training/full_train/pool_winnable.csv",
         "training/overfit30/pool.csv", "training/val10/pool.csv",
         "/abs/training/valfull/pool.csv", "training/valfull/pool.csv/"]
for _p in POOLS:
    check(f"writer and reader agree on {_p}",
          trainer_tag(_p) == eval_dirs.pool_tag(_p),
          f"{trainer_tag(_p)!r} vs {eval_dirs.pool_tag(_p)!r}")

check("valfull pool tags as 'valfull'", eval_dirs.pool_tag("training/valfull/pool.csv") == "valfull")
check("winnable pool tags as 'full_train'",
      eval_dirs.pool_tag("training/full_train/pool_winnable.csv") == "full_train")
# THE RENAME MUST CHANGE SOMETHING: a held-out pool must not be called "train" any more.
check("a held-out pool does NOT tag as 'train'",
      eval_dirs.pool_tag("training/valfull/pool.csv") != "train")

# ---- both conventions read back, and the pool-tagged one wins --------------------------------
import tempfile                                                             # noqa: E402

with tempfile.TemporaryDirectory() as d:
    for n in ("metric_train_ep5.json", "metric_train_ep10.json", "metric_valfull_ep10.json",
              "metric_valfull_ep15.json", "metric_ep99.json"):
        open(os.path.join(d, n), "w").write("{}")
    for n in ("train_eval_shards_ep5", "train_eval_shards_ep10", "valfull_eval_shards_ep10",
              "valfull_eval_shards_ep15", "eval_shards_ep99"):
        os.makedirs(os.path.join(d, n))
    mj, sd = eval_dirs.metric_jsons(d), eval_dirs.shard_dirs(d)
    check("legacy-only epoch is found", os.path.basename(mj.get(5, "")) == "metric_train_ep5.json")
    check("pool-tagged epoch is found",
          os.path.basename(mj.get(15, "")) == "metric_valfull_ep15.json")
    check("pool-tagged WINS over legacy on the same epoch",
          os.path.basename(mj.get(10, "")) == "metric_valfull_ep10.json",
          os.path.basename(mj.get(10, "")))
    check("the OTHER eval path is not picked up (metric_ep99.json)", 99 not in mj)
    check("shard dirs: legacy found", os.path.basename(sd.get(5, "")) == "train_eval_shards_ep5")
    check("shard dirs: pool-tagged wins",
          os.path.basename(sd.get(10, "")) == "valfull_eval_shards_ep10")
    check("shard dirs: the OTHER eval path is not picked up", 99 not in sd)
    check("epochs are ints, not strings", all(isinstance(k, int) for k in list(mj) + list(sd)))

# ---- the live run reads back whatever it is called right now ---------------------------------
_live = os.path.join(ROOT, "training", "runs", "o2qczq")
if os.path.isdir(_live):
    mj, sd = eval_dirs.metric_jsons(_live), eval_dirs.shard_dirs(_live)
    check("o2qczq's five eval epochs resolve", sorted(mj) == [5, 10, 15, 20, 25], str(sorted(mj)))
    check("o2qczq's five shard dirs resolve", sorted(sd) == [5, 10, 15, 20, 25], str(sorted(sd)))

print(f"\n  {'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks pass'}")
sys.exit(1 if FAIL else 0)
