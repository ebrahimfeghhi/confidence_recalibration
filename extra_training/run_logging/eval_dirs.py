#!/usr/bin/env python
"""Where a run's during-training evaluation landed, under either naming convention.

WHY THIS EXISTS. The trainer's --train-eval-* mechanism wrote metric_train_ep<N>.json and
train_eval_shards_ep<N>/ whatever pool it was pointed at. The name describes the MECHANISM, and
the mechanism was built to score the run's own --seeds -- but o2qczq points it at
training/valfull/pool.csv, which is HELD OUT, so five files called "metric_train" hold the one
number in this project that is not in-sample. That is the opposite of what the name says, and the
numbers get read off disk months later by someone who did not launch the run.

THE FIX IS TO NAME THE POOL, NOT THE MECHANISM. A pool is held out or it is not, and only the pool
knows which:

    metric_valfull_ep25.json      valfull_eval_shards_ep25/
    metric_full_train_ep25.json   full_train_eval_shards_ep25/

"heldout" would have been the wrong word to standardise on -- --train-eval-pool pointed at the
training pool really is in-sample, and a name that lies in the other direction is no better.

OLD NAMES KEEP WORKING, and not by a compatibility branch: the old name is exactly the new scheme
with the tag "train", so one glob reads both. That matters beyond history -- a resumed arm writes
to a FRESH run directory, so the chain o2qczq -> <resume> spans both conventions and every reader
has to handle the mix whether or not anything is renamed.

NOT EVERY SCRIPT USES THIS. The one-off analyses that read a single historical run still glob the
old name directly; they are correct on the runs they were written for, and rewriting twenty of
them to look for files those runs will never have would add risk without adding information. The
live plotting path -- plot_rel_bars, plot_rel_heldout, metrics_io, train_eval_metrics -- goes
through here.
"""
import glob
import os
import re

_MJ = re.compile(r"^metric_(?P<tag>.+)_ep(?P<ep>\d+)\.json$")
_SD = re.compile(r"^(?P<tag>.+)_eval_shards_ep(?P<ep>\d+)$")


def pool_tag(pool_path):
    """The tag naming a pool: the directory holding it, e.g. training/valfull/pool.csv -> valfull.

    The basename is nearly always "pool.csv", so it is the parent that carries the identity.
    """
    d = os.path.dirname(os.path.abspath(str(pool_path)))
    return os.path.basename(d.rstrip("/")) or "pool"


def metric_jsons(run_dir):
    """{epoch: path} for the during-training metric jsons, both conventions.

    A pool-tagged file WINS over a "train"-tagged one for the same epoch, so renaming a run by
    hand mid-flight cannot make an epoch read twice or read the stale copy.
    """
    out, legacy = {}, {}
    for p in glob.glob(os.path.join(run_dir, "metric_*_ep*.json")):
        m = _MJ.match(os.path.basename(p))
        if not m:
            continue
        (legacy if m.group("tag") == "train" else out)[int(m.group("ep"))] = p
    for e, p in legacy.items():
        out.setdefault(e, p)
    return out


def shard_dirs(run_dir):
    """{epoch: path} for the during-training shard work directories, both conventions."""
    out, legacy = {}, {}
    for p in glob.glob(os.path.join(run_dir, "*_eval_shards_ep*")):
        m = _SD.match(os.path.basename(p))
        if not m or not os.path.isdir(p):
            continue
        (legacy if m.group("tag") == "train" else out)[int(m.group("ep"))] = p
    for e, p in legacy.items():
        out.setdefault(e, p)
    return out


def chain_shard_dirs(rids, runs_root):
    """{epoch: path} across a resume chain, oldest first, later runs winning a repeated epoch."""
    out = {}
    for r in rids:
        out.update(shard_dirs(os.path.join(runs_root, r)))
    return out


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:] or ["."]:
        print(f"  {d}")
        for name, f in (("metric json", metric_jsons), ("shard dir", shard_dirs)):
            for e, p in sorted(f(d).items()):
                print(f"    {name:>11} ep{e:<4} {os.path.basename(p)}")
