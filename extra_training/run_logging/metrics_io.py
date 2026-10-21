#!/usr/bin/env python
"""THE single reader for the full-set run's metrics. The plot and the results table both use it.

WHY THIS EXISTS. plot_full_train.py and results_table.py were described as "two renderings of one
source". Review pointed out that was only half true: they read the same files but implemented
separate parsing, precedence and validation, and the two had already diverged where it mattered --
the plot warned when a metric json's budget differed from the plot's reference lines, and the table
silently took the number anyway. A budget-100 rescore could therefore have replaced a budget-30
value in the published table while the shipped-head comparison beside it stayed at budget 30. One
loader makes that a single decision instead of two.

A BUDGET MISMATCH IS A REJECTION, NOT A WARNING. 0.2626/0.5000 at 100 seeds and 0.2909/0.5556 at
30 are different pools, and the whole point of deriving the baseline from the scored pool was to
stop those being compared. A rejected json is reported by name so it is visible rather than
missing.
"""
import collections
import csv
import glob
import json
import os
import re

DOCKQ_LINE = re.compile(r"epoch (\d+) TRAIN: mean_dockq ([0-9.]+)\s+frac_acc ([0-9.]+)")
TOOK_LINE = re.compile(r"\[rank (\d+)\] epoch (\d+) took ([0-9.]+) min -- (.*)")
RESID_LINE = re.compile(r"\[rank (\d+)\] epoch (\d+) accounted ([0-9.]+)m of ([0-9.]+)m, "
                        r"residual ([0-9.-]+)m")
MEM_LINE = re.compile(r"\[rank (\d+)\] epoch (\d+) memory: peak allocated ([0-9.]+) GB, "
                      r"peak reserved ([0-9.]+) GB, blocks_per_ckpt=(\S+)")


class Metrics:
    """Per-epoch records for a joined lineage, with provenance and completeness."""

    def __init__(self):
        self.train_pae = {}        # epoch -> float
        self.train_steps = {}      # epoch -> int
        self.val_pae = {}          # epoch -> (float, n_poses)
        self.dockq = {}            # epoch -> (mean_dockq, frac_acc, source)
        self.segment = {}          # epoch -> run id that produced it
        self.timing = {}           # epoch -> {rank: (wall_min, buckets_str)}
        self.resid = {}            # epoch -> {rank: (accounted, wall, residual)}
        self.mem = {}              # epoch -> {rank: (peak_alloc, peak_reserved, ckpt)}
        self.rejected = []         # (path, reason)
        self.expected_steps = {}   # segment -> (full-epoch steps observed, threshold used)

    def epochs(self):
        return sorted(set(self.train_pae) | set(self.val_pae) | set(self.dockq))

    def complete(self, e, expect_poses=None):
        """All four metrics present, and the held-out pass covered the poses it should have.

        The watcher triggers on the DockQ line, which is the epoch boundary's last, so a complete
        epoch is the normal case. This exists so an INCOMPLETE one is labelled rather than
        rendered with em-dashes that read like a missing measurement.
        """
        if e not in self.train_pae or e not in self.dockq or e not in self.val_pae:
            return False
        v, n = self.val_pae[e]
        if v != v or v <= 0:          # NaN or nonsense
            return False
        if expect_poses and n != expect_poses:
            return False
        return True


def load(root, runs, budget, full_steps=0):
    """Read a '+'-joined lineage. `budget` is the seed budget the DockQ numbers must belong to.

    `full_steps` is an ABSOLUTE floor; 0 means derive the expectation per segment, which is now
    the only thing that works. The threshold used to be a flat 285 against a 290-step epoch, and
    when K-structures-per-step made an epoch 244 steps EVERY epoch of the new segment was
    silently rejected as partial -- epoch 7 reported a train PAE of "--" with 244 complete steps
    behind it. Segments do not share a step count, so no single number can serve them.

    Derived as the maximum steps any epoch of that segment reached, which is the full-epoch
    length by construction, minus 2% of slack. A partial final epoch falls below it and is still
    dropped, which is the behaviour that matters: an epoch's last step is a remainder and
    averaging it in is how a 265/290-step read once got published as a finished number.
    """
    m = Metrics()
    for part in runs.split("+"):
        d = os.path.join(root, part)
        f = os.path.join(d, "trajectory.csv")
        if os.path.exists(f):
            acc = collections.defaultdict(lambda: [0.0, 0, 0])
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    if r.get("step_taken") != "1":
                        continue
                    try:
                        e, v, n = int(r["epoch"]), float(r["batch_pae_loss"]), int(r["batch_n"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    acc[e][0] += v * n; acc[e][1] += n; acc[e][2] += 1
            seg_full = max((k for _s, _n, k in acc.values()), default=0)
            need = full_steps if full_steps else max(1, int(0.98 * seg_full))
            m.expected_steps[part] = (seg_full, need)
            # AN EPOCH IS NOT SETTLED UNTIL A LATER ONE EXISTS. A step threshold alone cannot
            # tell a finished epoch from one a few steps short: epoch 6 of 2w5to5 cleared the
            # flat 285 at 285/289 steps and its train PAE was published as 2.5301, which became
            # 2.5270 once the last four steps landed. The same near-miss is possible against any
            # threshold, so the threshold is no longer the only test.
            #
            # Settled means: a LATER epoch of this segment has started, or this segment is not
            # the newest in the lineage, or the run has written status.json and is over.
            _last_ep = max(acc) if acc else 0
            _is_newest = (part == runs.split("+")[-1])
            _ended = os.path.exists(os.path.join(d, "status.json"))
            # AN INDEPENDENT COUNT OF WHERE EACH EPOCH ENDED, so the newest epoch need not be
            # withheld. val_loss.csv's `step` is CUMULATIVE optimiser steps at the end of that
            # epoch, written by the boundary monitor and not derived from the trajectory at all --
            # so step(e) - step(e-1) is how many steps epoch e should contain, and an epoch with
            # all of them is finished whether or not a later epoch has started. This is the exact
            # test the 0.98 threshold could not be: epoch 6 of 2w5to5 had 285 of 289 and cleared
            # 0.98, and would fail this.
            _step_at = {}
            _vl = os.path.join(d, "val_loss.csv")
            if os.path.exists(_vl):
                with open(_vl) as fh:
                    for r in csv.DictReader(fh):
                        try:
                            _step_at[int(r["epoch"])] = int(r["step"])
                        except (ValueError, KeyError, TypeError):
                            continue

            def _expected(e, _sa=_step_at):
                """Steps epoch e should hold, or None when it cannot be known."""
                if e not in _sa:
                    return None
                if e - 1 in _sa:
                    return _sa[e] - _sa[e - 1]
                # No predecessor: only epoch 1 of a FRESH segment starts from step 0. A resumed
                # segment's first epoch starts from the bundle's step, which is not recorded here.
                return _sa[e] if e == 1 else None

            for e, (s, n, k) in acc.items():
                # WEIGHTED BY POSE COUNT, never a mean of per-step means, and a partial epoch is
                # dropped rather than averaged -- an epoch's last step is a remainder.
                m.train_steps[e] = m.train_steps.get(e, 0) + k
                _exp = _expected(e)
                _whole = _exp is not None and k >= _exp
                if k >= need and n > 0 and (_whole or e < _last_ep or not _is_newest or _ended):
                    m.train_pae[e] = s / n
                    m.segment[e] = part
        p = os.path.join(d, "val_loss.csv")
        if os.path.exists(p):
            with open(p) as fh:
                for r in csv.DictReader(fh):
                    try:
                        m.val_pae[int(r["epoch"])] = (float(r["val_pae"]), int(r["n_poses"]))
                    except (ValueError, KeyError):
                        continue
        log = os.path.join(d, "run.log")
        if os.path.exists(log):
            txt = open(log, errors="replace").read()
            for mm in DOCKQ_LINE.finditer(txt):
                m.dockq[int(mm.group(1))] = (float(mm.group(2)), float(mm.group(3)), "run.log")
                m.segment.setdefault(int(mm.group(1)), part)
            for mm in TOOK_LINE.finditer(txt):
                m.timing.setdefault(int(mm.group(2)), {})[int(mm.group(1))] = (
                    float(mm.group(3)), mm.group(4))
            for mm in RESID_LINE.finditer(txt):
                m.resid.setdefault(int(mm.group(2)), {})[int(mm.group(1))] = (
                    float(mm.group(3)), float(mm.group(4)), float(mm.group(5)))
            for mm in MEM_LINE.finditer(txt):
                m.mem.setdefault(int(mm.group(2)), {})[int(mm.group(1))] = (
                    float(mm.group(3)), float(mm.group(4)), mm.group(5))
        for jp in sorted(glob.glob(os.path.join(d, "metric_train_ep*.json"))):
            try:
                j = json.load(open(jp))
                ep = int(re.search(r"ep(\d+)\.json$", jp).group(1))
            except (ValueError, OSError, AttributeError) as e_:
                m.rejected.append((jp, f"unreadable ({type(e_).__name__})"))
                continue
            jb = j.get("budget")
            if jb is None:
                m.rejected.append((jp, "no budget recorded"))
                continue
            if int(jb) != int(budget):
                # REJECTED, NOT WARNED. A different budget is a different pool depth, and taking
                # its number while the shipped-head column stays at this budget is exactly the
                # comparison the derived baseline exists to prevent.
                m.rejected.append((jp, f"budget {int(jb)} != {int(budget)}"))
                continue
            # A matching-budget json WINS over the log line: it is what an explicit re-score writes.
            m.dockq[ep] = (float(j["mean_dockq"]), float(j["frac_acc"]), os.path.basename(jp))
            m.segment.setdefault(ep, part)
    return m


def slowest_rank(m, e):
    """The rank with the longest wall clock, and ITS accounted/residual -- not independent maxima.

    Taking max() of wall, accounted and residual separately produced a row whose accounted +
    residual did not equal its wall time, under a heading that said "worst rank". Memory stays an
    explicit across-rank maximum because the question there is whether ANY rank came near the card.
    """
    if e not in m.timing:
        return None
    rk = max(m.timing[e], key=lambda r: m.timing[e][r][0])
    wall, buckets = m.timing[e][rk]
    acc = res = None
    if e in m.resid and rk in m.resid[e]:
        acc, _w, res = m.resid[e][rk]
    pa = pr = ck = None
    if e in m.mem and m.mem[e]:
        pa = max(v[0] for v in m.mem[e].values())
        pr = max(v[1] for v in m.mem[e].values())
        ck = sorted({v[2] for v in m.mem[e].values()})[0]
    return {"rank": rk, "wall": wall, "buckets": buckets, "accounted": acc, "residual": res,
            "peak_alloc": pa, "peak_reserved": pr, "ckpt": ck}


def write_atomic(path, text):
    """So a reader never sees a half-written report."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
