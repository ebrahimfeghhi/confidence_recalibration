#!/usr/bin/env python
"""Back-fill spread.r<rank>.csv for a run whose trainer predates the csv writer.

    python -I run_logging/spread_from_log.py <rid> [--loop]

WHY. The within-interface spread -- median std(predicted) against std(true iTM) and their ratio --
has been printed at every epoch boundary for a long time and was only committed to a csv in
7730c333. A run already in flight keeps the old code for its whole life, so the quantity exists
only in run.log for it. This parses those lines into exactly the schema the trainer now writes, so
the two sources plot with one reader.

IT REFUSES TO OVERWRITE A TRAINER-WRITTEN FILE. Back-filled files carry a '# derived from run.log'
first line; a spread csv without that marker was written by the trainer itself, is authoritative,
and is left alone. Otherwise a stray invocation on a new run would silently replace measured rows
with rows re-parsed from a log format that is free to change.

IT REWRITES RATHER THAN APPENDS, which makes it idempotent and safe to loop: the log is the whole
truth, so a second pass over a longer log is a superset of the first.
"""
import argparse
import os
import re
import sys
import time

MARK = "# derived from run.log\n"
HEAD = "epoch,n_ifaces,rho,sd_pred,sd_true,sd_ratio\n"
RHO = re.compile(r"\[rank (\d+)\] epoch (\d+) WITHIN-INTERFACE SPEARMAN[^:]*: median "
                 r"([-+0-9.]+) over (\d+) interface-group")
SPR = re.compile(r"\[rank (\d+)\] epoch (\d+) SCORE SPREAD: median std\(predicted\) "
                 r"([0-9.]+) against std\(true iTM\) ([0-9.]+), median ratio ([0-9.]+)")

ap = argparse.ArgumentParser()
ap.add_argument("rid")
ap.add_argument("--runs", default="training/runs")
ap.add_argument("--loop", action="store_true", help="keep parsing every --every seconds")
ap.add_argument("--every", type=float, default=300.0)
a = ap.parse_args()

D = os.path.join(a.runs, a.rid)
LOG = os.path.join(D, "run.log")
if not os.path.isdir(D):
    sys.exit(f"STOP: no such run directory: {D}")
if not os.path.exists(LOG):
    sys.exit(f"STOP: {LOG} does not exist.")

for _r in range(4):
    _p = os.path.join(D, f"spread.r{_r}.csv")
    if os.path.exists(_p):
        with open(_p) as fh:
            if fh.readline() != MARK:
                sys.exit(f"STOP: {_p} was written by the trainer, not derived from the log.\n"
                         f"  It is the authoritative record and this script will not replace it.")


def once():
    by = {}      # rank -> epoch -> dict
    with open(LOG, errors="replace") as fh:
        for ln in fh:
            m = RHO.search(ln)
            if m:
                by.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), {}).update(
                    rho=float(m.group(3)), n=int(m.group(4)))
                continue
            m = SPR.search(ln)
            if m:
                by.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), {}).update(
                    sd_s=float(m.group(3)), sd_g=float(m.group(4)), ratio=float(m.group(5)))
    wrote = []
    for rk in sorted(by):
        rows = []
        for ep in sorted(by[rk]):
            d = by[rk][ep]
            # BOTH LINES OR NEITHER. A half-written epoch -- the rho line flushed and the spread
            # line not yet -- would otherwise land as a row of nans that looks like a measurement.
            if not {"rho", "sd_s"} <= set(d):
                continue
            rows.append("%d,%d,%.6f,%.6f,%.6f,%.6f\n"
                        % (ep, d["n"], d["rho"], d["sd_s"], d["sd_g"], d["ratio"]))
        if not rows:
            continue
        p = os.path.join(D, f"spread.r{rk}.csv")
        with open(p, "w") as fh:
            fh.write(MARK)
            fh.write(HEAD)
            fh.writelines(rows)
        wrote.append((p, len(rows)))
    return wrote


while True:
    w = once()
    stamp = time.strftime("%-I:%M %p", time.localtime())
    if not w:
        print(f"[{stamp}] no complete epoch yet in {LOG}")
    else:
        for p, n in w:
            print(f"[{stamp}] {p}: {n} epoch row(s)")
    if not a.loop:
        break
    sys.stdout.flush()
    time.sleep(a.every)
