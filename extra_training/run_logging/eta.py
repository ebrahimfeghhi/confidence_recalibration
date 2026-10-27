#!/usr/bin/env python
"""Pace and ETA for a running arm, ALWAYS in Eastern.

WHY THIS IS A FILE AND NOT A ONE-LINER. I hand-rolled this calculation at several epochs on
2026-10-26 with datetime.fromtimestamp(), which formats in the HOST's timezone -- the box is PDT,
the user reads Eastern -- so every ETA I reported that night was three hours early and I only
noticed when an ETA of "3:31 AM" printed at 3:36 AM. The clock times taken from `TZ=... date`
were right; only the projections were wrong, which is exactly the kind of error that survives a
glance. Putting it in one place with the timezone fixed means it cannot be retyped incorrectly.
"""
import argparse
import datetime
import glob
import os
import sys
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid")
    ap.add_argument("--total", type=int, default=20, help="epochs the run will finish at")
    ap.add_argument("--window", type=int, default=4, help="how many recent epochs to average")
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(ROOT, "training", "runs", a.rid,
                                       "metric_train_ep*.json")), key=os.path.getmtime)
    if len(fs) < 2:
        print(f"{a.rid}: {len(fs)} evaluated epoch(s); need 2 to measure a pace")
        return 1
    eps = [int(os.path.basename(f).split("metric_train_ep")[1].split(".")[0]) for f in fs]
    ts = [os.path.getmtime(f) for f in fs]
    gaps = [(ts[i + 1] - ts[i]) / 60 for i in range(len(ts) - 1)]
    w = gaps[-a.window:]
    avg = sum(w) / len(w)
    now = datetime.datetime.now(EASTERN)
    last = max(eps)
    left = a.total - last
    eta = now + datetime.timedelta(minutes=left * avg)
    print(f"{a.rid}: epoch {last} of {a.total} done at "
          f"{datetime.datetime.fromtimestamp(ts[-1], EASTERN).strftime('%-I:%M %p')} Eastern")
    print(f"  {avg:.1f} min/epoch over the last {len(w)}  (all: "
          + " ".join(f"{g:.0f}" for g in gaps) + ")")
    if left > 0:
        print(f"  {left} epoch(s) left -> ends ~{eta.strftime('%-I:%M %p')} Eastern "
              f"({'today' if eta.date() == now.date() else eta.strftime('%a %-d %b')})")
    else:
        print("  finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
