#!/usr/bin/env python
"""One-line health check for the live arm plus the standing box flags. No GPU, no model.

FLAGS THE USER ASKED FOR EXPLICITLY: a GPU holder gone, MemAvailable under 250 GB, /home over 90%.
A holder dying releases a card to the rest of the box; MemAvailable is other users' headroom.
"""
import csv
import os
import shutil
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RID = sys.argv[1] if len(sys.argv) > 1 else "nskyne"
EP_POSES = 5520
alarms = []

traj = os.path.join(ROOT, "training", "runs", RID, "trajectory.csv")
per = defaultdict(int)
wall = defaultdict(list)
for r in csv.DictReader(open(traj)):
    try:
        e = int(float(r["epoch"])); per[e] += 1; wall[e].append(float(r["wall_s"]))
    except (ValueError, KeyError, TypeError):
        continue
done = sorted(e for e in per if per[e] >= EP_POSES)
cur = max(per) if per else 0
print(f"  {RID}: {len(done)} complete epoch(s), {per[cur]}/{EP_POSES} into epoch {cur}")
if done:
    d = [max(wall[e]) - min(wall[e]) for e in done]
    print(f"    median epoch {sorted(d)[len(d)//2]/60:.1f} min")

# the three training ranks and the three holders, by memory, never by pattern
out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
ranks = {l.split(",")[0].strip() for l in out.splitlines()
         if l.strip() and float(l.split(",")[1]) > 50000}
holders = {l.split(",")[0].strip() for l in out.splitlines()
           if l.strip() and 500 < float(l.split(",")[1]) <= 50000}
print(f"    ranks up: {len(ranks)}   holder procs seen: {len(holders)}")
if len(ranks) != 3:
    alarms.append(f"expected 3 training ranks, see {len(ranks)}")
if len(holders) < 3:
    alarms.append(f"a GPU HOLDER may be gone: only {len(holders)} holder proc(s) visible")

mem = {}
for line in open("/proc/meminfo"):
    k, v = line.split(":")
    mem[k] = int(v.split()[0])
avail_gb = mem["MemAvailable"] / 1048576
print(f"    MemAvailable {avail_gb:.0f} GB")
if avail_gb < 250:
    alarms.append(f"MemAvailable {avail_gb:.0f} GB is under the 250 GB floor")

du = shutil.disk_usage("/home")
pct = 100 * du.used / du.total
print(f"    /home {pct:.0f}% used ({du.free/2**30:.0f} GB free)")
if pct > 90:
    alarms.append(f"/home is {pct:.0f}% full")

print("\n  " + ("OK" if not alarms else "ALARM: " + "; ".join(alarms)))
sys.exit(1 if alarms else 0)
