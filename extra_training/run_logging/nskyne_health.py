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

# RANK LIVENESS COMES FROM /proc, NOT FROM GPU MEMORY. The memory test ("> 50 GB is a rank") is
# wrong at exactly the moment this script runs. It is invoked at an epoch boundary, where torch
# releases its cache, so all three ranks drop out of the rank band and INTO the holder band: on
# 2026-10-29 at the epoch 49/50 boundary it printed "ranks up: 0, holder procs seen: 6" and raised
# a false alarm while all three ranks were alive, mid-step, and writing trajectory.csv that same
# second. Two separate failures in one reading -- a false rank alarm, and three ranks silently
# counted as holders, which would have masked a holder actually dying.
#
# The ranks have fixed pids for the life of the run, and the user's standing instruction is to
# derive them explicitly rather than scan. So: liveness is /proc/<pid>, and nvidia-smi is used
# only to say how much memory each one currently holds.
RANK_PIDS = [l.strip() for l in
             open(os.path.join(ROOT, "training", "runs", RID + "_rank_pids")).read().split()
             ] if os.path.exists(
                 os.path.join(ROOT, "training", "runs", RID + "_rank_pids")) else []
out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
mib = {}
for l in out.splitlines():
    if not l.strip():
        continue
    pid, m = l.split(",")[0].strip(), float(l.split(",")[1])
    mib[pid] = max(mib.get(pid, 0.0), m)

if not RANK_PIDS:
    alarms.append(f"no training/runs/{RID}_rank_pids file -- cannot check rank liveness by pid")
    print("    ranks up: UNKNOWN (no rank-pid file)")
else:
    alive = [p for p in RANK_PIDS if os.path.isdir(f"/proc/{p}")]
    print(f"    ranks up: {len(alive)}/{len(RANK_PIDS)} by pid"
          + (f"   GPU MiB {[int(mib.get(p, 0)) for p in RANK_PIDS]}" if mib else ""))
    if len(alive) != len(RANK_PIDS):
        alarms.append("rank pid(s) GONE: "
                      + ",".join(p for p in RANK_PIDS if p not in alive))

# HOLDERS ARE PINNED BY PID, for the same reason the ranks are. "Every other GPU process over
# 500 MiB" counts other users' jobs as holders: on 2026-10-29 two foreign processes appeared and
# the count read 5 when there are 3. The alarm only fires below 3, so an inflated count can MASK a
# holder dying -- 3 holders minus one plus two strangers is still 4 -- which is the exact failure
# the earlier memory-band version had, in a new guise. A holder dying releases a card to the rest
# of the box, and it is the thing the user asked to be flagged, so it is checked by identity.
HOLDER_F = os.path.join(ROOT, "training", "runs", "gpu_holder_pids")
HOLDER_PIDS = [l.strip() for l in open(HOLDER_F).read().split() if l.strip()] \
    if os.path.exists(HOLDER_F) else []
if not HOLDER_PIDS:
    alarms.append(f"no {os.path.relpath(HOLDER_F, ROOT)} file -- cannot check the GPU holders "
                  f"by pid, which is the one thing the user asked to be flagged")
    print("    holders: UNKNOWN (no holder-pid file)")
else:
    h_alive = [p for p in HOLDER_PIDS if os.path.isdir(f"/proc/{p}")]
    foreign = {p for p, m in mib.items()
               if p not in set(RANK_PIDS) | set(HOLDER_PIDS) and m > 500}
    print(f"    holders up: {len(h_alive)}/{len(HOLDER_PIDS)} by pid"
          + (f"   ({len(foreign)} foreign GPU proc(s) on the box)" if foreign else ""))
    if len(h_alive) != len(HOLDER_PIDS):
        alarms.append("GPU HOLDER pid(s) GONE: "
                      + ",".join(p for p in HOLDER_PIDS if p not in h_alive))

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
