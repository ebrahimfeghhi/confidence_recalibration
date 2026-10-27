#!/usr/bin/env python
"""Watch a running arm for a VOIDED held-out PAE epoch and recover it automatically.

WHY THIS EXISTS SEPARATELY FROM THE TRAINER'S OWN RETRY. The retry added to heldout_pae() only
helps a run whose PARENT process started after the edit -- a live arm has the old function in
memory and will keep voiding epochs until it is relaunched. tv6hue is mid-flight and must not be
restarted, so the guarantee "no PAE value is ever missing" has to be enforced from outside the
process for the arm that is actually running. It is ALSO a belt-and-braces net for future arms:
if a shard fails more times than --val-loss-shard-retries allows, the trainer still voids and
this still recovers it.

IT NEVER SIGNALS ANYTHING. It reads run.log, reads val_loss.csv, and launches read-only
measurement subprocesses against bundles already written to disk. There is no code path here
that can stop, kill or restart a run.

IT WAITS FOR ROOM. The recovery shards peak at ~4.5 GB reserved each, but the training ranks
spike to ~67 GB of a 96 GB card, so a recovery launched at the wrong instant could push a card
into OOM and take the ARM down to fix a plot. So it requires real headroom on every target card
and simply waits otherwise -- a late recovered point is fine, a dead arm is not.
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_BIN = sys.executable
VOID = re.compile(r"epoch (\d+) has no held-out point")


def stamp():
    return subprocess.run(["date", "+%F %-I:%M:%S %p %Z"], capture_output=True, text=True,
                          env={**os.environ, "TZ": "America/New_York"}).stdout.strip()


def voided_epochs(rid):
    out = set()
    for f in [os.path.join(ROOT, "training", "runs", rid, "run.log")]:
        try:
            with open(f, errors="replace") as fh:
                for ln in fh:
                    m = VOID.search(ln)
                    if m:
                        out.add(int(m.group(1)))
        except OSError:
            pass
    return out


def recorded_epochs(rid):
    vp = os.path.join(ROOT, "training", "runs", rid, "val_loss.csv")
    out = set()
    try:
        with open(vp) as fh:
            for row in csv.DictReader(fh):
                try:
                    out.add(int(row["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def free_mib(cards):
    """Free MiB per card, or None if nvidia-smi cannot be trusted. FAILS CLOSED."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
    except Exception:                             # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    free = {}
    for ln in r.stdout.strip().splitlines():
        try:
            i, tot, used = (x.strip() for x in ln.split(","))
            free[i] = int(tot) - int(used)
        except ValueError:
            return None
    if any(c not in free for c in cards):
        return None
    return {c: free[c] for c in cards}


def live_rids(window_min):
    """Run ids whose run.log was written recently -- i.e. arms that are actually going.

    NOT a process scan. pgrep/pkill matches the shell running it and has self-killed this session
    before, and a /proc cmdline scan matches itself; a file mtime answers the only question that
    matters here (is this run still producing output?) without touching the process table.
    """
    out = []
    cutoff = time.time() - window_min * 60
    for f in glob.glob(os.path.join(ROOT, "training", "runs", "*", "run.log")):
        d = os.path.dirname(f)
        if os.path.islink(d):
            continue
        try:
            if os.path.getmtime(f) >= cutoff:
                out.append(os.path.basename(d))
        except OSError:
            continue
    return sorted(out)


def arm_alive(rid):
    """True while the run's log is still growing. No pid, no pattern match, no /proc scan."""
    f = os.path.join(ROOT, "training", "runs", rid, "run.log")
    try:
        a = os.path.getmtime(f)
    except OSError:
        return False
    time.sleep(90)
    try:
        return os.path.getmtime(f) != a or (time.time() - os.path.getmtime(f)) < 1800
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid", nargs="?", default=None,
                    help="one run id, or omit with --follow-live to watch whichever arm is "
                         "actually running")
    ap.add_argument("--cards", default="4,5,7")
    ap.add_argument("--need-free-mib", type=int, default=12000,
                    help="per-card headroom required before a recovery is launched. The shards "
                         "peak near 4.5 GB reserved; the margin is for the training rank's own "
                         "spike, which reaches ~67 GB of 96 GB.")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--follow-live", action="store_true",
                    help="watch whatever arm is CURRENTLY running instead of a fixed rid, so the "
                         "guarantee survives a chain handing off to the next arm while nobody is "
                         "at the keyboard. A run counts as live only while its run.log has been "
                         "written within --live-window-min; without that, this would wake up and "
                         "re-measure voided epochs belonging to runs that ended days ago.")
    ap.add_argument("--live-window-min", type=int, default=60)
    a = ap.parse_args()
    if not a.rid and not a.follow_live:
        ap.error("give a run id, or --follow-live")
    cards = [c.strip() for c in a.cards.split(",") if c.strip()]
    _who = a.rid if a.rid else f"whichever arm is live (last {a.live_window_min} min)"
    print(f"{stamp()} val PAE watchdog armed for {_who}; cards {cards}; "
          f"needs {a.need_free_mib} MiB free per card", flush=True)
    tried = {}
    while True:
      for rid in (live_rids(a.live_window_min) if a.follow_live else [a.rid]):
        missing = sorted(voided_epochs(rid) - recorded_epochs(rid))
        for ep in missing:
            key = (rid, ep)
            if tried.get(key, 0) >= 5:
                continue
            fr = free_mib(cards)
            if fr is None:
                print(f"{stamp()} cannot read GPU memory; NOT launching a recovery", flush=True)
                break
            if min(fr.values()) < a.need_free_mib:
                print(f"{stamp()} {rid} epoch {ep} needs recovery; waiting for room "
                      f"(free {fr}, need {a.need_free_mib})", flush=True)
                break
            tried[key] = tried.get(key, 0) + 1
            print(f"{stamp()} recovering {rid} epoch {ep} (attempt {tried[key]}), "
                  f"free {fr}", flush=True)
            r = subprocess.run([PY_BIN, "-I", os.path.join(ROOT, "run_logging",
                                                           "recover_val_pae.py"),
                                rid, str(ep)], capture_output=True, text=True,
                               cwd=ROOT, timeout=3600)
            print((r.stdout or "").strip(), flush=True)
            if r.returncode != 0:
                print(f"{stamp()} {rid} recovery of epoch {ep} FAILED "
                      f"rc={r.returncode}:\n"
                      + (r.stderr or "").strip()[-800:], flush=True)
            else:
                print(f"{stamp()} {rid} epoch {ep} recovered and written to "
                      f"val_loss.csv", flush=True)
      if a.once:
          return 0
      if not a.follow_live and not arm_alive(a.rid):
          still = sorted(voided_epochs(a.rid) - recorded_epochs(a.rid))
          print(f"{stamp()} {a.rid} looks finished; "
                + (f"STILL MISSING {still}" if still else "no PAE value is missing"), flush=True)
          return 0 if not still else 1
      time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
