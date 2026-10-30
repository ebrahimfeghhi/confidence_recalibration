"""Count foreign processes occupying the trainer's cards. Prints the count; 0 means free.

    python -I run_logging/cards_busy.py [4,5,7]

READS GPU OCCUPANCY, NEVER COMMAND-LINE TEXT: the string a text check looks for is present in the
checking command itself, which is how a queued launch once refused against an idle box.

IT NOW TARGETS THE CARDS BY UUID, NOT BY WHERE THE HOLDERS HAPPEN TO BE. The previous version
derived the card set from the three holder pids:

    ours = {u for p, u in apps if p in HOLDERS}
    intruders = [(p, u) for p, u in apps if p not in HOLDERS and u in ours]

so if the holders were gone, `ours` was EMPTY, `intruders` was empty, and it printed 0 -- "cards
free" -- no matter what was actually running on them. A dead holder is precisely the condition
under which the cards become available to other users, i.e. exactly when a false "free" is most
dangerous, and the unattended chain gates a launch on this number. Found in review of fbf03447.

IT FAILS CLOSED. Any failure to establish occupancy -- nvidia-smi missing or erroring,
unparseable output, or a requested card index that does not exist -- prints a non-numeric token
and exits non-zero. Every caller tests `[ "$BUSY" = "0" ]`, so an unknown state reads as busy and
refuses the launch rather than proceeding on an assumption.

THE HOLDERS ARE STILL EXCLUDED, because they are ours and expected. The difference is that their
absence no longer hides anyone else.
"""
import subprocess
import sys

# The reservation processes. They are ours; never signal them, and do not count them as intruders.
# READ FROM training/runs/gpu_holder_pids, NOT HARDCODED. A holder restarted at a smaller --gib
# gets a NEW pid, and a holder pid this set does not know reads as an INTRUDER -- which makes this
# print non-zero, which makes every launch script and the watchdog refuse cards that are in fact
# ours. The literals below are the pids that ran from 2026-10-17, kept only as a fallback for a
# missing file.
def _holder_pids():
    import os
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "training", "runs", "gpu_holder_pids")
    try:
        with open(f) as fh:
            v = {int(x) for x in fh.read().split() if x.strip().isdigit()}
        if v:
            return v
    except Exception:
        pass
    return {306042, 306045, 306050}


HOLDERS = _holder_pids()
DEFAULT_CARDS = "4,5,7"


def _fail(msg):
    print(f"  cards_busy: {msg}", file=sys.stderr)
    # NOT A NUMBER, so `[ "$BUSY" = "0" ]` is false in every caller and the launch refuses.
    print("UNKNOWN")
    raise SystemExit(2)


def _smi(query):
    try:
        r = subprocess.run(["nvidia-smi", f"--query-{query[0]}={query[1]}",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        _fail(f"nvidia-smi failed ({e})")
    if r.returncode != 0:
        _fail(f"nvidia-smi exited {r.returncode}: {(r.stderr or '').strip()[:200]}")
    return r.stdout


def main():
    want = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CARDS)
    try:
        idx_want = [int(c.strip()) for c in want.split(",") if c.strip()]
    except ValueError:
        _fail(f"cannot parse card list {want!r}")
    if not idx_want:
        _fail("no cards requested")

    # index -> uuid for every GPU on the box, so the target set does not depend on any pid.
    uuid_of = {}
    for line in _smi(("gpu", "index,uuid")).strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            _fail(f"unparseable gpu line {line!r}")
        try:
            uuid_of[int(parts[0])] = parts[1]
        except ValueError:
            _fail(f"unparseable gpu index in {line!r}")
    missing = [i for i in idx_want if i not in uuid_of]
    if missing:
        _fail(f"card(s) {missing} do not exist on this box (have {sorted(uuid_of)})")
    target = {uuid_of[i] for i in idx_want}

    apps = []
    for line in _smi(("compute-apps", "pid,gpu_uuid")).strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            _fail(f"unparseable compute-app line {line!r}")
        try:
            apps.append((int(parts[0]), parts[1]))
        except ValueError:
            _fail(f"unparseable pid in {line!r}")

    intruders = [(p, u) for p, u in apps if p not in HOLDERS and u in target]
    for p, u in intruders:
        print(f"  occupying {u}: pid {p}", file=sys.stderr)
    print(len(intruders))
    return 0


if __name__ == "__main__":
    sys.exit(main())
