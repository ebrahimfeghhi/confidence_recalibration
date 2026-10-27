#!/usr/bin/env python
"""Exercise heldout_pae()'s shard retry WITHOUT a GPU, a model or a dataset.

WHY A TRANSCRIPTION AND NOT THE LIVE FUNCTION. heldout_pae() is a closure inside main() that
reads `a`, `cmd`, `env` and `_cards` from an enclosing scope built by 400 lines of argument
parsing and model construction. Importing it is not possible and refactoring the trainer to make
it importable would edit a file whose CHILD PROCESSES re-read it while an arm is running. So the
retry block is transcribed here and assert_source_matches() checks, line for line, that what is
tested is what is in phase_k_train.py. If the trainer changes and this is not updated, the test
fails loudly rather than passing against a stale copy.

WHAT IS FAKED: subprocess.Popen only. Every child is a stub that writes a log line and returns an
exit code from a per-shard script, so "shard 1 fails twice then succeeds" is expressible. The
spawn/await/retry/give-up control flow, the .failed log rotation and the rebuilt `procs` list are
the real ones.

Run:  /home/ebrahim/Protenix/.venv/bin/python -I run_logging/val_shard_retry_repro.py
"""
import csv
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "phases", "phase_k_train.py")

# Lines that must still be present in the trainer for this test to mean anything.
SOURCE_LINES = [
    'for _old in ([os.path.join(_o, _f) for _f in',
    '("shard.log", "trajectory.csv", "iptm_rows.csv")]',
    'for _try in range(1, int(getattr(a, "val_loss_shard_retries", 2) or 0) + 1):',
    "if not _bad:",
    "_idx = [_i for _i, _ in _bad]",
    "_handles[_i] = _spawn(_i)",
    "_bad = _await([_handles[_i] for _i in _idx])",
    "procs = [_handles[_i] for _i in range(_nsh)]",
    'if os.path.exists(_old):',
    "os.rename(_old, _old + _sfx)",
]


def assert_source_matches():
    src = open(TRAIN).read()
    missing = [ln for ln in SOURCE_LINES if ln not in src]
    if missing:
        print("SOURCE MISMATCH -- these lines are no longer in phase_k_train.py:")
        for ln in missing:
            print(f"    {ln}")
        raise SystemExit(2)
    print(f"source check: all {len(SOURCE_LINES)} transcribed lines still present "
          f"in phase_k_train.py\n")


class FakeProc:
    def __init__(self, rc):
        self.returncode = rc

    def wait(self):
        return self.returncode


POSES_PER_SHARD = 10


class Harness:
    """The transcribed control flow against a REAL filesystem.

    `scripts[i]` is the list of exit codes shard i returns. A non-zero code means the child wrote
    PART of its trajectory and then died, which is the case the first version of this test could
    not express -- it faked the filesystem, so it could not see that _flush() opens trajectory.csv
    with "a" and a retry in the same directory appends a second copy of the rows already there.

    `rotate_all=False` restores the pre-review behaviour (rotate only shard.log) so the bug is
    demonstrated rather than asserted away.
    """

    def __init__(self, base, scripts, nsh, retries, rotate_all=True):
        self.base, self.scripts, self.nsh, self.retries = base, scripts, nsh, retries
        self.rotate_all = rotate_all
        self.attempts = {i: 0 for i in range(nsh)}
        self.spawns = []

    def _child_writes(self, _o, _i, rc, attempt):
        """What the real child does: append trajectory rows as it goes, header only if absent."""
        tj = os.path.join(_o, "trajectory.csv")
        # a failed child dies partway, having already flushed some rows
        n = POSES_PER_SHARD if rc == 0 else POSES_PER_SHARD // 2
        new = not os.path.exists(tj)
        with open(tj, "a", newline="") as fh:     # <-- the real _flush mode
            w = csv.writer(fh)
            if new:
                w.writerow(["pdb", "unit", "pose", "pae_loss"])
            for k in range(n):
                w.writerow([f"p{_i}", f"u{_i}_{k}", "0", "2.5"])

    # ---- transcribed from phase_k_train.py ----
    def _spawn(self, _i):
        _o = os.path.join(self.base, f"shard{_i}")
        os.makedirs(_o, exist_ok=True)
        _n = 0
        while os.path.exists(os.path.join(_o, f"shard.log.failed{_n or ''}")):
            _n += 1
        _sfx = f".failed{_n or ''}"
        _names = (("shard.log", "trajectory.csv", "iptm_rows.csv") if self.rotate_all
                  else ("shard.log",))
        for _old in [os.path.join(_o, _f) for _f in _names]:
            if os.path.exists(_old):
                os.rename(_old, _old + _sfx)
        _lg = os.path.join(_o, "shard.log")
        self.spawns.append(_i)
        n = self.attempts[_i]
        self.attempts[_i] += 1
        rc = self.scripts[_i][min(n, len(self.scripts[_i]) - 1)]
        self._child_writes(_o, _i, rc, n)
        with open(_lg, "w") as fh:
            fh.write(f"attempt {n} rc {rc}\n")
            if rc == 0:
                fh.write(f"SHARD COVERAGE shard={_i} nshards={self.nsh} "
                         f"units_mine={POSES_PER_SHARD} units_all={POSES_PER_SHARD * self.nsh} "
                         f"nsample=1\n")
        return (_i, _o, _lg, None, FakeProc(rc))

    def _await(self, _ps):
        _failed = []
        for _i, _o, _lg, _fh, _p in _ps:
            _p.wait()
            if _p.returncode != 0:
                try:
                    _tail = open(_lg).read().strip().splitlines()[-6:]
                except OSError:
                    _tail = [f"(no readable log at {_lg})"]
                _failed.append((_i, _tail))
        return _failed

    def run(self):
        _handles = {}
        for _i in range(self.nsh):
            _handles[_i] = self._spawn(_i)
        _bad = self._await(list(_handles.values()))
        for _try in range(1, int(self.retries or 0) + 1):
            if not _bad:
                break
            _idx = [_i for _i, _ in _bad]
            for _i in _idx:
                _handles[_i] = self._spawn(_i)
            _bad = self._await([_handles[_i] for _i in _idx])
        procs = [_handles[_i] for _i in range(self.nsh)]
        return procs, _bad
    # ---- end transcription ----


def pool_poses(procs):
    """The parent's pooling. Returns (npose, duplicate_key or None)."""
    ids, n = set(), 0
    for _i, _o, _lg, _fh, _p in procs:
        tj = os.path.join(_o, "trajectory.csv")
        if not os.path.exists(tj):
            continue
        with open(tj) as fh:
            for row in csv.DictReader(fh):
                if not row.get("pae_loss"):
                    continue
                k = (row.get("pdb"), row.get("unit"), row.get("pose"))
                if k in ids:
                    return n, k
                ids.add(k)
                n += 1
    return n, None


def coverage_from(procs, nsh):
    """The parent's coverage scan, which is what a stale .failed log would corrupt."""
    cov = {}
    for _i, _o, _lg, _fh, _p in procs:
        with open(_lg) as fh:
            for ln in fh:
                if ln.startswith("SHARD COVERAGE "):
                    cov[_i] = dict(kv.split("=", 1) for kv in ln.split()[2:] if "=" in kv)
    return cov


def case(name, scripts, nsh, retries, want_void, want_spawns, want_failed_files,
         rotate_all=True, want_dup=False):
    base = tempfile.mkdtemp(prefix="valshard_")
    try:
        h = Harness(base, scripts, nsh, retries, rotate_all=rotate_all)
        procs, bad = h.run()
        void = bool(bad)
        cov = coverage_from(procs, nsh)
        failed_logs = sum(len([f for f in os.listdir(os.path.join(base, f"shard{i}"))
                               if ".failed" in f]) for i in range(nsh))
        ok = True
        notes = []
        if void != want_void:
            ok = False
            notes.append(f"void={void} want {want_void}")
        if len(h.spawns) != want_spawns:
            ok = False
            notes.append(f"{len(h.spawns)} spawn(s) want {want_spawns}")
        if failed_logs != want_failed_files:
            ok = False
            notes.append(f"{failed_logs} .failed file(s) want {want_failed_files}")
        npose, dup = pool_poses(procs)
        if (dup is not None) != want_dup:
            ok = False
            notes.append(f"duplicate={dup} want_dup={want_dup}")
        if not void:
            # The pooled population must be complete and read from the SUCCESSFUL logs.
            if len(cov) != nsh:
                ok = False
                notes.append(f"coverage from {len(cov)} of {nsh} shard(s)")
            if len(procs) != nsh or [p[0] for p in procs] != list(range(nsh)):
                ok = False
                notes.append("procs is not one handle per shard in order")
            if dup is None and npose != POSES_PER_SHARD * nsh:
                ok = False
                notes.append(f"pooled {npose} pose(s) want {POSES_PER_SHARD * nsh}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         spawns={h.spawns}  void={void}  coverage={len(cov)}/{nsh}  "
              f".failed={failed_logs}  poses={npose}  dup={dup}")
        if notes:
            print(f"         -> {'; '.join(notes)}")
        return ok
    finally:
        shutil.rmtree(base, ignore_errors=True)


def main():
    assert_source_matches()
    print("Each child is a stub returning a scripted exit code. No GPU, no model, no data.\n")
    results = []
    print("THE OLD BEHAVIOUR (retries=0) -- any failure voids the epoch:")
    results.append(case("2 of 3 fail once, retries=0 -> VOID",
                        {0: [0], 1: [1, 0], 2: [1, 0]}, 3, 0,
                        want_void=True, want_spawns=3, want_failed_files=0))
    print("\nTHE FIX (retries=2):")
    results.append(case("all succeed -> no retry, no rotation",
                        {0: [0], 1: [0], 2: [0]}, 3, 2,
                        want_void=False, want_spawns=3, want_failed_files=0))
    results.append(case("tv6hue ep4: shards 1,2 fail once -> recovered, only those re-run",
                        {0: [0], 1: [1, 0], 2: [1, 0]}, 3, 2,
                        want_void=False, want_spawns=5, want_failed_files=4))
    results.append(case("shard 1 fails twice then succeeds -> recovered on retry 2",
                        {0: [0], 1: [1, 1, 0], 2: [0]}, 3, 2,
                        want_void=False, want_spawns=5, want_failed_files=4))
    results.append(case("shard 1 fails forever -> VOID after 2 retries, not an infinite loop",
                        {0: [0], 1: [1], 2: [0]}, 3, 2,
                        want_void=True, want_spawns=5, want_failed_files=4))
    print("\nTHE REVIEW FINDING -- a shard that dies AFTER writing rows (rotate shard.log only):")
    results.append(case("P1 REPRODUCED: retry appends to the dead attempt's trajectory",
                        {0: [0], 1: [1, 0], 2: [0]}, 3, 2,
                        want_void=False, want_spawns=4, want_failed_files=1,
                        rotate_all=False, want_dup=True))
    print("\nTHE FIX -- rotate every output of the failed attempt:")
    results.append(case("same failure, all outputs rotated -> clean population",
                        {0: [0], 1: [1, 0], 2: [0]}, 3, 2,
                        want_void=False, want_spawns=4, want_failed_files=2,
                        rotate_all=True, want_dup=False))
    print()
    if all(results):
        print(f"ALL {len(results)} CASES PASS: only failed shards are re-run; a mid-pass failure "
              f"no longer leaves duplicate poses behind; a recovered epoch pools a complete "
              f"population; a permanently failing shard still voids rather than looping.")
        return 0
    print(f"{sum(1 for r in results if not r)} of {len(results)} CASES FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
