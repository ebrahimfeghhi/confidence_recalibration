#!/usr/bin/env python
"""CPU-ONLY reproduction of the multi-worker prefetch stall that f39c83ea asked for.

    python -I run_logging/prefetch_deadlock_repro.py

NO GPUS, NO REAL DATA, NO CUDA. Every "crop" and every "GPU step" is a sleep, so this runs in
seconds and costs essentially no CPU. It exists because the throughput fix -- more than one crop
worker -- was reverted in 7033430b with the mechanism UNESTABLISHED, and f39c83ea said plainly
that the path stays off until there is a CPU-only reproduction rather than another live run.

WHAT HUNG. ggfhj5 with --prefetch-workers 4 stopped at step 3 of 15 in epoch 12: all three ranks
at 0% GPU and ~0% CPU, every thread in futex_wait_queue or poll, no trajectory.csv write for nine
minutes. Epoch 12 was lost and exists in neither segment of the run.

THE HYPOTHESIS THIS TESTS. _pf_take's docstring justifies its 120 s budget with "the worker emits
exactly one result per unit, IN THE MAIN LOOP'S OWN ORDER, so the awaited result always comes".
With W>1 that premise is false by construction -- _pf_start hands worker i the stride units[i::W]
and the help text says results arrive out of order -- which is what _pf_pending is for. But
_pf_pending EVICTS once it exceeds max(16, 8*W):

    if len(_pf_pending) > max(16, 8 * max(1, a.prefetch_workers)):
        _pf_pending.pop(next(iter(_pf_pending)))

and it pops in INSERTION order, which is the lowest-numbered stashed unit -- precisely the one the
main loop will ask for next. A worker emits each unit exactly once, so an evicted result never
returns to the queue. The next _pf_take for it cannot hit, cannot be served, and blocks for the
WHOLE budget before falling back to an inline recrop. Then the unit after it is also gone, and so
on: a cascade of full-budget waits. Four of them at 120 s is eight minutes of every rank sitting
in a futex with no trajectory write -- which is the observation, and is NOT a deadlock, which is
why looking for a deadlock did not find it.

WHY IT TAKES A STALL TO SHOW. With all workers healthy the strides advance together, results
interleave, and _pf_pending stays far below 32. It needs ONE unit to be slow -- a cold page, a
contended socket, a large structure -- so the main loop waits on that unit while the other workers
race ahead and overflow the buffer. That is why it appeared mid-epoch, once, and never in a smoke
test.

FIDELITY, AND ITS LIMIT. _pf_take, _pf_worker, _pf_start and _pf_end are closures inside
phase_k_train.py's main function and cannot be imported, so the logic below is a transcription.
That means this tests the transcription, not the shipped code -- so assert_source_matches() reads
the real file and fails loudly if the lines this depends on have changed. The honest fix, once the
mechanism is confirmed, is to lift these into a module the trainer imports and test that.

BUDGET IS SCALED DOWN, not changed in kind: 2 s here against 120 s in the trainer, so a full-budget
wait is visible in seconds. A stall is reported whenever a take costs the whole budget, since in
the real thing that same event costs two minutes.
"""
import os
import queue
import statistics as st
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = os.path.join(ROOT, "phases", "phase_k_train.py")

# The exact lines this reproduction stands on. If any of them is edited, the transcription below
# may no longer describe the shipped code, and a passing or failing result here would say nothing.
REQUIRED = [
    # the bug's shape, still present: out-of-order delivery into a bounded buffer that evicts
    "if len(_pf_pending) > max(16, 8 * max(1, a.prefetch_workers)):",
    "_pf_q = queue.Queue(maxsize=max(1, a.prefetch_units) * _w)",
    "target=_pf_worker, args=(units[i::_w],), daemon=True)",
    "def _pf_take(lab_p, budget_s=120.0):",
    "if lab_p in _pf_pending:",
    # the fix, which must stay: evictions recorded, and both immediate-abandon paths
    "_pf_dropped.add(_dk)",
    "if lab_p in _pf_dropped:",
    "not any(_t.is_alive() for _t in _pf_thread)",
]


def assert_source_matches():
    src = open(TRAINER).read()
    missing = [ln for ln in REQUIRED if ln not in src]
    if missing:
        print("STOP: phase_k_train.py no longer contains the lines this reproduction transcribes.")
        for ln in missing:
            print("  missing: " + ln)
        print("Re-read the prefetch block and update REQUIRED and the transcription together.")
        raise SystemExit(2)
    print(f"source check: all {len(REQUIRED)} transcribed lines still present in phase_k_train.py")


class Harness:
    """Transcription of the trainer's prefetch block, with sleeps where the real work is."""

    def __init__(self, n_units, workers, prefetch_units=2, budget_s=2.0,
                 crop_s=0.005, gpu_s=0.006, stall_unit=None, stall_s=3.0,
                 max_stalls=40, fixed=False):
        self.units = [f"unit{i:04d}" for i in range(n_units)]
        self.workers = workers
        self.prefetch_units = prefetch_units
        self.budget_s = budget_s
        self.crop_s = crop_s
        self.gpu_s = gpu_s
        self.stall_unit = stall_unit
        self.stall_s = stall_s
        # BAIL ONCE THE CASCADE IS ESTABLISHED. Every stalled take costs the whole budget, so
        # without this the harness runs n * budget -- 400 units at 2 s is 13 minutes, which IS the
        # bug rather than a test of it. Forty stalls is far more than enough to call it.
        self.max_stalls = max_stalls
        # fixed=True transcribes the SHIPPED FIX rather than the shipped bug: evictions are
        # recorded, and a take is abandoned at once when its unit is gone or when every worker has
        # finished. Both live here so one run shows the bug AND that the fix removes it.
        self.fixed = fixed
        self._pf_dropped = set()
        self.stalls = 0
        self.stopped_early = False
        self._pf_q = None
        self._pf_thread = None
        self.live_workers = -1
        self._pf_stop = threading.Event()
        self._pf_pending = {}
        self._pf_stat = {"hit": 0, "miss": 0, "wait_s": 0.0}
        self.takes = []            # (unit, seconds) for every take
        self.emitted = set()        # what the workers actually handed back, for diagnosis only

    # --- transcribed from _pf_worker: one result per unit, its own stride, stop flag honoured ---
    def _pf_worker(self, my_units):
        for u in my_units:
            if self._pf_stop.is_set():
                return
            time.sleep(self.stall_s if u == self.stall_unit else self.crop_s)
            try:
                self._pf_q.put((u, {"payload": u}), timeout=30)
                self.emitted.add(u)
            except queue.Full:
                return

    # --- transcribed from _pf_take, including the eviction ---
    def _pf_take(self, lab_p):
        t0 = time.perf_counter()
        if lab_p in self._pf_pending:
            self._pf_stat["hit"] += 1
            v = self._pf_pending.pop(lab_p)
            self.takes.append((lab_p, time.perf_counter() - t0))
            return v
        if self._pf_q is None:
            self.takes.append((lab_p, time.perf_counter() - t0))
            return None
        if self.fixed and lab_p in self._pf_dropped:
            self._pf_dropped.discard(lab_p)
            self._pf_stat["miss"] += 1
            self.takes.append((lab_p, time.perf_counter() - t0))
            return None
        deadline = t0 + self.budget_s
        while True:
            if (self.fixed and self._pf_thread is not None
                    and not any(t.is_alive() for t in self._pf_thread)
                    and self._pf_q.empty()):
                self._pf_stat["miss"] += 1
                self.takes.append((lab_p, time.perf_counter() - t0))
                return None
            try:
                k, v = self._pf_q.get(timeout=max(0.05, deadline - time.perf_counter()))
            except queue.Empty:
                self._pf_stat["miss"] += 1
                self._pf_stat["wait_s"] += time.perf_counter() - t0
                self.takes.append((lab_p, time.perf_counter() - t0))
                return None
            if k == lab_p:
                self._pf_stat["hit"] += 1
                self._pf_stat["wait_s"] += time.perf_counter() - t0
                self.takes.append((lab_p, time.perf_counter() - t0))
                return v
            self._pf_pending[k] = v
            if len(self._pf_pending) > max(16, 8 * max(1, self.workers)):
                dk = next(iter(self._pf_pending))
                self._pf_pending.pop(dk)
                if self.fixed:
                    self._pf_dropped.add(dk)

    def _pf_start(self):
        self._pf_stop.clear()
        self._pf_pending.clear()
        self._pf_dropped.clear()
        w = max(1, self.workers)
        self._pf_q = queue.Queue(maxsize=max(1, self.prefetch_units) * w)
        self._pf_thread = [threading.Thread(target=self._pf_worker,
                                            args=(self.units[i::w],), daemon=True)
                           for i in range(w)]
        for t in self._pf_thread:
            t.start()

    def _pf_end(self):
        if self._pf_thread is None:
            return
        self.live_workers = sum(1 for t in self._pf_thread if t.is_alive())
        self._pf_stop.set()
        for t in self._pf_thread:
            while True:
                try:
                    self._pf_q.get_nowait()
                except queue.Empty:
                    break
            t.join(timeout=5)
        self._pf_thread = None
        self._pf_q = None
        self._pf_pending.clear()

    def run(self):
        self._pf_start()
        t0 = time.perf_counter()
        for u in self.units:                 # THE MAIN LOOP ASKS IN ORDER. That is the contract.
            self._pf_take(u)
            if self.takes and self.takes[-1][1] >= self.budget_s * 0.95:
                self.stalls += 1
                if self.stalls >= self.max_stalls:
                    self.stopped_early = True
                    break
            time.sleep(self.gpu_s)           # the GPU step
        self._pf_end()
        return time.perf_counter() - t0


def report(label, h, wall):
    full = [(u, s) for u, s in h.takes if s >= h.budget_s * 0.95]
    worst = max((s for _, s in h.takes), default=0.0)
    med = st.median([s for _, s in h.takes]) if h.takes else 0.0
    print(f"\n{label}")
    print(f"  wall {wall:6.2f}s   takes {len(h.takes)}   hits {h._pf_stat['hit']}  "
          f"misses {h._pf_stat['miss']}")
    print(f"  take time: median {med*1000:7.2f} ms   worst {worst:6.3f}s   "
          f"budget {h.budget_s:.1f}s")
    print(f"  FULL-BUDGET STALLS: {len(full)}"
          + ("   (stopped early -- the cascade does not end on its own)" if h.stopped_early else ""))
    print(f"  worker threads still alive at the end: {h.live_workers} of {h.workers}")
    if full:
        shown = ", ".join(u for u, _ in full[:8])
        print(f"    first few: {shown}{' ...' if len(full) > 8 else ''}")
        gone = [u for u, _ in full if u in h.emitted]
        print(f"    of those, {len(gone)} had ALREADY been emitted by a worker "
              f"-- delivered, evicted, then waited for")
    return len(full)


def main():
    assert_source_matches()
    print("\nEach 'crop' and 'GPU step' is a sleep. No CUDA, no data, ~0 CPU.")
    print("THE RATIO IS WHAT MATTERS, not the absolute times. In the trainer the budget is 120 s")
    print("while a slow unit costs a few seconds -- the budget is ~24x the stall, so a slow unit")
    print("alone can NEVER exhaust it. The first run of this harness used budget 0.5 s against a")
    print("3 s stall, inverting that, and W=1 stalled purely because the budget expired waiting on")
    print("a slow unit -- a failure that cannot happen in production. It reported INCONCLUSIVE")
    print("rather than claiming a reproduction, which is the only reason that was caught.")
    print("Below: budget 0.5 s, stall 0.15 s -- budget 3.3x the stall, same side as the real one.\n")
    print("=" * 78)

    n = 400
    stall = "unit0003"

    h1 = Harness(n, workers=1, stall_unit=stall, budget_s=0.5, stall_s=0.15)
    w1 = h1.run()
    s1 = report("W=1  (the shipped default)", h1, w1)

    h4 = Harness(n, workers=4, stall_unit=stall, budget_s=0.5, stall_s=0.15)
    w4 = h4.run()
    s4 = report("W=4  BEFORE the fix (the reverted setting)", h4, w4)

    hf = Harness(n, workers=4, stall_unit=stall, budget_s=0.5, stall_s=0.15, fixed=True)
    wf = hf.run()
    sf = report("W=4  WITH the fix (what the trainer now does)", hf, wf)

    print("\n" + "=" * 78)
    if s4 > 0 and s1 == 0 and sf == 0:
        print("REPRODUCED. W=1 never stalls; W=4 stalls on units a worker had already emitted.")
        print("In the trainer each of those costs 120 s, and the ranks drift apart by that much,")
        print("so they reach the gradient collective at different times and sit in futex_wait")
        print("with --dist-timeout-min 240 keeping anything from erroring out. That is ggfhj5's")
        print("nine minutes with no trajectory write.")
        print("\nAND THE FIX REMOVES IT: 0 stalls at W=4 once evictions are recorded and a take")
        print("is abandoned at once when its unit is gone or every worker has finished. The")
        print("budget goes back to meaning only what it was written to mean: the worker is dead.")
        print("THIS IS A REGRESSION TEST -- the fixed path must stay at 0 stalls.")
        return 0
    if sf > 0:
        print(f"THE FIX DID NOT HOLD: {sf} stalls remain at W=4 with it applied. Do not raise")
        print("--prefetch-workers. Read the take path again before anything else.")
        return 1
    if s4 == 0 and s1 == 0:
        print("NOT REPRODUCED. Neither setting stalled, so eviction is not the mechanism on this")
        print("shape. Widen the stall or the unit count before concluding -- and if it still does")
        print("not reproduce, the multi-worker path stays off and the crop cache is the route.")
        return 1
    print("INCONCLUSIVE: W=1 stalled too. With one worker there is no out-of-order delivery, so")
    print("eviction cannot be the cause there -- the harness is mis-scaled, not the trainer.")
    print("Check that budget_s is comfortably LARGER than stall_s, as it is in production.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
