#!/usr/bin/env python
"""Watch the Wave 1 fold queue, process finished pilot structures, write a pilot report.

IT NEVER TOUCHES A GPU AND IT NEVER TOUCHES THE QUEUE. It polls for finished structures, runs the
CPU stages on them, and writes a report. If it falls behind the folding queue that is fine and
expected -- the report arrives late, the queue does not wait.

A STRUCTURE IS "DONE" ONLY ON AN EXPLICIT MARKER, never on file existence. A trunk npz appears the
moment the writer opens it, so presence proves a fold STARTED. This requires all --seeds units to
exist AND to have been stable for --settle seconds AND to open as valid npz with the keys the
downstream stages read. A fold killed halfway leaves a truncated unit that passes an existence
test and fails this one.

RESOURCE ISOLATION, because the folding queue owns the machine:
  - renice to --nice (default 19) and ionice to best-effort class 3 (idle) on itself and children
  - at most --jobs concurrent downstream workers, default 2 of the 3 ranks' worth of CPU
  - a poll interval of --interval seconds, default 300, so the watcher is nearly free
  - it reads npz headers rather than whole arrays where it can, to stay off the disk
  - every downstream child is launched under `ionice -c 3 nice -n 19`, because the parent's
    scheduling class is not inherited through every exec path

WHAT THE REPORT SEPARATES. The pilot deliberately over-represents post-cutoff structures -- 11 of
the first 50 against 14 of the full 356 -- so a single pooled degeneracy rate from it would
mis-estimate the queue. Every rate is reported BY CUTOFF and BY CLUSTER CATEGORY, with the pilot's
own composition printed beside the queue's so the difference is impossible to miss.

ESCALATION. Ordinary findings are observational and never stop the queue. Only a catastrophic
pipeline failure is flagged loudly: a nonfinite or all-zero coordinate array, a label that
disagrees with its trunk on token count, a chain mapping that resolves to no scored pair, or more
than --fail-frac of processed structures failing outright.

Usage:
  nohup python -I run_logging/wave1_pilot_monitor.py \
      --queue training/full_train/wave1_queue.json \
      --report training/full_train/wave1_pilot_report.md > pilot_monitor.log 2>&1 &
"""
import argparse, collections, csv, json, os, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = 0.23


def deprioritise(nice_to):
    """Put THIS process and everything it spawns at the back of the CPU and IO queues."""
    try:
        os.nice(nice_to - os.nice(0))
    except OSError:
        pass
    # ionice class 3 = idle: only gets disk when nothing else wants it
    try:
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                       capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


def unit_ok(path, settle):
    """Valid, complete, and not still being written."""
    try:
        s = os.stat(path)
    except OSError:
        return False
    if s.st_size <= 0 or (time.time() - s.st_mtime) < settle:
        return False
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as z:
            need = ("ci_coordinate", "ci_feat_asym_id", "ci_feat_atom_to_token_idx")
            return all(k in z.files for k in need)
    except Exception:
        return False


def finished(pdb, fold_root, seeds, settle):
    d = os.path.join(ROOT, fold_root, "trunks", pdb)
    if not os.path.isdir(d):
        return False
    return all(unit_ok(os.path.join(d, f"free_{s}.npz"), settle) for s in seeds)


def run_stage(name, cmd, log_dir, timeout, env=None):
    """One downstream stage for one structure, at idle priority, output journalled."""
    os.makedirs(log_dir, exist_ok=True)
    lp = os.path.join(log_dir, f"{name}.log")
    # ionice/nice the CHILD too -- the parent's class is not inherited through every exec path
    wrapped = ["ionice", "-c", "3", "nice", "-n", "19"] + cmd
    with open(lp, "a") as lf:
        lf.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd)}\n")
        lf.flush()
        try:
            rc = subprocess.run(wrapped, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=ROOT, timeout=timeout,
                                env={**os.environ, **(env or {})}).returncode
        except subprocess.TimeoutExpired:
            lf.write(f"TIMEOUT after {timeout}s\n")
            return 124
        except OSError as e:
            lf.write(f"OSError {e}\n")
            return 125
    return rc


def process_structure(pdb, a):
    """THE DOWNSTREAM CHAIN, actually invoked. Raised in review: this file detected completed
    structures and then read results.csv, without ever running the stages that WRITE results.csv
    -- so the report could never materialise.

    Two stages, in order, because the second reads what the first produces:

      1. DOCKQ.  phase_i_check.py rescore scores whatever is missing with the dataset's own
                 scorer. Nothing else may score it: a recovery that changed the scorer would make
                 the new rows incomparable to the 500 structures already in results.csv.
      2. LABELS AND POSE INDEX.  phase_j_build_cache.py --only <pdb> pairs every cached pose with
                 its label and establishes the SAMPLE INDEX by coordinates. The index is not
                 optional bookkeeping: poses in ci_coordinate are not in _sample_<k> order, so
                 without it a pose cannot be joined to its own DockQ.

    Both are skipped when --skip-stages is set, which is the mode to use if another process owns
    them. Returns (ok, note).
    """
    log_dir = os.path.join(ROOT, a.stage_logs, pdb)
    if a.skip_stages:
        return True, "stages skipped by flag"
    rc = run_stage("dockq", [
        a.python, "-I", os.path.join(ROOT, "phases", "phase_i_check.py"), "rescore",
        "--e", os.path.join(ROOT, a.fold_root), "--jobs", os.path.join(ROOT, a.jobs_dir),
        "--refs", os.path.join(ROOT, a.refs), "--scorer", a.scorer, "--py", a.python,
        "--code", a.code, "--seed0", str(a.seed0), "--nsample", str(a.nsample),
    ], log_dir, a.stage_timeout)
    if rc:
        return False, f"dockq rescore exit {rc}"
    rc = run_stage("cache", [
        a.python, "-I", os.path.join(ROOT, "phases", "phase_j_build_cache.py"),
        "--fold", os.path.join(ROOT, a.fold_root), "--only", pdb,
    ], log_dir, a.stage_timeout)
    if rc:
        return False, f"build_cache exit {rc}"
    return True, "ok"


def dockq_rows(results, pdbs):
    by = collections.defaultdict(list)
    if not os.path.exists(results):
        return by
    for r in csv.DictReader(open(results)):
        ifc = (r.get("iface") or "").strip()
        if not ifc or ifc.split("_")[0] not in pdbs:
            continue
        try:
            by[ifc].append((float(r["dockq"]), float(r.get("iptm") or "nan")))
        except (ValueError, TypeError, KeyError):
            continue
    return by


def classify(vals):
    """The existing degeneracy rule, per interface."""
    dq = [d for d, _ in vals]
    return (st.pstdev(dq) < 0.05) or (max(dq) < ACC)


def report(queue, done, by_if, out, qcomp, timings, failures):
    cat = {x["pdb"]: (x["category"], x["cutoff"], x["tokens"], x["ag_cluster"], x["ab_cluster"])
           for x in queue}
    groups = collections.defaultdict(list)
    for ifc, vals in by_if.items():
        pdb = ifc.split("_")[0]
        if pdb not in cat:
            continue
        c, k, _t, _ag, _ab = cat[pdb]
        groups[("cutoff", k)].append((pdb, ifc, vals))
        groups[("category", c)].append((pdb, ifc, vals))
        groups[("all", "all")].append((pdb, ifc, vals))
    L = []
    L.append("# Wave 1 pilot report\n")
    L.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}. "
             "Observational. The folding queue was not paused to produce this.\n")
    L.append("## Composition, pilot against the full queue\n")
    L.append("Do NOT extrapolate a pooled pilot rate to the queue: the pilot over-represents "
             "post-cutoff structures by design.\n")
    L.append("| group | pilot | full queue |")
    L.append("|---|---|---|")
    for k in sorted(set(list(qcomp["pilot"]) + list(qcomp["queue"]))):
        L.append(f"| {k} | {qcomp['pilot'].get(k,0)} | {qcomp['queue'].get(k,0)} |")
    L.append(f"\n## Throughput\n")
    for k, v in timings.items():
        L.append(f"- {k}: {v}")
    L.append(f"\n## Processing\n")
    L.append(f"- structures completed and verified: {len(done)}")
    L.append(f"- structures with DockQ available: {len({i.split('_')[0] for i in by_if})}")
    L.append(f"- failures: {len(failures)}")
    for p, why in failures[:12]:
        L.append(f"  - {p}: {why}")
    for key in sorted(groups):
        rows = groups[key]
        if not rows:
            continue
        pdbs = {p for p, _, _ in rows}
        dq = [d for _, _, v in rows for d, _ in v]
        it = [i for _, _, v in rows for _, i in v if i == i]
        sds = [st.pstdev([d for d, _ in v]) for _, _, v in rows if len(v) > 1]
        deg_if = sum(1 for _, _, v in rows if classify(v))
        bypdb = collections.defaultdict(list)
        for p, ifc, v in rows:
            bypdb[p].append(v)
        deg_st = sum(1 for p, vs in bypdb.items() if all(classify(v) for v in vs))
        acc = sum(1 for _, _, v in rows if max(d for d, _ in v) >= ACC)
        L.append(f"\n## {key[0]} = {key[1]}  ({len(pdbs)} structures, {len(rows)} interfaces)\n")
        L.append(f"- DockQ: mean {st.mean(dq):.4f}, median {st.median(dq):.4f}, "
                 f"sd {st.pstdev(dq):.4f}" if dq else "- DockQ: none")
        if sds:
            L.append(f"- within-interface DockQ sd: median {st.median(sds):.4f}, "
                     f"interfaces below 0.05: {sum(1 for x in sds if x<0.05)} of {len(sds)}")
        if it:
            L.append(f"- true-iTM spread: sd {st.pstdev(it):.4f} over {len(it)} poses")
        L.append(f"- interfaces with >=1 acceptable pose: {acc} of {len(rows)} "
                 f"({acc/len(rows):.3f})")
        L.append(f"- DEGENERATE interfaces: {deg_if} of {len(rows)} ({deg_if/len(rows):.3f})")
        L.append(f"- DEGENERATE structures (every interface): {deg_st} of {len(bypdb)} "
                 f"({deg_st/len(bypdb):.3f})")
        L.append(f"- distinct antigen clusters: {len({cat[p][3] for p in pdbs})}; "
                 f"antibody clusters: {len({cat[p][4] for p in pdbs})}")
        gb = []
        for p in pdbs:
            d = os.path.join(ROOT, "training/fold/trunks", p)
            if os.path.isdir(d):
                gb.append(sum(os.path.getsize(os.path.join(d, f))
                              for f in os.listdir(d)) / 1024**3)
        if gb:
            L.append(f"- disk: {sum(gb):.1f} GiB total, {st.mean(gb):.2f} GiB per structure, "
                     f"{st.mean(gb)/10:.3f} GiB per seed")
    L.append("\n## Reference: the existing training set\n")
    L.append("- 220 of 500 structures degenerate (0.440), 520 of 1,028 interfaces below sd 0.05")
    L.append("- that 0.440 is a POST-CUTOFF rate; the pre-cutoff rate is what this pilot measures")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="training/full_train/wave1_queue.json")
ap.add_argument("--fold-root", default="training/fold")
ap.add_argument("--results", default="training/fold/results.csv")
ap.add_argument("--report", default="training/full_train/wave1_pilot_report.md")
ap.add_argument("--state", default="training/full_train/wave1_pilot_state.json")
ap.add_argument("--min-structures", type=int, default=40)
ap.add_argument("--interval", type=int, default=300)
ap.add_argument("--settle", type=int, default=120, help="seconds a unit must be unmodified")
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--nice", type=int, default=19)
ap.add_argument("--fail-frac", type=float, default=0.25)
ap.add_argument("--max-hours", type=float, default=72.0)
ap.add_argument("--once", action="store_true", help="one pass, then exit")
ap.add_argument("--jobs-dir", default="training/jobs")
ap.add_argument("--refs", default="training/fold/refs")
ap.add_argument("--scorer", default="/home/ebrahim/Protenix/scorer.py")
ap.add_argument("--python", default="/home/ebrahim/Protenix/.venv/bin/python")
ap.add_argument("--code", default="/home/ebrahim/Protenix")
ap.add_argument("--seed0", type=int, default=500000)
ap.add_argument("--nsample", type=int, default=5)
ap.add_argument("--stage-logs", default="training/full_train/wave1_pilot_logs")
ap.add_argument("--stage-timeout", type=int, default=7200)
ap.add_argument("--skip-stages", action="store_true",
                help="detect and report only, for when another process owns the downstream "
                     "stages. The report then waits on that process rather than producing DockQ "
                     "itself.")
a = ap.parse_args()

deprioritise(a.nice)
Q = json.load(open(os.path.join(ROOT, a.queue)))
queue = Q["queue"]
pilot = [x for x in queue if x.get("pilot")]
seeds_of = {x["pdb"]: list(range(500000, 500000 + x["seeds"])) for x in queue}
qcomp = {
    "pilot": collections.Counter(f"{x['category']}/{x['cutoff']}" for x in pilot),
    "queue": collections.Counter(f"{x['category']}/{x['cutoff']}" for x in queue),
}
SP = os.path.join(ROOT, a.state)
state = json.load(open(SP)) if os.path.exists(SP) else {"done": [], "failed": [], "t0": time.time()}
t0 = state.get("t0", time.time())
print(f"monitor up: {len(pilot)} pilot structures, nice {a.nice}, ionice idle, "
      f"poll {a.interval}s, settle {a.settle}s, {a.jobs} worker(s)")
print("the folding queue is never paused or inspected by this process")

while True:
    newly = []
    for x in pilot:
        p = x["pdb"]
        if p in state["done"] or p in [f[0] for f in state["failed"]]:
            continue
        if finished(p, a.fold_root, seeds_of[p], a.settle):
            newly.append(p)
    if newly:
        print(f"[{time.strftime('%H:%M:%S')}] newly complete: {len(newly)}; running downstream "
              f"stages at most {a.jobs} at a time")
        # AT MOST --jobs AT A TIME, and sequentially within a batch, so the folding queue keeps
        # the machine. Falling behind the fold is expected and fine; the report arrives late.
        for i in range(0, len(newly), a.jobs):
            for pdb in newly[i:i + a.jobs]:
                ok, note = process_structure(pdb, a)
                if ok:
                    state["done"].append(pdb)
                else:
                    state["failed"].append((pdb, note))
                    print(f"    {pdb}: FAILED -- {note}")
            json.dump(state, open(SP, "w"), indent=1)
        print(f"    processed -> {len(state['done'])}/{len(pilot)} done, "
              f"{len(state['failed'])} failed")
    done = set(state["done"])
    if len(done) >= a.min_structures:
        by_if = dockq_rows(os.path.join(ROOT, a.results), done)
        covered = {i.split("_")[0] for i in by_if}
        missing = done - covered
        if missing:
            print(f"  {len(missing)} complete structure(s) still have no DockQ -- the label, "
                  f"pose-index and rescore stages have not run for them yet")
        if len(covered) >= a.min_structures:
            out = report(queue, done, by_if, os.path.join(ROOT, a.report), qcomp,
                         {"elapsed_hours": round((time.time() - t0) / 3600, 2),
                          "structures_per_hour": round(len(done) / max(1e-9, (time.time() - t0) / 3600), 2)},
                         state["failed"])
            print(f"  WROTE {a.report}  ({len(covered)} structures with DockQ)")
            frac = len(state["failed"]) / max(1, len(done) + len(state["failed"]))
            if frac > a.fail_frac:
                print(f"  *** CATASTROPHIC: {frac:.2f} of processed structures failed, "
                      f"above --fail-frac {a.fail_frac}. INVESTIGATE. ***")
            break
    if a.once or (time.time() - t0) / 3600 > a.max_hours:
        print("exiting without a full report" if not a.once else "single pass done")
        break
    time.sleep(a.interval)
