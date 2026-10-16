"""THE ONE READER for a training run's records. Everything else here reads through this.

WHAT THE RECORDS ARE. The trainer already writes everything a report needs, and this module does not
add instrumentation to it -- deliberately. Coupling the trainer to a viewer (a SummaryWriter call, a
wandb handle) means a restart whenever the viewer changes, and we have spent four restarts on this
pilot already. The trainer writes plain files; readers come and go.

  <out>/trajectory.csv    one row per POSE, flushed every 20 rows. `step_taken` marks the pose on
                          which the optimiser stepped, so the per-batch series is the subset where
                          step_taken == 1. Carries the four raw loss terms, the weighted total,
                          grad_norm (PRE-clip, as clip_grad_norm_ returns it), lr, epoch, and the
                          structure and unit the pose came from.
  <out>/metric_ep<N>.json the validation metric after epoch N: mean_dockq, frac_acc, their oracle
                          counterparts, headroom, and which checkpoint was scored.
  <out>/args.json         every argument the run was launched with, so a curve can be read against
                          the settings that produced it.

THE BASELINE IS THE EVALUATOR'S OWN NUMBER. training/ft3_pilot_baseline.json holds the shipped head
scored by the same evaluator, which is what a trained head has to beat: 0.5775 mean DockQ, not the
0.5779 that comes from results.csv. results.csv stores ipTM rounded to 4 decimals, so near-ties are
broken differently there; measured, that moves the selection on 5 of 92 interfaces. Comparing a
recomputed metric against a rounded one would bake in a 0.0003 offset that has nothing to do with
the head.
"""
import csv, glob, json, os

TERMS = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def read_run(out_dir, baseline_path=""):
    """Everything about one run, as plain dicts. Missing pieces come back empty, not raised.

    A report is written WHILE the run is going, so absence is the normal case and not an error: no
    trajectory yet before the first flush, no metric_ep1.json before the first epoch boundary. The
    renderer shows what exists and says what is pending.
    """
    out = {"out_dir": out_dir, "args": {}, "poses": [], "steps": [], "epochs": [],
           "baseline": {}, "config": {}, "status": {}, "warnings": []}

    # config.json and status.json come from run_logging/launch.py: the settings, the commit the run
    # was launched from and whether the tree was dirty, then how it ended. A curve whose commit is
    # unknown cannot be compared with another, which is why they are read here rather than left as
    # files someone might open.
    for name, key in (("config.json", "config"), ("status.json", "status")):
        p_ = os.path.join(out_dir, name)
        if os.path.exists(p_):
            try:
                out[key] = json.load(open(p_))
            except (json.JSONDecodeError, OSError) as e:
                out["warnings"].append(f"{name} unreadable: {e}")

    ap = os.path.join(out_dir, "args.json")
    if os.path.exists(ap):
        try:
            out["args"] = json.load(open(ap))
        except (json.JSONDecodeError, OSError) as e:
            out["warnings"].append(f"args.json unreadable: {e}")

    tp = os.path.join(out_dir, "trajectory.csv")
    if os.path.exists(tp):
        try:
            with open(tp) as fh:
                for r in csv.DictReader(fh):
                    raw = sum(_num(r.get(k), 0.0) for k in TERMS)
                    rec = {"wall_s": _num(r.get("wall_s"), 0.0),
                           "step": int(_num(r.get("step"), 0)),
                           "epoch": int(_num(r.get("epoch"), 0)),
                           "pdb": r.get("pdb", ""),
                           "lr": _num(r.get("lr")),
                           "loss": _num(r.get("loss")),
                           "raw": raw,
                           "grad_norm": _num(r.get("grad_norm")),
                           "step_taken": r.get("step_taken") == "1",
                           "nonfinite": r.get("nonfinite_grad") == "1"}
                    for k in TERMS:
                        rec[k] = _num(r.get(k))
                    # THE EXACT BATCH NUMBERS, when the trainer wrote them: the mean of each term
                    # over all accum_eff poses in the step, reduced across every rank. Runs before
                    # 2026-10-16 have no such columns and fall back to rank-0 window means, which
                    # is the best that can be reconstructed -- the other ranks' poses were never
                    # written. `batch_n` distinguishes the two: None means reconstructed.
                    for k in TERMS:
                        rec["batch_" + k] = _num(r.get("batch_" + k))
                    rec["batch_n"] = _num(r.get("batch_n"))
                    rec["partial_step"] = r.get("partial_step") == "1"
                    rec["step_ok"] = r.get("step_ok") != "0"
                    out["poses"].append(rec)
        except OSError as e:
            out["warnings"].append(f"trajectory.csv unreadable: {e}")
    # THE PER-BATCH SERIES IS THE WINDOW MEAN, NOT THE STEPPING POSE. One optimiser step accumulates
    # accum_local poses and the row carrying step_taken == 1 is the LAST of them -- its loss fields
    # are that single pose's. Taking that row as the batch point was a reporting bug (found in
    # review 2026-10-16): it charted one pose out of 85 and labelled it "per batch", so a falling
    # batch mean could have been drawn as a flat line. Each point here averages the loss terms over
    # the whole accumulation window; lr, grad_norm and step stay from the stepping row, because those
    # are properties of the step and not of a pose.
    #
    # STILL RANK 0 ONLY. The trajectory is written by rank 0, which sees accum_local of the
    # accum_eff poses in a step (85 of 255 here). A true full-batch mean needs the trainer to
    # all-reduce the loss, which it does not do -- the gradient is reduced, the scalar is not.
    out["steps"] = _window_means(out["poses"], run_ended=bool(out["status"]))
    out["dropped_steps"] = [p for p in out["poses"] if p["nonfinite"]]

    for f in sorted(glob.glob(os.path.join(out_dir, "metric_ep*.json"))):
        try:
            m = json.load(open(f))
        except (json.JSONDecodeError, OSError) as e:
            out["warnings"].append(f"{os.path.basename(f)} unreadable: {e}")
            continue
        base = os.path.basename(f)
        try:
            ep = int(base.replace("metric_ep", "").replace(".json", ""))
        except ValueError:
            out["warnings"].append(f"cannot read an epoch number from {base}")
            continue
        m["epoch"] = ep
        out["epochs"].append(m)
    out["epochs"].sort(key=lambda m: m["epoch"])

    bp = baseline_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out_dir))),
                                       "training", "ft3_pilot_baseline.json")
    if os.path.exists(bp):
        try:
            out["baseline"] = json.load(open(bp))
        except (json.JSONDecodeError, OSError) as e:
            out["warnings"].append(f"baseline unreadable: {e}")
    else:
        out["warnings"].append(f"no baseline at {bp}; the reference lines will be absent")
    return out


def _window_means(poses, run_ended=False):
    """One record per optimiser step, averaging the loss terms over that step's accumulation window.

    See the note in read_run: the row flagged step_taken is the last pose of the window, so its own
    loss fields describe one pose.

    EPOCH BOUNDARIES CARRY A STEP THAT LEAVES NO ROW. At the end of an epoch (and of the run) the
    trainer calls _apply_partial, which scales the gradient up to a full-size step and takes it --
    but writes no trajectory row, because only the accum-boundary path appends one. Keying windows
    purely on step_taken therefore merged an epoch's trailing poses into the NEXT epoch's first
    window and discarded the run's final remainder (found in review 2026-10-16, before w23bjm
    reached its first boundary). A window is closed here on an epoch change as well, and whatever
    remains at the end is emitted too; both are marked `partial`, since the trainer's own count for
    them is reduced across ranks and is not recoverable from rank 0's rows alone.

    STILL RANK 0 ONLY. The trajectory is written by rank 0, which sees accum_local of the accum_eff
    poses in a step (85 of 255 here). A true full-batch mean needs the trainer to all-reduce the
    loss, which it does not do -- the gradient is reduced, the scalar is not.
    """
    out, cur = [], []

    def _close(window, partial):
        last = window[-1]
        if last["nonfinite"]:
            return
        rec = dict(last)
        for k in TERMS + ("raw", "loss"):
            vals = [q[k] for q in window if q.get(k) is not None]
            rec[k] = sum(vals) / len(vals) if vals else None
        rec["n_poses"] = len(window)
        rec["partial"] = partial
        out.append(rec)

    # IF THE TRAINER LOGGED THE BATCH ITSELF, use it and do not reconstruct anything. Those rows
    # already carry the cross-rank mean over every pose in the step, including the partial step an
    # epoch ends with, which rank 0's rows cannot express.
    exact = [p for p in poses if p.get("batch_n")]
    if exact:
        out = []
        for n, p in enumerate(exact, 1):
            rec = dict(p)
            for k in TERMS + ("raw", "loss"):
                v = p.get("batch_" + k)
                if v is not None:
                    rec[k] = v
            rec["raw"] = sum(p["batch_" + k] for k in TERMS
                             if p.get("batch_" + k) is not None and k != "loss")
            rec["n_poses"] = int(p["batch_n"])
            rec["partial"] = p.get("partial_step", False)
            rec["step_index"] = n
            out.append(rec)
        return out

    for p in poses:
        if cur and p["epoch"] != cur[-1]["epoch"]:
            # The epoch ended on the previous row and _apply_partial stepped without logging it.
            _close(cur, True)
            cur = []
        cur.append(p)
        if p["step_taken"] or p["nonfinite"]:
            _close(cur, False)
            cur = []
    if cur and run_ended:
        # THE RUN'S FINAL REMAINDER, stepped by _apply_partial at the end of train_loop and never
        # logged. Emitted ONLY once the run has ended: mid-run those trailing rows are a window
        # still ACCUMULATING, on which no step has been taken, so charting them would add a point
        # that moves with every flush and claim a step that has not happened.
        _close(cur, True)
    for n, rec in enumerate(out, 1):
        rec["step_index"] = n
    return out

def effective_batch(run):
    """The poses that ACTUALLY contribute to one optimiser step, which is not --accum.

    --accum is a request: the trainer splits it across ranks as accum // world, so 256 over three
    ranks is 85 each and 255 in total. The report printed the requested 256 and was wrong by a pose
    per rank. Derived here from the records rather than recomputed from the arguments, because the
    trajectory shows what happened: the gap between step_taken rows IS the per-rank accumulation.
    """
    steps = [i for i, p in enumerate(run["poses"]) if p["step_taken"]]
    gaps = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
    local = None
    if gaps:
        local = sorted(gaps)[len(gaps) // 2]
    world = (run.get("config") or {}).get("nproc")
    if not world:
        # Count the ranks that announced themselves, so a run without a config still resolves.
        try:
            with open(os.path.join(run["out_dir"], "run.log")) as fh:
                import re
                world = max((int(m.group(1)) for m in
                             re.finditer(r"\[rank \d+/(\d+)\]", fh.read())), default=None)
        except OSError:
            world = None
    if local and world:
        return local * world, local, world
    return run["args"].get("accum"), local, world


def pose_cost(run):
    """Per-pose wall time BUCKETED BY TOKEN COUNT, because the cost scales with the pair area.

    WHY THIS EXISTS. Timing on this pilot was misreported four times, and every time the mistake
    was the same: a median or a mean taken over whatever structures a run happened to reach, then
    compared against a different run that reached different ones. Measured, 699 tokens costs
    0.90 s/pose and 1364 costs 3.80 -- a 4.2x spread from size alone -- so ANY aggregate over mixed
    sizes says more about which structures were sampled than about the configuration. Comparisons
    belong within a token bucket, and n belongs beside every number.
    """
    tok = _token_counts()
    rows = []
    poses = run["poses"]
    for i in range(1, len(poses)):
        if poses[i]["pdb"] != poses[i - 1]["pdb"]:
            continue                      # a structure switch is item-build time, not pose time
        rows.append((tok.get(poses[i]["pdb"], -1), poses[i]["wall_s"] - poses[i - 1]["wall_s"],
                     poses[i]["pdb"]))
    out = []
    for lo, hi in ((0, 500), (500, 800), (800, 1200), (1200, 1600), (1600, 99999)):
        d = [t for n, t, _p in rows if lo <= n < hi]
        if not d:
            continue
        d.sort()
        out.append({"bucket": f"{lo}-{hi if hi < 99999 else '+'} tokens", "n": len(d),
                    "median": d[len(d) // 2], "mean": sum(d) / len(d),
                    "structures": len({p for n, _t, p in rows if lo <= n < hi})})
    return out


def _token_counts():
    import csv as _csv
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "train_val_splits", "cache_train_structures.csv")
    out = {}
    if os.path.exists(p):
        try:
            for r in _csv.DictReader(open(p)):
                if r.get("tokens", "").isdigit():
                    out[r["pdb"]] = int(r["tokens"])
        except OSError:
            pass
    return out


def epoch_times(run):
    """[(epoch, minutes, poses, structures)] from the per-pose wall clock.

    Derived rather than logged, so it is available for runs already in flight -- the trainer prints
    it too, but only from the next launch. An epoch's duration is the wall time of its last pose
    minus that of the previous epoch's last pose, which is exact because wall_s is measured from the
    same t0 throughout.
    """
    out, prev = [], 0.0
    byep = {}
    for p_ in run["poses"]:
        byep.setdefault(p_["epoch"], []).append(p_)
    for ep in sorted(byep):
        rows = byep[ep]
        end = rows[-1]["wall_s"]
        out.append({"epoch": ep, "minutes": (end - prev) / 60.0, "poses": len(rows),
                    "structures": len({r["pdb"] for r in rows}),
                    "complete": ep < max(byep)})
        prev = end
    return out


def summarise(run):
    """The few numbers a reader wants before any chart: where the run is and whether it is healthy."""
    poses, steps, args = run["poses"], run["steps"], run["args"]
    ep_total = args.get("epochs")
    accum = args.get("accum")
    s = {"poses": len(poses), "steps": len(steps),
         "dropped": len(run["dropped_steps"]),
         "epoch": poses[-1]["epoch"] if poses else 0,
         "epochs_total": ep_total,
         "wall_min": poses[-1]["wall_s"] / 60 if poses else 0.0,
         "lr": args.get("lr"), "accum": accum,
         "evals": len(run["epochs"])}
    if steps:
        n = max(1, len(steps) // 5)
        s["raw_first"] = sum(p["raw"] for p in steps[:n]) / n
        s["raw_last"] = sum(p["raw"] for p in steps[-n:]) / n
        gn = [p["grad_norm"] for p in steps if p["grad_norm"] is not None]
        if gn:
            s["grad_med"] = sorted(gn)[len(gn) // 2]
            s["grad_max"] = max(gn)
    if poses:
        s["structures"] = len({p["pdb"] for p in poses})
    return s
