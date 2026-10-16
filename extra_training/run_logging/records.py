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
                    out["poses"].append(rec)
        except OSError as e:
            out["warnings"].append(f"trajectory.csv unreadable: {e}")
    # THE PER-BATCH SERIES IS THE STEPS, NOT THE POSES. One optimiser step covers --accum poses, so
    # a per-pose curve is 85x denser than the thing being optimised and reads as noise.
    out["steps"] = [p for p in out["poses"] if p["step_taken"] and not p["nonfinite"]]
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
