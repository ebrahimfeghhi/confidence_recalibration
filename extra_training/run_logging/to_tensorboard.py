"""Stream a run's records into TensorBoard event files, without touching the trainer.

WHY A CONVERTER AND NOT A SummaryWriter IN THE TRAINER. Instrumenting the trainer couples it to a
viewer: adding a writer means a restart whenever the viewer changes, and this pilot has already spent
six launches. The records -- trajectory.csv, metric_ep<N>.json, config.json -- are the contract, and
this reads them. Consequences worth having: it works on runs that already finished or crashed, it can
be re-run to rebuild the event files from scratch, and if TensorBoard is abandoned later nothing in
the training path has to change back.

IT ALSO RUNS IN ITS OWN VENV. tensorboard pulls protobuf, grpcio and werkzeug, and the Protenix venv
is what trains; a version conflict there would break the thing we are trying to watch. So this
imports tensorboard's own EventFileWriter rather than torch.utils.tensorboard, needs no torch, and
lives in run_logging/.tbvenv.

WHAT IT WRITES

  train/plddt, train/pae, train/pde, train/resolved   the four raw loss terms, per optimiser step
  train/loss_sum                                      their sum, which is the curve to read
  train/grad_norm                                     PRE-clip, as clip_grad_norm_ returns it
  train/lr                                            including warmup
  val/mean_dockq, val/frac_acc                        per epoch, once an evaluation lands
  val/oracle_mean_dockq, val/headroom                 the ceiling and the gap to it
  baseline/mean_dockq, baseline/frac_acc              the shipped head, drawn at every step so it
                                                      reads as a horizontal line to compare against

THE X AXIS IS THE OPTIMISER STEP, not the pose. One step covers --accum poses (255 here: 85 per rank
x 3), so a per-pose series would be 85x denser than the thing being optimised and would read as
noise. The validation points are written at the step their epoch ended on, so both series share one
axis.

TAILING. With --follow it re-reads the records every --every seconds and writes only rows it has not
written before, so it can be left running beside a training job. It exits when the run's process is
gone and one final pass has been taken.
"""
import argparse, csv, glob, json, os, subprocess, sys, time

TERMS = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")


def writer_for(logdir):
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    os.makedirs(logdir, exist_ok=True)
    return EventFileWriter(logdir)


def scalar(w, tag, value, step, wall):
    from tensorboard.compat.proto import summary_pb2, event_pb2
    s = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))])
    w.add_event(event_pb2.Event(wall_time=float(wall), step=int(step), summary=s))


def read_steps(run_dir):
    """One record per OPTIMISER step -- DELEGATED to run_logging/records.py, not reimplemented.

    This used to walk trajectory.csv itself and take the step_taken row's own loss fields, which is
    one pose of 255 and is the bug review 2026-10-16 found in the HTML reporter. Two readers of the
    same records drifting apart is how that happened twice, so there is now one reader: records.py
    prefers the trainer's cross-rank batch columns, falls back to rank-0 window means for older
    runs, closes a window at an epoch boundary, and drops refused steps.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import records as _R
    _run = _R.read_run(run_dir)
    out = []
    for p in _run["steps"]:
        out.append({"step": p["step_index"], "wall_s": p["wall_s"], "epoch": p["epoch"],
                    "grad_norm": p["grad_norm"], "lr": p["lr"],
                    "loss_sum": p["loss"] if p["loss"] is not None else p["raw"],
                    **{k: p.get(k) for k in TERMS}})
    return out


def _read_steps_legacy(run_dir):
    """The old per-pose walk, kept only so the bug it caused stays visible in the diff."""
    p = os.path.join(run_dir, "trajectory.csv")
    if not os.path.exists(p):
        return []
    out = []
    try:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("step_taken") != "1" or r.get("nonfinite_grad") == "1":
                    continue
                try:
                    rec = {"step": int(float(r["step"])), "wall_s": float(r["wall_s"]),
                           "grad_norm": float(r["grad_norm"]) if r["grad_norm"] not in ("", "nan")
                           else None,
                           "lr": float(r["lr"]) if r.get("lr") else None,
                           "epoch": int(float(r.get("epoch", 0) or 0))}
                except (KeyError, ValueError):
                    continue
                tot = 0.0
                for k in TERMS:
                    try:
                        v = float(r[k])
                    except (KeyError, ValueError):
                        v = None
                    rec[k] = v
                    tot += v or 0.0
                rec["loss_sum"] = tot
                out.append(rec)
    except OSError:
        return []
    return out


def read_epochs(run_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "metric_ep*.json"))):
        try:
            m = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        try:
            m["epoch"] = int(os.path.basename(f).replace("metric_ep", "").replace(".json", ""))
        except ValueError:
            continue
        out.append(m)
    return sorted(out, key=lambda m: m["epoch"])


def baseline(run_dir, explicit=""):
    p = explicit or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(run_dir))),
                                 "training", "ft3_pilot_baseline.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def training_alive():
    """Bracketed, so the pattern cannot match this process's own command line."""
    try:
        return subprocess.run(["pgrep", "-f", r"[p]hase_k_train\.py"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="a run's output directory")
    ap.add_argument("--run-id", default="", help="a run id under training/runs/, or `latest`")
    ap.add_argument("--logdir", default="", help="where the event files go; default <run>/tb")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--follow", action="store_true", help="keep tailing until the run ends")
    ap.add_argument("--every", type=int, default=30)
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = a.run or (os.path.join(root, "training", "runs", a.run_id) if a.run_id else "")
    if not run or not os.path.isdir(run):
        sys.exit(f"STOP: pass --run <dir> or --run-id <id>; {run!r} is not a directory")
    logdir = a.logdir or os.path.join(run, "tb")
    base = baseline(run, a.baseline)

    w = writer_for(logdir)
    t0 = time.time()
    seen_steps, seen_epochs = set(), set()
    print(f"{os.path.basename(run)} -> {logdir}")
    while True:
        steps = read_steps(run)
        new = 0
        for r in steps:
            if r["step"] in seen_steps:
                continue
            seen_steps.add(r["step"])
            wall = t0 + r["wall_s"]
            for k in TERMS:
                if r[k] is not None:
                    scalar(w, f"train/{k.replace('_loss', '')}", r[k], r["step"], wall)
            scalar(w, "train/loss_sum", r["loss_sum"], r["step"], wall)
            if r["grad_norm"] is not None:
                scalar(w, "train/grad_norm", r["grad_norm"], r["step"], wall)
            if r["lr"] is not None:
                scalar(w, "train/lr", r["lr"], r["step"], wall)
            # THE BASELINE AT EVERY STEP, so it draws as a line across the validation charts rather
            # than a single point nobody can compare against.
            for k, tag in (("mean_dockq", "baseline/mean_dockq"),
                           ("frac_acc", "baseline/frac_acc")):
                if base.get(k) is not None:
                    scalar(w, tag, base[k], r["step"], wall)
            new += 1

        last_step = max(seen_steps) if seen_steps else 0
        for m in read_epochs(run):
            if m["epoch"] in seen_epochs:
                continue
            seen_epochs.add(m["epoch"])
            for k in ("mean_dockq", "frac_acc", "oracle_mean_dockq", "oracle_frac_acc", "headroom"):
                if m.get(k) is not None:
                    scalar(w, f"val/{k}", m[k], last_step, time.time())
            new += 1
        if new:
            w.flush()
            print(f"  +{new} record(s); {len(seen_steps)} step(s), {len(seen_epochs)} epoch(s)",
                  flush=True)
        if not a.follow:
            break
        if not training_alive():
            print("  no training process left; final pass written, exiting", flush=True)
            break
        time.sleep(a.every)
    w.close()


if __name__ == "__main__":
    main()
