"""Render a run's report, and say which artifact to publish it to. One script, any run.

WHY ONE SCRIPT. This replaces render_report.py + watch.sh, which had to be invoked with three paths
that were easy to get wrong -- and were: after a run directory was deleted, the watcher kept
re-rendering a path that no longer existed and the published page silently went stale. Everything is
derived from a run id here, so there is one argument to get right.

  python -I run_logging/report.py                      # the latest run, once
  python -I run_logging/report.py --run-id kbigru       # a specific run
  python -I run_logging/report.py --watch               # re-render until the run ends
  python -I run_logging/report.py --list                # what runs exist, and their artifacts

PUBLISHING IS NOT AUTOMATED, AND CANNOT BE. An artifact is published through Claude's own tool, so no
script can do it; a background loop can only keep the FILE current. This script therefore prints the
artifact URL to publish to, and remembers it per run in run_logging/artifacts.json, so a run keeps
the same URL across republishes instead of spawning a new page each time -- which is what makes a
pinned link keep working.

THE PAGE SAYS HOW STALE IT IS. Because the file and the published page can diverge, the report
carries its generation time and ticks "rendered N minutes ago", amber past 20. A reader should not
have to wonder whether they are looking at current numbers.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "training", "runs")
REPORTS = os.path.join(HERE, "reports")
ARTIFACTS = os.path.join(HERE, "artifacts.json")
BASELINE = os.path.join(ROOT, "training", "ft3_pilot_baseline.json")

sys.path.insert(0, HERE)
import records as R
import render_report as RR


def load_artifacts():
    if os.path.exists(ARTIFACTS):
        try:
            return json.load(open(ARTIFACTS))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_artifact(run_id, url):
    m = load_artifacts()
    m[run_id] = url
    with open(ARTIFACTS, "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)


def resolve(run_id):
    """A run id, `latest`, or nothing (meaning latest) -> an existing directory."""
    if not run_id or run_id == "latest":
        link = os.path.join(RUNS, "latest")
        if os.path.islink(link) and os.path.isdir(link):
            return os.path.realpath(link), os.path.basename(os.path.realpath(link))
        # no symlink: take the most recently started run from the index
        ids = [d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d))
               and d != "latest"] if os.path.isdir(RUNS) else []
        if not ids:
            sys.exit(f"STOP: no runs under {RUNS}. Launch one with run_logging/launch.py.")
        ids.sort(key=lambda d: os.path.getmtime(os.path.join(RUNS, d)))
        return os.path.join(RUNS, ids[-1]), ids[-1]
    d = os.path.join(RUNS, run_id)
    if not os.path.isdir(d):
        sys.exit(f"STOP: no run {run_id!r} under {RUNS}")
    return d, run_id


def listing():
    arts = load_artifacts()
    idx = os.path.join(RUNS, "index.csv")
    if os.path.exists(idx):
        import csv
        rows = list(csv.DictReader(open(idx)))
        print(f"{'run':8s} {'started':20s} {'commit':9s} {'lr':8s} {'ep':3s} artifact")
        for r in rows:
            print(f"{r['run_id']:8s} {r['started'][:19]:20s} {r['commit']:9s} "
                  f"{r.get('lr',''):8s} {r.get('epochs',''):3s} "
                  f"{arts.get(r['run_id'], '-')}")
            if r.get("note"):
                print(f"         {r['note'][:100]}")
    else:
        print(f"no index at {idx}")


def render(run_dir, run_id):
    import datetime
    run = R.read_run(run_dir, BASELINE)
    summ = R.summarise(run)
    os.makedirs(REPORTS, exist_ok=True)
    out = os.path.join(REPORTS, f"{run_id}.html")
    gen = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    with open(out, "w") as fh:
        fh.write(RR.build(run, summ, gen, int(time.time())))
    return out, summ, run


def training_alive():
    """Bracketed so the pattern cannot match this process's own command line."""
    try:
        return subprocess.run(["pgrep", "-f", r"[p]hase_k_train\.py"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="", help="a run id, or `latest` (the default)")
    ap.add_argument("--watch", action="store_true", help="re-render until the run ends")
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--list", action="store_true", help="the runs, and the artifact each publishes to")
    ap.add_argument("--set-artifact", default="", help="record the artifact URL for this run")
    a = ap.parse_args()

    if a.list:
        listing()
        return 0

    run_dir, run_id = resolve(a.run_id)
    if a.set_artifact:
        save_artifact(run_id, a.set_artifact)
        print(f"{run_id} -> {a.set_artifact}")
        return 0

    while True:
        out, summ, run = render(run_dir, run_id)
        ep = run["epochs"]
        val = (f"  val: " + ", ".join(f"ep{m['epoch']} {m.get('mean_dockq')}" for m in ep)) if ep \
            else "  val: none yet"
        print(f"{run_id}: epoch {summ['epoch']}/{summ.get('epochs_total','?')}, "
              f"{summ['steps']} step(s), {summ['poses']} pose(s), "
              f"{summ['dropped']} dropped, {summ['evals']} eval(s)")
        print(val)
        print(f"  file: {out}")
        url = load_artifacts().get(run_id)
        print(f"  publish to: {url if url else 'NOT YET PUBLISHED -- publish, then re-run with '
                                              '--set-artifact <url> so this run keeps one page'}")
        if not a.watch:
            return 0
        if not training_alive():
            print("  no training process left; final render written, exiting")
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
