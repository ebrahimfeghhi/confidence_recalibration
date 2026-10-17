"""Launch a training run into its own directory, with the settings and the commit recorded.

WHY THIS EXISTS. Launches used to share one directory keyed on the learning rate, so every restart
overwrote the last: on 2026-10-16 four launches collided in training/ft3_sweep/lr1e-3 and three of
the four loss curves were lost, because preserving them depended on me renaming files by hand before
each restart. The run logs that did survive are the only reason the eval-mode bug was diagnosable at
all. A run's identity is a LAUNCH, not a learning rate.

WHAT A RUN DIRECTORY HOLDS

  config.json    the settings, the git commit and whether the tree was dirty, the exact command, the
                 cards, the host, and the library versions. Written BEFORE the run starts, so a run
                 that dies in its first second still says what it was.
  status.json    how it ended: exit code, wall time, and the reason if one was given. Written after.
  trajectory.csv one row per pose (the trainer writes it); the per-step series is step_taken == 1
  metric_ep<N>.json  the validation metric after epoch N
  run.log        stdout of every rank
  head_*.pt      checkpoints. Kept because the per-epoch evaluation has to score something; saving
                 the finished head as a deliverable is deliberately NOT built yet.

THE COMMIT IS PART OF THE CONFIG, not a nicety. Labels carry a fingerprint over the builder's source,
so a run's validity depends on which code produced it; and --allow-code-drift lets a run proceed when
that source has changed. A curve whose commit is unknown cannot be compared with another.

training/runs/index.csv gets one row per launch, so runs are findable without encoding metadata in
their names.
"""
import argparse, csv, json, os, platform, secrets, shlex, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "training", "runs")
INDEX = os.path.join(RUNS, "index.csv")
INDEX_COLS = ["run_id", "started", "commit", "dirty", "lr", "epochs", "cards", "note", "out"]


def git(*args, default=""):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True,
                              timeout=20).stdout.strip() or default
    except (OSError, subprocess.SubprocessError):
        return default


def new_run_id(n=6):
    """A short random id. Random rather than sequential so two launches can never collide, and
    rather than a timestamp alone so a relaunch in the same minute is still distinct."""
    alphabet = "abcdefghijkmnopqrstuvwxyz23456789"      # no l/1, no 0/o
    while True:
        rid = "".join(secrets.choice(alphabet) for _ in range(n))
        if not os.path.exists(os.path.join(RUNS, rid)):
            return rid


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception as e:
        out["torch"] = f"unavailable: {type(e).__name__}"
    return out


def append_index(row):
    new = not os.path.exists(INDEX)
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in INDEX_COLS})


def main():
    ap = argparse.ArgumentParser(
        epilog="Anything after -- is passed to phase_k_train.py unchanged.")
    ap.add_argument("--note", default="", help="one line on why this launch exists; it goes in the "
                                               "config and the index, and a restart without a "
                                               "reason is how four launches became indistinguishable")
    ap.add_argument("--cards", default="0,1,3")
    ap.add_argument("--run-id", default="", help="default: a fresh random id")
    ap.add_argument("--dry-run", action="store_true", help="write the config and print the command")
    ap.add_argument("--skip-gate", action="store_true",
                    help="skip the label-provenance preflight. It takes ~2 min on CPU and iabg5i "
                         "burned 3 h and 65 steps on a fault it would have caught in that time, so "
                         "skipping it needs a reason.")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    rid = a.run_id or new_run_id()
    out = os.path.join(RUNS, rid)
    # A NON-EMPTY DIRECTORY IS REFUSED. new_run_id() already avoids collisions, but --run-id is
    # accepted from the caller and `exist_ok=True` let a second launch overwrite config.json while
    # the first run's trajectory.csv stayed behind -- a run whose settings and whose records came
    # from different launches, which is exactly the provenance failure this module exists to stop
    # (review 2026-10-16). A resumed run gets its own directory and names its parent in --note.
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"STOP: {out} already exists and is not empty ({len(os.listdir(out))} entry/ies). "
                 f"Its config.json would be overwritten while its trajectory.csv stayed, leaving a "
                 f"run whose settings and records come from different launches. Omit --run-id for a "
                 f"fresh one, or pick an unused id.")
    os.makedirs(out, exist_ok=True)

    train = os.path.join(ROOT, "phases", "phase_k_train.py")
    torchrun = "/home/ebrahim/Protenix/.venv/bin/torchrun"
    nproc = len([c for c in a.cards.split(",") if c.strip()])
    cmd = [torchrun, f"--nproc_per_node={nproc}", "--master_port=29540", train,
           *rest, "--out", out]

    # LR and epochs are pulled out of the passed-through arguments for the index, so the table is
    # readable without opening each config.
    def flag(name, default=""):
        return rest[rest.index(name) + 1] if name in rest and rest.index(name) + 1 < len(rest) \
            else default

    cfg = {"run_id": rid,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "note": a.note,
           "commit": git("rev-parse", "HEAD", default="unknown"),
           "commit_subject": git("log", "-1", "--format=%s", default=""),
           # DIRTY MEANS TRACKED FILES CHANGED, not that untracked files exist. The repo always
           # holds untracked data directories (folds, scored interfaces, prep outputs), so counting
           # those made every run "dirty" and the flag stopped meaning anything. What bears on
           # reproducibility is whether the CODE differs from the commit.
           "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           # SPLIT, DON'T SLICE. git() strips its output, which eats the leading space of the
           # FIRST porcelain line only (" M path" -> "M path"), so a fixed l[3:] cut one character
           # off exactly one filename and the record of which files were dirty was wrong for it.
           "dirty_files": [l.split(maxsplit=1)[1] for l in
                           git("status", "--porcelain", "--untracked-files=no").splitlines()
                           if l.split(maxsplit=1)[1:]][:40],
           "untracked_count": len(git("status", "--porcelain").splitlines())
           - len(git("status", "--porcelain", "--untracked-files=no").splitlines()),
           "branch": git("rev-parse", "--abbrev-ref", "HEAD", default=""),
           "cards": a.cards,
           "nproc": nproc,
           "command": " ".join(shlex.quote(c) for c in cmd),
           "passed_through": rest,
           "host": platform.node(),
           "versions": versions(),
           "out": out}
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)

    append_index({"run_id": rid, "started": cfg["started"], "commit": cfg["commit"][:8],
                  "dirty": "yes" if cfg["dirty"] else "no", "lr": flag("--lr"),
                  "epochs": flag("--epochs"), "cards": a.cards, "note": a.note, "out": out})

    # `latest` is a convenience for the report renderer and nothing depends on it being correct.
    link = os.path.join(RUNS, "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(rid, link)
    except OSError:
        pass

    print(f"run {rid}  ->  {out}")
    print(f"  commit {cfg['commit'][:8]}{' (DIRTY)' if cfg['dirty'] else ''}  cards {a.cards}")
    if cfg["dirty"]:
        print(f"  {len(cfg['dirty_files'])} MODIFIED tracked file(s): {cfg['dirty_files'][:4]}")
        print(f"  this run is not reproducible from a clean checkout of {cfg['commit'][:8]}")
    if a.note:
        print(f"  note: {a.note}")
    print(f"  {cfg['command']}")
    if a.dry_run:
        return 0

    # THE PREFLIGHT, RUN HERE RATHER THAN REMEMBERED. phase_k_train verifies a label's fingerprint
    # when it reaches that unit, which is correct but late: iabg5i died at step 65, three hours in,
    # on one structure of 491. The gate checks all of them up front in about two minutes on CPU.
    # It is given the SAME --manifest-skip and --seeds the training run gets, so the population it
    # checks is the population that will be trained on.
    if not a.skip_gate:
        gate = [sys.executable, "-I", os.path.join(ROOT, "phases", "phase_k_fingerprint_scan.py")]
        for flag in ("--manifest-skip", "--seeds"):
            if flag in rest:
                gate += [flag, rest[rest.index(flag) + 1]]
        print(f"  preflight: {' '.join(shlex.quote(c) for c in gate)}")
        g = subprocess.run(gate, env={**os.environ, "LAYERNORM_TYPE": "openfold"},
                           capture_output=True, text=True)
        tail = [l for l in g.stdout.splitlines() if "GATE" in l or "labels," in l]
        for l in tail:
            print(f"    {l}")
        with open(os.path.join(out, "preflight.log"), "w") as fh:
            fh.write(g.stdout + g.stderr)
        if g.returncode != 0:
            with open(os.path.join(out, "status.json"), "w") as fh:
                json.dump({"run_id": rid, "exit_code": g.returncode,
                           "reason": "label-provenance preflight failed; nothing was launched",
                           "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh, indent=1)
            print(f"STOP: the preflight failed (exit {g.returncode}); nothing launched. "
                  f"Full output in {os.path.join(out, 'preflight.log')}")
            return g.returncode

    # expandable_segments:True at the user's direction on 2026-10-16, after an OOM whose own error
    # message recommended it: 5.37 GB of the card was "reserved by PyTorch but unallocated", i.e.
    # lost to fragmentation, against 85.30 GB actually allocated. It lets the allocator grow a
    # segment instead of needing a contiguous block, so it reclaims that 5.37 GB. It is free and it
    # is not a fix on its own -- 6% of what was resident.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.cards, "LAYERNORM_TYPE": "openfold",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "a") as lg:
        rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env).returncode
    with open(os.path.join(out, "status.json"), "w") as fh:
        json.dump({"run_id": rid, "exit_code": rc, "wall_min": round((time.time() - t0) / 60, 1),
                   "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, fh, indent=1)
    print(f"run {rid} exit={rc} after {(time.time()-t0)/60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
