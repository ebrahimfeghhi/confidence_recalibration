#!/usr/bin/env python
"""Launch folder.py workers on a queue -- any queue, free or self-templated.

SUMMARY
    One launcher for both kinds of run: same driver, same claim directory, same GPU rules. Assembling
    the folder.py command by hand instead is how workers end up pointed at the wrong paths.

    Several workers read one queue at once. Each claims a line by creating a directory under the claim
    directory before folding it, so two workers never take the same fold and there is nothing to split
    up by hand. A worker killed mid-fold leaves a claim behind, which folder.py releases at startup.

    Devices are never guessed -- this is a shared machine -- and more than four is refused. Workers go
    in tmux, not background jobs, which die with the shell that started them and leave claims behind.

INPUTS
    --queue:      the work list, free or templated. Required.
    --out:        where folds land. Default the dataset's free pool.
    --templates:  template namespace. Required if the queue has any tmpl lines.
    --gpus:       explicit device ids, e.g. 0,1,2,5. Required by --run; never guessed.
    --allow-many: permit more than four GPUs. Default off.
    --run:        launch. Default off, in which case the plan is printed and nothing starts.
    --drop-cifs:  delete poses after scoring. Default OFF -- a pose whose coordinates are gone cannot
                  be templated on or used as a baseline, and folds do not reproduce.
    --csv:        results csv. Default <out>/results.csv.
    --claim-dir:  claim directory. Default <out>/claims.
    --log-dir:    where w<gpu>.log go. Default <out>.
    --env:        KEY=VALUE set in every worker's environment; repeatable. For the settings folder.py
                  reads from the environment (e.g. PROTENIX_MAX_TEMPLATE_CANDIDATES). A tmux session does not
                  inherit this shell's environment, so exporting them here would not reach the workers.
    --session:    tmux sessions are <session>w<gpu>. Default the log dir's name -- pass one when two
                  datasets have arms of the same name, or the watcher counts one arm's workers as both.
    --dataset:    pxm2025 or foldbench. Default pxm2025. Passed to every worker.
    --nsample:    poses per fold. Default 5.
    --overlap-scoring: workers score each fold on a background thread while the next one folds
                  (folder.py --overlap-scoring). Default off.

OUTPUTS
    tmux sessions: one per GPU, named <basename of log dir>w<gpu>, each running folder.py on the queue
    <log-dir>/w<gpu>.log: that worker's output

EXAMPLE
    call_folder.py --queue experimental_data/free/queue_free.tsv --out experimental_data/free \
                   --gpus 0,1,2,5 --run

    E=experimental_data/self_template_fixed_50_seed
    call_folder.py --queue $E/queue_tofold.tsv --out $E/poses --csv $E/results.csv \
                   --claim-dir $E/claims --log-dir $E --templates $E/templates --gpus 0 --run

FUNCTIONS
    check():   refuse a queue that cannot run: missing, templated with no namespace, or a templated
               queue aimed at the shared free pool
    launch():  one tmux session per GPU, all reading the same queue
    main():    validate, then launch
"""
import argparse
import os
import shlex
import subprocess
import sys

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_GPUS = 4


def check(queue, out, templates, free):
    """in: queue path, output dir, namespace or None, the dataset's free pool.
    out: (n folds, n templated); raises on refusal."""
    if not os.path.exists(queue):
        raise SystemExit(f"no queue at {queue}")
    kinds = [l.split("\t")[0] for l in open(queue) if l.strip()]
    n_tmpl = sum(1 for k in kinds if k != "free")
    if n_tmpl and not templates:
        raise SystemExit(f"{n_tmpl} templated folds need --templates (run the template builder first)")
    if n_tmpl and out == free:
        raise SystemExit("refusing to write a templated queue into the free pool; pass --out")
    return len(kinds), n_tmpl


def launch(gpus, queue, out, templates, dataset, nsample, prefix, csv_path, claim_dir,
           log_dir, drop_cifs=False, overlap_scoring=False, env=()):
    """in: devices and the run's paths.  out: the tmux session names started, one per GPU."""
    names = []
    for g in gpus:
        extra = "".join(f"{shlex.quote(e)} " for e in env)
        cmd = (f"cd {HERE} && CUDA_VISIBLE_DEVICES={g} LAYERNORM_TYPE=openfold {extra}"
               f"{paths.PROTENIX_PYTHON} folder.py --queue {shlex.quote(queue)} "
               f"--out {shlex.quote(out)} --csv {shlex.quote(csv_path)} "
               f"--claim-dir {shlex.quote(claim_dir)} --dataset {dataset} "
               f"--nsample {nsample} "
               + (f"--templates {shlex.quote(templates)} " if templates else "")
               + f"{'--drop-cifs ' if drop_cifs else ''}"
               + f"{'--overlap-scoring ' if overlap_scoring else ''}"
               + f"2>&1 | tee -a {shlex.quote(log_dir)}/w{g}.log")
        s = f"{prefix}{g}"
        subprocess.run(["tmux", "new-session", "-d", "-s", s, cmd], check=True)
        names.append(s)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--out", default=None, help="default the dataset's free pool")
    ap.add_argument("--templates", default=None, help="namespace, required for templated queues")
    ap.add_argument("--csv", default=None, help="default <out>/results.csv")
    ap.add_argument("--claim-dir", default=None, help="default <out>/claims")
    ap.add_argument("--log-dir", default=None, help="default <out>; worker logs are w<gpu>.log")
    ap.add_argument("--session", default=None, help="tmux name prefix; default the log dir's name")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every worker; repeatable")
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--gpus", default=None, help="explicit device ids, e.g. 0,1,2,5")
    ap.add_argument("--allow-many", action="store_true",
                    help=f"permit more than {MAX_GPUS} GPUs on this shared machine")
    ap.add_argument("--drop-cifs", action="store_true",
                    help="delete poses after scoring. OFF by default: a scored pose whose coordinates "
                         "are gone cannot be templated on, re-examined, or used as a baseline, and "
                         "folds do not reproduce so it cannot be recovered")
    ap.add_argument("--overlap-scoring", action="store_true",
                    help="workers score each fold in the background while the next one folds")
    ap.add_argument("--run", action="store_true", help="launch; otherwise only report the plan")
    A = ap.parse_args()
    for e in A.env:
        if "=" not in e or not e.split("=", 1)[0].isidentifier():
            raise SystemExit(f"--env {e!r}: expected KEY=VALUE")

    free = str(paths.DATASETS[A.dataset].free)
    out = A.out or free
    n, n_tmpl = check(A.queue, out, A.templates, free)
    print(f"  queue {A.queue}\n  folds {n} ({n_tmpl} templated)\n  out   {out}")
    print(f"  csv   {A.csv or out + '/results.csv'}\n  claims {A.claim_dir or out + '/claims'}")
    if A.env:
        print(f"  env   {' '.join(A.env)}")

    if not A.run:
        print("\n  nothing launched. re-run with --run --gpus <ids>")
        return 0
    if not A.gpus:
        raise SystemExit("--run needs --gpus; this machine is shared and devices are never guessed")
    gpus = [g.strip() for g in A.gpus.split(",") if g.strip()]
    if len(gpus) > MAX_GPUS and not A.allow_many:
        raise SystemExit(f"{len(gpus)} GPUs requested; cap is {MAX_GPUS}. Pass --allow-many.")
    csv_path = A.csv or f"{out}/results.csv"
    claim_dir = A.claim_dir or f"{out}/claims"
    log_dir = A.log_dir or out
    prefix = f"{A.session or os.path.basename(os.path.normpath(log_dir))}w"
    names = launch(gpus, A.queue, out, A.templates, A.dataset, A.nsample,
                   prefix, csv_path, claim_dir, log_dir, A.drop_cifs, A.overlap_scoring, A.env)
    print(f"\n  launched {len(names)} workers: {', '.join(names)}")
    print(f"  watch: tmux attach -t {names[0]}   |   tail -f {out}/w{gpus[0]}.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
