"""Run the validation metric across several GPUs instead of one, then merge.

WHY. A full pass is 5,300 validation units at 0.82 s/unit = ~73 minutes on one card, MEASURED on
GATE 1 2026-10-16 (73 units/min). The trainer evaluates every epoch, and during that evaluation the
other training ranks sit at a barrier doing nothing -- so a 3-epoch run spent ~3.7 h evaluating on
one card while two cards idled, against ~7.8 h of actual training.

The evaluator is embarrassingly parallel: every unit loads its own trunk and scores independently,
nothing is shared, and no gradient crosses between them. So there is no process group here and no
collective -- just N independent processes, one per card, each writing its rows, and a merge. That
also keeps it clear of the NCCL fault on this box (see phase_k_train.py --dist-backend).

WHAT IS AND IS NOT SPLIT. The per-pose work is split: the trunk, the head, the chain-pair resolution
and the pose-to-sample join. The AGGREGATE work is not, because it cannot be: a shard holds a slice
by construction, so the self-check, the coverage contract and the metric would all be computed over a
subset and labelled with the full budget. Those run once, on CPU, over the union.

THE SHARDS ARE STRIDED, not contiguous. Units are ordered by structure and structure sizes vary 2-3x,
so contiguous blocks would hand one shard the large complexes and the speedup would be set by the
slowest. Striding interleaves them. Perfect balance is still not expected -- the cards also contend
for the same disk reading trunks -- so treat the measured time, not 73/N, as the number.

EQUIVALENCE IS CHECKED, NOT ASSUMED. --verify-against runs the plain single-card path over the same
budget and requires the two metrics to be identical. Sharding changes which process computes a row,
and must change nothing else.
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
EVAL = os.path.join(ROOT, "phases", "phase_k_eval.py")


def visible_cards():
    """The PHYSICAL card ids this process may use, in the order CUDA numbers them.

    Under the trainer, CUDA_VISIBLE_DEVICES is already "0,1,3", so a subprocess asking for "the
    second card" must be given 1, not 2. Taking the parent's list rather than inventing indices is
    what keeps a shard off a card nobody granted.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [x.strip() for x in env.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="",
                    help="physical card ids, comma separated. Default: CUDA_VISIBLE_DEVICES as "
                         "inherited, so this never reaches a card the caller was not given.")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--work", default="", help="where the shard rows go; default beside --out")
    ap.add_argument("--extra", default="", help="passed through to every shard, space separated")
    ap.add_argument("--verify-against-single", action="store_true",
                    help="also run the unsharded path and require an identical metric. Doubles the "
                         "cost, so it is for proving the split once, not for every evaluation.")
    ap.add_argument("--timeout-min", type=float, default=240.0)
    a = ap.parse_args()

    cards = [x.strip() for x in a.cards.split(",") if x.strip()] or visible_cards()
    if not cards:
        sys.exit("STOP: no cards. Pass --cards or set CUDA_VISIBLE_DEVICES.")
    work = a.work or (os.path.dirname(os.path.abspath(a.out)) if a.out else os.getcwd())
    os.makedirs(work, exist_ok=True)
    tag = f"b{a.budget}_" + (os.path.basename(a.ckpt).replace(".pt", "") if a.ckpt else "shipped")
    rows = [os.path.join(work, f"rows_{tag}_s{i}.csv") for i in range(len(cards))]
    extra = a.extra.split() if a.extra else []

    t0 = time.time()
    procs = []
    for i, card in enumerate(cards):
        cmd = [PY, "-I", EVAL, "--budget", str(a.budget), "--shard", str(i),
               "--nshards", str(len(cards)), "--rows-out", rows[i], *extra]
        if a.ckpt:
            cmd += ["--ckpt", a.ckpt]
        lg = open(os.path.join(work, f"shard{i}.log"), "w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": card, "LAYERNORM_TYPE": "openfold"}
        print(f"  shard {i} -> card {card}")
        procs.append((i, card, subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, env=env),
                      lg))

    bad = []
    for i, card, pr, lg in procs:
        try:
            rc = pr.wait(timeout=a.timeout_min * 60)
        except subprocess.TimeoutExpired:
            pr.kill()
            rc = -9
        lg.close()
        if rc != 0:
            bad.append((i, card, rc))
    shard_s = time.time() - t0
    if bad:
        for i, card, rc in bad:
            print(f"  shard {i} on card {card} exited {rc}; tail of its log:")
            print("".join(open(os.path.join(work, f"shard{i}.log")).readlines()[-12:]))
        sys.exit(f"STOP: {len(bad)} of {len(cards)} shard(s) failed. The merge refuses a partial "
                 f"set, so no metric is reported.")
    print(f"  {len(cards)} shard(s) finished in {shard_s / 60:.1f} min")

    cmd = [PY, "-I", EVAL, "--budget", str(a.budget), "--rows-in", ",".join(rows), *extra]
    if a.out:
        cmd += ["--out", a.out]
    r = subprocess.run(cmd, env={**os.environ, "LAYERNORM_TYPE": "openfold"})
    if r.returncode:
        sys.exit(f"STOP: the merge failed ({r.returncode}). The shards are on disk at "
                 f"{work}, so it can be re-run without re-scoring.")
    print(f"  total {(time.time() - t0) / 60:.1f} min on {len(cards)} card(s)")

    if a.verify_against_single:
        print("\n== verifying against the unsharded path ==")
        single = os.path.join(work, f"metric_single_{tag}.json")
        cmd = [PY, "-I", EVAL, "--budget", str(a.budget), "--out", single, *extra]
        if a.ckpt:
            cmd += ["--ckpt", a.ckpt]
        t1 = time.time()
        lg = os.path.join(work, "single.log")
        with open(lg, "w") as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": cards[0],
                                    "LAYERNORM_TYPE": "openfold"})
        if r.returncode:
            print("".join(open(lg).readlines()[-12:]))
            sys.exit("STOP: the unsharded reference run failed, so the comparison proves nothing.")
        print(f"   unsharded: {(time.time() - t1) / 60:.1f} min on 1 card")
        if not a.out:
            sys.exit("STOP: --verify-against-single needs --out to compare against.")
        got, want = json.load(open(a.out)), json.load(open(single))
        keys = sorted(set(got) | set(want))
        diff = [k for k in keys if got.get(k) != want.get(k)]
        for k in keys:
            mark = "  <-- DIFFERS" if k in diff else ""
            print(f"   {k}: sharded={got.get(k)} single={want.get(k)}{mark}")
        if diff:
            sys.exit(f"STOP: sharding changed the metric on {diff}. It must change only which "
                     f"process computes a row.")
        print("   PASS: the sharded metric is identical to the unsharded one")


if __name__ == "__main__":
    main()
