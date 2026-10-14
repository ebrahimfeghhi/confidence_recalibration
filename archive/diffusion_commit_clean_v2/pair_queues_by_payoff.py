#!/usr/bin/env python
"""Order the baseline and method queues on ONE shared complex sequence, so rows land as pairs.

    ./pair_queues_by_payoff.py [--apply]

An interface enters the comparison only when BOTH arms have finished its complex. Ordering each queue
independently therefore wastes wall-clock on half-pairs: the method spent three GPUs finishing 8pn0,
whose baseline is still pending, while 8fbw sat 22 folds from delivering two decoy interfaces whose
baseline was already done.

So both queues take the same complex order, and the key is the pair's total cost:

    1. class: decoy, then nocov, then solved      the rows with something to prove
    2. folds remaining across BOTH arms           the cheapest complete pair first

Within a complex the existing line order is preserved. The rewrite is asserted to be a permutation of
each queue, and `folder.py` skips an (iface, seed, arm) already in results.csv, so partial complexes
keep their work. Restart the workers afterwards -- a queue is read once at startup.
"""
import argparse, collections, csv, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/analysis"); sys.path.insert(0, "/home/ebrahim/tts_abag")
import compare_arms as C                                           # noqa: E402
from analysis.free_vs_matched import matched_rows, BLOCK_BY_DATASET  # noqa: E402
from score_experiment import classes_at, pool_by_seed              # noqa: E402

FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
QUEUES = {"free": f"{FB}/free/queue.tsv", "method": f"{FB}/selftmpl_mask_on_nc10/queue.tsv"}
CSVS = {"free": f"{FB}/free/results.csv", "method": f"{FB}/selftmpl_mask_on_nc10/results.csv"}
RANK = {"decoy": 0, "nocov": 1, "solved": 2}


def load(q):
    by = collections.defaultdict(list)
    for l in open(q):
        if l.strip():
            by[l.rstrip("\n").split("\t")[1]].append(l.rstrip("\n"))
    return by


def scored(path):
    out = collections.defaultdict(set)
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r.get("dockq"):
                out[r["iface"].split("_")[0]].add(int(r["seed"]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    A = ap.parse_args()
    ds, _, _ = C.arm_defaults("foldbench")
    pool, first = pool_by_seed(50, str(ds.pool_csv))
    cls = classes_at(pool, first)
    per_cls = collections.defaultdict(set)
    for i, k in cls.items():
        per_cls[i.split("_")[0]].add(k)

    by = {a: load(q) for a, q in QUEUES.items()}
    sc = {a: scored(c) for a, c in CSVS.items()}
    left = {}
    for a in QUEUES:
        left[a] = {p: len({int(l.split("\t")[4]) for l in v} - sc[a][p]) for p, v in by[a].items()}

    def klass(p):
        s = per_cls.get(p, set())
        return min((k for k in s if k in RANK), key=lambda k: RANK[k]) if s & set(RANK) else "solved"

    allc = set(by["free"]) | set(by["method"])
    order = sorted(allc, key=lambda p: (RANK[klass(p)],
                                        left["free"].get(p, 0) + left["method"].get(p, 0), p))
    print(f"{'pdb':6}{'class':>7}{'free left':>11}{'method left':>13}")
    for p in order:
        f, m = left["free"].get(p, 0), left["method"].get(p, 0)
        if f or m:
            print(f"{p:6}{klass(p):>7}{f:>11}{m:>13}")
    for a, q in QUEUES.items():
        lines = [l for p in order if p in by[a] for l in by[a][p]]
        orig = [l.rstrip("\n") for l in open(q) if l.strip()]
        assert collections.Counter(lines) == collections.Counter(orig), f"{a}: not a permutation"
        if A.apply:
            bak = q.replace(".tsv", "_before_pairing.tsv.bak")
            if not os.path.exists(bak):
                shutil.copy2(q, bak)
            open(q, "w").write("\n".join(lines) + "\n")
            print(f"{a}: rewritten ({len(lines)} lines)")
    if not A.apply:
        print("\ndry run -- pass --apply")


if __name__ == "__main__":
    main()
