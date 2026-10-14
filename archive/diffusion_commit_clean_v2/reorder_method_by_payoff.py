#!/usr/bin/env python
"""Order the method queue by WHICH COMPLEX COMPLETES A TABLE ROW SOONEST.

    ./reorder_method_by_payoff.py [--apply]

An interface enters the comparison only when BOTH arms have finished it, so method folds on a complex
whose matched baseline is still pending buy nothing yet -- and size-ascending order had all three
workers on 8pn0, a complex in exactly that state, while 8fbw sat 22 folds from delivering two decoy
interfaces. Priority is therefore:

    1. is the complex's matched baseline already complete?   (otherwise the row cannot appear)
    2. class: decoy, then nocov, then solved                  (the rows with something to prove)
    3. fewest folds remaining                                 (soonest payoff first)

This only reorders. Rows are asserted to be a permutation of the queue, and `folder.py` skips an
(iface, seed, arm) already in results.csv, so partially folded complexes keep their work.
"""
import argparse, collections, csv, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/analysis")
sys.path.insert(0, "/home/ebrahim/tts_abag")
import compare_arms as C                                           # noqa: E402
from analysis.free_vs_matched import matched_rows, BLOCK_BY_DATASET  # noqa: E402
from score_experiment import classes_at, pool_by_seed              # noqa: E402

FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
N = f"{FB}/selftmpl_mask_on_nc10"
RANK = {"decoy": 0, "nocov": 1, "solved": 2}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    A = ap.parse_args()
    block = BLOCK_BY_DATASET["foldbench"]
    ds, _, _ = C.arm_defaults("foldbench")
    pool, first = pool_by_seed(50, str(ds.pool_csv))
    cls = classes_at(pool, first)
    per, _ = C.donors_used(N, first)
    mc = matched_rows(ds, *block)
    free_ip = {}
    for i in pool:
        r = [(p, d) for s, _, d, p in pool[i] if s in first[i.split("_")[0]]]
        if r:
            free_ip[i] = max(r)[0]
    scored = collections.defaultdict(set)
    for r in csv.DictReader(open(f"{N}/results.csv")):
        if r.get("dockq"):
            scored[r["iface"].split("_")[0]].add(int(r["seed"]))
    lines = [l.rstrip("\n") for l in open(f"{N}/queue.tsv") if l.strip()]
    by = collections.defaultdict(list)
    for l in lines:
        by[l.split("\t")[1]].append(l)

    def key(p):
        ifs = [i for i in pool if i.split("_")[0] == p and i in free_ip
               and free_ip[i] < 0.8 and cls.get(i) in RANK]
        want = min(len(per.get(p, ())), 50) or 50
        ready = bool(ifs) and all(len(mc.get(i, {})) >= want for i in ifs)
        k = min((cls[i] for i in ifs), key=lambda c: RANK[c]) if ifs else "solved"
        remaining = len({int(l.split("\t")[4]) for l in by[p]} - scored[p])
        return (not ready, RANK[k], remaining, p)

    order = sorted(by, key=key)
    out = [l for p in order for l in by[p]]
    assert collections.Counter(out) == collections.Counter(lines), "not a permutation"
    print("order:", " ".join(order))
    if A.apply:
        if not os.path.exists(f"{N}/queue_before_payoff_order.tsv.bak"):
            shutil.copy2(f"{N}/queue.tsv", f"{N}/queue_before_payoff_order.tsv.bak")
        open(f"{N}/queue.tsv", "w").write("\n".join(out) + "\n")
        print(f"written ({len(out)} lines)")
    else:
        print("\ndry run -- pass --apply")


if __name__ == "__main__":
    main()
