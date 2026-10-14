#!/usr/bin/env python
"""Build a fold queue that brings a benchmark to --target usable seeds per structure.

    ./build_topup_queue.py --dataset foldbench --credit-suspect-except-flagged --apply
    ./build_topup_queue.py --dataset pxm2025 --apply

WHAT COUNTS AS A SEED A STRUCTURE ALREADY HAS is the whole question, because foldbench's 151-200
block is `suspect` in the pooled table: 16 interfaces over 10 complexes disagree with the pool at
p<0.001 on how often a seed finds an acceptable pose, in BOTH directions -- 8up2_AC goes 48/50 to
23/50, 8vyl_AF 15/50 to 0/50. Reduced recycles are ruled out (measured fold_seconds sits at
0.86-1.02x the token model against 0.97x for units whose n_cycle=10 is recorded) and the paired mean
DockQ matches to 0.007, so the block is fine for averages. It is coverage statistics -- max over
seeds -- that the disagreement moves.

Three policies, so the choice is explicit rather than implied by whichever table got read:

  --no-credit-suspect                  count only tier != suspect. Clean, and for foldbench that
                                       means 40 fresh seeds for all 104 structures (~36 h on 3 GPUs)
  --credit-suspect                     count every seed. Cheapest, and carries the disagreement into
                                       coverage counts
  --credit-suspect-except-flagged      credit the block EXCEPT for the complexes in --flagged, which
                                       are refolded from scratch. The default policy for foldbench

Seeds are allocated from --seed0 upward, skipping any seed the structure already has in any tier, so
a top-up can never collide with an existing fold.
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

sys.path[:0] = ["/home/ebrahim/tts_abag"]
import paths  # noqa: E402

# the 10 complexes whose suspect block disagrees with the pool at p<0.001; see each benchmark's
# free_pool/INDEX.md and extra_training/docs/seed_scaling_impact.md
FLAGGED = "8up2,7x8p,8vyl,8oni,7zoz,8be3,8urf,8che,8byu,9fww"


def tokens(jobs):
    out = {}
    for p in glob.glob(f"{jobs}/*.json"):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        j = j[0] if isinstance(j, list) else j
        n = 0
        for s in j.get("sequences", []):
            for v in s.values():
                if isinstance(v, dict) and isinstance(v.get("sequence"), str):
                    n += len(v["sequence"]) * int(v.get("count", 1) or 1)
        if n:
            out[j.get("name") or os.path.basename(p).split("-")[0]] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=310000)
    ap.add_argument("--flagged", default=FLAGGED)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-credit-suspect", action="store_true")
    g.add_argument("--credit-suspect", action="store_true")
    g.add_argument("--credit-suspect-except-flagged", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    d = paths.DATASETS[a.dataset]
    flagged = {x for x in a.flagged.split(",") if x}

    usable, every = collections.defaultdict(set), collections.defaultdict(set)
    for r in csv.DictReader(open(d.free_pool)):
        every[r["pdb"]].add(int(r["seed"]))
        if r["tier"] != "suspect":
            usable[r["pdb"]].add(int(r["seed"]))

    if a.credit_suspect:
        policy, have = "credit every seed", every
    elif a.no_credit_suspect:
        policy, have = "count only tier != suspect", usable
    else:
        policy = f"credit the suspect block except {len(flagged)} flagged complexes"
        have = {p: (usable[p] if p in flagged else every[p]) for p in every}

    tok = tokens(str(d.jobs))
    f = lambda n: 0.00182 * n ** 1.57
    rows, miss_tok, secs, poses = [], [], 0.0, 0.0
    for pdb in sorted(have):
        need = a.target - len(have[pdb])
        if need <= 0:
            continue
        if pdb not in tok:
            miss_tok.append(pdb)
            continue
        taken = every[pdb]
        s = a.seed0
        for _ in range(need):
            while s in taken:
                s += 1
            rows.append((pdb, s))
            taken.add(s)
            secs += f(tok[pdb])
            poses += 0.00376 * tok[pdb]
            s += 1

    print(f"{a.dataset}: target {a.target} seeds, policy = {policy}")
    print(f"   structures needing folds: {len({p for p, _ in rows})}   folds: {len(rows)}")
    print(f"   GPU {secs/3600:.0f} h -> {secs/3/3600:.1f} h on 3 cards   poses ~{poses/1024:.0f} GiB")
    if miss_tok:
        print(f"   NO JOB JSON for {len(miss_tok)}: {miss_tok[:5]} -- these cannot be folded")
    seeds = [s for _, s in rows]
    if seeds:
        print(f"   seed range used: {min(seeds)}-{max(seeds)}, none colliding with an existing fold")

    out = a.out or f"{d.data}/free_top{a.target}/queue.tsv"
    if a.apply:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            for pdb, s in rows:
                fh.write(f"free\t{pdb}\t*\tfree\t{s}\n")
        print(f"\nwrote {out}")
    else:
        print(f"\n(dry run; --apply writes {out})")


if __name__ == "__main__":
    main()
