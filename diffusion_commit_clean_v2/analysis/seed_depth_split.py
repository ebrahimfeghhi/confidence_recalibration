#!/usr/bin/env python
"""How the decoy / acceptable / no-acceptable split moves with seed depth.

    ./seed_depth_split.py --dataset pxm2025
    ./seed_depth_split.py --dataset pxm2025 --depths 10,20,...,100 --boots 30

An interface is one of three things at a given seed depth:

  no-acceptable   no pose in the draw reaches DockQ 0.23 -- a SAMPLING failure; no reranking of
                  these poses can fix it
  decoy           an acceptable pose is in the draw but top pair-ipTM picks one below 0.23 -- a
                  SELECTION failure, winnable with no extra folding
  acceptable      top pair-ipTM picks an acceptable pose

WHY BOOTSTRAP. At depth k the answer depends on WHICH k seeds you drew, and the first k are not a
random draw -- an earlier 20-seed read of PXM was about 3 sd unlucky on decoys. Each depth is
therefore resampled `--boots` times WITHOUT replacement from the interface's own seed universe, and
the spread is reported next to the mean. At k equal to the universe size there is exactly one
possible draw, so it is computed once and its sd is 0 by construction, not by luck.

ONE UNIVERSE PER INTERFACE, AND IT IS EXACTLY `--universe` SEEDS. Interfaces hold different numbers
of seeds (pxm2025 ranges 61 to 145), and letting each contribute everything it has would mean the
depth-100 column was a different experiment per interface. Each interface contributes its
lowest-numbered `--universe` usable seeds and nothing else; interfaces with fewer are DROPPED, and
how many were dropped is printed, because a quietly shrinking denominator is how these counts go
wrong.

USABLE means `tier != "suspect"` in the benchmark's free_pool table (paths.DATASETS[ds].free_pool).
foldbench's 151-200 block is suspect: 16 interfaces disagree with the pool on how often a seed finds
an acceptable pose, in both directions, and these are max-over-seeds statistics -- exactly what such
a disagreement moves. See the free_pool INDEX.md.
"""
import argparse
import collections
import csv
import random
import statistics as st
import sys

sys.path[:0] = ["/home/ebrahim/tts_abag"]
import paths  # noqa: E402


def load(ds, universe, include_suspect=False):
    """in: dataset name, universe size.  out: {iface: {seed: [(dockq, pair_iptm), ...]}} with
    exactly `universe` seeds per interface, and the drop count."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(paths.DATASETS[ds].free_pool) as fh:
        for r in csv.DictReader(fh):
            if r["tier"] == "suspect" and not include_suspect:
                continue
            k = (r["pdb"], r["chain1"], r["chain2"])
            per[k][int(r["seed"])].append((float(r["dockq"]), float(r["pair_iptm"])))
    out, dropped = {}, 0
    for k, sd in per.items():
        if len(sd) < universe:
            dropped += 1
            continue
        keep = sorted(sd)[:universe]
        out[k] = {s: sd[s] for s in keep}
    return out, dropped


def split(data, k, rng, thresh):
    """the three counts for one draw of k seeds per interface."""
    dec = acc = noacc = 0
    for sd in data.values():
        seeds = list(sd) if k >= len(sd) else rng.sample(list(sd), k)
        poses = [p for s in seeds for p in sd[s]]
        if not poses:
            continue
        oracle = max(p[0] for p in poses)
        chosen = max(poses, key=lambda p: p[1])[0]
        if oracle < thresh:
            noacc += 1
        elif chosen < thresh:
            dec += 1
        else:
            acc += 1
    return dec, acc, noacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pxm2025")
    ap.add_argument("--depths", default="10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--boots", type=int, default=30)
    ap.add_argument("--universe", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=0.23)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-suspect", action="store_true")
    a = ap.parse_args()

    data, dropped = load(a.dataset, a.universe, a.include_suspect)
    rng = random.Random(a.seed)
    print(f"{a.dataset}: {len(data)} interfaces with >= {a.universe} usable seeds "
          f"({dropped} dropped for having fewer); {a.boots} draws per depth; "
          f"acceptable at DockQ >= {a.thresh}")
    print()
    print("| seeds | decoy | acceptable | no acceptable |")
    print("|---|---|---|---|")
    for k in [int(x) for x in a.depths.split(",")]:
        if k > a.universe:
            continue
        n = 1 if k == a.universe else a.boots
        draws = [split(data, k, rng, a.thresh) for _ in range(n)]
        cells = []
        for j in range(3):
            v = [d[j] for d in draws]
            cells.append(f"{st.mean(v):.1f}" if n == 1
                         else f"{st.mean(v):.1f} ± {st.stdev(v):.1f}")
        print(f"| {k} | {cells[0]} | {cells[1]} | {cells[2]} |")
    print()
    print("Rows are seed depth. Cells are interface counts, mean ± sd over draws; the last row is "
          "the single possible draw, so it has no spread.")


if __name__ == "__main__":
    main()
