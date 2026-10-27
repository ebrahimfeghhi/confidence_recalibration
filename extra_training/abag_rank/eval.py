#!/usr/bin/env python
"""Score valfull with a trained reranker, by phase_k_eval's selection rules. CPU or one GPU.

THE METRIC IS THE PROJECT'S, NOT A NEW ONE. Selection is max(iptm_like, -seed, -sample) and
acceptability is dockq >= 0.23, exactly as phase_k_eval does -- its tie-break exists because "row
order differs between a sharded run and a single-process one, so two poses with equal score
selected different DockQ depending on how the work was split", and a second implementation that
drifted from it would not be comparable to anything in paper_results.md. That drift already
happened once in train_eval_metrics.py and moved a baseline by 0.0003.

THE BASELINE IS RECOMPUTED FROM THE SAME ROWS, never quoted from another file: shipped selects
with its own stored ipTM over this same pose population, so the comparison is paired on identical
interfaces.
"""
import argparse
import csv
import glob
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/abag_rank_upstream/ABAG-Rank")

import features as FT                                   # noqa: E402
from dataset import InterfacePoses                      # noqa: E402
from src.models.deep_set import DeepSet                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEPTABLE = 0.23


def shipped_rows(results_csv, budget, seed_base):
    """iface -> [(seed, sample, stored_iptm, dockq)] within the budget."""
    out = {}
    with open(results_csv) as fh:
        for r in csv.DictReader(fh):
            try:
                s, k = int(r["seed"]), int(r["sample"])
                if not (seed_base <= s < seed_base + budget):
                    continue
                out.setdefault(r["iface"], []).append((s, k, float(r["iptm"]), float(r["dockq"])))
            except (KeyError, ValueError):
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/abag/run1/best.pt")
    ap.add_argument("--dump", default="validation/abag/pae_valfull")
    ap.add_argument("--restypes", default="validation/abag/restypes")
    ap.add_argument("--pool", default="training/valfull/pool.csv")
    ap.add_argument("--results", default="validation/fold/results.csv")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=400000)
    ap.add_argument("--allow-partial", action="store_true",
                    help="report a metric over an incomplete population. Off by default: such a "
                         "number is not comparable to the valfull baselines in paper_results.md.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ds = InterfacePoses(os.path.join(ROOT, a.pool), os.path.join(ROOT, a.dump),
                        os.path.join(ROOT, a.restypes), os.path.join(ROOT, a.results),
                        k=1, require_full=False)
    print(f"  {len(ds)} interface(s) with poses and labels")
    ship = shipped_rows(os.path.join(ROOT, a.results), a.budget, a.seed_base)

    # THE POPULATION IS REQUIRED, NOT WHATEVER SURVIVED. An interrupted export would otherwise
    # give a plausible metric over fewer than the benchmark's interfaces, or fewer poses each, and
    # the shipped baseline recomputed over that same reduced set would NOT make it comparable to
    # the valfull numbers in paper_results.md. Caught in review.
    want_if = set()
    with open(os.path.join(ROOT, a.pool)) as fh:
        for r in csv.DictReader(fh):
            want_if.add(f"{r['pdb']}_{r['chain1']}{r['chain2']}")
    have_if = {it["iface"] for it in ds.items}
    want_depth = a.budget * 5
    bad = []
    if have_if != want_if:
        bad.append(f"{len(want_if - have_if)} interface(s) absent, {len(have_if - want_if)} extra; "
                   f"first absent: {sorted(want_if - have_if)[:4]}")
    for it in ds.items:
        got = [(sd, smp) for _p, sd, smp in it["poses"]
               if a.seed_base <= sd < a.seed_base + a.budget]
        if len(got) != want_depth or len(set(got)) != want_depth:
            bad.append(f"{it['iface']}: {len(got)} pose(s) ({len(set(got))} unique), "
                       f"expected {want_depth}")
        if len(bad) >= 6:
            break
    if bad and not a.allow_partial:
        raise SystemExit("STOP: the valfull population is incomplete, so this would not be the "
                         "benchmark:\n    " + "\n    ".join(bad)
                         + "\n  Finish the export, or pass --allow-partial and say so wherever "
                           "the number is used.")
    print(f"  population: {len(have_if)} interface(s) x {want_depth} pose(s), complete")

    dev = a.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(ROOT, a.ckpt), map_location=dev)
    model = DeepSet(input_dim=ck.get("feature_dim", FT.FEATURE_DIM),
                    phi_hidden_dims=[512, 256, 128, 256],
                    rho_hidden_dims=[2048, 1024, 512, 256, 128], output_dim=1,
                    aggregator="attn_pool_concat_stats",
                    sample_level_dim=ck.get("sample_level_dim", 1)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  checkpoint from epoch {ck.get('epoch')} "
          f"(internal val top-1 DockQ {ck.get('val_top1', float('nan')):.4f})")

    picked, shipped, oracle, missing = [], [], [], 0
    with torch.no_grad():
        for it in ds.items:
            mean, letters = ds._complex(it["pdb"], 0)
            feats, lens, meta, sfs = [], [], [], []
            for path, sd, smp in it["poses"]:
                if not (a.seed_base <= sd < a.seed_base + a.budget):
                    continue
                z = np.load(path)
                ch = ds._chains(z, it["iface"], it["pdb"], sd, smp)
                if ch is None:
                    continue
                f, npair = FT.build(z, mean, letters, keep_chains=ch)
                if npair == 0:
                    continue
                feats.append(f)
                lens.append(npair)
                sfs.append([ds.iptm.get((it["iface"], sd, smp), 0.0)])
                meta.append((sd, smp, ds.labels[(it["iface"], sd, smp)]))
            # COVERAGE IS CHECKED ON WHAT WAS SCORED. The filesystem check above proves the npz
            # files exist; it cannot know that a pose's OST mapping resolves or that it keeps any
            # pair after the X filter and cutoff. Without this, an interface could pass the
            # 150-pose assertion and then be ranked on 140 candidates, silently. Caught in review.
            if len(meta) != want_depth and not a.allow_partial:
                raise SystemExit(
                    f"STOP: {it['iface']} scored {len(meta)} of {want_depth} candidate(s) -- "
                    f"poses were dropped for an unresolved chain mapping or an empty interface "
                    f"after filtering. Selecting from a reduced candidate set is not the "
                    f"benchmark. Pass --allow-partial to accept it deliberately.")
            if not feats:
                missing += 1
                continue
            # ALL CANDIDATES IN ONE FORWARD PASS, [1, K, N, F].
            #
            # SCORING POSES ONE AT A TIME IS MATHEMATICALLY WRONG with this aggregator.
            # attn_pool_concat_stats normalises its pooled features ACROSS THE K POSES of a
            # complex (deep_set.py:199-205), so at K=1 the mean IS the single pose, the numerator
            # is exactly zero, and every pooled feature vanishes -- verified numerically, max|v|
            # = 0.000e+00. The only surviving input is size_feat = sqrt(n_pairs), so a per-pose
            # evaluator ranks candidates BY THEIR PAIR COUNT and reports a perfectly plausible
            # DockQ from it. Caught in review; this is the bug that would have made us conclude
            # the model does not generalise when the evaluator was discarding everything it learned.
            K = len(feats)
            N = max(len(f) for f in feats)
            x = torch.zeros(1, K, N, FT.FEATURE_DIM)
            for i, f in enumerate(feats):
                x[0, i, :len(f)] = torch.from_numpy(f)
            ln = torch.tensor([lens], dtype=torch.long, device=dev)
            sfv = torch.tensor([sfs], dtype=torch.float32, device=dev)
            scores = model(x.to(dev), ln, sfv).squeeze(-1).squeeze(0).tolist()
            rows = [(sd, smp, sc, dq) for (sd, smp, dq), sc in zip(meta, scores)]
            # phase_k_eval's tie-break, exactly: score, then lowest seed, then lowest sample.
            picked.append(max(rows, key=lambda t: (t[2], -t[0], -t[1]))[3])
            oracle.append(max(t[3] for t in rows))
            have = {(t[0], t[1]) for t in rows}
            sr = [r for r in ship.get(it["iface"], []) if (r[0], r[1]) in have]
            shipped.append(max(sr, key=lambda t: (t[2], -t[0], -t[1]))[3] if sr else float("nan"))

    n = len(picked)
    d = [p - s for p, s in zip(picked, shipped) if not np.isnan(s)]
    print(f"\n  NOTE: trained at K=10 poses per complex, scored here at K={a.budget * 5}. The "
          f"aggregator normalises across K, so scores are group-dependent by construction.")
    print(f"\n  RERANKER on {n} interface(s) at budget {a.budget}"
          + (f"; {missing} interface(s) had no scorable pose" if missing else ""))
    print(f"    mean DockQ   {statistics.mean(picked):.4f}")
    print(f"    shipped      {statistics.mean([s for s in shipped if not np.isnan(s)]):.4f}")
    print(f"    paired delta {statistics.mean(d):+.4f} +- "
          f"{statistics.stdev(d) / len(d) ** 0.5:.4f}" if len(d) > 1 else "")
    print(f"    frac >= {ACCEPTABLE}  {sum(1 for x in picked if x >= ACCEPTABLE) / n:.4f}"
          f"   shipped {sum(1 for x in shipped if x >= ACCEPTABLE) / n:.4f}")
    print(f"    oracle       {statistics.mean(oracle):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
