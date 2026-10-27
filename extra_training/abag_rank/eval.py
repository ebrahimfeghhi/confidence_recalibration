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
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ds = InterfacePoses(os.path.join(ROOT, a.pool), os.path.join(ROOT, a.dump),
                        os.path.join(ROOT, a.restypes), os.path.join(ROOT, a.results),
                        k=1, require_full=False)
    print(f"  {len(ds)} interface(s) with poses and labels")
    ship = shipped_rows(os.path.join(ROOT, a.results), a.budget, a.seed_base)

    dev = a.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(ROOT, a.ckpt), map_location=dev)
    model = DeepSet(input_dim=ck.get("feature_dim", FT.FEATURE_DIM),
                    phi_hidden_dims=[512, 256, 128, 256],
                    rho_hidden_dims=[2048, 1024, 512, 256, 128], output_dim=1,
                    aggregator="attn_pool_concat_stats", sample_level_dim=0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  checkpoint from epoch {ck.get('epoch')} (internal val {ck.get('val'):.4f})")

    picked, shipped, oracle, missing = [], [], [], 0
    with torch.no_grad():
        for it in ds.items:
            mean, letters = ds._complex(it["pdb"], 0)
            rows = []
            for path, sd, smp in it["poses"]:
                if not (a.seed_base <= sd < a.seed_base + a.budget):
                    continue
                z = np.load(path)
                c1, c2 = ds._chains(z, it["iface"])
                f, npair = FT.build(z, mean, letters, keep_chains=(c1, c2))
                if npair == 0:
                    continue
                x = torch.from_numpy(f).unsqueeze(0).unsqueeze(0).to(dev)
                ln = torch.tensor([[npair]], dtype=torch.long, device=dev)
                score = float(model(x, ln).squeeze())
                rows.append((sd, smp, score, ds.labels[(it["iface"], sd, smp)]))
            if not rows:
                missing += 1
                continue
            # phase_k_eval's tie-break, exactly: score, then lowest seed, then lowest sample.
            picked.append(max(rows, key=lambda t: (t[2], -t[0], -t[1]))[3])
            oracle.append(max(t[3] for t in rows))
            sr = [r for r in ship.get(it["iface"], []) if (r[0], r[1]) in {(t[0], t[1]) for t in rows}]
            shipped.append(max(sr, key=lambda t: (t[2], -t[0], -t[1]))[3] if sr else float("nan"))

    n = len(picked)
    d = [p - s for p, s in zip(picked, shipped) if not np.isnan(s)]
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
