"""ipTM baseline and oracle on the held-out 30%, over EXACTLY the poses the training loop scores.

The run's `val top-1 DockQ` is not over all 50 poses of an interface. dataset.__getitem__ draws
K=10 of them from a stateful numpy Generator, and the DataLoader re-forks its workers every epoch
from a parent whose rng the parent never advances -- which is why the printed oracle is pinned at
the same value every epoch. So a baseline computed over all 50 poses, or over a fresh draw of 10,
would not be the same candidate set and the comparison would be void.

Rather than reverse-engineer which poses that draw selects, this iterates the SAME DataLoader with
the same seed, k, batch size, worker count and ordering, and reads ipTM and DockQ out of the
batches it hands back. `sample_features` is ipTM, `y` is the logit of DockQ, so no model is needed
and no GPU is touched. Reproducing the training loop's oracle to four decimals is the check that
the candidate sets really are identical; the script says so outright and exits nonzero if not.

The val membership is read from split.json rather than re-deriving the shuffle, so this cannot
drift from the run it describes.

Usage:  python -I abag_rank/val_baseline.py [--run training/abag/run_7030]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dataset import InterfacePoses, collate   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="training/abag/run_7030")
ap.add_argument("--dump", default="training/abag/pae_train")
ap.add_argument("--restypes", default="training/abag/restypes")
ap.add_argument("--pool", default="training/full_train/pool_insample.csv")
ap.add_argument("--results", default="training/fold/results.csv")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--expect-oracle", type=float, default=None,
                help="the training loop's printed oracle; mismatch is a hard failure")
ap.add_argument("--ckpt", default=None,
                help="also score this checkpoint's top-1 over the same poses")
ap.add_argument("--side", default="val", choices=["val", "train", "all"],
                help="which structures to score: the held-out 30%, the fitted 70%, or everything")
ap.add_argument("--tag", default=None, help="name for the output json; defaults to --side")
a = ap.parse_args()

# A run with --val-frac 0 trained on everything and has no split.json; it has no held-out side,
# so only --side all/train mean anything for it and asking for "val" is an error, not an empty set.
sf = os.path.join(ROOT, a.run, "split.json")
if os.path.exists(sf):
    split = json.load(open(sf))
    val_pdbs = set(split["val_pdbs"])
    print(f"  split.json: seed {split['seed']}, val_frac {split['val_frac']}, "
          f"{split['n_val_structures']}/{split['n_structures']} structures held out")
else:
    if a.side == "val":
        sys.exit(f"STOP: no split.json in {a.run}; that run has no held-out side.")
    split, val_pdbs = None, set()
    print(f"  no split.json in {a.run}: treating the whole pool as fitted (--side {a.side})")

torch.manual_seed(a.seed)
ds = InterfacePoses(os.path.join(ROOT, a.pool), os.path.join(ROOT, a.dump),
                    os.path.join(ROOT, a.restypes), os.path.join(ROOT, a.results),
                    k=a.k, seed=a.seed)
if a.side == "val":
    va_idx = [i for i, it in enumerate(ds.items) if it["pdb"] in val_pdbs]
    assert len(va_idx) == split["n_val_interfaces"], (len(va_idx), split["n_val_interfaces"])
elif a.side == "train":
    va_idx = [i for i, it in enumerate(ds.items) if it["pdb"] not in val_pdbs]
else:
    va_idx = list(range(len(ds.items)))
print(f"  side={a.side}: {len(va_idx)} interface(s), k={a.k} poses each")

# THE MODEL MUST SEE ALL K POSES IN ONE FORWARD. attn_pool_concat_stats normalises the pooled
# features across the K axis, so scoring poses one at a time collapses every feature to zero and
# returns the same number for every candidate. That bug cost this project a day; the DataLoader
# here hands over [B, K, N, F] groups, which is the shape the training loop uses.
model = None
if a.ckpt:
    sys.path.insert(0, "/home/ebrahim/abag_rank_upstream/ABAG-Rank")
    from src.models.deep_set import DeepSet
    import features as _FT
    ck = torch.load(os.path.join(ROOT, a.ckpt) if not os.path.isabs(a.ckpt) else a.ckpt,
                    map_location="cpu")
    model = DeepSet(input_dim=_FT.FEATURE_DIM, phi_hidden_dims=[512, 256, 128, 256],
                    rho_hidden_dims=[2048, 1024, 512, 256, 128], output_dim=1,
                    aggregator="attn_pool_concat_stats", sample_level_dim=1)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  scoring {a.ckpt} (epoch {ck.get('epoch', '?')}) on cpu")

dv = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=a.batch, shuffle=False,
                collate_fn=collate, num_workers=4)

iptm_dq, orc_dq, rnd_dq, mdl_dq = [], [], [], []
rng = np.random.default_rng(a.seed)
for b in dv:
    dq = torch.sigmoid(b["y"])                      # [B, K] true DockQ
    iptm = b["sample_features"].squeeze(-1)         # [B, K] the shipped selector's score
    # argmax without an explicit (-seed, -sample) tie-break: ipTM ties among 10 float32 scores do
    # not occur here, and the training loop's own top-1 is a bare argmax on the same axis.
    iptm_dq += dq.gather(1, iptm.argmax(dim=1, keepdim=True)).squeeze(1).tolist()
    orc_dq += dq.max(dim=1).values.tolist()
    # a random pick over the same candidates: the floor that says how much of ipTM's score is
    # selection rather than the pose set simply being good.
    pick = torch.from_numpy(rng.integers(0, dq.shape[1], size=dq.shape[0]))
    rnd_dq += dq.gather(1, pick[:, None]).squeeze(1).tolist()
    if model is not None:
        with torch.no_grad():
            pred = model(b["x"], b["lengths"], b["sample_features"]).squeeze(-1)
        mdl_dq += dq.gather(1, pred.argmax(dim=1, keepdim=True)).squeeze(1).tolist()


def frac(v):
    return sum(1 for x in v if x >= 0.23) / max(len(v), 1)


print(f"\n  interfaces scored        {len(iptm_dq)}")
print(f"  ipTM top-1 mean DockQ    {np.mean(iptm_dq):.4f}   frac acceptable {frac(iptm_dq):.4f}")
if mdl_dq:
    print(f"  MODEL top-1 mean DockQ   {np.mean(mdl_dq):.4f}   frac acceptable {frac(mdl_dq):.4f}"
          f"   delta vs ipTM {np.mean(mdl_dq) - np.mean(iptm_dq):+.4f}")
print(f"  random-pick mean DockQ   {np.mean(rnd_dq):.4f}   frac acceptable {frac(rnd_dq):.4f}")
print(f"  oracle     mean DockQ    {np.mean(orc_dq):.4f}   frac acceptable {frac(orc_dq):.4f}")

out = {"n_interfaces": len(iptm_dq), "k": a.k, "seed": a.seed, "side": a.side,
       "ckpt": a.ckpt,
       "model_dockq": float(np.mean(mdl_dq)) if mdl_dq else None,
       "model_frac": frac(mdl_dq) if mdl_dq else None,
       "iptm_dockq": float(np.mean(iptm_dq)), "iptm_frac": frac(iptm_dq),
       "random_dockq": float(np.mean(rnd_dq)), "random_frac": frac(rnd_dq),
       "oracle_dockq": float(np.mean(orc_dq)), "oracle_frac": frac(orc_dq)}
name = f"val_baseline{'' if (a.tag or a.side) == 'val' else '_' + (a.tag or a.side)}.json"
with open(os.path.join(ROOT, a.run, name), "w") as fh:
    json.dump(out, fh, indent=1)
print(f"  -> {os.path.join(a.run, name)}")

if a.expect_oracle is not None:
    got = round(float(np.mean(orc_dq)), 4)
    if abs(got - a.expect_oracle) > 5e-5:
        sys.exit(f"STOP: oracle {got:.4f} != training loop's {a.expect_oracle:.4f}. The candidate "
                 f"sets differ, so this baseline is NOT comparable to the run's val top-1.")
    print(f"  oracle matches the training loop ({got:.4f}): same candidate poses, comparable.")
