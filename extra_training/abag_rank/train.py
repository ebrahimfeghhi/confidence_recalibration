#!/usr/bin/env python
"""Train ABAG-Rank's Deep Set on our interfaces. Their model and their losses, our data.

THE MODEL IS IMPORTED, NOT REIMPLEMENTED. src/models/deep_set.py is the scientific content; a
reimplementation would be a second thing to get wrong for no gain. The three losses are ported
from their train.py for the same reason they are worth having:

  smooth-L1 on the logit label   get the VALUE right, per pose
  soft Spearman (1 - rho)        get the ORDER right, across a complex's K poses
  distance preservation          get the SPACING right, so predictions cannot collapse

THE RANKING TERM IS WHY THIS EXPERIMENT EXISTS. Every confidence-head arm we ran optimises a
per-pair PAE classification two steps removed from "which pose is best", and the best of them
moved selected DockQ by +0.0024 on valfull, about 2% of headroom. This optimises pose ORDER
directly.

ONE LIMIT WORTH KNOWING BEFORE READING THE RESULT: Spearman over all K poses is not a top-1
objective. A model can improve rho by sorting out the middle of the ranking and gain nothing on
the metric, which is exactly what our arms did -- rho 0.2301 -> ~0.255, worth ~+0.024 DockQ on a
simulated ranker, delivered +0.0024. If this model improves rho but not selected DockQ, that is
the same failure again, not a new one.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/abag_rank_upstream/ABAG-Rank")

import features as FT                                    # noqa: E402
from dataset import InterfacePoses, collate              # noqa: E402
from src.models.deep_set import DeepSet                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def soft_rank(x, tau):
    """Differentiable rank: how confidently each item beats each other, summed. Upstream's."""
    d = (x.unsqueeze(-1) - x.unsqueeze(-2)) / tau
    return torch.sigmoid(d).sum(-1)


def spearman_loss(pred, target, tau=0.05):
    """1 - soft Spearman, per complex, IN PROBABILITY SPACE.

    Upstream calls this with pred_sig/true_sig -- sigmoids of the logits -- not the logits
    themselves, and its training path uses tau 0.02, not 0.2. Both matter: logit gaps are far
    larger than DockQ gaps near 0 and 1, so running the soft rank on logits changes which pairs
    the sigmoid treats as "close enough to swap", and a 10x temperature changes it again. Caught
    in review.
    """
    K = pred.size(-1)
    if K < 2:
        return torch.zeros(pred.size(0), device=pred.device)
    mse = (soft_rank(pred, tau) - soft_rank(target, tau)).pow(2).sum(-1)
    rho = 1 - 6 * mse / (K * (K ** 2 - 1))
    return 1.0 - rho.clamp(-1 + 1e-6, 1 - 1e-6)


def distance_loss(pred, target):
    """Pairwise gaps among a complex's poses must match, SUMMED over unique pairs.

    Upstream sums the squared errors over the unique upper-triangular pose pairs; averaging over
    all K x K entries instead divides by K^2 and counts the zero diagonal, which at K=10 makes
    this term ~45x smaller than upstream's for the same errors. Combined with running it on
    logits rather than probabilities, the objective was materially different. Caught in review.
    """
    K = pred.size(-1)
    if K < 2:
        return torch.zeros(pred.size(0), device=pred.device)
    dp = (pred.unsqueeze(2) - pred.unsqueeze(1)).abs()
    dt = (target.unsqueeze(2) - target.unsqueeze(1)).abs()
    iu = torch.triu_indices(K, K, offset=1, device=pred.device)
    return (dp - dt).pow(2)[:, iu[0], iu[1]].sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="training/abag/pae_train")
    ap.add_argument("--restypes", default="training/abag/restypes")
    ap.add_argument("--pool", default="training/full_train/pool_insample.csv")
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--out", default="training/abag/run1")
    ap.add_argument("--k", type=int, default=10, help="poses per complex per step")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--smooth-l1-beta", type=float, default=0.5)
    ap.add_argument("--spearman-tau", type=float, default=0.05,
                    help="the EFFECTIVE upstream value: configs/losses/one_minus_rho.yaml sets "
                         "spearman_tau 0.05, which the config merge applies over train.py's own "
                         "0.02 fallback. Reading the function default alone gives the wrong "
                         "number. Caught in review.")
    ap.add_argument("--w-rank", type=float, default=1.0)
    ap.add_argument("--w-dist", type=float, default=0.1,
                    help="upstream's distance_lambda fallback (train.py:423), not 1.0")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="interfaces held out of TRAINING for model selection. valfull is never "
                         "touched here -- it is the external benchmark and selecting on it would "
                         "make every number in paper_results.md incomparable.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    ds = InterfacePoses(os.path.join(ROOT, a.pool), os.path.join(ROOT, a.dump),
                        os.path.join(ROOT, a.restypes), os.path.join(ROOT, a.results),
                        k=a.k, seed=a.seed)
    print(f"  {len(ds)} interface(s) with poses and labels")
    if not len(ds):
        raise SystemExit("STOP: no usable interfaces. Has the dump finished and been merged?")

    # SPLIT BY STRUCTURE, NOT BY INTERFACE. Two interfaces of one pdb share a trunk and most of
    # their poses, so splitting by interface would leak a structure across the boundary and make
    # the internal validation optimistic.
    pdbs = sorted({it["pdb"] for it in ds.items})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(pdbs)
    nval = max(1, int(len(pdbs) * a.val_frac))
    val_pdbs = set(pdbs[:nval])
    tr_idx = [i for i, it in enumerate(ds.items) if it["pdb"] not in val_pdbs]
    va_idx = [i for i, it in enumerate(ds.items) if it["pdb"] in val_pdbs]
    print(f"  internal split: {len(tr_idx)} train / {len(va_idx)} val interface(s) "
          f"over {len(pdbs) - nval}/{nval} structure(s)")

    dl = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=a.batch, shuffle=True,
                    collate_fn=collate, num_workers=8, drop_last=True)
    dv = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=a.batch, shuffle=False,
                    collate_fn=collate, num_workers=4)

    dev = a.device if torch.cuda.is_available() else "cpu"
    model = DeepSet(input_dim=FT.FEATURE_DIM, phi_hidden_dims=[512, 256, 128, 256],
                    rho_hidden_dims=[2048, 1024, 512, 256, 128], output_dim=1,
                    aggregator="attn_pool_concat_stats", sample_level_dim=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters on {dev}")

    best = -float("inf")   # selecting on top-1 DockQ, higher is better
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = n = 0.0
        for b in dl:
            x, lens, y = b["x"].to(dev), b["lengths"].to(dev), b["y"].to(dev)
            pred = model(x, lens, b["sample_features"].to(dev)).squeeze(-1)
            l_val = F.smooth_l1_loss(pred, y, beta=a.smooth_l1_beta)
            # PROBABILITY SPACE for the two group losses, and upstream's relevance weighting:
            # clamp(std(true_sig)/0.07, 0, 1) down-weights complexes whose poses are all much of
            # a muchness, where an ordering is noise. Applying them uniformly, on logits, was a
            # different objective wearing the same name.
            ps, ts = torch.sigmoid(pred), torch.sigmoid(y)
            # UNBIASED std, as upstream's torch default is -- not unbiased=False.
            relw = (ts.std(dim=-1) / 0.07).clamp(0.0, 1.0)
            l_rank = (spearman_loss(ps, ts, tau=a.spearman_tau) * relw).mean()
            l_dist = (distance_loss(ps, ts) * relw).mean()
            loss = l_val + a.w_rank * l_rank + a.w_dist * l_dist
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        model.eval()
        vt = vn = 0.0
        rhos, top1, orc = [], [], []
        with torch.no_grad():
            for b in dv:
                x, lens, y = b["x"].to(dev), b["lengths"].to(dev), b["y"].to(dev)
                pred = model(x, lens, b["sample_features"].to(dev)).squeeze(-1)
                vt += float(F.smooth_l1_loss(pred, y, beta=a.smooth_l1_beta)) * len(y)
                vn += len(y)
                rhos.append(float(1.0 - spearman_loss(torch.sigmoid(pred), torch.sigmoid(y),
                                                      tau=a.spearman_tau).mean()))
                # TOP-1 IS THE CRITERION THE PROJECT IS JUDGED ON. Selecting checkpoints on
                # regression loss would repeat the mistake this whole project has measured: a
                # better loss, and a better rank correlation, need not improve the single pick.
                # Our own arms moved rho 0.2301 -> ~0.255 and selected DockQ by +0.0024. So the
                # best checkpoint is the one that PICKS best, and the val DockQ here is recovered
                # from the logit label by the inverse of features.logit_label.
                dq = torch.sigmoid(y)
                sel = dq.gather(1, pred.argmax(dim=1, keepdim=True)).squeeze(1)
                top1 += sel.tolist()
                orc += dq.max(dim=1).values.tolist()
        vl = vt / max(vn, 1)
        t1 = float(np.mean(top1)) if top1 else 0.0
        print(f"  epoch {ep:3d}  train {tot / max(n, 1):.4f}  val {vl:.4f}  "
              f"soft-rho {np.mean(rhos):+.4f}  val top-1 DockQ {t1:.4f} "
              f"(oracle {np.mean(orc):.4f})", flush=True)
        if t1 > best:
            best = t1
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": vl,
                        "val_top1": t1, "feature_dim": FT.FEATURE_DIM, "sample_level_dim": 1},
                       os.path.join(ROOT, a.out, "best.pt"))
    print(f"  best internal val top-1 DockQ {best:.4f} -> {os.path.join(a.out, 'best.pt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
