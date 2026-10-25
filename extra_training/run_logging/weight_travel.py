#!/usr/bin/env python
"""Is the optimiser descending, or oscillating in place? Measured from a run's own checkpoints.

    python -I run_logging/weight_travel.py <rid> [--lambda 5e-05]

Three things, all CPU, all from head_ep*_step*.pt against l2sp_anchor.pt:

  ||W-W0||            distance from the shipped weights, per epoch. Flat = parked.
  2*lambda*||W-W0||   the L2-SP gradient magnitude, to compare against the task gradient norm in
                      trajectory.csv. This is how you tell whether the anchor is actually binding
                      instead of assuming it from lambda alone.
  straightness        net displacement / total path length over the epochs. 1.0 is a straight
                      descent; 0 is pure diffusion. Compare the net against sqrt(sum of squared
                      steps), which is what a free random walk of the same step sizes would give:
                      coming in BELOW that is confinement, i.e. oscillation around a point.

WHAT IT SETTLED ON 8fkrsu. The anchor is not the constraint -- its gradient is 2.1% of the median
task gradient (5.55e-04 against 2.6e-02), so lowering lambda cannot help. The weights travelled
25.5 in 6 epochs to end up 5.78 from where they started, against 10.5 for a free random walk of the
same steps. Sub-diffusive, and ||W-W0|| sat at 5.55-5.59 for epochs 5-7 while gradient norms held
at 0.026 and the training loss held at 0.1402-0.1408. That is an optimiser oscillating at the limit
its learning rate allows, not one that has run out of signal. lr is constant 0.01 after a 20-step
warmup -- phase_k_train.py has no decay schedule at all.
"""
import argparse, glob, math, re
import torch

ap = argparse.ArgumentParser()
ap.add_argument("rid")
ap.add_argument("--lam", type=float, default=None, help="defaults to the run's own --l2-sp")
a = ap.parse_args()

root = f"training/runs/{a.rid}"
anc = torch.load(f"{root}/l2sp_anchor.pt", map_location="cpu")
keys = list(anc.keys())
w0 = torch.cat([anc[k].float().flatten() for k in keys])

lam = a.lam
if lam is None:
    import json
    lam = float(json.load(open(f"{root}/args.json")).get("l2_sp") or 0.0)


def vec(p):
    sd = torch.load(p, map_location="cpu")
    sd = sd.get("model") or sd.get("state_dict") or sd
    return torch.cat([sd[k].float().flatten() for k in keys])


W = {}
for c in glob.glob(f"{root}/head_ep*_step*.pt"):
    W[int(re.search(r"_ep(\d+)_", c).group(1))] = vec(c)
if not W:
    raise SystemExit(f"no head_ep*_step*.pt in {root}")
eps = sorted(W)

print(f"{a.rid}: {len(keys)} anchored tensors, {w0.numel()} params, lambda {lam:g}")
print("epoch   ||W-W0||   2*lambda*||W-W0||   move from prev")
prev = None
for e in eps:
    r = float((W[e] - w0).norm())
    mv = f"{float((W[e]-W[prev]).norm()):8.4f}" if prev is not None else "       -"
    print(f"  {e:<5d}  {r:8.4f}      {2*lam*r:.3e}        {mv}")
    prev = e

steps = [float((W[b] - W[a_]).norm()) for a_, b in zip(eps, eps[1:])]
path = sum(steps)
net = float((W[eps[-1]] - W[eps[0]]).norm())
rw = math.sqrt(sum(s * s for s in steps))
print(f"\nepochs {eps[0]}..{eps[-1]}: path {path:.3f}, net {net:.3f}, "
      f"straightness {net/path:.3f}")
print(f"  free random walk of the same steps would give net {rw:.3f} "
      f"-> {'CONFINED (oscillating)' if net < rw else 'at or above diffusive'}")
