# Within-seed pairs, a share weight, and a top-k rank decay: three arms on overfit30 / val10

Everything here is **held out**: val10's 18 interfaces at budget 30, scored by each run's own
shards, with `shipped` derived from the same shards via `stored_iptm` so the comparison is
paired on identical interfaces and identical poses.

Reproduce the table with the script at the bottom. Raw rows:
`training/runs/<rid>/rank_rho_b30_v3.csv` and `training/runs/<rid>/train_eval_shards_ep<N>/`.

## The arms

All three are identical to `24u34d`/`24u342` except where stated: cross-seed softplus ranking
loss, `--alpha-pae 0`, distance **and** orientation injected at all four pairformer blocks with
untied per-block copies, `--train-only 'linear_no_bias_d,pose_perblock,frame_dir'` = 12 tensors /
44,032 parameters, L2-SP at 2.5e-4 over the 8 anchored tensors with `--l2-sp-exclude 'frame_dir'`,
overfit30 train, val10 held out every epoch, 15 epochs.

| arm | rid | `--rank-pairs` | `--rank-within-weight` | `--rank-topk-decay` | `--rank-tau` |
|---|---|---|---|---|---|
| **L8** | `hronsr` | all | **8** | 0 | 0.0270 |
| **L1** (control) | `bjwcm5` | all | **1** | 0 | 0.0270 |
| **L8K2** | `6v6bbf` | all | 8 | **2** | 0.0270 |

`24u342` is the fourth point of reference: the same arm with `--rank-pairs cross` at tau 0.0305,
i.e. no within-seed pairs at all.

**tau is 0.0270 in all three**, the median |ipTM gap| over all 1,146,184 training pairs of both
families. It is deliberately NOT retuned per arm: fixing tau to a property of the data rather
than of the knob is what makes L8 and L1 a one-variable comparison.

## Result

Paired `trained - shipped` on the same 18 interfaces, over epochs 8-15 where the arms had
settled:

| arm | mean paired DockQ | min | max | epochs positive | final frac |
|---|---|---|---|---|---|
| **L8** | **+0.0116** | +0.0084 | +0.0133 | **8 / 8** | 0.5556 (= shipped) |
| **L1** control | **-0.0073** | -0.0117 | -0.0061 | **0 / 8** | **0.6111** (> shipped) |
| **L8K2** | **-0.0219** | -0.0297 | -0.0106 | **0 / 8** | 0.5000 (< shipped) |

shipped on this set: DockQ 0.2909, frac 0.5556.

**The within-seed weighting is what produced L8's gain.** L8 and L1 share tau, share the pair
family, and differ only in L. They separate perfectly by sign -- positive in every one of eight
epochs against negative in every one of eight. The gain is therefore not attributable to tau
0.0270 nor to the mere presence of within-seed pairs, both of which the control also has.

**The top-k rank decay makes things worse, and by more than the weighting made them better.**
-0.0219 against +0.0116, from the same L=8 starting point. L8K2 is also the only arm below
shipped on DockQ *and* frac at once.

## What this does NOT establish

- **No arm is individually significant.** L8 peaks at 1.74 sigma (p ~ 0.08) and its final epoch
  is 1.31 sigma; L1 never exceeds 0.49 sigma; L8K2 reaches 1.29 sigma in the wrong direction.
  The 8/8-vs-0/8 sign split is more persuasive than any single epoch, but the epochs within an
  arm are serially correlated -- each arm contributes roughly **one** converged state, not eight
  observations. Three states is not a significance claim.
- **18 interfaces, and frac moves in steps of 0.0556.** L1's frac gain is ONE interface crossing
  0.23. It held for nine consecutive epochs, which is more persistence than `24u342`'s frac gain
  ever showed (that one flipped six times across 50 epochs), but it is still one interface.
- **L8 and L1 are not ordered versions of each other.** L8 wins DockQ and ties frac; L1 wins frac
  and loses DockQ. L1's standard error is **3x wider** (0.0260 vs 0.0088) on identical
  interfaces, so the control makes large per-interface bets while L8 makes small consistent ones.
  "Which is better" depends on the criterion, and on 18 interfaces the two criteria disagree.
- **L1K2 was never run**, so whether the decay is harmful on its own or only in combination with
  L=8 is untested. The 2x2 is three-quarters complete.
- **The held-out ranking loss plotted beside these is CROSS-SEED at tau 0.0305 for every arm**,
  because `rank_rho.py` computes it that way so figures stay comparable across the project. For
  these runs it is therefore not the validation twin of the objective being trained, and the
  training-vs-held-out curves must not be read as a generalisation gap. Selected DockQ is the
  criterion.

## A measurement error corrected along the way

Three files stated the within-seed median |ipTM gap| as 0.0081 and cross-seed pairs as 3.9x
wider. That 0.0081 came from a 5,400-pair sample taken **before DockQ ties were dropped**, and
ties are exactly the zero-gap pairs that drag a within-seed median down. After the tie filter the
full training set gives **0.0036**, so the ratio is **8.5x**. Reproduce with
`measure_tau.py --pairs within`.

The conclusion drawn from that figure was also wrong, and more consequentially. The claim was
that at the cross-seed tau a within-seed pair "sits at m/tau ~ 0 where softplus cannot tell a
correct pair from an inverted one, so the term is inert". The gradient neither vanishes nor
cancels: **both directions push toward the better-DockQ pose, so they add**. At tau 0.0270 an
overfit30-median within-seed pair (gap 0.0137) weights 0.376 correct against 0.624 inverted. What
a wide tau costs is *saturation* -- the loss never stops pushing pairs it has already ordered and
degenerates toward the linear loss. The original note's "inert" is about the NONLINEARITY, not
the gradient. This is why the fix is a weight and not a second temperature, and the arms above
are the test of that.

## Why the decay was expected to help, and what the runs suggest instead

Selection is argmax ipTM over the group, so the selected pose changes only when some pose crosses
the current top -- the pairs `(top, j)` are the entire selection-relevant subspace. Measured on
valfull at the shipped head they are **1.33% of pairs carrying 1.56% of the loss's weight**, while
ordering them all correctly is identical to the top pose being the best pose, i.e. the full
oracle (DockQ 0.5831 -> 0.6926 over 92 interfaces). Concentrating weight there should therefore
have been nearly free headroom.

It was not. A reading consistent with the runs, **not established by them**: those pairs are
where the head is already most confident and has least left to learn, so starving everything else
removes the signal that was doing the work. Three observations fit -- rho fell BELOW shipped in
L8K2 (no other arm has done that at any epoch), selection froze for four consecutive epochs while
rho kept climbing, and DockQ then oscillated between 0.261 and 0.281 through epoch 14 while L8
and L1 had both pinned to a single value by epoch 11. The oscillation is the predicted hazard of
a weight that depends on the current argmax: the objective changes identity whenever the top pose
flips, and k=2 is evidently not soft enough to prevent it.

## Reproduce

```
cd extra_training
python -I run_logging/measure_tau.py --pairs within            # 0.0036, the corrected figure
python -I run_logging/within_weight.py --tau 0.0270            # the L -> mass-share table
python -I phases/test_rank_loss.py                             # the knobs' invariants
```

The paired table above:

```python
import csv, glob, math, statistics as st
from collections import defaultdict
def paired(rid, ep):
    rows = []
    for f in glob.glob(f'training/runs/{rid}/train_eval_shards_ep{ep}/rows_b30_*.csv'):
        rows += list(csv.DictReader(open(f)))
    by = defaultdict(list)
    for r in rows:
        by[r['iface']].append((float(r['iptm']), float(r['stored_iptm']), float(r['dockq'])))
    dt = [max(ps, key=lambda t: t[0])[2] for ps in by.values()]
    ds = [max(ps, key=lambda t: t[1])[2] for ps in by.values()]
    d = [a - b for a, b in zip(dt, ds)]
    return sum(d) / len(d), st.stdev(d) / math.sqrt(len(d))
for rid in ('hronsr', 'bjwcm5', '6v6bbf'):
    ms = [paired(rid, e)[0] for e in range(8, 16)
          if glob.glob(f'training/runs/{rid}/train_eval_shards_ep{e}/rows_b30_*.csv')]
    print(rid, f'{sum(ms)/len(ms):+.4f}', f'positive in {sum(1 for m in ms if m>0)}/{len(ms)}')
```
