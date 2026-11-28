# 5-site PAE-only: lr 1e-4 against lr 1e-3

Two arms, identical except the learning rate, run to answer two questions: **is the ranking loss
necessary** (compare either against the rel arm `jyz3g2`) and **is lr 1e-4 underfitting**.

| | |
|---|---|
| arms | `k5x24w` lr 1e-4 (complete, 20 epochs), `4yu3s4` lr 1e-3 (live, at 11) |
| loss | PAE only -- no `--rank-loss`, `--rank-form` or `--rel-labels` in the argv |
| sites | 5: per-block + copy + frame-direction, readout prenorm + copy-distance, NO post-zero |
| trainable | 15 of 261 head tensors, 55,040 of 19,797,070 parameters, rest frozen |
| dropout | pair-dropout 0.25 |
| PAE balance | **OFF** (`pae_balance_accept 0`), deliberately, so this isolates the lr. Split stays 0.23 because it is a TALLY (`phase_k_train.py:6445`), not the objective |
| schedule | 20 epochs, val and slice at 10 and 20 |

Shipped: val 0.5819 / 72.3% / rho 0.1882; slice 30.7% / 0.2505.

## HELD OUT, AT THE MATCHED EPOCH 10

| | lr 1e-4 | **lr 1e-3** |
|---|---|---|
| max-pooled q | 0.5792 | **0.5819** (= shipped) |
| max-pooled top-1 | 72.3% | 72.3% |
| max-pooled rho | 0.2015 | **0.2118** |
| mean-pooled q | 0.5835 | **0.5853** |
| mean-pooled top-1 | 70.2% | **72.3%** |
| mean-pooled rho | 0.1906 | **0.2141** |

1e-3 is ahead on every column at the matched epoch. Two things worth noting rather than burying:
this is the FIRST arm in this project where mean-pooled rho exceeds max-pooled (0.2141 against
0.2118) -- on the rel arm `jyz3g2` max won rho at all 27 passes -- and it is the first where
mean-pooled top-1 reaches parity at epoch 10. It is also ONE PASS.

## IN SAMPLE, EPOCH 10

| | lr 1e-4 | lr 1e-3 |
|---|---|---|
| max frac acc | 29.3% / 0.2407 | 26.8% / 0.2355 |
| mean frac acc | 29.3% / 0.2398 | 29.3% / 0.2436 |
| max rho | 0.0470 | **0.0984** |
| mean rho | 0.0510 | **0.1396** |

In-sample rho is roughly DOUBLE at 1e-3 while max-pooled frac acceptable is LOWER (26.8% against
29.3%). So the higher rate learns the in-sample ordering much faster without converting it into
more acceptable picks -- the same ordering-without-selection disconnect the rel arm showed held
out. Both arms stay below shipped's 30.7% in sample.

## TRAINING LOSS (group_loss.csv, figures/paeonly_group_loss.png)

Per-epoch PAE loss split by pose quality, plus a total over ALL poses.

THE SPLIT DROPS ~6% OF POSES AND THE TOTAL DOES NOT. A pose has ONE PAE loss -- one crop, one
forward, one number -- while DockQ is per INTERFACE, and a structure carries several interfaces
sharing one crop. A pose acceptable for one interface and not another therefore has one loss and
two contradictory labels, so the tally puts it in neither group: 2,200 of 34,620 poses at epoch 10
"STRADDLED the threshold". The `pae_loss_all` column has no labels in it, so nothing is dropped,
and it is computed across ALL THREE rank files -- `trajectory.csv` is byte-identical to
`trajectory_rank0.csv` and reading it alone measures a third of the poses.

lr 1e-4 over 20 epochs: good 3.7263 -> 3.1668, bad 2.9003 -> 2.8052, all 3.4153 -> 3.0463.
lr 1e-3 over 10 epochs: good 3.4111 -> 3.1217, bad 2.9323 -> 2.7084, all 3.2436 -> 2.9816.

The good/bad asymmetry is lopsided at 1e-4 (-0.560 against -0.095) and comparable at 1e-3 (-0.289
against -0.224). Those spans are not matched -- 20 epochs against 10 -- so they are reported as
what each arm did, not as a comparison.

## What to challenge

1. Epoch 10 is one pass per arm and the metrics are within a few thousandths on q. Is "1e-3 is
   ahead on every column" distinguishable from noise with n=1 per arm?
2. The `oowhyw` precedent cuts against it: PAE-only at lr 1e-3 with 113 trainable tensors was the
   hardest-overfitting arm on record (49.6% in-sample by epoch 50 while held-out q fell 0.5873 ->
   0.5591). This arm has 15 trainable tensors, so far less capacity to memorise -- but epoch 20 is
   the first point that could show the turn.
3. In-sample rho doubles while in-sample frac acceptable falls. Does that favour a higher rate, or
   is it the start of the same ordering-without-selection ceiling?
