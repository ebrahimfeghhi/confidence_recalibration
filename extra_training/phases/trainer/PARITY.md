# The GPU parity ladder

Nothing here runs without an idle card; every script refuses rather than competing with a live
arm. Run them in order — each answers a question the next one depends on.

| | script | question |
|---|---|---|
| 0 | `replay_variance.sh` | how reproducible is **legacy against itself**? Gives the tolerance. |
| 1 | `gpu_smoke.sh` | does the copy reach a forward and a backward at all? |
| 2 | `parity_compare.py` | same inputs, same outputs? Order, PAE, loss. |

## Why step 0 comes first

Row-wise dropout is live at `p_drop 0.25`, and `_xs_prepare` saves and restores the CPU and CUDA
RNG so pass B **redraws** pass A's masks. `phase_k_train.py` is therefore not guaranteed
bit-identical to itself. A tolerance chosen before that is measured is either tighter than the
floor — failing on noise, sending someone after a bug that is not there — or looser, certifying
nothing. `parity_compare.py` refuses to run without `--tol` for this reason.

## Why the fixture is multi-seed

`8yvm`, **15 seeds × 5 poses = 75 poses**, 1,004 tokens (measured 56.81 GB uncheckpointed).

A single-pose fixture would prove wiring and certify **nothing** about the ranking term: `rel` is
mean-centred within an interface *across its seeds*, so with one pose the coefficient is
identically zero and the loss is vacuous. It would pass while testing the thing it exists to
test — the same shape as the zero-init orientation trap, where every added path contributes
exactly zero and a naive step-zero comparison passes without exercising any of it.

## What is NOT covered yet

- no optimiser step (`--loss-only`), so no AdamW moment comparison
- one rank, so no collective, no 3-rank step composition, no resume
- the four injection sites are **not** perturbed, so their wiring is unproven: in `87ja6a` the
  orientation arms are zero-init and site 5 is absent, so they contribute exactly zero and a
  comparison passes without touching them

Those are steps 3 onward and need the owner's approval for more GPU time.
