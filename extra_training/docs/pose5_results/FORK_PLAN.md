# Adding the post-LayerNorm site by forking the five-site run at epoch 40

Proposal under review. **Nothing has been launched and the live arm is untouched.**

## The design

Fork `jyz3g2`'s `resume_ep40.pt`, add `--pose-readout-post-zero` (a second readout injection AFTER
`pae_ln`, through its own zero-init matrices), keep lr 1e-4 and every other setting, and replay
epochs 41-50 with the same data order. The five-site run's val passes at 42, 44, 46, 48 and 50 are
already recorded in `jyz3g2_val_slice.csv`, so the comparison is against the same held-out
interfaces at the same epochs from a shared parent -- a matched A/B for a sixth of the compute of
a fresh 60-epoch arm.

What the sixth site buys, in principle: the pre-norm site can only ROTATE the pair feature,
because the layernorm divides its magnitude back out; a post-norm site can SCALE the readout. They
are two independent sets of weights, so neither is the other at a different scale.

## Why `--resume` alone cannot do it, verified in the code

- `phase_k_train.py:4593` raises `SystemExit` when any trainable tensor is missing from the
  bundle. The five-site bundle has 15 tensors and no `pose_readout_post_*`, so adding the flag
  hard-stops before the first step.
- `phase_k_train.py:4604` calls `opt.load_state_dict(_b["opt"])` directly, which would put 15
  saved param slots against 18 live ones.

## Why appending is safe here, and the assertion that enforces it

AdamW state is keyed by integer position in `model.named_parameters()` order
(`phase_k_train.py:4151` builds the optimiser from that order, not a sorted one). Appending is
therefore only correct if the new tensors land AFTER all fifteen existing ones -- otherwise every
restored moment silently attaches to the wrong weight.

Checked on the real bundle: its head dict is in `named_parameters()` order and each optimiser
moment matches its tensor by shape, with `pose_readout_*` last at indices 12, 13, 14.
`pose_readout.py:128-131` registers the three `post_*` modules immediately after
`pose_readout_frame_dir`, so they should be 15, 16, 17.

`run_logging/fork_add_postnorm_site.py` does not trust that. It takes `--order`, the six-site
model's trainable parameter list as the trainer enumerates it, and refuses to write unless the
first fifteen names are exactly the bundle's order and the remaining three are exactly the
expected `post_*` names.

## What the fork preserves

| | |
|---|---|
| all 15 existing weights | copied unchanged |
| all 15 AdamW moments | copied unchanged, same indices |
| step counter | 6120, unchanged |
| the 3 new tensors | zero, and NO optimiser state -- AdamW creates their moments at their first step |

## The pre-flight that has to pass first

The three new matrices are zero, so **the forked head must reproduce the parent's PAE logits
exactly** on the same poses before any update. If it does not, the post-norm site is not actually
zero-init in this build and nothing measured afterwards is interpretable. Run both heads
`--loss-only` over the same pool and compare, and only then start epoch 41.

## What is NOT yet done

- The `--order` file has not been produced; that needs the six-site model built once, which has
  not been run.
- The pre-flight logits comparison has not been run.
- No training has been launched.
