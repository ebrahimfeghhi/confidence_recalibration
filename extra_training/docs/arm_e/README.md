# ARM E — Arm B + the live pair-track LayerNorms + `linear_no_bias_pae`

Run `73rf4v`, 20 epochs, started 2026-11-18 4:05 PM EST, ~8 min/epoch on cards 4,5,7.
Reference arm for every figure is Arm B (`w57hjo`). **Results here are through epoch 14;
the epoch-20 val is what decides Arm F.**

## What the arm is

Arm B unchanged — six pose-injection sites, `pae_ln` unfrozen, PAE-only loss, pair-dropout 0.25,
`--pae-split-accept 0.23` with `--pae-balance-accept 0` — plus two additions:

1. **the gain and bias of every LayerNorm on the pair track inside the four pairformer blocks**,
   named explicitly rather than by pattern. The single-track norms are excluded because
   `layernorm_z` only feeds the single-track attention bias. **The norms belonging to the ten
   provably dead pair modules are also excluded**, at the user's direction — `--lean-single`
   independently agrees, dropping 0 of 53 as gradient-free.
2. **`linear_no_bias_pae`**, the 256 -> 64 map into the error bins. PDE, pLDDT and resolved stay
   frozen.

**53 trainable tensors, 91,136 of 464.5M parameters.** Against Arm B that is +33 tensors and
+24,576 parameters, a 37% increase over Arm B's 66,560.

`check_unfreeze.py` on ep1 vs ep2: **53 of 53 selected tensors moved, 0 unselected tensors moved**
— "exactly the selected set trained".

## Results, epoch 10

Mean-pooled ipTM in parentheses; shipped is max-pooled, which is its own convention.

| HELD-OUT val (n=141) | mean DockQ | frac acceptable | rho |
|---|---|---|---|
| shipped | 0.5812 | 72.3% | 0.1882 |
| Arm E ep10 | 0.5806 (0.5937) | 72.3% (73.0%) | 0.2075 (**0.2232**) |
| Arm B ep10 | 0.5814 (**0.5940**) | 72.3% (73.0%) | 0.2083 (0.2166) |

| IN-SAMPLE (n=280) | mean DockQ | frac acceptable | rho |
|---|---|---|---|
| shipped | 0.2505 | 30.7% | 0.0819 |
| Arm E ep10 | 0.2401 (0.2498) | 28.9% (30.7%) | 0.1188 (**0.1606**) |
| Arm B ep10 | — (0.2471) | — (30.4%) | — (0.1577) |

**The extra 33 tensors bought rho and nothing in DockQ.** 0.2232 is the best held-out rho any arm
has posted, and val mean DockQ is 0.5937 against Arm B's 0.5940 — 0.0003, well under one
interface changing hands on 141. The in-sample-below-shipped sign flip recorded for Arms A and B
reproduces here.

## THE RESULT THAT MATTERS: a much lower loss floor, the SAME turn-up epoch

Arm E's training loss is below both Arm B and the rank-8 LoRA arm at every matched epoch, by far
more than LoRA ever bought:

| ep | Arm E good | Arm B good | LoRA8 good | Arm E bad |
|---|---|---|---|---|
| 1 | 3.2432 | 3.3017 | 3.2952 | 2.7780 |
| 5 | 3.0968 | 3.1519 | 3.1411 | 2.6400 |
| 10 | 3.0460 | 3.1076 | 3.0988 | 2.5856 |
| 12 | **3.0397** | **3.1010** | — | 2.5659 |
| 13 | 3.0423 | 3.1046 | — | 2.5719 |
| 14 | 3.0413 | 3.1048 | — | 2.5723 |

Rank 8 bought ~0.009 of good-pose loss over Arm B for +30,720 parameters. Arm E buys **0.061** for
+24,576 — roughly seven times the return per parameter, and the norms are where it came from.

**And the good-pose curve still bottoms at epoch 12 and turns up.** Same epoch as Arm B
(3.1010 at 12) and as the rank-8 LoRA arm. The floor moved down a long way; the shape did not
move at all.

That makes **three** explanations for the turn-up now ruled out by a matched arm:

1. **learning rate** — the cosine fork `32khid` is identical to Arm B to four decimals.
2. **LoRA capacity on `blocks.3.pair_transition`** — rank 8 bottoms at the same epoch 12.
3. **capacity in the pair-track norms** — Arm E lowers the floor by 0.061 and bottoms at
   epoch 12 regardless.

A lower floor that does not move the turn-up, and 0.061 of training loss that does not move val
DockQ, are the same observation twice: the objective and the metric are only loosely coupled here.

## WHAT TO CHALLENGE

1. Three arms now bottom at epoch 12 with floors spread over 0.06. What is special about epoch 12
   that is not capacity and not the step size? Data order is the remaining shared factor — every
   arm uses `--seed 0` and the same epoch balance.
2. rho improved to 0.2232 while DockQ did not move. The project has now recorded this decoupling
   six times. Is rho worth optimising at all, if selection never follows?
3. `--pae-balance-accept` is **0** on this arm, inherited from Arm B so the A/B is clean. It
   remains the one obvious lever never tested on any arm.
4. Is 0.0003 of val mean DockQ (Arm E vs Arm B) a basis for choosing Arm F's base at all? Both
   sit inside the noise of one interface on 141, so the choice is effectively arbitrary between
   them, and the rho difference is the only thing that distinguishes them.
