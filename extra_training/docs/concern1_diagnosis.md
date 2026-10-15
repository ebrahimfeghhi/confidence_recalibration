# Concern 1: one cause confirmed, one claim RETRACTED

Written 2026-10-15 8:40 AM ET. Replaces the previous revision of this file, whose central claim was
wrong. Review of 9cbe93c3 was correct on every point.

## RETRACTED: there is no column-order bug, and no mis-attribution

The previous revision claimed `eval/ost_dockq.py:155` read OST's `dockq_interfaces` columns in the
wrong order, that `results.csv` rows therefore carried other interfaces' DockQ values, and that 454
of 501 structures were exposed. **All of that is withdrawn.** The installed OST 2.8.0 documents the
order explicitly, at `ost/mol/alg/scoring.py:1113`:

    :type: :class:`list` of :class:`tuple` with 4 elements each:
           (trg_ch1, trg_ch2, mdl_ch1, mdl_ch2)

and the same property builds each entry as `(i[0], i[1], flat_mapping[i[0]], flat_mapping[i[1]])`
where `i` is a TARGET interface, so `GetFlatMapping()` is reference -> model. The parser as written
is correct: its keys are reference pairs, and `model_chains` really does hold model chains.

My argument rested on reading `chain_mapping` as model -> reference. It is reference -> model. Under
the correct direction every entry for 9vb4 is consistent and nothing is mis-attributed:

| `dockq_interfaces` entry | target pair | model pair | roles | consistent with `chain_mapping` {A:C, B:D, C:A} |
|---|---|---|---|---|
| `[A,B,C,D]` | A,B | C,D | ag0-ag1 | A->C, B->D, yes |
| `[A,C,C,A]` | A,C | C,A | ag0-H | A->C, C->A, yes |
| `[B,C,D,A]` | B,C | D,A | ag1-H | B->D, C->A, yes |

So `9vb4_BC` = DockQ 0.922 really is the ag1-H interface. **No historical score is shown to be
wrong by this analysis, and the proposed column swap must not be applied** -- it would have
introduced exactly the silent mis-attribution it was meant to fix.

## CONFIRMED: 27 of 31 never-scored pairs are not interfaces at all

Re-measured at OST's 5 A definition as asked, alongside the original 4.5 A. Identical both ways:

| | 4.5 A | 5.0 A |
|---|---|---|
| never-scored pool pairs that ARE real reference interfaces | 4 | 4 |
| never-scored pool pairs with NO reference contact | 27 | 27 |

`phase_g_build_val_jobs.py:344` builds interfaces as "every (antibody chain, antigen chain) pair";
nothing measures contact. 27 of the 31 pairs the pool claims for these 20 structures do not touch in
the reference at either threshold, so having no DockQ for them is correct. That is open review item
**J2**, and the 27 can be removed from the pool on this evidence.

## CONFIRMED: 4 real interfaces are lost, because a target chain goes unmapped

| pair | roles | reference contacts at 5 A |
|---|---|---|
| 9vaw_AB | ag0-H | 185 |
| 9vb4_AD | ag0-L | 164 |
| 9q5v_EJ | ag1-H | 123 |
| 9q5m_EH | ag2-H | 81 |

The mechanism, from the same OST source: `dockq_interfaces` includes a target interface only
`if i[0] in flat_mapping and i[1] in flat_mapping` (`scoring.py:1121`). So an interface is dropped
whenever either of its target chains is absent from the mapping.

For 9vb4 the mapping covers target A, B and C and omits target **D**, the 111-residue light chain,
although model chain B is a 111-residue match at identity 1.00. Chains, as OST loaded them:

| target label | auth | residues | role |
|---|---|---|---|
| A | A | 351 | ag0 |
| B | B | 338 | ag1 |
| C | D1 | 119 | H |
| D | D2 | 111 | L |

A lead, not a conclusion: OST's `chem_groups` for the target are `[[A], [B,D], [C]]` and its
`chem_mapping` is `[[C], [D], [A,B]]` -- model chains A AND B are both placed in the group holding
target C alone, so only one of them can map and target D is left with nothing. H and L are
homologous immunoglobulin V domains, which would explain a chemical grouping that cannot separate
them. Not verified.

The never-scored pairs are not confined to one role, so "the light chain is always dropped" is not
the rule:

| role appearing in a never-scored pair | count |
|---|---|
| L | 18 |
| ag0 | 17 |
| H | 13 |
| ag1 | 10 |
| ag2 | 4 |

## What this does and does not affect

| | affected |
|---|---|
| trunks, poses, folds | no -- all 723 units intact. Folds kept; nothing needs refolding |
| training labels | no -- `phase_j_job_cifs.py` reads the deposited structure and never touches `pool.csv` or `chain_roles.csv` |
| `results.csv` attribution | **no** -- retracted above |
| coverage of `results.csv` | yes, narrowly: 4 real interfaces across 4 structures have no DockQ |
| published FoldBench / PXM numbers | not shown to be wrong. The earlier "unknown, needs checking" framing came from the retracted claim |

## Next steps

| step | why |
|---|---|
| remove the 27 non-contacting pairs from the pool (J2) | they are measured non-interfaces at 4.5 A and 5 A |
| establish why OST leaves a target chain unmapped on 4 structures | the chem_groups/chem_mapping lead above; affects 4 real interfaces |
| re-score those 4 with an explicit `-c` chain map | `ost_dockq.invoke` already accepts `chain_map`, so it can be forced without changing the parser |

---

# Steps 1-3 executed, 2026-10-15 9:30 AM ET

## Step 1: the phantom pairs, measured across all 501 structures

Measured with OST's own criterion, which `scoring.py` sets as `contact_d = 5.0` and
`contact_mode = "aa"` -- all-atom contacts at 5 A, the same test OST applies when deciding what a
target interface is.

| | pairs |
|---|---|
| pool pairs, all 501 training structures | 1,137 |
| with reference contacts (kept) | 1,037 |
| **ZERO reference contacts (phantom)** | **100, over 81 structures** |

Contact distribution of the 1,037 that are real: min 1, p10 38, median 210, max 4,927. The phantom
pairs are not a borderline tail -- they are exactly zero.

Most common role combinations among the 100: ag0-L 30, ag1-L 22, ag1-H 12, L-ag1 8, L-ag0 8.

Written out: `train_val_splits/pool_reference_contacts.csv` (every pair with its contact count),
`train_val_splits/pool_phantom_pairs.csv` (the 100), and `training/fold/pool.contacting.csv` (the
1,037). **The live `pool.csv` is NOT overwritten** -- swapping it in changes the denominator of every
coverage statistic this project has quoted, so it should be a reviewed step rather than a side
effect of this investigation.

### This explains the old run's unsolved cases

`10op_FN` -- flagged early on as "zero samples on every seed" and carried as an open mystery all
session -- is a phantom pair: roles ag1-L, zero reference contacts. It was never a scorer failure.

### Why only 20 structures looked affected

`partial_scores.csv` covers the 74 structures in the regeneration queue, not all 501. The other 61
phantom-pair structures were folded in the earlier run and their results live in
`training/fold/results.csv`. Checked directly: `10op_FN`, `10or_BT`, `10or_FT`, `7stz_AF` and
`7tn9_FH` each have 0 rows there, so nothing was scored for them and there is no contradiction
between the 100 and the 20.

## Steps 2-3: the four real interfaces

`-c` is not a usable lever. OST validates a custom map against its own chemical grouping and refuses
the correct one:

    RuntimeError: Chem group mismatch in custom mapping: target chain "D" groups with the following
    chemically equivalent target chains: ['B', 'D'] but model chain "B" maps to the following target
    chains: ['C']

Model B is 9vb4's 111-residue light chain and target C its 119-residue heavy chain; OST holds them
chemically equivalent, which is what the chem-grouping hypothesis predicted. Scoring the interface's
two chains in isolation puts one chain per group and the mapping comes out right.

| interface | roles | DockQ | pair_iptm | model chains returned | expected | verdict |
|---|---|---|---|---|---|---|
| 9vb4_AD | ag0-L | **0.894** | 0.919 | (C, B) | (C, B) | accepted |
| 9vaw_AB | ag0-H | **0.005** | 0.178 | (C, A) | (C, A) | accepted |
| 9q5v_EJ | ag1-H | -- | -- | -- | -- | REFUSED, ambiguous map |
| 9q5m_EH | ag2-H | -- | -- | -- | -- | REFUSED, ambiguous map |

Step 3 is the "model chains returned vs expected" columns: a recovered number is accepted only when
the model chains OST reports are the ones `chain_roles.csv` names for those roles, so it cannot be
filed under the wrong physical pair.

9vaw_AB at 0.005 is the result that matters. That interface is a near-total failure and it was being
omitted rather than recorded, so dropping unmappable interfaces biases evaluation OPTIMISTICALLY.

The two refusals are references carrying several identical antigen copies -- 9q5m has three
372-residue chains, each matching more than one model chain at 0.82 -- so sequence alone does not
determine a one-to-one map. They need symmetry-aware mapping, which is what OST's chain mapper is
for and what fails here. Refused rather than guessed.

## Scope, restated after the review's qualification

"Training labels unaffected" means only that this scorer issue does not change how they were
constructed. It is NOT the label and cache validation still owed before training:
`phase_j_replay.py` has never been run, and `phase_i_check.py`'s validator still accepts NaN
coordinates and bad trunk shapes.
