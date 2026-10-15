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
