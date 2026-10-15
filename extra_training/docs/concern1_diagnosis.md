# Concern 1 diagnosed: two causes, one of them a scorer bug

Written 2026-10-15 8:10 AM ET, after review of a4f65258. Supersedes that commit's CONCERN 1
section, which was wrong in two ways recorded at the bottom.

## Representative case: 9vb4

`chain_roles.csv` and sequence identity agree on the mapping, so the role assignment is correct:

| model chain | best reference match | identity | role | reference label |
|---|---|---|---|---|
| A (120 res) | auth D1 | 1.00 | H | C |
| B (111) | auth D2 | 1.00 | L | D |
| C (354) | auth A | 1.00 | ag0 | A |
| D (358) | auth B | 0.97 (model carries an N-term His tag) | ag1 | B |

Measured heavy-atom contacts under 4.5 A in the reference, against what the pool claims:

| pool pair | roles | reference contacts | row written? |
|---|---|---|---|
| A,C | ag0-H | 84 | yes, DockQ 0.945 |
| B,C | ag1-H | 88 | yes, DockQ 0.922 -- **but see below** |
| A,D | ag0-L | 94 | **no** |
| B,D | ag1-L | **0** | no -- correct, there is no interface |

## Cause 1: the pool is a cross product, not measured contacts

`phase_g_build_val_jobs.py:344` -- "Interfaces are every (antibody chain, antigen chain) pair ...
antigen against each antibody chain". Nothing measures whether the pair touches.

Across all 31 never-scored pool pairs in the 20 affected structures:

| | pairs |
|---|---|
| zero reference contacts, correctly unscoreable | **27** |
| a real reference interface that was lost | **4** |

The 27 are not a defect in the folds or the scorer; they are interfaces the pool should never have
claimed. This is open review item **J2**. The four that are real:

| pair | roles | reference contacts |
|---|---|---|
| 9vaw_AB | ag0-H | 101 |
| 9vb4_AD | ag0-L | 94 |
| 9q5v_EJ | ag1-H | 64 |
| 9q5m_EH | ag2-H | 45 |

## Cause 2: `ost_dockq.parse` reads OST's interface columns in the wrong order

`eval/ost_dockq.py:155` keys the result by `t[0],t[1]` and records `t[2],t[3]` as `model_chains`:

    ref_pair = f"{t[0]},{t[1]}"
    res[ref_pair] = {"dockq": ..., "model_chains": (t[2], t[3])}

The columns are the other way round. OST's own `chain_mapping` is model -> reference and for 9vb4 is
`{A: C, B: D, C: A}`, with model D unmapped. Its `dockq_interfaces` are
`[[A,B,C,D], [A,C,C,A], [B,C,D,A]]`:

| entry | read as model-first | read as reference-first |
|---|---|---|
| `[B,C,D,A]` | model B->D and C->A, both exactly `chain_mapping` | requires model D->ref B, which `chain_mapping` does not contain |

So `t[0],t[1]` are MODEL chains. The parsed keys for 9vb4 are `A,B` `A,C` `B,C`, which are model
pairs, while `pool.csv` queries reference labels. Consequences:

| pool query (reference labels) | matched result key (model letters) | physical pair asked for | physical pair returned |
|---|---|---|---|
| A,C = ag0-H | A,C = H-ag0 | {ag0,H} | {ag0,H} -- correct by coincidence |
| B,C = ag1-H | B,C = L-ag0 | {ag1,H} | **{L,ag0} -- wrong interface** |
| A,D = ag0-L | none | {ag0,L} | nothing, though its value sits in the B,C row |
| B,D = ag1-L | none | {ag1,L} | nothing, correctly |

So `9vb4_BC` in `results.csv` carries DockQ 0.922 labelled ag1-H when that number is the ag0-L
interface, and the genuine ag1-H interface (88 contacts) was never scored at all, because OST left
model chain D unmapped.

A lookup succeeds only when the pool's letter pair happens to denote the same physical pair as the
identical model letter pair. Scope of the risk:

| | structures |
|---|---|
| model letter == reference label for every chain (lookup safe) | 47 |
| they differ (lookup can match the wrong interface) | **454** |
| total training structures | 501 |

Only 20 of the 454 showed MISSING rows. The other 434 got a row for every pool interface, which is
not evidence that the rows are right -- a wrong match is silent.

## What this does and does not affect

| | affected |
|---|---|
| trunks, poses, folds | **no** -- all 723 units intact; 9vb4 logs N_asym 4, N_token 705, a 40 s forward pass |
| training labels | **no** -- built by `phase_j_job_cifs.py` from the deposited structure, an independent path that never reads `pool.csv` or `chain_roles.csv` |
| DockQ and ipTM in `results.csv` | **YES**, wherever model and reference naming differ |
| the already-published FoldBench / PXM numbers | **UNKNOWN** -- those datasets' `chain_roles.csv` has a different schema (`chain_1`/`chain_2`, no model/reference columns), so the test above does not apply to them. Needs its own check before any of those numbers are relied on again. |

`pair_iptm` is indexed off `model_chains`, which under the correct reading holds the REFERENCE
chains, so ipTM attribution is suspect by the same mechanism.

## Proposed fix, NOT applied

Swap the columns in `parse()` so the key is the reference pair `t[2],t[3]` and `model_chains` is
`t[0],t[1]`, then re-derive every affected number. This is deliberately left unapplied: it changes
the attribution of every DockQ value this scorer has produced, including numbers already in the
report, so it should be a reviewed decision rather than an overnight edit. The folds are preserved
and nothing needs refolding -- the poses are correct, only the bookkeeping over them is wrong.

Separately, J2 would remove the 27 phantom pairs by building the pool from measured contacts.

## Corrections to a4f65258

| claim there | correct |
|---|---|
| "folder.py keeps the claim by design" after SCORING INCOMPLETE | WRONG. `folder.py` calls `release()` on that path. The unit is retryable; the new completion logic treats the logged failure as terminal without proving it is unrecoverable. |
| "evaluation-only ... the folds are not at fault" | NOT ESTABLISHED THEN, and only half right now. Folds and labels are indeed unaffected, but 4 real interfaces were lost and an unknown number of rows are mis-attributed, which is more than a missing-value problem. |
| "9vaw scored nothing of its two" | one seed's row read as the structure's. 9vaw seed 500002 is short only `AC`. |
| equal chain counts cited as evidence the right copies were selected | they are not evidence. The mapping was confirmed here by per-chain sequence identity and by measured reference contacts. |
