# Label audit: our RCSB-derived labels vs SAbDab2's independent coordinates

Reproduce both audits with `phases/phase_j_labels_and_audit.sh both`; prove the audit can fail with
`phases/phase_j_label_audit_selftest.py`. Per-structure reports in
`train_val_splits/label_audit_{training,validation}.csv`.

> **REVISED 2026-10-15 12:40 PM ET.** The pairing was rewritten: solved jointly per structure and, for
> a fused chain, by concatenation. Verified counts rose from 481/501 to **497/501** (training) and
> 48/53 to **52/53** (validation), with no role left unaudited in either set. Earlier headline
> figures on this page (496/501, 51/53, "0 residue mismatches over 2.03M atoms", then 481/501 and
> 48/53) are superseded by the table below.

## Verdict, 2026-10-15 12:40 PM ET

| verdict | training (n=501) | validation (n=53) |
|---|---|---|
| **ok** -- every role audited, no substitution, max deviation 0.0000 A | **497** | **52** |
| COORD MISMATCH -- sequences exact, coordinates differ | 4 | 1 |
| INCOMPLETE -- a role could not be paired | 0 | 0 |
| SUBSTITUTION -- a residue identity differs | 0 | 0 |

Columns: `training`/`validation` count STRUCTURES, not chains or residues. A structure is `ok` only
when every role it declares was paired and compared; see "Three holes" below for why that is stated.

## What the audit had to solve: chain-to-chain correspondence

Both files hold the same molecule and divide it into chains differently. Every structure left
unverified before this revision failed for one of three reasons, none of them a label defect.

| reason | measured example |
|---|---|
| the two sources disagree on whether an scFv is one chain or two, in BOTH directions | `32nz` our label A=122 + B=115, SAbDab2 A=238 fused; `10fj` our label A=220 fused, SAbDab2 B=120 + C=113 |
| `chain_roles` records a chain id absent from SAbDab2's file, or one carrying a copy suffix (`B1`/`B2`) | pairing by name fails; pairing is by sequence |
| the crystal holds two copies of the complex, and the two files took different copies | `8dpl` every chain fits at 0.000-0.020 A alone, 28.5 A pooled |

Two changes, neither introducing a tolerance:

| change | why | measured effect |
|---|---|---|
| pair all roles at once (`scipy.optimize.linear_sum_assignment`) instead of per role | a per-role choice let one role take a chain another needed: on `9vb4` role H scored best against the ANTIGEN chain C (33/351) and claimed it, leaving both antibody roles unaudited | 481 -> 494 training |
| compare roles GROUPED BY REFERENCE CHAIN, concatenating our chains when they share one | the claim "these two chains are the pieces of that one" is tested directly, with the same matcher and acceptance test as any one-to-one chain | 494 -> 497 training, 50 -> 52 validation |

### An approach that was abandoned, and why

Slicing the fused chain to one role's window needed a register tolerance to discard blocks in the
homologous other domain. That constant had no principled value and had to be hand-tuned per
structure: 30 passed `29tj` and failed `32nz` (122 matched over a 144-wide span); 8 passed `32nz`.
Forcing the window to the label chain's length instead invented substitutions at the boundary --
`29tj` H 142-144 reported ASP/ILE/VAL against ALA/ALA/SER. Both failures came from the slice, not the
data. Concatenation needs no such constant and is what ships.

## The one threshold, and evidence it is not load-bearing

`MIN_COVER = 0.95` is the single threshold in `phase_j_label_audit.py`, used for both pairing and
acceptance. `--coverage-hist` re-measures its margin on every run; the driver passes it always.

| set | chain groups | lowest accepted | below 0.99 | below 0.97 | below MIN_COVER |
|---|---|---|---|---|---|
| training | 1,517 | 0.9871 | 1 | 0 | 0 |
| validation | 140 | 0.9848 | 1 | 0 | 0 |

Columns count chain GROUPS (one per reference chain, so a fused chain is one group covering two of
our chains). Nothing lies between 0.95 and 0.98, so any cutoff in that range gives identical
verdicts and nothing passes by sitting just above the line.

**OPEN QUESTION FOR REVIEW.** A reference chain not FULLY accounted for could be a hard failure with
the shortfall itemised, removing the last threshold from the file. It changes no verdict today, so it
is a design decision rather than something to settle by tuning.

## The 5 COORD MISMATCH structures

Sequences match exactly in all five (`n_subs` 0). `per_chain_rmsd` superposes each reference chain on
its own, alongside the pooled fit the verdict uses, which separates mixed copies from wrong atoms.

| structure | per-chain rmsd (A) | pooled rmsd (A) | pooled max (A) | reading |
|---|---|---|---|---|
| `8dpl` | 0.000, 0.020, 0.020 | 28.50 | 68.41 | copies mixed; our atoms are right |
| `8f6x` | 0.000, 0.443, 0.443 | 34.87 | 71.91 | copies mixed |
| `8d53` | 0.000, 0.000, 0.000, 0.453 | 0.34 | 10.18 | one antigen chain differs locally |
| `9djw` | 1.558, 1.596 | 15.77 | 47.02 | both chains differ locally |
| `8v8f` (val) | 0.254, 0.725 | 0.29 | 4.36 | small local difference |

`8dpl` and `8f6x` are harmless. The other three carry real local differences (max 4-13 A) and are 3
of 554 structures; not yet explained.

## Proving the audit can fail

`phases/phase_j_label_audit_selftest.py` damages copies of real labels and asserts the audit notices,
on a one-to-one structure (`10gh`) and a fused one (`32nz`). It builds its fixtures from the repo and
exits non-zero if any damage goes unnoticed. Result 2026-10-15: 10 of 10 detected.

| case | damage | verdict, 10gh | verdict, 32nz |
|---|---|---|---|
| clean | none (positive control) | ok | ok |
| substitution | one residue renamed, coordinates untouched | SUBSTITUTION, n_subs 1 | SUBSTITUTION, n_subs 1 |
| swap | coordinates of two chains exchanged | COORD MISMATCH, 7.97 A | COORD MISMATCH, 14.01 A |
| truncate | 30 residues deleted inside the variable domain | INCOMPLETE | INCOMPLETE |
| shift | one chain translated 5 A | COORD MISMATCH, 1.26 A | COORD MISMATCH, 2.24 A |

A truncation must be AIMED. Deleting the END of a chain was tried first and the audit reported `ok`
-- correctly: SAbDab2 crops to the variable region, so `10gh` label chain A is 238 residues (VH+CH1)
against SAbDab2's 119, and the deleted quarter was CH1, which the reference does not contain. An
audit cannot verify residues its reference lacks. That is this audit's coverage limit, and the reason
an unaimed truncation test would have proved nothing.

## DECISION FOR REVIEW: keep our RCSB labels, or switch to SAbDab2's?

Raised 2026-10-15 by the user, who asked for the pros and cons weighed rather than assumed. Numbers
below are measured, not estimated; the script that produced them is in this commit's message.

**What the audit says about label quality.** Our labels now verify against SAbDab2's independent
coordinates at 497 of 501 (training) and 52 of 53 (validation), with 0 substitutions and 0 unaudited
roles, and the audit is shown to be able to fail (10 of 10 planted defects detected). So there is no
label-QUALITY argument for switching. The question is coverage versus simplicity.

**Coverage cost of switching**, over all 501 training structures:

| quantity | residues |
|---|---|
| in our RCSB-derived labels | 311,923 |
| in SAbDab2's labels | 266,336 |
| lost by switching | 45,587 (14.6% of our labelled residues) |

Structures where SAbDab2 has FEWER residues than ours: 284 of 501. Where it has MORE: 4. The gap is
the constant domains -- SAbDab2 crops to the variable region, so `10gh` is 1,474 residues for us
against 1,234, and its heavy chain is 238 (VH+CH1) against 119.

**Why the constant domains may not be discardable.** `chain_pair_iptm` in Protenix
(`sample_confidence.py:536`) builds `pair_mask` as the UNION of all tokens in both chains, with NO
contact filter. The per-interface ipTM we are training the head to predict therefore depends on
constant-domain tokens. The folded input contains them either way, so switching labels would leave
those tokens present in the input but without ground truth in the FT3 losses (pLDDT, PAE, PDE,
resolved), which are per-token and per-pair.

| | keep our RCSB labels | switch to SAbDab2's |
|---|---|---|
| labelled residues | 311,923 | 266,336 (-14.6%) |
| constant domains labelled | yes | no |
| tokens in input with no label | none measured | the 45,587 above |
| verified against an independent source | yes, 497/501 and 52/53 | it IS the source; no second opinion possible |
| chain-correspondence code needed | yes (this file) | none |
| already built and folded | yes | labels yes, nothing refolded |

**OPEN QUESTIONS FOR THE AGENT.**
1. Does the FT3 loss mask unlabelled tokens cleanly, or would 14.6% unlabelled residues bias the
   pLDDT/PAE targets? This is the question that decides it, and it is not answered here.
2. Is a label set that cannot be cross-checked against anything acceptable, given that four earlier
   checks in this project were wrong in the same direction?
3. If we keep our labels, are the 3 unexplained local differences (`8d53`, `9djw`, `8v8f`, max
   4-13 A) worth chasing, or is 3 of 554 tolerable?

## Three holes found in review, and closed

| hole | effect | fix |
|---|---|---|
| a role that could not be paired never reached the verdict | validation's **9nr6 passed `ok` with BOTH antibody chains unpaired** -- only its antigen had been compared | a pass now requires `roles_audited == roles_expected`; otherwise INCOMPLETE, naming the skipped roles |
| the residue-identity check walked `get_matching_blocks()`, which returns only EQUAL blocks | a substitution fell outside every block and could not be reported. **"0 residue mismatches" was unfalsifiable** | the walk is over `get_opcodes()`; `replace` is a SUBSTITUTION, `delete`/`insert` are residues in only one file |
| coverage was counted as rows with rmsd < 0.01 | **496 of 501 was overstated** -- only 481 were complete passes; 15 were exact over an incomplete set of roles | the report carries `roles_expected`/`roles_audited`; only `verdict == ok` is fully verified |

NEGATIVE CONTROL, which the earlier version failed. One residue of 10gh's heavy chain was mutated
ARG->TRP in a copy of the label. Before: `ok`, 0 mismatches. After: `SUBSTITUTION`, `n_subs 1`,
`subs H:41->41:ARG!=TRP`. The check can now fail, so its passing means something.

## Verdict

| verdict | training | validation |
|---|---|---|
| **ok -- every role audited, no substitution, max deviation 0.0000 A** | **481 of 501** | **48 of 53** |
| INCOMPLETE -- one or more roles could not be paired | 18 | 5 |
| COORD MISMATCH | 2 | 0 |

Within the fully-verified passes:

| | training | validation |
|---|---|---|
| substitutions | **0** | **0** |
| residues in SAbDab2 but NOT in our label | **10** | **2** |
| residues in our label only (the constant domain SAbDab2 crops) | 45,594 | 5,012 |
| atoms compared | 1,978,685 | 137,052 |
| worst max deviation | **0.0000 A** | **0.0000 A** |

So where a structure is fully audited, the agreement is exact -- not approximate, not
within-tolerance, bit-identical on every compared atom.

### The 12 residues SAbDab2 has and our label lacks are deliberate, logged drops

Inspected. Not losses: every one is the `drop_geometrically_impossible` veto in
`phase_j_job_cifs.py` (`max_ca=4.5`), and the writer's own `unplaced_residues.csv` accounts for all
12 with matching counts.

| structure | role | kept / expected | vetoed |
|---|---|---|---|
| 7ugn | ag0 | 448 / 451 | 3 |
| 7ugq | ag0 | 444 / 447 | 3 |
| 7ugq | ag1 | 444 / 447 | 3 |
| 9mnq | ag1 | 168 / 169 | 1 |
| 7t2p | H | 130 / 132 | 2 |

10 training plus 2 validation = 12, exactly the figure this audit reached independently.

WHY THE VETO EXISTS, measured rather than assumed: sequence alignment can place a residue at a
position it does not belong to, because it only needs the same amino acid and the true residue there
is often unresolved. Without the veto, 104 of 501 structures lost an ENTIRE CHAIN to AF3's 10 A
filter (`filter.py:146`), and 34 more carried a 4.5-10 A break that the filter passes, so they would
have trained against a wrong residue silently. Twelve dropped residues against 104 destroyed chains
is the right trade.

That the audit rediscovered an already-logged quantity, with counts matching the writer's own
record, is a small positive signal for the audit itself.

## What was compared, and why it is independent

| artifact | provenance |
|---|---|
| `training/prep/job_cifs/<pdb>.cif` | the TRAINING LABEL, built by `phase_j_job_cifs` from `rcsb_training_cifs` |
| `SAbDab2_data/training_cifs/<instance>.cif` | SAbDab2's own curated file for the exact instance that produced the selection |

Not the same file and not derived from each other: for 10fj the label carries 220/220/172-residue
chains, SAbDab2 carries 120/113/172 -- full Fab plus antigen against variable domains plus antigen.
`training/refs/<pdb>.cif` IS md5-identical to SAbDab2's file, but that is the DockQ reference, a
different artifact from the label.

Method: pair chains through `chain_roles.csv` (`model_chain` for the label, `ref_chain` for
SAbDab2), establish residue correspondence by SEQUENCE ALIGNMENT, compare only atoms present in
both, and fit ONE rigid transform over the whole complex -- never per chain, which would hide a
swapped copy.

## The 5 structures that are not exact

| pdb | rmsd | max dev | what it is |
|---|---|---|---|
| 8d53 | 0.339 | 10.18 | H, L and ag0 exact at 0.000. 15 of 3,516 ag1 atoms (0.4%) differ. Localised -- alternate conformation or a few differently modelled residues |
| 8dpl | 22.3 | 39.1 | a different copy |
| 8f6x | 21.2 | 45.3 | a different copy |
| 8s6k | >= 1.0 | -- | a different copy |
| 9djw | 15.8 | 47.0 | a different copy |

The four copy cases are NOT broken assemblies. Checked for 9djw: our label's two chains make 180
inter-chain contacts and SAbDab2's make 204, so both are genuine contacting antibody-antigen pairs.
They are different copies of the same complex, which a single rigid fit cannot reconcile.

That is worth stating precisely because of what else it touches: the DockQ reference
(`training/refs/`) is SAbDab2's file, so for these four the label and the scoring reference describe
different copies. OST maps chains by superposition, so DockQ is still computed against whichever
copy fits, but the label and the reference are not the same instance for these four.

## Methodological notes

14 structures had at least one role paired by SEQUENCE rather than by the name in `chain_roles`,
recorded in the report with a `~seq` marker so such a row is never mistaken for a named pairing. The
cause is `chain_roles` carrying a copy index for single-chain constructs (`B1`/`B2` for the two
domains of one chain) which does not appear as a chain id in SAbDab2's file.

Four bugs of mine were found and fixed while building this, each by the next check rather than the
one that should have caught it:

| | |
|---|---|
| keyed the label on `subchain` | gemmi reports the label's subchains as `Ap`/`Bp`/`Cp` while `chain_roles` names `A`/`B`/`C`, so the first run reported 0 shared atoms for every structure |
| compared residues at equal `label_seq` | the files number independently -- 10fj's heavy chain is 20..139 in SAbDab2 and 1..220 in the label, a constant offset of 18. Comparing at equal index matched 8 of 120 residue identities; aligning gives an exact 120-residue block |
| earlier: 57 roles reported "chain not found" | 42 were the copy-index artefact and 15 were me checking auth names where `label_asym_id` was meant. ALL 57 resolve; there was no mis-map |
| earlier: resolved positions pooled across chain copies | invented a 935-residue input/label mismatch that does not exist |

## Validation, built and audited 2026-10-15 1:50 PM ET

| | structures |
|---|---|
| **RMSD < 0.01 A -- exact** | **51 of 53** |
| >= 1.0 A, different copy | **0** |
| residue-identity mismatches | **0** |
| atoms compared | 141,881 |
| not auditable, single-chain construct | 2 (7tzh, 8v8f) |

The validation labels did not exist before this -- `validation/prep/job_cifs` was empty and no
`job_cifs` directory existed outside training. They are not needed for the validation arm's own
metrics, which are decoy, coverage and regret from a scalar DockQ per pose that the SAbDab2
reference supplies. They are built because those structures are expected to become TRAINING data
once the training setup is settled, and training needs per-atom coordinates across the whole folded
chain, which SAbDab2's variable-region crop cannot provide.

THE SOURCE DEPOSITS EXISTED NOWHERE PERMANENT. All 53 RCSB cifs were only in a session scratchpad
that would have been wiped. Copied to `SAbDab2_data/rcsb_validation_cifs` (297 files, 341 MB) before
anything else was run.

## Reproducibility

    bash phases/phase_j_labels_and_audit.sh training     # build labels, then audit
    bash phases/phase_j_labels_and_audit.sh validation
    bash phases/phase_j_labels_and_audit.sh both

The driver exists because the two steps must agree on four paths, and because the validation
selection is not a file that otherwise exists -- the 53 structures are split across
`val_split_proposal.csv` (40 first representatives) and `val_split_second_reps.csv` (13 seconds). It
rebuilds that union fresh each run, into `train_val_splits/val_all_structures.csv`, and stops if any
structure lacks a SAbDab2 instance id.

## The 4 validation flags are all single-chain constructs

7tzh, 8v8f, 29tj and 9cn2 are scFv instances (`A1_A2`, `C1_C2`, `B1_B2`). SAbDab2 keeps the construct
as ONE fused chain; our label splits it into two. 8v8f and 9cn2 sum exactly to the fused length --
236 = 125+111 and 223 = 116+107 -- so nothing is missing, the pairing is simply 2-to-1 and this audit
pairs one-to-one. Their ANTIGEN roles audit exactly. Same class as open review item J1/J3.

Normalising the sequence-coverage fallback by the label chain rather than the SAbDab2 chain was
tried, left 7tzh and 8v8f unpaired anyway, and regressed 9nr6 from ok to ambiguous, so it was
reverted. A proper 2-to-1 mode is not written.

`validation/prep/job_cifs` is absent, and no `job_cifs` directory exists anywhere except the training
one. The validation label pipeline has never been run, so there is nothing on that side to audit. The
review asked to start with validation; that is not currently possible. 53 structures, and it is a
prerequisite for training on the validation arm.
