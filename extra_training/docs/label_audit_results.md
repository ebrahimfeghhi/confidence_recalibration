# Label audit: our RCSB-derived labels vs SAbDab2's independent coordinates

Run 2026-10-15 1:10 PM ET by `phases/phase_j_label_audit.py`. Full per-structure report in
`train_val_splits/label_audit_training.csv`.

> **REVISED 2026-10-15 2:40 PM ET** after review of c8137d8 found three holes in this audit. The
> earlier headline figures (496 of 501, 51 of 53, "0 residue mismatches over 2.03M atoms") were
> wrong or unfalsifiable and are replaced below.

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
within-tolerance, bit-identical on every compared atom. The 10 and 2 residues present in SAbDab2 but
absent from our label are the only real coverage gaps found and have not been individually
inspected.

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
