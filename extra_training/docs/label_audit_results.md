# Label audit: our RCSB-derived labels vs SAbDab2's independent coordinates

Run 2026-10-15 1:10 PM ET by `phases/phase_j_label_audit.py`. Full per-structure report in
`train_val_splits/label_audit_training.csv`.

## Verdict

| | structures |
|---|---|
| **RMSD < 0.01 A -- exact to file-rounding precision** | **496 of 501** |
| 0.01-1.0 A, minor and localised | 1 |
| >= 1.0 A, a DIFFERENT COPY of the complex | 4 |
| residue-identity mismatches, anywhere | **0** |
| total atoms compared | **2,027,629** |

The labels are correct. 496 of 501 structures agree atom for atom with a separately curated copy of
the same deposit, and not one residue identity disagrees across two million atoms.

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

## NOT DONE: the validation labels do not exist

`validation/prep/job_cifs` is absent, and no `job_cifs` directory exists anywhere except the training
one. The validation label pipeline has never been run, so there is nothing on that side to audit. The
review asked to start with validation; that is not currently possible. 53 structures, and it is a
prerequisite for training on the validation arm.
