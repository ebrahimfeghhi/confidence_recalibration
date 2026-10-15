# RETRACTION: the inputs were never truncated. `Hseq` is resolved-only by definition.

Written 2026-10-15 11:00 AM ET. Supersedes the premise of `sabdab_labels.md`, the stated rationale
for `refold_audit_train.csv`, and the parts of `overnight_2026-10-15_concerns.md` that rest on an
input/label mismatch. Found by reading the README bundled in the SAbDab2 download, which I had
argued was not worth fetching.

## What the dataset actually promised

From `splits_final/README.md` inside `splits.tar.gz`:

| column | definition, verbatim |
|---|---|
| `Hseq` | "sequence of heavy-chain residues **which are structurally resolved**. These are always sequence-contiguous." |
| `Hseq_expected` | "complete sequence of heavy-chain residues, as annotated in the PDB, **including unresolved ones**" |

Our fold input is `Hseq`, verified byte-identical for every case checked. So the input is, by
construction, exactly the residues that have coordinates.

## The measurement that was wrong

I compared the input against the FULL chain and called the difference a truncation. Re-checked
against each role's own chain, the chain `chain_roles.csv` names and the label actually uses:

| of the 935 residues the refold would restore | residues |
|---|---|
| **not resolved in their own chain** -- no coordinates, so never a mismatch | **919 (98.3%)** |
| resolved in their own chain | 16 (1.7%) |

And the 16 are alignment noise, not data. Each is a single residue at the END of its run:

| role | missing run | resolved within it |
|---|---|---|
| 8w2e H | 184-193 | 1 of 10 -- only 193 |
| 8tq7 H | 132-140 | 1 of 9 -- only 140 |
| 7ssc H | 139-142 | 1 of 4 -- only 142 |

That is a difflib run boundary placed one position too far, not a gap in the data.

## Why I did not see it

The check asked "is this residue resolved?" against the UNION of every chain in the file. These
structures hold several copies of the antibody, and a residue resolved in one copy is often
unresolved in another. 9p1m:

| | 135-144 resolved |
|---|---|
| union over all chains | 10 of 10 |
| **chain F, the one we use** | **0 of 10** |

SAbDab2's instance is `pdb_00009p1m_F_E`, Hchain `F`. `chain_roles.csv` maps role H to auth chain
`F`. The pipeline picked the right chain all along; only my verification was wrong.

## Three bugs of mine, same shape

Each was caught by a later cross-check rather than by the check itself.

| bug | effect |
|---|---|
| resolved positions pooled across all chain copies | invented a 935-residue input/label mismatch that does not exist |
| `difflib.ratio`, which is symmetric, used to match a partial chain | an 81-residue reference fragment scored 0.26 against its own 532-residue chain; two interfaces refused as "ambiguous" that were not |
| `gemmi.remove_ligands_and_waters()` before a contact search | deleted short peptide ANTIGENS, inventing 2 phantom interfaces and nearly pruning 2 real ones from the pool |

## What this changes

| | |
|---|---|
| the input/label mismatch | **does not exist**. Input and label are the same resolved set of the same chain |
| the corrected-query refold | **permanently unnecessary**. No 96 GB UniRef30 re-fetch, no ~700 GB envdb, no 2,480 extra folds. It existed only to fix this |
| `building_msa/queries_corrected_v2/` | 149 corrected sequences, now moot. Kept as a record, not to be used |
| the refold RUNNING now | **unaffected and still required**. It folds the ORIGINAL queries to regenerate trunks for the 119 structures whose trunks were deleted; a trunk needs a fresh inference pass regardless of the query |
| the 98.6%-constant-domain argument | moot in both directions -- there is nothing to restore |

## What is still true

`Hseq` being resolved-only also explains the "internal omissions" cleanly: a chain whose middle is
unresolved yields an `Hseq` that skips it, and SAbDab2 states these are "always sequence-contiguous",
i.e. reported as one contiguous string. Folding that string is folding a sequence the crystal
supports. Whether the benchmarks' practice of folding the FULL deposited sequence creates a
train/test difference is a separate, open question, and the only one left standing.
