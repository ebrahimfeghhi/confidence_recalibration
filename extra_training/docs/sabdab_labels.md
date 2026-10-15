# Where the input/label mismatch comes from, and why SAbDab2's own labels cannot replace RCSB's

Written 2026-10-15 10:50 AM ET, from a discussion that corrected several of my own claims. Numbers
measured locally; no external source was consulted, which is this document's main limitation.

## The mismatch, in one example

| | residues |
|---|---|
| INPUT we folded, from SAbDab2's sequence table | 193 |
| LABEL, residues with real coordinates in the RCSB deposit | 218 |
| present in the label, absent from the input | 28, mid-chain |

9p1m heavy chain. The LABEL IS RICHER THAN THE INPUT: the crystal holds coordinates for residues our
prediction never contained, so at comparison time they have nothing on the model side to align to.

This affects 136 roles over 118 training structures and 15 roles over 13 validation structures.

### They are not merely unresolved residues

A plausible defence would be that the omitted residues have no coordinates anyway, making the
omission harmless. Checked on the first 40 refold roles: **218 of 218 sampled residues are resolved**
and carry real coordinates. The mismatch is genuine.

## Why two sources were joined at all

| source | provides | why it alone is not enough |
|---|---|---|
| SAbDab2 sequence tables (`Hseq`, `Lseq`, `agexpectedseqs`) | the full Fab construct sequence | no coordinates |
| SAbDab2 structures (`SAbDab2_data/training_cifs/`, 4,903 CIFs, instance-named) | coordinates, and NO SEQRES records at all (`full_sequence` is empty for every entity) | trimmed; see the measurement below |
| RCSB deposits (`SAbDab2_data/rcsb_training_cifs/`, 559 CIFs) | complete coordinates AND SEQRES | its SEQRES disagrees with SAbDab2's sequence for ~25% of structures |

So sequences came from SAbDab2 and labels from RCSB -- two independent curations of the same
molecule, agreeing for about three quarters of structures and differing by a few internal residues
for the rest.

## SAbDab2 is NOT internally inconsistent

Tested directly: is SAbDab2's own structure a clean substring of SAbDab2's own sequence?

| | roles |
|---|---|
| structure fully contained in the sequence | **30 of 30** |
| not contained | 0 |

Their dataset is self-consistent. The mismatch is a consequence of our join, not a defect they
shipped.

## SAbDab2's labels CAN be used; the limitation is which regions they supervise

CORRECTED after review. The earlier heading here said their labels "cannot replace RCSB's", which
was too absolute. What is true is a coverage limitation, not an invalidity:

| supervision wanted | can SAbDab2's cropped coordinates supply it? |
|---|---|
| variable domain and antigen | **yes**, given validated residue mapping |
| the full Fab, constant domain included | only partly -- constant-domain coordinates must come from elsewhere |
| full-Fab inputs with losses restricted to chosen regions | **yes**, with explicit supervision masks |

That third option needs care: "cropped out by the dataset" must not be allowed to become
"experimentally unresolved", which would corrupt the `resolved` head's supervision in particular.

RCSB is the right choice for THIS pipeline because our inputs carry substantial constant-domain
sequence and we supervise across the whole chain. That is an implementation consequence of our
coverage needs, not a defect in SAbDab2. Missing SEQRES in their CIFs likewise means reconstruction
work, not unusable coordinates.

### The measurement that drove the original wording

The deciding test: for each role we want to refold, does SAbDab2's own structure already contain the
sequence we need?

| coverage of the corrected query by SAbDab2's own structure | roles |
|---|---|
| >= 99% | **0** |
| 90-99% | **0** |
| < 90% (typically 54%, range 50-72%) | **136 of 136** |

SAbDab2's structures cover roughly half of each chain and the residues at issue are not in them, so
they cannot supply CONSTANT-DOMAIN labels for a full-Fab fold. They remain usable for the variable
domain and the antigen.

## The part of this that argues against refolding at all

The region SAbDab2 trims is the constant domain, and that is where the disputed residues live. Three
independent lines point the same way:

| | |
|---|---|
| 1 | of the 1,059 residues a refold would restore, **1,044 (98.6%) are constant domain or termini**; only 15 touch a variable domain, and both of those straddle its C-terminal edge rather than a CDR |
| 2 | ~~SAbDab2's curated structures omit that region, so the field does not treat it as load-bearing~~ **RETRACTED.** The cropping is a deliberate scoping choice by the curators, not evidence about biological relevance. And SAbDab2 is itself derived from PDB depositions, so it is a curated subset of the same experimental data rather than an independent source of truth -- it cannot serve as the external check I claimed. Their site also offers full processed structures separately from the ML dataset |
| 3 | ~~the reported metric does not read them~~ **WRONG.** `sample_confidence.py:536` builds `pair_mask` as the union of ALL tokens in both chains and passes it straight to `calculate_iptm`, which applies no contact or distance filter. Per-interface ipTM is therefore computed over every inter-chain PAE entry between the complete heavy chain -- VH **and CH1** -- and the complete antigen. Constant-domain residues DO affect the score |

WHERE THAT LEAVES THE CASE AGAINST REFOLDING: largely collapsed. Argument 2 is retracted and
argument 3 was simply wrong. Argument 1 survives only as a geometric statement -- the residues are
far from the paratope -- and its relevance is much reduced now that the reported metric is known not
to be geometry-restricted. Restoring those residues changes the ipTM we report, so the refold is
better justified than I argued. It was launched on 2026-10-15 and is running.

**Decision taken 2026-10-15:** refold the 118 training and 13 validation structures with their
ORIGINAL queries, which needs no MSA rebuild and runs immediately, rather than wait on the
corrected-query path (96 GB re-fetch, ~700 GB envdb, effectively one-shot).

## The trimming is deliberate, and the authors say so

Confirmed from the Zenodo record rather than inferred from our measurements. SAbDab2's own
description of what each structure file contains:

> "Each structure file contains a single paired-chain or single-chain antibody, **cropped to the
> variable region**, alongside any antigen chains."

So the 50-72% coverage measured above is the dataset working as designed, not a defect.

IT IS NOT, HOWEVER, AN EXTERNAL CHECK ON WHETHER THE CONSTANT DOMAIN MATTERS, which is how an
earlier revision of this file used it. Cropping is a scoping decision by the curators, and SAbDab2
derives from PDB depositions, so it is a curated subset of the same experimental data rather than
independent evidence. Their site offers full processed structures separately from the ML dataset.

Version in use: **v0.2.0** (Zenodo 22019991, published 2026-08-20). **v0.3.0** was published
2026-10-13 (record 23343734) and we are not on it; its record carries no changelog, and the authors
state splits are backward-compatible across versions, so nothing here is invalidated. See
`training_data_overview.md` for the full citation and the version table.

## Open, and not traced

THE CROPPING DOES NOT EXPLAIN THE SEQUENCE OMISSIONS, and matching their cropped variable-region
coordinates does nothing to validate the omitted constant-region sequence. These are two separate
questions and only the first is answered.

**Why** SAbDab2's sequence table differs from RCSB's SEQRES in those ~25% of cases -- a curation
choice, a different construct record, or something else -- has never been investigated. That is the
actual origin of this whole issue. No README or download URL is recorded under `SAbDab2_data/` itself, though
`cached_training.md` does name the version -- an earlier revision of this file wrongly said no
provenance existed anywhere. The trimming question IS answered by the Zenodo description, quoted
above; what remains unanswered is why their sequence TABLE differs from RCSB's SEQRES.

## Claims of mine that were wrong on the way here

| claim | corrected |
|---|---|
| "SAbDab2's CIFs are variable-domain-only" (from 3 examples) | varies by structure; the generalisation was unfounded. The measurement that holds is the 136-role coverage table above |
| a follow-up count of "15 of 60 V-domain only" | measured the longest chain, which is the ANTIGEN. Wrong statistic |
| SAbDab2's sequences do not match their own structures (ratio 0.000) | my extraction read SEQRES, which their files do not carry. Against resolved residues they match exactly, 30 of 30 |
