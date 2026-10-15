# train_val_splits

> **SUPERSEDED IN PART, 2026-10-15.** The input/label mismatch this file is built on DOES NOT
> EXIST. SAbDab2's `Hseq` is defined as the residues that are structurally RESOLVED, so our
> input already matched the label's coordinate set; 919 of the 935 "missing" residues have no
> coordinates in their own chain. See **`docs/hseq_correction.md`**. The refold still running
> is unaffected: it regenerates deleted trunks using the ORIGINAL queries.


Which clusters train, which validate, and what was excluded. Written by phases A, B and J0; read by
the later phases rather than recomputed.

| file | what it is |
|---|---|
| `train_clusters.csv` | the 483 training clusters, after leakage removal and the validation split |
| `train_clusters.meta.json` | provenance for the above, including a sha256 of the validation file it was built against |
| `val_split_proposal.csv` | the 40-cluster validation split: one named representative per cluster |
| `val_split_proposal_clusters.csv` | just those 40 cluster ids |
| `val_split_second_reps.csv` | the second representative per cluster, where one exists |
| `excluded_homolog_clusters.csv` | the 34 clusters dropped for homology with the benchmarks, with the matching query and identity |
| `cache_train_structures.csv` | the 501 structures selected for the training cache fold |
| `refold_audit_train.csv` | **the refold list, training**: 136 roles over 118 structures, 935 residues. Written from the domain audit; carries `corrected_q` verbatim |
| `refold_audit_val.csv` | the same for validation: 15 roles over 13 structures, 124 residues |
| `skip_boundary_train.csv` | 10 roles NOT refolded because the correction needs a boundary decision, with the measured reason |
| `skip_boundary_val.csv` | the same for validation: 3 roles |
| `refold_internal_omissions.csv` | SUPERSEDED by `refold_audit_train.csv`. Built from a G/S composition guess at the VH/VL splice point, which queued six pure-tether restorations |
| `refold_internal_omissions_validation.csv` | SUPERSEDED by `refold_audit_val.csv` |
| `excluded_interdomain_linkers.csv` | runs deliberately NOT restored, training: 4 single-chain-construct tethers (`GGGGSGGGGSGGGGS` and the like) that belong to neither variable domain |
| `excluded_interdomain_linkers_validation.csv` | the same for validation: 1 |
| `archive/` | the superseded 50-cluster validation list |

## The refold list

It is derived, not asserted. `phase_j_domain_audit.py` reads the queries Protenix ACTUALLY folded
out of the job JSONs -- roles from the MSA directory names embedded in them, never from this repo's
chain mapper, because a log written by the mapper under audit is provenance and not validation --
aligns each against its deposited chain, and reads variable-domain boundaries from HMMER against
ANARCI's own germline profiles. `phase_j_refold_from_audit.py` then turns that into the list.

    python phases/phase_j_domain_audit.py --out audit_train.csv
    python phases/phase_j_refold_from_audit.py --audit audit_train.csv \
        --out train_val_splits/refold_audit_train.csv \
        --skip train_val_splits/skip_boundary_train.csv

A role is refolded only when the correction is unambiguous: one variable domain on the chain, and
native residues missing strictly INSIDE a span the query already covers at >= 80%. Everything else
is skipped with its reason rather than guessed. What that rule exists to prevent, measured:

| the earlier rule did this | the audit finds |
|---|---|
| queued `KGGG`, `GGG`, `GGGG`, `SGGG`, `SG`, `TTPP` for restoration on 8cz8, 9ejz, 8vy9, 8pww, 9tuy, 32nz | pure inter-domain tether on two-domain chains; restoring it writes a worse input than the one it replaces |
| classified 8dpl 115-136 as tether | it opens with `TVSS`, the real end of VH (hmmscan H 1-118, E=1.2e-58) |
| would rebuild 28 scFv queries to the profile edge | those differ from it by one or two residues; the domain edge is not crisp to the residue |
| reported 437 residues "missing" from three antigen roles | 9-10 residue peptide epitopes scattered across a long chain by chance; 7tr4 ag2 covers 9% of its span |
| counted 8qp6's 104 residues past the V domain as contamination | chain B is a single-chain Fab; those are CH1 |

ANARCI's Python API returns `None` for every chain here -- its Biopython 1.81 layer cannot read
HMMER 3.4 domain coordinates and discards the hits. `hmmscan --domtblout` on the identical `ALL.hmm`
reports `mouse_H 1-118, E=1.1e-59`. The audit drives the HMMs directly for that reason.

The superseded files were written by `phases/phase_j_refold_list.py` from
`phases/phase_j_job_cifs.py --native-gaps`:

    python phases/phase_j_job_cifs.py --native-gaps <gaps.csv>            # per-run report
    python phases/phase_j_refold_list.py --gaps <gaps.csv> --out ... \
        --linkers ... --seeds 10 --mb-per-unit 280                        # the list + its cost

`--seeds` is 10 for training and 100 for validation; `--mb-per-unit` is the measured trunk size,
280 MB for training folds and 136 MB for validation ones.
