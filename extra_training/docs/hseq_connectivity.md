# `Hseq` fuses resolved fragments across unresolved gaps. 138 roles, 119 structures.

Written 2026-10-15 11:20 AM ET. Raised in review of f92996c and confirmed. This REVERSES part of
`hseq_correction.md`, which concluded that corrected-query refolding was permanently unnecessary.

## The distinction that was missed

Matching the observed residue set establishes COORDINATE COVERAGE. It says nothing about SEQUENCE
CONNECTIVITY. A residue with no coordinates still belongs to the polymer, and splicing across it
asserts a peptide bond that does not exist.

SAbDab2's README says `Hseq` residues "are always sequence-contiguous". That means stored as one
string, NOT contiguous in the polymer. 9p1m chain F, 193 resolved residues spanning positions
1-221:

| contiguous runs | residues |
|---|---|
| 1-134 | 134 |
| 145-164 | 20 |
| 173-194 | 22 |
| 205-221 | 17 |

Unresolved gaps the string jumps: 135-144, 165-172, 195-204 -- exactly the "missing runs" the audit
flagged. Folded as one chain, the input asserts that residue 134 is bonded to residue 145, ten
positions apart in the real protein.

## Scale, measured over every role in the selection

| | roles |
|---|---|
| one contiguous run, connectivity intact | 1,340 |
| **2 fragments fused** | 116 |
| **3 fragments** | 17 |
| **4 fragments** | 3 |
| **5+ fragments** | 2 |
| **total fused** | **138, over 119 structures** |
| chain named by chain_roles not found in the cif | 57 |

Worst: 9jqz H and 9p1m L at 5 fragments; 9p1m H, 9d72 H, 9qqf H at 4.

The 119 structures are, within one, the same set the refold list already identified. The original
finding was right; the DIAGNOSIS was wrong, and so was the retraction.

## The 16 residues, verified individually as asked

Each is the FIRST residue of the following fragment, not a gap member:

| role | flagged run | resolved within it | position-1 | position+1 |
|---|---|---|---|---|
| 7ssc H | 139-142 | 142 only | 141 unresolved | 143 resolved |
| 7t82 H | 136-139 | 139 only | 138 unresolved | 140 resolved |
| 7t87 H | 140-143 | 143 only | 142 unresolved | 144 resolved |
| 7xy8 H | 133-138 | 138 only | 137 unresolved | 139 resolved |
| 8av9 L | 112-115 | 115 only | 114 unresolved | 116 resolved |
| 8cn9 H | 142-143 | 143 only | 142 unresolved | 144 resolved |

16 of 16 follow this pattern: the difflib run boundary extends one position too far and swallows the
next fragment's first residue. An off-by-one in the audit, not a data defect -- but confirmed
individually rather than inferred from three cases.

## What this means for the refold

| claim | status |
|---|---|
| "the label has coordinates the input lacks" | still WITHDRAWN -- 919 of 935 have no coordinates in their own chain |
| "there is no defect in these inputs" | **WRONG.** The defect is false connectivity, for 138 roles over 119 structures |
| "corrected-query refolding is permanently unnecessary" | **RETRACTED.** It is the only fix: restoring the unresolved residues makes the chain a true contiguous polymer, and the label simply carries no coordinates at those positions |

AND THE REFOLD RUNNING NOW FOLDS EXACTLY THE AFFECTED SET. All 118 structures in
`training/fold_deferred/queue.tsv` have at least one fused role; none has intact connectivity. It
uses the ORIGINAL queries, so it reproduces the chimeric inputs. It restores the cache to its prior
state and is not destructive, but it does not fix this.

Not stopped pending a decision. The corrected queries already exist, validated, in
`building_msa/queries_corrected_v2/` (149 sequences); using them needs the MSAs rebuilt, since the
a3m column count must match the longer query -- a 96 GB UniRef30 re-fetch plus the envdb build.

## Still open

Whether a fused input measurably harms the fold. The geometric consequence is a false peptide bond
at each junction; whether Protenix's output degrades enough to matter, and by how much, has not been
measured. That is testable cheaply: fold one affected structure both ways and compare, which needs
one corrected MSA rather than the whole rebuild.
