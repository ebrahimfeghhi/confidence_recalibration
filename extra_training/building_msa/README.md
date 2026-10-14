# building_msa

Every fold needs an MSA per chain, and the search is the expensive part. Built once, by phases C–E.

| | |
|---|---|
| `queries/` | the deduplicated sequences to search for, and which complex each came from |
| `work/` | the MSAs those searches produced — read by every fold, referenced by absolute path inside each job JSON |
