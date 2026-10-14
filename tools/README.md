# tools

Downloaded binaries and databases the MSA pipeline shells out to. 5.3 GB on disk; only the small text
is tracked here, so the layout is visible without the weight.

| | |
|---|---|
| `colabfold/` | the ColabFold MSA search and its database setup scripts; phases D and E run these |
| `mmseqs/` | the sequence search engine ColabFold drives |
| `hmmer-3.1b2-linux-intel-x86_64/` | HMMER **3.1b2** specifically — ANARCI parses hmmscan's text output with a Biopython parser that cannot read HMMER 3.4, which silently returns no domains |
| `anarci_hmms/` | the ANARCI antibody-numbering profiles, re-pressed with 3.1b2 to match |
