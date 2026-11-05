# Forced chain mapping vs OST's automatic mapping — measured

Review asked for this before anything is removed from the pool: same poses, same references,
automatic mapping against a mapping forced from `chain_roles.csv`, measuring recovered interface
coverage, DockQ, and the model chain pair ipTM is read at. Script:
`run_logging/compare_forced_chain_map.py` (its `--out-dir` refuses to sit inside the poses tree,
because `ost_dockq.run()` reuses a usable json at its output path and a probe written there would
become the cached answer for a production rescore).

`-c` is confirmed to take `reference:model` pairs, and `chain_mapping` in the output is
reference -> model: OST's own help says so, and 4zto's forced run reproduces the mapping we passed.

## Result: forcing the mapping must NOT be adopted globally

| case | pool ifaces | automatic DockQ | forced DockQ | note |
|---|---|---|---|---|
| `3rhw` 500000 s0 | 4 | **0.964 / 0.900 / 0.978**, 1 absent | **0.010 / 0.010 / 0.008**, 1 absent | two identical antigen copies |
| `8w2e` 500007 s1 | 3 | **0.756 / 0.661 / 0.455** | **0.012 / 0.006 / 0.016** | two copies |
| `7uij` 500007 s3 | 4 | 0.014 / 0.013 / 0.015 / 0.016 | 0.009 / 0.007 / 0.015 / 0.015 | two copies; attributions swap |
| `4zto` 500000 s0 | 2 | **no usable json** | **0.736 / 0.906** | 10-residue peptide antigen |
| `5tl5` 500000 s0 | 2 | 0 of 2 (only H-L scored) | **OST refuses the map** | see below |

For the multi-copy structures the automatic mapper is choosing the copy assignment that fits the
pose and `chain_roles` names the other one, so forcing it does not recover interfaces — it replaces
good scores with near-zero ones. 3rhw alone would go from 0.964/0.900/0.978 to 0.010/0.010/0.008.

For `5tl5` forcing is not even available:

    RuntimeError: Chem group mismatch in custom mapping: target chain "C" groups with the following
    chemically equivalent target chains: ['A', 'C'] but model chain "C" maps to the following
    target chains: ['B']

## Two concrete findings worth acting on

**1. The identity-map retry in `ost_dockq.run()` is dead code in practice.** `_identity_chain_map`
builds its pairs from gemmi chain names, which are AUTH names, while OST works in label space:

    identity map 4zto -> ['M:M', 'I:I', 'Q:Q']
    OST            -> RuntimeError: Target chain "M" is not available in target processed for
                      chain mapping ({'F', 'E', 'D'})

So the fallback that exists to rescue structures whose automatic mapping crashes cannot fire for any
structure whose auth and label names differ. For `4zto` that is the difference between no json at
all and two interfaces at 0.736 and 0.906.

**2. Not every missing interface is a mapper failure.** `3rhw_DK` is absent under both mappings
while `C,G`, `D,G` and `C,K` all score, and both runs scored `C,D` and `G,K` as well — the second
antigen copy simply does not contact the light chain in the reference. So the "56 interfaces with no
row" figure mixes mapper failures with legitimately non-contacting pairs, and I should split it
before quoting it again.

## Still open: the `5tl5` class (20 structures)

OST's chemical grouping of the REFERENCE puts the light chain and the antigen in one group:

    reference_chains ['A','B','C']   chem_groups [['A','C'], ['B']]   chem_mapping [['B'], ['A','C']]
    chain_mapping {'A': 'B', 'B': 'A'}        (reference C never mapped)

Reference A is the 112-residue kappa light chain, reference C the 99-residue antigen, and their
sequences share essentially nothing (difflib ratio 0.000); the model's antigen chain is the same
sequence as reference C at 109 residues. The grouping also claims the model's heavy chain and its
antigen are equivalent to reference B. Both claims are false on sequence, and they are what stops
the antigen being mapped and what makes a custom mapping inconsistent.

Normalising the reference does not change it: a gemmi `setup_entities()` rewrite, same label chains,
gives byte-identical behaviour (`rewritten: 0/2 pool interfaces`), so this is not the staged
instance cif's entity records.

Next probe I would run: score against a reference cut down to the antibody plus the antigen, to
separate "the grouping over the full chain set is wrong" from "this antigen chain is being rejected".

## What I have NOT changed

The scorer is untouched. Three fold workers are scoring with it right now, so `eval/ost_dockq.py`
should not change mid-run even for the identity-map bug.
