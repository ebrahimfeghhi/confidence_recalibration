# top_50_complex -- per-COMPLEX donor selection

## Procedure

1. **Per interface**: order its poses by ipTM; keep a pose only if its pose-pose DockQ to everything
   already kept is < 0.23; stop at 50. -> that interface's ranked leader list.
2. **Per complex**: round-robin those lists (each interface's 1st pick, then each 2nd, ...), skipping
   poses already taken, until 50 distinct poses.
3. **Fold** each donor once: one template, `target=*`, seed 430000.
4. **Score** every interface of the complex from each fold.
5. **Evaluate** per interface: pool the free poses + all the complex's templated poses, take the
   highest-ipTM pose, report its DockQ. Compare to the same rule on free poses alone.

3313 donors across 123 complexes, median 27 each (decoy complexes 36, solved 11, no-coverage 50).

## Why per complex

A template is a whole complex and a fold scores every interface of it, so donors cannot meaningfully
be chosen "for interface A-B". The previous run (`k50`, now `k50_per_interface`) selected per
interface and tagged leaders `K00..K49` per interface, so tags collided between siblings: `fold.py`
identifies a unit as `(seed, tag)` checked across ALL the complex's interfaces -- `target` is not in
the key -- so once A-B's `K00` had written rows for every interface, A-C's `K00` was skipped.
Whichever interface was claimed first supplied the donors for the whole complex. 43 of 98 completed
interfaces got leaders they never selected; several got none of their own.

## Round-robin, not majority vote

Capping the union by "poses the most interfaces asked for" starves the idiosyncratic interface -- on
9uza one interface kept 14% of its own picks. Round-robin gives each interface an equal share and
raises that floor to 36%.

## Reuse

428 folds are carried over from `k50_per_interface`, retagged `K##` -> `C##` by matching the donor
POSE (the tag is only a name; the pose is the identity). Only SINGLE-interface complexes are reused:
there one fold = one tag = one pose. On multi-interface complexes 114 `pdb+tag` pairs were claimed by
two interfaces, so a stored row cannot be traced to its template, and those are refolded.

Their rows were copied into `results.csv` before launch, so `fold.py` skips them as already-scored --
verified: 428 skipped, 2885 run.
