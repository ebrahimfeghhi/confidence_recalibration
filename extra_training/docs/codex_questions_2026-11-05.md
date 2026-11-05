# Review requests — 2026-11-05 morning

Context: Wave 1 training fold is on its last 61 GPU units of 3,560. The validation fold (30
structures x 100 seeds) is armed behind it and has not launched. Commits to review:
`6a1468cc` (single-instance locks) and `88d3b83f` (duplicate resolution via OST's own mapping).

## 1. OST never maps the antigen chain for 21 structures, so 56 pool interfaces can never score

Measured, not inferred. For every unit the gate calls "CPU only -- unscored, poses on disk", I read
the retained OST output at `<run>/ost_json/<sample>.json`:

- 200 units (20 structures) **already scored successfully**. OST found only the intra-antibody
  interface and left the antigen chain out of `chain_mapping`, so the pool's antibody-antigen pairs
  were never produced. `5tl5` seed 500000: `dockq_interfaces [['A','B','B','A']]`,
  `chain_mapping {'A':'B','B':'A'}` — reference chain C absent. `chem_groups [['A','C'],['B']]`,
  i.e. two reference chains in one chemical group against one model chain in that group, so the
  automatic mapper can only map one of them.
- 10 units (`4zto`) scored with no interface at all; its antigen is a 10-residue **PEPTIDE**.
- Across the whole pool: 56 of 1,711 interfaces (3.3%) over 33 structures have no row at all, and
  21 structures scored some interfaces but not others (e.g. `3rhw` is missing only `3rhw_DK`).
  Some of the 33 are simply not folded yet.

**Questions.** Is this a pool-definition problem (interfaces named against a reference copy the
model does not contain) or a scorer-invocation problem (we rely on OST's automatic chain mapping
where we could pass an explicit model->reference mapping, which we already hold in
`chain_roles.csv`)? Should `ost_dockq` pass an explicit mapping so the pool's pair is always the
pair scored? And should pool-build exclude an interface whose reference chain is one of several
identical copies, so the queue count stops implying work that cannot produce a row?

## 2. chain_roles and OST's mapping disagree for 2 structures — how far does that reach?

`88d3b83f` resolves duplicate rows using OST's retained `chain_mapping` rather than
`chain_roles.csv`, because the mapping that identifies a RECORDED row must be the one the scorer
used when it wrote it. On 8 of 10 duplicates the two agree. On 2 they do not:

    7uij_DF sample 3   chain_roles (D,F) -> model (B,D) = 0.1487   ost -> (B,C) = 0.1481
    8w2e_BC sample 1   chain_roles (B,C) -> model (D,B) = 0.4209   ost -> (C,B) = 0.6320

with the duplicate rows carrying 0.1482/0.1481 and 0.6273/0.6320 respectively.

**Question.** If `chain_roles` is wrong for those two structures, then every row for them — not
only the duplicates — has an `iface` name implying one reference pair while its `iptm` came from
whatever pair OST chose. Is the CSV self-consistent (DockQ and ipTM both from OST's pair, so only
the NAME is suspect), or can the two columns come from different pairs in the same row? I would
like this checked across the whole CSV rather than at the 10 duplicates.

## 3. Is `_chain_index` the right way into `chain_pair_iptm`?

Both the old and the new resolution assume `chain_pair_iptm` is indexed in model asym order such
that `ost_dockq._chain_index('B') == 1`. If the matrix is ordered by the job json's chain order
rather than by letter, both mappings could be subtly wrong in the same direction and the agreement
on 8 duplicates would not catch it. Please confirm the matrix ordering independently.

Also: I assume `chain_mapping` is reference -> model. The `aln` entries (`>reference:A` /
`>model:B`) support that, but it is an assumption worth a second pair of eyes.

## 4. The gate's CPU bucket describes the poses, not recoverability

`training_queue_drained.py` reports "CPU only -- unscored, poses on disk: 210 (rescore, no
refold)". That is true about the poses and false about recovery: ~200 of those already have a
successful OST score that simply contains no such interface, so a rescore reproduces the same
nothing. Proposed fix: split that bucket by consulting the retained `ost_json` — "never scored
(OST failed, or no output)" versus "scored, interface not produced by OST". Is that the right
split, and is there a case where re-running OST on an existing successful score legitimately
yields a different interface set?

Related: with close to zero recoverable units, should `rescore_training.sh` still run when the
workers exit? It is armed and will run. It is CPU-bounded and takes a backup, so I believe it is
harmless, but it will claim to be a recovery that recovers nothing.

## 5. Lock lifetime in the launcher

`launch_wave1_val_fold.sh` now takes `flock -n` on `validation/fold/.launch.lock` and also refuses
if any `w1val` tmux session exists. The flock is released when the launcher exits, a few seconds
after it starts the three sessions; from then on a late arm is caught only by the session check.
Is there a window worth closing (a session that dies immediately after creation, say), or should
the lock be held for the lifetime of the fold instead of the launcher?

## 6. Replacing the canonical results.csv

With all 10 duplicates now resolved, `dedupe_results.py --replace` is unblocked on evidence. It
refuses while any fold session writes, takes a backup and swaps with `os.replace`. Is there a
reason to keep holding it beyond "wait for the workers to exit"?
