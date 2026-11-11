# The cropped arm dies in epoch 1: pass A emits no score for a structure whose crop drops a chain

Status: **open**. One cause confirmed, a second still unidentified. Written for review — the
section "What I could not explain" is the part that needs a second pair of eyes.

## The crash

Run `3pdv7i` (the cropped arm: full head, `rel` + PAE, alpha-rank 75, lr 1e-3, 256-token crops
from `training/crop_final`, targets `training/fold/itm_labels_crop`) died at **11:01 PM EST on
2026-11-10**, 21 minutes into epoch 1 of 40, ~73% of the way through the epoch.

```
STOP: pass A scored 70 of 100 expected (pdb, seed, pose, interface) key(s) for this step.
  missing e.g. [('9o4o', 500000, 0, '9o4o_DF'), ('9o4o', 500000, 0, '9o4o_EF'), ...]
  unexpected e.g. []
```

30 keys missing = 3 seeds x 2 interfaces x 5 poses. Every key of `9o4o` in that step, both
interfaces, every pose. Nothing unexpected appeared. Rank 1 raised, torchrun tore down all three.

The guard is `phase_k_train.py:5928`. It exists precisely because a missing score is otherwise
silent: `group_coeffs` intersects surviving scores with labels and skips groups left with fewer
than two poses, so the objective quietly stops being the documented full-group ranking loss.

## Confirmed: the crop is NOT at fault

`9o4o` has two scored interfaces, `9o4o_DF` and `9o4o_EF`, which `interface_chain_pairs` resolves
to model asym pairs **(0,3)** and **(1,3)**.

The pool did its job. `build_crop_pool.py:155` computes `valid = usable.all(0)` — a crop counts
only if **every** scored interface keeps a cross-chain pair — and `select_crop_pool.py:48` filters
on it. Checked directly: all 146 selected crops for `9o4o` are marked valid, and none of them is
absent from the raw pool. The epoch-1 crop keeps 99 tokens on the small side of DF and 53 on EF.

## Confirmed bug #1: Protenix compacts asym ids, and the training path does not follow

`9o4o` uncropped has chains 0,1,2,3 with 127 / 109 / 478 / 478 tokens, all framed — contiguous, so
no compaction. **The 256-token crop drops chain 2 entirely**, leaving asym `{0, 1, 3}`.

`sample_confidence.py:508`:

```python
unique_asym_ids = torch.unique(asym_id)
if len(unique_asym_ids) != asym_id.max() + 1:
    remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
```

`len(unique)=3 != max+1=4`, so Protenix **remaps asym ids to 0..N-1** and sizes `chain_pair_iptm`
to the compacted count. Chain 3 becomes row 2. Any lookup using the uncropped index is then either
out of range or — worse — **in range and pointing at a different chain pair**, returning a
plausible number with no error.

`phase_k_eval.py:92` already solves exactly this (`_cpi_index`, "replicating Protenix's
compaction… exactly when a crop has removed a whole chain"), and notes it is the identity in the
contiguous case so it is safe to apply unconditionally. The eval path uses it. `rank_loss.iptm`
(line 239) does not — it does a bare bounds check and returns None:

```python
i, j = pair
if i >= cpi.shape[0] or j >= cpi.shape[1]:
    return None
```

This is a real defect that must be fixed regardless of the crash, because the in-range case is
silent. **But it is not what killed this run** — see below.

## What I could not explain

This arm runs `--rank-pool mean`, which goes through `iptm_pooled`, and that function indexes by
asym **value** (`tok = (asym == i) | (asym == j)`, `rank_loss.py:970`), not by position — so it is
immune to the compaction above.

I replicated its exact None conditions on the real crop masks and the real `asym`/`has_frame`
tensors, on CPU:

| crop idx | iface | pair | tok | live | tm | `iptm_pooled` |
|---|---|---|---|---|---|---|
| 0 | 9o4o_DF | (0,3) | 203 | 256 | 20,592 | returns a value |
| 0 | 9o4o_EF | (1,3) | 157 | 256 | 11,024 | returns a value |
| 1 | 9o4o_DF | (0,3) | 202 | 256 | 20,274 | returns a value |
| 1 | 9o4o_EF | (1,3) | 147 | 256 | 10,044 | returns a value |

All three None branches (`tok.sum()==0 or live.sum()==0`, `tm.sum()==0`, and downstream
`t is None` / `not keep.any()`) are clear. So by this analysis pass A should have emitted both
interfaces for every pose, and it emitted neither.

### The leading hypothesis, untested

Expected and emitted keys come from **two different sources**:

- `_exp` (`phase_k_train.py:5915`) is built from `_rel_ifaces[pdb]` — the interface names from
  `--pae-scored-pairs`, i.e. the **static** pool.
- `_items` (`phase_k_train.py:5488`) comes from `_pose_pairs(pdb, lab_p, _samp_a)` — the
  **per-pose OST join**.

If `_pose_pairs` returns nothing for `9o4o` at those samples, `_items` is empty, no key is ever
emitted for either interface, and the count is short by exactly the 10 keys per seed we observe —
without `iptm_pooled` ever being called. That shape matches the evidence better than any None
branch does, and it would also explain why `unexpected` is empty.

This is consistent with a known asymmetry: `chain_roles` assigns one reference chain per structure
for the whole run, while OST decides per pose, and the two were measured to disagree on 3,154 of
36,900 rows across 52 structures. The cropped **labels** were built with the per-pose OST join
deliberately; the guard's expected set is built from the static one.

**Not yet tested**, and it should be tested before anything is changed: call `_pose_pairs` for
`9o4o` at the seeds in that step and see what comes back.

## What is NOT affected

Every uncropped arm, including `87ja6a` and all prior results. No cropping means contiguous asym
ids means no compaction, and the uncropped runs have never tripped this guard. `3pdv7i` wrote no
epoch checkpoint, so nothing downstream is contaminated.

## Questions for review

1. Is the `_exp` / `_items` asymmetry the real cause, or is it masking a third thing?
2. Should `_exp` be built from the same per-pose join that `_items` uses, or should a structure
   whose OST join is missing an interface be refused at startup rather than mid-epoch?
3. Is applying `_cpi_index` in `rank_loss.iptm` sufficient for bug #1, or does `iptm_pooled`'s
   value-based indexing need an equivalent guard for the case where a scored chain is dropped
   entirely (where it currently returns None and the step dies the same way)?
