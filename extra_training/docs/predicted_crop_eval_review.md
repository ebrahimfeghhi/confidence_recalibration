# Predicted-crop eval of the epoch-30 cropped head — for review BEFORE it runs

**Nothing here has been run on a GPU.** This document is the thing to review. The code is
committed; the eval is not launched.

## The question

Section 9b: the epoch-30 cropped head beats a shipped head by **+0.0376 mean selected DockQ (MAX
pooling) / +0.0226 (MEAN)** when both are scored on ground-truth interface crops. Section 9c:
that crop is defined from `label_coordinate`, so it cannot be built at inference and none of
those numbers are deployment numbers.

This asks whether a **label-free** crop recovers any of it.

## The crop

For every candidate pose, take that pose's own **predicted** antibody–antigen contacts (5 Å,
heavy atom, from `ci_coordinate`), keep tokens within a margin of them, and **union over all 150
poses** of the structure. One token set per structure.

- **No labels.** The union includes the interfaces of poses that dock on the wrong face, so the
  mask does not encode where the true epitope is.
- **One mask per structure, not per pose.** ipTM's `d0` is set from the chain pair's token count,
  so per-pose cropping changes the scoring function *between the candidates being ranked* —
  measured at `d0` 3.00 → 4.81 Å inside one interface (`run_logging/predcrop_size_control.py`).
  A shared mask gives every pose the same tokens, the same `n`, the same `d0`.

## What the crop actually looks like, measured (`run_logging/union_crop_margin_sweep.py`)

VALFULL, 53 structures, full = 481 tokens, ground-truth crop = 51.0% of full:

| margin | % of full | recall vs GT crop | interface coverage | cov p10 |
|--------|-----------|-------------------|--------------------|---------|
| 2.5 Å  | 57.1%     | 0.7248            | 1.0000             | 0.9359  |
| 5 Å    | 74.4%     | 0.8873            | 1.0000             | 1.0000  |
| 10 Å   | 90.6%     | 1.0000            | 1.0000             | 1.0000  |

**Running 5 Å first, then 2.5 Å. 10 Å skipped** — at 90.6% of the structure it is barely a crop.

The margin that matches the ground-truth size differs by pool (train: 5 Å → 59.7% against GT
53.3%; valfull: 2.5 Å → 57.1% against GT 51.0%), so the margin must be fixed by one rule and
reported on both, never tuned per pool.

## Code to review

**`run_logging/build_union_crop_masks.py`** — writes `<pdb> -> bool[n_token]`.
- `--verify-rule` (on by default) runs the same function on the **label** coordinates and asserts
  it reproduces `interface_crop_tokens` **bit-identically**. 53/53. Without this the masks would
  be a different crop's reimplementation.
- `--require-poses 150` hard-stops if a structure **examined** fewer candidates than the eval
  scores. It counts poses *examined*, not poses that contributed: `8piz free_400001` pose 1 has no
  cross-chain contact at all (fully dissociated decoy) and contributes nothing **by definition**.
  The first version conflated the two and refused a correct mask.
- Provenance is written into the npz: margin, contact, pool/trunk/label roots, builder git commit,
  build time, and a SHA of the exact unit list per structure.
- **Reproducibility checked, not assumed:** rebuilt from scratch, 106 arrays, **0 differ**, unit
  digest `22d4c9246e51d750` both times; only `built_utc` moves.

**`phases/phase_k_eval.py`** — new `--crop-tokens <npz>`.
- Refuses at **parse time** if `--crop-margin` is also set (two crop rules, one log line) or if
  the file is missing.
- A structure absent from the mask is **fatal**, never a silent fall-through to the full
  structure — a half-cropped pass would still write a complete metric under the crop's name.
- Refuses if a mask's token count disagrees with the cached trunk's.
- Still reads the label npz, **to slice the unit's label arrays**. The crop is label-free; the
  slicing bookkeeping is not. Worth a reviewer's attention.
- Proven to slice a real unit: `22gi` 430 → 194 tokens, `z_trunk (194,194,256)`,
  `s_trunk (194,384)`, shapes consistent.

## The run plan (NOT executed)

At an epoch boundary of `erb7nd`, after its train + val complete, pause, run, resume.

```
# ours
phase_k_eval --ckpt training/runs/t4jtjr/head_ep30_step3660.pt \
  --crop-tokens training/runs/union_crop/valfull_m5.npz --pool-mean --select-by mean ...
# shipped, SAME crops -- NO --ckpt. There is no --shipped flag; an empty --ckpt IS the
# shipped head (phase_k_eval.py:687), and `not a.ckpt` is also what enables the self-check.
phase_k_eval --crop-tokens training/runs/union_crop/valfull_m5.npz ...
```

**Two passes, not one.** `stored_iptm` is the fold's *uncropped* value; scoring a cropped head
against it is section 8c's error. The baseline must be a shipped pass over the same crops — which
is what `training/runs/cropped_eval/valfull_shipped` was in 9b.

## What would make this a negative result

The union keeps 74.4% of the structure at 5 Å where the ground-truth crop keeps 51.0%. Section 12
measured the available gain as *concentrating* supervision, and removing a quarter of the tokens
concentrates about half as much. **Expect materially less than +0.0376.** If it comes back near
zero, that says 9b's gain needed the ground-truth crop specifically — which is itself the answer
to 9c and worth the two passes.

## Open questions for the reviewer

1. Is reading the label npz for slicing acceptable, given the crop is label-free?
2. `--require-poses 150` is pool-specific. Should it be derived from the eval budget instead?
3. The mask is built over the same 30 seeds the b30 eval scores — but only because the label npzs
   happen to exist for exactly those seeds. Should the seed list be passed explicitly?
