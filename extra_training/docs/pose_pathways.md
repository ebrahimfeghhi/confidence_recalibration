# More pose pathways into the confidence head

Status: proposal, nothing implemented. Written 2026-10-19, revised the same day after review.

**C is the one to build.** A is a reasonable follow-up but this doc does not establish its
premise. B is ruled out — see below, it cannot reach PAE at all.

## Where the pose enters today

`ConfidenceHead.memory_efficient_forward`, in `protenix/model/modules/confidence.py`:

```python
distance_pred = torch.cdist(x_pred_rep_coords, x_pred_rep_coords)      # [N_token, N_token]
z_pair = z_pair + self.linear_no_bias_d(one_hot(distance_pred, ...))   # 39 bins -> c_z
z_pair = z_pair + self.linear_no_bias_d_wo_onehot(distance_pred[..., None])  # 1 -> c_z
s_single, z_pair = self.pairformer_stack(s_trunk, z_pair, pair_mask, ...)
pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
```

So there are **two** pose paths today, both scalar distance, both added into the pair
representation **once**, before a stack of **4** pairformer blocks whose weights are all
pose-agnostic. The binned path carries resolution; the raw path covers the whole range
continuously, including distances under the first bin edge.

Relevant dimensions: `n_blocks=4`, `c_z=128`, `c_s=384`, `num_bins=39`
(`arange(3.25, 52.0, 1.25)`).

Two details of that encoding, both verified:

- `one_hot` is `(x > lower_bins) * (x < upper_bins)`, and the first lower edge is 3.25 A. So a
  distance of 0 — the diagonal — produces an all-zero vector, and so does *any* distance under
  3.25 A. Only the raw path sees those. `linear_no_bias_d_wo_onehot` has no bias, so an input
  of 0 gives an output of 0. Both existing pose paths therefore contribute exactly zero on the
  diagonal.
- `x_pred_rep_coords` is selected by `distogram_rep_atom_mask`, which is **one atom per token**
  (measured: 625 rep atoms for 625 tokens on `10fj`). `cdist` is a CA-only distance matrix.

---

## C. Frame-relative direction

### The argument

PAE is not a distance. It is: superpose on token i's local frame, then ask how far token j
ended up from where it should be. `PAELoss.calculate_label` computes exactly that — express j's
position in i's frame for the predicted structure, do the same for the true structure, take the
distance between those two vectors.

The head's pose input is a **CA-only** distance matrix. A residue's backbone frame is defined by
its N, CA and C atoms, and N and C are not in that matrix at all. So the frame's orientation —
which way the residue is "facing" — is not merely hard to recover from the input, it is
**absent**, beyond whatever is inferable from neighbouring CA geometry. C supplies it directly.

That is the justification. It is about information the input does not contain.

### What this argument is NOT

An earlier version of this doc argued that because `cdist` is symmetric and PAE is not, the pose
**cannot** contribute to the asymmetric part of the target. That is false, and the error matters
enough to record.

By the time the distance is added, `z_pair` is already asymmetric: `z_init[i][j]` is
`linear_no_bias_s1(s_j) + linear_no_bias_s2(s_i)` with two *different* matrices, and `z_trunk`
is asymmetric too. The pairformer blocks are nonlinear. So the same symmetric distance value
lands in different contexts at `[i][j]` and `[j][i]`, and changing it moves the two outputs by
different amounts. The pose can modulate an asymmetry it did not create.

What it still cannot do is state that in *this* pose, j sits up-and-left of i. That is C's case,
and it does not need the impossibility claim.

### The asymmetry measurement, and what it is good for

Measured 2026-10-19 on 3 structures (`10fj`, `10gh`, `10op`) x 2 seeds x 5 poses, computing the
true PAE label through `expressCoordinatesInFrame` and binning it the way the loss does
(64 bins over 0-32 A, everything beyond saturating in the last bin):

| scope | frac of pairs where bin(i,j) != bin(j,i) | corr(bin, bin^T) |
|---|---|---|
| all pairs | 0.690 | 0.733 |
| inter-chain pairs | 0.553 | 0.651 |

Over half of inter-chain pairs carry a different label in the two directions, so the target is
strongly direction-dependent. That is a fact about the target and it motivates giving the model
directional input. It is **not** evidence that the current model cannot produce asymmetric
output.

Caveats: 43% of the inter-chain pairs are both saturated in the top bin (>= 32 A), where
asymmetry cannot show even if present, so the true figure is higher than 55%. And these are
whole uncropped structures from the fold set, not the `overfit30` interface crops we train on.
Re-measuring on the crops is a few minutes of CPU.

### What goes in: the unit direction vector

For each ordered pair (i, j), the displacement of j expressed in i's frame, **divided by its own
length** — a unit direction vector, three numbers bounded in [-1, 1].

Direction only, because the magnitude is already supplied by the two existing distance paths.
That also removes any free scale parameter, which matters for the reason below.

For a residue the frame atoms are N, CA, C, giving origin CA, `e1` the bisector of the two bond
directions, `e2` perpendicular to `e1` in the N-CA-C plane, `e3` normal to that plane. The three
numbers are j's direction along those axes.

This is asymmetric in (i, j), like the target, and still invariant to globally rotating or
translating the complex, because i's frame rotates with it.

### Why normalisation is required, not optional

Zero-initialising the new layer makes its *output* zero at step 0. It does not make the first
**update** small: the gradient on the weight is the output error times the input, so raw
Angstrom components of +/-30 give an update 30x larger than inputs of order 1. The weight does
not stay near zero — it jumps, and the head's output is disrupted after one step. This is a
broken head, not a slow start.

The unit vector fixes it with no constant to choose. (Binning each axis the way distance is
binned would also work — 3 x 39 one-hot inputs of magnitude 1 — but it is 117 inputs instead of
3 for no clear gain.)

### Implementation

Everything needed already exists:

- `protenix/model/modules/frames.py:expressCoordinatesInFrame(coordinate, frames)` returns
  `[..., N_frame, N_atom, 3]`. It is what `PAELoss.calculate_label` already calls.
- `gather_frame_atom_by_indices` in the same module builds the frames from coordinates.
- `frame_atom_index` (shape `(num_token, 3)`, atom indices after crop) and `has_frame` are
  already standard keys in `input_feature_dict` — `protenix/data/utils.py:1057`. Our cached
  label npz carries them as `lf_frame_atom_index` / `lf_has_frame`.

Steps:

1. Build frames from `x_pred_coords` via `gather_frame_atom_by_indices` on
   `frame_atom_index[has_frame]`. Compute in float32 — the existing code already disables
   autocast around `cdist` for the same reason.
2. `expressCoordinatesInFrame(x_pred_rep_coords, frames)` -> `[N_frame, N_token, 3]`.
3. Divide by the vector length to get unit directions. **Zero the diagonal explicitly** rather
   than relying on an eps: the displacement there is the zero vector and the division is ours,
   not `expressCoordinatesInFrame`'s. Note the existing distance paths already contribute zero
   on the diagonal, so this matches.
4. Scatter into `[N_token, N_token, 3]`, zeros for tokens with no frame. **Mask degenerate
   frames** — `featurizer.py:206` writes `frame_atom_index = [-1, b_idx, -1]` when a frame
   cannot be built, and near-collinear frame atoms make `e3` unstable through the cross product.
   (`10fj` happens to have all 625 tokens framed with no -1 rows, so a smoke test on it will not
   exercise this path. Ligand and ion tokens will.)
5. `z_pair = z_pair + LinearNoBias(3 -> c_z)(that)`, weight zero-initialised.

Cost: `N_token^2 * 3` floats, about 5 MB at 656 tokens, and one `(N, N, 3) @ (3, 128)` matmul.
Negligible.

### Keep both existing distance paths

Do not replace `linear_no_bias_d` or `linear_no_bias_d_wo_onehot`. Their weights are trained and
doing real work, the 39-bin one-hot is a nonlinear high-resolution encoding a linear layer on
three components cannot reproduce, and the unit vector deliberately discards magnitude — which
those paths supply. C is a *third* path, not a substitute.

### Two things found while implementing it

**The target atom is the PAE representative, not the distogram one.** Protenix has two per-token
representative atoms and they are different atoms. `distogram_rep_atom_mask` is CB, CA for
glycine (AF3 SI 4.4, `parser.py:add_distogram_rep_atom_mask`); `pae_rep_atom_mask` is the token
CENTRE atom, CA for standard amino acids. They differ on every non-glycine residue — 1,158 of
`10fj`'s atoms. `memory_efficient_forward` is handed the DISTOGRAM subset, because that is what
`cdist` wants, so the obvious implementation feeds directions to CB while the PAE label is built
on CA. The implementation selects the PAE subset from the full coordinates instead.

**That also makes the self-entry exactly zero.** The frame origin is the middle frame atom, CA,
so with CA targets `[i][i]` is the zero vector. With the distogram subset it was the CA->CB
direction at unit length — a near-constant local vector carrying no pose information. The
self-test caught this: it failed `max|diag| = 1.00` on all three structures before the switch,
and reads exactly `0.00` after. The two existing distance paths contribute zero on the diagonal
too, so a zero diagonal is also what the head already sees there.

### Launching it

**Relaunch from scratch rather than migrating a checkpoint.** Zero-init does not make an
existing bundle loadable: `phase_k_train.py` refuses a bundle missing any trainable tensor, and
the new layer is trainable and absent from every bundle we have; `opt.load_state_dict` also
requires matching parameter groups, which adding a layer changes. A migration is writable —
restore old weights and moments by name, zero the new weight, give it fresh optimiser state —
but it is code to get wrong for no benefit here, and a fresh run gives a clean A/B against the
existing curve. The cost is the ~110 epochs accumulated in `hhzddi`/`jy6sah`.

If a migration is ever wanted, note that step-0 equivalence then means matching the **loaded
checkpoint**, not the shipped head.

---

## A. Inject the pose at every block

### The argument, and its weakness

The pose is added into `z_pair` once, then passes through 4 pairformer blocks whose weights know
nothing about it. Giving each block its own injection gives every block direct access and
shortens the gradient path: the PAE output would reach block 4's injection weight through zero
blocks instead of four.

**This doc does not establish the premise.** The hypothesis is that the pose signal degrades as
it crosses the stack, and with only 4 blocks it does not have far to travel. Nothing measured
here shows geometry is being lost. A adds access and capacity, not information — unlike C. Build
C first; A only becomes interesting if something shows the signal is actually being attenuated.

### What it would be

Each of the 4 blocks gets its own `LinearNoBias(39 -> c_z)`, zero-initialised, added into `z`
before that block runs. Block 1 keeps the existing pretrained `linear_no_bias_d`, so this adds
**3** matrices: `3 * 39 * 128 = 14,976` parameters, 0.08% of the head's 19.8M trainable.

`PairformerStack.forward` builds a list of per-block partials and hands it to
`checkpoint_blocks(blocks, args=(s, z), blocks_per_ckpt)`, each taking `(s, z)` and returning
`(s, z)`. **Wrap each block** — `z = z + W_i(onehot); return block(s, z)` — rather than
inserting standalone callables, which would add 3 checkpoint boundaries for nothing. We run with
`--blocks-per-ckpt 1`, so each block is already its own checkpoint. **Compute the one-hot once**
outside the stack and share it across the four weight matrices; at 656 tokens it is 67 MB held,
and recomputing per block would be four times that.

Each extra matmul is roughly 2 GFLOP at 656 tokens against a block's triangle operations.
Effectively free.

---

## B. Injecting into the single track — ruled out

Do not build this. An earlier version of this doc proposed pooling each token's distance row
into `s_trunk`, on the reasoning that the single track is pose-blind and that making it
pose-aware would influence how the pair track evolves. **Both halves are wrong.**

`PairformerBlock.forward`: the pair update is `tri_mul_out`, `tri_mul_in`, `tri_att_start`,
`tri_att_end`, `pair_transition` — every one reading only `z` and `pair_mask`. Then
`s = s + attention_pair_bias(a=s, s=None, z=z)` and `s = s + single_transition(s)`. Information
flows `z -> s` and never back.

So: the single track is **already** pose-aware, because pose-dependent `z` is read into it —
which means pLDDT already receives pose information. And geometry injected into `s_trunk` could
never reach `pae_pred`, which is computed from `z_pair`. B cannot affect PAE or ipTM without a
further architectural change that lets the pair update read `s`.

---

## The control that distinguishes the two stories

C and A are both representation and access changes. If the binding constraint is **capacity** —
19.8M parameters, 4 blocks, a 128-dim pair channel — neither helps much.

The clean control is `n_blocks 4 -> 8`: it doubles head compute and adds real capacity without
touching the pose pathway. It cannot resume from an existing bundle either, since the new blocks
have no pretrained weights.

An entropy-floor measurement was considered — for each token pair, the entropy of the true PAE
bin across a seed's 5 poses, which is the best a pose-blind predictor could do — and dropped. It
is **one-sided**: a CE below that floor would prove the pose is being used, but a CE at or above
it is equally consistent with a pose-aware model that is simply under-optimised. It would not
change the decision to build C, so it is not worth the setup.

## Why this is worth doing now

Measured 2026-10-19 across every arm's epochs (78 paired points), within-arm slopes of mean
DockQ on pose-weighted PAE cross-entropy run -0.185 to -0.375, averaging about -0.28. The best
arm (`hhzddi`/`jy6sah`) has driven CE from 2.34 at the shipped head to 1.61, and mean DockQ from
0.0420 to a best of 0.2101 (25 of 77 interfaces acceptable).

What that supports is a direction, not a magnitude: **lower PAE cross-entropy correlates with
better selected DockQ.** Two things it does not support, both of which an earlier version of
this doc overstated:

- The slope was fitted over CE 1.67-2.58. Projecting it to CE 1.0 is extrapolation past the
  observed range and does not establish that further CE reductions keep paying at the same rate.
- The figure 0.4216 / 53 of 77 is what the **true** PAE scores when pushed through `ipTM`. It is
  a reference point, not an upper bound on ipTM-based selection: a deliberately miscalibrated
  PAE field could order poses better than the true one does. Calling it a ceiling was wrong.

So driving CE lower is the right target on current evidence, and C is a bet that part of what
makes it slow is that the head cannot see the geometry the target is defined on.
