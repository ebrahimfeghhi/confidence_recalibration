# More pose pathways into the confidence head

Status: proposal, nothing implemented. Written 2026-10-19.

Two changes to how the predicted pose reaches the confidence head. They are independent and
can be built separately, but both are zero-initialised, so with either or both installed the
head at step 0 is **bit-identical** to the shipped head and a run can resume from an existing
bundle rather than starting over.

## Where the pose enters today

`ConfidenceHead.memory_efficient_forward`, in `protenix/model/modules/confidence.py`:

```python
distance_pred = torch.cdist(x_pred_rep_coords, x_pred_rep_coords)      # [N_token, N_token]
z_pair = z_pair + self.linear_no_bias_d(one_hot(distance_pred, ...))   # 39 bins -> c_z
z_pair = z_pair + self.linear_no_bias_d_wo_onehot(distance_pred[..., None])  # 1 -> c_z
s_single, z_pair = self.pairformer_stack(s_trunk, z_pair, pair_mask, ...)
pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
```

So the pose is a grid of scalar distances, embedded and added into the pair representation
**once**, before a stack of **4** pairformer blocks whose weights are all pose-agnostic.
`s_trunk` is pose-independent: nothing pose-dependent ever reaches the single track.

Relevant dimensions: `n_blocks=4`, `c_z=128`, `c_s=384`, and the distance one-hot has
`num_bins=39` (`arange(3.25, 52.0, 1.25)`).

---

## C. Frame-relative displacement

### The argument

PAE is not a distance. It is: superpose on token i's local frame, then ask how far token j
ended up from where it should be. `PAELoss.calculate_label` computes exactly that — express
j's position in i's frame for the predicted structure, do the same for the true structure,
take the distance between those two vectors.

`cdist` is **symmetric** by construction: entry `[i][j]` equals entry `[j][i]`, always. PAE is
not. `PAE[i][j]` is j's error in i's frame; `PAE[j][i]` is i's error in j's frame. Different
quantities.

So the head is asked to predict an asymmetric target from a symmetric pose input. Whatever
asymmetry it produces has to come from `z_trunk` and `s_inputs`, which are pose-independent
and identical across all five poses of a seed. **The pose cannot contribute to the asymmetric
part of the target at all.**

### How asymmetric is it

Measured 2026-10-19 on 3 structures (`10fj`, `10gh`, `10op`) x 2 seeds x 5 poses, computing the
true PAE label through `expressCoordinatesInFrame` and binning it the way the loss does
(64 bins over 0-32 A, everything beyond saturating in the last bin):

| scope | frac of pairs where bin(i,j) != bin(j,i) | corr(bin, bin^T) |
|---|---|---|
| all pairs | 0.690 | 0.733 |
| inter-chain pairs | 0.553 | 0.651 |

Over half of inter-chain pairs carry a different label in the two directions.

**Two caveats on that number.** 43% of the inter-chain pairs are both saturated in the top bin
(>= 32 A), where asymmetry cannot show even if present — so 55% is measured on a set where
nearly half the pairs are incapable of disagreeing, and the true figure is higher. But these
are whole uncropped structures from the fold set, not the `overfit30` interface crops we
actually train on. **Re-measure on the crops before building.** It is a CPU-only job of a few
minutes and it is the measurement that justifies C over A.

### What goes in

For each ordered pair (i, j), the displacement of j expressed in i's frame — three numbers.
For a residue the frame atoms are N, CA, C, giving origin CA, `e1` the bisector of the two bond
directions, `e2` perpendicular to `e1` in the N-CA-C plane, `e3` normal to that plane. The three
numbers are j's position along those axes.

This is asymmetric in (i, j), like the target. It is still invariant to globally rotating or
translating the complex, because i's frame rotates with it — the same symmetry the PAE label
has, so nothing breaks under a rigid motion.

### Implementation

Everything needed already exists:

- `protenix/model/modules/frames.py:expressCoordinatesInFrame(coordinate, frames)` returns
  `[..., N_frame, N_atom, 3]`. It is what `PAELoss.calculate_label` already calls.
- `gather_frame_atom_by_indices` in the same module builds the frames from coordinates.
- `frame_atom_index` (shape `(num_token, 3)`, atom indices after crop) and `has_frame` are
  already standard keys in `input_feature_dict` — see `protenix/data/utils.py:1057`. No new
  data plumbing, and our cached label npz already carries them as `lf_frame_atom_index` /
  `lf_has_frame`.

Steps:

1. Build frames from `x_pred_coords` with `gather_frame_atom_by_indices` on
   `frame_atom_index[has_frame]`.
2. `expressCoordinatesInFrame(x_pred_rep_coords, frames)` -> `[N_frame, N_token, 3]`.
3. Scatter into `[N_token, N_token, 3]`, zeros for tokens without a frame.
4. `z_pair = z_pair + LinearNoBias(3 -> c_z)(that)`, weight zero-initialised.

Cost: the tensor is `N_token^2 * 3` floats, about 5 MB at 656 tokens. One `(N, N, 3) @ (3, 128)`
matmul. Negligible.

### Keep the distance input

Do not replace `linear_no_bias_d`. Its weights are trained and doing real work; the 39-bin
one-hot is a nonlinear high-resolution encoding of distance that a linear layer reading three
raw components cannot reproduce, since length is not linear in the components; and keeping it is
what lets the new layer start at zero so step 0 is identical to the current head.

### Scaling

The three components are in Angstroms and unbounded, where the existing input is a one-hot.
Either normalise them, or bin each axis the way the distance is binned and feed
`LinearNoBias(3 * num_bins -> c_z)`. Binning is closer to the existing design and avoids a
scale mismatch against the one-hot path; raw with a normalisation is fewer parameters. Worth
trying raw first, since zero-init means a bad scale shows up as a slow start rather than a
broken head.

---

## A. Inject the pose at every block

### The argument

The pose is added into `z_pair` once, then passes through 4 pairformer blocks whose weights
know nothing about it. Giving each block its own injection gives every block direct access, and
shortens the gradient path: the PAE output reaches block 4's injection weight through zero
blocks instead of four.

**The honest case against it:** the hypothesis is that the pose signal degrades as it passes
through the stack, and with only 4 blocks it does not have far to travel. A adds capacity and
access, not information — unlike C, which adds information the input does not currently contain.
That is why C is ranked first.

### What goes in

Each of the 4 blocks gets its own `LinearNoBias(39 -> c_z)`, zero-initialised, whose output is
added into `z` before that block runs. Block 1 keeps the existing pretrained
`linear_no_bias_d`, so this adds **3** new matrices: `3 * 39 * 128 = 14,976` parameters, 0.08%
of the head's 19.8M trainable.

### Implementation

`PairformerStack.forward` builds a list of per-block partials and hands it to
`checkpoint_blocks(blocks, args=(s, z), blocks_per_ckpt)`, each taking `(s, z)` and returning
`(s, z)`.

**Wrap each block, do not insert new callables between them.** Wrapping —
`z = z + W_i(onehot); return block(s, z)` — keeps the checkpoint boundary count at 4. Inserting
standalone callables would add 3 more boundaries and cost memory for nothing. We run with
`--blocks-per-ckpt 1`, so every block is already its own checkpoint.

**Compute the one-hot once**, outside the stack, and let all four injections share that tensor
with their own weight matrices. At 656 tokens it is `656^2 * 39 * 4 B` = 67 MB held. Computing
it per block would be four times that for no benefit.

Each extra matmul is roughly 2 GFLOP at 656 tokens, against a pairformer block's triangle
operations. Effectively free.

### C and A compose

C at every block is the natural combination: 4 copies of `LinearNoBias(3 -> c_z)` alongside the
distance injections. All zero-init, so any subset is identical to the current head at step 0,
and an ablation afterwards is just zeroing one group.

---

## B. The single track, not pursued yet

`s_trunk` is pose-blind. Pool each token's distance row into a per-token summary — mean and max
over its one-hot row, or a small learned attention pool — and add a zero-init
`LinearNoBias(39 -> c_s)` into `s_trunk`. One `(N, 39) @ (39, 384)` matmul.

This matters because the Pairformer's single track is updated by attention with pair bias, so
`s` influences how the pair track evolves, and today it does so with no knowledge of which pose
it is looking at. It is also the only pathway that would make pLDDT (which reads `s_single`)
pose-aware, though that is not what we are optimising.

Ranked below C and A because the PAE head reads `z_pair`, not `s_single`, so this is the most
indirect of the three.

---

## The control that distinguishes the two stories

All of C, A and B are representation and access changes. If the binding constraint is
**capacity** — 19.8M parameters, 4 blocks, a 128-dim pair channel — none of them help much.

The clean control is `n_blocks 4 -> 8`: it doubles head compute and adds real capacity without
touching the pose pathway. Running that alongside C+A says which story is right. Note it breaks
zero-init equivalence, since the new blocks have no pretrained weights, so it cannot resume from
an existing bundle.

The cheaper version of the same question is the **entropy floor**, still unrun: for each token
pair, the entropy of the true PAE bin across a seed's 5 poses. That is the minimum
cross-entropy a pose-blind predictor can reach. If it sits near where we currently are, the
pose is contributing almost nothing and these pathways are the right place to work. If it sits
well above us, we are already extracting pose information and the limit is elsewhere.

## Why this is worth doing now

Measured 2026-10-19 across every arm's epochs (78 paired points), within-arm slopes of mean
DockQ on pose-weighted PAE cross-entropy run -0.185 to -0.375, averaging about -0.28. The best
arm (`hhzddi`/`jy6sah`) has driven CE from 2.34 at the shipped head to 1.61, and mean DockQ from
0.0420 to a best of 0.2101 (25 of 77 interfaces acceptable). Extrapolating the within-arm slope
to CE 1.0 lands somewhere in 0.29-0.43 against a perfect-PAE ceiling of 0.4216 (53 of 77) and an
oracle of 0.5098 (68 of 77).

So driving CE lower is the right target, and the question is what is making it slow. These
pathways are a bet that part of the answer is that the head cannot see the geometry the target
is defined on.
