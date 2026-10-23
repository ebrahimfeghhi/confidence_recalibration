# 73% of the confidence head cannot change ipTM

Reproduce by adding `--grad-probe` to any arm with the whole head trainable; the gate prints the
dead tensors and refuses to launch. Measured in probes `upnohf` (all 248 head tensors trainable)
and `oqkgp9` (family-level patterns), and the derived set confirmed in `jrkg84`.

## The question

Four arms plateaued at a held-out ranking loss of ~0.038-0.042 with between 10,240 and 44,032
trainable parameters. The parameter gradient collapsing while mistakes remained was traced in
`docs/gradient_alignment.md` to **lost sensitivity** -- the trainable weights losing leverage over
ipTM -- rather than to growing cancellation. The obvious next move is to make more of the head
trainable and hold it near the shipped weights with L2-SP instead of freezing it.

## What the gate found

With all 248 confidence-head tensors trainable, one accumulation window of 100 poses:

| | tensors | parameters | |
|---|---|---|---|
| **no ranking gradient at all** | 158 | 14,438,734 | **73.1%** |
| reachable | 90 | 5,313,536 | 26.9% |

`--grad-probe` refused the launch. A second probe with family-level `--train-only` patterns
(`pair_transition,tri_mul_out,tri_mul_in,tri_att_start,...`) was refused too, with 49 dead
tensors -- **every one of them also dead in the first probe**, so the two measurements agree
exactly.

## Why, for most of it

Confirmed in `protenix/model/modules/confidence.py`:

```python
pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
pde_pred = self.linear_no_bias_pde(self.pde_ln(z_pair + z_pair.transpose(-2, -3)))
a = broadcast_token_to_atom(s_single, ...)
plddt_pred    = einsum(..., self.plddt_ln(a),    self.plddt_weight[...])
resolved_pred = einsum(..., self.resolved_ln(a), self.resolved_weight[...])
```

ipTM is computed from `pae_pred` alone, so it depends on `z_pair` and never on `s_single`. Inside
each pairformer block the triangle operations update `z` from `z`, while `attention_pair_bias` and
`single_transition` update `s` *using* `z` as input -- **`z` never depends on `s`**. So:

- the whole single-representation path (`attention_pair_bias`, `single_transition`, `input_strunk_ln`)
- the plDDT, PDE and resolved output heads (`plddt_ln`, `plddt_weight`, `resolved_ln`,
  `resolved_weight`, `pde_ln`, `linear_no_bias_pde`, `lower_bins`, `upper_bins`)

cannot change ipTM no matter how long they train. Training them would put 14.4M dead parameters in
the optimiser and, worse, in the anchor's `||W - W0||^2`, which prices lambda against drift that
cannot matter.

## What is NOT explained

The live/dead split inside the pairformer is **per block, not per family**, and irregularly so:

| block | live |
|---|---|
| 0 | `pair_transition` only |
| 1 | `tri_mul_in`, `tri_mul_out` only |
| 2 | `pair_transition`, `tri_mul_in`, `tri_mul_out` |
| 3 | `pair_transition`, `tri_att_start`, `tri_mul_in`, `tri_mul_out` |

`tri_att_end` is dead in all four blocks although it updates `z`. The s/z argument above explains
`attention_pair_bias`, `single_transition` and the output heads; it does **not** explain why block
0's `tri_mul` is dead while block 2's is live. One accumulation window cannot distinguish
"structurally disconnected" from "numerically zero on these 100 poses", and this has not been
pinned down. The reachable set used by `nzqhdf` is therefore **derived from the measurement, not
from a theory**, and was handed back to the gate for independent confirmation -- `jrkg84` passed
with 100 tensors / 5.3M parameters and no dead entries.

## The consequence for lambda, and for the underfitting hypothesis

`run_logging/l2sp_lambda.py` transferred lambda from `zfahpb`/`ygpn6t` using `jrkg84`:

```
old: lambda 0.00025   |g_task| 4.0207e-02      40,960 params
new:                  |g_task| 3.9572e-01   5,346,560 params
task-gradient factor   9.8421
parameter-count factor 11.4250
lambda_new = 0.00025 * 9.8421 / 11.4250 = 0.0002154  -> 2.5e-4
```

Two things worth keeping:

1. **`|g_rank|` is 9.84x larger on the same data under the same loss**, purely from unfreezing. That
   is direct evidence that the trainable set, not the objective, was capping the gradient -- the
   "lost sensitivity" of `docs/gradient_alignment.md`, now with a cause.
2. The two factors nearly cancel, so lambda lands on **the same 2.5e-4** that `zfahpb` and `24u34d`
   used, which makes their drift figures directly comparable. It is also **25x stiffer than the
   1e-5** at which `wpg274` ran this parameter set unrestrained to a drift of 3,801 and ended at
   DockQ 0.1616 / frac 0.2778 against shipped 0.2909 / 0.5556.

Cancellation is unchanged by any of this: rank alignment at `jrkg84` is 0.1254 against a
`1/sqrt(N)` baseline of 0.1000, so the ranking objective still destroys ~90% of its gradient
magnitude every step, exactly as it did with 40,960 parameters.
