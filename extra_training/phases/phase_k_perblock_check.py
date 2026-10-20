#!/usr/bin/env python
"""Check pathway A's wiring on CPU, with a real PairformerStack but a small one. No GPU, no fold.

Six properties. Each is something that would be a real bug, and two of them are bugs that
actually happened in pathway C and would happen here the same way.

1. ZERO-INIT IS EXACTLY ZERO. Every injected projection must return 0.0, not something small --
   "step 0 is bit-identical" depends on exactness, not closeness.
2. BIT-IDENTITY AT STEP 0. The stack's output with A installed must equal the output without it,
   to the last bit, on the same inputs. This is the property that lets A be installed on a head
   whose numbers we already trust.
3. THE RIGHT BLOCKS ARE WRAPPED. n_blocks - 1 of them: block 0 keeps the pretrained
   linear_no_bias_d applied before the stack, and wrapping it too would duplicate that injection
   rather than extend it.
4. ONCE TRAINED, IT CHANGES THE OUTPUT. With a non-zero weight the output must differ -- a
   pathway that is inert after training is worse than no pathway, and zero-init makes that easy
   to miss.
5. TIED USES ONE MATRIX, UNTIED USES n-1, and tied is NOT the same function as untied with equal
   weights... it is, so instead: tied must share storage, so writing one matrix changes every
   injection.
6. NO TENSOR IS LEFT REACHABLE between poses. The shared one-hot is 67 MB at 656 tokens; leaving
   it live after the forward is what a per-pose leak looks like from the outside.

Run: /home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_perblock_check.py
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FakeHead(torch.nn.Module):
    """The attributes pose_perblock.install touches, and nothing else.

    A substitute rather than the real ConfidenceHead because the real one needs a 464M-parameter
    model and a GPU to instantiate, while A's wiring is entirely about the pair stack and the
    projections. The STACK is the genuine PairformerStack, so the checkpointing path and the
    _prep_blocks contract are real.
    """

    def __init__(self, n_blocks, c_z, c_s, num_bins):
        super().__init__()
        from protenix.model.modules.pairformer import PairformerStack
        from protenix.model.modules.primitives import LinearNoBias
        self.c_z, self.num_bins = c_z, num_bins
        self.linear_no_bias_d = LinearNoBias(in_features=num_bins, out_features=c_z)
        lower = torch.arange(3.25, 3.25 + 1.25 * num_bins, 1.25)[:num_bins]
        self.lower_bins = torch.nn.Parameter(lower, requires_grad=False)
        self.upper_bins = torch.nn.Parameter(
            torch.cat([lower[1:], lower.new_tensor([1e6])]), requires_grad=False)
        self.pairformer_stack = PairformerStack(n_blocks=n_blocks, c_z=c_z, c_s=c_s,
                                                n_heads=2, dropout=0.0, blocks_per_ckpt=None)

    def memory_efficient_forward(self, input_feature_dict, s_trunk, z_pair, pair_mask,
                                 x_pred_rep_coords, **kw):
        """A stand-in for the real one: embed the distance once, then run the stack."""
        from protenix.model.utils import one_hot
        d = torch.cdist(x_pred_rep_coords.float(), x_pred_rep_coords.float())
        z = z_pair + self.linear_no_bias_d(
            one_hot(x=d, lower_bins=self.lower_bins, upper_bins=self.upper_bins))
        return self.pairformer_stack(s_trunk, z, pair_mask)


class FakeModel(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.confidence_head = FakeHead(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--c-z", type=int, default=16)
    ap.add_argument("--c-s", type=int, default=8)
    ap.add_argument("--num-bins", type=int, default=39)
    ap.add_argument("--n-token", type=int, default=12)
    a = ap.parse_args()

    import pose_perblock

    torch.manual_seed(0)
    kw = dict(n_blocks=a.n_blocks, c_z=a.c_z, c_s=a.c_s, num_bins=a.num_bins)

    def inputs():
        g = torch.Generator().manual_seed(7)
        return (torch.randn(a.n_token, a.c_s, generator=g),
                torch.randn(a.n_token, a.n_token, a.c_z, generator=g),
                torch.randn(a.n_token, 3, generator=g) * 8.0)

    s, z, x = inputs()
    fails = []

    # ONE INSTANCE, BEFORE AND AFTER INSTALL. An earlier version of this check built two models
    # under the same torch.manual_seed and compared them -- and they disagreed by 0.69, because
    # constructing a PairformerStack is not reproducible from the global seed alone. That made
    # check 2 fail on a model that was in fact bit-identical. Installing onto the instance already
    # measured removes the question entirely: the weights cannot differ because they are the same
    # weights.
    torch.manual_seed(0)
    m = FakeModel(**kw).eval()
    with torch.no_grad():
        b_s, b_z = m.confidence_head.memory_efficient_forward({}, s, z, None, x)
    info = pose_perblock.install(m, tied=False, loud=False)

    # 1. zero-init exact
    worst = max(float(p.weight.abs().max()) for p in m.confidence_head.pose_perblock_proj)
    ok1 = worst == 0.0
    # 3. the right count
    ok3 = info["injected"] == a.n_blocks - 1 and info["params"] == (a.n_blocks - 1) * a.num_bins * a.c_z
    # 2. bit-identity
    with torch.no_grad():
        a_s, a_z = m.confidence_head.memory_efficient_forward({}, s, z, None, x)
    ok2 = torch.equal(a_s, b_s) and torch.equal(a_z, b_z)
    dmax = float((a_z - b_z).abs().max())
    # 6. nothing left reachable
    ok6 = pose_perblock._CALL["onehot"] is None

    # 4. a trained weight changes the output
    with torch.no_grad():
        for p in m.confidence_head.pose_perblock_proj:
            p.weight.normal_(0.0, 0.05)
        t_s, t_z = m.confidence_head.memory_efficient_forward({}, s, z, None, x)
    ok4 = not torch.equal(t_z, b_z)
    tdiff = float((t_z - b_z).abs().max())

    # 5. tied uses one matrix, and that matrix reaches every injected block. Same
    # one-instance discipline: measure the model, install tied, then write the single weight.
    torch.manual_seed(1)
    mt = FakeModel(**kw).eval()
    with torch.no_grad():
        bb_s, bb_z = mt.confidence_head.memory_efficient_forward({}, s, z, None, x)
    it = pose_perblock.install(mt, tied=True, loud=False)
    ok5 = it["params"] == a.num_bins * a.c_z and len(mt.confidence_head.pose_perblock_proj) == 1
    with torch.no_grad():
        zero_s, zero_z = mt.confidence_head.memory_efficient_forward({}, s, z, None, x)
        ok5 = ok5 and torch.equal(zero_z, bb_z)          # still bit-identical at zero-init
        mt.confidence_head.pose_perblock_proj[0].weight.normal_(0.0, 0.05)
        tt_s, tt_z = mt.confidence_head.memory_efficient_forward({}, s, z, None, x)
    ok5 = ok5 and not torch.equal(tt_z, bb_z)

    print(f"stack: {a.n_blocks} blocks, c_z {a.c_z}, {a.num_bins} bins, {a.n_token} tokens")
    print(f"  1 zero-init exact      max|w| {worst:.1e}                      {'ok' if ok1 else 'FAIL'}")
    print(f"  2 bit-identical at 0   max|dz| {dmax:.1e}                     {'ok' if ok2 else 'FAIL'}")
    print(f"  3 blocks injected      {info['injected']} of {a.n_blocks}, {info['params']} params        {'ok' if ok3 else 'FAIL'}")
    print(f"  4 trained weight acts  max|dz| {tdiff:.3f}                   {'ok' if ok4 else 'FAIL'}")
    print(f"  5 tied shares storage  {it['params']} params, 1 matrix          {'ok' if ok5 else 'FAIL'}")
    print(f"  6 one-hot released     {pose_perblock._CALL['onehot']}                        {'ok' if ok6 else 'FAIL'}")
    for i, ok in ((1, ok1), (2, ok2), (3, ok3), (4, ok4), (5, ok5), (6, ok6)):
        if not ok:
            fails.append(str(i))
    if fails:
        raise SystemExit(f"\nFAILED: check(s) {', '.join(fails)}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
