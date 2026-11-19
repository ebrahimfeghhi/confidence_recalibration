#!/usr/bin/env python
"""--tm-centered must REMOVE a per-pair level and KEEP a within-structure ordering error.

Run: python -I phases/test_tm_centered.py

WHY THIS EXISTS. A shape test would pass on a no-op. The arm exists for exactly one property: a
head whose per-pair error is the same on every pose of a structure must get ~zero loss, and a head
whose error VARIES across the structure's poses must not -- because only the second changes which
pose selection picks. So the asserts here are on the VALUES: the centred loss of a level-shifted
head is ~0 while its uncentred mean square is three orders of magnitude larger, and the centred
loss of an ordering-shifted head equals the variance of the shift it was given.

WHAT IS REAL AND WHAT IS MIRRORED. tm_centered_loss is imported, and _set_pae_mask is EXECUTED out
of phase_k_train.py by its first and last lines (the house pattern, see test_preflight_cover.py) so
the blocks and the per-pair d0 under test are the ones a launch would build. The six-line pass-A
accumulator (sum / cnt / poses) is mirrored here rather than extracted, because it sits inside
run_unit's try block; it is the only mirrored code and it is marked where it appears.

THE FOUR THINGS ASSERTED
  1. d0 PER BLOCK equals calculate_normalization(N_d) for the N_d protenix's own calculate_iptm
     uses for that chain pair -- token_mask.sum() over the two chains, frames included. A wrong
     N_d makes every number here a different metric.
  2. BLOCKS DO NOT OVERLAP, so `m` is 0/1 and nothing is double counted before centring: a token
     pair's two chains determine its block uniquely, and pair_errors' sum-then-count is therefore
     a sum over one block, never two.
  3. THE CENTRING PROPERTY, above, including across poses that arrive in two separate run_unit
     calls (two seeds of one structure in one step).
  4. THE DETACHED e_mean COSTS NOTHING: the gradient of sum_poses mean((e - detach(e_mean))^2)
     equals the gradient of the same expression with e_mean differentiable, on the masks actually
     used -- not in the idealised case.
"""
import os
import sys
import textwrap

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import tm_centered_loss as TMC                                                  # noqa: E402
from protenix.model.sample_confidence import calculate_normalization            # noqa: E402

SRC = os.path.join(ROOT, "phases", "phase_k_train.py")
FIRST = "def _set_pae_mask(pdb, asym, has_frame, unit=None, pairs_override=None):"
LAST = '_pae_mask_stats["masked"] += 1'
MIN_BIN, MAX_BIN, NO_BINS = 0.0, 32.0, 64


def real_set_pae_mask(scored):
    """The REAL _set_pae_mask, out of phase_k_train.py, over a stub closure."""
    lines = open(SRC).read().splitlines()
    lo = next(i for i, ln in enumerate(lines) if ln.strip() == FIRST)
    hi = next(i for i, ln in enumerate(lines) if ln.strip() == LAST)
    env = {"torch": torch, "_scored_asym": scored,
           "_pae_keep": {"m": None, "blocks": None, "pdb": None, "unit": None},
           "_pae_mask_stats": {"masked": 0, "unmasked": 0},
           "_pae_mask_check": lambda: None}
    exec(compile(textwrap.dedent("\n".join(lines[lo:hi + 1])), "<mask>", "exec"), env)
    return env["_set_pae_mask"], env["_pae_keep"]


def toy(sizes):
    """asym_id and an all-True has_frame for chains of these token counts."""
    asym = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)])
    return asym, torch.ones(asym.numel(), dtype=torch.bool)


def logits_for(target, d0):
    """Logits whose softmax puts s_pred exactly on `target`.

    Two live bins, the first and the last, so s_pred = t*w_first + (1-t)*w_last is linear in one
    free parameter and can be solved for any target inside [w_last, w_first].
    """
    w = TMC.per_bin_weight(d0, MIN_BIN, MAX_BIN, NO_BINS, "cpu", dtype=torch.float64)
    lo, hi = float(w[-1]), float(w[0])
    if not ((target > lo + 1e-9).all() and (target < hi - 1e-9).all()):
        raise AssertionError(f"target outside [{lo:.6f}, {hi:.6f}]: "
                             f"[{float(target.min()):.6f}, {float(target.max()):.6f}]")
    t = (target.double() - lo) / (hi - lo)
    lg = torch.full(tuple(target.shape) + (NO_BINS,), -1.0e4, dtype=torch.float64)
    lg[..., 0] = torch.log(t)
    lg[..., -1] = torch.log1p(-t)
    return lg.float()


def accumulate(e_list, m_list):
    """MIRRORS phase_k_train's pass-A accumulator (_tm_acc): sum, per-pair pose count, poses."""
    acc = None
    for e, m in zip(e_list, m_list):
        if acc is None:
            acc = {"sum": e.sum(dim=0).clone(),
                   "cnt": (m > 0).to(torch.float32).sum(dim=0).clone(),
                   "poses": int(e.shape[0])}
        else:
            acc["sum"] += e.sum(dim=0)
            acc["cnt"] += (m > 0).to(torch.float32).sum(dim=0)
            acc["poses"] += int(e.shape[0])
    return acc, acc["sum"] / acc["cnt"].clamp_min(1.0), \
        (acc["cnt"] >= float(acc["poses"])).to(torch.float32)


# ----------------------------------------------------------------- 1 and 2: d0, and no overlap
def test_d0_and_blocks():
    sizes = [40, 30, 17]
    asym, hf = toy(sizes)
    pairs = {(0, 1), (0, 2), (1, 2)}
    setm, keep = real_set_pae_mask({"toy": pairs})
    setm("toy", asym, hf, "seed0")
    blocks = keep["blocks"]
    assert len(blocks) == 3, blocks
    seen = {}
    for blk, npt in blocks:
        # WHICH PAIR THIS BLOCK IS, read back from the mask rather than assumed.
        rows = sorted({int(asym[i]) for i in blk.any(dim=1).nonzero().reshape(-1)})
        cols = sorted({int(asym[j]) for j in blk.any(dim=0).nonzero().reshape(-1)})
        pr = tuple(sorted(set(rows) | set(cols)))
        # N_d AS protenix COMPUTES IT: calculate_iptm takes token_mask = chain x OR chain y, then
        # N_d = has_frame.shape[-1] AFTER that mask -- i.e. the token count of the two chains.
        n_d = int(((asym == pr[0]) | (asym == pr[1])).sum())
        assert npt == n_d, (pr, npt, n_d)
        assert abs(TMC.d0_for_pair(npt) - calculate_normalization(n_d)) == 0.0
        seen[pr] = npt
    assert set(seen) == pairs, seen
    assert seen[(0, 1)] == 70 and seen[(0, 2)] == 57 and seen[(1, 2)] == 47, seen
    # NO DOUBLE COUNTING, ASSERTED ON THE REAL BLOCKS. A token pair's two chains pick its block
    # uniquely, so `m` can only be 0 or 1 -- which is what makes the sum-then-count in pair_errors
    # a per-pair mean and not a per-contribution one. 3 pairs is the worst case this pool has.
    lg = torch.zeros(2, asym.numel(), asym.numel(), NO_BINS)
    sq = torch.zeros(2, asym.numel(), asym.numel())
    _, m = TMC.pair_errors(lg, sq, blocks, MIN_BIN, MAX_BIN, NO_BINS)
    assert float(m.max()) == 1.0, float(m.max())
    assert float(m.sum()) / 2 == sum(float(b.sum()) for b, _ in blocks)
    print(f"  d0 per block: {len(blocks)} block(s), N_d {sorted(seen.values())} "
          f"= protenix's own, d0 {[round(TMC.d0_for_pair(v), 6) for v in sorted(seen.values())]}; "
          f"m.max()=1 so no pair is counted twice")


# --------------------------------------------------------- 3: the property the arm exists for
def test_centring():
    sizes, pairs = [40, 30], [(0, 1)]
    asym, hf = toy(sizes)
    n = asym.numel()
    setm, keep = real_set_pae_mask({"toy": set(pairs)})
    setm("toy", asym, hf, "seed")
    blocks = keep["blocks"]
    d0 = TMC.d0_for_pair(blocks[0][1])
    gen = torch.Generator().manual_seed(7)
    s_true = (0.25 + 0.5 * torch.rand(n, n, generator=gen)).double()
    per_pair = (0.03 * (torch.rand(n, n, generator=gen) - 0.5)).double()
    sq = ((d0 ** 2) * (1.0 / s_true - 1.0)).float().expand(1, n, n)

    # TWO run_unit CALLS, 4 poses then 2, which is how a structure's seeds arrive inside one step.
    split = [4, 2]
    shifts = {
        # a per-STRUCTURE level shift plus a per-PAIR bias: identical on every pose, so the
        # centring must remove all of it.
        "level": torch.zeros(sum(split)).double() + 0.05,
        # the same, but the error now VARIES across the structure's poses -- this is the part
        # selection can read, and the loss must keep it.
        "ordering": torch.tensor([-0.06, -0.02, 0.0, 0.02, 0.04, 0.02]).double() + 0.05,
    }
    res = {}
    for name, g in shifts.items():
        es, ms, uncentred = [], [], []
        at = 0
        for k in split:
            tgt = (s_true + per_pair).unsqueeze(0) + g[at:at + k].reshape(-1, 1, 1)
            at += k
            e, m = TMC.pair_errors(logits_for(tgt, d0), sq.expand(k, n, n), blocks,
                                   MIN_BIN, MAX_BIN, NO_BINS)
            es.append(e)
            ms.append(m)
            uncentred.append(float(((e * (m > 0)) ** 2).sum() / (m > 0).sum()))
        _, e_mean, m_all = accumulate(es, ms)
        losses = [float(TMC.centred_loss(e[p:p + 1], m[p:p + 1], e_mean, m_all))
                  for e, m in zip(es, ms) for p in range(e.shape[0])]
        res[name] = (losses, sum(uncentred) / len(uncentred))

    lvl, lvl_unc = res["level"]
    ordg, ordg_unc = res["ordering"]
    # THE LEVEL-SHIFTED HEAD IS WRONG BY 0.05 ON EVERY PAIR AND THE CENTRED LOSS DOES NOT SEE IT.
    assert max(lvl) < 1e-8, lvl
    assert lvl_unc > 1e-3, lvl_unc
    assert lvl_unc / max(max(lvl), 1e-30) > 1e4, (lvl_unc, max(lvl))
    # THE ORDERING-SHIFTED HEAD IS WRONG BY THE SAME 0.05 ON AVERAGE AND THE LOSS IS NOT ZERO:
    # it is the variance of the part that moved, pose by pose.
    gg = shifts["ordering"]
    want = [float((v - gg.mean()) ** 2) for v in gg]
    assert all(abs(a - b) < 2e-6 for a, b in zip(ordg, want)), (ordg, want)
    assert sum(ordg) / len(ordg) > 1e-4, ordg
    print(f"  level-shifted head: centred {max(lvl):.3e} vs uncentred {lvl_unc:.3e} "
          f"({lvl_unc / max(max(lvl), 1e-30):.1e}x) -- the level is removed")
    print(f"  ordering-shifted head: centred per pose "
          f"{[round(v, 6) for v in ordg]} == var of the shift {[round(v, 6) for v in want]}")


# ------------------------------------------------------- 3b: only pairs valid in EVERY pose
def test_every_pose_mask():
    sizes = [20, 14, 11]
    asym, hf = toy(sizes)
    n = asym.numel()
    # POSE 0 SCORES TWO INTERFACES, THE OTHERS ONE -- which is what --rank-form rel's per-pose
    # pairs_override can produce. The (0,2) pairs are then valid in one pose of three and must
    # take no part: their mean would be over a different population than they are subtracted from.
    setm, keep = real_set_pae_mask({"toy": {(0, 1)}})
    setm("toy", asym, hf, "s", pairs_override={(0, 1), (0, 2)})
    blocks_wide = keep["blocks"]
    setm("toy", asym, hf, "s", pairs_override={(0, 1)})
    blocks_narrow = keep["blocks"]
    assert len(blocks_wide) == 2 and len(blocks_narrow) == 1
    gen = torch.Generator().manual_seed(11)
    sq = 4.0 * torch.rand(1, n, n, generator=gen)
    es, ms = [], []
    # A DIFFERENT PREDICTION PER POSE, so the equality below is not 0 == 0: with identical poses
    # every centred error is zero and the mask could be anything.
    for b in (blocks_wide, blocks_narrow, blocks_narrow):
        lg = torch.randn(1, n, n, NO_BINS, generator=gen)
        e, m = TMC.pair_errors(lg, sq, b, MIN_BIN, MAX_BIN, NO_BINS)
        es.append(e)
        ms.append(m)
    _, e_mean, m_all = accumulate(es, ms)
    narrow = (blocks_narrow[0][0]).to(torch.float32)
    assert torch.equal(m_all, narrow), "m_all is not the AND across poses"
    # AND THE LOSS OF THE WIDE POSE IS THE LOSS OVER THE NARROW BLOCK ALONE.
    wide = float(TMC.centred_loss(es[0], ms[0], e_mean, m_all))
    only = float(((es[0] - e_mean) * narrow).pow(2).sum() / narrow.sum())
    assert wide > 1e-6, wide
    assert abs(wide - only) < 1e-9, (wide, only)
    # AND THE EXCLUDED INTERFACE WOULD HAVE CHANGED IT: the same pose scored with its own wide
    # mask, uncentred, is a different number -- so the every-pose rule is doing work.
    assert abs(float((es[0] * (ms[0] > 0)).pow(2).sum() / (ms[0] > 0).sum()) - wide) > 1e-6
    print(f"  every-pose mask: {int(narrow.sum())} pair(s) kept of "
          f"{int(blocks_wide[0][0].sum() + blocks_wide[1][0].sum())}; the pose-0-only interface "
          f"takes no part ({wide:.6f} == {only:.6f})")


# ------------------------------------------------------------- 4: detaching e_mean is exact
def test_gradient_is_exact():
    sizes = [12, 10]
    asym, hf = toy(sizes)
    n = asym.numel()
    setm, keep = real_set_pae_mask({"toy": {(0, 1)}})
    setm("toy", asym, hf, "s")
    blocks = keep["blocks"]
    gen = torch.Generator().manual_seed(3)
    sq = 4.0 * torch.rand(1, n, n, generator=gen)
    base = torch.randn(4, n, n, NO_BINS, generator=gen)

    def loss(detach_mean):
        lg = base.clone().requires_grad_(True)
        e, m = TMC.pair_errors(lg, sq.expand(4, n, n), blocks, MIN_BIN, MAX_BIN, NO_BINS)
        cnt = (m > 0).to(torch.float32).sum(dim=0)
        em = e.sum(dim=0) / cnt.clamp_min(1.0)
        m_all = (cnt >= float(e.shape[0])).to(torch.float32)
        if detach_mean:
            em = em.detach()
        tot = 0.0
        for p in range(e.shape[0]):
            k = (m[p:p + 1] > 0).to(torch.float32) * m_all
            tot = tot + ((e[p:p + 1] - em) * k).pow(2).sum() / k.sum()
        tot.backward()
        return float(tot), lg.grad

    l_det, g_det = loss(True)
    l_live, g_live = loss(False)
    assert abs(l_det - l_live) < 1e-9, (l_det, l_live)
    rel = float((g_det - g_live).abs().max() / g_live.abs().max())
    assert rel < 1e-5, rel
    # AND THE GRADIENT IS NOT ZERO, which is the whole claim's precondition.
    assert float(g_live.abs().max()) > 1e-6
    print(f"  detached e_mean: loss {l_det:.8f} == {l_live:.8f}, max grad diff "
          f"{float((g_det - g_live).abs().max()):.3e} ({rel:.2e} relative), "
          f"grad scale {float(g_live.abs().max()):.3e}")
    # SUM OF THE CENTRED ERRORS OVER THE POSES, ON THE MASKS ACTUALLY USED, is what makes it so.
    e, m = TMC.pair_errors(base, sq.expand(4, n, n), blocks, MIN_BIN, MAX_BIN, NO_BINS)
    cnt = (m > 0).to(torch.float32).sum(dim=0)
    em = e.sum(dim=0) / cnt.clamp_min(1.0)
    m_all = (cnt >= float(e.shape[0])).to(torch.float32)
    s = float((((e - em) * m_all)).sum())
    assert abs(s) < 1e-3, s
    print(f"  sum of centred errors over poses: {s:+.3e} (0 by construction)")


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in (test_d0_and_blocks, test_centring, test_every_pose_mask, test_gradient_is_exact):
        print(fn.__name__)
        fn()
    print("OK")


def test_rank_merge_is_exact():
    """Splitting a structure's poses across ranks and merging must give the single-rank e_mean.

    THIS IS THE PROPERTY THE all_gather_object IN _xs_prepare EXISTS FOR. All of a structure's
    poses are in the step, but build_epoch_cross_seed deals them in contiguous slices, so a rank
    can hold 10 units of one structure and none of another. Before the gather each pose was centred
    against this RANK's share -- 25 of 75 at world=3 -- and which 25 depended on the shard, so the
    objective depended on the world size.

    The merge is addition because sums and counts are additive, and that is what this asserts: no
    rank needs to know how the units were dealt. `poses` is summed too, so the every-pose test
    (cnt == poses) refers to the STRUCTURE's pose count and not to a rank's share of it.
    """
    import numpy as np
    import torch
    torch.manual_seed(7)
    Nf, Nt, B, npt = 9, 14, 64, 300
    blk = torch.zeros(Nf, Nt, dtype=torch.bool)
    blk[:5, 7:] = True
    blocks = [(blk, npt)]
    # 9 poses of ONE structure, dealt 4 / 3 / 2 across three ranks -- deliberately uneven, which
    # is the case a "poses // world" shortcut would get wrong.
    logits = torch.randn(9, Nf, Nt, B)
    sq = torch.rand(9, Nf, Nt) * 400.0
    e_all, m_all = TMC.pair_errors(logits, sq, blocks, 0.0, 32.0, 64)
    ref_sum = e_all.sum(dim=0).numpy()
    ref_cnt = (m_all > 0).to(torch.float32).sum(dim=0).numpy()
    ref_poses = 9

    shards, i = [], 0
    for n in (4, 3, 2):
        e, m = TMC.pair_errors(logits[i:i + n], sq[i:i + n], blocks, 0.0, 32.0, 64)
        shards.append((e.sum(dim=0).numpy(),
                       (m > 0).to(torch.float32).sum(dim=0).numpy(), n))
        i += n
    got = None
    for s, c, n in shards:
        got = (s.copy(), c.copy(), n) if got is None else (got[0] + s, got[1] + c, got[2] + n)

    d_sum = float(np.abs(got[0] - ref_sum).max())
    d_cnt = float(np.abs(got[1] - ref_cnt).max())
    assert d_sum < 1e-5, f"merged sum differs by {d_sum}"
    assert d_cnt == 0.0, f"merged count differs by {d_cnt}"
    assert got[2] == ref_poses, f"merged pose count {got[2]} != {ref_poses}"
    # AND THE MEAN THAT COMES OUT OF IT, which is what pass B divides.
    ref_mean = ref_sum / np.maximum(ref_cnt, 1.0)
    got_mean = got[0] / np.maximum(got[1], 1.0)
    d_mean = float(np.abs(got_mean - ref_mean).max())
    assert d_mean < 1e-6, f"merged e_mean differs by {d_mean}"
    # A PER-RANK MEAN IS GENUINELY DIFFERENT, so this test would catch the ungathered version.
    r0_mean = shards[0][0] / np.maximum(shards[0][1], 1.0)
    d_r0 = float(np.abs(r0_mean - ref_mean).max())
    print(f"  merged over 4/3/2 poses: sum diff {d_sum:.2e}, count diff {d_cnt:.0f}, "
          f"poses {got[2]}, e_mean diff {d_mean:.2e}")
    print(f"  one rank's own mean differs from the true mean by {d_r0:.4f} "
          f"-- which is what the gather removes")
    assert d_r0 > 1e-4, ("a single rank's mean matched the true mean, so this test cannot "
                         "distinguish the gathered version from the ungathered one")


if __name__ == "__main__":
    print("test_rank_merge_is_exact")
    test_rank_merge_is_exact()
    print("OK")
