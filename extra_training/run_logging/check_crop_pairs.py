"""GATE: every selected crop must keep a scorable cross-chain pair for EVERY pose's OST pair.

THE FAILURE THIS EXISTS TO CATCH. 3pdv7i died 21 minutes into epoch 1 because the crop pool was
validated against the STATIC chain_roles join while phase_k_train scores the PER-POSE OST join.
On 9o4o those name different chains, so the pool certified a crop that deleted the chain training
needed, iptm_pooled returned None for both interfaces on every pose, and the pass-A completeness
guard stopped the run -- after three GPUs had been held for 21 minutes.

Nothing about that failure needed a GPU to find. This asserts the same invariant on CPU, over
every structure and every crop in the selected pool, in about a minute.

THE CONDITION IS iptm_pooled's OWN, not a proxy. It returns None unless there is a LIVE row token
(has_frame, non-ligand) in one of the pair's chains and ANY token in the other -- so checking
"both chains survive" is not enough, and checking "the interface retains contacts" is not the
same question. This reproduces the three branches exactly.

Exit 0 and print OK, or exit 1 naming every structure and crop that would die.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phases"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import crop_itm_rank as C                                           # noqa: E402
import phase_j_build_cache as BC                                    # noqa: E402
import phase_k_eval as PKE                                          # noqa: E402


def ost_pairs_union(fold, pdb, refs):
    """{(iface, lo, hi)} over every pose -- one crop is shared by all of them."""
    seen = set()
    for seed in C.SEEDS:
        try:
            ud = BC.seed_dir(fold, pdb, int(seed))
        except Exception:                                           # noqa: BLE001
            continue
        for samp in range(5):
            got, _ = PKE.ost_chain_pairs(ud, pdb, samp, refs)
            for i, x, y in (got or []):
                seen.add((i, min(x, y), max(x, y)))
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="training/crop_final")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--scored-pairs", default="training/combined/pool.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    refs_by_pdb = PKE.pool_ref_pairs(a.scored_pairs)
    files = sorted(f for f in os.listdir(a.pool) if f.endswith(".npz"))
    if a.limit:
        files = files[:a.limit]

    bad, checked, crops_checked = [], 0, 0
    for fn in files:
        pdb = fn[:-4]
        got = None
        for seed in C.SEEDS:
            got = C.unit_tensors(a.fold, pdb, seed)
            if got is not None:
                break
        if got is None:
            bad.append((pdb, -1, "no cached tensors"))
            continue
        asym = got[0].reshape(-1).long()
        hf = got[1].reshape(-1).bool()
        pairs = ost_pairs_union(a.fold, pdb, refs_by_pdb.get(pdb) or [])
        if not pairs:
            bad.append((pdb, -1, "no OST pair for any pose"))
            continue

        z = np.load(os.path.join(a.pool, fn), allow_pickle=True)
        rows, bi = [], 0
        while f"c{bi}" in z.files:
            for r in np.asarray(z[f"c{bi}"]):
                rows.append(np.asarray(r))
            bi += 1
        checked += 1
        for ci, idx in enumerate(rows):
            keep = torch.zeros(asym.numel(), dtype=torch.bool)
            keep[torch.as_tensor(idx.astype(np.int64))] = True
            ca, ch = asym[keep], hf[keep]
            crops_checked += 1
            for iface, i, j in pairs:
                in_i, in_j = (ca == i), (ca == j)
                # iptm_pooled's own three None branches, in order.
                if not bool((in_i | in_j).any()) or not bool(ch.any()):
                    bad.append((pdb, ci, f"{iface} pair({i},{j}): no token of either chain"))
                    continue
                live_i = bool((in_i & ch).any())
                live_j = bool((in_j & ch).any())
                if not ((live_i and bool(in_j.any())) or (live_j and bool(in_i.any()))):
                    bad.append((pdb, ci,
                                f"{iface} pair({i},{j}): no live cross-chain row "
                                f"(chain {i}: {int(in_i.sum())} tok/{int((in_i & ch).sum())} live, "
                                f"chain {j}: {int(in_j.sum())} tok/{int((in_j & ch).sum())} live)"))

    print(f"checked {checked} structure(s), {crops_checked} crop(s) against every pose's OST pair")
    if bad:
        print(f"REFUSING: {len(bad)} crop/pair combination(s) would score None and stop the run:")
        for pdb, ci, why in bad[:20]:
            print(f"    {pdb} crop {ci}: {why}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
        return 1
    print("OK: every selected crop keeps a scorable cross-chain pair for every pose's OST pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
