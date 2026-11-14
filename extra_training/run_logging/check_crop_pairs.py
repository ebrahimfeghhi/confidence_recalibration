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


def ost_pairs_union(fold, pdb, refs, unmapped=None):
    """{(iface, lo, hi)} over every pose -- one crop is shared by all of them.

    AN UNMAPPED POSE IS APPENDED TO `unmapped`, NOT SWALLOWED. `(got or [])` alone meant a pose
    whose OST mapping cannot be read contributed nothing and said nothing, so a structure with no
    mappable pose returned an EMPTY union -- and an empty union passes every check below for the
    reason that there is nothing to check. That is a gate reporting OK about a structure it never
    looked at. The caller names them.
    """
    seen = set()
    for seed in C.SEEDS:
        try:
            ud = BC.seed_dir(fold, pdb, int(seed))
        except Exception:                                           # noqa: BLE001
            continue
        for samp in range(5):
            got, why = PKE.ost_chain_pairs(ud, pdb, samp, refs)
            if not got and unmapped is not None:
                unmapped.append(f"{pdb}/{os.path.basename(ud)} sample {samp}: "
                                f"{why or 'mapping resolved no scored interface'}")
            for i, x, y in (got or []):
                seen.add((i, min(x, y), max(x, y)))
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="training/crop_final")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--scored-pairs", default="training/combined/pool.csv")
    ap.add_argument("--limit", type=int, default=0)
    # WHICH JOIN THE CONSUMER USES, because the two disagree on 90 of 846 structures and the
    # crop only has to survive the one that will actually be asked for.
    #   ost    -- the per-pose OST join, which --rank-form rel scores (_pose_pairs).
    #   static -- the chain_roles join, which the PAE mask uses (phase_k_train.py:2096) and
    #             which build_crop_pool.py validated against.
    ap.add_argument("--join", choices=["ost", "static"], default="ost")
    ap.add_argument("--roles", default="training/fold/chain_roles_all.csv")
    a = ap.parse_args()

    refs_by_pdb = PKE.pool_ref_pairs(a.scored_pairs)
    static_pairs = {}
    if a.join == "static":
        _sp, _ = PKE.interface_chain_pairs(a.scored_pairs, a.roles)
        static_pairs = {k: [(i, min(x, y), max(x, y)) for i, x, y in v] for k, v in _sp.items()}
    files = sorted(f for f in os.listdir(a.pool) if f.endswith(".npz"))
    if a.limit:
        files = files[:a.limit]

    bad, checked, crops_checked = [], 0, 0
    # A PARTIALLY UNMAPPED STRUCTURE NARROWS THE UNION, WHICH IS THE QUIET VERSION OF THE FAILURE
    # THIS GATE EXISTS FOR. "no OST pair for any pose" below catches the all-unmapped case, but if
    # 50 of 75 poses are unreadable the union is built from the other 25 -- and a crop certified
    # against that smaller union can be missing a pair the unreadable poses would have asked for,
    # the moment they become readable. So every unmapped pose is named in the output.
    unmapped = []
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
        pairs = (ost_pairs_union(a.fold, pdb, refs_by_pdb.get(pdb) or [], unmapped)
                 if a.join == "ost" else static_pairs.get(pdb, []))
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

    print(f"checked {checked} structure(s), {crops_checked} crop(s) against the "
          f"{a.join.upper()} chain-pair join")
    if unmapped:
        _pdbs = sorted({u.split("/")[0] for u in unmapped})
        print(f"OST MAPPING UNREADABLE on {len(unmapped)} pose(s) across {len(_pdbs)} "
              f"structure(s) -- the union below was built WITHOUT them, so it may be short of a "
              f"pair those poses need: {', '.join(_pdbs)}")
        for u in unmapped[:40]:
            print(f"    {u}")
        if len(unmapped) > 40:
            print(f"    ... and {len(unmapped) - 40} more")
        print("RECOVERY: python -I run_logging/rescue_ost_mapping.py --pdb <pdb> --verify "
              "(CPU only), then re-run this gate.")
    if bad:
        print(f"REFUSING: {len(bad)} crop/pair combination(s) would score None and stop the run:")
        for pdb, ci, why in bad[:20]:
            print(f"    {pdb} crop {ci}: {why}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
        return 1
    print(f"OK: every selected crop keeps a scorable cross-chain pair under the {a.join.upper()} join")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
